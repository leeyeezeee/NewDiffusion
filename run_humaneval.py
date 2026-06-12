import math
from typing import Optional, Iterator, Any, List
from tqdm import tqdm
import copy
import sys
import os
import torch
import pickle
import time
import asyncio
import argparse
import random

from utils import (
    attach_edge_semantic_gains,
    build_semantic_judge_from_args,
    compute_effective_size_reward,
    compute_sparsity_reward,
    get_kwargs,
    save_graph_with_features,
)
from mas.prompt.humaneval_prompt_set import ROLE_DESCRIPTION

sys.stdout.reconfigure(encoding='utf-8')

from mas.graph.graph import Graph, TestGraph
from mas.utils.const import mas_ROOT
from mas.utils.globals import Cost, PromptTokens, CompletionTokens
from mas.tools.reader.readers import JSONLReader
from mas.tools.coding.python_executor import PyExecutor

from process_datasets import load_graph_dataset, PyGGraphDataset
from sentence_transformers import SentenceTransformer

from model.gd import GDFramework
from model.denoising import DenoisingNetwork
from model.ordering import DiffusionOrderingNetwork
from model.utils import NodeMasking


def parse_args():
    parser = argparse.ArgumentParser(description="Run HumanEval experiment.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--diffusion_batch_size', type=int, default=32, help="Batch size for diffusion training")
    parser.add_argument('--update_freq', type=int, default=10, help="Utility loss update frequency for the model training")
    parser.add_argument('--sample_ratio', type=float, default=1.0, help="Sample ratio for the model utility loss training")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'], help='List of agent names')
    parser.add_argument('--agent_nums', type=int, default=5, help='Specify the number of agents')
    parser.add_argument('--train_set_size', type=int, default=50, help="Size of the training set")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds for each query")
    parser.add_argument('--device', type=str, default='cuda:0', help='Specify cuda devices')
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument('--dataset', type=str, default="humaneval", help="Dataset name")
    parser.add_argument('--domain', type=str, default="humaneval", help="Domain name, same as dataset name")
    parser.add_argument('--decision_method', type=str, default="FinalWriteCode", help="Decision method for the final node")
    parser.add_argument('--num_workers', type=int, default=6, help="Number of workers for data loading")
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of epochs for training")
    parser.add_argument('--num_trajectories', type=int, default=4, help="Number of trajectories for training")
    parser.add_argument('--semantic_entropy_samples', type=int, default=0, help="Samples per edge side for collection-time semantic entropy; <=1 disables it")
    parser.add_argument('--semantic_judge_llm_name', type=str, default=None, help="LLM used to judge semantic equivalence")
    parser.add_argument('--semantic_judge_api_key', type=str, default="", help="API key for the semantic judge")
    parser.add_argument('--semantic_judge_base_url', type=str, default="", help="Optional OpenAI-compatible base URL for the semantic judge")
    parser.add_argument('--semantic_judge_max_concurrency', type=int, default=16, help="Max concurrent semantic judge requests")
    parser.add_argument('--collect_correct_semantic_graphs', action='store_true', help="Only collect correct graphs with edge semantic entropy gains, then exit")

    args = parser.parse_args()

    if len(args.agent_names) != 1:
        parser.error("The number of agent names must match the number of agent counts.")

    return args


def get_initial_dataset_configs():
    """
    Return initial dataset configurations.
    """
    configs = set()
    for agent_num in (5,):
        configs.add(('FullConnected', agent_num))
        configs.add(('Mesh', agent_num))
        configs.add(('Star', agent_num))
        configs.add(('Layered', agent_num))
        configs.add(('Random', agent_num))

    return list(configs)


async def generate_initial_dataset(args):
    train_set_size = args.train_set_size
    dataset = JSONLReader.parse_file('data/humaneval/humaneval-py.jsonl')
    print(f"Loaded {len(dataset)} HumanEval dataset")

    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    initial_dataset_indices = all_indices[:train_set_size]
    initial_dataset = [dataset[i] for i in initial_dataset_indices]

    test_indices = all_indices[train_set_size:]
    test_dataset = [dataset[i] for i in test_indices]

    # Generate data for each initial dataset configuration
    configs = get_initial_dataset_configs()
    print(f"Generating initial dataset for {len(configs)} configurations...")

    for mode, agent_num in configs:
        print(f"\n=== Processing configuration: Mode={mode}, Agent Nums={agent_num} ===")

        # Build Graph instance for this configuration
        current_agent_names = [args.agent_names[0]] * agent_num
        kwargs = get_kwargs(mode, agent_num)
        available_roles = list(ROLE_DESCRIPTION.keys())
        random_roles = random.choices(available_roles, k=agent_num)
        kwargs['node_kwargs'] = [{'role': role} for role in random_roles]
        graph = Graph(
            domain=args.domain,
            llm_name=args.llm_name,
            agent_names=current_agent_names,
            decision_method=args.decision_method,
            **kwargs
        )

        # Evaluate and save data
        await evaluate(
            graph=graph,
            dataset=initial_dataset,
            args=args,
            current_mode=mode,
            current_agent_num=agent_num
        )

    print("All initial dataset generation complete.")

    return test_dataset


async def evaluate(
        graph: Graph,
        dataset,
        args,
        current_mode: str = None,
        current_agent_num: int = None
) -> float:
    """
    Run multi-agent inference on an initial dataset and save successful graphs.
    """
    executor = PyExecutor()
    num_batches = math.ceil(len(dataset) / args.batch_size)
    total_solved = 0

    dirpath = mas_ROOT / "cache/humaneval/graphs"

    os.makedirs(dirpath, exist_ok=True)
    if args is not None and args.collect_correct_semantic_graphs and args.semantic_entropy_samples <= 1:
        args.semantic_entropy_samples = 3
    semantic_judge = build_semantic_judge_from_args(args)
    if args is not None and args.collect_correct_semantic_graphs and semantic_judge is None:
        raise ValueError(
            "--collect_correct_semantic_graphs requires a configured semantic judge. "
            "Set OPENAI_API_KEY or pass --semantic_judge_api_key."
        )

    for i_batch in tqdm(range(num_batches), desc=f"Processing {current_mode}-{current_agent_num}"):
        batch_records = dataset[i_batch * args.batch_size: (i_batch + 1) * args.batch_size]
        
        if not batch_records:
            continue
        
        tasks = []
        for record in batch_records:
            g_copy = copy.deepcopy(graph)
            input_dict = {"task": record["prompt"]}
            flow_graph = g_copy.to_pyg_graph(input_dict)
            tg = TestGraph(
                domain=args.domain,
                llm_name=args.llm_name,
                decision_method=args.decision_method,
                pyg_data=flow_graph
            )

            metadata = {
                "record": record,
                "flow_graph": flow_graph,
                "question": record["prompt"],
                "input_dict": input_dict,
                "test_graph": tg,
            }
            tasks.append((tg.arun(input_dict, args.num_rounds), metadata))

        coroutines_to_run = [task for task, meta in tasks]
        results = await asyncio.gather(*coroutines_to_run, return_exceptions=True)

        save_items = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task execution error: {result}")
                continue

            metadata = tasks[i][1]
            record = metadata['record']

            raw_answer = result
            if isinstance(raw_answer, list) and raw_answer:
                raw_answer = raw_answer[0]
            answer_code = raw_answer.lstrip("```python\n").rstrip("\n```")

            is_solved, _, _ = executor.execute(answer_code, [record["test"]], timeout=10)

            total_solved += 1
            if args is not None and args.collect_correct_semantic_graphs and not is_solved:
                continue
            save_items.append((i, is_solved))

        if semantic_judge is not None and save_items:
            semantic_tasks = [
                attach_edge_semantic_gains(
                    tasks[i][1]["test_graph"],
                    tasks[i][1]["flow_graph"],
                    tasks[i][1]["input_dict"],
                    tasks[i][1]["question"],
                    semantic_judge,
                    args.semantic_entropy_samples,
                )
                for i, _ in save_items
            ]
            semantic_results = await asyncio.gather(*semantic_tasks, return_exceptions=True)
            for result in semantic_results:
                if isinstance(result, Exception):
                    print(f"Semantic entropy computation failed: {result}")

        for i, is_solved in save_items:
            metadata = tasks[i][1]
            record = metadata['record']
            record_id = record.get('task_id', f"task_{i_batch * args.batch_size + i}")
            name = "_".join(map(str, ['humaneval', record_id, current_mode, current_agent_num, is_solved]))
            filepath = dirpath / f'{name}.pt'

            save_graph_with_features(
                metadata['flow_graph'],
                str(filepath), 
                {
                    "mode": current_mode,
                    "num_nodes": current_agent_num,
                    "is_correct": is_solved,
                    "question": metadata['question'],
                    "record": metadata['record']
                }
            )
                
    print(f"Config {current_mode}-{current_agent_num} finished. Successfully solved {total_solved} / {len(dataset)} tasks.")


async def evaluate_generated_graphs_batch(
    gd_framework, 
    denoising_batch, 
    args, 
    id_to_role,
    role_to_id
):
    """
    Helper function to evaluate a batch of generated graphs asynchronously.
    
    Args:
        gd_framework: GDFramework instance
        denoising_batch: List of original graphs
        args: Arguments object
        id_to_role: Dictionary mapping role IDs to role names
        role_to_id: Dictionary mapping role names to IDs
    
    Returns:
        results: List of dicts with keys:
            - 'generated_reward': float (0.0 or 1.0)
            - 'original_reward': float (0.0 or 1.0)
            - 'log_probs_list': List of log probs
            - 'connections_list': List of connections
            - 'graph': Original graph
    """
    evaluation_tasks = []
    graph_data_list = []
    executor = PyExecutor()
    
    for graph in denoising_batch:
        # Get original graph's correctness (baseline)
        original_reward = graph.is_correct.item()
        original_sparsity_reward = compute_sparsity_reward(graph)
        original_effective_size_reward = compute_effective_size_reward(graph.x, graph.edge_index)

        original_reward = original_reward * 0.8 + original_sparsity_reward * 0.1 + original_effective_size_reward * 0.1
        # Generate graph with log probs
        task_embedding = graph.task_embedding
        num_nodes = graph.x.shape[0] if hasattr(graph, 'x') and hasattr(graph.x, 'shape') else args.agent_nums
        
        generated_graph, log_probs_list, connections_list = gd_framework.generate_graph(
            num_nodes, task_embedding, return_log_probs=True
        )
        
        # Convert generated graph format for evaluation
        generated_graph_copy = generated_graph.clone()
        generated_graph_copy.num_nodes = generated_graph.x.shape[0]
        if hasattr(generated_graph_copy, 'x') and generated_graph_copy.x.dim() > 0:
            if isinstance(generated_graph_copy.x, torch.Tensor):
                generated_graph_copy.x = [{'role': id_to_role[idx.item()]} for idx in generated_graph_copy.x]
        
        # Create TestGraph for evaluation
        tg = TestGraph(
            domain=args.domain,
            llm_name=args.llm_name,
            decision_method=args.decision_method,
            pyg_data=generated_graph_copy
        )
        
        # Create input dict
        input_dict = {"task": graph.record["prompt"]}
        metadata = {
            "record": graph.record,
            "flow_graph": generated_graph_copy,
            "question": graph.record["prompt"],
        }
        
        # Store data for later processing
        graph_data_list.append({
            'graph': generated_graph_copy,
            'log_probs_list': log_probs_list,
            'connections_list': connections_list,
            'original_reward': original_reward,
            'record': graph.record
        })
        
        # Create async evaluation task
        evaluation_tasks.append(asyncio.create_task(tg.arun(input_dict, num_rounds=args.num_rounds)))
    
    # Wait for all evaluations to complete
    if len(evaluation_tasks) == 0:
        return []
    
    raw_answers = await asyncio.gather(*evaluation_tasks)
    
    # Process results
    results = []

    for i, (raw_answer, graph_data) in enumerate(zip(raw_answers, graph_data_list)):
        if isinstance(raw_answer, Exception):
            print(f"Task execution error: {raw_answer}")
            continue
        
        # Get correct answer
        correct_answer = graph_data['record']["test"]
        if isinstance(raw_answer, list) and raw_answer:
            raw_answer = raw_answer[0]

        answer_code = raw_answer.lstrip("```python\n").rstrip("\n```")
        is_solved, _, _ = executor.execute(answer_code, [correct_answer], timeout=10)
        generated_reward = 1.0 if is_solved else 0.0
        generated_sparsity_reward = compute_sparsity_reward(graph_data['graph'])
        generated_x = torch.tensor([role_to_id[node['role']] for node in graph_data['graph'].x], device=args.device)
        generated_edge_index = graph_data['graph'].edge_index
        generated_effective_size_reward = compute_effective_size_reward(generated_x, generated_edge_index)
        generated_reward = generated_reward * 0.8 + generated_sparsity_reward * 0.1 + generated_effective_size_reward * 0.1
        
        results.append({
            'generated_reward': generated_reward,
            'original_reward': graph_data['original_reward'],
            'log_probs_list': graph_data['log_probs_list'],
            'connections_list': graph_data['connections_list'],
            'graph': graph_data['graph']
        })
    
    return results


async def train_graph_diffusion_model(args):
    # load cached dataset
    graph_dir = mas_ROOT / "cache/humaneval/graphs"
    role_dir = mas_ROOT / "cache/humaneval/roles"

    dataset = load_graph_dataset(args, graph_dir=graph_dir, role_dir=role_dir)

    role_to_id = dataset.role_to_id
    id_to_role = dataset.id_to_role

    pyg_graph_dataset = PyGGraphDataset(dataset.graph_list)

    diff_ord_net = DiffusionOrderingNetwork(
        node_feature_dim=1,
        num_node_types=len(role_to_id),
        num_edge_types=1,
        num_layers=3,
        out_channels=1,
        device=args.device
    )

    denoising_net = DenoisingNetwork(
        node_feature_dim=1,
        edge_feature_dim=1,
        task_feature_dim=pyg_graph_dataset.task_embedding_dim,
        num_node_types=len(role_to_id),
        num_edge_types=1,
        num_layers=3,
        device=args.device
    )

    gd_framework = GDFramework(
        dataset=pyg_graph_dataset,
        denoising_network=denoising_net,
        diffusion_ordering_network=diff_ord_net,
        device=args.device
    )

    dataloader = torch.utils.data.DataLoader(
        pyg_graph_dataset,
        batch_size=args.diffusion_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=lambda _: _
    )

    denoising_optimizer = torch.optim.Adam(gd_framework.denoising_network.parameters(), lr=1e-5, betas=(0.9, 0.999))
    ordering_optimizer = torch.optim.Adam(gd_framework.diffusion_ordering_network.parameters(), lr=5e-5, betas=(0.9, 0.999))

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            mid = len(batch) // 2
            denoising_batch = batch[:mid]
            ordering_batch = batch[mid:]
            
            batch_denoising_loss, batch_ordering_loss = gd_framework.train_step(
                denoising_batch, ordering_batch, M=args.num_trajectories
            )
            
            # Compute REINFORCE loss
            batch_reinforce_loss = torch.tensor(0.0, device=args.device, requires_grad=True)
            if epoch % args.update_freq == 0 and epoch > 0:
                sample_size = max(1, int(args.sample_ratio * len(denoising_batch)))
                sample_indices = random.sample(range(len(denoising_batch)), sample_size)
                sample_denoising_batch = [denoising_batch[i] for i in sample_indices]

                # Evaluate generated graphs for REINFORCE
                evaluation_results = await evaluate_generated_graphs_batch(
                    gd_framework, 
                    sample_denoising_batch, 
                    args, 
                    id_to_role,
                    role_to_id
                )
                
                if len(evaluation_results) > 0:
                    reinforce_losses = []
                    for result in evaluation_results:
                        reinforce_loss = gd_framework.compute_graph_utility_loss(
                            log_probs_list=result['log_probs_list'],
                            generated_reward=torch.tensor(result['generated_reward'], device=args.device, dtype=torch.float32),
                            original_reward=torch.tensor(result['original_reward'], device=args.device, dtype=torch.float32),
                            connections_list=result['connections_list']
                        )
                        reinforce_losses.append(reinforce_loss)
                    batch_reinforce_loss = sum(reinforce_losses) / len(reinforce_losses)
            
            # Combine losses: NLL + REINFORCE
            total_denoising_loss = batch_denoising_loss + batch_reinforce_loss
            total_ordering_loss = batch_ordering_loss
            
            # Backward pass
            denoising_optimizer.zero_grad()
            ordering_optimizer.zero_grad()
            
            total_denoising_loss.backward()
            total_ordering_loss.backward()
            
            denoising_optimizer.step()
            ordering_optimizer.step()
            
            # Print progress
            if batch_idx % args.update_freq == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx}, "
                      f"Denoising Loss: {total_denoising_loss.item():.4f} "
                      f"(NLL: {batch_denoising_loss.item():.4f}, REINFORCE: {batch_reinforce_loss.item():.4f}), "
                      f"Ordering Loss: {total_ordering_loss.item():.4f}")

    model_dir = mas_ROOT / "cache/humaneval/models"
    os.makedirs(model_dir, exist_ok=True)
    gd_framework.save_model(model_dir)
    print("Training finished.")

    return gd_framework


async def evaluate_graph_diffusion_model(gd_framework, args, test_dataset):
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    executor = PyExecutor()

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentence_model.to(args.device)

    if test_dataset is not None:
        dataset = test_dataset
    else:
        dataset = JSONLReader.parse_file('data/humaneval/humaneval-py.jsonl')
        # randomly shuffle the dataset
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        dataset = [dataset[i] for i in all_indices[args.train_set_size:]]
        
    total_solved = 0
    total_tasks = len(dataset)

    sorted_roles = sorted(ROLE_DESCRIPTION.keys())
    role_to_id = {role: i for i, role in enumerate(sorted_roles)}
    id_to_role = {i: role for role, i in role_to_id.items()}

    def eval_loader(data: List[Any], batch_size: int) -> Iterator[List[Any]]:
        records = []
        for record in data:
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    num_batches = int(math.ceil(len(dataset) / args.batch_size))

    for i_batch, record_batch in tqdm(enumerate(eval_loader(dataset, batch_size=args.batch_size)), total=num_batches):
        print(f"{'-' * 80}")

        start_ts = time.time()
        answer_tasks = []

        for i, record in enumerate(record_batch):
            input_dict = {"task": record["prompt"]}
            task_text = record['prompt']

            task_embedding = torch.tensor(
                sentence_model.encode(task_text), device=args.device).float()

            generated_graph = gd_framework.generate_graph(args.agent_nums, task_embedding)
            generated_graph.task = task_text
            generated_graph.num_nodes = args.agent_nums
            generated_graph.x = list({'role': id_to_role[idx.item()]} for idx in generated_graph.x)

            tg = TestGraph(
                domain=args.domain,
                llm_name=args.llm_name,
                decision_method=args.decision_method,
                pyg_data=generated_graph
            )
            answer_tasks.append(asyncio.create_task(tg.arun(input_dict, args.num_rounds)))
        
        raw_results = await asyncio.gather(*answer_tasks)

        for raw_answer, record in zip(raw_results, record_batch):
            if isinstance(raw_answer, Exception):
                print(f"Task execution error: {raw_answer}")
                continue
            
            if isinstance(raw_answer, list) and raw_answer:
                raw_answer = raw_answer[0]
            answer_code = raw_answer.lstrip("```python\n").rstrip("\n```")
            is_solved, _, _ = executor.execute(answer_code, [record["test"]], timeout=10)

            if is_solved:
                total_solved += 1
        
        print(f"Batch time: {time.time() - start_ts:.3f}s")
    
    acc = total_solved / total_tasks * 100
    print(f"Accuracy: {acc:.2f}% ({total_solved}/{total_tasks})")

    final_cost = Cost.instance().value
    final_prompt_tokens = PromptTokens.instance().value
    final_completion_tokens = CompletionTokens.instance().value

    print("\n" + "=" * 50 + "\nEvaluation Summary")
    print(f"Total tasks: {total_tasks}\nFinal accuracy : {acc:.2f}%")
    print("-" * 50)
    print(f"Total cost: ${final_cost:.6f}")
    print(f"Total Prompt Tokens: {int(final_prompt_tokens)}")
    print(f"Total Completion Tokens: {int(final_completion_tokens)}")
    print("-" * 50)

    # write the accuracy to a text file
    with open('humaneval_accuracy.txt', 'a+', encoding='utf-8') as f:
        f.write(f"Total tasks: {total_tasks}\nFinal accuracy : {acc:.2f}%\nTotal cost: ${final_cost:.6f}\nTotal Prompt Tokens: {int(final_prompt_tokens)}\nTotal Completion Tokens: {int(final_completion_tokens)}\n")


async def main():
    args = parse_args()

    test_dataset = None

    if args.collect_correct_semantic_graphs:
        await generate_initial_dataset(args)
        print("Correct semantic graph collection finished.")
        return

    # step 1: generate initial dataset
    graphs_dir = mas_ROOT / "cache/humaneval/graphs"
    if graphs_dir.exists() and any(graphs_dir.iterdir()):
        print("Initial dataset already generated.")
    else:
        test_dataset = await generate_initial_dataset(args)
    
    # step 2: train graph diffusion model
    models_dir = mas_ROOT / "cache/humaneval/models"
    if models_dir.exists() and any(models_dir.iterdir()):
        graph_dir = mas_ROOT / "cache/humaneval/graphs"
        role_dir = mas_ROOT / "cache/humaneval/roles"

        dataset = load_graph_dataset(args, graph_dir=graph_dir, role_dir=role_dir)

        role_to_id = dataset.role_to_id

        pyg_graph_dataset = PyGGraphDataset(dataset.graph_list)

        diff_ord_net = DiffusionOrderingNetwork(
            node_feature_dim=1,
            num_node_types=len(role_to_id),
            num_edge_types=1,
            num_layers=3,
            out_channels=1,
            device=args.device
        )

        denoising_net = DenoisingNetwork(
            node_feature_dim=1,
            edge_feature_dim=1,
            task_feature_dim=pyg_graph_dataset.task_embedding_dim,
            num_node_types=len(role_to_id),
            num_edge_types=1,
            num_layers=3,
            device=args.device
        )

        gd_framework = GDFramework(
            dataset=pyg_graph_dataset,
            denoising_network=denoising_net,
            diffusion_ordering_network=diff_ord_net,
            device=args.device
        )
        gd_framework.load_model(models_dir)
    else:
        gd_framework = await train_graph_diffusion_model(args)
        print("Graph diffusion model training finished.")

    # step 3: evaluate the performance of the graph diffusion model
    await evaluate_graph_diffusion_model(gd_framework, args, test_dataset)
    print("Evaluation finished.")

if __name__ == "__main__":
    asyncio.run(main())
