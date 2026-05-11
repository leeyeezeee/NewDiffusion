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

from utils import get_kwargs, save_graph_with_features
from accuracy import Accuracy
from mas.prompt.mmlu_prompt_set import ROLE_DESCRIPTION

sys.stdout.reconfigure(encoding='utf-8')

from mas.datasets.mmlu_dataset import MMLUDataset
from mas.graph.graph import Graph, TestGraph
from mas.utils.const import mas_ROOT
from mas.utils.globals import Cost, PromptTokens, CompletionTokens

from process_datasets import load_graph_dataset, PyGGraphDataset
from sentence_transformers import SentenceTransformer

from model.gd import GDFramework
from model.denoising import DenoisingNetwork
from model.ordering import DiffusionOrderingNetwork
from model.utils import NodeMasking


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMLU experiment.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--diffusion_batch_size', type=int, default=32, help="Batch size for diffusion training")
    parser.add_argument('--update_freq', type=int, default=10, help="Utility loss update frequency for the model training")
    parser.add_argument('--sample_ratio', type=float, default=0.5, help="Sample ratio for the model utility loss training")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'], help='List of agent names')
    parser.add_argument('--agent_nums', type=int, default=5, help='Specify the number of agents')
    parser.add_argument('--train_set_size', type=int, default=12, help="Size of the training set")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds for each query")
    parser.add_argument('--device', type=str, default='cuda:1', help='Specify cuda devices')
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument('--dataset', type=str, default="mmlu", help="Dataset name")
    parser.add_argument('--domain', type=str, default="mmlu", help="Domain name, same as dataset name")
    parser.add_argument('--decision_method', type=str, default="FinalRefer", help="Decision method for the final node")
    parser.add_argument('--num_workers', type=int, default=6, help="Number of workers for data loading")
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of epochs for training")
    parser.add_argument('--num_trajectories', type=int, default=4, help="Number of trajectories for training")
    parser.add_argument('--limit_questions', type=int, default=153, help="Limit number of questions to evaluate")

    args = parser.parse_args()

    if len(args.agent_names) != 1:
        parser.error("The number of agent names must match the number of agent counts.")

    return args


def get_initial_dataset_configs():
    """
    Return initial dataset configurations.
    """
    configs = set()
    for agent_num in range(3, 5):
        configs.add(('FullConnected', agent_num))
        configs.add(('Mesh', agent_num))
        configs.add(('Star', agent_num))
        configs.add(('Layered', agent_num))
        configs.add(('Random', agent_num))

    return list(configs)


async def generate_initial_dataset(args):
    train_set_size = args.train_set_size
    dataset = MMLUDataset('MMLU','dev')
    print()
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    initial_dataset_indices = all_indices[:train_set_size]
    initial_dataset = torch.utils.data.Subset(dataset, initial_dataset_indices)

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
            num_rounds=args.num_rounds,
            limit_questions=None,
            eval_batch_size=args.batch_size,
            args=args,
            current_mode=mode,
            current_agent_num=agent_num
        )

    print("All initial dataset generation complete.")


async def evaluate(
        graph: Graph,
        dataset,  # Subset of MMLUDataset
        num_rounds: int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 1,
        args=None,
        current_mode: str = None,
        current_agent_num: int = None
) -> float:
    """
    Run multi-agent inference on an initial dataset and save successful graphs.
    """
    accuracy = Accuracy()
    original_dataset = dataset.dataset

    sorted_roles = sorted(ROLE_DESCRIPTION.keys())
    role_to_id = {role: i for i, role in enumerate(sorted_roles)}
    id_to_role = {i: role for role, i in role_to_id.items()}

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = len(dataset) if limit_questions is None else min(len(dataset), limit_questions)
    num_batches = math.ceil(data_len / eval_batch_size)

    dirpath = mas_ROOT / "cache/MMLU/graphs"

    os.makedirs(dirpath, exist_ok=True)

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        tasks = []
        questions = []
        flow_graphs = []

        for record in record_batch:
            g_copy = copy.deepcopy(graph)
            input_dict = original_dataset.record_to_input(record)
            flow_graph = g_copy.to_pyg_graph(input_dict)
            tg = TestGraph(
                domain=args.domain,
                llm_name=args.llm_name,
                decision_method=args.decision_method,
                pyg_data=flow_graph
            )
            tasks.append(asyncio.create_task(tg.arun(input_dict, num_rounds)))
            questions.append(input_dict['task'])
            flow_graphs.append(flow_graph)

        is_corrects = []
        raw_results = await asyncio.gather(*tasks)

        for raw_answer, record in zip(raw_results, record_batch):
            answer = original_dataset.postprocess_answer(raw_answer)
            correct_answer = original_dataset.record_to_target_answer(record)
            is_correct = accuracy.update(answer, correct_answer)
            accuracy.print()
            is_corrects.append(is_correct)

        for i, record in enumerate(record_batch):
            record_id = record.get('id', f"task_{i_batch * eval_batch_size + i}")

            name = "_".join(map(str, ['mmlu', record_id, current_mode, current_agent_num, is_corrects[i]]))
            filepath = dirpath / f'{name}.pt'
            save_graph_with_features(
                flow_graphs[i],
                str(filepath),
                {
                    "mode": current_mode,
                    "num_nodes": current_agent_num,
                    "is_correct": is_corrects[i],
                    "question": questions[i],
                    "record": record
                }
            )

    accuracy.print()
    print(f"Finished Mode={current_mode}, Agent Nums={current_agent_num}, Accuracy: {accuracy.get():.2f}")
    return accuracy.get()

def compute_sparsity_reward(graph):
    """
    Compute sparsity reward: ratio of missing edges to max possible edges.
    For directed graph with n nodes: max_edges = n * (n-1)
    """
    n = graph.num_nodes
    max_possible_edges = n * (n - 1)  # No self-loops
    actual_edges = graph.edge_index.shape[1]
    sparsity_reward = 1.0 - (actual_edges / max_possible_edges) if max_possible_edges > 0 else 0.0
    return sparsity_reward

async def evaluate_generated_graphs_batch(
    gd_framework, 
    denoising_batch, 
    args, 
    id_to_role
):
    """
    Helper function to evaluate a batch of generated graphs asynchronously.
    
    Args:
        gd_framework: GDFramework instance
        denoising_batch: List of original graphs
        args: Arguments object
        id_to_role: Dictionary mapping role IDs to role names
    
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
    
    for graph in denoising_batch:
        # Get original graph's correctness (baseline)
        original_reward = graph.is_correct.item()
        original_sparsity_reward = compute_sparsity_reward(graph)

        original_reward = original_reward * 0.8 + original_sparsity_reward * 0.2
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
        input_dict = MMLUDataset.record_to_input(graph.record)
        
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
        # Get correct answer
        correct_answer = MMLUDataset.record_to_target_answer(graph_data['record'])
        
        # Postprocess answer and check correctness
        postprocessed_answer = MMLUDataset.postprocess_answer(raw_answer)
        generated_reward = 1.0 if (postprocessed_answer == correct_answer) else 0.0
        generated_sparsity_reward = compute_sparsity_reward(graph_data['graph'])
        generated_reward = generated_reward * 0.8 + generated_sparsity_reward * 0.2
        
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
    graph_dir = mas_ROOT / "cache/MMLU/graphs"
    role_dir = mas_ROOT / "cache/MMLU/roles"

    dataset = load_graph_dataset(args, graph_dir=graph_dir, role_dir=role_dir)

    role_to_id = dataset.role_to_id
    id_to_role = dataset.id_to_role

    pyg_graph_dataset = PyGGraphDataset(dataset.graph_list)

    role_embeddings_path = role_dir / 'precomputed_role_embeddings.pkl'
    with open(str(role_embeddings_path), 'rb') as f:
        role_embeddings_dict = pickle.load(f)
        print(f"Loaded {len(role_embeddings_dict)} role embeddings")

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
        drop_last=False,
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
            if batch_idx % args.update_freq == 0:
                sample_size = max(1, int(args.sample_ratio * len(denoising_batch)))
                sample_indices = random.sample(range(len(denoising_batch)), sample_size)
                sample_denoising_batch = [denoising_batch[i] for i in sample_indices]

                # Evaluate generated graphs for REINFORCE
                evaluation_results = await evaluate_generated_graphs_batch(
                    gd_framework, 
                    sample_denoising_batch, 
                    args, 
                    id_to_role
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

    model_dir = mas_ROOT / "cache/MMLU/models"
    os.makedirs(model_dir, exist_ok=True)
    gd_framework.save_model(model_dir)
    print("Training finished.")

    return gd_framework


async def evaluate_graph_diffusion_model(gd_framework, args):
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentence_model.to(args.device)

    dataset = MMLUDataset('MMLU','val')

    accuracy = Accuracy()
    limit_questions = args.limit_questions

    sorted_roles = sorted(ROLE_DESCRIPTION.keys())
    role_to_id = {role: i for i, role in enumerate(sorted_roles)}
    id_to_role = {i: role for role, i in role_to_id.items()}

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / args.batch_size))

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=args.batch_size)), total=num_batches):
        print(f"{'-' * 80}")

        start_ts = time.time()
        answer_tasks = []
        questions = []

        for i, record in enumerate(record_batch):
            input_dict = dataset.record_to_input(record)
            task_text = input_dict['task']
            questions.append(task_text)

            question_id = i_batch * args.batch_size + i + 1

            # Add task information by transforming task text to embedding
            task_embedding = torch.tensor(
                sentence_model.encode(task_text, device=args.device)
            ).float()

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
        is_corrects = []

        for raw_answer, record in zip(raw_results, record_batch):
            answer = dataset.postprocess_answer(raw_answer)
            correct_answer = dataset.record_to_target_answer(record)
            is_correct = accuracy.update(answer, correct_answer)
            
            acc = accuracy.get() * 100
            print(f"Accuracy: {acc:.2f}% | "
                  f"Cost: ${Cost.instance().value:.4f} | "
                  f"Tokens: P({int(PromptTokens.instance().value)}), C({int(CompletionTokens.instance().value)})")

            is_corrects.append(is_correct)
        
        print(f"Batch time: {time.time() - start_ts:.3f}s")
    
    acc = accuracy.get() * 100
    print(f"Accuracy: {acc:.2f}%")

    final_cost = Cost.instance().value
    final_prompt_tokens = PromptTokens.instance().value
    final_completion_tokens = CompletionTokens.instance().value
    total_tasks = min(len(dataset), args.limit_questions) if args.limit_questions is not None else len(dataset_test)

    print("\n" + "=" * 50 + "\nEvaluation Summary")
    print(f"Total tasks: {total_tasks}\nFinal accuracy : {acc:.2f}%")
    print("-" * 50)
    print(f"Total cost: ${final_cost:.6f}")
    print(f"Total Prompt Tokens: {int(final_prompt_tokens)}")
    print(f"Total Completion Tokens: {int(final_completion_tokens)}")
    print("-" * 50)

    # write the accuracy to a text file
    with open('mmlu_accuracy.txt', 'a+', encoding='utf-8') as f:
        f.write(f"Total tasks: {total_tasks}\nFinal accuracy : {acc:.2f}%\nTotal cost: ${final_cost:.6f}\nTotal Prompt Tokens: {int(final_prompt_tokens)}\nTotal Completion Tokens: {int(final_completion_tokens)}\n")


async def main():
    args = parse_args()

    # step 1: generate initial dataset
    graphs_dir = mas_ROOT / "cache/MMLU/graphs"
    if graphs_dir.exists() and any(graphs_dir.iterdir()):
        print("Initial dataset already generated.")
    else:
        await generate_initial_dataset(args)
    
    # step 2: train graph diffusion model
    models_dir = mas_ROOT / "cache/MMLU/models"
    if models_dir.exists() and any(models_dir.iterdir()):
        graph_dir = mas_ROOT / "cache/MMLU/graphs"
        role_dir = mas_ROOT / "cache/MMLU/roles"

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
    await evaluate_graph_diffusion_model(gd_framework, args)
    print("Evaluation finished.")

if __name__ == "__main__":
    asyncio.run(main())
