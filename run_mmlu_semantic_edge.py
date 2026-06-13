import argparse
import asyncio
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any, Tuple

import networkx as nx
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from accuracy import Accuracy
from mas.datasets.mmlu_dataset import MMLUDataset
from mas.graph.graph import Graph, TestGraph
from mas.prompt.mmlu_prompt_set import ROLE_DESCRIPTION
from mas.utils.const import mas_ROOT
from mas.utils.globals import CompletionTokens, Cost, PromptTokens
from model.denoising import SemanticEdgeDenoisingNetwork
from model.gd import GDFramework
from model.ordering import DiffusionOrderingNetwork, TaskAwareTopologyOrderNetwork
from process_datasets import PyGGraphDataset, load_graph_dataset
from utils import build_semantic_judge_from_args, get_kwargs, save_graph_with_features
from model.semantic_entropy import _sample_node_outputs, semantic_uncertainty


RL_CORRECTNESS_REWARD_WEIGHT = 0.75
RL_SPARSITY_REWARD_WEIGHT = 0.15
RL_ENTROPY_REWARD_WEIGHT = 0.10
DEFAULT_SEMANTIC_ENTROPY_SAMPLES = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MMLU with supervised topology ordering and edge-only semantic diffusion."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--domain", type=str, default="mmlu")
    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--diffusion_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--order_epochs", type=int, default=5)
    parser.add_argument("--edge_epochs", type=int, default=15)
    parser.add_argument("--rl_epochs", type=int, default=5)
    parser.add_argument("--update_freq", type=int, default=10)
    parser.add_argument("--sample_ratio", type=float, default=0.5)
    parser.add_argument("--agent_nums", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--limit_questions", type=int, default=153)
    parser.add_argument("--lr_order", type=float, default=5e-5)
    parser.add_argument("--lr_edge", type=float, default=1e-5)
    parser.add_argument("--hidden_dim_order", type=int, default=128)
    parser.add_argument("--hidden_dim_edge", type=int, default=256)
    parser.add_argument("--graph_dir", type=str, default=str(mas_ROOT / "cache/MMLU/graphs"))
    parser.add_argument("--role_dir", type=str, default=str(mas_ROOT / "cache/MMLU/roles"))
    parser.add_argument("--model_dir", type=str, default=str(mas_ROOT / "cache/MMLU/semantic_edge_models"))
    parser.add_argument("--bootstrap_graphs", action="store_true")
    parser.add_argument("--bootstrap_train_size", type=int, default=64)
    parser.add_argument("--bootstrap_agent_min", type=int, default=3)
    parser.add_argument("--bootstrap_agent_max", type=int, default=5)
    parser.add_argument(
        "--bootstrap_modes",
        nargs="+",
        default=["FullConnected", "Layered", "Star", "Random"],
    )
    parser.add_argument("--lambda_order_rl", type=float, default=0.05)
    parser.add_argument("--lambda_edge_rl", type=float, default=0.05)
    parser.add_argument("--rl_sft_weight", type=float, default=0.1)
    parser.add_argument("--hard_semantic_pruning", action="store_true")
    parser.add_argument("--semantic_prune_threshold", type=float, default=0.0)
    parser.add_argument("--semantic_entropy_samples", type=int, default=DEFAULT_SEMANTIC_ENTROPY_SAMPLES)
    parser.add_argument("--semantic_judge_llm_name", type=str, default=None)
    parser.add_argument("--semantic_judge_api_key", type=str, default="")
    parser.add_argument("--semantic_judge_base_url", type=str, default="")
    parser.add_argument("--semantic_judge_max_concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.semantic_entropy_samples <= 1:
        parser.error("--semantic_entropy_samples must be greater than 1 because RL reward always includes average agent semantic entropy.")
    return args


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def topological_order_from_pyg(data):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    edge_index = data.edge_index.detach().cpu()
    graph.add_edges_from(
        (int(edge_index[0, idx]), int(edge_index[1, idx]))
        for idx in range(edge_index.shape[1])
        if int(edge_index[0, idx]) != int(edge_index[1, idx])
    )

    if nx.is_directed_acyclic_graph(graph):
        return list(nx.topological_sort(graph))

    # Fallback for malformed cached candidates: keep a stable node order.
    return list(range(int(data.num_nodes)))


def role_ids_for_generation(graph, agent_nums, device):
    node_types = graph.x.squeeze().long().to(device)
    if node_types.numel() >= agent_nums:
        return node_types[:agent_nums]
    pad = torch.zeros(agent_nums - node_types.numel(), dtype=torch.long, device=device)
    return torch.cat([node_types, pad], dim=0)


def compute_sparsity_reward(graph):
    n = int(getattr(graph, "num_nodes", 0) or len(getattr(graph, "x", [])))
    max_possible_edges = n * (n - 1)
    if max_possible_edges <= 0:
        return 0.0
    actual_edges = int(graph.edge_index.shape[1]) if hasattr(graph, "edge_index") else 0
    return 1.0 - (actual_edges / max_possible_edges)


def scalar_float(value, default=0.0):
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def edge_policy_log_prob(log_probs_list):
    total_log_prob = None
    count = 0
    for log_probs in log_probs_list or []:
        edge_log_prob = log_probs.get("edge_log_prob")
        if edge_log_prob is None:
            continue
        valid_mask = log_probs.get("valid_edge_mask")
        if valid_mask is not None:
            edge_log_prob = edge_log_prob[valid_mask]
        step_log_prob = edge_log_prob.sum()
        total_log_prob = step_log_prob if total_log_prob is None else total_log_prob + step_log_prob
        count += int(edge_log_prob.numel())
    return total_log_prob, count


def order_policy_log_prob(order_log_probs):
    if not order_log_probs:
        return None, 0
    return torch.stack(order_log_probs).sum(), len(order_log_probs)


def policy_gradient_loss(log_prob, count, advantage, device):
    if log_prob is None or count <= 0:
        return torch.tensor(0.0, device=device)
    advantage = torch.as_tensor(advantage, dtype=torch.float32, device=device).detach()
    return -advantage * log_prob / max(count, 1)


def weighted_reward(correctness, sparsity, avg_entropy, args):
    if avg_entropy is None:
        raise ValueError("avg_entropy is required because RL reward always includes average agent semantic entropy.")
    entropy_reward = 1.0 / (1.0 + max(float(avg_entropy), 0.0))
    reward = RL_CORRECTNESS_REWARD_WEIGHT * float(correctness)
    reward += RL_SPARSITY_REWARD_WEIGHT * float(sparsity)
    reward += RL_ENTROPY_REWARD_WEIGHT * entropy_reward
    return reward, entropy_reward


def baseline_reward(correctness, sparsity):
    return (
        RL_CORRECTNESS_REWARD_WEIGHT * float(correctness)
        + RL_SPARSITY_REWARD_WEIGHT * float(sparsity)
    )


def semantic_prune_threshold(args):
    return args.semantic_prune_threshold if args.hard_semantic_pruning else None


async def average_agent_semantic_entropy(test_graph, input_dict, question, judge, num_samples):
    if judge is None or num_samples <= 1:
        return None, []

    async def node_entropy_item(node):
        history = getattr(node, "execution_history", [])
        if not history:
            return None
        history_item = history[-1]
        outputs = await _sample_node_outputs(
            node,
            input_dict,
            history_item.get("spatial_info", {}),
            history_item.get("temporal_info", {}),
            num_samples,
        )
        if not outputs:
            return None
        entropy, labels = await semantic_uncertainty(question, outputs, judge)
        return {
            "node_id": node.id,
            "role": node.role,
            "entropy": float(entropy),
            "labels": labels,
        }

    entropy_results = await asyncio.gather(
        *[node_entropy_item(node) for node in test_graph.nodes.values()],
        return_exceptions=True,
    )
    entropy_items = [
        item
        for item in entropy_results
        if isinstance(item, dict)
    ]

    if not entropy_items:
        return 0.0, []
    avg_entropy = sum(item["entropy"] for item in entropy_items) / len(entropy_items)
    return avg_entropy, entropy_items


def build_order_graph(gd_framework, node_types, task_embedding):
    order_graph = gd_framework.masker.generate_fully_masked(n_nodes=node_types.numel()).to(gd_framework.device)
    order_graph.x = node_types.view(-1, 1).long().to(gd_framework.device)
    order_graph.task_embedding = task_embedding
    return order_graph


async def evaluate_generated_graphs_for_rl(
    args,
    gd_framework,
    denoising_batch,
    id_to_role,
    semantic_judge=None,
):
    evaluation_tasks = []
    graph_data_list = []

    for graph in denoising_batch:
        if not hasattr(graph, "record"):
            continue

        task_embedding = graph.task_embedding
        node_types = role_ids_for_generation(graph, args.agent_nums, args.device)
        order_graph = build_order_graph(gd_framework, node_types, task_embedding)
        node_order, order_log_probs = gd_framework.topology_order_network.sample_order(
            order_graph,
            task_embedding=task_embedding,
            sampling_method="sample",
            return_log_probs=True,
        )

        generated_graph, edge_log_probs_list, connections_list = gd_framework.generate_graph(
            args.agent_nums,
            task_embedding,
            return_log_probs=True,
            node_types=node_types,
            node_order=node_order,
        )

        generated_graph_copy = generated_graph.clone()
        generated_graph_copy.num_nodes = args.agent_nums
        default_role = next(iter(id_to_role.values()), "Normal")
        generated_graph_copy.x = [
            {"role": id_to_role.get(int(node_type.item()), default_role)}
            for node_type in node_types
        ]

        test_graph = TestGraph(
            domain=args.domain,
            llm_name=args.llm_name,
            decision_method=args.decision_method,
            pyg_data=generated_graph_copy,
        )
        input_dict = MMLUDataset.record_to_input(graph.record)
        graph_data_list.append({
            "original_graph": graph,
            "generated_graph": generated_graph_copy,
            "test_graph": test_graph,
            "input_dict": input_dict,
            "record": graph.record,
            "order_log_probs": order_log_probs,
            "edge_log_probs_list": edge_log_probs_list,
            "connections_list": connections_list,
            "node_order": node_order,
        })
        evaluation_tasks.append(asyncio.create_task(test_graph.arun(input_dict, num_rounds=args.num_rounds)))

    if not evaluation_tasks:
        return []

    raw_answers = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
    entropy_tasks = [
        average_agent_semantic_entropy(
            graph_data["test_graph"],
            graph_data["input_dict"],
            graph_data["input_dict"]["task"],
            semantic_judge,
            args.semantic_entropy_samples,
        )
        for graph_data in graph_data_list
    ]
    entropy_results = await asyncio.gather(*entropy_tasks, return_exceptions=True)

    results = []
    for raw_answer, graph_data, entropy_result in zip(raw_answers, graph_data_list, entropy_results):
        record = graph_data["record"]
        correct_answer = MMLUDataset.record_to_target_answer(record)
        if isinstance(raw_answer, BaseException):
            postprocessed_answer = ""
            is_correct = False
        else:
            postprocessed_answer = MMLUDataset.postprocess_answer(raw_answer)
            is_correct = postprocessed_answer == correct_answer

        if isinstance(entropy_result, BaseException):
            avg_entropy, entropy_details = 0.0, []
        else:
            avg_entropy, entropy_details = entropy_result

        generated_sparsity = compute_sparsity_reward(graph_data["generated_graph"])
        generated_reward, entropy_reward = weighted_reward(
            float(is_correct),
            generated_sparsity,
            avg_entropy,
            args,
        )

        original_correct = scalar_float(getattr(graph_data["original_graph"], "is_correct", None))
        original_sparsity = compute_sparsity_reward(graph_data["original_graph"])
        original_reward = baseline_reward(original_correct, original_sparsity)

        results.append({
            "generated_reward": generated_reward,
            "original_reward": original_reward,
            "advantage": generated_reward - original_reward,
            "is_correct": bool(is_correct),
            "generated_sparsity": generated_sparsity,
            "original_sparsity": original_sparsity,
            "avg_semantic_entropy": avg_entropy,
            "entropy_reward": entropy_reward,
            "entropy_details": entropy_details,
            "order_log_probs": graph_data["order_log_probs"],
            "edge_log_probs_list": graph_data["edge_log_probs_list"],
            "connections_list": graph_data["connections_list"],
            "node_order": graph_data["node_order"],
            "predicted": postprocessed_answer,
            "target": correct_answer,
        })

    return results


def bootstrap_candidate_graphs(args, graph_dir):
    """
    Creates unscreened candidate graph cache from traditional topology templates.
    This path does not run LLM inference; it only builds PyG graphs with task
    metadata so order/edge diffusion has supervised structures to learn from.
    """
    graph_dir.mkdir(parents=True, exist_ok=True)
    dataset = MMLUDataset("MMLU", "dev")
    available_roles = list(ROLE_DESCRIPTION.keys())
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[: min(args.bootstrap_train_size, len(indices))]

    saved = 0
    configs = []
    for agent_num in range(args.bootstrap_agent_min, args.bootstrap_agent_max + 1):
        for mode in args.bootstrap_modes:
            configs.append((mode, agent_num))

    for record_idx in tqdm(indices, desc="Bootstrapping MMLU graph cache"):
        record = dataset[record_idx]
        input_dict = dataset.record_to_input(record)
        for mode, agent_num in configs:
            kwargs = get_kwargs(mode, agent_num)
            random_roles = random.choices(available_roles, k=agent_num)
            kwargs["node_kwargs"] = [{"role": role} for role in random_roles]
            graph = Graph(
                domain=args.domain,
                llm_name=args.llm_name,
                agent_names=["AnalyzeAgent"] * agent_num,
                decision_method=args.decision_method,
                **kwargs,
            )
            flow_graph = graph.to_pyg_graph(input_dict)
            filename = f"mmlu_bootstrap_{record_idx}_{mode}_{agent_num}_{saved}.pt"
            save_graph_with_features(
                flow_graph,
                str(graph_dir / filename),
                {
                    "mode": mode,
                    "num_nodes": agent_num,
                    "is_correct": False,
                    "question": input_dict["task"],
                    "record": record,
                    "bootstrap": True,
                },
            )
            saved += 1

    print(f"Bootstrapped {saved} unscreened candidate graphs into {graph_dir}")


def build_framework(args, pyg_graph_dataset, num_role_types):
    task_dim = pyg_graph_dataset.task_embedding_dim
    if task_dim is None:
        raise ValueError("task_embedding is missing. Regenerate cached MMLU graphs with task text embeddings.")

    legacy_order_net = DiffusionOrderingNetwork(
        node_feature_dim=1,
        num_node_types=num_role_types,
        num_edge_types=1,
        task_feature_dim=task_dim,
        num_layers=3,
        out_channels=1,
        hidden_dim=32,
        device=args.device,
    )
    topology_order_net = TaskAwareTopologyOrderNetwork(
        node_feature_dim=1,
        num_node_types=num_role_types,
        num_edge_types=1,
        task_feature_dim=task_dim,
        num_layers=3,
        hidden_dim=args.hidden_dim_order,
        device=args.device,
    )
    denoising_net = SemanticEdgeDenoisingNetwork(
        node_feature_dim=1,
        edge_feature_dim=1,
        task_feature_dim=task_dim,
        num_node_types=num_role_types,
        num_edge_types=1,
        num_layers=3,
        hidden_dim=args.hidden_dim_edge,
        device=args.device,
    )
    return GDFramework(
        dataset=pyg_graph_dataset,
        denoising_network=denoising_net,
        diffusion_ordering_network=legacy_order_net,
        topology_order_network=topology_order_net,
        device=args.device,
    )


def load_cached_graph_dataset(args):
    graph_dir = mas_ROOT / args.graph_dir if not os.path.isabs(args.graph_dir) else args.graph_dir
    role_dir = mas_ROOT / args.role_dir if not os.path.isabs(args.role_dir) else args.role_dir

    graph_dir = Path(graph_dir)
    role_dir = Path(role_dir)
    if not graph_dir.exists() or not any(graph_dir.glob("*.pt")):
        if args.bootstrap_graphs:
            bootstrap_candidate_graphs(args, graph_dir)
        else:
            raise FileNotFoundError(
                f"No cached MMLU graph files found in {graph_dir}. "
                "Pass --bootstrap_graphs to create unscreened template graphs from the raw MMLU dev split, "
                "run the original run_mmlu.py once to generate screened cache/MMLU/graphs, "
                "or pass --graph_dir to your existing graph cache."
            )

    nx_dataset = load_graph_dataset(args, graph_dir=graph_dir, role_dir=role_dir)
    pyg_graph_dataset = PyGGraphDataset(nx_dataset.graph_list)
    return nx_dataset, pyg_graph_dataset


def save_semantic_edge_models(gd_framework, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(gd_framework.topology_order_network.state_dict(), os.path.join(model_dir, "topology_order_network.pt"))
    torch.save(gd_framework.denoising_network.state_dict(), os.path.join(model_dir, "semantic_edge_denoising_network.pt"))


async def train_semantic_edge_diffusion(args, gd_framework, pyg_graph_dataset, id_to_role):
    dataloader = torch.utils.data.DataLoader(
        pyg_graph_dataset,
        batch_size=args.diffusion_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
    )

    order_optimizer = torch.optim.Adam(
        gd_framework.topology_order_network.parameters(),
        lr=args.lr_order,
        betas=(0.9, 0.999),
    )
    edge_optimizer = torch.optim.Adam(
        gd_framework.denoising_network.parameters(),
        lr=args.lr_edge,
        betas=(0.9, 0.999),
    )

    semantic_judge = build_semantic_judge_from_args(args)
    if semantic_judge is None:
        raise ValueError(
            "RL reward always includes average agent semantic entropy. Configure "
            "--semantic_judge_llm_name plus OPENAI_API_KEY/--semantic_judge_api_key, "
            "or --semantic_judge_base_url for an OpenAI-compatible local server."
        )

    print("Stage 1: SFT topology-order training")
    for epoch in range(args.order_epochs):
        total_loss = 0.0
        steps = 0
        for batch in dataloader:
            target_orders = [topological_order_from_pyg(graph) for graph in batch]
            loss = gd_framework.train_order_step(batch, target_orders=target_orders)
            order_optimizer.zero_grad()
            loss.backward()
            order_optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        print(f"Order epoch {epoch + 1}/{args.order_epochs}, loss={total_loss / max(steps, 1):.4f}")

    print("Stage 2: SFT semantic edge denoising/pruning training")
    for epoch in range(args.edge_epochs):
        total_loss = 0.0
        steps = 0
        for batch in dataloader:
            node_orders = [topological_order_from_pyg(graph) for graph in batch]
            loss = gd_framework.train_edge_step(
                batch,
                node_orders=node_orders,
                prune_low_gain_threshold=semantic_prune_threshold(args),
            )
            edge_optimizer.zero_grad()
            loss.backward()
            edge_optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        print(f"Edge epoch {epoch + 1}/{args.edge_epochs}, loss={total_loss / max(steps, 1):.4f}")

    print("Stage 3: reward fine-tuning for topology order and edge denoising")
    for epoch in range(args.rl_epochs):
        total_order_sft = 0.0
        total_edge_sft = 0.0
        total_order_rl = 0.0
        total_edge_rl = 0.0
        total_reward = 0.0
        total_entropy = 0.0
        entropy_count = 0
        reward_steps = 0
        steps = 0

        for batch_idx, batch in enumerate(dataloader):
            target_orders = [topological_order_from_pyg(graph) for graph in batch]
            order_sft_loss = gd_framework.train_order_step(batch, target_orders=target_orders)
            edge_sft_loss = gd_framework.train_edge_step(
                batch,
                node_orders=target_orders,
                prune_low_gain_threshold=semantic_prune_threshold(args),
            )

            order_rl_loss = torch.tensor(0.0, device=args.device)
            edge_rl_loss = torch.tensor(0.0, device=args.device)
            batch_reward = 0.0
            batch_entropy = 0.0
            batch_entropy_count = 0

            if batch_idx % args.update_freq == 0:
                sample_size = max(1, int(args.sample_ratio * len(batch)))
                sample_indices = random.sample(range(len(batch)), sample_size)
                sample_batch = [batch[i] for i in sample_indices]

                prompt_before = int(PromptTokens.instance().value)
                completion_before = int(CompletionTokens.instance().value)
                cost_before = float(Cost.instance().value)

                evaluation_results = await evaluate_generated_graphs_for_rl(
                    args,
                    gd_framework,
                    sample_batch,
                    id_to_role,
                    semantic_judge=semantic_judge,
                )

                if evaluation_results:
                    order_losses = []
                    edge_losses = []
                    for result in evaluation_results:
                        order_log_prob, order_count = order_policy_log_prob(result["order_log_probs"])
                        edge_log_prob, edge_count = edge_policy_log_prob(result["edge_log_probs_list"])
                        order_losses.append(policy_gradient_loss(
                            order_log_prob,
                            order_count,
                            result["advantage"],
                            args.device,
                        ))
                        edge_losses.append(policy_gradient_loss(
                            edge_log_prob,
                            edge_count,
                            result["advantage"],
                            args.device,
                        ))
                        batch_reward += float(result["generated_reward"])
                        if result["avg_semantic_entropy"] is not None:
                            batch_entropy += float(result["avg_semantic_entropy"])
                            batch_entropy_count += 1

                    order_rl_loss = torch.stack(order_losses).mean()
                    edge_rl_loss = torch.stack(edge_losses).mean()
                    reward_steps += len(evaluation_results)

                prompt_delta = int(PromptTokens.instance().value) - prompt_before
                completion_delta = int(CompletionTokens.instance().value) - completion_before
                cost_delta = float(Cost.instance().value) - cost_before
                avg_reward = batch_reward / max(len(evaluation_results), 1) if evaluation_results else 0.0
                avg_entropy = batch_entropy / batch_entropy_count if batch_entropy_count else None
                entropy_text = f"{avg_entropy:.4f}" if avg_entropy is not None else "n/a"
                print(
                    f"RL epoch {epoch + 1}/{args.rl_epochs}, batch {batch_idx}: "
                    f"reward={avg_reward:.4f}, avg_semantic_entropy={entropy_text}, "
                    f"tokens=P(+{prompt_delta}), C(+{completion_delta}), cost=+${cost_delta:.4f}"
                )

            total_loss = (
                args.rl_sft_weight * (order_sft_loss + edge_sft_loss)
                + args.lambda_order_rl * order_rl_loss
                + args.lambda_edge_rl * edge_rl_loss
            )

            order_optimizer.zero_grad()
            edge_optimizer.zero_grad()
            total_loss.backward()
            order_optimizer.step()
            edge_optimizer.step()

            total_order_sft += float(order_sft_loss.item())
            total_edge_sft += float(edge_sft_loss.item())
            total_order_rl += float(order_rl_loss.item())
            total_edge_rl += float(edge_rl_loss.item())
            total_reward += batch_reward
            total_entropy += batch_entropy
            entropy_count += batch_entropy_count
            steps += 1

        avg_epoch_reward = total_reward / max(reward_steps, 1)
        avg_epoch_entropy = total_entropy / entropy_count if entropy_count else None
        entropy_text = f"{avg_epoch_entropy:.4f}" if avg_epoch_entropy is not None else "n/a"
        print(
            f"RL epoch {epoch + 1}/{args.rl_epochs} summary: "
            f"order_sft={total_order_sft / max(steps, 1):.4f}, "
            f"edge_sft={total_edge_sft / max(steps, 1):.4f}, "
            f"order_rl={total_order_rl / max(steps, 1):.4f}, "
            f"edge_rl={total_edge_rl / max(steps, 1):.4f}, "
            f"reward={avg_epoch_reward:.4f}, avg_semantic_entropy={entropy_text}, "
            f"tokens=P({int(PromptTokens.instance().value)}), C({int(CompletionTokens.instance().value)}), "
            f"cost=${Cost.instance().value:.4f}"
        )

    save_semantic_edge_models(gd_framework, args.model_dir)
    print(f"Saved models to {args.model_dir}")


def save_evaluation_result(summary_text):
    result_path = Path("mmlu_semantic_edge_accuracy.txt")
    with open(result_path, "a+", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    return result_path


async def evaluate_semantic_edge_diffusion(args, gd_framework, pyg_graph_dataset, id_to_role):
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentence_model.to(args.device)
    dataset = MMLUDataset("MMLU", "val")
    accuracy = Accuracy()
    total_correct = 0
    total_tasks = 0

    cached_graphs = [pyg_graph_dataset[idx] for idx in range(len(pyg_graph_dataset))]

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for idx, record in enumerate(dataset):
            if args.limit_questions is not None and idx >= args.limit_questions:
                break
            records.append((idx, record))
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = min(len(dataset), args.limit_questions) if args.limit_questions is not None else len(dataset)
    num_batches = math.ceil(data_len / args.batch_size)

    for batch_idx, record_batch in tqdm(enumerate(eval_loader(args.batch_size)), total=num_batches):
        tasks = []
        batch_metadata = []
        for item_idx, (record_idx, record) in enumerate(record_batch):
            input_dict = dataset.record_to_input(record)
            task_text = input_dict["task"]
            task_embedding = torch.tensor(sentence_model.encode(task_text, device=args.device)).float().to(args.device)

            template_graph = cached_graphs[(batch_idx * args.batch_size + item_idx) % len(cached_graphs)]
            node_types = role_ids_for_generation(template_graph, args.agent_nums, args.device)
            generated_graph = gd_framework.generate_graph(
                num_nodes=args.agent_nums,
                task_embedding=task_embedding,
                node_types=node_types,
            )
            generated_graph.num_nodes = args.agent_nums
            default_role = next(iter(id_to_role.values()), "Normal")
            generated_graph.x = [
                {"role": id_to_role.get(int(node_type.item()), default_role)}
                for node_type in node_types
            ]

            test_graph = TestGraph(
                domain=args.domain,
                llm_name=args.llm_name,
                decision_method=args.decision_method,
                pyg_data=generated_graph,
            )
            tasks.append(asyncio.create_task(test_graph.arun(input_dict, args.num_rounds)))
            batch_metadata.append({
                "index": record_idx,
                "question": str(record["question"]),
                "task": task_text,
                "target": dataset.record_to_target_answer(record),
            })

        raw_answers = await asyncio.gather(*tasks)
        for raw_answer, metadata in zip(raw_answers, batch_metadata):
            answer = dataset.postprocess_answer(raw_answer)
            target = metadata["target"]
            is_correct = accuracy.update(answer, target)
            total_correct += int(is_correct)
            total_tasks += 1

    acc = accuracy.get() * 100
    final_cost = Cost.instance().value
    final_prompt_tokens = PromptTokens.instance().value
    final_completion_tokens = CompletionTokens.instance().value
    summary_text = (
        f"Total tasks: {total_tasks}\n"
        f"Final accuracy : {acc:.2f}%\n"
        f"Total cost: ${final_cost:.6f}\n"
        f"Total Prompt Tokens: {int(final_prompt_tokens)}\n"
        f"Total Completion Tokens: {int(final_completion_tokens)}"
    )
    print("\n" + "=" * 50 + "\nEvaluation Summary")
    print(summary_text)
    print("-" * 50)
    result_path = save_evaluation_result(summary_text)
    print(f"Saved evaluation summary to {result_path}")


async def main():
    args = parse_args()
    set_seed(args.seed)

    nx_dataset, pyg_graph_dataset = load_cached_graph_dataset(args)
    gd_framework = build_framework(args, pyg_graph_dataset, num_role_types=len(nx_dataset.role_to_id))

    await train_semantic_edge_diffusion(args, gd_framework, pyg_graph_dataset, nx_dataset.id_to_role)
    await evaluate_semantic_edge_diffusion(args, gd_framework, pyg_graph_dataset, nx_dataset.id_to_role)


if __name__ == "__main__":
    asyncio.run(main())
