import argparse
import asyncio
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import networkx as nx
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from mas.graph.graph import TestGraph
from mas.utils.const import mas_ROOT
from mas.utils.globals import CompletionTokens, Cost, PromptTokens
from model.denoising import SemanticEdgeDenoisingNetwork
from model.gd import GDFramework
from model.ordering import DiffusionOrderingNetwork, TaskAwareTopologyOrderNetwork
from process_datasets import PyGGraphDataset, load_graph_dataset


@dataclass
class SemanticEdgeDatasetSpec:
    dataset: str
    split: str
    cache_name: str
    result_prefix: str
    default_domain: str
    default_decision_method: str
    default_batch_size: int
    default_train_set_size: int
    default_graph_dir: str
    default_role_dir: str
    default_model_dir: str
    load_eval_records: Callable[[Any], List[Any]]
    make_input: Callable[[Any], Dict[str, str]]
    task_text: Callable[[Any], str]
    target_answer: Callable[[Any], Any]
    predict_answer: Callable[[Any, Any], Any]
    is_correct: Callable[[Any, Any, Any, Any], bool]
    question_text: Callable[[Any], str]
    default_limit_questions: Optional[int] = None


def parse_semantic_edge_args(description: str, spec: SemanticEdgeDatasetSpec):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--domain", type=str, default=spec.default_domain)
    parser.add_argument("--dataset", type=str, default=spec.dataset)
    parser.add_argument("--decision_method", type=str, default=spec.default_decision_method)
    parser.add_argument("--batch_size", type=int, default=spec.default_batch_size)
    parser.add_argument("--diffusion_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--order_epochs", type=int, default=5)
    parser.add_argument("--edge_epochs", type=int, default=15)
    parser.add_argument("--agent_nums", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--train_set_size", type=int, default=spec.default_train_set_size)
    parser.add_argument("--limit_questions", type=int, default=spec.default_limit_questions)
    parser.add_argument("--lr_order", type=float, default=5e-5)
    parser.add_argument("--lr_edge", type=float, default=1e-5)
    parser.add_argument("--hidden_dim_order", type=int, default=128)
    parser.add_argument("--hidden_dim_edge", type=int, default=256)
    parser.add_argument("--graph_dir", type=str, default=str(mas_ROOT / spec.default_graph_dir))
    parser.add_argument("--role_dir", type=str, default=str(mas_ROOT / spec.default_role_dir))
    parser.add_argument("--model_dir", type=str, default=str(mas_ROOT / spec.default_model_dir))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def shuffled_eval_split(records: List[Any], args) -> List[Any]:
    indices = list(range(len(records)))
    random.shuffle(indices)
    return [records[i] for i in indices[args.train_set_size:]]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else mas_ROOT / path


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
    return list(range(int(data.num_nodes)))


def role_ids_for_generation(graph, agent_nums, device):
    node_types = graph.x.squeeze().long().to(device).view(-1)
    if node_types.numel() >= agent_nums:
        return node_types[:agent_nums]
    pad = torch.zeros(agent_nums - node_types.numel(), dtype=torch.long, device=device)
    return torch.cat([node_types, pad], dim=0)


def build_framework(args, pyg_graph_dataset, num_role_types):
    task_dim = pyg_graph_dataset.task_embedding_dim
    if task_dim is None:
        raise ValueError(
            "task_embedding is missing. Regenerate cached graphs with task text embeddings."
        )

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


def load_cached_graph_dataset(args, spec: SemanticEdgeDatasetSpec):
    graph_dir = _resolve_path(args.graph_dir)
    role_dir = _resolve_path(args.role_dir)
    if not graph_dir.exists() or not any(graph_dir.glob("*.pt")):
        raise FileNotFoundError(
            f"No cached {spec.dataset} graph files found in {graph_dir}. "
            f"Run run_{spec.dataset}.py --collect_correct_semantic_graphs first, "
            "or pass --graph_dir to an existing graph cache."
        )

    nx_dataset = load_graph_dataset(args, graph_dir=graph_dir, role_dir=role_dir)
    pyg_graph_dataset = PyGGraphDataset(nx_dataset.graph_list)
    if len(pyg_graph_dataset) == 0:
        raise ValueError(f"No usable graph data loaded from {graph_dir}.")

    has_semantic_gain = any(
        hasattr(graph, attr_name)
        for graph in pyg_graph_dataset
        for attr_name in ("edge_semantic_gain", "semantic_gain", "edge_entropy_gain")
    )
    if not has_semantic_gain:
        print("Warning: no edge semantic gain attributes found; edge training will use zero semantic gains.")

    return nx_dataset, pyg_graph_dataset


def save_semantic_edge_models(gd_framework, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        gd_framework.topology_order_network.state_dict(),
        os.path.join(model_dir, "topology_order_network.pt"),
    )
    torch.save(
        gd_framework.denoising_network.state_dict(),
        os.path.join(model_dir, "semantic_edge_denoising_network.pt"),
    )


async def train_semantic_edge_diffusion(args, gd_framework, pyg_graph_dataset):
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

    print("Stage 1: supervised topology-order training")
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

    print("Stage 2: edge-only denoising training")
    for epoch in range(args.edge_epochs):
        total_loss = 0.0
        steps = 0
        for batch in dataloader:
            node_orders = [topological_order_from_pyg(graph) for graph in batch]
            loss = gd_framework.train_edge_step(batch, node_orders=node_orders)
            edge_optimizer.zero_grad()
            loss.backward()
            edge_optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        print(f"Edge epoch {epoch + 1}/{args.edge_epochs}, loss={total_loss / max(steps, 1):.4f}")

    save_semantic_edge_models(gd_framework, args.model_dir)
    print(f"Saved models to {args.model_dir}")


def _batch_iter(records: List[Any], batch_size: int) -> Iterator[List[Any]]:
    batch = []
    for idx, record in enumerate(records):
        batch.append((idx, record))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _jsonable(value):
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def save_evaluation_result(args, spec: SemanticEdgeDatasetSpec, result):
    result_dir = mas_ROOT / "cache" / spec.cache_name / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_name = _safe_filename(args.llm_name)
    filename = f"{spec.result_prefix}_{llm_name}_{timestamp}.json"
    result_path = result_dir / filename
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, ensure_ascii=False, indent=2)
    return result_path


async def evaluate_semantic_edge_diffusion(args, spec, gd_framework, pyg_graph_dataset, id_to_role):
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentence_model.to(args.device)
    records = spec.load_eval_records(args)
    if args.limit_questions is not None:
        records = records[: args.limit_questions]

    total_correct = 0
    item_results = []
    cached_graphs = [pyg_graph_dataset[idx] for idx in range(len(pyg_graph_dataset))]
    num_batches = math.ceil(len(records) / args.batch_size)

    for batch_idx, record_batch in tqdm(enumerate(_batch_iter(records, args.batch_size)), total=num_batches):
        tasks = []
        batch_metadata = []
        for item_idx, (record_idx, record) in enumerate(record_batch):
            input_dict = spec.make_input(record)
            task_text = spec.task_text(record)
            task_embedding = torch.tensor(
                sentence_model.encode(task_text, device=args.device),
                device=args.device,
            ).float()

            template_graph = cached_graphs[(batch_idx * args.batch_size + item_idx) % len(cached_graphs)]
            node_types = role_ids_for_generation(template_graph, args.agent_nums, args.device)
            generated_graph = gd_framework.generate_graph(
                num_nodes=args.agent_nums,
                task_embedding=task_embedding,
                node_types=node_types,
            )
            generated_graph.num_nodes = args.agent_nums
            generated_graph.x = [
                {"role": id_to_role.get(int(node_type.item()), next(iter(id_to_role.values())))}
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
                "question": spec.question_text(record),
                "task": task_text,
                "target": spec.target_answer(record),
                "record": record,
            })

        raw_answers = await asyncio.gather(*tasks, return_exceptions=True)
        for raw_answer, metadata in zip(raw_answers, batch_metadata):
            if isinstance(raw_answer, BaseException):
                predicted = ""
                is_correct = False
                error = str(raw_answer)
            else:
                predicted = spec.predict_answer(raw_answer, metadata["record"])
                is_correct = spec.is_correct(predicted, metadata["target"], raw_answer, metadata["record"])
                error = None
            total_correct += int(is_correct)
            item_results.append({
                "index": metadata["index"],
                "question": metadata["question"],
                "task": metadata["task"],
                "target": metadata["target"],
                "raw_answer": raw_answer,
                "predicted": predicted,
                "is_correct": bool(is_correct),
                "error": error,
            })

        running_accuracy = total_correct / max(len(item_results), 1) * 100
        print(
            f"Accuracy={running_accuracy:.2f}% | "
            f"Cost=${Cost.instance().value:.4f} | "
            f"Tokens=P({int(PromptTokens.instance().value)}), C({int(CompletionTokens.instance().value)})"
        )

    total_tasks = len(item_results)
    accuracy = total_correct / total_tasks if total_tasks else 0.0
    print(f"Final accuracy: {accuracy * 100:.2f}% ({total_correct}/{total_tasks})")
    result = {
        "dataset": args.dataset,
        "split": spec.split,
        "llm_name": args.llm_name,
        "domain": args.domain,
        "decision_method": args.decision_method,
        "agent_nums": args.agent_nums,
        "num_rounds": args.num_rounds,
        "train_set_size": args.train_set_size,
        "limit_questions": args.limit_questions,
        "model_dir": args.model_dir,
        "graph_dir": args.graph_dir,
        "total_tasks": total_tasks,
        "correct": total_correct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "cost": Cost.instance().value,
        "prompt_tokens": int(PromptTokens.instance().value),
        "completion_tokens": int(CompletionTokens.instance().value),
        "items": item_results,
    }
    result_path = save_evaluation_result(args, spec, result)
    print(f"Saved evaluation result to {result_path}")
    return result


async def run_semantic_edge_pipeline(args, spec: SemanticEdgeDatasetSpec):
    set_seed(args.seed)
    nx_dataset, pyg_graph_dataset = load_cached_graph_dataset(args, spec)
    gd_framework = build_framework(args, pyg_graph_dataset, num_role_types=len(nx_dataset.role_to_id))
    await train_semantic_edge_diffusion(args, gd_framework, pyg_graph_dataset)
    await evaluate_semantic_edge_diffusion(
        args,
        spec,
        gd_framework,
        pyg_graph_dataset,
        nx_dataset.id_to_role,
    )
