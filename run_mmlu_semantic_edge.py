import argparse
import asyncio
import math
import os
import random
from typing import Iterator, List, Optional, Any

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
from utils import get_kwargs, save_graph_with_features


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
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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

    from pathlib import Path

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


def load_semantic_edge_models(gd_framework, model_dir, device):
    gd_framework.topology_order_network.load_state_dict(
        torch.load(os.path.join(model_dir, "topology_order_network.pt"), map_location=device)
    )
    gd_framework.denoising_network.load_state_dict(
        torch.load(os.path.join(model_dir, "semantic_edge_denoising_network.pt"), map_location=device)
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


async def evaluate_semantic_edge_diffusion(args, gd_framework, pyg_graph_dataset, id_to_role):
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentence_model.to(args.device)
    dataset = MMLUDataset("MMLU", "val")
    accuracy = Accuracy()

    cached_graphs = [pyg_graph_dataset[idx] for idx in range(len(pyg_graph_dataset))]

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for idx, record in enumerate(dataset):
            if args.limit_questions is not None and idx >= args.limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = min(len(dataset), args.limit_questions) if args.limit_questions is not None else len(dataset)
    num_batches = math.ceil(data_len / args.batch_size)

    for batch_idx, record_batch in tqdm(enumerate(eval_loader(args.batch_size)), total=num_batches):
        tasks = []
        for item_idx, record in enumerate(record_batch):
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

        raw_answers = await asyncio.gather(*tasks)
        for raw_answer, record in zip(raw_answers, record_batch):
            answer = dataset.postprocess_answer(raw_answer)
            target = dataset.record_to_target_answer(record)
            accuracy.update(answer, target)

        print(
            f"Accuracy={accuracy.get() * 100:.2f}% | "
            f"Cost=${Cost.instance().value:.4f} | "
            f"Tokens=P({int(PromptTokens.instance().value)}), C({int(CompletionTokens.instance().value)})"
        )

    print(f"Final accuracy: {accuracy.get() * 100:.2f}%")


async def main():
    args = parse_args()
    set_seed(args.seed)

    nx_dataset, pyg_graph_dataset = load_cached_graph_dataset(args)
    gd_framework = build_framework(args, pyg_graph_dataset, num_role_types=len(nx_dataset.role_to_id))

    if args.eval_only:
        load_semantic_edge_models(gd_framework, args.model_dir, args.device)
    else:
        await train_semantic_edge_diffusion(args, gd_framework, pyg_graph_dataset)

    await evaluate_semantic_edge_diffusion(args, gd_framework, pyg_graph_dataset, nx_dataset.id_to_role)


if __name__ == "__main__":
    asyncio.run(main())
