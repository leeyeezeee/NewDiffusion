import math
import sys
import os
import torch
import pickle
import random
import json
import numpy as np
import asyncio
from typing import Any, Dict, List
from torch_geometric.utils import to_dense_adj

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

def get_kwargs(mode: str, N: int):
    initial_spatial_probability = 0.5
    initial_temporal_probability = 0.5
    fixed_spatial_masks = None
    fixed_temporal_masks = None
    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
        adj = [[0] * N for _ in range(N)]
        base = N // layer_num
        rem = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base + (1 if i < rem else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            for j in range(N):
                if layers[j] == layers[i] + 1:
                    adj[i][j] = 1
        return adj

    def generate_mesh_graph(N):
        if N > 4 and int(math.sqrt(N))**2 == N:
            size = int(math.sqrt(N))
            adj = [[0] * N for _ in range(N)]
            for i in range(N):
                if (i + 1) % size != 0:
                    adj[i][i+1] = adj[i+1][i] = 1
                if i < N - size:
                    adj[i][i+size] = adj[i+size][i] = 1
            return adj
        return [[1 if i != j else 0 for i in range(N)] for j in range(N)]

    def generate_star_graph(N):
        adj = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adj[0][i] = adj[i][0] = 1
        return adj

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Normal'}]
    elif mode in ('FullConnected', 'FakeFullConnected', 'FakeAGFull'):
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1] * N for _ in range(N)]
    elif mode in ('Random', 'FakeRandom', 'FakeAGRandom'):
        fixed_spatial_masks = [[random.randint(0,1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0,1) for _ in range(N)] for _ in range(N)]
    elif mode in ('Chain', 'FakeChain'):
        fixed_spatial_masks = [[1 if abs(i-j)==1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==j else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]
    elif mode in ('Mesh', 'FakeMesh'):
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]
    elif mode in ('Star', 'FakeStar'):
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]

    elif 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': 'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': None} for i in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs
    }


def save_graph_with_features(flow_graph, filepath, metadata):
    """
    Attach metadata to the graph and save it.
    """
    for key, value in metadata.items():
        setattr(flow_graph, key, value)
    torch.save(flow_graph, filepath)


def build_semantic_judge_from_args(args):
    num_samples = getattr(args, "semantic_entropy_samples", 0)
    if num_samples <= 1:
        return None

    from model.semantic_entropy import SemanticEntailmentJudge

    judge = SemanticEntailmentJudge(
        llm_name=getattr(args, "semantic_judge_llm_name", None),
        api_key=getattr(args, "semantic_judge_api_key", ""),
        base_url=getattr(args, "semantic_judge_base_url", ""),
        max_concurrency=getattr(args, "semantic_judge_max_concurrency", None),
    )
    if not judge.is_configured:
        print(
            "Semantic entropy is enabled but SemanticEntailmentJudge is not configured; "
            "set OPENAI_API_KEY or --semantic_judge_api_key. Skipping edge semantic gain."
        )
        return None
    return judge


def _latest_node_output(node):
    outputs = getattr(node, "outputs", [])
    if isinstance(outputs, list):
        return outputs[-1] if outputs else None
    return outputs


def _node_info_from_output(node, output):
    return {"role": getattr(node, "role", ""), "output": output}


def _current_spatial_info(target_node) -> Dict[str, Dict[str, Any]]:
    spatial_info = {}
    for predecessor in getattr(target_node, "spatial_predecessors", []):
        output = _latest_node_output(predecessor)
        if output is not None:
            spatial_info[predecessor.id] = _node_info_from_output(predecessor, output)
    return spatial_info


def _current_temporal_info(target_node) -> Dict[str, Dict[str, Any]]:
    temporal_info = {}
    for predecessor in getattr(target_node, "temporal_predecessors", []):
        outputs = getattr(predecessor, "last_memory", {}).get("outputs", [])
        output = outputs[-1] if isinstance(outputs, list) and outputs else outputs
        if output is not None:
            temporal_info[predecessor.id] = _node_info_from_output(predecessor, output)
    return temporal_info


def _flatten_outputs(results: List[Any]) -> List[Any]:
    outputs = []
    for result in results:
        if isinstance(result, list):
            outputs.extend(result)
        else:
            outputs.append(result)
    return outputs


async def _sample_node_outputs(
    node,
    input_data: Dict[str, Any],
    spatial_info: Dict[str, Dict[str, Any]],
    temporal_info: Dict[str, Dict[str, Any]],
    num_samples: int,
) -> List[Any]:
    tasks = []
    for _ in range(max(1, int(num_samples))):
        tasks.append(
            asyncio.create_task(
                node._async_execute(
                    input_data,
                    dict(spatial_info),
                    dict(temporal_info),
                )
            )
        )
    return _flatten_outputs(await asyncio.gather(*tasks, return_exceptions=False))


async def attach_edge_semantic_gains(
    test_graph,
    flow_graph,
    input_data: Dict[str, Any],
    question: str,
    judge,
    num_entropy_samples: int,
    negative_reward_scale: float = 1.0,
    nonpositive_penalty: float = 0.01,
):
    """
    Compute and attach edge-level semantic entropy gains during graph collection.

    The returned tensor is aligned with flow_graph.edge_index: for edge src -> dst,
    gain = entropy_without_edge - entropy_with_edge.
    """
    if judge is None or num_entropy_samples <= 1 or flow_graph.edge_index.numel() == 0:
        return flow_graph

    from model.semantic_entropy import semantic_uncertainty

    node_list = list(test_graph.nodes.values())
    gains = torch.zeros(flow_graph.edge_index.shape[1], dtype=torch.float)
    details = []
    after_cache = {}

    for edge_pos, edge_index in enumerate(flow_graph.edge_index.T):
        src_idx = int(edge_index[0].item())
        dst_idx = int(edge_index[1].item())
        if src_idx >= len(node_list) or dst_idx >= len(node_list):
            continue

        source_node = node_list[src_idx]
        target_node = node_list[dst_idx]
        spatial_info = _current_spatial_info(target_node)
        temporal_info = _current_temporal_info(target_node)
        if source_node.id not in spatial_info:
            continue

        without_edge_spatial_info = dict(spatial_info)
        without_edge_spatial_info.pop(source_node.id, None)

        try:
            before_outputs = await _sample_node_outputs(
                target_node,
                input_data,
                without_edge_spatial_info,
                temporal_info,
                num_entropy_samples,
            )
            after_key = target_node.id
            if after_key not in after_cache:
                after_outputs = await _sample_node_outputs(
                    target_node,
                    input_data,
                    spatial_info,
                    temporal_info,
                    num_entropy_samples,
                )
                after_cache[after_key] = await semantic_uncertainty(
                    question,
                    after_outputs,
                    judge,
                )

            before_entropy, before_labels = await semantic_uncertainty(
                question,
                before_outputs,
                judge,
            )
            after_entropy, after_labels = after_cache[after_key]
            entropy_delta = before_entropy - after_entropy
            reward = (
                entropy_delta
                if entropy_delta > 0
                else negative_reward_scale * entropy_delta - nonpositive_penalty
            )
            gains[edge_pos] = float(reward)
            details.append(
                {
                    "edge_pos": edge_pos,
                    "source": src_idx,
                    "target": dst_idx,
                    "before_entropy": before_entropy,
                    "after_entropy": after_entropy,
                    "entropy_delta": entropy_delta,
                    "reward": reward,
                    "before_labels": before_labels,
                    "after_labels": after_labels,
                }
            )
        except Exception as exc:
            print(f"Failed to compute semantic entropy for edge {src_idx}->{dst_idx}: {exc}")

    flow_graph.edge_semantic_gain = gains
    flow_graph.edge_entropy_gain = gains.clone()
    flow_graph.edge_semantic_details = details
    return flow_graph


def compute_effective_size_reward(x, edge_index, alpha=0.7):
    """
    Compute effective size for each node in a directed graph.
    For directed graphs, effective size has two components:
    - In-neighbor effective size = |n_in| - |tie_in| / |n_in|
    - Out-neighbor effective size = |n_out| - |tie_out| / |n_out|
    
    Returns normalized effective size [0,1].
    """
    num_nodes = x.shape[0]
    effective_sizes = torch.zeros(num_nodes, device=x.device)
    
    # Convert to dense adjacency matrix (directed)
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    
    for i in range(num_nodes):
        # Get in-neighbors (nodes pointing TO node i)
        in_neighbors = torch.where(adj[:, i] > 0)[0]  # adj[:, i] means edges TO i
        n_in = len(in_neighbors)
        
        # Get out-neighbors (nodes pointing FROM node i)
        out_neighbors = torch.where(adj[i, :] > 0)[0]  # adj[i, :] means edges FROM i
        n_out = len(out_neighbors)
        
        # Compute in-neighbor effective size
        if n_in > 0:
            # Count edges among in-neighbors (ties)
            in_neighbor_adj = adj[in_neighbors][:, in_neighbors]
            in_neighbor_x = x[in_neighbors]

            if in_neighbor_x.dim() > 1:
                in_neighbor_x = in_neighbor_x.squeeze(-1)
                
            if in_neighbor_x.dim() == 0:
                in_neighbor_x = in_neighbor_x.unsqueeze(0)

            role_matrix = (in_neighbor_x.unsqueeze(0) == in_neighbor_x.unsqueeze(1))
            role_matrix.fill_diagonal_(False)
            tie_mask = (in_neighbor_adj > 0) & role_matrix
            tie_in = tie_mask.sum().item()

            eff_size_in = n_in - (tie_in / n_in)
        else:
            eff_size_in = 0.0
        
        # Compute out-neighbor effective size
        if n_out > 0:
            # Count edges among out-neighbors (ties)
            out_neighbor_adj = adj[out_neighbors][:, out_neighbors]
            out_neighbor_x = x[out_neighbors]

            if out_neighbor_x.dim() > 1:
                out_neighbor_x = out_neighbor_x.squeeze(-1)
                
            if out_neighbor_x.dim() == 0:
                out_neighbor_x = out_neighbor_x.unsqueeze(0)

            role_matrix = (out_neighbor_x.unsqueeze(0) == out_neighbor_x.unsqueeze(1))
            role_matrix.fill_diagonal_(False)
            tie_mask = (out_neighbor_adj > 0) & role_matrix
            tie_out = tie_mask.sum().item()
            eff_size_out = n_out - (tie_out / n_out)
        else:
            eff_size_out = 0.0
        
        # Combine in and out effective sizes (sum)
        effective_sizes_combined = eff_size_in * (1 - alpha) + eff_size_out * alpha

        max_possible = num_nodes - 1
        if max_possible > 0:
            effective_sizes[i] = effective_sizes_combined / max_possible
        else:
            effective_sizes[i] = 0.0
    
    effective_sizes_reward = effective_sizes.mean().item()

    return effective_sizes_reward

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
