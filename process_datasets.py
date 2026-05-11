import os
import glob
import torch
import networkx as nx
import pickle
import random

import torch.utils.data
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from sentence_transformers import SentenceTransformer
from mas.prompt.mmlu_prompt_set import ROLE_DESCRIPTION as MMLU_ROLE_DESCRIPTION
from mas.prompt.humaneval_prompt_set import ROLE_DESCRIPTION as HUMAN_ROLE_DESCRIPTION
from mas.prompt.aqua_prompt_set import ROLE_DESCRIPTION as AQUA_ROLE_DESCRIPTION
from mas.prompt.gsm8k_prompt_set import ROLE_DESCRIPTION as GSM8K_ROLE_DESCRIPTION


def precompute_role_embeddings(dsets, save_name, dirpath, device):
    """
    Precompute embeddings for roles defined in ROLE_DESCRIPTION.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.to(device)
    role_embeddings = {}

    if dsets == 'mmlu':
        ROLE_DESCRIPTION = MMLU_ROLE_DESCRIPTION
    elif dsets == 'humaneval':
        ROLE_DESCRIPTION = HUMAN_ROLE_DESCRIPTION
    elif dsets == 'aqua':
        ROLE_DESCRIPTION = AQUA_ROLE_DESCRIPTION
    elif dsets == 'gsm8k':
        ROLE_DESCRIPTION = GSM8K_ROLE_DESCRIPTION
    else:
        ROLE_DESCRIPTION = GSM8K_ROLE_DESCRIPTION

    for role, description in ROLE_DESCRIPTION.items():
        embedding = model.encode(f"{role}: {description.strip()}", device=device)
        role_embeddings[role] = torch.tensor(embedding)
        
    os.makedirs(dirpath, exist_ok=True)
    save_path = dirpath / save_name

    with open(str(save_path), 'wb') as f:
        pickle.dump(role_embeddings, f)

    print(f"Precomputed {len(role_embeddings)} role embeddings, saved to {str(save_path)}")
    return role_embeddings


class NXGraphDataset:
    """
    Adapter for graph data.
    """

    def __init__(self, args, graph_dir, role_dir, sample_size=0):
        self.graph_dir = graph_dir
        self.role_dir = role_dir
        self.device = args.device
        self.dataset = args.dataset
        cache_path = self.role_dir / 'precomputed_role_embeddings.pkl'

        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model.to(self.device)
        
        if not cache_path.exists():
            self.precomputed_embeddings = precompute_role_embeddings(args.dataset, save_name='precomputed_role_embeddings.pkl', dirpath=self.role_dir, device=self.device)
        else:
            with open(str(cache_path), 'rb') as f:
                self.precomputed_embeddings = pickle.load(f)
            print(f"Loaded {len(self.precomputed_embeddings)} precomputed embeddings")
        
        self.graph_list = self._load_and_convert_graphs(sample_size)

    def _load_and_convert_graphs(self, sample_size):
        """
        Load PyG graphs from disk, filter and sample, convert to NetworkX DAGs.
        """
        graph_files = list(self.graph_dir.glob('*.pt'))
        print(f"Found {len(graph_files)} graph files")

        if sample_size and sample_size > 0 and len(graph_files) > sample_size:
            random.seed(42)
            graph_files = random.sample(graph_files, sample_size)
        else:
            print(f"Using all {len(graph_files)} graph files")

        # Create global role mapping
        if self.dataset == 'mmlu':
            ROLE_DESCRIPTION = MMLU_ROLE_DESCRIPTION
        elif self.dataset == 'humaneval':
            ROLE_DESCRIPTION = HUMAN_ROLE_DESCRIPTION
        elif self.dataset == 'aqua':
            ROLE_DESCRIPTION = AQUA_ROLE_DESCRIPTION
        elif self.dataset == 'gsm8k':
            ROLE_DESCRIPTION = GSM8K_ROLE_DESCRIPTION
        else:
            ROLE_DESCRIPTION = GSM8K_ROLE_DESCRIPTION
            
        sorted_roles = sorted(ROLE_DESCRIPTION.keys())
        role_to_id = {role: i for i, role in enumerate(sorted_roles)}
        id_to_role = {int(i): role for i, role in enumerate(sorted_roles)}
        self.role_to_id = role_to_id
        self.id_to_role = id_to_role

        nx_graphs = []
        for file in graph_files:
            try:
                pyg_graph = torch.load(file, weights_only=False)
                num_nodes = pyg_graph.num_nodes
                nx_graph = nx.DiGraph()
                nx_graph.add_nodes_from(range(num_nodes))
                nx_graph.role_embeddings = {}
                task = getattr(pyg_graph, "question", "")
                is_correct = getattr(pyg_graph, "is_correct", False)
                nx_graph.graph['is_correct'] = is_correct
                nx_graph.graph['record'] = getattr(pyg_graph, "record", None)
                
                for i, node_data in enumerate(pyg_graph.x):
                    role = node_data.get('role')
                    embedding = self.precomputed_embeddings[role]

                    nx_graph.nodes[i]['role'] = role
                    nx_graph.role_embeddings[i] = embedding
                    nx_graph.nodes[i]['feat'] = role_to_id.get(role, 0)

                if hasattr(pyg_graph, 'edge_index'):
                    edge_index = pyg_graph.edge_index.numpy()
                    edges = [(int(edge_index[0, j]), int(edge_index[1, j])) for j in range(edge_index.shape[1])]
                    nx_graph.add_edges_from(edges)
                    nx.set_edge_attributes(nx_graph, 1, "edge_attr")

                if not nx.is_directed_acyclic_graph(nx_graph):
                    try:
                        _ = list(nx.topological_sort(nx_graph))
                    except nx.NetworkXUnfeasible:
                        for u, v in list(nx_graph.edges()):
                            nx_graph.remove_edge(u, v)
                        for i in range(num_nodes - 1):
                            nx_graph.add_edge(i, i + 1, label=0)

                task_embedding = self.embedding_model.encode(task, device=self.device)
                nx_graph.graph['task_embedding'] = task_embedding
                nx_graph.graph['is_dag'] = True

                nx_graphs.append(nx_graph)

            except Exception as e:
                print(f"Error processing file {file}: {e}")

        dag_count = sum(nx.is_directed_acyclic_graph(g) for g in nx_graphs)
        print(f"DAG check: {dag_count}/{len(nx_graphs)} graphs are DAG")

        return nx_graphs

    def __getitem__(self, index):
        return self.graph_list[index]

    def __len__(self):
        return len(self.graph_list)


def load_graph_dataset(args, graph_dir, role_dir):
    return NXGraphDataset(args, graph_dir=graph_dir, role_dir=role_dir, sample_size=0)


class PyGGraphDataset(InMemoryDataset):
    def __init__(self, nx_graphs):
        super().__init__()
        data_list = []
        task_embedding_dim = None

        for g in nx_graphs:
            # Temporarily remove pandas objects from g.graph before from_networkx
            # to avoid deprecation warnings
            pandas_attrs = {}
            for key in list(g.graph.keys()):
                if isinstance(g.graph[key], (pd.Series, pd.DataFrame)):
                    pandas_attrs[key] = g.graph.pop(key)

            # Convert node/edge attributes to PyG format:
            # - 'feat' -> x
            # - 'edge_attr' -> edge_attr
            data = from_networkx(
                g,
                group_node_attrs=['feat'],      # node feature -> x (N, 1)
                group_edge_attrs=['edge_attr'], # edge feature -> edge_attr (E, 1)
            )

            data.num_nodes = g.number_of_nodes()

            # Restore pandas attributes after conversion
            for key, value in pandas_attrs.items():
                g.graph[key] = value
                data[key] = value

            # Optional: add graph-level features
            if 'task_embedding' in g.graph:
                # ensure tensor float dtype
                data.task_embedding = torch.as_tensor(
                    g.graph['task_embedding'], dtype=torch.float
                )

                if task_embedding_dim is None:
                    task_embedding_dim = data.task_embedding.shape[0]
                else:
                    assert task_embedding_dim == data.task_embedding.shape[0], "Task embedding dimension mismatch"

            if 'is_correct' in g.graph:
                data.is_correct = torch.as_tensor(
                    g.graph['is_correct'], dtype=torch.float
                )
            
            if 'record' in g.graph:
                data.record = g.graph['record']

            data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        self.task_embedding_dim = task_embedding_dim
