import torch
from torch import nn
from torch_geometric.nn.conv import RGCNConv
from torch.nn import functional as F
import math


class RGCN(nn.Module):
    def __init__(self, num_relations, hidden_dim, out_channels=1, num_layers=3, device='cpu'):
        super(RGCN, self).__init__()
        self.device = device
        self.embedding_dim = hidden_dim
        self.num_layers = num_layers

        self.conv = nn.ModuleList()

        # Define R-GCN layers
        for layer in range(num_layers - 1):
            self.conv.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=num_relations, num_bases=2).to(self.device))
        
        self.conv.append(RGCNConv(in_channels=hidden_dim, out_channels=out_channels, num_relations=num_relations, num_bases=2).to(self.device))
    
    def forward(self, x, edge_index, edge_type):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        
        # R-GCN layers
        for layer in range(self.num_layers):
            x = self.conv[layer](x, edge_index, edge_type)
        return x

class DiffusionOrderingNetwork(nn.Module):
    '''
    at each diffusion step t, we sample from this network to select a node 
    v_sigma(t) to be absorbed and obtain the corresponding masked graph Gt
    The original effective-size redundancy input is intentionally removed.
    Optional task conditioning can be enabled by passing task_feature_dim.
    '''
    def __init__(self,
                 node_feature_dim,
                 num_node_types,
                 num_edge_types,
                 task_feature_dim=None,
                 num_layers=3,
                 out_channels=1,
                 hidden_dim=32,
                 num_heads=6,
                 device='cpu'):
        super(DiffusionOrderingNetwork, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        num_node_types += 1 # add one for masked node type
        num_edge_types += 2 # add one for masked edge type and one for empty edge type
        
        # add positional encodings into node features
        self.embedding = nn.Embedding(num_embeddings=num_node_types, embedding_dim=hidden_dim).to(self.device)
        self.task_proj = nn.Linear(task_feature_dim, hidden_dim).to(self.device) if task_feature_dim is not None else None
        
        # Create an instance of the RGCN model
        self.gat = RGCN(num_relations=num_edge_types,
                        hidden_dim=self.hidden_dim,
                        out_channels=self.out_channels,
                        num_layers=num_layers,
                        device=device).to(self.device)
        
        # initialize positional encodings
        MAX_NODES = 10000
        self.pe = self.positionalencoding(MAX_NODES).to(self.device)

    def positionalencoding(self, lengths):
        '''
        From Chen, et al. 2021 (Order Matters: Probabilistic Modeling of Node Sequences for Graph Generation)
        * lengths: length(s) of graph in the batch
        '''
        l_t = lengths # .max() # use when parallelizing
        pes = torch.zeros([l_t, self.out_channels], device=self.device)
        position = torch.arange(0, l_t, device=self.device).unsqueeze(1) + 1
        div_term = torch.exp((torch.arange(0, self.out_channels, 2, dtype=torch.float, device=self.device) *
                              -(math.log(10000.0) / self.out_channels)))
        pes[:,0::2] = torch.sin(position.float() * div_term)
        pes[:,1::2] = torch.cos(position.float() * div_term)
        return pes

    def forward(self, G, node_order=None, task_embedding=None):
        '''
        node_order: list of absorbed nodes so far
        '''
        if node_order is None:
            node_order = []
        # list of not absorbed nodes (G.x.shape[0], except for nodes in node_order)
        node_order = [int(node.item()) if torch.is_tensor(node) else int(node) for node in node_order]
        unmasked = torch.tensor([node for node in range(G.x.shape[0]) if node not in node_order], device=self.device)

        h = self.embedding(G.x.squeeze().long().to(self.device))
        if self.task_proj is not None and task_embedding is not None:
            task_h = self.task_proj(task_embedding.float().to(self.device)).view(1, -1)
            h = h + task_h.repeat(h.shape[0], 1)

        # # Positional encoding
        for t in range(len(node_order)):
            h[node_order[t], :] += self.pe[t, :].to(self.device)
        h = self.gat(h, G.edge_index.long().to(self.device), G.edge_attr.long().to(self.device))
        logits = h.squeeze(-1)

        if unmasked.numel() > 0:
            masked_logits = torch.full_like(logits, -torch.inf)
            masked_logits[unmasked] = logits[unmasked]
            probs = F.softmax(masked_logits, dim=0)
        else:
            probs = F.softmax(logits, dim=0)
        
        return probs.view(-1, 1)  # outputs probabilities for a categorical distribution over nodes


class TaskAwareTopologyOrderNetwork(nn.Module):
    '''
    Supervised topology ordering model.

    It is independent from the denoising loss: given a candidate graph, task
    embedding, and agent-role node ids, it autoregressively selects the next
    node in the topology order. The loss is plain next-node cross entropy over a
    provided target topological order.
    '''
    def __init__(self,
                 node_feature_dim,
                 num_node_types,
                 num_edge_types,
                 task_feature_dim,
                 num_layers=3,
                 hidden_dim=128,
                 max_nodes=256,
                 device='cpu'):
        super(TaskAwareTopologyOrderNetwork, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        num_node_types += 1
        num_edge_types += 2

        self.node_embedding = nn.Embedding(
            num_embeddings=num_node_types,
            embedding_dim=hidden_dim,
        ).to(self.device)
        self.task_proj = nn.Linear(task_feature_dim, hidden_dim).to(self.device)
        self.step_embedding = nn.Embedding(max_nodes, hidden_dim).to(self.device)
        self.encoder = RGCN(
            num_relations=num_edge_types,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            device=device,
        ).to(self.device)
        self.scorer = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    def _as_order_list(self, node_order):
        if node_order is None:
            return []
        return [int(node.item()) if torch.is_tensor(node) else int(node) for node in node_order]

    def _task_context(self, task_embedding):
        if task_embedding is None:
            return torch.zeros(1, self.hidden_dim, device=self.device)
        return self.task_proj(task_embedding.float().to(self.device)).view(1, -1)

    def encode_nodes(self, G, task_embedding=None):
        h = self.node_embedding(G.x.squeeze().long().to(self.device))
        task_h = self._task_context(task_embedding)
        h = h + task_h.repeat(h.shape[0], 1)
        return self.encoder(
            h,
            G.edge_index.long().to(self.device),
            G.edge_attr.long().to(self.device),
        )

    def forward(self, G, selected_order=None, task_embedding=None, return_logits=False):
        '''
        Returns next-node probabilities over all nodes. Already selected nodes
        are masked out. Set return_logits=True for cross entropy training.
        '''
        selected_order = self._as_order_list(selected_order)
        h = self.encode_nodes(G, task_embedding)
        num_nodes = h.shape[0]

        if selected_order:
            selected_tensor = torch.tensor(selected_order, dtype=torch.long, device=self.device)
            selected_context = h[selected_tensor].mean(dim=0, keepdim=True)
        else:
            selected_context = torch.zeros(1, self.hidden_dim, device=self.device)

        step_idx = min(len(selected_order), self.max_nodes - 1)
        step_context = self.step_embedding(torch.tensor([step_idx], dtype=torch.long, device=self.device))
        context = (selected_context + step_context).repeat(num_nodes, 1)
        task_context = self._task_context(task_embedding).repeat(num_nodes, 1)

        logits = self.scorer(torch.cat([h, context, task_context], dim=-1)).squeeze(-1)
        if selected_order:
            logits[selected_tensor] = -torch.inf

        if return_logits:
            return logits
        return F.softmax(logits, dim=0).view(-1, 1)

    def supervised_loss(self, G, target_order, task_embedding=None):
        '''
        Cross entropy for a complete target topological order.
        target_order can be a Python list or a 1-D tensor of node ids.
        '''
        target_order = self._as_order_list(target_order)
        if len(target_order) == 0:
            return torch.tensor(0.0, device=self.device)

        losses = []
        selected_order = []
        for target_node in target_order:
            logits = self.forward(
                G,
                selected_order=selected_order,
                task_embedding=task_embedding,
                return_logits=True,
            )
            target = torch.tensor([target_node], dtype=torch.long, device=self.device)
            losses.append(F.cross_entropy(logits.view(1, -1), target))
            selected_order.append(target_node)

        return torch.stack(losses).mean()

    def sample_order(self, G, task_embedding=None, sampling_method="argmax"):
        assert sampling_method in ["argmax", "sample"], "sampling_method must be either 'argmax' or 'sample'"
        selected_order = []
        for _ in range(G.x.shape[0]):
            probs = self.forward(G, selected_order=selected_order, task_embedding=task_embedding).view(-1)
            if sampling_method == "sample":
                next_node = torch.distributions.Categorical(probs=probs).sample()
            else:
                next_node = torch.argmax(probs)
            selected_order.append(int(next_node.item()))
        return selected_order
