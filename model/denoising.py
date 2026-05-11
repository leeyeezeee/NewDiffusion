import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj

    
class MPLayer(MessagePassing):
    '''
    Custom message passing layer for the GraphARM model
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum') #  "Max" aggregation.
        self.f = nn.Sequential(Linear(3 * in_channels, out_channels),
                       nn.ReLU(),
                       Linear(out_channels, out_channels)) # MLP for message construction
        self.g = nn.Sequential(Linear(3 * in_channels, out_channels),
                          nn.ReLU(),
                          Linear(out_channels, out_channels)) # MLP for attention coefficients
        
        self.gru = nn.GRU(2*out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        '''
        x has shape [N, in_channels]
        edge_index has shape [2, E]
        **self-loops should be added in the preprocessing step (fully connecting the graph)
        '''

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out, _ = self.gru(torch.cat([x, out], dim=-1)) # discard final hidden state
        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        h_vi = x_i
        h_vj = x_j
        h_eij = edge_attr

        m_ij = self.f(torch.cat([h_vi, h_vj, h_eij], dim=-1))
        a_ij = self.g(torch.cat([h_vi, h_vj, h_eij], dim=-1))
        return m_ij * a_ij


class DenoisingNetwork(nn.Module):
    def __init__(self,
                node_feature_dim,
                edge_feature_dim,
                task_feature_dim,
                num_node_types,
                num_edge_types,
                num_layers=5,
                hidden_dim=256,
                K=20,
                device='cpu'):
        super().__init__()
        self.device = device
        num_edge_types += 1 # add one for empty edge type
        self.K = K
        self.num_layers = num_layers
        self.node_embedding = Linear(node_feature_dim, hidden_dim).to(self.device)
        self.edge_embedding = Linear(edge_feature_dim, hidden_dim).to(self.device)
        self.task_proj = Linear(task_feature_dim, hidden_dim).to(self.device)
        self.effective_size_weight = nn.Parameter(torch.tensor(1.0, device=self.device))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(MPLayer(hidden_dim, hidden_dim)).to(self.device)

        self.mlp_alpha = nn.Sequential(Linear(3*hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       Linear(hidden_dim, self.K)).to(self.device)
        
        self.node_pred_layer = nn.Sequential(Linear(2*hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       Linear(hidden_dim, num_node_types)).to(self.device)
        
        self.edge_pred_layer = nn.Sequential(Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       Linear(hidden_dim, num_edge_types*K)).to(self.device)


        
    def forward(self, x, edge_index, edge_attr, task_embedding, v_t=None):
        # make sure x and edge_attr are of type float, for the MLPs
        x = x.float().to(self.device)
        edge_attr = edge_attr.float().to(self.device)
        task_embedding = task_embedding.float().to(self.device)
        
        h_v = self.node_embedding(x)
        h_e = self.edge_embedding(edge_attr.reshape(-1, 1))
        
        for l in range(self.num_layers):
            h_v = self.layers[l](h_v, edge_index, h_e)

        effective_sizes = self.compute_effective_size(x, edge_index)
        h_v = h_v + self.effective_size_weight * effective_sizes.unsqueeze(1)

        # graph-level embedding
        graph_embedding = self.task_proj(task_embedding) # + torch.mean(h_v, dim=0, keepdim=True)

        # repeat graph embedding to have the same shape as h_v
        graph_embedding = graph_embedding.repeat(h_v.shape[0], 1)

        node_pred = self.node_pred_layer(torch.cat([graph_embedding, h_v], dim=1)) # hidden_dim + 1
        
        # edge prediction follows a mixture of multinomial distribution, with
        # the Softmax(sum(mlp_alpha([graph_embedding, h_vi, h_vj])))
        alphas = torch.zeros(h_v.shape[0], self.K)
        if v_t is None:
            v_t = h_v.shape[0] - 1 # node being masked, this assumes that the masked node is the last node in the graph
        h_v_t = h_v[v_t, :].repeat(h_v.shape[0], 1)

        alphas = self.mlp_alpha(torch.cat([graph_embedding, h_v_t, h_v], dim=1))

        alphas = F.softmax(torch.sum(alphas, dim=0, keepdim=True), dim=1)

        p_v = F.softmax(node_pred, dim=-1)
        log_theta = self.edge_pred_layer(h_v)
        log_theta = log_theta.view(h_v.shape[0], -1, self.K) # h_v.shape[0] is the number of steps (nodes) (block size)
        p_e = torch.sum(alphas * F.softmax(log_theta, dim=1), dim=-1) # softmax over edge types

        p_v = p_v.to(self.device) 
        p_e = p_e.to(self.device) 

        return p_v, p_e
    

    def compute_effective_size(self, x, edge_index, alpha=0.7):
        """
        Compute effective size for each node in a directed graph.
        For directed graphs, effective size has two components:
        - In-neighbor effective size = |n_in| - |tie_in| / |n_in|
        - Out-neighbor effective size = |n_out| - |tie_out| / |n_out|
    
        Returns combined effective size (sum of in and out components).
        """
        num_nodes = x.shape[0]
        effective_sizes = torch.zeros(num_nodes, device=self.device)
    
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

        return effective_sizes
