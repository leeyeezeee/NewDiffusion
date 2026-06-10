import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear, ReLU
from torch_geometric.nn import MessagePassing

    
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
                predict_node_types=False,
                use_semantic_gain=True,
                device='cpu'):
        super().__init__()
        self.device = device
        num_edge_types += 1 # add one for empty edge type
        self.K = K
        self.num_layers = num_layers
        self.predict_node_types = predict_node_types
        self.num_edge_types = num_edge_types
        self.node_embedding = Linear(node_feature_dim, hidden_dim).to(self.device)
        self.edge_embedding = Linear(edge_feature_dim, hidden_dim).to(self.device)
        self.task_proj = Linear(task_feature_dim, hidden_dim).to(self.device)
        self.use_semantic_gain = use_semantic_gain
        self.semantic_gain_weight = nn.Parameter(torch.tensor(1.0, device=self.device)) if use_semantic_gain else None

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(MPLayer(hidden_dim, hidden_dim)).to(self.device)

        self.mlp_alpha = nn.Sequential(Linear(3*hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       Linear(hidden_dim, self.K)).to(self.device)
        
        self.node_pred_layer = None
        if self.predict_node_types:
            self.node_pred_layer = nn.Sequential(Linear(2*hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           Linear(hidden_dim, num_node_types)).to(self.device)
        
        self.edge_pred_layer = nn.Sequential(Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       Linear(hidden_dim, num_edge_types*K)).to(self.device)


        
    def forward(self, x, edge_index, edge_attr, task_embedding, v_t=None, semantic_gain=None, return_node_probs=None):
        # make sure x and edge_attr are of type float, for the MLPs
        x = x.float().to(self.device)
        edge_attr = edge_attr.float().to(self.device)
        task_embedding = task_embedding.float().to(self.device)
        if return_node_probs is None:
            return_node_probs = self.predict_node_types
        
        h_v = self.node_embedding(x)
        h_e = self.edge_embedding(edge_attr.reshape(-1, 1))
        
        for l in range(self.num_layers):
            h_v = self.layers[l](h_v, edge_index, h_e)

        # graph-level embedding
        graph_embedding = self.task_proj(task_embedding) # + torch.mean(h_v, dim=0, keepdim=True)

        # repeat graph embedding to have the same shape as h_v
        graph_embedding = graph_embedding.repeat(h_v.shape[0], 1)

        node_pred = None
        if self.predict_node_types and return_node_probs:
            node_pred = self.node_pred_layer(torch.cat([graph_embedding, h_v], dim=1)) # hidden_dim + 1
        
        # edge prediction follows a mixture of multinomial distribution, with
        # the Softmax(sum(mlp_alpha([graph_embedding, h_vi, h_vj])))
        alphas = torch.zeros(h_v.shape[0], self.K, device=self.device)
        if v_t is None:
            v_t = h_v.shape[0] - 1 # node being masked, this assumes that the masked node is the last node in the graph
        h_v_t = h_v[v_t, :].repeat(h_v.shape[0], 1)

        alphas = self.mlp_alpha(torch.cat([graph_embedding, h_v_t, h_v], dim=1))

        alphas = F.softmax(torch.sum(alphas, dim=0, keepdim=True), dim=1)

        log_theta = self.edge_pred_layer(h_v)
        log_theta = log_theta.view(h_v.shape[0], -1, self.K) # h_v.shape[0] is the number of steps (nodes) (block size)
        if self.use_semantic_gain and semantic_gain is not None:
            semantic_gain = semantic_gain.float().to(self.device).view(-1)
            semantic_gain = self._normalize_semantic_gain(semantic_gain)
            if semantic_gain.shape[0] != h_v.shape[0]:
                raise ValueError("semantic_gain must have one score per node/candidate edge")
            semantic_bias = torch.zeros(
                h_v.shape[0],
                self.num_edge_types,
                1,
                device=self.device,
                dtype=log_theta.dtype,
            )
            # Higher semantic entropy gain means the edge is useful: increase
            # non-empty edge logits and decrease EMPTY_EDGE (last class).
            semantic_bias[:, :-1, :] = semantic_gain.view(-1, 1, 1)
            semantic_bias[:, -1:, :] = -semantic_gain.view(-1, 1, 1)
            log_theta = log_theta + self.semantic_gain_weight * semantic_bias
        p_e = torch.sum(alphas * F.softmax(log_theta, dim=1), dim=-1) # softmax over edge types

        p_e = p_e.to(self.device) 

        if self.predict_node_types and return_node_probs:
            p_v = F.softmax(node_pred, dim=-1).to(self.device)
            return p_v, p_e
        return p_e
    
    def _normalize_semantic_gain(self, semantic_gain, eps=1e-8):
        if semantic_gain.numel() <= 1:
            return torch.clamp(semantic_gain, min=0.0)
        min_value = semantic_gain.min()
        max_value = semantic_gain.max()
        return (semantic_gain - min_value) / (max_value - min_value + eps)


class SemanticEdgeDenoisingNetwork(DenoisingNetwork):
    '''
    Edge-only denoiser for fixed agents and a known/generated topology order.
    It keeps the RADAR message passing and mixture edge head, but removes node
    type prediction from the denoising objective.
    '''
    def __init__(self, *args, **kwargs):
        kwargs["predict_node_types"] = False
        kwargs.setdefault("use_semantic_gain", True)
        super().__init__(*args, **kwargs)
