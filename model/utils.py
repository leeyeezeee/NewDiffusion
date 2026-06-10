import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

def random_node_decay_ordering(datapoint):
    # create random list of nodes
    return torch.randperm(datapoint.x.shape[0]).tolist()



class NodeMasking:
    def __init__(self, dataset):
        self.dataset = dataset
        assert dataset.x.shape[1] == 1, "Only one feature per node is supported"

        # Maps: type to id
        self.node_types_to_idx = {node_type.item(): idx for idx, node_type in enumerate(dataset.x.unique())}
        self.edge_types_to_idx = {edge_type.item(): idx for idx, edge_type in enumerate(dataset.edge_attr.unique())}

        # Inverse maps: id to type
        self.idx_to_node_types = {idx: node_type for node_type, idx in self.node_types_to_idx.items()}
        self.idx_to_edge_types = {idx: edge_type for edge_type, idx in self.edge_types_to_idx.items()}
        
        # Mask values
        self.NODE_MASK = dataset.x.unique().shape[0]
        self.EMPTY_EDGE = dataset.edge_attr.unique().shape[0]
        self.EDGE_MASK = dataset.edge_attr.unique().shape[0] + 1

    def _edge_side_attrs(self, graph):
        return [name for name in ("edge_semantic_gain", "semantic_gain", "edge_entropy_gain") if hasattr(graph, name)]

    def _filter_edge_side_attrs(self, graph, mask):
        for attr_name in self._edge_side_attrs(graph):
            setattr(graph, attr_name, getattr(graph, attr_name)[mask])
        return graph
    
    def idxify(self, datapoint):
        '''
        Converts node and edge types to indices starting from 0
        '''
        datapoint = datapoint.clone()
        datapoint.x = torch.tensor([self.node_types_to_idx[node_type.item()] for node_type in datapoint.x]).reshape(-1, 1)
        datapoint.edge_attr = torch.tensor([self.edge_types_to_idx[edge_type.item()] for edge_type in datapoint.edge_attr])
        return datapoint
    
    def deidxify(self, datapoint):
        '''
        Converts node and edge indices back to their original types
        '''
        datapoint = datapoint.clone()
        datapoint.x = torch.tensor([self.idx_to_node_types.get(node_idx.item(), self.NODE_MASK) for node_idx in datapoint.x]).reshape(-1, 1)
        datapoint.edge_attr = torch.tensor([self.idx_to_edge_types.get(edge_idx.item(), self.EDGE_MASK) for edge_idx in datapoint.edge_attr])
        return datapoint

    def is_masked(self, datapoint, node=None):
        '''
        returns if node is masked or not, or array of masked nodes if node == None
        '''
        if node is None:
            return datapoint.x == self.NODE_MASK
        return datapoint.x[node] == self.NODE_MASK

    def remove_node(self, datapoint, node):
        '''
        Removes node from graph, and all edges connected to it
        '''
        assert node < datapoint.x.shape[0], "Node does not exist"
        if datapoint.x.shape[0] == 1:
            return datapoint.clone()
        datapoint = datapoint.clone()
        # remove node
        datapoint.x = torch.cat([datapoint.x[:node], datapoint.x[node+1:]])
        
        # remove edges from edge_index (remove elements containing node in tuple of edge_index) (if datapoint.edge_index[:, 0] == node or datapoint.edge_index[:, 1] == node)
        if datapoint.edge_index.shape[1] > 1:

            # remove edges (remove elements containing node)
            keep_mask = torch.tensor([node not in edge_index for edge_index in datapoint.edge_index.T], device=datapoint.edge_index.device, dtype=torch.bool)
            datapoint.edge_attr = datapoint.edge_attr[keep_mask]
            datapoint = self._filter_edge_side_attrs(datapoint, keep_mask)

            edge_index_T = torch.stack([edge_index_tuple for edge_index_tuple in datapoint.edge_index.T if node not in edge_index_tuple])
            datapoint.edge_index = edge_index_T.T
            # update indices of edge_index
            datapoint.edge_index[datapoint.edge_index > node] -= 1
        return datapoint

    def add_masked_node(self, datapoint):
        '''
        Adds a masked node to the graph
        '''
        datapoint = datapoint.clone()
        n_nodes = datapoint.x.shape[0]
        datapoint.x = torch.cat([datapoint.x.reshape(-1,1), torch.tensor([[self.NODE_MASK]]).to(datapoint.x.device)], dim=0)
        datapoint.edge_attr = torch.cat([datapoint.edge_attr.reshape(-1,1), torch.tensor([self.EDGE_MASK]).repeat(n_nodes+1, 1).to(datapoint.edge_attr.device)], dim=0)
        new_edges = torch.tensor([(node, n_nodes) for node in range(n_nodes+1)], dtype=torch.long).transpose(1,0).to(datapoint.edge_index.device)
        for attr_name in self._edge_side_attrs(datapoint):
            attr = getattr(datapoint, attr_name).reshape(-1, 1)
            pad = torch.zeros(n_nodes + 1, 1, dtype=attr.dtype, device=attr.device)
            setattr(datapoint, attr_name, torch.cat([attr, pad], dim=0).view(-1))

        datapoint.edge_index = torch.cat([datapoint.edge_index, new_edges], dim=1)
        return datapoint

    def mask_node(self, datapoint, selected_node):
        '''
        Masking node mechanism
        1. Masked node (x = -1)
        2. Connected to all other nodes in graph by masked edges (edge_attr = -1)
        
        datapoint.x: node feature matrix
        datapoint.edge_index: edge index matrix
        datapoint.edge_attr: edge attribute matrix
        datapoint.y: target value
        '''
        # mask node
        datapoint = datapoint.clone()
        datapoint.x[selected_node] = self.NODE_MASK
        
        # mask edges
        datapoint.edge_attr[datapoint.edge_index[0] == selected_node] = self.EDGE_MASK
        datapoint.edge_attr[datapoint.edge_index[1] == selected_node] = self.EDGE_MASK
        return datapoint

    def mask_edge(self, datapoint, src, dst):
        '''
        Masks one directed edge src -> dst without changing node types.
        '''
        datapoint = datapoint.clone()
        src = int(src.item()) if torch.is_tensor(src) else int(src)
        dst = int(dst.item()) if torch.is_tensor(dst) else int(dst)
        mask = torch.logical_and(datapoint.edge_index[0] == src, datapoint.edge_index[1] == dst)
        datapoint.edge_attr[mask] = self.EDGE_MASK
        return datapoint

    def mask_edges_to_node(self, datapoint, selected_node, predecessors=None):
        '''
        Masks directed incoming edges i -> selected_node without masking the
        selected node itself. If predecessors is None, all incoming edges are
        masked.
        '''
        datapoint = datapoint.clone()
        selected_node = int(selected_node.item()) if torch.is_tensor(selected_node) else int(selected_node)
        mask = datapoint.edge_index[1] == selected_node
        if predecessors is not None:
            predecessors = [
                int(node.item()) if torch.is_tensor(node) else int(node)
                for node in predecessors
            ]
            pred_mask = torch.zeros_like(mask, dtype=torch.bool)
            for predecessor in predecessors:
                pred_mask = torch.logical_or(pred_mask, datapoint.edge_index[0] == predecessor)
            mask = torch.logical_and(mask, pred_mask)
        datapoint.edge_attr[mask] = self.EDGE_MASK
        return datapoint
    
    def _reorder_edge_attr_and_index(self, graph):
        '''
        Reorders edge_attr and edge_index to be like on nx graph
        (0, 0), (0, 1), (0, 2), ..., (0, n), (1, 0), (1, 1), ..., (n, n)
        '''
        graph = graph.clone()
        # reorder edge_attr
        edge_attr = torch.full((graph.x.shape[0], graph.x.shape[0]), self.EMPTY_EDGE, dtype=torch.long)
        for edge_attr_value, edge_index in zip(graph.edge_attr, graph.edge_index.T):
            edge_attr[edge_index[0], edge_index[1]] = edge_attr_value
        graph.edge_attr = edge_attr.view(-1)
        
        # reorder edge_index
        edge_index = torch.stack([torch.tensor([i, j]) for i in range(graph.x.shape[0]) for j in range(graph.x.shape[0])], dim=1)
        graph.edge_index = edge_index.long()
        return graph

    def remove_empty_edges(self, graph):
        '''
        Removes empty edges from graph
        '''
        graph = graph.clone()
        # remove masker.EMPTY_EDGE from edge_attr, and equivalent in edge_index
        mask = graph.edge_attr.view(-1) != self.EMPTY_EDGE
        graph.edge_index = graph.edge_index[:, mask]
        graph.edge_attr = graph.edge_attr[mask]
        graph = self._filter_edge_side_attrs(graph, mask)

        return graph
    
    def remove_masked_edges(self, graph):
        '''
        Removes masked edges from graph
        '''
        graph = graph.clone()
        # remove masker.EDGE_MASK from edge_attr, and equivalent in edge_index
        mask = graph.edge_attr.view(-1) != self.EDGE_MASK
        graph.edge_index = graph.edge_index[:, mask]
        graph.edge_attr = graph.edge_attr[mask]
        graph = self._filter_edge_side_attrs(graph, mask)

        return graph
    
    def remove_self_loops(self, graph):
        '''
        Removes self-loops from graph
        '''
        graph = graph.clone()
        mask = graph.edge_index[0] != graph.edge_index[1]
        # remove self-loops from edge_index
        graph.edge_index = graph.edge_index[:, mask]
        # remove self-loops from edge_attr
        graph.edge_attr = graph.edge_attr[mask]
        graph = self._filter_edge_side_attrs(graph, mask)

        return graph

    def demask_node(self, graph, selected_node, node_type, connections_types):
        '''
        Demasking node mechanism
        1. Unmasked node (graph.x = node_type)
        2. Connected to all other nodes in graph by unmasked edges (graph.edge_attr <= connections_types)
        '''
        assert connections_types.shape[0] == graph.x.shape[0], "Number of connections must be equal to number of nodes"
        
        # demask node
        graph = graph.clone()
        graph.x[selected_node] = node_type
        # demask edge_attr
        for i, connection in enumerate(connections_types):
            if not self.is_masked(graph, node=i):
                graph.edge_attr[torch.logical_and(graph.edge_index[0] == i, graph.edge_index[1] == selected_node)] = connection * 1.0
                graph.edge_attr[torch.logical_and(graph.edge_index[1] == i, graph.edge_index[0] == selected_node)] = connection * 1.0
        
        return graph
    
    def fully_connect(self, graph, keep_original_edges=True):
        '''
        Fully connect graph with edge attribute value
        '''
        adjacency_matrix = to_dense_adj(graph.edge_index)[0]
        adjacency_matrix[adjacency_matrix == 0] = 1

        fully_connected = graph.clone()
        fully_connected.edge_attr = torch.ones(fully_connected.x.shape[0]**2) * self.EMPTY_EDGE
        
        fully_connected.edge_attr = fully_connected.edge_attr.long()
        semantic_attrs = {}
        for attr_name in self._edge_side_attrs(graph):
            attr = getattr(graph, attr_name)
            semantic_attrs[attr_name] = torch.zeros(
                fully_connected.x.shape[0]**2,
                dtype=attr.dtype,
                device=attr.device,
            )

        if keep_original_edges:
            # restore values of original edges
            for edge_pos, (edge_attr, edge_index) in enumerate(zip(graph.edge_attr, graph.edge_index.T)):
                fully_connected.edge_attr[edge_index[0] * fully_connected.x.shape[0] + edge_index[1]] = edge_attr
                fully_connected.edge_attr[edge_index[1] * fully_connected.x.shape[0] + edge_index[0]] = edge_attr  # Ensure symmetry
                for attr_name, dense_attr in semantic_attrs.items():
                    source_attr = getattr(graph, attr_name)
                    if edge_pos < source_attr.numel():
                        dense_attr[edge_index[0] * fully_connected.x.shape[0] + edge_index[1]] = source_attr[edge_pos]
                        dense_attr[edge_index[1] * fully_connected.x.shape[0] + edge_index[0]] = source_attr[edge_pos]

        fully_connected.edge_index = torch.nonzero(adjacency_matrix).T
        flat_indices = fully_connected.edge_index[0] * fully_connected.x.shape[0] + fully_connected.edge_index[1]
        for attr_name, dense_attr in semantic_attrs.items():
            setattr(fully_connected, attr_name, dense_attr[flat_indices])
        return fully_connected
    
    def generate_fully_masked(self, n_nodes):
        '''
        Generates a fully masked graph like the one provided
        '''
        
        fully_masked = Data(
            x=torch.ones((n_nodes, 1))*self.NODE_MASK,
            edge_index=torch.tensor([(i, j) for i in range(n_nodes) for j in range(n_nodes)], dtype=torch.int64).transpose(1,0),
            edge_attr=torch.ones(n_nodes**2)*self.EDGE_MASK,
        )
        return fully_masked

    def get_denoised_nodes(self, graph):
        '''
        Returns a list of nodes that are denoised
        '''
        denoised_nodes = []
        for node in range(graph.x.shape[0]):
            if not self.is_masked(graph, node):
                denoised_nodes.append(node)

        return denoised_nodes
