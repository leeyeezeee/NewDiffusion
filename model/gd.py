import os
import torch
import torch.nn as nn
import logging

from model.denoising import DenoisingNetwork, SemanticEdgeDenoisingNetwork
from model.ordering import DiffusionOrderingNetwork, TaskAwareTopologyOrderNetwork
from model.utils import NodeMasking

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GDFramework(nn.Module):
    '''
    Class to encapsulate DiffusionOrderingNetwork and DenoisingNetwork, as well as the training loop
    for both with diffusion and denoising steps.
    '''
    def __init__(self,
                 dataset,
                 denoising_network,
                 diffusion_ordering_network,
                 topology_order_network=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(GDFramework, self).__init__()
        self.device = device
        self.diffusion_ordering_network = diffusion_ordering_network.to(device)
        self.topology_order_network = (
            topology_order_network.to(device)
            if topology_order_network is not None
            else diffusion_ordering_network.to(device)
        )
        self.denoising_network = denoising_network.to(device)
        self.masker = NodeMasking(dataset)
        

    def node_decay_ordering(self, datapoint):
        '''
        Returns node order for a given graph, using the diffusion ordering network.
        '''
        p = datapoint.clone().to(self.device)
        node_order = []
        sigma_t_dist_list = []

        for i in range(p.x.shape[0]):
            # use diffusion ordering network to get probabilities
            sigma_t_dist = self.diffusion_ordering_network(
                p,
                node_order,
                task_embedding=getattr(p, "task_embedding", None),
            )
            # sample (only unmasked nodes) from categorical distribution to get node to mask
            unmasked = torch.tensor([i not in node_order for i in range(p.x.shape[0])]).to(self.device)

            sigma_t_dist_list.append(sigma_t_dist.flatten())
            sigma_t = torch.distributions.Categorical(probs=sigma_t_dist[unmasked].flatten()).sample()

            # get node index
            sigma_t = torch.where(unmasked.flatten())[0][sigma_t.long()]
            sigma_t = int(sigma_t.item())
            node_order.append(sigma_t)
        return node_order, sigma_t_dist_list

    def uniform_node_decay_ordering(self, datapoint):
        '''
        Samples next node from uniform distribution 
        '''
        p = datapoint.clone()
        return torch.randperm(p.x.shape[0]).tolist()

    def topology_ordering(self, graph, task_embedding=None, sampling_method="argmax"):
        '''
        Returns a topology order from the supervised order model. This is the
        preferred path for the semantic edge-only variant.
        '''
        task_embedding = task_embedding if task_embedding is not None else getattr(graph, "task_embedding", None)
        if hasattr(self.topology_order_network, "sample_order"):
            return self.topology_order_network.sample_order(
                graph.to(self.device),
                task_embedding=task_embedding,
                sampling_method=sampling_method,
            )
        node_order, _ = self.node_decay_ordering(graph)
        return node_order

    def _get_target_order(self, graph, target_order=None):
        if target_order is not None:
            return target_order
        for attr_name in ("target_order", "topological_order", "node_order", "order"):
            if hasattr(graph, attr_name):
                return getattr(graph, attr_name)
        raise ValueError("target_order is required unless graph has target_order/topological_order/node_order/order")

    def compute_topology_order_loss(self, ordering_batch, target_orders=None):
        '''
        Supervised loss for the independent order model. This should be
        optimized separately from the edge denoising loss.
        '''
        if not hasattr(self.topology_order_network, "supervised_loss"):
            raise TypeError("topology_order_network must implement supervised_loss")

        if not isinstance(ordering_batch, (list, tuple)):
            ordering_batch = [ordering_batch]
        if target_orders is None:
            target_orders = [None] * len(ordering_batch)

        losses = []
        for graph, target_order in zip(ordering_batch, target_orders):
            target_order = self._get_target_order(graph, target_order)
            graph = self.preprocess_order_graph(graph).to(self.device)
            task_embedding = getattr(graph, "task_embedding", None)
            losses.append(self.topology_order_network.supervised_loss(
                graph,
                target_order=target_order,
                task_embedding=task_embedding,
            ))
        return torch.stack(losses).mean()

    def generate_diffusion_trajectories(self, graph, M):
        '''
        Generates M diffusion trajectories for a given graph,
        using the node decay ordering mechanism.
        '''
        print(f"Generating {M} diffusion trajectories for graph with {graph.x.shape[0]} nodes")
        original_data = graph.clone().to(self.device)
        diffusion_trajectories = []
        for m in range(M):
            node_order, sigma_t_dist = self.node_decay_ordering(graph)
            original_node_order = list(node_order)
            current_node_order = []

            # create diffusion trajectory
            diffusion_trajectory = [original_data]
            reference_trajectory = []
            masked_data = graph.clone()
            for i in range(len(node_order)):
                node = node_order[i]
                masked_data = masked_data.clone().to(self.device)
                reference_trajectory.append(masked_data.clone())
                current_node_order.append(node)
                masked_data = self.masker.mask_node(masked_data, node)
                diffusion_trajectory.append(masked_data)
                if i < len(node_order) - 1:
                    masked_data = self.masker.remove_node(masked_data, node)
                    node_order = [n - 1 if n > node else n for n in node_order]  # update node order to account for removed node

            diffusion_trajectories.append([
                diffusion_trajectory,
                current_node_order,
                sigma_t_dist,
                reference_trajectory,
                original_node_order,
            ])
        return diffusion_trajectories

    def preprocess(self, graph):
        '''
        Preprocesses graph to be used by the denoising network.
        '''
        graph = graph.clone()
        graph = self.masker.idxify(graph)
        graph = self.masker.fully_connect(graph)
        return graph

    def preprocess_order_graph(self, graph):
        '''
        Preprocesses graph for topology-order supervision while preserving the
        original directed candidate structure.
        '''
        graph = graph.clone()
        return self.masker.idxify(graph)

    def compute_nll_node(self, node_type_probs, correct_node_type, sigma_t_dist):
        '''
        Computes the negative log-likelihood for node types.
        '''
        # Compute NLL for edge type
        node_probs = node_type_probs * sigma_t_dist.view(-1, 1).clone()
        # get original edge index for each node being unmasked
        nll_node = -torch.log(node_probs[:, correct_node_type].sum() + 1e-8)
        return nll_node.mean()

    def compute_nll_edge(self, edge_type_probs, correct_edge_type):
        '''
        Computes the negative log-likelihood for edge types.
        - get probability of choosing edge type for each edge
        - compose edge_type_probs with sigma_t_dist to get probability of choosing edge type for each edge
        '''
        edge_probs = edge_type_probs.view(-1, edge_type_probs.shape[-1])
        correct_edge_type = correct_edge_type.to(self.device).long().view(-1)
        edge_probs = torch.gather(edge_probs, 1, correct_edge_type.view(-1, 1))
        nll_edge = -torch.log(edge_probs + 1e-8).mean()

        return nll_edge

    def _semantic_gain_attr(self, graph):
        for attr_name in ("edge_semantic_gain", "semantic_gain", "edge_entropy_gain"):
            if hasattr(graph, attr_name):
                return getattr(graph, attr_name)
        return None

    def semantic_gain_to_node(self, graph, selected_node):
        '''
        Returns one semantic entropy gain score per candidate i -> selected_node.
        If the graph has no edge-level semantic gains, zeros are returned and the
        denoiser behaves like the normal RADAR edge head without effective size.
        '''
        n_nodes = graph.x.shape[0]
        gains = torch.zeros(n_nodes, device=self.device)
        edge_gain = self._semantic_gain_attr(graph)
        if edge_gain is None:
            return gains
        edge_gain = edge_gain.to(self.device).view(-1)
        selected_node = int(selected_node.item()) if torch.is_tensor(selected_node) else int(selected_node)
        for edge_pos, edge_index in enumerate(graph.edge_index.T):
            src = int(edge_index[0].item())
            dst = int(edge_index[1].item())
            if dst == selected_node and edge_pos < edge_gain.numel():
                gains[src] = edge_gain[edge_pos]
        return gains

    def edge_types_to_node(self, graph, selected_node):
        '''
        Returns the correct edge type for each candidate i -> selected_node.
        Missing candidates are treated as EMPTY_EDGE.
        '''
        n_nodes = graph.x.shape[0]
        edge_types = torch.full(
            (n_nodes,),
            int(self.masker.EMPTY_EDGE),
            dtype=torch.long,
            device=self.device,
        )
        selected_node = int(selected_node.item()) if torch.is_tensor(selected_node) else int(selected_node)
        for edge_attr, edge_index in zip(graph.edge_attr, graph.edge_index.T):
            src = int(edge_index[0].item())
            dst = int(edge_index[1].item())
            if dst == selected_node:
                edge_types[src] = edge_attr.long().to(self.device)
        return edge_types

    def edge_type_for_pair(self, graph, src, dst):
        '''
        Returns the edge type for src -> dst, or EMPTY_EDGE if the edge is absent.
        '''
        src = int(src.item()) if torch.is_tensor(src) else int(src)
        dst = int(dst.item()) if torch.is_tensor(dst) else int(dst)
        for edge_attr, edge_index in zip(graph.edge_attr, graph.edge_index.T):
            if int(edge_index[0].item()) == src and int(edge_index[1].item()) == dst:
                return edge_attr.long().to(self.device)
        return torch.tensor(int(self.masker.EMPTY_EDGE), dtype=torch.long, device=self.device)

    def semantic_gain_for_pair(self, graph, src, dst):
        edge_gain = self._semantic_gain_attr(graph)
        if edge_gain is None:
            return 0.0
        src = int(src.item()) if torch.is_tensor(src) else int(src)
        dst = int(dst.item()) if torch.is_tensor(dst) else int(dst)
        edge_gain = edge_gain.to(self.device).view(-1)
        for edge_pos, edge_index in enumerate(graph.edge_index.T):
            if int(edge_index[0].item()) == src and int(edge_index[1].item()) == dst and edge_pos < edge_gain.numel():
                return float(edge_gain[edge_pos].item())
        return 0.0

    def edge_mask_order(self, node_order, semantic_gain_matrix=None, graph=None, mask_low_gain_first=True):
        '''
        Builds a legal DAG edge mask schedule under a known topology order.

        The forward diffusion order masks only pi_i -> pi_j where i < j.
        By default low semantic-gain edges are masked first, so the reverse
        denoising trajectory recovers high semantic-gain edges first.
        '''
        node_order = [int(node.item()) if torch.is_tensor(node) else int(node) for node in node_order]
        semantic_gain_matrix_tensor = None
        if semantic_gain_matrix is not None:
            semantic_gain_matrix_tensor = torch.as_tensor(semantic_gain_matrix, dtype=torch.float, device=self.device)

        schedule = []
        for dst_pos in range(1, len(node_order)):
            dst = node_order[dst_pos]
            node_edges = []
            for src_pos in range(dst_pos):
                src = node_order[src_pos]
                if semantic_gain_matrix_tensor is not None:
                    gain = float(semantic_gain_matrix_tensor[src, dst].item())
                elif graph is not None:
                    gain = self.semantic_gain_for_pair(graph, src, dst)
                else:
                    gain = 0.0
                node_edges.append({"src": src, "dst": dst, "semantic_gain": gain})
            node_edges.sort(key=lambda item: item["semantic_gain"], reverse=not mask_low_gain_first)
            schedule.extend(node_edges)
        return schedule

    def compute_edge_denoising_loss(self, graph, node_order=None, semantic_gain_matrix=None, mask_low_gain_first=True):
        '''
        Edge-only diffusion loss. Nodes/agent roles remain visible; only legal
        directed edges under node_order are progressively masked. The denoiser
        predicts the edge type for the current masked edge.
        '''
        graph = self.preprocess(graph).to(self.device)
        if node_order is None:
            node_order = self.topology_ordering(graph, sampling_method="argmax")

        schedule = self.edge_mask_order(
            node_order=node_order,
            semantic_gain_matrix=semantic_gain_matrix,
            graph=graph,
            mask_low_gain_first=mask_low_gain_first,
        )
        if not schedule:
            return torch.tensor(0.0, device=self.device)

        original_graph = graph.clone()
        masked_graph = graph.clone()
        task_embedding = getattr(original_graph, "task_embedding", None)
        if task_embedding is None:
            raise ValueError("graph.task_embedding is required for edge denoising")
        losses = []
        for item in schedule:
            src = item["src"]
            dst = item["dst"]
            masked_graph = self.masker.mask_edge(masked_graph, src, dst)

            if semantic_gain_matrix is not None:
                semantic_gain = torch.as_tensor(semantic_gain_matrix, dtype=torch.float, device=self.device)[:, dst]
            else:
                semantic_gain = self.semantic_gain_to_node(original_graph, dst)

            edge_type_probs = self.denoising_network(
                masked_graph.x,
                masked_graph.edge_index,
                masked_graph.edge_attr,
                task_embedding,
                v_t=dst,
                semantic_gain=semantic_gain,
                return_node_probs=False,
            )
            if isinstance(edge_type_probs, tuple):
                edge_type_probs = edge_type_probs[1]

            correct_edge_type = self.edge_type_for_pair(original_graph, src, dst)
            losses.append(-torch.log(edge_type_probs[src, correct_edge_type] + 1e-8))

        return torch.stack(losses).mean()

    def compute_denoising_loss(
            self,
            diffusion_trajectory,
            node_order_invariate,
            sigma_t_dist_list,
            reference_trajectory=None):
        '''
        Computes the loss for the denoising network based on negative log-likelihood (NLL).
        '''
        loss = 0
        T = len(diffusion_trajectory) - 1 # Total number of time steps
        sigma_t = torch.stack(sigma_t_dist_list, dim=0)
        G_0 = diffusion_trajectory[0]  # Original graph

        for t in range(0, T):
            graph_t_next = diffusion_trajectory[t + 1] # G_{t+1}
            reference_graph = reference_trajectory[t] if reference_trajectory is not None else G_0
            selected_node = node_order_invariate[t]
            semantic_gain = self.semantic_gain_to_node(graph_t_next, selected_node)
            denoising_output = self.denoising_network(
                graph_t_next.x,
                graph_t_next.edge_index,
                graph_t_next.edge_attr,
                graph_t_next.task_embedding,
                v_t=selected_node,
                semantic_gain=semantic_gain,
            )

            if isinstance(denoising_output, tuple):
                node_type_probs, edge_type_probs = denoising_output
                # Compute NLL for node type. Edge-only denoisers skip this branch.
                sigma_t_dist = sigma_t[t]
                sigma_t_dist = sigma_t_dist[sigma_t_dist != 0]
                original_node_type = reference_graph.x[selected_node]
                nll_node = self.compute_nll_node(node_type_probs, original_node_type, sigma_t_dist)
            else:
                edge_type_probs = denoising_output
                nll_node = torch.tensor(0.0, device=self.device)

            original_edge_types = self.edge_types_to_node(reference_graph, selected_node)
            nll_edge = self.compute_nll_edge(edge_type_probs, original_edge_types)

            loss += nll_node + nll_edge
        
        print(f"Denoising loss: {loss.item()}")

        return loss / T

    def compute_ordering_loss(self, diffusion_trajectories, M):
        '''
        Computes the loss for the diffusion ordering network using the REINFORCE algorithm.
        '''
        ordering_loss = 0
        for trajectory_item in diffusion_trajectories:
            trajectory, node_order, sigma_t_dist_list = trajectory_item[:3]
            reference_trajectory = trajectory_item[3] if len(trajectory_item) > 3 else None
            log_prob_node_order = trajectory_item[4] if len(trajectory_item) > 4 else node_order
            # Compute the reward as the negative denoising loss
            with torch.no_grad():
                reward = -self.compute_denoising_loss(
                    trajectory,
                    node_order,
                    sigma_t_dist_list,
                    reference_trajectory=reference_trajectory,
                )
                print(f"Reward: {reward.item()}")
            # REINFORCE update (policy gradient)
            # Calculate probability of trajectory using sigma_t_dist_list
            log_prob = torch.tensor(0.0, device=self.device)
            for t in range(len(sigma_t_dist_list)):
                log_prob += torch.log(sigma_t_dist_list[t][log_prob_node_order[t]])
            print(f"Log prob of sigma_t: {log_prob.item()}")
            ordering_loss += reward * log_prob

        return ordering_loss / M

    def train_step(self, denoising_batch, ordering_batch, M):
        '''
        Performs one training step for both the denoising and diffusion ordering networks.
        '''
        # Generate diffusion trajectories for each graph in the batch
        batch_denoising_loss = 0
        batch_ordering_loss = 0

        for graph in denoising_batch:
            graph = self.preprocess(graph)
            diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)
            
            # Compute denoising loss
            denoising_loss = sum([
                self.compute_denoising_loss(
                    traj[0],
                    traj[1],
                    traj[2],
                    reference_trajectory=traj[3] if len(traj) > 3 else None,
                )
                for traj in diffusion_trajectories
            ])
            batch_denoising_loss += denoising_loss
        
        for graph in ordering_batch:
            graph = self.preprocess(graph)
            diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)

            # Compute ordering loss for REINFORCE
            ordering_loss = self.compute_ordering_loss(diffusion_trajectories, M)
            batch_ordering_loss += ordering_loss
        
        return batch_denoising_loss, batch_ordering_loss

    def train_order_step(self, ordering_batch, target_orders=None):
        '''
        Computes only the supervised topology-order loss. The caller should
        backpropagate this loss with an optimizer over topology_order_network.
        '''
        return self.compute_topology_order_loss(ordering_batch, target_orders=target_orders)

    def train_edge_step(self, denoising_batch, node_orders=None, semantic_gain_matrices=None):
        '''
        Computes only the edge denoising loss. The caller should backpropagate
        this loss with an optimizer over denoising_network.
        '''
        if not isinstance(denoising_batch, (list, tuple)):
            denoising_batch = [denoising_batch]
        if node_orders is None:
            node_orders = [None] * len(denoising_batch)
        if semantic_gain_matrices is None:
            semantic_gain_matrices = [None] * len(denoising_batch)

        losses = []
        for graph, node_order, semantic_gain_matrix in zip(denoising_batch, node_orders, semantic_gain_matrices):
            losses.append(self.compute_edge_denoising_loss(
                graph,
                node_order=node_order,
                semantic_gain_matrix=semantic_gain_matrix,
            ))
        return torch.stack(losses).mean()
    
    def compute_graph_utility_loss(self, log_probs_list, generated_reward, original_reward, connections_list=None):
        '''
        Computes the utility loss for a generated graph based on the log probabilities of the node and edge types.
        '''
        total_log_prob = torch.tensor(0.0, device=self.device, requires_grad=True)
    
        for step_idx, log_probs in enumerate(log_probs_list):
            node_log_prob = log_probs.get('node_log_prob', torch.tensor(0.0, device=self.device))
            edge_log_prob = log_probs['edge_log_prob']  # shape: [num_existing_nodes]
        
            # Filter out masked edges if connections_list and masker are provided
            if connections_list is not None and step_idx < len(connections_list):
                connections = connections_list[step_idx]
                # Set log prob to 0 for masked edges
                valid_mask = (connections != self.masker.EDGE_MASK)
                edge_log_prob = edge_log_prob * valid_mask.float()
        
            # Sum log probs: node log prob + sum of all valid edge log probs
            step_log_prob = node_log_prob + edge_log_prob.sum()
            total_log_prob = total_log_prob + step_log_prob
    
        # Compute advantage
        advantage = generated_reward - original_reward

        min_advantage = -1.0  # Worst case: generated incorrect, original perfect
        max_advantage = 1.0   # Best case: generated perfect, original minimum sparsity
        
        # Normalize advantage to [0, 1]
        advantage = (advantage - min_advantage) / (max_advantage - min_advantage)

        # Normalize by number of steps to stabilize gradients
        num_steps = len(log_probs_list)
        if num_steps > 0:
            advantage = advantage / max(num_steps, 1.0)
    
        # REINFORCE with baseline
        utility_loss = -total_log_prob * advantage
    
        return utility_loss

    def predict_new_node(self, graph, task_embedding: torch.Tensor, sampling_method="sample", preprocess=True, return_log_probs=False):
        '''
        Predicts the value of a new node for graph as well as its connection to all previously denoised nodes.
        sampling_method: "argmax" or "sample"
        - argmax: select node and edge type with highest probability
        - sample: sample node and edge type from multinomial distribution
        '''
        assert sampling_method in ["argmax", "sample"], "sampling_method must be either 'argmax' or 'sample'"
        if not getattr(self.denoising_network, "predict_node_types", True):
            raise ValueError("The configured denoising network is edge-only and cannot predict new node types")
        
        if not return_log_probs:
            with torch.no_grad():
                if preprocess:
                    graph = self.preprocess(graph)
                # predict node type
                semantic_gain = self.semantic_gain_to_node(graph, graph.x.shape[0] - 1)
                node_type_probs, edge_type_probs = self.denoising_network(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    task_embedding,
                    semantic_gain=semantic_gain,
                )
        else:
            if preprocess:
                graph = self.preprocess(graph)
            
            semantic_gain = self.semantic_gain_to_node(graph, graph.x.shape[0] - 1)
            node_type_probs, edge_type_probs = self.denoising_network(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                task_embedding,
                semantic_gain=semantic_gain,
            )
            
        node_type_probs = node_type_probs[-1] # only predict for last node
        # edge_type_probs shape: [num_existing_nodes, num_edge_types]

        # sample node type
        if sampling_method == "sample":
            node_dist = torch.distributions.Categorical(probs=node_type_probs.squeeze())
            node_type = node_dist.sample()
            node_log_prob = node_dist.log_prob(node_type) if return_log_probs else None
        elif sampling_method == "argmax":
            node_type = torch.argmax(node_type_probs.squeeze(), dim=-1).reshape(-1, 1)
            node_log_prob = torch.log(node_type_probs.squeeze()[node_type] + 1e-8) if return_log_probs else None

        # sample edge type
        if sampling_method == "sample":
            # new_connections = torch.multinomial(edge_type_probs.squeeze(), num_samples=1, replacement=True)
            edge_dist = torch.distributions.Categorical(probs=edge_type_probs)
            new_connections = edge_dist.sample()
            edge_log_prob = edge_dist.log_prob(new_connections) if return_log_probs else None
        elif sampling_method == "argmax":
            new_connections = torch.argmax(edge_type_probs, dim=-1) # shape: [num_existing_nodes]
            # no need to filter connection to previously denoised nodes, assuming only one new node is added at a time
            selected_probs = edge_type_probs[torch.arange(edge_type_probs.shape[0]), new_connections]
            edge_log_prob = torch.log(selected_probs + 1e-8) if return_log_probs else None
        
        if return_log_probs:
            return node_type, new_connections, {'node_log_prob': node_log_prob, 'edge_log_prob': edge_log_prob}
        else:
            return node_type, new_connections

    def predict_new_edges(self, graph, task_embedding: torch.Tensor, selected_node=None, sampling_method="sample", return_log_probs=False, semantic_gain=None):
        '''
        Edge-only denoising step for fixed agents. It predicts candidate
        connections i -> selected_node and does not predict a node type.
        '''
        assert sampling_method in ["argmax", "sample"], "sampling_method must be either 'argmax' or 'sample'"
        if selected_node is None:
            selected_node = graph.x.shape[0] - 1
        if semantic_gain is None:
            semantic_gain = self.semantic_gain_to_node(graph, selected_node)

        if not return_log_probs:
            with torch.no_grad():
                edge_type_probs = self.denoising_network(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    task_embedding,
                    v_t=selected_node,
                    semantic_gain=semantic_gain,
                    return_node_probs=False,
                )
        else:
            edge_type_probs = self.denoising_network(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                task_embedding,
                v_t=selected_node,
                semantic_gain=semantic_gain,
                return_node_probs=False,
            )

        if isinstance(edge_type_probs, tuple):
            edge_type_probs = edge_type_probs[1]

        if sampling_method == "sample":
            edge_dist = torch.distributions.Categorical(probs=edge_type_probs)
            new_connections = edge_dist.sample()
            edge_log_prob = edge_dist.log_prob(new_connections) if return_log_probs else None
        else:
            new_connections = torch.argmax(edge_type_probs, dim=-1)
            selected_probs = edge_type_probs[torch.arange(edge_type_probs.shape[0]), new_connections]
            edge_log_prob = torch.log(selected_probs + 1e-8) if return_log_probs else None

        if return_log_probs:
            return new_connections, {'edge_log_prob': edge_log_prob}
        return new_connections

    def generate_edge_graph(self, node_types, task_embedding: torch.Tensor, node_order=None, semantic_gain_matrix=None, return_log_probs=False):
        '''
        Generates only communication edges for fixed nodes/agent roles.
        DAG is guaranteed by only allowing edges from earlier to later nodes in
        node_order. semantic_gain_matrix[i, j] optionally controls edge i -> j.
        '''
        from torch_geometric.data import Data

        node_types = torch.as_tensor(node_types, dtype=torch.long, device=self.device).view(-1, 1)
        num_nodes = node_types.shape[0]
        if node_order is None:
            order_graph = self.masker.generate_fully_masked(n_nodes=num_nodes).to(self.device)
            order_graph.x = node_types.clone()
            order_graph.task_embedding = task_embedding
            node_order = self.topology_ordering(
                order_graph,
                task_embedding=task_embedding,
                sampling_method="argmax",
            )
        node_order = [int(node.item()) if torch.is_tensor(node) else int(node) for node in node_order]

        candidate_edges = []
        for dst_pos in range(num_nodes):
            dst = node_order[dst_pos]
            for src_pos in range(dst_pos):
                src = node_order[src_pos]
                candidate_edges.append([src, dst])

        if candidate_edges:
            edge_index = torch.tensor(candidate_edges, dtype=torch.long, device=self.device).T
            edge_attr = torch.full(
                (len(candidate_edges),),
                int(self.masker.EDGE_MASK),
                dtype=torch.long,
                device=self.device,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0,), dtype=torch.long, device=self.device)

        graph = Data(x=node_types.clone(), edge_index=edge_index, edge_attr=edge_attr).to(self.device)
        kept_edges = []
        kept_attrs = []
        log_probs_list = [] if return_log_probs else None
        connections_list = [] if return_log_probs else None

        semantic_gain_matrix_tensor = None
        if semantic_gain_matrix is not None:
            semantic_gain_matrix_tensor = torch.as_tensor(semantic_gain_matrix, dtype=torch.float, device=self.device)

        for dst_pos in range(1, num_nodes):
            dst = node_order[dst_pos]
            semantic_gain = torch.zeros(num_nodes, device=self.device)
            if semantic_gain_matrix_tensor is not None:
                semantic_gain = semantic_gain_matrix_tensor[:, dst]

            if return_log_probs:
                connections, log_probs = self.predict_new_edges(
                    graph,
                    task_embedding,
                    selected_node=dst,
                    sampling_method="sample",
                    return_log_probs=True,
                    semantic_gain=semantic_gain,
                )
                log_probs_list.append(log_probs)
                connections_list.append(connections)
            else:
                connections = self.predict_new_edges(
                    graph,
                    task_embedding,
                    selected_node=dst,
                    sampling_method="sample",
                    return_log_probs=False,
                    semantic_gain=semantic_gain,
                )

            for src_pos in range(dst_pos):
                src = node_order[src_pos]
                edge_type = connections[src]
                if int(edge_type.item()) != int(self.masker.EMPTY_EDGE):
                    kept_edges.append([src, dst])
                    kept_attrs.append(edge_type)

        if kept_edges:
            graph.edge_index = torch.tensor(kept_edges, dtype=torch.long, device=self.device).T
            graph.edge_attr = torch.stack(kept_attrs).long().to(self.device)
        else:
            graph.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            graph.edge_attr = torch.empty((0,), dtype=torch.long, device=self.device)
        graph = self.masker.deidxify(graph)

        if return_log_probs:
            return graph, log_probs_list, connections_list
        return graph

    def generate_graph(self, num_nodes: int, task_embedding: torch.Tensor, return_log_probs=False, node_types=None, node_order=None, semantic_gain_matrix=None):
        '''
        Generates a graph with num_nodes nodes using the GDFramework.
        '''
        if not getattr(self.denoising_network, "predict_node_types", True):
            if node_types is None:
                node_types = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
            return self.generate_edge_graph(
                node_types=node_types,
                task_embedding=task_embedding,
                node_order=node_order,
                semantic_gain_matrix=semantic_gain_matrix,
                return_log_probs=return_log_probs,
            )

        # generate a new graph from an empty graph
        empty_graph = self.masker.generate_fully_masked(n_nodes=1)
        empty_graph = empty_graph.to(self.device)
        log_probs_list = [] if return_log_probs else None
        connections_list = [] if return_log_probs else None

        # generate a new graph using predict_new_node until the graph is complete
        while empty_graph.x.shape[0] < num_nodes + 1:
            # Step 1: Predict node type and connections for the masked node (last node)
            if return_log_probs:
                node_type, connections, log_probs = self.predict_new_node(empty_graph, task_embedding, sampling_method='sample', preprocess=False, return_log_probs=return_log_probs)
                log_probs_list.append(log_probs)
                connections_list.append(connections)
            else:
                node_type, connections = self.predict_new_node(empty_graph, task_embedding, sampling_method='sample', preprocess=False, return_log_probs=return_log_probs)
    
            # Step 2: Demask the last node with predicted values
            empty_graph = self.masker.demask_node(empty_graph, empty_graph.x.shape[0]-1, node_type, connections)
    
            # Step 3: Remove empty edges
            empty_graph = self.masker.remove_empty_edges(empty_graph)
    
            # Step 4: Remove masked edges
            empty_graph = self.masker.remove_masked_edges(empty_graph)

            # Step 5: Add a new masked node for the next iteration
            # if the graph is complete, break the loop
            if empty_graph.x.shape[0] == num_nodes:
                break
            empty_graph = self.masker.add_masked_node(empty_graph)

        # Final step: Remove self-loops
        empty_graph = self.masker.remove_self_loops(empty_graph)

        # Deidxify the graph
        empty_graph = self.masker.deidxify(empty_graph)
        
        if return_log_probs:
            return empty_graph, log_probs_list, connections_list
        else:
            return empty_graph

    
    def save_model(self, model_dir):
        denoising_network_path = os.path.join(model_dir, "denoising_network.pt")
        diffusion_ordering_network_path = os.path.join(model_dir, "diffusion_ordering_network.pt")
        torch.save(self.denoising_network.state_dict(), denoising_network_path)
        torch.save(self.diffusion_ordering_network.state_dict(), diffusion_ordering_network_path)
        print(f"Model saved to {model_dir}")

    def load_model(self, model_dir):
        denoising_network_path = os.path.join(model_dir, "denoising_network.pt")
        diffusion_ordering_network_path = os.path.join(model_dir, "diffusion_ordering_network.pt")
        self.denoising_network.load_state_dict(torch.load(denoising_network_path, map_location=self.device))
        self.diffusion_ordering_network.load_state_dict(torch.load(diffusion_ordering_network_path, map_location=self.device))
        print(f"Model loaded from {model_dir}")

