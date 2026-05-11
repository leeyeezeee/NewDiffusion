import os
import torch
import torch.nn as nn
import logging

from model.denoising import DenoisingNetwork
from model.ordering import DiffusionOrderingNetwork
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
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(GDFramework, self).__init__()
        self.device = device
        self.diffusion_ordering_network = diffusion_ordering_network.to(device)
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
            sigma_t_dist = self.diffusion_ordering_network(p, node_order)
            # sample (only unmasked nodes) from categorical distribution to get node to mask
            unmasked = torch.tensor([i not in node_order for i in range(p.x.shape[0])]).to(self.device)

            sigma_t_dist_list.append(sigma_t_dist.flatten())
            sigma_t = torch.distributions.Categorical(probs=sigma_t_dist[unmasked].flatten()).sample()

            # get node index
            sigma_t = torch.where(unmasked.flatten())[0][sigma_t.long()]
            node_order.append(sigma_t)
        return node_order, sigma_t_dist_list

    def uniform_node_decay_ordering(self, datapoint):
        '''
        Samples next node from uniform distribution 
        '''
        p = datapoint.clone()
        return torch.randperm(p.x.shape[0]).tolist()

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
            node_order_invariate = node_order

            # create diffusion trajectory
            diffusion_trajectory = [original_data]
            masked_data = graph.clone()
            for i in range(len(node_order)):
                node = node_order[i]
                masked_data = masked_data.clone().to(self.device)
                masked_data = self.masker.mask_node(masked_data, node)
                diffusion_trajectory.append(masked_data)
                if i < len(node_order) - 1:
                    masked_data = self.masker.remove_node(masked_data, node)
                    node_order = [n - 1 if n > node else n for n in node_order]  # update node order to account for removed node

            diffusion_trajectories.append([diffusion_trajectory, node_order_invariate, sigma_t_dist])
        return diffusion_trajectories

    def preprocess(self, graph):
        '''
        Preprocesses graph to be used by the denoising network.
        '''
        graph = graph.clone()
        graph = self.masker.idxify(graph)
        graph = self.masker.fully_connect(graph)
        return graph

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
        edge_probs = torch.gather(edge_probs, 1, correct_edge_type.view(-1, 1))
        nll_edge = -torch.log(edge_probs + 1e-8).mean()

        return nll_edge

    def compute_denoising_loss(self, diffusion_trajectory, node_order_invariate, sigma_t_dist_list):
        '''
        Computes the loss for the denoising network based on negative log-likelihood (NLL).
        '''
        loss = 0
        T = len(diffusion_trajectory) - 1 # Total number of time steps
        sigma_t = torch.stack(sigma_t_dist_list, dim=0)
        G_0 = diffusion_trajectory[0]  # Original graph

        for t in range(0, T):
            graph_t_next = diffusion_trajectory[t + 1] # G_{t+1}
            node_type_probs, edge_type_probs = self.denoising_network(graph_t_next.x, graph_t_next.edge_index, graph_t_next.edge_attr, graph_t_next.task_embedding)

            # Compute NLL for node type
            # compute for all nodes, weight them by the sigma_t_dist at the original node order
            sigma_t_dist = sigma_t[t]
            sigma_t_dist = sigma_t_dist[sigma_t_dist != 0]

            original_node_type = G_0.x[node_order_invariate[t]]
            nll_node = self.compute_nll_node(node_type_probs, original_node_type, sigma_t_dist)
            # get original edge type for each edge in G_0
            
            original_edge_types = G_0.edge_attr[(G_0.edge_index[0] == node_order_invariate[t]) & 
                                              (torch.tensor([G_0.edge_index[1][i] in node_order_invariate[t:] 
                                                             for i in range(G_0.edge_index.shape[1])]).to(self.device))]
            nll_edge = self.compute_nll_edge(edge_type_probs, original_edge_types)

            loss += nll_node + nll_edge
        
        print(f"Denoising loss: {loss.item()}")

        return loss / T

    def compute_ordering_loss(self, diffusion_trajectories, M):
        '''
        Computes the loss for the diffusion ordering network using the REINFORCE algorithm.
        '''
        ordering_loss = 0
        for trajectory, node_order, sigma_t_dist_list in diffusion_trajectories:
            # Compute the reward as the negative denoising loss
            with torch.no_grad():
                reward = -self.compute_denoising_loss(trajectory, node_order, sigma_t_dist_list)
                print(f"Reward: {reward.item()}")
            # REINFORCE update (policy gradient)
            # Calculate probability of trajectory using sigma_t_dist_list
            log_prob = torch.tensor(0.0, device=self.device)
            for t in range(len(sigma_t_dist_list)):
                log_prob += torch.log(sigma_t_dist_list[t][node_order[t]])
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
            denoising_loss = sum([self.compute_denoising_loss(traj[0], traj[1], traj[2]) for traj in diffusion_trajectories])
            batch_denoising_loss += denoising_loss
        
        for graph in ordering_batch:
            graph = self.preprocess(graph)
            diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)

            # Compute ordering loss for REINFORCE
            ordering_loss = self.compute_ordering_loss(diffusion_trajectories, M)
            batch_ordering_loss += ordering_loss
        
        return batch_denoising_loss, batch_ordering_loss
    
    def compute_graph_utility_loss(self, log_probs_list, generated_reward, original_reward, connections_list=None):
        '''
        Computes the utility loss for a generated graph based on the log probabilities of the node and edge types.
        '''
        total_log_prob = torch.tensor(0.0, device=self.device, requires_grad=True)
    
        for step_idx, log_probs in enumerate(log_probs_list):
            node_log_prob = log_probs['node_log_prob']
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
        
        if not return_log_probs:
            with torch.no_grad():
                if preprocess:
                    graph = self.preprocess(graph)
                # predict node type
                node_type_probs, edge_type_probs = self.denoising_network(graph.x, graph.edge_index, graph.edge_attr, task_embedding)
        else:
            if preprocess:
                graph = self.preprocess(graph)
            
            node_type_probs, edge_type_probs = self.denoising_network(graph.x, graph.edge_index, graph.edge_attr, task_embedding)
            
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

    def generate_graph(self, num_nodes: int, task_embedding: torch.Tensor, return_log_probs=False):
        '''
        Generates a graph with num_nodes nodes using the GDFramework.
        '''
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

