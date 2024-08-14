import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import MessagePassing
# from torch_geometric.data import DataLoader, Batch
import torch.nn.functional as F

# Define the model
class DepthGNNModel(MessagePassing):
    def __init__(self, node_features_size, edge_features_size, hidden_channels, output_size):
        super(DepthGNNModel, self).__init__(aggr='add')  # Aggregation: sum, mean, or max

        # MLP to generate messages
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_features_size + edge_features_size, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, hidden_channels)
        )

        # MLP to update node features
        self.node_mlp = nn.Sequential(
            nn.Linear(node_features_size + hidden_channels, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, output_size)  # Output a flattened 25x25 depth map
        )

    def forward(self, x, edge_index, edge_attr):
        # Perform message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate source node feature, target node feature, and edge feature
        message_input = torch.cat([x_i, edge_attr, x_j], dim=-1)
        return self.message_mlp(message_input)

    def update(self, aggr_out, x):
        # Concatenate original node feature with aggregated message
        updated_node_features = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(updated_node_features)
