import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthGNNModel(nn.Module):
    def __init__(self, node_features_size, depth_features_size, edge_features_size):
        super(DepthGNNModel, self).__init__()
        
        self.node_mlp = nn.Sequential(
            nn.Linear(node_features_size + depth_features_size, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(4096, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(1024, 625)  # Output a flattened 25x25 depth map
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear((node_features_size + depth_features_size) * 2 + edge_features_size, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(1024, edge_features_size)  # Output edge features
        )

    def forward(self, node1_features, node2_features, edges):
        edge_input = torch.cat([node1_features, edges, node2_features], dim=-1)
        updated_edges = self.edge_mlp(edge_input)

        depth_map1 = self.node_mlp(node1_features)
        depth_map2 = self.node_mlp(node2_features)

        return depth_map1, depth_map2, updated_edges