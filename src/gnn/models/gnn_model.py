"""
GNN Model for Antimicrobial Activity Prediction
================================================

Implements Graph Attention Network (GAT) for molecular property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import DataLoader


class AntimicrobialGNN(nn.Module):
    """
    Graph Attention Network for antimicrobial activity prediction
    
    Architecture:
    1. Graph convolution layers (GAT) to learn from molecular structure
    2. Global pooling to get graph-level representation
    3. MLP classifier for binary prediction
    """
    
    def __init__(self, 
                 num_node_features: int,
                 num_edge_features: int = 0,
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.3):
        """
        Args:
            num_node_features: Dimension of node features
            num_edge_features: Dimension of edge features (optional)
            hidden_channels: Hidden layer size
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(num_node_features, hidden_channels, heads=heads, dropout=dropout)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Last layer - reduce to single head
        self.convs.append(
            GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Global pooling + MLP classifier
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)  # *2 for mean+max pooling
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = nn.Linear(hidden_channels // 2, 2)  # Binary classification
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Data object with x, edge_index, batch
            
        Returns:
            Logits for binary classification
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (combine mean and max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # MLP classifier
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.lin2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        
        return x
    
    def predict_proba(self, data):
        """Get probability predictions"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            probs = F.softmax(logits, dim=1)
        return probs


class SimplerGNN(nn.Module):
    """
    Simpler GNN model for smaller datasets
    Uses basic graph convolutions without attention
    """
    
    def __init__(self, num_node_features: int, hidden_channels: int = 32):
        super().__init__()
        
        from torch_geometric.nn import GCNConv
        
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 2)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classifier
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        
        return x


def get_model(model_type: str = 'gat', num_node_features: int = 38, 
              num_edge_features: int = 6, **kwargs):
    """
    Factory function to get model
    
    Args:
        model_type: 'gat' (Graph Attention) or 'gcn' (simpler)
        num_node_features: Number of node features
        num_edge_features: Number of edge features
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    if model_type == 'gat':
        return AntimicrobialGNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            **kwargs
        )
    elif model_type == 'gcn':
        return SimplerGNN(
            num_node_features=num_node_features,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    print("="*70)
    print("🧪 Testing GNN Model Architecture")
    print("="*70)
    
    # Create dummy data
    from torch_geometric.data import Data
    
    num_nodes = 20
    num_edges = 40
    num_node_features = 38
    
    x = torch.randn(num_nodes, num_node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test GAT model
    print("\n📊 Testing GAT Model:")
    model = get_model('gat', num_node_features=num_node_features)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    output = model(data)
    print(f"   Input nodes: {num_nodes}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ GAT model works!")
    
    # Test GCN model
    print("\n📊 Testing GCN Model:")
    model_gcn = get_model('gcn', num_node_features=num_node_features)
    print(f"   Parameters: {sum(p.numel() for p in model_gcn.parameters()):,}")
    
    output_gcn = model_gcn(data)
    print(f"   Output shape: {output_gcn.shape}")
    print(f"   ✅ GCN model works!")
    
    print("\n✅ Model architectures validated!")