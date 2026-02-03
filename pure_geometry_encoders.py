"""
pure_geometry_encoders.py — ENCODERS FOR PURE GEOMETRY JEPA

CRITICAL: These encoders NEVER see coordinates
- Input: atom features + bond graph
- Output: latent representations
- NO geometric information

Three options:
1. GraphTransformer (BEST - attention over graph)
2. GIN (Graph Isomorphism Network - expressive)
3. GINE (GIN with edge features - if you have bond types)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# GRAPH TRANSFORMER (BEST FOR MOLECULAR STRUCTURE)
# ============================================================================

class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer with multi-head attention over graph structure.
    
    WHY BEST: 
    - Captures long-range dependencies
    - Learns structural patterns
    - No geometric bias
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge bias (optional - captures bond structure)
        # Bond types: 0=unknown, 1=single, 2=double, 3=triple, 12=aromatic
        self.edge_bias = nn.Embedding(13, num_heads)  # Support all bond types
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, hidden_dim] node features
            edge_index: [2, E] graph connectivity
            edge_attr: [E] optional edge types
        """
        N = x.shape[0]
        
        # Multi-head attention
        residual = x
        x = self.norm1(x)
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)  # [N, H, D]
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)
        
        # Compute attention only over edges
        src, dst = edge_index
        
        # Attention scores: Q[dst] * K[src]
        attn_scores = (Q[dst] * K[src]).sum(dim=-1) / math.sqrt(self.head_dim)  # [E, H]
        
        # Optional: Add edge bias
        if edge_attr is not None:
            # Clamp edge_attr to valid range [0, 12]
            edge_attr_clamped = torch.clamp(edge_attr, 0, 12)
            edge_bias = self.edge_bias(edge_attr_clamped)  # [E, H]
            attn_scores = attn_scores + edge_bias
        
        # Softmax per destination node
        attn_scores_exp = torch.exp(attn_scores - attn_scores.max())  # Numerical stability
        
        # Sum of attention scores per destination node
        attn_sum = torch.zeros(N, self.num_heads, device=x.device)
        attn_sum.index_add_(0, dst, attn_scores_exp)
        
        # Normalized attention weights
        attn_weights = attn_scores_exp / (attn_sum[dst] + 1e-8)  # [E, H]
        
        # Aggregate messages: sum of weighted values
        messages = V[src] * attn_weights.unsqueeze(-1)  # [E, H, D]
        
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        for h in range(self.num_heads):
            out[:, h, :].index_add_(0, dst, messages[:, h, :])
        
        out = out.view(N, self.hidden_dim)
        out = self.o_proj(out)
        out = self.dropout(out)
        x = residual + out
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class GraphTransformerEncoder(nn.Module):
    """
    BEST CHOICE: Graph Transformer encoder for pure geometry JEPA
    
    Properties:
    - NO coordinate information
    - Learns molecular structure from graph
    - Multi-head attention captures dependencies
    - Suitable for chemistry (handles rings, conjugation, etc.)
    """
    def __init__(self, in_dim, hidden_dim, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, in_dim] atom features
            edge_index: [2, E] graph connectivity
            edge_attr: [E] optional edge types (1=single, 2=double, etc.)
        
        Returns:
            h: [N, hidden_dim] node embeddings
        """
        h = self.input_proj(x)
        
        # If edge_attr is provided, make sure it's valid
        if edge_attr is not None:
            edge_attr = torch.clamp(edge_attr.long(), 0, 12)
        
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        
        h = self.output_norm(h)
        return h


# ============================================================================
# GIN (GRAPH ISOMORPHISM NETWORK) - SIMPLE BUT EXPRESSIVE
# ============================================================================

class GINLayer(nn.Module):
    """
    GIN layer with epsilon-aggregation.
    Theoretically as powerful as WL-test for graph isomorphism.
    """
    def __init__(self, in_dim, out_dim, epsilon_learnable=True):
        super().__init__()
        
        if epsilon_learnable:
            self.epsilon = nn.Parameter(torch.zeros(1))
        else:
            self.epsilon = 0.0
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index):
        """
        GIN update: h_i = MLP((1 + epsilon) * h_i + sum_j h_j)
        """
        src, dst = edge_index
        
        # Aggregate neighbors
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        
        # Update
        out = (1 + self.epsilon) * x + agg
        out = self.mlp(out)
        
        return out


class GINEncoder(nn.Module):
    """
    GOOD CHOICE: GIN encoder for pure geometry JEPA
    
    Properties:
    - NO coordinate information
    - Provably expressive (as powerful as WL-test)
    - Simple and fast
    - Good baseline
    """
    def __init__(self, in_dim, hidden_dim, num_layers=4):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, in_dim] atom features
            edge_index: [2, E] graph connectivity
            edge_attr: ignored (GIN doesn't use edge features)
        
        Returns:
            h: [N, hidden_dim] node embeddings
        """
        h = self.input_proj(x)
        
        for layer in self.layers:
            h = layer(h, edge_index)
        
        h = self.output_norm(h)
        return h


# ============================================================================
# GINE (GIN WITH EDGE FEATURES) - IF YOU HAVE BOND TYPES
# ============================================================================

class GINELayer(nn.Module):
    """
    GINE layer - GIN with edge features.
    Good for chemistry where bond types matter.
    """
    def __init__(self, in_dim, out_dim, edge_dim=32, epsilon_learnable=True):
        super().__init__()
        
        if epsilon_learnable:
            self.epsilon = nn.Parameter(torch.zeros(1))
        else:
            self.epsilon = 0.0
        
        # Edge embedding
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, out_dim),
            nn.ReLU()
        )
        
        # Node update MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        GINE update: h_i = MLP((1 + epsilon) * h_i + sum_j (h_j + edge_ij))
        """
        src, dst = edge_index
        
        # Edge features
        edge_feat = self.edge_embed(edge_attr)  # [E, out_dim]
        
        # Aggregate: neighbors + edge features
        messages = x[src] + edge_feat
        agg = torch.zeros(x.shape[0], edge_feat.shape[1], device=x.device)
        agg.index_add_(0, dst, messages)
        
        # Update
        out = (1 + self.epsilon) * x + agg
        out = self.mlp(out)
        
        return out


class GINEEncoder(nn.Module):
    """
    BEST FOR CHEMISTRY: GINE encoder with bond type features
    
    Properties:
    - NO coordinate information
    - Uses bond types (single, double, triple, aromatic)
    - Expressive for molecules
    - Good for QM9
    """
    def __init__(self, in_dim, hidden_dim, edge_dim=32, num_layers=4):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Bond type embedding (1=single, 2=double, 3=triple, 12=aromatic)
        self.bond_type_embed = nn.Embedding(13, edge_dim)
        
        self.layers = nn.ModuleList([
            GINELayer(hidden_dim, hidden_dim, edge_dim)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, in_dim] atom features
            edge_index: [2, E] graph connectivity
            edge_attr: [E] bond types (1, 2, 3, 12)
        
        Returns:
            h: [N, hidden_dim] node embeddings
        """
        h = self.input_proj(x)
        
        # Embed edge attributes if provided
        if edge_attr is not None:
            edge_feat = self.bond_type_embed(edge_attr)
        else:
            # Default to single bonds
            edge_feat = self.bond_type_embed(
                torch.ones(edge_index.shape[1], dtype=torch.long, device=x.device)
            )
        
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat)
        
        h = self.output_norm(h)
        return h


# ============================================================================
# HELPER: CREATE ENCODER BY NAME
# ============================================================================

def create_encoder(encoder_type, in_dim, hidden_dim, num_layers=4, **kwargs):
    """
    Factory function to create encoders.
    
    Args:
        encoder_type: 'transformer', 'gin', or 'gine'
        in_dim: input feature dimension
        hidden_dim: hidden dimension
        num_layers: number of layers
        **kwargs: additional arguments (num_heads, dropout, etc.)
    
    Returns:
        Encoder module
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == 'transformer':
        return GraphTransformerEncoder(
            in_dim, hidden_dim, num_layers,
            num_heads=kwargs.get('num_heads', 8),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    elif encoder_type == 'gin':
        return GINEncoder(in_dim, hidden_dim, num_layers)
    
    elif encoder_type == 'gine':
        return GINEEncoder(
            in_dim, hidden_dim,
            edge_dim=kwargs.get('edge_dim', 32),
            num_layers=num_layers
        )
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Choose from: 'transformer', 'gin', 'gine'")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TESTING PURE GEOMETRY ENCODERS")
    print("="*70)
    
    # Create dummy data
    N = 20  # 20 atoms
    E = 40  # 40 bonds
    in_dim = 10
    hidden_dim = 128
    
    x = torch.randn(N, in_dim)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randint(1, 4, (E,))  # Bond types 1, 2, 3
    
    print(f"\nInput: {N} atoms, {E} bonds")
    print(f"Atom features: {x.shape}")
    print(f"Edge index: {edge_index.shape}")
    print(f"Edge attr: {edge_attr.shape}")
    
    # Test 1: GraphTransformer
    print("\n" + "-"*70)
    print("1. GraphTransformer")
    print("-"*70)
    model = GraphTransformerEncoder(in_dim, hidden_dim, num_layers=4, num_heads=8)
    h = model(x, edge_index, edge_attr)
    print(f"✅ Output shape: {h.shape}")
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 2: GIN
    print("\n" + "-"*70)
    print("2. GIN")
    print("-"*70)
    model = GINEncoder(in_dim, hidden_dim, num_layers=4)
    h = model(x, edge_index)
    print(f"✅ Output shape: {h.shape}")
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 3: GINE
    print("\n" + "-"*70)
    print("3. GINE")
    print("-"*70)
    model = GINEEncoder(in_dim, hidden_dim, edge_dim=32, num_layers=4)
    h = model(x, edge_index, edge_attr)
    print(f"✅ Output shape: {h.shape}")
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 4: Factory
    print("\n" + "-"*70)
    print("4. Factory function")
    print("-"*70)
    for enc_type in ['transformer', 'gin', 'gine']:
        model = create_encoder(enc_type, in_dim, hidden_dim, num_layers=4)
        h = model(x, edge_index, edge_attr)
        print(f"✅ {enc_type}: {h.shape}")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)
    print("\n📋 RECOMMENDATION:")
    print("   1st choice: 'transformer' (best for molecules)")
    print("   2nd choice: 'gine' (if bond types available)")
    print("   3rd choice: 'gin' (simple baseline)")
    print("="*70)
