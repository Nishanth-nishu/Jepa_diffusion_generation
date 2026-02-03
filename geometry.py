"""
geometry.py — GEOMETRY RELATIONS ONLY

Purpose:
    Everything derived from the bond graph, NOT coordinates

Functions:
    1. get_angles(edge_index, num_nodes) -> Tensor[num_angles, 3]
    2. get_torsions(edge_index, num_nodes) -> Tensor[num_torsions, 4]
    3. get_nonbonded_pairs(edge_index, num_nodes) -> Tensor[2, num_pairs]

What is NOT here:
    ❌ RDKit
    ❌ SMILES
    ❌ Dataset logic
    ❌ Batching
"""

import torch


# ============================================================================
# ANGLE EXTRACTION
# ============================================================================

def get_angles(edge_index, num_nodes, max_angles=None):
    """
    Extract valid angles (i-j-k) from bond graph.
    
    Rules:
        - i-j is a bond
        - j-k is a bond
        - i ≠ k
    
    Args:
        edge_index: Tensor[2, E], directed edges
        num_nodes: int, number of nodes
        max_angles: int, optional limit on number of angles
    
    Returns:
        angle_triplets: Tensor[num_angles, 3], each row is (i, j, k)
    
    Example:
        If bonds are: 0-1, 1-2
        Then angles are: 0-1-2
    """
    if edge_index.shape[1] == 0:
        return torch.zeros((0, 3), dtype=torch.long, device=edge_index.device)
    
    # Build adjacency list
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
        if d not in adj[s]:  # Avoid duplicates
            adj[s].append(d)
    
    # Find all angles
    angles = []
    for j in range(num_nodes):
        neighbors = adj[j]
        
        if len(neighbors) < 2:
            continue
        
        # All pairs of neighbors form valid angles
        for idx_i, i in enumerate(neighbors):
            for k in neighbors[idx_i + 1:]:
                angles.append([i, j, k])
                
                if max_angles and len(angles) >= max_angles:
                    break
            
            if max_angles and len(angles) >= max_angles:
                break
        
        if max_angles and len(angles) >= max_angles:
            break
    
    if len(angles) == 0:
        return torch.zeros((0, 3), dtype=torch.long, device=edge_index.device)
    
    return torch.tensor(angles, dtype=torch.long, device=edge_index.device)


# ============================================================================
# TORSION EXTRACTION
# ============================================================================

def get_torsions(edge_index, num_nodes, max_torsions=200):
    """
    Extract torsion angles (i-j-k-l) from bond graph.
    
    Rules:
        - i-j is a bond
        - j-k is a bond
        - k-l is a bond
        - Forms a connected path i-j-k-l
    
    Args:
        edge_index: Tensor[2, E], directed edges
        num_nodes: int, number of nodes
        max_torsions: int, limit on number of torsions (prevents explosion)
    
    Returns:
        torsion_quads: Tensor[num_torsions, 4], each row is (i, j, k, l)
    
    Example:
        If bonds are: 0-1, 1-2, 2-3
        Then torsions are: 0-1-2-3
    """
    if edge_index.shape[1] == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=edge_index.device)
    
    # Build adjacency list
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
        if d not in adj[s]:
            adj[s].append(d)
    
    # Find all torsions along j-k edges
    torsions = []
    
    for j in range(num_nodes):
        j_neighbors = adj[j]
        
        if len(j_neighbors) == 0:
            continue
        
        for k in j_neighbors:
            if k <= j:  # Avoid duplicates (process each edge once)
                continue
            
            k_neighbors = adj[k]
            
            if len(k_neighbors) == 0:
                continue
            
            # Get neighbors of j (excluding k)
            i_candidates = [n for n in j_neighbors if n != k]
            
            # Get neighbors of k (excluding j)
            l_candidates = [n for n in k_neighbors if n != j]
            
            if len(i_candidates) == 0 or len(l_candidates) == 0:
                continue
            
            # Create torsions i-j-k-l
            for i in i_candidates[:2]:  # Limit to avoid explosion
                for l in l_candidates[:2]:
                    torsions.append([i, j, k, l])
                    
                    if len(torsions) >= max_torsions:
                        break
                
                if len(torsions) >= max_torsions:
                    break
            
            if len(torsions) >= max_torsions:
                break
        
        if len(torsions) >= max_torsions:
            break
    
    if len(torsions) == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=edge_index.device)
    
    return torch.tensor(torsions, dtype=torch.long, device=edge_index.device)


# ============================================================================
# NON-BONDED PAIRS
# ============================================================================

def get_nonbonded_pairs(edge_index, num_nodes, max_pairs=500):
    """
    Extract non-bonded atom pairs for repulsion modeling.
    
    Rules:
        - Exclude 1-2 bonded pairs
        - Exclude 1-3 pairs (bonded through one atom)
        - Include all other pairs
    
    Args:
        edge_index: Tensor[2, E], directed edges
        num_nodes: int, number of nodes
        max_pairs: int, limit on number of pairs
    
    Returns:
        nonbond_pairs: Tensor[2, num_pairs], each column is (i, j) with i < j
    
    Why exclude 1-3?
        - 1-2: bonded, handled by bond length head
        - 1-3: angle, handled by angle head
        - 1-4+: repulsion, handled here
    """
    if edge_index.shape[1] == 0 or num_nodes < 4:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    
    # Build 1-2 bonded set
    bonded_12 = set()
    src, dst = edge_index
    for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
        bonded_12.add((min(s, d), max(s, d)))
    
    # Build adjacency for 1-3 detection
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
        if d not in adj[s]:
            adj[s].append(d)
    
    # Build 1-3 pairs (bonded through one atom)
    bonded_13 = set()
    for j in range(num_nodes):
        neighbors = adj[j]
        
        # All pairs of neighbors are 1-3 connected
        for idx_i, i in enumerate(neighbors):
            for k in neighbors[idx_i + 1:]:
                bonded_13.add((min(i, k), max(i, k)))
    
    # Combine exclusions
    excluded = bonded_12 | bonded_13
    
    # Get all non-excluded pairs
    pairs = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) not in excluded:
                pairs.append([i, j])
                
                if len(pairs) >= max_pairs:
                    break
        
        if len(pairs) >= max_pairs:
            break
    
    if len(pairs) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    
    # Return as (2, num_pairs)
    return torch.tensor(pairs, dtype=torch.long, device=edge_index.device).T


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TESTING geometry.py")
    print("="*70)
    
    # Test 1: Simple chain 0-1-2-3
    print("\n1. Linear chain (0-1-2-3):")
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)
    
    angles = get_angles(edge_index, num_nodes=4)
    print(f"   Angles: {angles.tolist()}")
    print(f"   Expected: [[0,1,2], [1,2,3]]")
    
    torsions = get_torsions(edge_index, num_nodes=4)
    print(f"   Torsions: {torsions.tolist()}")
    print(f"   Expected: [[0,1,2,3]]")
    
    nonbond = get_nonbonded_pairs(edge_index, num_nodes=4)
    print(f"   Non-bonded: {nonbond.T.tolist()}")
    print(f"   Expected: [[0,3]] (1-4 pair)")
    
    # Test 2: Branched (water-like)
    print("\n2. Branched structure (H-O-H):")
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long)
    
    angles = get_angles(edge_index, num_nodes=3)
    print(f"   Angles: {angles.tolist()}")
    print(f"   Expected: [[0,1,2]] (H-O-H angle)")
    
    torsions = get_torsions(edge_index, num_nodes=3)
    print(f"   Torsions: {torsions.tolist()}")
    print(f"   Expected: [] (no torsions in 3-atom molecule)")
    
    nonbond = get_nonbonded_pairs(edge_index, num_nodes=3)
    print(f"   Non-bonded: {nonbond.T.tolist()}")
    print(f"   Expected: [] (no 1-4+ pairs)")
    
    # Test 3: Real batch
    print("\n3. Real batch:")
    from dataset import QM9Text3DDataset
    from collate import collate_fn
    from bonds import get_bonds_from_batch
    from torch.utils.data import DataLoader
    
    dataset = QM9Text3DDataset(
        jsonl_path='/scratch/nishanth.r/egnn/data/qm9_100k.jsonl',
        atom_feature_dim=10
    )
    
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    # Get bonds
    edge_index = get_bonds_from_batch(batch['smiles'], batch['mask'])
    num_atoms = int(batch['mask'].sum().item())
    
    # Get geometry
    angles = get_angles(edge_index, num_atoms)
    torsions = get_torsions(edge_index, num_atoms)
    nonbond = get_nonbonded_pairs(edge_index, num_atoms)
    
    print(f"   Total atoms: {num_atoms}")
    print(f"   Total bonds: {edge_index.shape[1]}")
    print(f"   Total angles: {angles.shape[0]}")
    print(f"   Total torsions: {torsions.shape[0]}")
    print(f"   Total non-bonded pairs: {nonbond.shape[1]}")
