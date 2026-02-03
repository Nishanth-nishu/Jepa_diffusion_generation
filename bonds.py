"""
bonds.py — CHEMISTRY ONLY

Purpose:
    Convert SMILES → chemical graph

This file is the ONLY place RDKit should exist.

Functions:
    1. get_bonds_from_smiles(smiles: str) -> edge_index
    2. get_bonds_from_batch(smiles_list, mask, device) -> edge_index

What is NOT here:
    ❌ Distances
    ❌ Coordinates
    ❌ Angles/torsions
    ❌ Padding logic
"""

import torch
import warnings

try:
    from rdkit import Chem
    from rdkit import RDLogger
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Using fallback bond extraction.")


# ============================================================================
# SINGLE MOLECULE BOND EXTRACTION
# ============================================================================

def get_bonds_from_smiles(smiles, num_atoms=None, verbose=False):
    """
    Extract chemical bonds from SMILES using RDKit.
    
    Args:
        smiles: str, SMILES string
        num_atoms: int, expected number of atoms (for validation)
        verbose: bool, print warnings
    
    Returns:
        edge_index: Tensor[2, E], directed edges (i→j and j→i for each bond)
    
    Raises:
        ValueError: if RDKit parsing fails critically
    """
    if not RDKIT_AVAILABLE:
        if verbose:
            print(f"⚠️ RDKit not available, using fallback for SMILES: {smiles}")
        return _fallback_bonds(num_atoms) if num_atoms else torch.zeros((2, 0), dtype=torch.long)
    
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            if verbose:
                print(f"⚠️ RDKit failed to parse SMILES: {smiles}")
            return _fallback_bonds(num_atoms) if num_atoms else torch.zeros((2, 0), dtype=torch.long)
        
        # Validate number of atoms
        if num_atoms is not None and mol.GetNumAtoms() != num_atoms:
            if verbose:
                print(f"⚠️ Atom count mismatch: SMILES has {mol.GetNumAtoms()}, expected {num_atoms}")
            return _fallback_bonds(num_atoms)
        
        # Extract bonds (directed: i→j and j→i)
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])  # Make undirected
        
        if len(edges) == 0:
            # No bonds (e.g., single atom)
            return torch.zeros((2, 0), dtype=torch.long)
        
        edge_index = torch.tensor(edges, dtype=torch.long).T  # (2, E)
        return edge_index
    
    except Exception as e:
        if verbose:
            print(f"⚠️ RDKit error for SMILES '{smiles}': {e}")
        return _fallback_bonds(num_atoms) if num_atoms else torch.zeros((2, 0), dtype=torch.long)


def _fallback_bonds(num_atoms):
    """
    Fallback: Create local connectivity graph.
    
    WARNING: This is NOT chemically correct. Use only when RDKit fails.
    
    Args:
        num_atoms: int, number of atoms
    
    Returns:
        edge_index: Tensor[2, E], local connectivity
    """
    if num_atoms is None or num_atoms == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    
    edges = []
    for i in range(num_atoms):
        for j in range(i + 1, min(i + 4, num_atoms)):  # Connect to 3 nearest neighbors
            edges.append([i, j])
            edges.append([j, i])
    
    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    
    return torch.tensor(edges, dtype=torch.long).T


# ============================================================================
# BATCH BOND EXTRACTION
# ============================================================================

def get_bonds_from_batch(smiles_list, mask, device='cpu', verbose=False):
    """
    Extract bonds for an entire batch.
    
    Args:
        smiles_list: List[str], length B
        mask: Tensor[B, max_atoms], 1 = real atom, 0 = padding
        device: torch device
        verbose: bool, print warnings
    
    Returns:
        edge_index: Tensor[2, E_total], global edge indices across batch
        
    How it works:
        1. For each molecule in batch:
           - Get num_atoms from mask
           - Extract bonds from SMILES
           - Offset atom indices by cumulative count
        2. Concatenate all edges
    
    Example:
        Batch of 2 molecules:
            Molecule 0: 3 atoms, bonds [(0,1), (1,2)]
            Molecule 1: 2 atoms, bonds [(0,1)]
        
        Global edge_index:
            [(0,1), (1,0), (1,2), (2,1),  # Molecule 0
             (3,4), (4,3)]                # Molecule 1 (offset by 3)
    """
    batch_size = len(smiles_list)
    
    all_edges = []
    node_offset = 0
    
    for b in range(batch_size):
        # Get number of atoms for this molecule
        n_atoms = int(mask[b].sum().item())
        
        if n_atoms == 0:
            continue
        
        # Get bonds for this molecule
        smiles = smiles_list[b]
        edge_index = get_bonds_from_smiles(smiles, num_atoms=n_atoms, verbose=verbose)
        
        # Offset atom indices to global batch indexing
        if edge_index.shape[1] > 0:
            edge_index_offset = edge_index + node_offset
            all_edges.append(edge_index_offset)
        
        node_offset += n_atoms
    
    # Concatenate all edges
    if len(all_edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    global_edge_index = torch.cat(all_edges, dim=1).to(device)
    
    return global_edge_index


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TESTING bonds.py")
    print("="*70)
    
    # Test 1: Single molecule
    print("\n1. Single molecule bond extraction:")
    smiles = "[H]O[H]"  # Water
    bonds = get_bonds_from_smiles(smiles, num_atoms=3, verbose=True)
    print(f"   SMILES: {smiles}")
    print(f"   Bonds: {bonds.T.tolist()}")
    
    # Test 2: Acetylene
    print("\n2. Acetylene:")
    smiles = "[H]C#C[H]"
    bonds = get_bonds_from_smiles(smiles, num_atoms=4, verbose=True)
    print(f"   SMILES: {smiles}")
    print(f"   Bonds: {bonds.T.tolist()}")
    
    # Test 3: Batch extraction
    print("\n3. Batch extraction:")
    from dataset import QM9Text3DDataset
    from collate import collate_fn
    from torch.utils.data import DataLoader
    
    dataset = QM9Text3DDataset(
        jsonl_path='/scratch/nishanth.r/egnn/data/qm9_100k.jsonl',
        atom_feature_dim=10
    )
    
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    edge_index = get_bonds_from_batch(
        batch['smiles'],
        batch['mask'],
        device='cpu',
        verbose=True
    )
    
    print(f"   Batch size: {len(batch['smiles'])}")
    print(f"   Total atoms: {int(batch['mask'].sum().item())}")
    print(f"   Total edges: {edge_index.shape[1]}")
    print(f"   Edge index shape: {edge_index.shape}")
    
    # Verify edges are within bounds
    max_node = int(batch['mask'].sum().item())
    if edge_index.shape[1] > 0:
        print(f"   Edge range: [{edge_index.min().item()}, {edge_index.max().item()}]")
        print(f"   Valid: {edge_index.max().item() < max_node}")
