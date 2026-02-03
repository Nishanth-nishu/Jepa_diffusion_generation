"""
bond_types.py — BOND TYPE EXTRACTION

Purpose:
    Extract bond types and expected lengths from SMILES using RDKit
    
Returns:
    - Bond type (single=1, double=2, triple=3, aromatic=12)
    - Expected bond length based on chemistry
    - Atom types for valence checking
"""

import torch
import warnings

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Bond type features disabled.")


# ============================================================================
# CHEMISTRY CONSTANTS
# ============================================================================

# Expected bond lengths in Angstroms (from experimental data)
BOND_LENGTHS = {
    # Carbon bonds
    'C-C': 1.54, 'C=C': 1.34, 'C#C': 1.20,
    'C-N': 1.47, 'C=N': 1.29, 'C#N': 1.16,
    'C-O': 1.43, 'C=O': 1.23,
    'C-S': 1.82, 'C=S': 1.60,
    'C-F': 1.35, 'C-Cl': 1.77, 'C-Br': 1.94,
    'C-H': 1.09,
    
    # Nitrogen bonds
    'N-N': 1.45, 'N=N': 1.25, 'N#N': 1.10,
    'N-O': 1.40, 'N=O': 1.21,
    'N-H': 1.01,
    
    # Oxygen bonds
    'O-O': 1.48, 'O=O': 1.21,
    'O-H': 0.96,
    
    # Other
    'S-S': 2.05, 'S=O': 1.43,
    'P-O': 1.63, 'P=O': 1.50,
    
    # Aromatic (average)
    'aromatic': 1.40,
    
    # Fallback
    'default': 1.50
}

# Maximum valence by atom type
MAX_VALENCE = {
    1: 1,   # H
    6: 4,   # C
    7: 5,   # N (can be 3 or 5)
    8: 2,   # O
    9: 1,   # F
    15: 5,  # P
    16: 6,  # S
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}


# ============================================================================
# SINGLE MOLECULE BOND TYPE EXTRACTION
# ============================================================================

def get_bond_types_from_smiles(smiles, num_atoms=None, verbose=False):
    """
    Extract bond types, expected lengths, and atom info from SMILES.
    
    Args:
        smiles: str, SMILES string
        num_atoms: int, expected number of atoms (for validation)
        verbose: bool, print warnings
    
    Returns:
        dict with keys:
            - 'bond_info': List[Dict] with keys:
                * 'edge': (i, j) tuple
                * 'type': int (1=single, 2=double, 3=triple, 12=aromatic)
                * 'expected_length': float (Angstroms)
            - 'atom_info': List[Dict] with keys:
                * 'atomic_num': int
                * 'max_valence': int
                * 'is_hydrogen': bool
    """
    if not RDKIT_AVAILABLE:
        return {'bond_info': [], 'atom_info': []}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if verbose:
                print(f"⚠️ RDKit failed to parse SMILES: {smiles}")
            return {'bond_info': [], 'atom_info': []}
        
        # Validate atom count
        if num_atoms is not None and mol.GetNumAtoms() != num_atoms:
            if verbose:
                print(f"⚠️ Atom count mismatch: SMILES has {mol.GetNumAtoms()}, expected {num_atoms}")
            return {'bond_info': [], 'atom_info': []}
        
        # Extract atom information
        atom_info = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            atom_info.append({
                'atomic_num': atomic_num,
                'max_valence': MAX_VALENCE.get(atomic_num, 4),
                'is_hydrogen': atomic_num == 1
            })
        
        # Extract bond information
        bond_info = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            atom_i = mol.GetAtomWithIdx(i)
            atom_j = mol.GetAtomWithIdx(j)
            
            symbol_i = atom_i.GetSymbol()
            symbol_j = atom_j.GetSymbol()
            
            # Get bond type
            bond_type_double = bond.GetBondTypeAsDouble()  # 1.0, 1.5, 2.0, 3.0
            is_aromatic = bond.GetIsAromatic()
            
            # Map to integer code
            if is_aromatic:
                type_int = 12  # Special code for aromatic
                lookup_key = 'aromatic'
            else:
                type_int = int(round(bond_type_double))
                
                # Build lookup key for expected length
                symbols_sorted = tuple(sorted([symbol_i, symbol_j]))
                bond_symbol = {1: '-', 2: '=', 3: '#'}.get(type_int, '-')
                
                # Try specific lookup
                lookup_key = f"{symbols_sorted[0]}{bond_symbol}{symbols_sorted[1]}"
                
                # Fallback to element pair
                if lookup_key not in BOND_LENGTHS:
                    lookup_key = f"{symbols_sorted[0]}-{symbols_sorted[1]}"
            
            # Get expected length with fallback
            expected_length = BOND_LENGTHS.get(lookup_key, BOND_LENGTHS['default'])
            
            bond_info.append({
                'edge': (i, j),
                'type': type_int,
                'expected_length': expected_length
            })
        
        return {
            'bond_info': bond_info,
            'atom_info': atom_info
        }
    
    except Exception as e:
        if verbose:
            print(f"⚠️ Error extracting bond types from '{smiles}': {e}")
        return {'bond_info': [], 'atom_info': []}


# ============================================================================
# BATCH BOND TYPE EXTRACTION
# ============================================================================

def get_bond_types_from_batch(smiles_list, mask, edge_index, device='cpu', verbose=False):
    """
    Extract bond types and atom info for entire batch.
    
    Args:
        smiles_list: List[str], length B
        mask: Tensor[B, max_atoms], 1 = real atom, 0 = padding
        edge_index: Tensor[2, E], global edge indices
        device: torch device
        verbose: bool, print warnings
    
    Returns:
        dict with keys:
            - 'bond_types': Tensor[E], type for each edge (1,2,3,12)
            - 'expected_lengths': Tensor[E], expected length for each edge
            - 'atom_atomic_nums': Tensor[N], atomic number for each atom
            - 'atom_max_valences': Tensor[N], max valence for each atom
            - 'atom_is_hydrogen': Tensor[N], 1 if hydrogen, 0 otherwise
    """
    bond_types = []
    expected_lengths = []
    
    all_atomic_nums = []
    all_max_valences = []
    all_is_hydrogen = []
    
    node_offset = 0
    
    for b, smiles in enumerate(smiles_list):
        n_atoms = int(mask[b].sum().item())
        if n_atoms == 0:
            continue
        
        # Get chemistry info for this molecule
        chem_info = get_bond_types_from_smiles(smiles, n_atoms, verbose)
        
        # Process atom info
        if len(chem_info['atom_info']) == n_atoms:
            for atom_data in chem_info['atom_info']:
                all_atomic_nums.append(atom_data['atomic_num'])
                all_max_valences.append(atom_data['max_valence'])
                all_is_hydrogen.append(1 if atom_data['is_hydrogen'] else 0)
        else:
            # Fallback if atom info not available
            for _ in range(n_atoms):
                all_atomic_nums.append(6)  # Assume carbon
                all_max_valences.append(4)
                all_is_hydrogen.append(0)
        
        # Build bond lookup for this molecule
        bond_dict = {}
        for bond_data in chem_info['bond_info']:
            i, j = bond_data['edge']
            bond_dict[(i, j)] = (bond_data['type'], bond_data['expected_length'])
            bond_dict[(j, i)] = (bond_data['type'], bond_data['expected_length'])  # Undirected
        
        # Match edges with bond info
        for k in range(edge_index.shape[1]):
            src, dst = edge_index[0, k].item(), edge_index[1, k].item()
            
            # Check if this edge belongs to current molecule
            if node_offset <= src < node_offset + n_atoms:
                local_src = src - node_offset
                local_dst = dst - node_offset
                
                if (local_src, local_dst) in bond_dict:
                    btype, exp_len = bond_dict[(local_src, local_dst)]
                    bond_types.append(btype)
                    expected_lengths.append(exp_len)
                else:
                    # Edge exists but no bond info (shouldn't happen with RDKit)
                    bond_types.append(1)  # Assume single bond
                    expected_lengths.append(1.5)  # Default length
        
        node_offset += n_atoms
    
    # Convert to tensors
    if len(bond_types) == 0:
        bond_types_tensor = torch.zeros(0, dtype=torch.long, device=device)
        expected_lengths_tensor = torch.zeros(0, dtype=torch.float, device=device)
    else:
        bond_types_tensor = torch.tensor(bond_types, dtype=torch.long, device=device)
        expected_lengths_tensor = torch.tensor(expected_lengths, dtype=torch.float, device=device)
    
    if len(all_atomic_nums) == 0:
        atomic_nums_tensor = torch.zeros(0, dtype=torch.long, device=device)
        max_valences_tensor = torch.zeros(0, dtype=torch.long, device=device)
        is_hydrogen_tensor = torch.zeros(0, dtype=torch.float, device=device)
    else:
        atomic_nums_tensor = torch.tensor(all_atomic_nums, dtype=torch.long, device=device)
        max_valences_tensor = torch.tensor(all_max_valences, dtype=torch.long, device=device)
        is_hydrogen_tensor = torch.tensor(all_is_hydrogen, dtype=torch.float, device=device)
    
    return {
        'bond_types': bond_types_tensor,
        'expected_lengths': expected_lengths_tensor,
        'atom_atomic_nums': atomic_nums_tensor,
        'atom_max_valences': max_valences_tensor,
        'atom_is_hydrogen': is_hydrogen_tensor
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TESTING bond_types.py")
    print("="*70)
    
    # Test 1: Water
    print("\n1. Water (H-O-H):")
    smiles = "[H]O[H]"
    info = get_bond_types_from_smiles(smiles, num_atoms=3, verbose=True)
    print(f"   SMILES: {smiles}")
    print(f"   Bond info: {info['bond_info']}")
    print(f"   Atom info: {info['atom_info']}")
    
    # Test 2: Ethylene (C=C)
    print("\n2. Ethylene (C=C):")
    smiles = "C=C"
    info = get_bond_types_from_smiles(smiles, verbose=True)
    print(f"   SMILES: {smiles}")
    print(f"   Bond info: {info['bond_info']}")
    
    # Test 3: Acetylene (C#C)
    print("\n3. Acetylene (C#C):")
    smiles = "C#C"
    info = get_bond_types_from_smiles(smiles, verbose=True)
    print(f"   SMILES: {smiles}")
    print(f"   Bond info: {info['bond_info']}")
    
    # Test 4: Benzene (aromatic)
    print("\n4. Benzene (aromatic):")
    smiles = "c1ccccc1"
    info = get_bond_types_from_smiles(smiles, verbose=True)
    print(f"   SMILES: {smiles}")
    print(f"   Bond info (first 3): {info['bond_info'][:3]}")
    
    # Test 5: Batch extraction
    print("\n5. Batch extraction:")
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
    
    edge_index = get_bonds_from_batch(batch['smiles'], batch['mask'])
    chem_info = get_bond_types_from_batch(batch['smiles'], batch['mask'], edge_index)
    
    print(f"   Batch size: {len(batch['smiles'])}")
    print(f"   Total atoms: {int(batch['mask'].sum().item())}")
    print(f"   Total edges: {edge_index.shape[1]}")
    print(f"   Bond types shape: {chem_info['bond_types'].shape}")
    print(f"   Expected lengths shape: {chem_info['expected_lengths'].shape}")
    print(f"   Bond types (first 10): {chem_info['bond_types'][:10].tolist()}")
    print(f"   Expected lengths (first 10): {chem_info['expected_lengths'][:10].tolist()}")
    print(f"   Hydrogen count: {chem_info['atom_is_hydrogen'].sum().item():.0f}")
