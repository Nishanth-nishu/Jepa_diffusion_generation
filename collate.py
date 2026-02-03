"""
collate.py — BATCH ASSEMBLY ONLY

Purpose:
    Convert list of molecules → batched tensors

Responsibilities:
    1. Pad atoms to max_atoms_in_batch
    2. Pad coordinates
    3. Build batch masks
    4. Preserve SMILES as a list

Output of collate_fn:
    {
        "x": Tensor[B, N, atom_dim],
        "pos": Tensor[B, N, 3],
        "mask": Tensor[B, N],         # 1 = real atom, 0 = padding
        "smiles": List[str],          # length B
    }

Critical invariants:
    - Padding atoms must never leak into model
    - mask must be correct
    - smiles[i] corresponds to x[i], pos[i], mask[i]

What is NOT here:
    ❌ RDKit
    ❌ Graph construction
    ❌ Flattening
    ❌ Loss logic
"""

import torch


def collate_fn(batch):
    """
    Collate a list of molecules into batched tensors.
    
    Args:
        batch: List[Dict], each dict from dataset.__getitem__:
            {
                "x": Tensor[n_atoms, atom_dim],
                "pos": Tensor[n_atoms, 3],
                "mask": Tensor[n_atoms],
                "smiles": str,
            }
    
    Returns:
        dict:
            {
                "x": Tensor[B, max_atoms, atom_dim],
                "pos": Tensor[B, max_atoms, 3],
                "mask": Tensor[B, max_atoms],  # 1 = real atom, 0 = padding
                "smiles": List[str],  # length B
            }
    """
    # Extract batch size
    batch_size = len(batch)
    
    # Find max number of atoms in this batch
    max_atoms = max(sample["x"].shape[0] for sample in batch)
    
    # Get feature dimension
    atom_dim = batch[0]["x"].shape[1]
    
    # Initialize padded tensors
    x_batched = torch.zeros(batch_size, max_atoms, atom_dim, dtype=torch.float32)
    pos_batched = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    mask_batched = torch.zeros(batch_size, max_atoms, dtype=torch.float32)
    
    # Collect SMILES
    smiles_list = []
    
    # Fill batch
    for i, sample in enumerate(batch):
        n_atoms = sample["x"].shape[0]
        
        # Copy data into padded tensors
        x_batched[i, :n_atoms] = sample["x"]
        pos_batched[i, :n_atoms] = sample["pos"]
        mask_batched[i, :n_atoms] = sample["mask"]  # Should be all ones
        
        smiles_list.append(sample["smiles"])
    
    return {
        "x": x_batched,           # (B, max_atoms, atom_dim)
        "pos": pos_batched,       # (B, max_atoms, 3)
        "mask": mask_batched,     # (B, max_atoms) - 1 = real, 0 = padding
        "smiles": smiles_list,    # List[str] length B
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    from dataset import QM9Text3DDataset
    from torch.utils.data import DataLoader
    
    # Load dataset
    dataset = QM9Text3DDataset(
        jsonl_path='/scratch/nishanth.r/egnn/data/qm9_100k.jsonl',
        atom_feature_dim=10
    )
    
    # Create dataloader with collate_fn
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Test batch
    batch = next(iter(loader))
    
    print("Batch shapes:")
    print(f"  x: {batch['x'].shape}")
    print(f"  pos: {batch['pos'].shape}")
    print(f"  mask: {batch['mask'].shape}")
    print(f"  smiles: {len(batch['smiles'])} molecules")
    
    print("\nBatch contents:")
    for i in range(len(batch['smiles'])):
        n_atoms = int(batch['mask'][i].sum().item())
        print(f"  Molecule {i}: {n_atoms} atoms, SMILES={batch['smiles'][i]}")
    
    print("\nMask validation:")
    for i in range(len(batch['smiles'])):
        mask_sum = batch['mask'][i].sum().item()
        print(f"  Molecule {i}: {int(mask_sum)} real atoms")
