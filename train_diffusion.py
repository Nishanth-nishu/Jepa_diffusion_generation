"""
train_diffusion.py — TRAIN GEOMETRY DIFFUSION MODEL

Trains the diffusion model on molecular geometry from QM9.

Pipeline:
    1. Load QM9 data
    2. Extract geometry (bond lengths, angles, torsions) from each molecule
    3. Train diffusion to denoise geometry
    4. Use pretrained JEPA as initialization/denoiser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import math

# Local imports
from geometry_diffusion import GeometryDiffusion, GeometryRepresentation
from diffusion_jepa import JEPADiffusion
from dataset_optimized import QM9OptimizedDataset, collate_fn_optimized
import geometry


# ============================================================================
# GEOMETRY EXTRACTION
# ============================================================================

def extract_geometry_from_batch(batch, device):
    """
    Extract geometry vectors from a batch of molecules.
    
    Args:
        batch: Dict from QM9OptimizedDataset
        device: Target device
    
    Returns:
        geometry_batch: Tensor of shape (batch_size, max_atoms, geometry_dim)
        atom_types_batch: Tensor of shape (batch_size, max_atoms)
        mask: Tensor of shape (batch_size, max_atoms)
    """
    x = batch['x'].to(device)
    pos = batch['pos'].to(device)
    mask = batch['mask'].to(device)
    edge_index = batch['edge_index'].to(device)
    bond_types = batch['bond_types'].to(device)
    
    B, max_atoms, atom_dim = x.shape
    
    # We'll store geometry per-atom
    # Each atom has: local bond info + angle info + torsion info
    # Simplified: use a fixed geometry_dim per atom
    geometry_dim = 32  # Can be tuned
    
    geometry_batch = torch.zeros(B, max_atoms, geometry_dim, device=device)
    atom_types_batch = torch.zeros(B, max_atoms, dtype=torch.long, device=device)
    
    for b in range(B):
        mol_mask = mask[b].bool()
        n_atoms = mol_mask.sum().item()
        if n_atoms < 2:
            continue
        
        mol_pos = pos[b, mol_mask]
        mol_x = x[b, mol_mask]
        
        # Get atom types from one-hot encoding
        # Assuming x contains one-hot or atomic numbers
        if mol_x.dim() == 2 and mol_x.shape[1] > 1:
            atom_types_batch[b, :n_atoms] = mol_x.argmax(dim=-1)
        else:
            atom_types_batch[b, :n_atoms] = mol_x.squeeze(-1).long()
        
        # Extract per-atom local geometry features
        for i in range(n_atoms):
            geom_vec = []
            
            # Distance to all other atoms (normalized)
            dists = torch.norm(mol_pos - mol_pos[i], dim=-1)
            dists = dists[dists > 0]  # Exclude self
            if len(dists) > 0:
                # Statistics of distances
                geom_vec.extend([
                    dists.min().item(),
                    dists.max().item(),
                    dists.mean().item(),
                    dists.std().item() if len(dists) > 1 else 0.0
                ])
            else:
                geom_vec.extend([0.0, 0.0, 0.0, 0.0])
            
            # Normalized position (relative to center)
            center = mol_pos.mean(dim=0)
            rel_pos = mol_pos[i] - center
            geom_vec.extend(rel_pos.tolist())
            
            # Pad to geometry_dim
            while len(geom_vec) < geometry_dim:
                geom_vec.append(0.0)
            
            geometry_batch[b, i, :] = torch.tensor(geom_vec[:geometry_dim], device=device)
    
    return geometry_batch, atom_types_batch, mask


def extract_global_geometry(pos, edge_index, num_atoms):
    """
    Extract global geometry representation for a molecule.
    
    Returns:
        geometry: Dict with:
            - bond_lengths: Tensor (num_bonds,)
            - angles: Tensor (num_angles,)
            - torsions: Tensor (num_torsions, 2) - sin/cos
    """
    device = pos.device
    
    # Bond lengths
    if edge_index.shape[1] > 0:
        src, dst = edge_index
        bond_lengths = torch.norm(pos[src] - pos[dst], dim=-1)
    else:
        bond_lengths = torch.zeros(0, device=device)
    
    # Angles
    angle_triplets = geometry.get_angles(edge_index, num_atoms)
    if angle_triplets.shape[0] > 0:
        i, j, k = angle_triplets[:, 0], angle_triplets[:, 1], angle_triplets[:, 2]
        v1 = pos[i] - pos[j]
        v2 = pos[k] - pos[j]
        cos_angle = F.cosine_similarity(v1, v2, dim=-1).clamp(-1.0, 1.0)
        angles = torch.acos(cos_angle)
    else:
        angles = torch.zeros(0, device=device)
    
    # Torsions
    torsion_quads = geometry.get_torsions(edge_index, num_atoms)
    if torsion_quads.shape[0] > 0:
        i, j, k, l = torsion_quads[:, 0], torsion_quads[:, 1], torsion_quads[:, 2], torsion_quads[:, 3]
        v1 = pos[j] - pos[i]
        v2 = pos[k] - pos[j]
        v3 = pos[l] - pos[k]
        n1 = F.normalize(torch.cross(v1, v2, dim=-1) + 1e-6, dim=-1)
        n2 = F.normalize(torch.cross(v2, v3, dim=-1) + 1e-6, dim=-1)
        v2_norm = F.normalize(v2 + 1e-6, dim=-1)
        cos_torsion = (n1 * n2).sum(dim=-1)
        sin_torsion = (torch.cross(n1, n2, dim=-1) * v2_norm).sum(dim=-1)
        torsions = torch.stack([sin_torsion, cos_torsion], dim=-1)
    else:
        torsions = torch.zeros((0, 2), device=device)
    
    return {
        'bond_lengths': bond_lengths,
        'angles': angles,
        'torsions': torsions
    }


# ============================================================================
# GEOMETRY DATASET
# ============================================================================

class GeometryDataset(Dataset):
    """
    Dataset that provides pre-extracted geometry vectors.
    
    Each sample is a fixed-size geometry vector representing a molecule.
    """
    
    def __init__(self, qm9_dataset, geometry_dim=64, max_atoms=29):
        """
        Args:
            qm9_dataset: QM9OptimizedDataset instance
            geometry_dim: Dimension of geometry vector
            max_atoms: Maximum number of atoms
        """
        self.qm9 = qm9_dataset
        self.geometry_dim = geometry_dim
        self.max_atoms = max_atoms
        
        # Pre-extract all geometry (can be slow for large datasets)
        print("Pre-extracting geometry from QM9...")
        self.geometry_data = []
        
        for idx in tqdm(range(len(qm9_dataset)), desc="Extracting geometry"):
            sample = qm9_dataset[idx]
            geom = self._extract_single(sample)
            if geom is not None:
                self.geometry_data.append(geom)
        
        print(f"Extracted {len(self.geometry_data)} valid geometries")
    
    def _extract_single(self, sample):
        """Extract geometry from a single sample.
        
        QM9OptimizedDataset returns trimmed molecules with:
            - 'pos': (n_atoms, 3)
            - 'x': (n_atoms, atom_dim)
            - 'n_atoms': int
            - 'atom_atomic_nums': (n_atoms,)
        """
        try:
            pos = sample['pos']
            n_atoms = sample['n_atoms']
            
            if n_atoms < 3:  # Need at least 3 atoms for angles
                return None
            
            # pos is already trimmed to n_atoms
            mol_pos = pos[:n_atoms] if pos.shape[0] > n_atoms else pos
            
            # Simple geometry: pairwise distances
            # Flatten upper triangle of distance matrix
            dists = torch.cdist(mol_pos, mol_pos, p=2)
            n = mol_pos.shape[0]
            triu_idx = torch.triu_indices(n, n, offset=1)
            dist_vec = dists[triu_idx[0], triu_idx[1]]
            
            # Normalize distances
            dist_mean = 1.5  # Typical bond length
            dist_std = 1.0
            dist_vec = (dist_vec - dist_mean) / dist_std
            
            # Pad or truncate to geometry_dim
            if len(dist_vec) < self.geometry_dim:
                padded = torch.zeros(self.geometry_dim)
                padded[:len(dist_vec)] = dist_vec
                geometry_vec = padded
            else:
                geometry_vec = dist_vec[:self.geometry_dim]
            
            # Get atom types - use atom_atomic_nums if available
            if 'atom_atomic_nums' in sample:
                atom_types = sample['atom_atomic_nums'][:n_atoms]
            else:
                # Fallback to x features
                x = sample['x']
                mol_x = x[:n_atoms] if x.shape[0] > n_atoms else x
                if mol_x.dim() == 2:
                    atom_types = mol_x.argmax(dim=-1)
                else:
                    atom_types = mol_x.long()
            
            # Pad atom types
            atom_types_padded = torch.zeros(self.max_atoms, dtype=torch.long)
            atom_types_padded[:n_atoms] = atom_types[:n_atoms]
            
            return {
                'geometry': geometry_vec.float(),
                'atom_types': atom_types_padded,
                'n_atoms': n_atoms
            }
        except Exception as e:
            # Uncomment for debugging:
            # print(f"Error extracting geometry: {e}")
            return None
    
    def __len__(self):
        return len(self.geometry_data)
    
    def __getitem__(self, idx):
        return self.geometry_data[idx]


def geometry_collate_fn(batch):
    """Collate function for GeometryDataset."""
    geometries = torch.stack([b['geometry'] for b in batch])
    atom_types = torch.stack([b['atom_types'] for b in batch])
    n_atoms = torch.tensor([b['n_atoms'] for b in batch])
    
    return {
        'geometry': geometries,
        'atom_types': atom_types,
        'n_atoms': n_atoms
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_diffusion(args):
    """Main training function."""
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load QM9 dataset
    print(f"Loading QM9 from {args.data_path}...")
    qm9 = QM9OptimizedDataset(args.data_path, verbose=True)
    
    # Create geometry dataset
    geometry_dataset = GeometryDataset(qm9, geometry_dim=args.geometry_dim)
    
    # Split
    n_total = len(geometry_dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        geometry_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=geometry_collate_fn,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=geometry_collate_fn,
        num_workers=args.num_workers
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create models
    diffusion = GeometryDiffusion(
        geometry_dim=args.geometry_dim,
        num_timesteps=args.num_timesteps,
        schedule=args.schedule
    ).to(device)
    
    denoiser = JEPADiffusion(
        pretrained_path=args.pretrained_jepa if args.use_pretrained else None,
        geometry_dim=args.geometry_dim,
        latent_dim=args.latent_dim,
        num_timesteps=args.num_timesteps,
        prediction_type=args.prediction_type
    ).to(device)
    
    print(f"Diffusion model: {sum(p.numel() for p in diffusion.parameters())} parameters")
    print(f"Denoiser model: {sum(p.numel() for p in denoiser.parameters())} parameters")
    
    # Optimizer
    optimizer = AdamW(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Training
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            diffusion, denoiser, train_loader, optimizer, device, epoch, args
        )
        
        # Validate
        val_loss = validate(diffusion, denoiser, val_loader, device, args)
        
        # Update scheduler
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                diffusion, denoiser, optimizer, epoch, val_loss,
                args.save_path / 'best_diffusion_model.pt'
            )
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                diffusion, denoiser, optimizer, epoch, val_loss,
                args.save_path / f'diffusion_epoch_{epoch+1}.pt'
            )
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


def train_epoch(diffusion, denoiser, loader, optimizer, device, epoch, args):
    """Train for one epoch."""
    denoiser.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        geometry = batch['geometry'].to(device)
        atom_types = batch['atom_types'].to(device)
        
        # Sample random timesteps
        t = torch.randint(0, args.num_timesteps, (geometry.shape[0],), device=device)
        
        # Forward diffusion
        noisy_geometry, noise = diffusion.forward_diffusion(geometry, t)
        
        # Predict noise
        if args.prediction_type == 'noise':
            pred = denoiser(noisy_geometry, t, atom_types)
            target = noise
        else:  # 'x0'
            pred = denoiser(noisy_geometry, t, atom_types)
            target = geometry
        
        # Loss
        loss = F.mse_loss(pred, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), args.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(diffusion, denoiser, loader, device, args):
    """Validate on held-out data."""
    denoiser.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        geometry = batch['geometry'].to(device)
        atom_types = batch['atom_types'].to(device)
        
        # Sample random timesteps
        t = torch.randint(0, args.num_timesteps, (geometry.shape[0],), device=device)
        
        # Forward diffusion
        noisy_geometry, noise = diffusion.forward_diffusion(geometry, t)
        
        # Predict
        if args.prediction_type == 'noise':
            pred = denoiser(noisy_geometry, t, atom_types)
            target = noise
        else:
            pred = denoiser(noisy_geometry, t, atom_types)
            target = geometry
        
        loss = F.mse_loss(pred, target)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(diffusion, denoiser, optimizer, epoch, val_loss, path):
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'diffusion_state_dict': diffusion.state_dict(),
        'denoiser_state_dict': denoiser.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Geometry Diffusion Model')
    
    # Data
    parser.add_argument('--data_path', type=str, default='data/qm9_100k.jsonl',
                        help='Path to QM9 JSONL file')
    parser.add_argument('--save_path', type=str, default='checkpoints/diffusion',
                        help='Directory to save checkpoints')
    
    # Model
    parser.add_argument('--geometry_dim', type=int, default=64,
                        help='Dimension of geometry vector')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=['cosine', 'linear'], help='Noise schedule')
    parser.add_argument('--prediction_type', type=str, default='noise',
                        choices=['noise', 'x0'], help='What to predict')
    
    # Pretrained JEPA
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Use pretrained JEPA for initialization')
    parser.add_argument('--pretrained_jepa', type=str, 
                        default='best_pure_jepa_transformer.pt',
                        help='Path to pretrained JEPA')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    args.save_path = Path(args.save_path)
    
    print("=" * 60)
    print("GEOMETRY DIFFUSION TRAINING")
    print("=" * 60)
    print(f"Config: {vars(args)}")
    print("=" * 60)
    
    train_diffusion(args)


if __name__ == '__main__':
    main()
