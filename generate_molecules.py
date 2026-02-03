"""
generate_molecules.py — MOLECULE GENERATION PIPELINE

Generates novel 3D molecules using the trained diffusion model.

Pipeline:
    1. Generate random atom types (C, N, O, F, etc.)
    2. Generate geometry via diffusion (noise → geometry)
    3. Reconstruct 3D coordinates from geometry
    4. Validate with RDKit and save
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings

# RDKit imports (for chemistry validation)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Molecule validation will be limited.")

# Local imports
from geometry_diffusion import GeometryDiffusion
from diffusion_jepa import JEPADiffusion


# ============================================================================
# ATOM TYPE GENERATION
# ============================================================================

# Common organic atom types with atomic numbers and typical valences
ATOM_INFO = {
    6: {'symbol': 'C', 'valence': 4, 'weight': 0.4},   # Carbon - most common
    7: {'symbol': 'N', 'valence': 3, 'weight': 0.2},   # Nitrogen
    8: {'symbol': 'O', 'valence': 2, 'weight': 0.25},  # Oxygen
    9: {'symbol': 'F', 'valence': 1, 'weight': 0.05},  # Fluorine
    16: {'symbol': 'S', 'valence': 2, 'weight': 0.05}, # Sulfur
    17: {'symbol': 'Cl', 'valence': 1, 'weight': 0.03}, # Chlorine
    35: {'symbol': 'Br', 'valence': 1, 'weight': 0.02}, # Bromine
}

# Add hydrogen separately (often added implicitly)
HYDROGEN = {1: {'symbol': 'H', 'valence': 1}}


def generate_atom_types(n_atoms, include_h=False, mode='weighted'):
    """
    Generate random atom types for a molecule.
    
    Args:
        n_atoms: Number of atoms to generate
        include_h: Whether to include explicit hydrogens
        mode: 'weighted' (based on frequency) or 'uniform'
    
    Returns:
        atom_types: List of atomic numbers
    """
    if include_h:
        atoms = {**ATOM_INFO, **HYDROGEN}
    else:
        atoms = ATOM_INFO
    
    atomic_nums = list(atoms.keys())
    
    if mode == 'weighted':
        weights = [atoms[a].get('weight', 0.1) for a in atomic_nums]
        weights = np.array(weights) / sum(weights)
        atom_types = np.random.choice(atomic_nums, size=n_atoms, p=weights).tolist()
    else:
        atom_types = random.choices(atomic_nums, k=n_atoms)
    
    return atom_types


def generate_valid_atom_composition(n_atoms):
    """
    Generate chemically reasonable atom composition.
    
    Ensures:
    - At least some carbons (organic molecules)
    - Reasonable heteroatom ratios
    """
    if n_atoms < 3:
        return [6] * n_atoms  # Just carbons
    
    # Start with C as backbone
    n_carbons = max(1, n_atoms // 2)
    n_heteroatoms = n_atoms - n_carbons
    
    atom_types = [6] * n_carbons
    
    # Add heteroatoms
    heteroatom_types = [7, 8, 9]  # N, O, F
    hetero_weights = [0.4, 0.5, 0.1]
    
    for _ in range(n_heteroatoms):
        atom = np.random.choice(heteroatom_types, p=hetero_weights)
        atom_types.append(atom)
    
    # Shuffle
    random.shuffle(atom_types)
    
    return atom_types


# ============================================================================
# MOLECULE GENERATOR
# ============================================================================

class MoleculeGenerator:
    """
    Main class for generating molecules via diffusion.
    """
    
    def __init__(self, diffusion_model, denoiser_model, jepa_model=None, device='cuda'):
        """
        Args:
            diffusion_model: Trained GeometryDiffusion
            denoiser_model: Trained JEPADiffusion
            jepa_model: Optional trained JEPA for coordinate reconstruction
            device: Target device
        """
        self.diffusion = diffusion_model.to(device)
        self.denoiser = denoiser_model.to(device)
        self.jepa = jepa_model.to(device) if jepa_model else None
        self.device = device
        
        self.diffusion.eval()
        self.denoiser.eval()
        if self.jepa:
            self.jepa.eval()
    
    @torch.no_grad()
    def generate_geometry(self, atom_types, num_steps=50, use_ddim=True):
        """
        Generate geometry via diffusion.
        
        Args:
            atom_types: List of atomic numbers
            num_steps: Number of diffusion steps
            use_ddim: Whether to use DDIM (faster) or DDPM
        
        Returns:
            geometry: Tensor of shape (geometry_dim,)
        """
        n_atoms = len(atom_types)
        geometry_dim = self.diffusion.geometry_dim
        
        # Convert atom types to tensor
        atom_types_tensor = torch.tensor(atom_types, device=self.device).unsqueeze(0)
        
        # Generate geometry via diffusion
        if use_ddim:
            geometry = self.diffusion.ddim_sample(
                self.denoiser,
                shape=(1, geometry_dim),
                atom_types=atom_types_tensor,
                num_steps=num_steps,
                eta=0.0  # Deterministic
            )
        else:
            geometry = self.diffusion.sample(
                self.denoiser,
                shape=(1, geometry_dim),
                atom_types=atom_types_tensor,
                num_steps=num_steps,
                progress=False
            )
        
        return geometry.squeeze(0)
    
    def geometry_to_coords(self, geometry, atom_types):
        """
        Convert geometry representation to 3D coordinates.
        
        Uses the geometry to define pairwise distances, then optimizes
        coordinates to match those distances.
        
        Args:
            geometry: Geometry tensor (geometry_dim,)
            atom_types: List of atomic numbers
        
        Returns:
            coords: Tensor of shape (n_atoms, 3)
        """
        n_atoms = len(atom_types)
        
        # Ensure geometry is detached from any computation graph
        geometry = geometry.detach().clone()
        
        # Denormalize geometry to get distances
        dist_mean = 1.5
        dist_std = 1.0
        distances = geometry[:n_atoms * (n_atoms - 1) // 2] * dist_std + dist_mean
        distances = torch.clamp(distances, min=0.8, max=5.0)  # Reasonable range
        
        # Optimize 3D coordinates to match distances
        coords = self._optimize_coords_from_distances(distances, n_atoms)
        
        return coords
    
    def _optimize_coords_from_distances(self, distances, n_atoms, n_steps=100):
        """
        Optimize 3D coordinates to match target pairwise distances.
        
        Uses gradient descent to find coordinates that satisfy the
        distance constraints.
        """
        # Need to enable grad since this might be called from @torch.no_grad() context
        with torch.enable_grad():
            # Initialize random coordinates
            coords = torch.randn(n_atoms, 3, device=self.device) * 0.5
            coords = coords.clone().detach().requires_grad_(True)
            
            optimizer = torch.optim.Adam([coords], lr=0.1)
            
            # Get pair indices
            triu_idx = torch.triu_indices(n_atoms, n_atoms, offset=1, device=self.device)
            n_pairs = triu_idx.shape[1]
            
            # Ensure we have enough distances and detach from diffusion graph
            if len(distances) >= n_pairs:
                target_dists = distances[:n_pairs].detach().clone()
            else:
                target_dists = F.pad(distances.detach().clone(), (0, n_pairs - len(distances)), value=1.5)
            
            for step in range(n_steps):
                optimizer.zero_grad()
                
                # Compute current distances
                diff = coords[triu_idx[0]] - coords[triu_idx[1]]
                current_dists = torch.norm(diff, dim=-1)
                
                # Distance loss
                loss = F.mse_loss(current_dists, target_dists)
                
                # Add regularization to keep structure compact
                center = coords.mean(dim=0)
                loss = loss + 0.01 * ((coords - center) ** 2).sum()
                
                loss.backward()
                optimizer.step()
            
            return coords.detach()
    
    @torch.no_grad()
    def generate(self, n_atoms=None, atom_types=None, num_steps=50, use_ddim=True):
        """
        Full generation pipeline.
        
        Args:
            n_atoms: Number of atoms (randomly sampled if None)
            atom_types: Atom types (randomly generated if None)
            num_steps: Diffusion steps
            use_ddim: Use DDIM sampling
        
        Returns:
            atom_types: List of atomic numbers
            coords: 3D coordinates (n_atoms, 3)
        """
        # Generate atom types if not provided
        if atom_types is None:
            if n_atoms is None:
                n_atoms = random.randint(5, 15)
            atom_types = generate_valid_atom_composition(n_atoms)
        else:
            n_atoms = len(atom_types)
        
        # Generate geometry
        geometry = self.generate_geometry(atom_types, num_steps, use_ddim)
        
        # Convert to 3D coordinates
        coords = self.geometry_to_coords(geometry, atom_types)
        
        return atom_types, coords


# ============================================================================
# RDKIT CONVERSION AND VALIDATION
# ============================================================================

def coords_to_molecule(atom_types, coords, add_hydrogens=True):
    """
    Convert atom types and coordinates to RDKit molecule.
    
    Args:
        atom_types: List of atomic numbers
        coords: Numpy array or tensor of shape (n_atoms, 3)
        add_hydrogens: Whether to add implicit hydrogens
    
    Returns:
        mol: RDKit Mol object or None if failed
    """
    if not RDKIT_AVAILABLE:
        return None
    
    if torch.is_tensor(coords):
        coords = coords.cpu().numpy()
    
    try:
        # Create editable molecule
        mol = Chem.RWMol()
        
        # Add atoms
        for atomic_num in atom_types:
            atom = Chem.Atom(int(atomic_num))
            mol.AddAtom(atom)
        
        # Add bonds based on distances
        n_atoms = len(atom_types)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                
                # Determine bond type from distance
                bond_type = get_bond_type_from_distance(
                    atom_types[i], atom_types[j], dist
                )
                
                if bond_type is not None:
                    mol.AddBond(i, j, bond_type)
        
        # Convert to regular molecule
        mol = mol.GetMol()
        
        # Try to sanitize
        try:
            Chem.SanitizeMol(mol)
        except:
            pass  # May fail for unusual structures
        
        # Add 3D conformer
        conf = Chem.Conformer(n_atoms)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, coords[i].tolist())
        mol.AddConformer(conf, assignId=True)
        
        # Add hydrogens if requested
        if add_hydrogens:
            try:
                mol = Chem.AddHs(mol, addCoords=True)
            except:
                pass
        
        return mol
    
    except Exception as e:
        if verbose:
            print(f"Failed to create molecule: {e}")
        return None


def get_bond_type_from_distance(atom1, atom2, dist):
    """
    Infer bond type from atomic numbers and distance.
    
    Uses typical bond lengths for different bond orders.
    """
    # Typical bond lengths (in Angstroms)
    # C-C: 1.54 (single), 1.34 (double), 1.20 (triple)
    # C-N: 1.47 (single), 1.29 (double)
    # C-O: 1.43 (single), 1.23 (double)
    # C-H: 1.09
    
    # Covalent radii (approximate)
    covalent_radii = {
        1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
        16: 1.05, 17: 1.02, 35: 1.20
    }
    
    r1 = covalent_radii.get(atom1, 0.77)
    r2 = covalent_radii.get(atom2, 0.77)
    
    # Expected single bond length
    single_bond = r1 + r2
    
    # Tolerance
    tol = 0.4
    
    if dist < single_bond * 0.85 - tol:
        return Chem.BondType.TRIPLE
    elif dist < single_bond * 0.95 - tol:
        return Chem.BondType.DOUBLE
    elif dist < single_bond + tol:
        return Chem.BondType.SINGLE
    else:
        return None  # No bond


def validate_molecule(mol):
    """
    Validate a generated molecule.
    
    Checks:
    - Valency correctness
    - Connectivity
    - Ring strain
    - Reasonable molecular weight
    
    Returns:
        is_valid: Boolean
        issues: List of issues found
    """
    if mol is None:
        return False, ['Molecule is None']
    
    if not RDKIT_AVAILABLE:
        return True, []  # Can't validate without RDKit
    
    issues = []
    
    try:
        # Check valency
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            issues.append(f'Sanitization failed: {e}')
        
        # Check connectivity
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            issues.append(f'Molecule is fragmented ({len(frags)} fragments)')
        
        # Check molecular weight
        mw = Descriptors.MolWt(mol)
        if mw < 30:
            issues.append(f'Molecular weight too low: {mw:.1f}')
        if mw > 500:
            issues.append(f'Molecular weight too high: {mw:.1f}')
        
        # Check for unreasonable ring sizes
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) < 3:
                issues.append(f'Invalid ring size: {len(ring)}')
            elif len(ring) > 8:
                issues.append(f'Large ring: {len(ring)} atoms')
        
        # Check for charged atoms (might indicate issues)
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                issues.append(f'Atom {atom.GetSymbol()} has charge {atom.GetFormalCharge()}')
        
    except Exception as e:
        issues.append(f'Validation error: {e}')
    
    is_valid = len(issues) == 0
    return is_valid, issues


def compute_molecule_properties(mol):
    """
    Compute molecular properties for a valid molecule.
    
    Returns:
        properties: Dict of computed properties
    """
    if mol is None or not RDKIT_AVAILABLE:
        return {}
    
    try:
        props = {
            'molecular_formula': CalcMolFormula(mol),
            'molecular_weight': Descriptors.MolWt(mol),
            'logP': Descriptors.MolLogP(mol),
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rings': Descriptors.RingCount(mol),
        }
        return props
    except:
        return {}


# ============================================================================
# BATCH GENERATION
# ============================================================================

def generate_molecules(generator, num_molecules=100, n_atoms_range=(5, 15),
                       num_steps=50, use_ddim=True, verbose=True):
    """
    Generate multiple molecules.
    
    Args:
        generator: MoleculeGenerator instance
        num_molecules: Number to generate
        n_atoms_range: (min, max) atoms per molecule
        num_steps: Diffusion steps
        use_ddim: Use DDIM sampling
        verbose: Print progress
    
    Returns:
        molecules: List of (atom_types, coords, mol, properties) tuples
        stats: Generation statistics
    """
    molecules = []
    valid_count = 0
    
    iterator = range(num_molecules)
    if verbose:
        iterator = tqdm(iterator, desc='Generating molecules')
    
    for i in iterator:
        try:
            # Random number of atoms
            n_atoms = random.randint(*n_atoms_range)
            
            # Generate
            atom_types, coords = generator.generate(
                n_atoms=n_atoms,
                num_steps=num_steps,
                use_ddim=use_ddim
            )
            
            # Convert to RDKit molecule
            mol = coords_to_molecule(atom_types, coords)
            
            # Validate
            is_valid, issues = validate_molecule(mol)
            
            if is_valid:
                valid_count += 1
                props = compute_molecule_properties(mol)
                molecules.append({
                    'atom_types': atom_types,
                    'coords': coords.cpu().numpy(),
                    'mol': mol,
                    'properties': props
                })
                
                if verbose:
                    formula = props.get('molecular_formula', 'Unknown')
                    tqdm.write(f"  Valid: {formula}")
            else:
                if verbose and len(issues) > 0:
                    tqdm.write(f"  Invalid: {issues[0]}")
        
        except Exception as e:
            if verbose:
                tqdm.write(f"  Error: {e}")
    
    stats = {
        'total_generated': num_molecules,
        'valid_count': valid_count,
        'validity_rate': valid_count / num_molecules * 100 if num_molecules > 0 else 0
    }
    
    return molecules, stats


def save_molecules(molecules, output_path, format='sdf'):
    """
    Save generated molecules to file.
    
    Args:
        molecules: List of molecule dicts
        output_path: Path to save
        format: 'sdf' or 'mol2'
    """
    if not RDKIT_AVAILABLE:
        print("Cannot save molecules without RDKit")
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'sdf':
        with Chem.SDWriter(str(output_path)) as writer:
            for i, mol_data in enumerate(molecules):
                mol = mol_data['mol']
                if mol is not None:
                    # Add properties
                    for key, value in mol_data.get('properties', {}).items():
                        mol.SetProp(key, str(value))
                    mol.SetProp('_Name', f'Generated_{i}')
                    writer.write(mol)
    
    print(f"Saved {len(molecules)} molecules to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate molecules via diffusion')
    
    # Model paths
    parser.add_argument('--diffusion_path', type=str, 
                        default='checkpoints/diffusion/best_diffusion_model.pt',
                        help='Path to trained diffusion model')
    parser.add_argument('--geometry_dim', type=int, default=64,
                        help='Geometry dimension')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    
    # Generation
    parser.add_argument('--num_molecules', type=int, default=100,
                        help='Number of molecules to generate')
    parser.add_argument('--min_atoms', type=int, default=5,
                        help='Minimum atoms per molecule')
    parser.add_argument('--max_atoms', type=int, default=15,
                        help='Maximum atoms per molecule')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of diffusion steps')
    parser.add_argument('--use_ddim', action='store_true', default=True,
                        help='Use DDIM sampling (faster)')
    
    # Output
    parser.add_argument('--output', type=str, default='generated_molecules.sdf',
                        help='Output SDF file')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("MOLECULE GENERATION VIA DIFFUSION")
    print("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create models
    diffusion = GeometryDiffusion(
        geometry_dim=args.geometry_dim,
        num_timesteps=args.num_timesteps,
        schedule='cosine'
    ).to(device)
    
    denoiser = JEPADiffusion(
        pretrained_path=None,
        geometry_dim=args.geometry_dim,
        latent_dim=args.latent_dim,
        num_timesteps=args.num_timesteps
    ).to(device)
    
    # Load checkpoint if exists
    if Path(args.diffusion_path).exists():
        print(f"Loading checkpoint from {args.diffusion_path}")
        checkpoint = torch.load(args.diffusion_path, map_location=device)
        diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
        print("  Loaded successfully!")
    else:
        print("No checkpoint found - using randomly initialized model")
        print("  (For meaningful generation, train the model first)")
    
    # Create generator
    generator = MoleculeGenerator(diffusion, denoiser, device=device)
    
    # Generate molecules
    print(f"\nGenerating {args.num_molecules} molecules...")
    molecules, stats = generate_molecules(
        generator,
        num_molecules=args.num_molecules,
        n_atoms_range=(args.min_atoms, args.max_atoms),
        num_steps=args.num_steps,
        use_ddim=args.use_ddim,
        verbose=True
    )
    
    # Print stats
    print("\n" + "=" * 60)
    print("GENERATION STATISTICS")
    print("=" * 60)
    print(f"Total generated: {stats['total_generated']}")
    print(f"Valid molecules: {stats['valid_count']}")
    print(f"Validity rate: {stats['validity_rate']:.1f}%")
    
    if molecules:
        mws = [m['properties'].get('molecular_weight', 0) for m in molecules]
        print(f"Avg molecular weight: {np.mean(mws):.1f} ± {np.std(mws):.1f}")
        
        # Save molecules
        save_molecules(molecules, args.output)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
