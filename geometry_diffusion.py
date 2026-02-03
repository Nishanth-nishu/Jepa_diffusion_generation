"""
geometry_diffusion.py — GEOMETRY DIFFUSION MODEL

Core diffusion model for molecular geometry generation.
Uses JEPA as the denoising network to generate valid molecular geometry
(bond lengths, angles, torsions) from noise.

Architecture:
    1. Forward diffusion: Clean geometry → Noisy geometry
    2. Reverse diffusion: Noisy geometry → Clean geometry (via JEPA)
    3. Reconstruction: Geometry → 3D coordinates (via JEPA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# NOISE SCHEDULE
# ============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine noise schedule (from Improved DDPM paper).
    
    Better than linear for molecular data because:
    - Slower noise addition at start (preserves structure longer)
    - Faster at end (less computation wasted on pure noise)
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent beta=0 at t=0
    
    Returns:
        betas: Tensor of shape (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Linear noise schedule (original DDPM).
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        betas: Tensor of shape (timesteps,)
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def timestep_embedding(t, dim, max_period=10000):
    """
    Sinusoidal timestep embedding (from Transformer/DDPM).
    
    Args:
        t: Tensor of timesteps, shape (batch_size,)
        dim: Embedding dimension
        max_period: Maximum period for sinusoidal encoding
    
    Returns:
        embedding: Tensor of shape (batch_size, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ============================================================================
# GEOMETRY DIFFUSION MODEL
# ============================================================================

class GeometryDiffusion(nn.Module):
    """
    Diffusion model for molecular geometry generation.
    
    The "geometry" is a vector containing:
    - Bond lengths (normalized)
    - Bond angles (normalized to [0, 1])
    - Torsion angles (sin/cos representation)
    
    Uses a denoiser network (the JEPA) to predict noise at each step.
    """
    
    def __init__(self, geometry_dim=64, num_timesteps=1000, schedule='cosine'):
        """
        Args:
            geometry_dim: Dimension of geometry vector per atom
            num_timesteps: Number of diffusion timesteps
            schedule: 'cosine' or 'linear' noise schedule
        """
        super().__init__()
        self.geometry_dim = geometry_dim
        self.num_timesteps = num_timesteps
        
        # Build noise schedule
        if schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            betas = linear_beta_schedule(num_timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers (moved to device with model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Precompute values for q(x_t | x_0) and posterior
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
    
    def _extract(self, a, t, x_shape):
        """Extract coefficients at timesteps t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def forward_diffusion(self, x_0, t):
        """
        Forward diffusion: q(x_t | x_0)
        
        Add noise to clean geometry according to noise schedule.
        
        Args:
            x_0: Clean geometry, shape (batch, num_atoms, geometry_dim) or (batch, geometry_dim)
            t: Timesteps, shape (batch,)
        
        Returns:
            x_t: Noisy geometry at timestep t
            noise: The noise that was added (for training)
        """
        noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def predict_x0_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise.
        
        Args:
            x_t: Noisy geometry at timestep t
            t: Timesteps
            noise: Predicted noise
        
        Returns:
            x_0: Predicted clean geometry
        """
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def q_posterior(self, x_0, x_t, t):
        """
        Compute posterior q(x_{t-1} | x_t, x_0).
        
        Returns mean and variance for the posterior distribution.
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(self, denoiser, x_t, t, atom_types=None, clip_denoised=True):
        """
        Compute mean and variance for p(x_{t-1} | x_t) using denoiser.
        
        Args:
            denoiser: Network that predicts noise (the JEPA)
            x_t: Noisy geometry at timestep t
            t: Timesteps
            atom_types: Optional atom type conditioning
            clip_denoised: Whether to clip predicted x_0
        
        Returns:
            model_mean, posterior_variance, posterior_log_variance, pred_x0
        """
        # Predict noise using denoiser (JEPA)
        pred_noise = denoiser(x_t, t, atom_types)
        
        # Get predicted x_0
        pred_x0 = self.predict_x0_from_noise(x_t, t, pred_noise)
        
        if clip_denoised:
            # Clip to reasonable range for geometry
            pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)
        
        # Get posterior distribution
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(pred_x0, x_t, t)
        
        return model_mean, posterior_variance, posterior_log_variance, pred_x0
    
    @torch.no_grad()
    def p_sample(self, denoiser, x_t, t, atom_types=None, clip_denoised=True):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t).
        
        Single reverse diffusion step.
        """
        model_mean, _, model_log_variance, pred_x0 = self.p_mean_variance(
            denoiser, x_t, t, atom_types, clip_denoised
        )
        
        noise = torch.randn_like(x_t)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        x_t_minus_1 = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        
        return x_t_minus_1, pred_x0
    
    @torch.no_grad()
    def sample(self, denoiser, shape, atom_types=None, num_steps=None, 
               return_intermediates=False, progress=True):
        """
        Generate geometry via full reverse diffusion.
        
        Args:
            denoiser: Network that predicts noise (JEPA)
            shape: Shape of geometry to generate (batch, num_atoms, geometry_dim)
            atom_types: Optional atom type conditioning
            num_steps: Number of steps (default: num_timesteps)
            return_intermediates: Whether to return all intermediate steps
            progress: Whether to show progress bar
        
        Returns:
            x_0: Generated geometry
            intermediates: (optional) List of intermediate geometries
        """
        device = next(denoiser.parameters()).device
        
        if num_steps is None:
            num_steps = self.num_timesteps
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device)
        
        intermediates = [x_t] if return_intermediates else None
        
        # Reverse diffusion
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        iterator = timesteps
        if progress:
            from tqdm import tqdm
            iterator = tqdm(timesteps, desc='Sampling')
        
        for t in iterator:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x_t, pred_x0 = self.p_sample(denoiser, x_t, t_batch, atom_types)
            
            if return_intermediates:
                intermediates.append(x_t)
        
        if return_intermediates:
            return x_t, intermediates
        return x_t
    
    @torch.no_grad()
    def ddim_sample(self, denoiser, shape, atom_types=None, num_steps=50, eta=0.0):
        """
        DDIM sampling for faster generation.
        
        DDIM allows using fewer steps (e.g., 50) than training timesteps (e.g., 1000)
        while maintaining quality.
        
        Args:
            denoiser: Network that predicts noise
            shape: Shape of geometry to generate
            atom_types: Optional conditioning
            num_steps: Number of sampling steps (can be << num_timesteps)
            eta: Noise level (0 = deterministic, 1 = DDPM)
        
        Returns:
            x_0: Generated geometry
        """
        device = next(denoiser.parameters()).device
        
        # Create sampling timesteps (evenly spaced)
        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)
        
        # Start from noise
        x_t = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            pred_noise = denoiser(x_t, t_batch, atom_types)
            
            # Get alpha values
            alpha_t = self._extract(self.alphas_cumprod, t_batch, x_t.shape)
            
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                t_prev_batch = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
                alpha_t_prev = self._extract(self.alphas_cumprod, t_prev_batch, x_t.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)
            
            # Predict x_0
            pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)
            
            # Direction pointing to x_t
            sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            direction = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * pred_noise
            
            # Sample
            noise = torch.randn_like(x_t) if i + 1 < len(timesteps) else torch.zeros_like(x_t)
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + direction + sigma * noise
        
        return x_t


# ============================================================================
# GEOMETRY REPRESENTATION
# ============================================================================

class GeometryRepresentation:
    """
    Utilities for converting between molecular geometry representations.
    
    Geometry vector format (per molecule):
    - Bond lengths: N_bonds values (normalized)
    - Bond angles: N_angles values (in [0, 1], representing [0, π])
    - Torsion angles: N_torsions * 2 values (sin, cos representation)
    """
    
    def __init__(self, max_bonds=50, max_angles=100, max_torsions=50):
        self.max_bonds = max_bonds
        self.max_angles = max_angles
        self.max_torsions = max_torsions
        
        # Geometry vector dimension
        self.dim = max_bonds + max_angles + max_torsions * 2
    
    def encode(self, bond_lengths, angles, torsions, bond_length_mean=1.5, bond_length_std=0.3):
        """
        Encode geometry into fixed-size vector.
        
        Args:
            bond_lengths: Tensor of bond lengths
            angles: Tensor of angles in radians
            torsions: Tensor of torsion sin/cos values, shape (N, 2)
        
        Returns:
            geometry: Tensor of shape (dim,)
        """
        device = bond_lengths.device
        geometry = torch.zeros(self.dim, device=device)
        
        # Normalize and pad bond lengths
        n_bonds = min(len(bond_lengths), self.max_bonds)
        normalized_bonds = (bond_lengths[:n_bonds] - bond_length_mean) / bond_length_std
        geometry[:n_bonds] = normalized_bonds
        
        # Normalize and pad angles (to [0, 1])
        n_angles = min(len(angles), self.max_angles)
        normalized_angles = angles[:n_angles] / math.pi
        offset = self.max_bonds
        geometry[offset:offset + n_angles] = normalized_angles
        
        # Pad torsions (already sin/cos)
        n_torsions = min(len(torsions), self.max_torsions)
        offset = self.max_bonds + self.max_angles
        geometry[offset:offset + n_torsions * 2] = torsions[:n_torsions].flatten()
        
        return geometry
    
    def decode(self, geometry, bond_length_mean=1.5, bond_length_std=0.3):
        """
        Decode geometry vector back to bond lengths, angles, torsions.
        
        Args:
            geometry: Tensor of shape (dim,)
        
        Returns:
            bond_lengths, angles, torsions
        """
        # Extract bond lengths
        bond_lengths_norm = geometry[:self.max_bonds]
        bond_lengths = bond_lengths_norm * bond_length_std + bond_length_mean
        
        # Extract angles
        offset = self.max_bonds
        angles_norm = geometry[offset:offset + self.max_angles]
        angles = angles_norm * math.pi
        
        # Extract torsions
        offset = self.max_bonds + self.max_angles
        torsions_flat = geometry[offset:]
        torsions = torsions_flat.reshape(-1, 2)
        
        return bond_lengths, angles, torsions


if __name__ == '__main__':
    # Quick test
    print("Testing GeometryDiffusion...")
    
    diffusion = GeometryDiffusion(geometry_dim=64, num_timesteps=100, schedule='cosine')
    print(f"✓ Created diffusion model with {diffusion.num_timesteps} timesteps")
    
    # Test forward diffusion
    batch_size = 4
    num_atoms = 10
    clean_geometry = torch.randn(batch_size, num_atoms, 64)
    t = torch.randint(0, 100, (batch_size,))
    
    noisy, noise = diffusion.forward_diffusion(clean_geometry, t)
    print(f"✓ Forward diffusion: {clean_geometry.shape} → {noisy.shape}")
    
    # Test noise schedule
    print(f"✓ Beta range: [{diffusion.betas.min():.6f}, {diffusion.betas.max():.6f}]")
    print(f"✓ Alpha cumprod range: [{diffusion.alphas_cumprod.min():.6f}, {diffusion.alphas_cumprod.max():.6f}]")
    
    # Test timestep embedding
    emb = timestep_embedding(t, 128)
    print(f"✓ Timestep embedding: {t.shape} → {emb.shape}")
    
    print("\n✓ All tests passed!")
