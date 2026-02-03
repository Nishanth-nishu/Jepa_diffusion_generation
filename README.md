# JEPA Diffusion Molecule Generator

A 3D molecule generator combining **Joint-Embedding Predictive Architecture (JEPA)** with **denoising diffusion** for generating novel, chemically valid molecular structures.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  JEPA (Pre-training)                                            │
│  Atom Types + Bond Graph → GraphTransformer → Geometry Heads    │
│  (Predicts bond lengths, angles, torsions)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Diffusion Model (Generation)                                   │
│  Noise → Iterative Denoising → Molecular Geometry → 3D Coords   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Validation (RDKit)                                             │
│  3D Coordinates → Bond Inference → Chemistry Validation → SDF  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/Nishanth-nishu/Jepa_diffusion_generation.git
cd Jepa_diffusion_generation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### PyTorch with CUDA (optional)
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Train JEPA Model
Pre-train the geometry prediction model on QM9 dataset:

```bash
python train_geo_com_op.py \
    --data_path data/qm9_100k.jsonl \
    --epochs 100 \
    --batch_size 32 \
    --encoder_type transformer
```

**Key outputs:**
- `best_pure_jepa_transformer.pt` - Best model checkpoint

### 2. Train Diffusion Model
Train the diffusion model using pre-trained JEPA:

```bash
python train_diffusion.py \
    --data_path data/qm9_100k.jsonl \
    --epochs 200 \
    --batch_size 64 \
    --use_pretrained \
    --pretrained_jepa best_pure_jepa_transformer.pt
```

### 3. Generate Molecules
Generate novel molecules:

```bash
python generate_molecules.py \
    --checkpoint checkpoints/diffusion_best.pt \
    --num_molecules 100 \
    --output generated_molecules.sdf
```

### 4. Convert to PDB (optional)
```bash
python convert_sdf_to_pdb.py generated_molecules.sdf pdb_molecules/
```

## Project Structure

```
├── train_geo_com_op.py      # JEPA training script
├── train_diffusion.py       # Diffusion training script
├── generate_molecules.py    # Molecule generation pipeline
├── geometry_diffusion.py    # Core diffusion model
├── diffusion_jepa.py        # JEPA wrapper for diffusion
├── pure_geometry_encoders.py # Graph transformer encoder
├── dataset_optimized.py     # Optimized QM9 data loader
├── geometry.py              # Geometry extraction utilities
├── bonds.py                 # Bond extraction from SMILES
├── bond_types.py            # Bond type classification
├── convert_sdf_to_pdb.py    # SDF to PDB converter
├── models/                  # EGNN model components
├── data/                    # Dataset directory
├── checkpoints/             # Saved model checkpoints
└── requirements.txt         # Python dependencies
```

## Key Features

- **Pure Graph Encoder**: Uses GraphTransformer (no coordinates in encoder)
- **Multi-head Geometry Prediction**: Bond lengths, angles, torsions, repulsion
- **Contrastive Learning**: Energy-based discrimination of valid vs corrupted graphs
- **DDIM Sampling**: Fast generation with configurable steps
- **RDKit Validation**: Chemistry-aware validation and property calculation

## Model Details

### JEPA Pre-training
- **Encoder**: 4-layer GraphTransformer with 8 attention heads
- **Heads**: BondLength, Angle, Torsion, Repulsion, Energy, Valence
- **Loss**: Multi-task geometry prediction + contrastive energy

### Diffusion Model
- **Timesteps**: 1000 (training), 50 (DDIM sampling)
- **Noise Schedule**: Cosine beta schedule
- **Denoiser**: JEPA model wrapped with time embedding

## Requirements

- Python 3.8+
- PyTorch 2.0+
- RDKit 2023.3+
- NumPy, Pandas, tqdm, matplotlib

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgements

- EGNN implementation based on [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844)
- QM9 dataset from [Quantum Machine](http://quantum-machine.org/datasets/)
