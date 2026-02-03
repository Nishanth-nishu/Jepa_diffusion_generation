"""
convert_sdf_to_pdb.py - Convert SDF file to individual PDB files

Usage:
    python convert_sdf_to_pdb.py input.sdf output_dir/
"""

import sys
from pathlib import Path
from rdkit import Chem

def convert_sdf_to_pdb(sdf_path, output_dir):
    """Convert all molecules in SDF to individual PDB files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all molecules from SDF
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    
    count = 0
    for i, mol in enumerate(suppl):
        if mol is None:
            print(f"Skipping invalid molecule {i}")
            continue
        
        # Get molecule name or use index
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
        
        # Write to PDB
        pdb_path = output_dir / f"{name}.pdb"
        Chem.MolToPDBFile(mol, str(pdb_path))
        print(f"Saved: {pdb_path}")
        count += 1
    
    print(f"\nConverted {count} molecules to PDB format")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_sdf_to_pdb.py input.sdf output_dir/")
        sys.exit(1)
    
    sdf_path = sys.argv[1]
    output_dir = sys.argv[2]
    convert_sdf_to_pdb(sdf_path, output_dir)
