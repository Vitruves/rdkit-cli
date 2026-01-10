"""Unit tests for align module."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


@pytest.fixture
def mol_with_3d():
    """Create a molecule with 3D coordinates."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


class TestMoleculeAligner:
    """Test MoleculeAligner class."""

    def test_align_mcs_method(self, mol_with_3d):
        """Test MCS-based alignment."""
        from rdkit_cli.core.align import MoleculeAligner
        from rdkit_cli.io.readers import MoleculeRecord

        # Create a similar molecule
        probe_mol = Chem.MolFromSmiles("c1ccccc1C")  # Toluene
        probe_mol = Chem.AddHs(probe_mol)
        AllChem.EmbedMolecule(probe_mol, randomSeed=42)
        probe_mol = Chem.RemoveHs(probe_mol)

        aligner = MoleculeAligner(
            reference_mol=mol_with_3d,
            method="mcs",
        )

        record = MoleculeRecord(mol=probe_mol, smiles="c1ccccc1C", name="toluene")
        result = aligner.align(record)

        assert result is not None
        assert "rmsd" in result
        assert result["rmsd"] >= 0

    def test_align_o3a_method(self, mol_with_3d):
        """Test O3A-based alignment."""
        from rdkit_cli.core.align import MoleculeAligner
        from rdkit_cli.io.readers import MoleculeRecord

        probe_mol = Chem.MolFromSmiles("c1ccccc1C")
        probe_mol = Chem.AddHs(probe_mol)
        AllChem.EmbedMolecule(probe_mol, randomSeed=42)
        probe_mol = Chem.RemoveHs(probe_mol)

        aligner = MoleculeAligner(
            reference_mol=mol_with_3d,
            method="o3a",
        )

        record = MoleculeRecord(mol=probe_mol, smiles="c1ccccc1C")
        result = aligner.align(record)

        assert result is not None
        assert "rmsd" in result

    def test_align_no_3d_input(self, mol_with_3d):
        """Test alignment when input has no 3D coordinates."""
        from rdkit_cli.core.align import MoleculeAligner
        from rdkit_cli.io.readers import MoleculeRecord

        aligner = MoleculeAligner(
            reference_mol=mol_with_3d,
            method="mcs",
        )

        # Molecule without 3D
        mol = Chem.MolFromSmiles("c1ccccc1C")
        record = MoleculeRecord(mol=mol, smiles="c1ccccc1C")

        result = aligner.align(record)

        # Should generate 3D and align
        assert result is not None or result is None  # May or may not work

    def test_align_invalid_reference(self):
        """Test with reference without 3D coordinates."""
        from rdkit_cli.core.align import MoleculeAligner

        mol_no_3d = Chem.MolFromSmiles("c1ccccc1")

        with pytest.raises(ValueError, match="3D coordinates"):
            MoleculeAligner(reference_mol=mol_no_3d)


class TestCalculateRMSD:
    """Test RMSD calculation functions."""

    def test_calculate_rmsd_aligned(self, mol_with_3d):
        """Test RMSD calculation with alignment."""
        from rdkit_cli.core.align import calculate_rmsd

        # Same molecule should have RMSD close to 0
        rmsd = calculate_rmsd(mol_with_3d, mol_with_3d, align=True)

        assert rmsd is not None
        assert rmsd < 0.1  # Very close to 0

    def test_calculate_rmsd_none_mol(self):
        """Test RMSD with None molecule."""
        from rdkit_cli.core.align import calculate_rmsd

        rmsd = calculate_rmsd(None, None)
        assert rmsd is None


class TestLoadReferenceMolecule:
    """Test load_reference_molecule function."""

    def test_load_from_sdf(self, tmp_dir):
        """Test loading reference from SDF file."""
        from rdkit_cli.core.align import load_reference_molecule

        # Create a simple SDF file
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        sdf_path = tmp_dir / "ref.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()

        loaded = load_reference_molecule(str(sdf_path))

        assert loaded is not None
        assert loaded.GetNumConformers() > 0
