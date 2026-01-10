"""Unit tests for rmsd module."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


@pytest.fixture
def mol_with_conformers():
    """Create a molecule with multiple conformers."""
    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=42)
    return mol


class TestCalculateRMSD:
    """Test calculate_rmsd function."""

    def test_rmsd_same_molecule(self, mol_with_conformers):
        """Test RMSD between same molecule conformers."""
        from rdkit_cli.core.rmsd import calculate_rmsd

        # Create two copies
        mol1 = Chem.Mol(mol_with_conformers)
        mol2 = Chem.Mol(mol_with_conformers)

        rmsd = calculate_rmsd(mol1, mol2, align=True, symmetry=True)

        assert rmsd is not None
        assert rmsd >= 0

    def test_rmsd_no_align(self, mol_with_conformers):
        """Test RMSD without alignment."""
        from rdkit_cli.core.rmsd import calculate_rmsd

        mol1 = Chem.Mol(mol_with_conformers)
        mol2 = Chem.Mol(mol_with_conformers)

        rmsd = calculate_rmsd(mol1, mol2, align=False)

        assert rmsd is not None

    def test_rmsd_none_molecule(self):
        """Test RMSD with None molecule."""
        from rdkit_cli.core.rmsd import calculate_rmsd

        rmsd = calculate_rmsd(None, None)
        assert rmsd is None

    def test_rmsd_no_conformer(self):
        """Test RMSD with molecule without conformers."""
        from rdkit_cli.core.rmsd import calculate_rmsd

        mol1 = Chem.MolFromSmiles("CCC")
        mol2 = Chem.MolFromSmiles("CCC")

        rmsd = calculate_rmsd(mol1, mol2)
        assert rmsd is None


class TestConformerRMSDMatrix:
    """Test conformer RMSD matrix calculation."""

    def test_matrix_calculation(self, mol_with_conformers):
        """Test calculating conformer RMSD matrix."""
        from rdkit_cli.core.rmsd import calculate_conformer_rmsd_matrix

        matrix = calculate_conformer_rmsd_matrix(mol_with_conformers)

        n_conf = mol_with_conformers.GetNumConformers()
        assert len(matrix) == n_conf
        assert len(matrix[0]) == n_conf

        # Diagonal should be 0
        for i in range(n_conf):
            assert matrix[i][i] == 0.0

        # Matrix should be symmetric
        for i in range(n_conf):
            for j in range(i + 1, n_conf):
                assert abs(matrix[i][j] - matrix[j][i]) < 0.001

    def test_matrix_no_conformers(self):
        """Test with molecule without conformers."""
        from rdkit_cli.core.rmsd import calculate_conformer_rmsd_matrix

        mol = Chem.MolFromSmiles("CCC")
        matrix = calculate_conformer_rmsd_matrix(mol)

        assert matrix == []


class TestClusterConformers:
    """Test conformer clustering by RMSD."""

    def test_cluster_conformers(self, mol_with_conformers):
        """Test clustering conformers."""
        from rdkit_cli.core.rmsd import cluster_conformers_by_rmsd

        clusters = cluster_conformers_by_rmsd(mol_with_conformers, threshold=2.0)

        assert len(clusters) >= 1
        # All conformers should be assigned
        all_indices = []
        for cluster in clusters:
            all_indices.extend(cluster)
        assert len(all_indices) == mol_with_conformers.GetNumConformers()


class TestRMSDCalculator:
    """Test RMSDCalculator class."""

    def test_calculator(self, mol_with_conformers):
        """Test RMSDCalculator."""
        from rdkit_cli.core.rmsd import RMSDCalculator

        # Use first conformer as reference
        ref = Chem.Mol(mol_with_conformers)

        calculator = RMSDCalculator(
            reference_mol=ref,
            align=True,
            symmetry=True,
        )

        # Calculate RMSD to same molecule
        probe = Chem.Mol(mol_with_conformers)
        rmsd = calculator.calculate(probe)

        assert rmsd is not None
        assert rmsd >= 0

    def test_calculator_no_reference(self):
        """Test calculator with None reference."""
        from rdkit_cli.core.rmsd import RMSDCalculator

        with pytest.raises(ValueError):
            RMSDCalculator(reference_mol=None)


class TestConformerRMSDAnalyzer:
    """Test ConformerRMSDAnalyzer class."""

    def test_analyzer(self, mol_with_conformers):
        """Test analyzing conformer RMSDs."""
        from rdkit_cli.core.rmsd import ConformerRMSDAnalyzer

        analyzer = ConformerRMSDAnalyzer(symmetry=True)
        result = analyzer.analyze(mol_with_conformers)

        assert result is not None
        assert "num_conformers" in result
        assert "min_rmsd" in result
        assert "max_rmsd" in result
        assert "mean_rmsd" in result
        assert result["num_conformers"] == mol_with_conformers.GetNumConformers()

    def test_analyzer_single_conformer(self):
        """Test with single conformer."""
        from rdkit_cli.core.rmsd import ConformerRMSDAnalyzer

        mol = Chem.MolFromSmiles("CCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        analyzer = ConformerRMSDAnalyzer()
        result = analyzer.analyze(mol)

        assert result is not None
        assert result["num_conformers"] == 1
