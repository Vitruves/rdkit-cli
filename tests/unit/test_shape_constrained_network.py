"""Tests for new subcommands: shape similarity, constrained embedding,
reaction mapping, scaffold network, charges/crippen, BRICS build."""

import json
import tempfile
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit_cli.io.readers import MoleculeRecord


def _make_record(smiles: str, name: str = "") -> MoleculeRecord:
    mol = Chem.MolFromSmiles(smiles)
    return MoleculeRecord(mol=mol, smiles=smiles, name=name)


def _make_3d_sdf(smiles: str, path: Path):
    """Write a single molecule with 3D coords to an SDF file."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


# ---------------------------------------------------------------------------
# 1. Shape similarity
# ---------------------------------------------------------------------------


class TestShapeSimilaritySearcher:

    def test_shape_search_self(self, tmp_path):
        from rdkit_cli.core.similarity import ShapeSimilaritySearcher

        ref_path = tmp_path / "ref.sdf"
        _make_3d_sdf("c1ccccc1", ref_path)

        searcher = ShapeSimilaritySearcher(
            reference_file=str(ref_path),
            threshold=0.0,
            metric="tanimoto",
        )
        result = searcher.search(_make_record("c1ccccc1", "benzene"))

        assert result is not None
        assert "shape_similarity" in result
        assert result["shape_similarity"] > 0

    def test_shape_search_protrude(self, tmp_path):
        from rdkit_cli.core.similarity import ShapeSimilaritySearcher

        ref_path = tmp_path / "ref.sdf"
        _make_3d_sdf("c1ccccc1", ref_path)

        searcher = ShapeSimilaritySearcher(
            reference_file=str(ref_path),
            threshold=0.0,
            metric="protrude",
        )
        result = searcher.search(_make_record("Cc1ccccc1", "toluene"))
        assert result is not None
        assert isinstance(result["shape_similarity"], float)

    def test_shape_search_tversky(self, tmp_path):
        from rdkit_cli.core.similarity import ShapeSimilaritySearcher

        ref_path = tmp_path / "ref.sdf"
        _make_3d_sdf("c1ccccc1", ref_path)

        searcher = ShapeSimilaritySearcher(
            reference_file=str(ref_path),
            threshold=0.0,
            metric="tversky",
            tversky_alpha=0.8,
            tversky_beta=0.2,
        )
        result = searcher.search(_make_record("c1ccccc1", "benzene"))
        assert result is not None

    def test_shape_threshold_filters(self, tmp_path):
        from rdkit_cli.core.similarity import ShapeSimilaritySearcher

        ref_path = tmp_path / "ref.sdf"
        _make_3d_sdf("c1ccccc1", ref_path)

        searcher = ShapeSimilaritySearcher(
            reference_file=str(ref_path),
            threshold=0.99,
        )
        # Methane is very different from benzene in shape
        result = searcher.search(_make_record("C", "methane"))
        # Likely filtered out (or None due to shape mismatch)
        # either filtered or very low similarity
        if result is not None:
            assert result["shape_similarity"] >= 0.99

    def test_shape_invalid_reference(self, tmp_path):
        from rdkit_cli.core.similarity import ShapeSimilaritySearcher

        bad_path = tmp_path / "empty.sdf"
        bad_path.write_text("")

        with pytest.raises(ValueError, match="Cannot load"):
            ShapeSimilaritySearcher(reference_file=str(bad_path))

    def test_shape_none_molecule(self, tmp_path):
        from rdkit_cli.core.similarity import ShapeSimilaritySearcher

        ref_path = tmp_path / "ref.sdf"
        _make_3d_sdf("c1ccccc1", ref_path)

        searcher = ShapeSimilaritySearcher(
            reference_file=str(ref_path), threshold=0.0,
        )
        record = MoleculeRecord(mol=None, smiles="invalid")
        assert searcher.search(record) is None


# ---------------------------------------------------------------------------
# 2. Constrained embedding
# ---------------------------------------------------------------------------


class TestConstrainedEmbedder:

    def test_constrained_embed(self, tmp_path):
        from rdkit_cli.core.conformers import ConstrainedEmbedder

        ref_path = tmp_path / "ref.sdf"
        _make_3d_sdf("c1ccccc1", ref_path)

        embedder = ConstrainedEmbedder(
            reference_file=str(ref_path),
            force_field="mmff",
        )
        # Phenol contains benzene as substructure
        result = embedder.embed(_make_record("c1ccc(O)cc1", "phenol"))

        assert result is not None
        assert "mol" in result
        assert result["mol"].GetNumConformers() > 0

    def test_constrained_embed_none_mol(self, tmp_path):
        from rdkit_cli.core.conformers import ConstrainedEmbedder

        ref_path = tmp_path / "ref.sdf"
        _make_3d_sdf("c1ccccc1", ref_path)

        embedder = ConstrainedEmbedder(reference_file=str(ref_path))
        record = MoleculeRecord(mol=None, smiles="invalid")
        assert embedder.embed(record) is None

    def test_constrained_embed_invalid_ref(self, tmp_path):
        from rdkit_cli.core.conformers import ConstrainedEmbedder

        bad_path = tmp_path / "empty.sdf"
        bad_path.write_text("")

        with pytest.raises(ValueError, match="Cannot load"):
            ConstrainedEmbedder(reference_file=str(bad_path))


# ---------------------------------------------------------------------------
# 3. Reaction atom mapping
# ---------------------------------------------------------------------------


class TestReactionAtomMapping:

    def test_mapped_reaction(self):
        from rdkit_cli.core.reactions import get_atom_mapping

        # Simple mapped reaction: esterification
        smarts = "[C:1](=[O:2])[OH].[O:3][C:4]>>[C:1](=[O:2])[O:3][C:4]"
        result = get_atom_mapping(smarts)

        assert result["has_mapping"] is True
        assert result["num_reactants"] == 2
        assert result["num_products"] == 1
        assert len(result["reactant_maps"]) == 2
        assert len(result["product_maps"]) == 1
        # Check atom map numbers are present
        assert 1 in result["reactant_maps"][0]
        assert 2 in result["reactant_maps"][0]

    def test_unmapped_reaction(self):
        from rdkit_cli.core.reactions import get_atom_mapping

        smarts = "[C](=O)O>>[C](=O)N"
        result = get_atom_mapping(smarts)

        assert result["has_mapping"] is False
        assert result["num_reactants"] == 1
        assert result["num_products"] == 1

    def test_invalid_smarts(self):
        from rdkit_cli.core.reactions import get_atom_mapping

        with pytest.raises(ValueError):
            get_atom_mapping("not>>valid>>smarts>>here")

    def test_mapping_structure(self):
        from rdkit_cli.core.reactions import get_atom_mapping

        smarts = "[C:1][OH:2]>>[C:1]=[O:2]"
        result = get_atom_mapping(smarts)

        rmap = result["reactant_maps"][0]
        pmap = result["product_maps"][0]

        # Map number 1 should be C in both
        assert rmap[1]["symbol"] == "C"
        assert pmap[1]["symbol"] == "C"
        # idx should be integers
        assert isinstance(rmap[1]["idx"], int)


# ---------------------------------------------------------------------------
# 4. Scaffold network
# ---------------------------------------------------------------------------


class TestScaffoldNetwork:

    def test_build_network(self):
        from rdkit_cli.core.scaffold import build_scaffold_network

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),       # benzene
            Chem.MolFromSmiles("Cc1ccccc1"),       # toluene
            Chem.MolFromSmiles("c1ccc2ccccc2c1"),  # naphthalene
        ]

        network = build_scaffold_network(mols)

        assert "nodes" in network
        assert "edges" in network
        assert "counts" in network
        assert len(network["nodes"]) > 0
        assert network["num_molecules"] == 3

    def test_network_contains_benzene(self):
        from rdkit_cli.core.scaffold import build_scaffold_network

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("Cc1ccccc1"),
        ]

        network = build_scaffold_network(mols)
        # Benzene should be a node
        assert "c1ccccc1" in network["nodes"]

    def test_network_empty_input(self):
        from rdkit_cli.core.scaffold import build_scaffold_network

        network = build_scaffold_network([])
        assert network["nodes"] == []
        assert network["edges"] == []

    def test_network_counts(self):
        from rdkit_cli.core.scaffold import build_scaffold_network

        mols = [
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("Cc1ccccc1"),
            Chem.MolFromSmiles("CCc1ccccc1"),
        ]

        network = build_scaffold_network(mols)
        # Benzene scaffold should match all 3 molecules
        benzene_idx = network["nodes"].index("c1ccccc1")
        assert network["counts"][benzene_idx] == 3


# ---------------------------------------------------------------------------
# 5. Props charges and crippen (CLI-level integration)
# ---------------------------------------------------------------------------


class TestPropsCharges:

    def test_charges_cli(self, tmp_path):
        from rdkit_cli.commands.props import run_charges
        import argparse

        csv_path = tmp_path / "mols.csv"
        csv_path.write_text("smiles,name\nCCO,ethanol\nc1ccccc1,benzene\n")

        out_path = tmp_path / "charges.csv"
        args = argparse.Namespace(
            input=str(csv_path),
            output=str(out_path),
            smiles_column="smiles",
            no_header=False,
        )
        ret = run_charges(args)
        assert ret == 0
        assert out_path.exists()

        import pandas as pd
        df = pd.read_csv(out_path)
        assert "gasteiger_charges" in df.columns
        assert "min_charge" in df.columns
        assert "max_charge" in df.columns
        assert len(df) == 2


class TestPropsCrippen:

    def test_crippen_cli(self, tmp_path):
        from rdkit_cli.commands.props import run_crippen
        import argparse

        csv_path = tmp_path / "mols.csv"
        csv_path.write_text("smiles,name\nCCO,ethanol\nc1ccccc1,benzene\n")

        out_path = tmp_path / "crippen.csv"
        args = argparse.Namespace(
            input=str(csv_path),
            output=str(out_path),
            smiles_column="smiles",
            no_header=False,
        )
        ret = run_crippen(args)
        assert ret == 0
        assert out_path.exists()

        import pandas as pd
        df = pd.read_csv(out_path)
        assert "logp_contribs" in df.columns
        assert "mr_contribs" in df.columns
        assert "crippen_logp" in df.columns
        assert "crippen_mr" in df.columns
        assert len(df) == 2


# ---------------------------------------------------------------------------
# 6. BRICS build
# ---------------------------------------------------------------------------


class TestBRICSBuild:

    def test_brics_build_basic(self):
        from rdkit_cli.core.fragment import brics_build

        # Simple BRICS fragments
        fragments = [
            "[3*]O",
            "[4*]c1ccccc1",
            "[3*]N",
        ]
        products = brics_build(fragments, max_molecules=10)

        assert len(products) > 0
        # All should be valid SMILES
        for smi in products:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Invalid product: {smi}"

    def test_brics_build_max_limit(self):
        from rdkit_cli.core.fragment import brics_build

        fragments = [
            "[3*]O",
            "[4*]c1ccccc1",
            "[3*]N",
            "[16*]c1ccccc1",
        ]
        products = brics_build(fragments, max_molecules=5)

        assert len(products) <= 5

    def test_brics_build_empty_input(self):
        from rdkit_cli.core.fragment import brics_build

        products = brics_build([])
        assert products == []

    def test_brics_build_invalid_fragments(self):
        from rdkit_cli.core.fragment import brics_build

        products = brics_build(["not_a_smiles", "also_invalid"])
        assert products == []
