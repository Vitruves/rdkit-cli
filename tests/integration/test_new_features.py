"""Integration tests for all new features (Phase 1/2/3).
Tests full CLI invocation end-to-end."""

import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


def run_cli(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run rdkit-cli command and return result."""
    cmd = [sys.executable, "-m", "rdkit_cli"] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def mol_csv(tmp_path):
    """Standard molecule CSV."""
    p = tmp_path / "mols.csv"
    p.write_text(
        "smiles,name\n"
        "c1ccccc1,benzene\n"
        "Cc1ccccc1,toluene\n"
        "CCO,ethanol\n"
        "CC(=O)O,acetic_acid\n"
        "c1ccc(O)cc1,phenol\n"
    )
    return p


@pytest.fixture
def chiral_csv(tmp_path):
    """CSV with chiral molecules."""
    p = tmp_path / "chiral.csv"
    p.write_text(
        "smiles,name\n"
        "C[C@H](O)F,chiral1\n"
        "CC(O)(F)Cl,potential\n"
        "CCO,achiral\n"
    )
    return p


@pytest.fixture
def ref_sdf(tmp_path):
    """Reference molecule SDF with 3D coords."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    p = tmp_path / "ref.sdf"
    w = Chem.SDWriter(str(p))
    w.write(mol)
    w.close()
    return p


@pytest.fixture
def out(tmp_path):
    return tmp_path / "output.csv"


# ===================== Phase 1: Registry expansions ========================


class TestPhase1Fingerprints:
    """New FP types via CLI."""

    @pytest.mark.parametrize("fp_type", ["avalon", "mhfp", "pharmacophore"])
    def test_compute_new_fp(self, mol_csv, tmp_path, fp_type):
        out = tmp_path / f"fp_{fp_type}.csv"
        result = run_cli([
            "fingerprints", "compute",
            "-i", str(mol_csv), "-o", str(out),
            "-t", fp_type, "-q",
        ])
        assert result.returncode == 0
        assert out.exists()
        content = out.read_text()
        assert "fingerprint" in content
        assert len(content.strip().split("\n")) >= 5

    def test_fingerprints_list_includes_new(self):
        result = run_cli(["fingerprints", "list"])
        assert result.returncode == 0
        for name in ["avalon", "mhfp", "pharmacophore"]:
            assert name in result.stdout


class TestPhase1Descriptors:
    """MQN and 3D descriptors via CLI."""

    def test_mqn_descriptors(self, mol_csv, out):
        result = run_cli([
            "descriptors", "compute",
            "-i", str(mol_csv), "-o", str(out),
            "--mqn", "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "MQN1" in content
        assert "MQN42" in content

    def test_3d_descriptors(self, mol_csv, out):
        result = run_cli([
            "descriptors", "compute",
            "-i", str(mol_csv), "-o", str(out),
            "--3d", "--generate-conformers", "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "PMI1" in content
        assert "Asphericity" in content

    def test_list_mqn_category(self):
        result = run_cli(["descriptors", "list", "--category", "mqn"])
        assert result.returncode == 0
        assert "MQN1" in result.stdout

    def test_list_3d_category(self):
        result = run_cli(["descriptors", "list", "--category", "3d"])
        assert result.returncode == 0
        assert "PMI1" in result.stdout


class TestPhase1Similarity:
    """New similarity metrics via CLI."""

    @pytest.mark.parametrize("metric", [
        "braunblanquet", "kulczynski", "tversky",
    ])
    def test_search_new_metric(self, mol_csv, tmp_path, metric):
        out = tmp_path / f"sim_{metric}.csv"
        args = [
            "similarity", "search",
            "-i", str(mol_csv), "-o", str(out),
            "--query", "c1ccccc1", "-m", metric,
            "-t", "0.01", "-q",
        ]
        if metric == "tversky":
            args += ["--tversky-alpha", "0.8", "--tversky-beta", "0.2"]
        result = run_cli(args)
        assert result.returncode == 0
        assert out.exists()


class TestPhase1Filters:
    """Expanded alert catalogs via CLI."""

    @pytest.mark.parametrize("catalog", ["brenk", "nih", "zinc", "all"])
    def test_filter_catalog(self, mol_csv, tmp_path, catalog):
        out = tmp_path / f"filter_{catalog}.csv"
        result = run_cli([
            "filter", "pains",
            "-i", str(mol_csv), "-o", str(out),
            "--catalog", catalog, "-q",
        ])
        assert result.returncode == 0
        assert out.exists()

    def test_filter_alerts_alias(self, mol_csv, tmp_path):
        out = tmp_path / "alerts.csv"
        result = run_cli([
            "filter", "alerts",
            "-i", str(mol_csv), "-o", str(out),
            "--catalog", "brenk", "-q",
        ])
        assert result.returncode == 0


# ===================== Phase 2: New subcommands ============================


class TestPhase2ShapeSimilarity:

    def test_shape_search(self, mol_csv, ref_sdf, tmp_path):
        out = tmp_path / "shape.csv"
        result = run_cli([
            "similarity", "shape",
            "-i", str(mol_csv), "-o", str(out),
            "-r", str(ref_sdf),
            "-t", "0.01", "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "shape_similarity" in content


class TestPhase2ConstrainedEmbed:

    def test_constrained(self, mol_csv, ref_sdf, tmp_path):
        out = tmp_path / "constrained.sdf"
        result = run_cli([
            "conformers", "constrained",
            "-i", str(mol_csv), "-o", str(out),
            "-r", str(ref_sdf), "-q",
        ])
        assert result.returncode == 0


class TestPhase2ReactionsMap:

    def test_map_text(self):
        result = run_cli([
            "reactions", "map",
            "-s", "[C:1](=[O:2])[OH].[O:3][C:4]>>[C:1](=[O:2])[O:3][C:4]",
        ])
        assert result.returncode == 0
        assert "Has mapping: True" in result.stdout

    def test_map_json(self):
        result = run_cli([
            "reactions", "map",
            "-s", "[C:1]=[O:2]>>[C:1][O:2]",
            "-f", "json",
        ])
        assert result.returncode == 0
        import json
        data = json.loads(result.stdout)
        assert data["has_mapping"] is True


class TestPhase2ScaffoldNetwork:

    def test_network_csv(self, mol_csv, tmp_path):
        out = tmp_path / "network.csv"
        result = run_cli([
            "scaffold", "network",
            "-i", str(mol_csv), "-o", str(out), "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "scaffold" in content
        assert "count" in content

    def test_network_json(self, mol_csv, tmp_path):
        out = tmp_path / "network.json"
        result = run_cli([
            "scaffold", "network",
            "-i", str(mol_csv), "-o", str(out),
            "-f", "json", "-q",
        ])
        assert result.returncode == 0
        import json
        data = json.loads(out.read_text())
        assert "nodes" in data
        assert "edges" in data


class TestPhase2PropsCharges:

    def test_charges(self, mol_csv, out):
        result = run_cli([
            "props", "charges",
            "-i", str(mol_csv), "-o", str(out),
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "gasteiger_charges" in content

    def test_crippen(self, mol_csv, out):
        result = run_cli([
            "props", "crippen",
            "-i", str(mol_csv), "-o", str(out),
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "crippen_logp" in content


class TestPhase2BRICSBuild:

    def test_brics_build(self, tmp_path):
        # First fragment molecules
        mol_csv = tmp_path / "mols.csv"
        mol_csv.write_text(
            "smiles,name\nc1ccccc1,benzene\nCCO,ethanol\n"
        )
        frags = tmp_path / "frags.csv"
        run_cli([
            "fragment", "brics",
            "-i", str(mol_csv), "-o", str(frags), "-q",
        ])

        # Then build from fragments
        out = tmp_path / "built.csv"
        result = run_cli([
            "fragment", "brics-build",
            "-i", str(frags), "-o", str(out),
            "--max-molecules", "5", "-q",
        ])
        assert result.returncode == 0


# ===================== Phase 3: New commands ===============================


class TestPhase3Stereo:

    def test_assign(self, chiral_csv, out):
        result = run_cli([
            "stereo", "assign",
            "-i", str(chiral_csv), "-o", str(out), "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "cip_labels" in content
        assert "num_stereocenters" in content

    def test_perceive(self, chiral_csv, out):
        result = run_cli([
            "stereo", "perceive",
            "-i", str(chiral_csv), "-o", str(out), "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "num_potential" in content

    def test_enhanced(self, chiral_csv, out):
        result = run_cli([
            "stereo", "enhanced",
            "-i", str(chiral_csv), "-o", str(out), "-q",
        ])
        assert result.returncode == 0

    def test_clean(self, chiral_csv, out):
        result = run_cli([
            "stereo", "clean",
            "-i", str(chiral_csv), "-o", str(out), "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "original_smiles" in content


class TestPhase3Energy:

    def test_compute(self, mol_csv, out):
        result = run_cli([
            "energy", "compute",
            "-i", str(mol_csv), "-o", str(out), "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "energy" in content

    def test_minimize(self, mol_csv, out):
        result = run_cli([
            "energy", "minimize",
            "-i", str(mol_csv), "-o", str(out), "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "energy_before" in content
        assert "energy_after" in content


class TestPhase3Pharmacophore:

    def test_perceive(self, mol_csv, out):
        result = run_cli([
            "pharmacophore", "perceive",
            "-i", str(mol_csv), "-o", str(out), "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "num_features" in content

    def test_search(self, mol_csv, out):
        result = run_cli([
            "pharmacophore", "search",
            "-i", str(mol_csv), "-o", str(out),
            "--query", "c1ccc(O)cc1",
            "-t", "0.01", "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "pharmacophore_similarity" in content


class TestPhase3DepictHighlight:

    def test_highlight_svg(self, tmp_path):
        out = tmp_path / "hl.svg"
        result = run_cli([
            "depict", "highlight",
            "c1ccc(O)cc1",
            "-s", "c1ccccc1",
            "-o", str(out),
        ])
        assert result.returncode == 0
        assert out.exists()
        assert "<svg" in out.read_text()


class TestPhase3ConformersTorsion:

    def test_torsion_scan(self, tmp_path):
        csv = tmp_path / "butane.csv"
        csv.write_text("smiles,name\nCCCC,butane\n")
        out = tmp_path / "torsion.csv"
        result = run_cli([
            "conformers", "torsion",
            "-i", str(csv), "-o", str(out),
            "--atoms", "0,1,2,3",
            "--step", "60", "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "min_angle" in content
        assert "barrier" in content


class TestPhase3ReactionsFingerprint:

    def test_reaction_fp(self, tmp_path):
        csv = tmp_path / "rxns.csv"
        csv.write_text(
            "reaction\n"
            "[C:1]=[O:2]>>[C:1][O:2]\n"
            "[N:1][C:2]>>[N:1]=[C:2]\n"
        )
        out = tmp_path / "rxn_fp.csv"
        result = run_cli([
            "reactions", "fingerprint",
            "-i", str(csv), "-o", str(out),
            "-t", "structural", "-q",
        ])
        assert result.returncode == 0
        content = out.read_text()
        assert "fingerprint" in content


# ===================== Cross-feature integration ===========================


class TestCrossFeatureIntegration:
    """Test that features work together in pipelines."""

    def test_descriptors_then_filter(self, mol_csv, tmp_path):
        """Compute descriptors, then filter by property."""
        desc = tmp_path / "desc.csv"
        run_cli([
            "descriptors", "compute",
            "-i", str(mol_csv), "-o", str(desc),
            "-d", "MolWt,MolLogP", "-q",
        ])
        filtered = tmp_path / "filtered.csv"
        result = run_cli([
            "filter", "property",
            "-i", str(mol_csv), "-o", str(filtered),
            "-r", "MolWt<200", "-q",
        ])
        assert result.returncode == 0

    def test_standardize_then_fingerprint(self, mol_csv, tmp_path):
        """Standardize molecules, then compute new FP types."""
        std = tmp_path / "std.csv"
        run_cli([
            "standardize",
            "-i", str(mol_csv), "-o", str(std),
            "--cleanup", "-q",
        ])
        fp_out = tmp_path / "avalon_fp.csv"
        result = run_cli([
            "fingerprints", "compute",
            "-i", str(std), "-o", str(fp_out),
            "-t", "avalon", "-q",
        ])
        assert result.returncode == 0
        content = fp_out.read_text()
        assert "fingerprint" in content

    def test_fragment_then_build(self, mol_csv, tmp_path):
        """Fragment via BRICS, then recombine."""
        frags = tmp_path / "frags.csv"
        run_cli([
            "fragment", "brics",
            "-i", str(mol_csv), "-o", str(frags), "-q",
        ])
        built = tmp_path / "built.csv"
        result = run_cli([
            "fragment", "brics-build",
            "-i", str(frags), "-o", str(built),
            "--max-molecules", "3", "-q",
        ])
        assert result.returncode == 0
