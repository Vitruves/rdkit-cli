"""Tests for stereo, energy, pharmacophore, depict highlight,
conformers torsion, and reactions fingerprint."""

import pytest
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit_cli.io.readers import MoleculeRecord


def _rec(smiles, name=""):
    mol = Chem.MolFromSmiles(smiles)
    return MoleculeRecord(mol=mol, smiles=smiles, name=name)


# ---------------------------------------------------------------------------
# stereo
# ---------------------------------------------------------------------------


class TestStereoAssign:

    def test_assign_r_s(self):
        from rdkit_cli.core.stereo import StereoAssigner

        a = StereoAssigner()
        result = a.assign(_rec("C[C@H](O)F", "chiral"))

        assert result is not None
        assert result["num_stereocenters"] >= 1
        assert "cip_labels" in result

    def test_assign_no_stereo(self):
        from rdkit_cli.core.stereo import StereoAssigner

        a = StereoAssigner()
        result = a.assign(_rec("CCO", "ethanol"))

        assert result is not None
        assert result["num_stereocenters"] == 0

    def test_assign_none_mol(self):
        from rdkit_cli.core.stereo import StereoAssigner

        a = StereoAssigner()
        r = MoleculeRecord(mol=None, smiles="bad")
        assert a.assign(r) is None


class TestStereoPerceive:

    def test_perceive_potential(self):
        from rdkit_cli.core.stereo import StereoPerceiver

        p = StereoPerceiver()
        result = p.perceive(_rec("CC(O)(F)Cl", "potential"))

        assert result is not None
        assert result["num_potential"] >= 1

    def test_perceive_specified(self):
        from rdkit_cli.core.stereo import StereoPerceiver

        p = StereoPerceiver()
        result = p.perceive(_rec("C[C@H](O)F"))
        assert result["num_specified"] >= 1


class TestStereoEnhanced:

    def test_enhanced_groups(self):
        from rdkit_cli.core.stereo import get_enhanced_stereo

        mol = Chem.MolFromSmiles("C[C@H](O)F")
        groups = get_enhanced_stereo(mol)
        # Most simple molecules have no enhanced groups
        assert isinstance(groups, list)


class TestStereoFunctions:

    def test_assign_cip_labels(self):
        from rdkit_cli.core.stereo import assign_cip_labels

        mol = Chem.MolFromSmiles("C[C@H](O)F")
        labels = assign_cip_labels(mol)
        assert len(labels) >= 1
        assert labels[0]["cip"] in ("R", "S")

    def test_perceive_stereo(self):
        from rdkit_cli.core.stereo import perceive_stereo

        mol = Chem.MolFromSmiles("CC(O)(F)Cl")
        info = perceive_stereo(mol)
        assert len(info) >= 1
        assert "type" in info[0]


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------


class TestEnergyCalculator:

    def test_compute_mmff(self):
        from rdkit_cli.core.energy import EnergyCalculator

        calc = EnergyCalculator(force_field="mmff")
        result = calc.compute(_rec("CCO", "ethanol"))

        assert result is not None
        assert "energy" in result
        assert isinstance(result["energy"], float)
        assert result["force_field"] == "MMFF"

    def test_compute_uff(self):
        from rdkit_cli.core.energy import EnergyCalculator

        calc = EnergyCalculator(force_field="uff")
        result = calc.compute(_rec("c1ccccc1", "benzene"))

        assert result is not None
        assert isinstance(result["energy"], float)

    def test_compute_none_mol(self):
        from rdkit_cli.core.energy import EnergyCalculator

        calc = EnergyCalculator()
        r = MoleculeRecord(mol=None, smiles="bad")
        assert calc.compute(r) is None


class TestEnergyMinimizer:

    def test_minimize(self):
        from rdkit_cli.core.energy import EnergyMinimizer

        m = EnergyMinimizer(force_field="mmff", max_iterations=100)
        result = m.minimize(_rec("CCCC", "butane"))

        assert result is not None
        assert "energy_before" in result
        assert "energy_after" in result
        assert isinstance(result["converged"], bool)

    def test_minimize_lowers_energy(self):
        from rdkit_cli.core.energy import EnergyMinimizer

        m = EnergyMinimizer(force_field="mmff", max_iterations=500)
        result = m.minimize(_rec("CCCC"))

        if result is not None and result["energy_before"] is not None:
            assert result["energy_after"] <= result["energy_before"] + 0.1


class TestComputeEnergy:

    def test_basic(self):
        from rdkit_cli.core.energy import compute_energy

        mol = Chem.MolFromSmiles("CCO")
        e = compute_energy(mol, "mmff")
        assert e is not None
        assert isinstance(e, float)


# ---------------------------------------------------------------------------
# pharmacophore
# ---------------------------------------------------------------------------


class TestPharmacophorePerceiver:

    def test_perceive_phenol(self):
        from rdkit_cli.core.pharmacophore import PharmacophorePerceiver

        p = PharmacophorePerceiver()
        result = p.perceive(_rec("c1ccc(O)cc1", "phenol"))

        assert result is not None
        assert result["num_features"] > 0
        # Phenol should have HD (donor) and HA (acceptor) features
        assert result.get("n_HD", 0) > 0 or result.get("n_HA", 0) > 0

    def test_perceive_aliphatic(self):
        from rdkit_cli.core.pharmacophore import PharmacophorePerceiver

        p = PharmacophorePerceiver()
        result = p.perceive(_rec("CCCCCC", "hexane"))

        assert result is not None
        # Hexane has no donors or acceptors
        assert result.get("n_HD", 0) == 0
        assert result.get("n_HA", 0) == 0

    def test_perceive_none(self):
        from rdkit_cli.core.pharmacophore import PharmacophorePerceiver

        p = PharmacophorePerceiver()
        r = MoleculeRecord(mol=None, smiles="bad")
        assert p.perceive(r) is None


class TestPharmacophoreSearcher:

    def test_search_similar(self):
        from rdkit_cli.core.pharmacophore import PharmacophoreSearcher

        s = PharmacophoreSearcher(
            query_smiles="c1ccc(O)cc1", threshold=0.0,
        )
        result = s.search(_rec("c1ccc(O)cc1", "phenol"))

        assert result is not None
        assert result["pharmacophore_similarity"] == pytest.approx(1.0)

    def test_search_threshold_filters(self):
        from rdkit_cli.core.pharmacophore import PharmacophoreSearcher

        s = PharmacophoreSearcher(
            query_smiles="c1ccc(O)cc1", threshold=0.99,
        )
        result = s.search(_rec("CCCCCC", "hexane"))
        # Hexane vs phenol should be very dissimilar
        assert result is None

    def test_invalid_query(self):
        from rdkit_cli.core.pharmacophore import PharmacophoreSearcher

        with pytest.raises(ValueError, match="Invalid query"):
            PharmacophoreSearcher(query_smiles="not_a_smiles")


class TestPerceiveFeatures:

    def test_basic(self):
        from rdkit_cli.core.pharmacophore import perceive_features

        mol = Chem.MolFromSmiles("c1ccc(O)cc1")
        feats = perceive_features(mol)
        assert len(feats) > 0
        assert "family" in feats[0]
        assert "atoms" in feats[0]


# ---------------------------------------------------------------------------
# depict highlight
# ---------------------------------------------------------------------------


class TestDepictHighlight:

    def test_highlight_svg(self, tmp_path):
        from rdkit_cli.commands.depict import run_highlight
        import argparse

        out = tmp_path / "out.svg"
        args = argparse.Namespace(
            smiles="c1ccccc1O",
            smarts="c1ccccc1",
            output=str(out),
            width=400,
            height=300,
            color="1.0,0.0,0.0",
        )
        ret = run_highlight(args)
        assert ret == 0
        assert out.exists()
        content = out.read_text()
        assert "<svg" in content

    def test_highlight_png(self, tmp_path):
        from rdkit_cli.commands.depict import run_highlight
        import argparse

        out = tmp_path / "out.png"
        args = argparse.Namespace(
            smiles="CCO",
            smarts="O",
            output=str(out),
            width=400,
            height=300,
            color="0.0,0.0,1.0",
        )
        ret = run_highlight(args)
        assert ret == 0
        assert out.exists()
        assert out.stat().st_size > 0

    def test_highlight_no_match(self, tmp_path):
        from rdkit_cli.commands.depict import run_highlight
        import argparse

        out = tmp_path / "out.svg"
        args = argparse.Namespace(
            smiles="CCCC",
            smarts="c1ccccc1",
            output=str(out),
            width=400,
            height=300,
            color="1.0,0.0,0.0",
        )
        ret = run_highlight(args)
        assert ret == 0  # Still succeeds, just no highlights

    def test_highlight_invalid_smiles(self, tmp_path):
        from rdkit_cli.commands.depict import run_highlight
        import argparse

        out = tmp_path / "out.svg"
        args = argparse.Namespace(
            smiles="not_valid",
            smarts="C",
            output=str(out),
            width=400,
            height=300,
            color="1.0,0.0,0.0",
        )
        ret = run_highlight(args)
        assert ret == 1


# ---------------------------------------------------------------------------
# conformers torsion
# ---------------------------------------------------------------------------


class TestTorsionScanner:

    def test_scan_butane(self):
        from rdkit_cli.core.conformers import TorsionScanner

        scanner = TorsionScanner(
            atom_indices=(0, 1, 2, 3),
            start_angle=-180,
            end_angle=180,
            step=60,
        )
        result = scanner.scan(_rec("CCCC", "butane"))

        assert result is not None
        assert "min_angle" in result
        assert "min_energy" in result
        assert "barrier" in result
        assert result["barrier"] >= 0

    def test_scan_none_mol(self):
        from rdkit_cli.core.conformers import TorsionScanner

        scanner = TorsionScanner(atom_indices=(0, 1, 2, 3))
        r = MoleculeRecord(mol=None, smiles="bad")
        assert scanner.scan(r) is None


# ---------------------------------------------------------------------------
# reactions fingerprint
# ---------------------------------------------------------------------------


class TestReactionFingerprint:

    def test_difference_fp(self):
        from rdkit_cli.core.reactions import compute_reaction_fingerprint

        fp = compute_reaction_fingerprint(
            "[C:1]=[O:2]>>[C:1][O:2]",
            fp_type="difference",
        )
        assert fp is not None
        assert fp.GetLength() > 0

    def test_structural_fp(self):
        from rdkit_cli.core.reactions import compute_reaction_fingerprint

        fp = compute_reaction_fingerprint(
            "[C:1]=[O:2]>>[C:1][O:2]",
            fp_type="structural",
        )
        assert fp is not None
        assert fp.GetNumBits() > 0

    def test_invalid_smarts(self):
        from rdkit_cli.core.reactions import compute_reaction_fingerprint

        with pytest.raises(ValueError):
            compute_reaction_fingerprint("invalid")

    def test_different_reactions_differ(self):
        from rdkit_cli.core.reactions import compute_reaction_fingerprint

        fp1 = compute_reaction_fingerprint(
            "[C:1]=[O:2]>>[C:1][O:2]",
            fp_type="structural",
        )
        fp2 = compute_reaction_fingerprint(
            "[N:1][C:2]>>[N:1]=[C:2]",
            fp_type="structural",
        )
        assert fp1.ToBitString() != fp2.ToBitString()
