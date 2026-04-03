"""Tests for extended fingerprints, descriptors, similarity metrics, and alerts."""

import pytest
from rdkit import Chem

from rdkit_cli.io.readers import MoleculeRecord


def _make_record(smiles: str, name: str = "") -> MoleculeRecord:
    """Helper to create a MoleculeRecord from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    return MoleculeRecord(mol=mol, smiles=smiles, name=name)


# ---------------------------------------------------------------------------
# 1. New fingerprint types: Avalon, MHFP, Pharmacophore
# ---------------------------------------------------------------------------


class TestAvalonFingerprint:
    """Test Avalon fingerprint computation."""

    def test_compute_avalon(self):
        from rdkit_cli.core.fingerprints import FingerprintCalculator, FingerprintType

        calc = FingerprintCalculator(fp_type=FingerprintType.AVALON, n_bits=512)
        result = calc.compute(_make_record("c1ccccc1", "benzene"))

        assert result is not None
        assert "fingerprint" in result
        assert len(result["fingerprint"]) > 0

    def test_avalon_different_molecules_differ(self):
        from rdkit_cli.core.fingerprints import FingerprintType, compute_fingerprint

        mol1 = Chem.MolFromSmiles("c1ccccc1")
        mol2 = Chem.MolFromSmiles("CCCCCC")

        fp1 = compute_fingerprint(mol1, FingerprintType.AVALON, n_bits=512)
        fp2 = compute_fingerprint(mol2, FingerprintType.AVALON, n_bits=512)

        assert fp1 is not None
        assert fp2 is not None
        assert fp1.ToBitString() != fp2.ToBitString()

    def test_avalon_default_bits(self):
        from rdkit_cli.core.fingerprints import FINGERPRINT_INFO, FingerprintType

        assert FINGERPRINT_INFO[FingerprintType.AVALON].default_bits == 512

    def test_avalon_none_molecule(self):
        from rdkit_cli.core.fingerprints import FingerprintType, compute_fingerprint

        result = compute_fingerprint(None, FingerprintType.AVALON)
        assert result is None


class TestMHFPFingerprint:
    """Test MHFP (MinHash) fingerprint computation."""

    def test_compute_mhfp(self):
        from rdkit_cli.core.fingerprints import FingerprintCalculator, FingerprintType

        calc = FingerprintCalculator(fp_type=FingerprintType.MHFP, n_bits=2048, radius=3)
        result = calc.compute(_make_record("c1ccccc1", "benzene"))

        assert result is not None
        assert "fingerprint" in result
        assert len(result["fingerprint"]) > 0

    def test_mhfp_different_radii(self):
        from rdkit_cli.core.fingerprints import FingerprintType, compute_fingerprint

        mol = Chem.MolFromSmiles("c1ccc(CC)cc1")
        fp_r2 = compute_fingerprint(mol, FingerprintType.MHFP, radius=2)
        fp_r3 = compute_fingerprint(mol, FingerprintType.MHFP, radius=3)

        assert fp_r2 is not None
        assert fp_r3 is not None
        # Different radii should generally produce different fingerprints
        assert fp_r2.ToBitString() != fp_r3.ToBitString()

    def test_mhfp_has_radius(self):
        from rdkit_cli.core.fingerprints import FINGERPRINT_INFO, FingerprintType

        assert FINGERPRINT_INFO[FingerprintType.MHFP].has_radius is True


class TestPharmacophoreFingerprint:
    """Test 2D pharmacophore (Gobbi) fingerprint computation."""

    def test_compute_pharmacophore(self):
        from rdkit_cli.core.fingerprints import FingerprintCalculator, FingerprintType

        calc = FingerprintCalculator(fp_type=FingerprintType.PHARMACOPHORE)
        result = calc.compute(_make_record("c1ccc(O)cc1", "phenol"))

        assert result is not None
        assert "fingerprint" in result
        assert len(result["fingerprint"]) > 0

    def test_pharmacophore_fixed_size(self):
        from rdkit_cli.core.fingerprints import FingerprintCalculator, FingerprintType

        calc = FingerprintCalculator(fp_type=FingerprintType.PHARMACOPHORE)
        # n_bits should be overridden to 39972
        assert calc.n_bits == 39972

    def test_pharmacophore_different_molecules(self):
        from rdkit_cli.core.fingerprints import FingerprintType, compute_fingerprint

        mol1 = Chem.MolFromSmiles("c1ccc(O)cc1")  # phenol (donor + aromatic)
        mol2 = Chem.MolFromSmiles("CCCCCC")  # hexane (no pharmacophoric features)

        fp1 = compute_fingerprint(mol1, FingerprintType.PHARMACOPHORE)
        fp2 = compute_fingerprint(mol2, FingerprintType.PHARMACOPHORE)

        assert fp1 is not None
        assert fp2 is not None
        assert fp1.GetNumOnBits() != fp2.GetNumOnBits()


class TestFingerprintListIncludesNewTypes:
    """Test that fingerprint list includes all new types."""

    def test_list_contains_new_types(self):
        from rdkit_cli.core.fingerprints import list_fingerprints

        fps = list_fingerprints()
        names = [fp.name for fp in fps]

        assert "avalon" in names
        assert "mhfp" in names
        assert "pharmacophore" in names
        assert len(fps) == 9  # 6 original + 3 new


# ---------------------------------------------------------------------------
# 2. MQN and 3D descriptors
# ---------------------------------------------------------------------------


class TestMQNDescriptors:
    """Test Molecular Quantum Number descriptors."""

    def test_compute_mqn(self):
        from rdkit_cli.core.descriptors import MQN_DESCRIPTORS, DescriptorCalculator

        calc = DescriptorCalculator(descriptors=MQN_DESCRIPTORS)
        result = calc.compute(_make_record("c1ccccc1", "benzene"))

        assert result is not None
        assert "MQN1" in result
        assert "MQN42" in result
        assert len([k for k in result if k.startswith("MQN")]) == 42

    def test_mqn_values_are_numeric(self):
        from rdkit_cli.core.descriptors import MQN_DESCRIPTORS, DescriptorCalculator

        calc = DescriptorCalculator(descriptors=MQN_DESCRIPTORS)
        result = calc.compute(_make_record("CCO", "ethanol"))

        for name in MQN_DESCRIPTORS:
            assert isinstance(result[name], (int, float)), f"{name} is not numeric"

    def test_mqn_category_filter(self):
        from rdkit_cli.core.descriptors import list_descriptors

        mqn_descs = list_descriptors(category="mqn")
        assert len(mqn_descs) == 42

    def test_mqn_different_molecules(self):
        from rdkit_cli.core.descriptors import DescriptorCalculator

        calc = DescriptorCalculator(descriptors=["MQN1", "MQN2", "MQN3"])

        r1 = calc.compute(_make_record("c1ccccc1", "benzene"))
        r2 = calc.compute(_make_record("CCCCCCCCCCC", "undecane"))

        # Benzene and undecane should differ in atom counts
        assert r1["MQN1"] != r2["MQN1"]


class TestThreeDDescriptors:
    """Test 3D shape descriptors."""

    def test_compute_3d_with_conformer_generation(self):
        from rdkit_cli.core.descriptors import THREE_D_DESCRIPTORS, DescriptorCalculator

        calc = DescriptorCalculator(
            descriptors=THREE_D_DESCRIPTORS,
            generate_conformers=True,
        )
        result = calc.compute(_make_record("c1ccccc1", "benzene"))

        assert result is not None
        assert "PMI1" in result
        assert "NPR1" in result
        assert "Asphericity" in result
        assert "Eccentricity" in result
        assert "SpherocityIndex" in result
        assert "PBF" in result

    def test_3d_values_are_numeric(self):
        from rdkit_cli.core.descriptors import THREE_D_DESCRIPTORS, DescriptorCalculator

        calc = DescriptorCalculator(
            descriptors=THREE_D_DESCRIPTORS,
            generate_conformers=True,
        )
        result = calc.compute(_make_record("CCO", "ethanol"))

        for name in THREE_D_DESCRIPTORS:
            val = result[name]
            assert isinstance(val, (int, float)), f"{name} = {val} is not numeric"

    def test_3d_category_filter(self):
        from rdkit_cli.core.descriptors import list_descriptors

        three_d = list_descriptors(category="3d")
        assert len(three_d) == 10

    def test_3d_without_conformers_returns_error_value(self):
        from rdkit_cli.core.descriptors import DescriptorCalculator

        calc = DescriptorCalculator(
            descriptors=["PMI1"],
            generate_conformers=False,
            error_value="NaN",
        )
        result = calc.compute(_make_record("c1ccccc1", "benzene"))

        # Without 3D coords and without generate_conformers, should get error value
        assert result is not None
        assert result["PMI1"] == "NaN"

    def test_total_descriptor_count_increased(self):
        from rdkit_cli.core.descriptors import list_descriptors

        all_descs = list_descriptors()
        # Original ~133 + QED + 42 MQN + 10 3D = ~186
        assert len(all_descs) > 180


# ---------------------------------------------------------------------------
# 3. New similarity metrics
# ---------------------------------------------------------------------------


class TestNewSimilarityMetrics:
    """Test 8 new similarity metrics."""

    @pytest.fixture
    def benzene_toluene_fps(self):
        from rdkit_cli.core.similarity import get_morgan_fingerprint

        mol1 = Chem.MolFromSmiles("c1ccccc1")
        mol2 = Chem.MolFromSmiles("Cc1ccccc1")
        return get_morgan_fingerprint(mol1), get_morgan_fingerprint(mol2)

    @pytest.mark.parametrize("metric_name", [
        "allbit", "asymmetric", "braunblanquet", "kulczynski",
        "mcconnaughey", "onbit", "rogotgoldberg", "tversky",
    ])
    def test_new_metric_returns_float(self, metric_name, benzene_toluene_fps):
        from rdkit_cli.core.similarity import (
            SimilarityMetric,
            compute_similarity,
        )

        fp1, fp2 = benzene_toluene_fps
        metric = SimilarityMetric(metric_name)
        sim = compute_similarity(fp1, fp2, metric)

        assert isinstance(sim, float)

    @pytest.mark.parametrize("metric_name", [
        "allbit", "asymmetric", "braunblanquet", "kulczynski",
        "onbit", "rogotgoldberg", "tversky",
    ])
    def test_new_metric_identical_is_one(self, metric_name):
        from rdkit_cli.core.similarity import (
            SimilarityMetric,
            compute_similarity,
            get_morgan_fingerprint,
        )

        mol = Chem.MolFromSmiles("c1ccccc1")
        fp = get_morgan_fingerprint(mol)
        metric = SimilarityMetric(metric_name)

        sim = compute_similarity(fp, fp, metric)
        assert sim == pytest.approx(1.0)

    def test_tversky_alpha_beta_affect_result(self, benzene_toluene_fps):
        from rdkit_cli.core.similarity import (
            SimilarityMetric,
            compute_similarity,
        )

        fp1, fp2 = benzene_toluene_fps

        sim_a = compute_similarity(
            fp1, fp2, SimilarityMetric.TVERSKY,
            tversky_alpha=0.9, tversky_beta=0.1,
        )
        sim_b = compute_similarity(
            fp1, fp2, SimilarityMetric.TVERSKY,
            tversky_alpha=0.1, tversky_beta=0.9,
        )

        # Asymmetric: different alpha/beta should give different results
        assert sim_a != sim_b

    def test_tversky_symmetric_equals_tanimoto(self):
        from rdkit_cli.core.similarity import (
            SimilarityMetric,
            compute_similarity,
            get_morgan_fingerprint,
        )

        mol1 = Chem.MolFromSmiles("c1ccccc1")
        mol2 = Chem.MolFromSmiles("Cc1ccccc1")
        fp1 = get_morgan_fingerprint(mol1)
        fp2 = get_morgan_fingerprint(mol2)

        tanimoto = compute_similarity(fp1, fp2, SimilarityMetric.TANIMOTO)
        tversky = compute_similarity(
            fp1, fp2, SimilarityMetric.TVERSKY,
            tversky_alpha=1.0, tversky_beta=1.0,
        )

        assert tanimoto == pytest.approx(tversky)

    def test_searcher_with_tversky(self):
        from rdkit_cli.core.similarity import (
            SimilarityMetric,
            SimilaritySearcher,
        )

        searcher = SimilaritySearcher(
            query_smiles="c1ccccc1",
            threshold=0.1,
            metric=SimilarityMetric.TVERSKY,
            tversky_alpha=0.8,
            tversky_beta=0.2,
        )
        result = searcher.search(_make_record("Cc1ccccc1", "toluene"))

        assert result is not None
        assert 0 < result["similarity"] <= 1.0

    def test_similarity_matrix_with_new_metric(self):
        from rdkit_cli.core.similarity import (
            SimilarityMetric,
            compute_similarity_matrix,
        )

        mols = [Chem.MolFromSmiles(s) for s in ["c1ccccc1", "Cc1ccccc1", "CCO"]]
        matrix = compute_similarity_matrix(mols, metric=SimilarityMetric.BRAUNBLANQUET)

        assert len(matrix) == 3
        assert matrix[0][0] == pytest.approx(1.0)
        assert matrix[0][1] == matrix[1][0]  # symmetric

    def test_all_metrics_in_enum(self):
        from rdkit_cli.core.similarity import SimilarityMetric

        expected = {
            "tanimoto", "dice", "cosine", "sokal", "russel",
            "allbit", "asymmetric", "braunblanquet", "kulczynski",
            "mcconnaughey", "onbit", "rogotgoldberg", "tversky",
        }
        actual = {m.value for m in SimilarityMetric}
        assert actual == expected


# ---------------------------------------------------------------------------
# 4. Filter alert catalogs: Brenk, NIH, ZINC
# ---------------------------------------------------------------------------


class TestFilterAlertCatalogs:
    """Test expanded structural alert catalogs."""

    def test_brenk_catalog_loads(self):
        from rdkit_cli.core.filters import PAINSFilter

        filt = PAINSFilter(catalog_name="brenk")
        assert filt.catalog.GetNumEntries() > 0

    def test_nih_catalog_loads(self):
        from rdkit_cli.core.filters import PAINSFilter

        filt = PAINSFilter(catalog_name="nih")
        assert filt.catalog.GetNumEntries() > 0

    def test_zinc_catalog_loads(self):
        from rdkit_cli.core.filters import PAINSFilter

        filt = PAINSFilter(catalog_name="zinc")
        assert filt.catalog.GetNumEntries() > 0

    def test_all_catalog_loads(self):
        from rdkit_cli.core.filters import PAINSFilter

        filt = PAINSFilter(catalog_name="all")
        # "all" should have more entries than any individual catalog
        pains_filt = PAINSFilter(catalog_name="pains")
        assert filt.catalog.GetNumEntries() > pains_filt.catalog.GetNumEntries()

    def test_pains_a_b_c_catalogs(self):
        from rdkit_cli.core.filters import PAINSFilter

        for cat in ["pains_a", "pains_b", "pains_c"]:
            filt = PAINSFilter(catalog_name=cat)
            assert filt.catalog.GetNumEntries() > 0

    def test_invalid_catalog_raises(self):
        from rdkit_cli.core.filters import PAINSFilter

        with pytest.raises(ValueError, match="Unknown catalog"):
            PAINSFilter(catalog_name="nonexistent")

    def test_brenk_filters_problematic_molecule(self):
        from rdkit_cli.core.filters import PAINSFilter

        filt = PAINSFilter(catalog_name="brenk", exclude=True)

        # Acyl halide (known Brenk alert)
        record = _make_record("CC(=O)Cl", "acetyl_chloride")
        result = filt.filter(record)
        assert result is None  # Should be filtered out

    def test_clean_molecule_passes_all_catalogs(self):
        from rdkit_cli.core.filters import PAINSFilter

        filt = PAINSFilter(catalog_name="all", exclude=True)

        # Simple benign molecule
        record = _make_record("CCO", "ethanol")
        result = filt.filter(record)
        assert result is not None

    def test_backward_compatible_default_is_pains(self):
        from rdkit_cli.core.filters import PAINSFilter

        # Default should still be PAINS
        filt = PAINSFilter()
        pains_filt = PAINSFilter(catalog_name="pains")
        assert filt.catalog.GetNumEntries() == pains_filt.catalog.GetNumEntries()

    def test_alert_catalogs_dict(self):
        from rdkit_cli.core.filters import ALERT_CATALOGS

        expected_keys = {"pains", "pains_a", "pains_b", "pains_c", "brenk", "nih", "zinc"}
        assert set(ALERT_CATALOGS.keys()) == expected_keys
