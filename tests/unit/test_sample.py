"""Unit tests for sample module."""

import pytest
from rdkit import Chem


class TestMoleculeSampler:
    """Test MoleculeSampler class."""

    def test_init_requires_one_param(self):
        """Test that either n or fraction must be specified."""
        from rdkit_cli.core.sample import MoleculeSampler

        with pytest.raises(ValueError, match="Either n or fraction"):
            MoleculeSampler()

        with pytest.raises(ValueError, match="Only one of"):
            MoleculeSampler(n=10, fraction=0.5)

    def test_invalid_fraction(self):
        """Test that fraction must be between 0 and 1."""
        from rdkit_cli.core.sample import MoleculeSampler

        with pytest.raises(ValueError, match="fraction must be between"):
            MoleculeSampler(fraction=1.5)

        with pytest.raises(ValueError, match="fraction must be between"):
            MoleculeSampler(fraction=0.0)

    def test_sample_by_count(self):
        """Test sampling by count."""
        from rdkit_cli.core.sample import MoleculeSampler
        from rdkit_cli.io.readers import MoleculeRecord

        records = [MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C") for _ in range(100)]
        sampler = MoleculeSampler(n=10, seed=42)
        result = sampler.sample(records)

        assert len(result) == 10

    def test_sample_by_fraction(self):
        """Test sampling by fraction."""
        from rdkit_cli.core.sample import MoleculeSampler
        from rdkit_cli.io.readers import MoleculeRecord

        records = [MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C") for _ in range(100)]
        sampler = MoleculeSampler(fraction=0.1, seed=42)
        result = sampler.sample(records)

        assert len(result) == 10

    def test_sample_reproducible(self):
        """Test that sampling is reproducible with seed."""
        from rdkit_cli.core.sample import MoleculeSampler
        from rdkit_cli.io.readers import MoleculeRecord

        records = [MoleculeRecord(Chem.MolFromSmiles("C"), smiles=f"mol{i}") for i in range(100)]

        sampler1 = MoleculeSampler(n=10, seed=42)
        sampler2 = MoleculeSampler(n=10, seed=42)

        result1 = sampler1.sample(records)
        result2 = sampler2.sample(records)

        assert [r.smiles for r in result1] == [r.smiles for r in result2]

    def test_sample_more_than_available(self):
        """Test sampling more than available."""
        from rdkit_cli.core.sample import MoleculeSampler
        from rdkit_cli.io.readers import MoleculeRecord

        records = [MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C") for _ in range(5)]
        sampler = MoleculeSampler(n=100, seed=42)
        result = sampler.sample(records)

        assert len(result) == 5

    def test_empty_input(self):
        """Test with empty input."""
        from rdkit_cli.core.sample import MoleculeSampler

        sampler = MoleculeSampler(n=10)
        result = sampler.sample([])

        assert len(result) == 0

    def test_stratified_sampling(self):
        """Test stratified sampling maintains ratio."""
        from rdkit_cli.core.sample import MoleculeSampler
        from rdkit_cli.io.readers import MoleculeRecord

        # 80 valid, 20 invalid
        valid = [MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C") for _ in range(80)]
        invalid = [MoleculeRecord(None, smiles="invalid") for _ in range(20)]
        records = valid + invalid

        sampler = MoleculeSampler(n=50, seed=42, stratify_valid=True)
        result = sampler.sample(records)

        # Should maintain roughly 80/20 ratio
        valid_count = sum(1 for r in result if r.is_valid)
        invalid_count = len(result) - valid_count

        # Allow some tolerance
        assert 35 <= valid_count <= 45
        assert 5 <= invalid_count <= 15


class TestReservoirSampler:
    """Test ReservoirSampler class."""

    def test_reservoir_sampling(self):
        """Test reservoir sampling."""
        from rdkit_cli.core.sample import ReservoirSampler
        from rdkit_cli.io.readers import MoleculeRecord

        sampler = ReservoirSampler(n=10, seed=42)

        for i in range(100):
            record = MoleculeRecord(Chem.MolFromSmiles("C"), smiles=f"mol{i}")
            sampler.add(record)

        result = sampler.get_sample()
        assert len(result) == 10

    def test_reservoir_fewer_than_n(self):
        """Test reservoir when fewer items than n."""
        from rdkit_cli.core.sample import ReservoirSampler
        from rdkit_cli.io.readers import MoleculeRecord

        sampler = ReservoirSampler(n=100, seed=42)

        for i in range(5):
            record = MoleculeRecord(Chem.MolFromSmiles("C"), smiles=f"mol{i}")
            sampler.add(record)

        result = sampler.get_sample()
        assert len(result) == 5

    def test_reservoir_reproducible(self):
        """Test reservoir sampling is reproducible."""
        from rdkit_cli.core.sample import ReservoirSampler
        from rdkit_cli.io.readers import MoleculeRecord

        def run_sampling():
            sampler = ReservoirSampler(n=10, seed=42)
            for i in range(100):
                record = MoleculeRecord(Chem.MolFromSmiles("C"), smiles=f"mol{i}")
                sampler.add(record)
            return [r.smiles for r in sampler.get_sample()]

        result1 = run_sampling()
        result2 = run_sampling()

        assert result1 == result2
