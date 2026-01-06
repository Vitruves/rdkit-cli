"""Unit tests for deduplicate module."""

import pytest
from rdkit import Chem


class TestDeduplicator:
    """Test Deduplicator class."""

    def test_deduplicate_by_smiles(self):
        """Test deduplication by canonical SMILES."""
        from rdkit_cli.core.deduplicate import Deduplicator
        from rdkit_cli.io.readers import MoleculeRecord

        records = [
            MoleculeRecord(Chem.MolFromSmiles("CCO"), smiles="CCO"),
            MoleculeRecord(Chem.MolFromSmiles("OCC"), smiles="OCC"),  # Same as CCO
            MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C"),
            MoleculeRecord(Chem.MolFromSmiles("CCO"), smiles="CCO"),  # Duplicate
        ]

        dedup = Deduplicator(key_type="smiles")
        unique, n_removed = dedup.deduplicate(records)

        assert len(unique) == 2
        assert n_removed == 2

    def test_deduplicate_keep_first(self):
        """Test keeping first occurrence."""
        from rdkit_cli.core.deduplicate import Deduplicator
        from rdkit_cli.io.readers import MoleculeRecord

        records = [
            MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C", name="first"),
            MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C", name="second"),
        ]

        dedup = Deduplicator(key_type="smiles", keep="first")
        unique, _ = dedup.deduplicate(records)

        assert len(unique) == 1
        assert unique[0].name == "first"

    def test_deduplicate_keep_last(self):
        """Test keeping last occurrence."""
        from rdkit_cli.core.deduplicate import Deduplicator
        from rdkit_cli.io.readers import MoleculeRecord

        records = [
            MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C", name="first"),
            MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C", name="second"),
        ]

        dedup = Deduplicator(key_type="smiles", keep="last")
        unique, _ = dedup.deduplicate(records)

        assert len(unique) == 1
        assert unique[0].name == "second"

    def test_deduplicate_by_inchikey(self):
        """Test deduplication by InChIKey."""
        from rdkit_cli.core.deduplicate import Deduplicator
        from rdkit_cli.io.readers import MoleculeRecord

        records = [
            MoleculeRecord(Chem.MolFromSmiles("CCO"), smiles="CCO"),
            MoleculeRecord(Chem.MolFromSmiles("OCC"), smiles="OCC"),  # Same molecule
            MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C"),
        ]

        dedup = Deduplicator(key_type="inchikey")
        unique, n_removed = dedup.deduplicate(records)

        assert len(unique) == 2
        assert n_removed == 1

    def test_deduplicate_by_scaffold(self):
        """Test deduplication by Murcko scaffold."""
        from rdkit_cli.core.deduplicate import Deduplicator
        from rdkit_cli.io.readers import MoleculeRecord

        records = [
            MoleculeRecord(Chem.MolFromSmiles("c1ccccc1C"), smiles="c1ccccc1C"),  # toluene
            MoleculeRecord(Chem.MolFromSmiles("c1ccccc1CC"), smiles="c1ccccc1CC"),  # ethylbenzene
            MoleculeRecord(Chem.MolFromSmiles("CCO"), smiles="CCO"),  # ethanol (no scaffold)
        ]

        dedup = Deduplicator(key_type="scaffold")
        unique, n_removed = dedup.deduplicate(records)

        # Toluene and ethylbenzene have same scaffold (benzene)
        assert n_removed == 1

    def test_deduplicate_handles_none(self):
        """Test that None molecules are preserved."""
        from rdkit_cli.core.deduplicate import Deduplicator
        from rdkit_cli.io.readers import MoleculeRecord

        records = [
            MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C"),
            MoleculeRecord(None, smiles="invalid"),
            MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C"),  # Duplicate
            MoleculeRecord(None, smiles="also_invalid"),
        ]

        dedup = Deduplicator(key_type="smiles")
        unique, n_removed = dedup.deduplicate(records)

        # Should keep 1 valid + 2 invalid
        assert len(unique) == 3
        assert n_removed == 1

    def test_invalid_key_type(self):
        """Test error on invalid key type."""
        from rdkit_cli.core.deduplicate import Deduplicator

        with pytest.raises(ValueError, match="Unknown key_type"):
            Deduplicator(key_type="invalid")

    def test_invalid_keep(self):
        """Test error on invalid keep value."""
        from rdkit_cli.core.deduplicate import Deduplicator

        with pytest.raises(ValueError, match="keep must be"):
            Deduplicator(key_type="smiles", keep="middle")

    def test_stream_deduplication(self):
        """Test stream-based deduplication."""
        from rdkit_cli.core.deduplicate import Deduplicator
        from rdkit_cli.io.readers import MoleculeRecord

        records = [
            MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C"),
            MoleculeRecord(Chem.MolFromSmiles("C"), smiles="C"),
            MoleculeRecord(Chem.MolFromSmiles("CC"), smiles="CC"),
        ]

        dedup = Deduplicator(key_type="smiles", keep="first")
        unique = list(dedup.deduplicate_stream(iter(records)))

        assert len(unique) == 2

    def test_stream_deduplication_requires_first(self):
        """Test that stream deduplication requires keep='first'."""
        from rdkit_cli.core.deduplicate import Deduplicator
        from rdkit_cli.io.readers import MoleculeRecord

        dedup = Deduplicator(key_type="smiles", keep="last")

        with pytest.raises(ValueError, match="only supports keep='first'"):
            list(dedup.deduplicate_stream(iter([])))

    def test_available_key_types(self):
        """Test available key types list."""
        from rdkit_cli.core.deduplicate import Deduplicator

        key_types = Deduplicator.available_key_types()
        assert "smiles" in key_types
        assert "inchi" in key_types
        assert "inchikey" in key_types
        assert "scaffold" in key_types
