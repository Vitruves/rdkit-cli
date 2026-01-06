"""Unit tests for validate module."""

import pytest
from rdkit import Chem


class TestMoleculeValidator:
    """Test MoleculeValidator class."""

    def test_validate_valid_molecule(self):
        """Test validating a valid molecule."""
        from rdkit_cli.core.validate import MoleculeValidator

        validator = MoleculeValidator()
        mol = Chem.MolFromSmiles("CCO")
        result = validator.validate(mol, "CCO")

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_none_molecule(self):
        """Test validating a None molecule."""
        from rdkit_cli.core.validate import MoleculeValidator

        validator = MoleculeValidator()
        result = validator.validate(None, "invalid_smiles")

        assert not result.is_valid
        assert "Failed to parse" in result.errors[0]

    def test_validate_max_atoms(self):
        """Test max atoms constraint."""
        from rdkit_cli.core.validate import MoleculeValidator

        validator = MoleculeValidator(max_atoms=5)

        # Valid - 3 atoms
        mol = Chem.MolFromSmiles("CCO")
        result = validator.validate(mol)
        assert result.is_valid

        # Invalid - too many atoms
        mol = Chem.MolFromSmiles("CCCCCCCCCC")
        result = validator.validate(mol)
        assert not result.is_valid
        assert "Too many atoms" in result.errors[0]

    def test_validate_max_rings(self):
        """Test max rings constraint."""
        from rdkit_cli.core.validate import MoleculeValidator

        validator = MoleculeValidator(max_rings=1)

        # Valid - 1 ring
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = validator.validate(mol)
        assert result.is_valid

        # Invalid - 2 rings
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        result = validator.validate(mol)
        assert not result.is_valid
        assert "Too many rings" in result.errors[0]

    def test_validate_allowed_elements(self):
        """Test allowed elements constraint."""
        from rdkit_cli.core.validate import MoleculeValidator

        validator = MoleculeValidator(allowed_elements={"C", "H", "O"})

        # Valid - only C, H, O
        mol = Chem.MolFromSmiles("CCO")
        result = validator.validate(mol)
        assert result.is_valid

        # Invalid - contains N
        mol = Chem.MolFromSmiles("CCN")
        result = validator.validate(mol)
        assert not result.is_valid
        assert "Disallowed elements" in result.errors[0]

    def test_validate_check_stereo(self):
        """Test stereochemistry checking."""
        from rdkit_cli.core.validate import MoleculeValidator

        validator = MoleculeValidator(check_stereo=True)

        # Check for stereo warnings
        mol = Chem.MolFromSmiles("CC(O)C")  # Has potential chiral center
        result = validator.validate(mol)
        # This is just a warning, not an error
        assert result.is_valid

    def test_validate_basic_info(self):
        """Test that basic info is included in result."""
        from rdkit_cli.core.validate import MoleculeValidator

        validator = MoleculeValidator()
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = validator.validate(mol)

        assert "num_atoms" in result.info
        assert "num_heavy_atoms" in result.info
        assert "num_rings" in result.info
        assert result.info["num_rings"] == 1

    def test_validate_to_dict(self):
        """Test converting result to dictionary."""
        from rdkit_cli.core.validate import MoleculeValidator

        validator = MoleculeValidator()
        mol = Chem.MolFromSmiles("CCO")
        result = validator.validate(mol)

        d = result.to_dict()
        assert "is_valid" in d
        assert "errors" in d
        assert "warnings" in d

    def test_validate_radical(self):
        """Test detection of radical electrons."""
        from rdkit_cli.core.validate import MoleculeValidator

        validator = MoleculeValidator(check_atoms=True)

        # Create molecule with radical
        mol = Chem.MolFromSmiles("[CH3]")
        result = validator.validate(mol)

        # Should have a warning about radical
        assert any("radical" in w.lower() for w in result.warnings)


class TestValidateSmiles:
    """Test validate_smiles function."""

    def test_valid_smiles(self):
        """Test with valid SMILES."""
        from rdkit_cli.core.validate import validate_smiles

        is_valid, mol, error = validate_smiles("CCO")
        assert is_valid
        assert mol is not None
        assert error == ""

    def test_invalid_smiles(self):
        """Test with invalid SMILES."""
        from rdkit_cli.core.validate import validate_smiles

        is_valid, mol, error = validate_smiles("not_a_smiles")
        assert not is_valid
        assert mol is None
        assert "Failed to parse" in error

    def test_empty_smiles(self):
        """Test with empty SMILES."""
        from rdkit_cli.core.validate import validate_smiles

        is_valid, mol, error = validate_smiles("")
        assert not is_valid
        assert mol is None
        assert "Empty" in error

    def test_whitespace_smiles(self):
        """Test with whitespace-only SMILES."""
        from rdkit_cli.core.validate import validate_smiles

        is_valid, mol, error = validate_smiles("   ")
        assert not is_valid
        assert mol is None


class TestValidationResult:
    """Test ValidationResult class."""

    def test_add_error(self):
        """Test adding errors."""
        from rdkit_cli.core.validate import ValidationResult

        result = ValidationResult()
        assert result.is_valid

        result.add_error("Test error")
        assert not result.is_valid
        assert "Test error" in result.errors

    def test_add_warning(self):
        """Test adding warnings."""
        from rdkit_cli.core.validate import ValidationResult

        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.is_valid  # Warnings don't invalidate
        assert "Test warning" in result.warnings

    def test_to_dict_formatting(self):
        """Test that errors/warnings are joined with semicolons."""
        from rdkit_cli.core.validate import ValidationResult

        result = ValidationResult()
        result.add_error("Error 1")
        result.add_error("Error 2")
        result.add_warning("Warning 1")

        d = result.to_dict()
        assert d["errors"] == "Error 1;Error 2"
        assert d["warnings"] == "Warning 1"
