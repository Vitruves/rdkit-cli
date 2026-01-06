"""Molecule standardization engine."""

from typing import Optional, Any

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from rdkit_cli.io.readers import MoleculeRecord


class MoleculeStandardizer:
    """Standardizer for molecular structures."""

    def __init__(
        self,
        canonicalize: bool = True,
        remove_stereo: bool = False,
        disconnect_metals: bool = False,
        normalize: bool = False,
        reionize: bool = False,
        uncharge: bool = False,
        fragment_parent: bool = False,
        tautomer_parent: bool = False,
        include_original: bool = False,
    ):
        """
        Initialize standardizer.

        Args:
            canonicalize: Canonicalize SMILES
            remove_stereo: Remove stereochemistry information
            disconnect_metals: Disconnect metal atoms
            normalize: Apply normalization transforms
            reionize: Standardize ionization state
            uncharge: Neutralize charges
            fragment_parent: Keep only largest fragment
            tautomer_parent: Canonicalize tautomer
            include_original: Include original SMILES in output
        """
        self.canonicalize = canonicalize
        self.remove_stereo = remove_stereo
        self.disconnect_metals = disconnect_metals
        self.normalize = normalize
        self.reionize = reionize
        self.uncharge = uncharge
        self.fragment_parent = fragment_parent
        self.tautomer_parent = tautomer_parent
        self.include_original = include_original

        # Initialize standardizers
        self._metal_disconnector = rdMolStandardize.MetalDisconnector() if disconnect_metals else None
        self._normalizer = rdMolStandardize.Normalizer() if normalize else None
        self._reionizer = rdMolStandardize.Reionizer() if reionize else None
        self._uncharger = rdMolStandardize.Uncharger() if uncharge else None
        self._fragment_chooser = rdMolStandardize.LargestFragmentChooser() if fragment_parent else None
        self._tautomer_canon = rdMolStandardize.TautomerCanonicalizer() if tautomer_parent else None

    def standardize(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Standardize a molecule record.

        Args:
            record: MoleculeRecord to process

        Returns:
            Dictionary with standardized SMILES or None if failed
        """
        if record.mol is None:
            return None

        try:
            mol = record.mol

            # Apply transformations in order
            if self._metal_disconnector:
                mol = self._metal_disconnector.Disconnect(mol)

            if self._normalizer:
                mol = self._normalizer.normalize(mol)

            if self._reionizer:
                mol = self._reionizer.reionize(mol)

            if self._uncharger:
                mol = self._uncharger.uncharge(mol)

            if self._fragment_chooser:
                mol = self._fragment_chooser.choose(mol)

            if self._tautomer_canon:
                mol = self._tautomer_canon.canonicalize(mol)

            if self.remove_stereo:
                Chem.RemoveStereochemistry(mol)

            # Generate output SMILES
            if self.canonicalize:
                output_smiles = Chem.MolToSmiles(mol, canonical=True)
            else:
                output_smiles = Chem.MolToSmiles(mol)

            result: dict[str, Any] = {}

            if self.include_original:
                result["original_smiles"] = record.smiles

            result["smiles"] = output_smiles

            if record.name:
                result["name"] = record.name

            return result

        except Exception:
            return None

    def get_column_names(self) -> list[str]:
        """Get output column names in order."""
        cols = []
        if self.include_original:
            cols.append("original_smiles")
        cols.append("smiles")
        cols.append("name")
        return cols


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize a SMILES string.

    Args:
        smiles: Input SMILES

    Returns:
        Canonical SMILES or None if parsing failed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)
