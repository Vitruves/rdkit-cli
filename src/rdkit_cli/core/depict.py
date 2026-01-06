"""Molecular depiction/visualization engine."""

from typing import Optional, Any
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from rdkit_cli.io.readers import MoleculeRecord


class MoleculeDepiction:
    """Generate 2D depictions of molecules."""

    def __init__(
        self,
        width: int = 300,
        height: int = 300,
        image_format: str = "svg",
        add_atom_indices: bool = False,
        add_stereo_annotation: bool = False,
        highlight_atoms: Optional[list[int]] = None,
        highlight_bonds: Optional[list[int]] = None,
        use_kekulize: bool = True,
        wedge_bonds: bool = True,
        add_chiral_hs: bool = True,
    ):
        """
        Initialize molecule depiction.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            image_format: Output format ('svg' or 'png')
            add_atom_indices: Add atom index labels
            add_stereo_annotation: Add stereo annotations
            highlight_atoms: Atom indices to highlight
            highlight_bonds: Bond indices to highlight
            use_kekulize: Use Kekule form for drawing
            wedge_bonds: Draw wedged bonds
            add_chiral_hs: Add chiral Hs
        """
        self.width = width
        self.height = height
        self.image_format = image_format.lower()
        self.add_atom_indices = add_atom_indices
        self.add_stereo_annotation = add_stereo_annotation
        self.highlight_atoms = highlight_atoms or []
        self.highlight_bonds = highlight_bonds or []
        self.use_kekulize = use_kekulize
        self.wedge_bonds = wedge_bonds
        self.add_chiral_hs = add_chiral_hs

    def depict(self, mol: Chem.Mol) -> Optional[str]:
        """
        Generate depiction of a molecule.

        Args:
            mol: RDKit molecule

        Returns:
            SVG or PNG data as string/bytes
        """
        if mol is None:
            return None

        try:
            # Prepare molecule
            mol = Chem.Mol(mol)  # Copy
            if self.add_chiral_hs:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                mol = Chem.RemoveHs(mol)

            # Generate 2D coords
            rdDepictor.Compute2DCoords(mol)

            # Create drawer
            if self.image_format == "svg":
                drawer = rdMolDraw2D.MolDraw2DSVG(self.width, self.height)
            else:
                drawer = rdMolDraw2D.MolDraw2DCairo(self.width, self.height)

            # Configure options
            opts = drawer.drawOptions()
            opts.addAtomIndices = self.add_atom_indices
            opts.addStereoAnnotation = self.add_stereo_annotation

            # Draw
            if self.highlight_atoms or self.highlight_bonds:
                drawer.DrawMolecule(
                    mol,
                    highlightAtoms=self.highlight_atoms,
                    highlightBonds=self.highlight_bonds,
                )
            else:
                drawer.DrawMolecule(mol)

            drawer.FinishDrawing()

            return drawer.GetDrawingText()

        except Exception:
            return None

    def depict_record(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Generate depiction of a molecule record.

        Args:
            record: MoleculeRecord to process

        Returns:
            Dictionary with image data or None
        """
        if record.mol is None:
            return None

        image_data = self.depict(record.mol)
        if image_data is None:
            return None

        result: dict[str, Any] = {
            "smiles": record.smiles,
            "image": image_data,
        }

        if record.name:
            result["name"] = record.name

        return result


class GridDepiction:
    """Generate grid of molecule depictions."""

    def __init__(
        self,
        mols_per_row: int = 4,
        mol_width: int = 200,
        mol_height: int = 200,
        legends: Optional[list[str]] = None,
        use_svg: bool = True,
    ):
        """
        Initialize grid depiction.

        Args:
            mols_per_row: Molecules per row
            mol_width: Width per molecule
            mol_height: Height per molecule
            legends: List of labels for molecules
            use_svg: Output SVG instead of PNG
        """
        self.mols_per_row = mols_per_row
        self.mol_width = mol_width
        self.mol_height = mol_height
        self.legends = legends
        self.use_svg = use_svg

    def depict(self, mols: list[Chem.Mol]) -> Optional[str]:
        """
        Generate grid depiction.

        Args:
            mols: List of molecules

        Returns:
            SVG or PNG data
        """
        if not mols:
            return None

        try:
            # Prepare molecules
            prepared_mols = []
            for mol in mols:
                if mol is not None:
                    mol = Chem.Mol(mol)
                    rdDepictor.Compute2DCoords(mol)
                    prepared_mols.append(mol)
                else:
                    prepared_mols.append(None)

            legends = self.legends or [""] * len(prepared_mols)

            if self.use_svg:
                return Draw.MolsToGridImage(
                    prepared_mols,
                    molsPerRow=self.mols_per_row,
                    subImgSize=(self.mol_width, self.mol_height),
                    legends=legends[:len(prepared_mols)],
                    useSVG=True,
                )
            else:
                img = Draw.MolsToGridImage(
                    prepared_mols,
                    molsPerRow=self.mols_per_row,
                    subImgSize=(self.mol_width, self.mol_height),
                    legends=legends[:len(prepared_mols)],
                )
                # Convert to bytes
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

        except Exception:
            return None


def depict_smiles(
    smiles: str,
    width: int = 300,
    height: int = 300,
    image_format: str = "svg",
) -> Optional[str]:
    """
    Convenience function to depict a SMILES string.

    Args:
        smiles: SMILES string
        width: Image width
        height: Image height
        image_format: Output format

    Returns:
        Image data or None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    depictor = MoleculeDepiction(
        width=width,
        height=height,
        image_format=image_format,
    )

    return depictor.depict(mol)
