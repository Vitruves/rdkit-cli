"""Stereochemistry analysis and manipulation engine."""

from typing import Any

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from rdkit_cli.io.readers import MoleculeRecord


def assign_cip_labels(mol: Chem.Mol) -> list[dict[str, Any]]:
    """Assign CIP labels (R/S, E/Z) to a molecule."""
    rdCIPLabeler.AssignCIPLabels(mol)
    labels = []
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPCode"):
            labels.append({
                "idx": atom.GetIdx(),
                "symbol": atom.GetSymbol(),
                "cip": atom.GetProp("_CIPCode"),
            })
    for bond in mol.GetBonds():
        if bond.HasProp("_CIPCode"):
            labels.append({
                "idx": bond.GetIdx(),
                "type": "bond",
                "begin": bond.GetBeginAtomIdx(),
                "end": bond.GetEndAtomIdx(),
                "cip": bond.GetProp("_CIPCode"),
            })
    return labels


def perceive_stereo(mol: Chem.Mol) -> list[dict[str, Any]]:
    """Find potential stereocenters and stereobonds."""
    info = Chem.FindPotentialStereo(mol)
    results = []
    for si in info:
        results.append({
            "type": str(si.type),
            "centered_on": si.centeredOn,
            "specified": str(si.specified),
            "descriptor": str(si.descriptor) if si.descriptor else "",
        })
    return results


def get_enhanced_stereo(mol: Chem.Mol) -> list[dict[str, Any]]:
    """Get enhanced stereo group information."""
    groups = []
    for sg in mol.GetStereoGroups():
        atoms = [a.GetIdx() for a in sg.GetAtoms()]
        groups.append({
            "type": str(sg.GetGroupType()),
            "atoms": atoms,
        })
    return groups


class StereoAssigner:
    """Assign and report CIP labels for molecules."""

    def assign(self, record: MoleculeRecord):
        if record.mol is None:
            return None

        labels = assign_cip_labels(record.mol)
        cip_str = "; ".join(
            f"{l['symbol']}@{l['idx']}={l['cip']}"
            for l in labels if "symbol" in l
        )
        result = {
            "smiles": record.smiles,
            "cip_labels": cip_str,
            "num_stereocenters": len(
                [l for l in labels if "symbol" in l]
            ),
        }
        if record.name:
            result["name"] = record.name
        return result


class StereoPerceiver:
    """Perceive potential stereocenters."""

    def perceive(self, record: MoleculeRecord):
        if record.mol is None:
            return None

        info = perceive_stereo(record.mol)
        n_specified = sum(
            1 for i in info if "Specified" in i["specified"]
        )
        n_unspecified = sum(
            1 for i in info if "Unspecified" in i["specified"]
        )
        result = {
            "smiles": record.smiles,
            "num_potential": len(info),
            "num_specified": n_specified,
            "num_unspecified": n_unspecified,
        }
        if record.name:
            result["name"] = record.name
        return result
