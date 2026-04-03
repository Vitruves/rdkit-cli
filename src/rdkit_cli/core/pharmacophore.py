"""Pharmacophore feature perception and matching engine."""

from typing import Any

from rdkit import Chem
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

from rdkit_cli.io.readers import MoleculeRecord

# Feature families recognized by Gobbi factory
FEATURE_FAMILIES = [
    "HD", "HA", "AR", "AG", "BG", "LH", "RR", "X",
]

FAMILY_LABELS = {
    "HD": "Donor",
    "HA": "Acceptor",
    "AR": "Aromatic",
    "AG": "Anion",
    "BG": "Cation",
    "LH": "LumpedHydrophobe",
    "RR": "RingRing",
    "X": "Halogen",
}


def perceive_features(mol: Chem.Mol) -> list[dict[str, Any]]:
    """Perceive pharmacophore features in a molecule."""
    feat_factory = Gobbi_Pharm2D.factory.featFactory
    feats = feat_factory.GetFeaturesForMol(mol)
    results = []
    for feat in feats:
        results.append({
            "family": feat.GetFamily(),
            "type": feat.GetType(),
            "atoms": list(feat.GetAtomIds()),
        })
    return results


class PharmacophorePerceiver:
    """Perceive pharmacophoric features for molecules."""

    def perceive(self, record: MoleculeRecord):
        if record.mol is None:
            return None

        try:
            feats = perceive_features(record.mol)

            # Count by family
            family_counts = {}
            for f in feats:
                fam = f["family"]
                family_counts[fam] = family_counts.get(fam, 0) + 1

            result = {
                "smiles": record.smiles,
                "num_features": len(feats),
            }
            if record.name:
                result["name"] = record.name

            for fam in FEATURE_FAMILIES:
                result[f"n_{fam}"] = family_counts.get(fam, 0)

            return result
        except Exception:
            return None


class PharmacophoreSearcher:
    """Search molecules by 2D pharmacophore fingerprint similarity."""

    def __init__(
        self,
        query_smiles: str,
        threshold: float = 0.5,
    ):
        from rdkit import DataStructs

        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            raise ValueError(f"Invalid query SMILES: {query_smiles}")

        self.query_fp = Generate.Gen2DFingerprint(
            query_mol, Gobbi_Pharm2D.factory,
        )
        self.threshold = threshold
        self._DataStructs = DataStructs

    def search(self, record: MoleculeRecord):
        if record.mol is None:
            return None

        try:
            fp = Generate.Gen2DFingerprint(
                record.mol, Gobbi_Pharm2D.factory,
            )
            sim = self._DataStructs.TanimotoSimilarity(
                self.query_fp, fp,
            )

            if sim < self.threshold:
                return None

            result = {
                "smiles": record.smiles,
                "pharmacophore_similarity": round(sim, 4),
            }
            if record.name:
                result["name"] = record.name
            return result
        except Exception:
            return None
