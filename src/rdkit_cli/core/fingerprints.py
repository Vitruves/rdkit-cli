"""Molecular fingerprint computation engine."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

from rdkit_cli.io.readers import MoleculeRecord


class FingerprintType(Enum):
    """Supported fingerprint types."""

    MORGAN = "morgan"
    MACCS = "maccs"
    RDKIT = "rdkit"
    ATOMPAIR = "atompair"
    TORSION = "torsion"
    PATTERN = "pattern"


@dataclass
class FingerprintInfo:
    """Information about a fingerprint type."""

    name: str
    description: str
    default_bits: int
    has_radius: bool


FINGERPRINT_INFO: dict[FingerprintType, FingerprintInfo] = {
    FingerprintType.MORGAN: FingerprintInfo(
        name="morgan",
        description="Morgan/ECFP circular fingerprints",
        default_bits=2048,
        has_radius=True,
    ),
    FingerprintType.MACCS: FingerprintInfo(
        name="maccs",
        description="MACCS structural keys (166 bits)",
        default_bits=167,
        has_radius=False,
    ),
    FingerprintType.RDKIT: FingerprintInfo(
        name="rdkit",
        description="RDKit/Daylight-like path-based fingerprints",
        default_bits=2048,
        has_radius=False,
    ),
    FingerprintType.ATOMPAIR: FingerprintInfo(
        name="atompair",
        description="Atom pair fingerprints",
        default_bits=2048,
        has_radius=False,
    ),
    FingerprintType.TORSION: FingerprintInfo(
        name="torsion",
        description="Topological torsion fingerprints",
        default_bits=2048,
        has_radius=False,
    ),
    FingerprintType.PATTERN: FingerprintInfo(
        name="pattern",
        description="SMARTS pattern fingerprints (for screening)",
        default_bits=2048,
        has_radius=False,
    ),
}


def list_fingerprints() -> list[FingerprintInfo]:
    """List available fingerprint types."""
    return list(FINGERPRINT_INFO.values())


def compute_fingerprint(
    mol: Chem.Mol,
    fp_type: FingerprintType,
    n_bits: int = 2048,
    radius: int = 2,
    use_counts: bool = False,
) -> Optional[DataStructs.ExplicitBitVect]:
    """
    Compute fingerprint for a molecule.

    Args:
        mol: RDKit molecule
        fp_type: Type of fingerprint
        n_bits: Number of bits
        radius: Radius for Morgan fingerprints
        use_counts: Use count fingerprints (Morgan only)

    Returns:
        Fingerprint bit vector or None on failure
    """
    try:
        if fp_type == FingerprintType.MORGAN:
            if use_counts:
                return rdMolDescriptors.GetHashedMorganFingerprint(
                    mol, radius, nBits=n_bits
                )
            else:
                return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius, nBits=n_bits
                )

        elif fp_type == FingerprintType.MACCS:
            return MACCSkeys.GenMACCSKeys(mol)

        elif fp_type == FingerprintType.RDKIT:
            return Chem.RDKFingerprint(mol, fpSize=n_bits)

        elif fp_type == FingerprintType.ATOMPAIR:
            return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=n_bits
            )

        elif fp_type == FingerprintType.TORSION:
            return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol, nBits=n_bits
            )

        elif fp_type == FingerprintType.PATTERN:
            return Chem.PatternFingerprint(mol, fpSize=n_bits)

        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

    except Exception:
        return None


def fingerprint_to_hex(fp) -> str:
    """Convert fingerprint to hex string."""
    if fp is None:
        return ""

    if hasattr(fp, "GetNonzeroElements"):
        # Count fingerprint - convert to bit vector first
        bit_string = fp.ToBitString()
        return hex(int(bit_string, 2))[2:]

    # Bit vector
    return fp.ToBase64()


def fingerprint_to_bitstring(fp) -> str:
    """Convert fingerprint to bit string."""
    if fp is None:
        return ""
    return fp.ToBitString()


def fingerprint_to_numpy(fp):
    """Convert fingerprint to numpy array."""
    import numpy as np

    if fp is None:
        return None

    arr = np.zeros((len(fp),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class FingerprintCalculator:
    """Calculator for molecular fingerprints."""

    def __init__(
        self,
        fp_type: FingerprintType = FingerprintType.MORGAN,
        n_bits: int = 2048,
        radius: int = 2,
        use_counts: bool = False,
        output_format: str = "hex",
        include_smiles: bool = True,
        include_name: bool = True,
    ):
        """
        Initialize fingerprint calculator.

        Args:
            fp_type: Type of fingerprint
            n_bits: Number of bits
            radius: Radius for Morgan fingerprints
            use_counts: Use count fingerprints
            output_format: Output format (hex, bitstring, bits)
            include_smiles: Include SMILES in output
            include_name: Include molecule name in output
        """
        self.fp_type = fp_type
        self.n_bits = n_bits
        self.radius = radius
        self.use_counts = use_counts
        self.output_format = output_format
        self.include_smiles = include_smiles
        self.include_name = include_name

        # Override n_bits for MACCS
        if fp_type == FingerprintType.MACCS:
            self.n_bits = 167

    def compute(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Compute fingerprint for a molecule record.

        Args:
            record: MoleculeRecord to process

        Returns:
            Dictionary with fingerprint or None if molecule is invalid
        """
        if record.mol is None:
            return None

        fp = compute_fingerprint(
            record.mol,
            self.fp_type,
            n_bits=self.n_bits,
            radius=self.radius,
            use_counts=self.use_counts,
        )

        if fp is None:
            return None

        result: dict[str, Any] = {}

        if self.include_smiles:
            result["smiles"] = record.smiles
        if self.include_name and record.name:
            result["name"] = record.name

        # Format fingerprint
        if self.output_format == "hex":
            result["fingerprint"] = fingerprint_to_hex(fp)
        elif self.output_format == "bitstring":
            result["fingerprint"] = fingerprint_to_bitstring(fp)
        elif self.output_format == "bits":
            # Individual bit columns
            bits = fingerprint_to_bitstring(fp)
            for i, bit in enumerate(bits):
                result[f"bit_{i}"] = int(bit)
        else:
            result["fingerprint"] = fingerprint_to_hex(fp)

        return result

    def get_column_names(self) -> list[str]:
        """Get output column names in order."""
        cols = []
        if self.include_smiles:
            cols.append("smiles")
        if self.include_name:
            cols.append("name")

        if self.output_format == "bits":
            cols.extend([f"bit_{i}" for i in range(self.n_bits)])
        else:
            cols.append("fingerprint")

        return cols
