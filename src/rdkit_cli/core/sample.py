"""Random sampling engine for molecular datasets."""

import random
from typing import Optional

from rdkit_cli.io.readers import MoleculeRecord


class MoleculeSampler:
    """Randomly sample molecules from a dataset."""

    def __init__(
        self,
        n: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
        stratify_valid: bool = False,
    ):
        """
        Initialize sampler.

        Args:
            n: Number of molecules to sample
            fraction: Fraction of molecules to sample (0.0-1.0)
            seed: Random seed for reproducibility
            stratify_valid: If True, maintain valid/invalid ratio in sample

        Note: Exactly one of n or fraction must be specified.
        """
        if n is None and fraction is None:
            raise ValueError("Either n or fraction must be specified")
        if n is not None and fraction is not None:
            raise ValueError("Only one of n or fraction can be specified")
        if fraction is not None and not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must be between 0 and 1")

        self.n = n
        self.fraction = fraction
        self.seed = seed
        self.stratify_valid = stratify_valid

    def sample(self, records: list[MoleculeRecord]) -> list[MoleculeRecord]:
        """
        Sample records from the dataset.

        Args:
            records: List of molecule records

        Returns:
            Sampled list of records
        """
        if not records:
            return []

        # Set random seed
        if self.seed is not None:
            random.seed(self.seed)

        # Calculate sample size
        if self.n is not None:
            sample_size = min(self.n, len(records))
        else:
            sample_size = int(len(records) * self.fraction)
            sample_size = max(1, sample_size)  # At least 1

        if self.stratify_valid:
            return self._stratified_sample(records, sample_size)
        else:
            return random.sample(records, sample_size)

    def _stratified_sample(
        self,
        records: list[MoleculeRecord],
        sample_size: int,
    ) -> list[MoleculeRecord]:
        """
        Sample while maintaining valid/invalid ratio.

        Args:
            records: List of molecule records
            sample_size: Number of records to sample

        Returns:
            Stratified sample of records
        """
        valid = [r for r in records if r.is_valid]
        invalid = [r for r in records if not r.is_valid]

        if not valid or not invalid:
            # Can't stratify, just do regular sample
            return random.sample(records, sample_size)

        # Calculate proportional sizes
        valid_ratio = len(valid) / len(records)
        n_valid = int(sample_size * valid_ratio)
        n_invalid = sample_size - n_valid

        # Ensure we don't exceed available
        n_valid = min(n_valid, len(valid))
        n_invalid = min(n_invalid, len(invalid))

        # Sample from each group
        sampled_valid = random.sample(valid, n_valid)
        sampled_invalid = random.sample(invalid, n_invalid)

        # Combine and shuffle
        result = sampled_valid + sampled_invalid
        random.shuffle(result)
        return result


class ReservoirSampler:
    """
    Stream-based sampling using reservoir sampling.

    Useful for sampling from large files without loading all records.
    """

    def __init__(self, n: int, seed: Optional[int] = None):
        """
        Initialize reservoir sampler.

        Args:
            n: Number of items to sample
            seed: Random seed for reproducibility
        """
        self.n = n
        self.seed = seed
        self._reservoir: list[MoleculeRecord] = []
        self._count = 0
        self._rng = random.Random(seed)

    def add(self, record: MoleculeRecord) -> None:
        """
        Add a record to the reservoir.

        Args:
            record: Molecule record to potentially include
        """
        if len(self._reservoir) < self.n:
            self._reservoir.append(record)
        else:
            # Replace with decreasing probability
            j = self._rng.randint(0, self._count)
            if j < self.n:
                self._reservoir[j] = record
        self._count += 1

    def get_sample(self) -> list[MoleculeRecord]:
        """
        Get the current sample.

        Returns:
            List of sampled records
        """
        return list(self._reservoir)
