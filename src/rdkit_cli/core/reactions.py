"""Reaction transformation engine."""

from typing import Optional, Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

from rdkit_cli.io.readers import MoleculeRecord


class ReactionTransformer:
    """Apply SMIRKS transformations to molecules."""

    def __init__(
        self,
        smirks: str,
        max_products: int = 100,
    ):
        """
        Initialize reaction transformer.

        Args:
            smirks: SMIRKS reaction pattern
            max_products: Maximum number of products to generate
        """
        self.reaction = AllChem.ReactionFromSmarts(smirks)
        if self.reaction is None:
            raise ValueError(f"Invalid SMIRKS pattern: {smirks}")

        self.max_products = max_products

    def transform(self, record: MoleculeRecord) -> Optional[dict[str, Any]]:
        """
        Apply transformation to a molecule.

        Args:
            record: MoleculeRecord to transform

        Returns:
            Dictionary with products or None if no reaction
        """
        if record.mol is None:
            return None

        try:
            products = self.reaction.RunReactants((record.mol,))

            if not products:
                return None

            # Collect unique products
            unique_smiles = set()
            for product_set in products[:self.max_products]:
                for prod in product_set:
                    try:
                        Chem.SanitizeMol(prod)
                        smi = Chem.MolToSmiles(prod)
                        unique_smiles.add(smi)
                    except Exception:
                        continue

            if not unique_smiles:
                return None

            # Return first product (or could return all)
            product_smiles = list(unique_smiles)[0]

            result: dict[str, Any] = {
                "smiles": product_smiles,
                "reactant": record.smiles,
                "num_products": len(unique_smiles),
            }

            if record.name:
                result["name"] = record.name

            return result

        except Exception:
            return None


class ReactionEnumerator:
    """Enumerate products from reaction templates."""

    def __init__(
        self,
        reaction_smarts: str,
        max_products: int = 1000,
    ):
        """
        Initialize reaction enumerator.

        Args:
            reaction_smarts: Reaction SMARTS
            max_products: Maximum products to generate
        """
        self.reaction = AllChem.ReactionFromSmarts(reaction_smarts)
        if self.reaction is None:
            raise ValueError(f"Invalid reaction SMARTS: {reaction_smarts}")

        self.max_products = max_products
        self.num_reactants = self.reaction.GetNumReactantTemplates()

    def enumerate(
        self,
        reactant_lists: list[list[Chem.Mol]],
    ) -> list[dict[str, Any]]:
        """
        Enumerate reaction products from lists of reactants.

        Args:
            reactant_lists: List of reactant lists (one per reactant template)

        Returns:
            List of product dictionaries
        """
        if len(reactant_lists) != self.num_reactants:
            raise ValueError(
                f"Expected {self.num_reactants} reactant lists, got {len(reactant_lists)}"
            )

        results = []
        unique_products = set()

        # Generate all combinations
        from itertools import product as iterproduct

        for reactants in iterproduct(*reactant_lists):
            if len(results) >= self.max_products:
                break

            try:
                products = self.reaction.RunReactants(reactants)

                for product_set in products:
                    for prod in product_set:
                        try:
                            Chem.SanitizeMol(prod)
                            smi = Chem.MolToSmiles(prod)

                            if smi not in unique_products:
                                unique_products.add(smi)
                                results.append({
                                    "smiles": smi,
                                    "reactants": ".".join(
                                        Chem.MolToSmiles(r) for r in reactants
                                    ),
                                })

                                if len(results) >= self.max_products:
                                    break
                        except Exception:
                            continue

            except Exception:
                continue

        return results
