"""Quick molecule info module."""

from typing import Optional


def get_molecule_info(smiles: str, include_3d: bool = False) -> Optional[dict]:
    """
    Get comprehensive information about a molecule from its SMILES.

    Args:
        smiles: SMILES string
        include_3d: Whether to include 3D-related info

    Returns:
        Dictionary with molecule properties or None if parsing failed
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, inchi

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Basic info
    info = {
        "input_smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mol_weight": round(Descriptors.MolWt(mol), 4),
        "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
    }

    # InChI
    try:
        info["inchi"] = inchi.MolToInchi(mol)
        info["inchikey"] = inchi.MolToInchiKey(mol)
    except Exception:
        info["inchi"] = ""
        info["inchikey"] = ""

    # Counts
    info["heavy_atom_count"] = mol.GetNumHeavyAtoms()
    info["atom_count"] = mol.GetNumAtoms()
    info["bond_count"] = mol.GetNumBonds()
    info["ring_count"] = rdMolDescriptors.CalcNumRings(mol)
    info["aromatic_ring_count"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    info["rotatable_bond_count"] = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # H-bond
    info["hbd"] = rdMolDescriptors.CalcNumHBD(mol)
    info["hba"] = rdMolDescriptors.CalcNumHBA(mol)

    # Lipophilicity
    info["logp"] = round(Descriptors.MolLogP(mol), 4)
    info["tpsa"] = round(Descriptors.TPSA(mol), 4)

    # Fraction sp3
    info["fraction_csp3"] = round(rdMolDescriptors.CalcFractionCSP3(mol), 4)

    # Stereocenters
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    info["stereocenters"] = len(chiral_centers)
    info["undefined_stereocenters"] = sum(1 for _, s in chiral_centers if s == "?")

    # Formal charge
    info["formal_charge"] = Chem.GetFormalCharge(mol)

    # Element composition
    elements = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        elements[sym] = elements.get(sym, 0) + 1
    info["elements"] = elements

    # Lipinski violations
    violations = 0
    if info["mol_weight"] > 500:
        violations += 1
    if info["logp"] > 5:
        violations += 1
    if info["hbd"] > 5:
        violations += 1
    if info["hba"] > 10:
        violations += 1
    info["lipinski_violations"] = violations

    return info


def format_info_text(info: dict) -> str:
    """Format molecule info as human-readable text."""
    lines = []

    lines.append(f"SMILES:          {info['canonical_smiles']}")
    lines.append(f"Formula:         {info['formula']}")
    lines.append(f"Mol Weight:      {info['mol_weight']:.2f}")
    lines.append(f"Exact Mass:      {info['exact_mass']:.4f}")
    lines.append("")
    lines.append(f"InChI:           {info['inchi']}")
    lines.append(f"InChIKey:        {info['inchikey']}")
    lines.append("")
    lines.append(f"Heavy Atoms:     {info['heavy_atom_count']}")
    lines.append(f"Bonds:           {info['bond_count']}")
    lines.append(f"Rings:           {info['ring_count']}")
    lines.append(f"Aromatic Rings:  {info['aromatic_ring_count']}")
    lines.append(f"Rotatable Bonds: {info['rotatable_bond_count']}")
    lines.append("")
    lines.append(f"HB Donors:       {info['hbd']}")
    lines.append(f"HB Acceptors:    {info['hba']}")
    lines.append(f"LogP:            {info['logp']:.2f}")
    lines.append(f"TPSA:            {info['tpsa']:.2f}")
    lines.append(f"Frac. sp3:       {info['fraction_csp3']:.2f}")
    lines.append("")
    lines.append(f"Stereocenters:   {info['stereocenters']} ({info['undefined_stereocenters']} undefined)")
    lines.append(f"Formal Charge:   {info['formal_charge']}")
    lines.append(f"Lipinski Viol.:  {info['lipinski_violations']}")
    lines.append("")
    elem_str = ", ".join(f"{k}:{v}" for k, v in sorted(info['elements'].items()))
    lines.append(f"Elements:        {elem_str}")

    return "\n".join(lines)


def format_info_json(info: dict) -> str:
    """Format molecule info as JSON."""
    import json
    return json.dumps(info, indent=2)
