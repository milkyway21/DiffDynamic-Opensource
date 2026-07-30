"""Element-aware valence checking."""

from __future__ import annotations

from typing import List, Optional, Tuple

from rdkit import Chem

from .atom_mapping import bond_map_pair
from .models import StructureIssue

# Allowed explicit valence totals (bond order sum + formal charge adjustments handled separately)
# We use RDKit GetTotalValence when available; otherwise bond-order sum + explicit H.


def _bond_order_sum(atom: Chem.Atom) -> float:
    total = 0.0
    for bond in atom.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.BondType.AROMATIC:
            total += 1.5
        elif bt == Chem.BondType.SINGLE:
            total += 1.0
        elif bt == Chem.BondType.DOUBLE:
            total += 2.0
        elif bt == Chem.BondType.TRIPLE:
            total += 3.0
        elif bt == Chem.BondType.QUADRUPLE:
            total += 4.0
        else:
            try:
                total += float(bond.GetBondTypeAsDouble())
            except Exception:  # noqa: BLE001
                total += 1.0
    return total


def _explicit_h(atom: Chem.Atom) -> int:
    return int(atom.GetNumExplicitHs())


def _implicit_h_safe(atom: Chem.Atom) -> int:
    try:
        return int(atom.GetNumImplicitHs())
    except Exception:  # noqa: BLE001
        return 0


def expected_valence_range(atomic_num: int, formal_charge: int, aromatic: bool) -> Tuple[float, float]:
    """Return (min, max) acceptable total valence (bonds + H), charge-aware."""
    z = atomic_num
    fc = formal_charge

    if z == 6:  # C
        # C+: 3, C-: 3, neutral: 4
        if fc == 1:
            return (3.0, 3.0)
        if fc == -1:
            return (3.0, 3.0)
        return (4.0, 4.0)
    if z == 7:  # N
        if fc == 1:
            return (4.0, 4.0)  # ammonium / nitro N+
        if fc == -1:
            return (2.0, 2.0)
        if aromatic:
            return (3.0, 3.0)
        return (3.0, 3.0)
    if z == 8:  # O
        if fc == 1:
            return (3.0, 3.0)
        if fc == -1:
            return (1.0, 1.0)
        return (2.0, 2.0)
    if z in (9, 17, 35, 53):  # F Cl Br I
        if fc == 0:
            return (1.0, 1.0)
        if fc == -1:
            return (0.0, 0.0)
        return (1.0, 3.0)
    if z == 5:  # B
        if fc == -1:
            return (4.0, 4.0)
        return (3.0, 3.0)
    if z == 15:  # P
        return (3.0, 5.0)
    if z == 16:  # S
        return (2.0, 6.0)
    if z == 14:  # Si
        return (4.0, 4.0)
    # fallback
    return (1.0, 6.0)


def _issue_code_for(atomic_num: int, over: bool) -> str:
    names = {
        6: "CARBON",
        7: "NITROGEN",
        8: "OXYGEN",
        5: "BORON",
        15: "PHOSPHORUS",
        16: "SULFUR",
        9: "HALOGEN",
        17: "HALOGEN",
        35: "HALOGEN",
        53: "HALOGEN",
    }
    base = names.get(atomic_num, "UNEXPLAINED")
    if base == "UNEXPLAINED":
        return "UNEXPLAINED_VALENCE_STATE"
    if base in ("BORON", "PHOSPHORUS", "SULFUR"):
        return f"{base}_VALENCE_ERROR"
    if over:
        return f"{base}_OVERVALENT"
    return f"{base}_UNDERVALENT"


def check_atom_valence(atom: Chem.Atom) -> Optional[StructureIssue]:
    z = atom.GetAtomicNum()
    if z == 1:
        return None
    fc = atom.GetFormalCharge()
    aromatic = atom.GetIsAromatic()
    bond_sum = _bond_order_sum(atom)
    h_sum = _explicit_h(atom) + _implicit_h_safe(atom)
    # Prefer RDKit total valence when available (correct for aromatic heteroatoms)
    try:
        total = float(atom.GetTotalValence())
    except Exception:  # noqa: BLE001
        total = bond_sum + h_sum
    radicals = atom.GetNumRadicalElectrons()
    vmin, vmax = expected_valence_range(z, fc, aromatic)

    tol = 0.6 if aromatic else 0.05

    over = total > vmax + tol
    under = total < vmin - tol and radicals == 0

    # Special: nitro-like N with two double O and charge 0 is overvalent → charge issue
    if z == 7 and fc == 0 and bond_sum >= 4.5 and not aromatic:
        over = True
        total = bond_sum + h_sum

    # Aromatic heterocycles: do not flag when RDKit total valence is in range
    if aromatic and vmin - tol <= total <= vmax + tol:
        return None

    if not over and not under:
        return None

    mol = atom.GetOwningMol()
    amap = atom.GetAtomMapNum()
    bonds = []
    for bond in atom.GetBonds():
        bonds.append(bond_map_pair(mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    code = _issue_code_for(z, over=over)
    return StructureIssue(
        issue_code=code,
        severity="error",
        atom_map_ids=[amap],
        bond_atom_map_pairs=bonds,
        description=(
            f"Atom map {amap} ({atom.GetSymbol()}) valence={total:.2f} "
            f"expected [{vmin},{vmax}] charge={fc}"
        ),
        evidence={
            "atomic_num": z,
            "formal_charge": fc,
            "bond_order_sum": bond_sum,
            "explicit_h": _explicit_h(atom),
            "implicit_h": _implicit_h_safe(atom),
            "total_valence": total,
            "expected_min": vmin,
            "expected_max": vmax,
            "radicals": radicals,
            "aromatic": aromatic,
        },
        repairable=True,
        candidate_rule_ids=["VALENCE_REPAIR", "CHARGE_REPAIR", "FUNCTIONAL_GROUP_REPAIR"],
    )


def detect_valence_issues(mol: Chem.Mol) -> List[StructureIssue]:
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:  # noqa: BLE001
        pass
    issues: List[StructureIssue] = []
    for atom in mol.GetAtoms():
        issue = check_atom_valence(atom)
        if issue is not None:
            issues.append(issue)
    return issues
