"""Reaction fingerprints + Tanimoto similarity.

A reaction is encoded as ``ECFP(products) − ECFP(reactants)``: the
product-side ECFP bit vector minus the reactant-side, expressed as a
sparse counter. Two reactions of the same class score high Tanimoto on
this representation. It's the simplest fingerprint that survives the
"different solvent / catalyst, same bond-forming step" test.

This is deliberately implemented with only RDKit primitives we already
ship — no `rxnfp` torch model, no `DRFP` install. The API
(:func:`reaction_fingerprint`, :func:`tanimoto`) is stable; we can swap
in a smarter backend in Tier 2 without changing callers.

If parsing fails on either side, the fingerprint is an empty Counter —
the retrieval layer treats that as "no match" rather than crashing.
"""

from __future__ import annotations

from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem


def _smi_to_mol(smi: str) -> Chem.Mol | None:
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:  # pragma: no cover — defensive, matches chemistry.parse_mol
        return None


def _side_ecfp_counter(side_smiles: str, radius: int = 2, n_bits: int = 2048) -> Counter:
    """ECFP bit counter for one side of a reaction (multi-component '.').

    The side is the dot-joined SMILES of every reactant (or every product).
    Each individual molecule contributes its bits; we sum across molecules
    so a side with two reactants is the bag-of-bits of both.
    """
    counter: Counter = Counter()
    for fragment in side_smiles.split("."):
        if not fragment:
            continue
        mol = _smi_to_mol(fragment)
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        for i in bv.GetOnBits():
            counter[i] += 1
    return counter


def reaction_fingerprint(rxn_smiles: str, *, radius: int = 2, n_bits: int = 2048) -> Counter:
    """Difference fingerprint: product ECFP counts minus reactant ECFP counts.

    Reagents (between the two ``>``) are deliberately ignored — the
    user's question is about the bond-forming transformation; reagent
    choice is a recommendation we produce in the next layer, not a
    retrieval key.
    """
    if rxn_smiles.count(">") != 2:
        return Counter()
    left, _middle, right = rxn_smiles.split(">")
    left_c = _side_ecfp_counter(left, radius=radius, n_bits=n_bits)
    right_c = _side_ecfp_counter(right, radius=radius, n_bits=n_bits)

    diff: Counter = Counter()
    for k, v in right_c.items():
        delta = v - left_c.get(k, 0)
        if delta:
            diff[k] = delta
    for k, v in left_c.items():
        if k not in right_c:
            diff[k] = -v
    return diff


def tanimoto(a: Counter, b: Counter) -> float:
    """Generalised Tanimoto on signed-integer counters.

    For two empty counters we return 0.0 (genuinely no signal), not 1.0
    (degenerate self-match) — the retrieval layer treats unparseable
    reactions as misses.
    """
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    intersection = sum(min(abs(a.get(k, 0)), abs(b.get(k, 0))) for k in keys)
    union = sum(max(abs(a.get(k, 0)), abs(b.get(k, 0))) for k in keys)
    return intersection / union if union else 0.0
