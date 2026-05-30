"""Static DFT activation-energy table.

Lookup for the most common reaction class × substrate-class
combinations the predictor will see. Values are kJ/mol activation
barriers at the canonical condition (room temperature, standard
literature catalyst). They're meant as a sanity-check signal, not a
quantitative prediction — "this substrate class is harder than that
one" — and the predictor uses them to nudge the ranker, not replace
it.

The table is small, hand-maintained, and conservative: when a key
isn't present we return ``None`` and the ranker reads "no DFT
signal" instead of defaulting to optimism.
"""

from __future__ import annotations

from eln_structurer.reaction_class import ReactionClass


# (reaction_class, substrate_descriptor_key) → activation Ea (kJ/mol)
# substrate_descriptor_key is a free-text tag the composer or agent
# emits; we keep a small set of canonical tags so adding new entries
# is straightforward.
_DFT_EA: dict[tuple[ReactionClass, str], float] = {
    # Suzuki couplings: aryl bromide easiest, aryl chloride harder,
    # heteroaryl chloride hardest.
    (ReactionClass.SUZUKI_COUPLING, "aryl_bromide"):     63.0,
    (ReactionClass.SUZUKI_COUPLING, "aryl_iodide"):      54.0,
    (ReactionClass.SUZUKI_COUPLING, "aryl_chloride"):    81.0,
    (ReactionClass.SUZUKI_COUPLING, "heteroaryl_chloride"): 90.0,

    # Buchwald-Hartwig couplings: similar trend.
    (ReactionClass.BUCHWALD_HARTWIG, "aryl_bromide"):    71.0,
    (ReactionClass.BUCHWALD_HARTWIG, "aryl_chloride"):   89.0,
    (ReactionClass.BUCHWALD_HARTWIG, "aryl_triflate"):   60.0,

    # Amide couplings: HATU is fast; DCC requires more enthalpy.
    (ReactionClass.AMIDE_FORMATION, "hatu"):             45.0,
    (ReactionClass.AMIDE_FORMATION, "edc"):              52.0,
    (ReactionClass.AMIDE_FORMATION, "dcc"):              58.0,

    # Reductive amination: STAB faster than NaBH3CN.
    (ReactionClass.REDUCTIVE_AMINATION, "stab"):         50.0,
    (ReactionClass.REDUCTIVE_AMINATION, "nabh3cn"):      57.0,

    # Grignard formation: ether reflux is the rate-limiting step.
    (ReactionClass.GRIGNARD, "primary_alkyl_bromide"):   45.0,
    (ReactionClass.GRIGNARD, "primary_alkyl_chloride"):  68.0,
    (ReactionClass.GRIGNARD, "aryl_bromide"):            55.0,

    # NaBH4 / LAH / DIBAL ranges for reductions.
    (ReactionClass.REDUCTION, "nabh4_aldehyde"):         42.0,
    (ReactionClass.REDUCTION, "nabh4_ketone"):           48.0,
    (ReactionClass.REDUCTION, "lialh4_ester"):           55.0,
    (ReactionClass.REDUCTION, "dibal_nitrile"):          62.0,

    # DMP and Swern oxidations.
    (ReactionClass.OXIDATION, "dmp_primary_alcohol"):    50.0,
    (ReactionClass.OXIDATION, "dmp_secondary_alcohol"):  55.0,
    (ReactionClass.OXIDATION, "swern_primary_alcohol"):  53.0,

    # Boc deprotection with TFA / HCl.
    (ReactionClass.BOC_DEPROTECTION, "tfa_dcm"):         38.0,
    (ReactionClass.BOC_DEPROTECTION, "hcl_dioxane"):     42.0,

    # Mitsunobu activation step.
    (ReactionClass.MITSUNOBU, "phenol_carboxylic_acid"): 60.0,

    # Wittig + HWE.
    (ReactionClass.WITTIG, "stabilised_ylide"):          50.0,
    (ReactionClass.WITTIG, "unstabilised_ylide"):        58.0,
    (ReactionClass.WITTIG, "hwe_phosphonate"):           55.0,

    # Esterification (Fischer; Steglich).
    (ReactionClass.ESTERIFICATION, "fischer_h2so4"):     70.0,
    (ReactionClass.ESTERIFICATION, "steglich_dcc_dmap"): 50.0,

    # Halogenation (NBS / NCS).
    (ReactionClass.HALOGENATION, "nbs_benzylic"):        48.0,
    (ReactionClass.HALOGENATION, "ncs_aromatic"):        55.0,
    (ReactionClass.HALOGENATION, "selectfluor_alpha"):   62.0,

    # N-alkylation.
    (ReactionClass.N_ALKYLATION, "primary_amine_alkyl_bromide"): 60.0,
    (ReactionClass.N_ALKYLATION, "secondary_amine_alkyl_iodide"): 50.0,
}


def lookup_activation_energy(
    reaction_class: ReactionClass, substrate_tag: str,
) -> float | None:
    """Return Ea in kJ/mol or None when not tabulated."""
    return _DFT_EA.get((reaction_class, substrate_tag.lower()))


def known_substrate_tags(reaction_class: ReactionClass) -> list[str]:
    """Surface the tags the table knows for a given class — useful so
    the composer can suggest valid substrate descriptors back to the
    agent."""
    return [tag for (cls, tag) in _DFT_EA if cls is reaction_class]


__all__ = ["lookup_activation_energy", "known_substrate_tags"]
