"""Per-reaction-class protocol skeletons.

A skeleton declares the *shape* of a valid protocol for one reaction
class: which slots must be filled, what kind of value each holds,
typical equivalents/ranges, and the canonical workup sequence. The
composition layer in :mod:`eln_structurer.predict.composition` fills
the slots by voting across the channel evidence (exact match,
literature RAG, HTE corpus, LLM priors).

Skeletons are hand-encoded — small, finite (13 classes), and stable.
Updates ride alongside changes to :mod:`eln_structurer.reaction_class`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from eln_structurer.reaction_class import ReactionClass


SlotRole = Literal[
    "REACTANT", "REAGENT", "SOLVENT", "CATALYST", "INTERNAL_STANDARD",
]


@dataclass(frozen=True)
class Slot:
    """One named slot in a protocol skeleton.

    ``role`` is the ORD ReactionRole the slot will carry once filled.
    ``description`` tells the composer what kind of chemical to look
    for; ``typical_equiv_range`` is a soft guide used when the agent
    invents a fallback (no retrieval hits).
    """
    name: str
    role: SlotRole
    description: str
    typical_equiv_range: tuple[float, float] | None = None
    required: bool = True
    # Fallback values used when neither retrieval nor LLM proposes
    # anything plausible. Kept conservative — these are "what every
    # textbook would say"-level defaults.
    fallback_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProtocolSkeleton:
    """Class-conditional template for a protocol."""
    reaction_class: ReactionClass
    description: str
    slots: list[Slot]
    workup_sequence: list[str]
    # Soft hint on the typical reaction temperature in °C — used when
    # no retrieval evidence is available.
    typical_temperature_c: tuple[float, float] | None = None
    typical_duration_minutes: tuple[float, float] | None = None
    inert_atmosphere_required: bool = False
    extra_notes: str = ""


# ---------------------------------------------------------------------------
# The skeletons. One per reaction class the classifier recognises.
# ---------------------------------------------------------------------------


_SKELETONS: dict[ReactionClass, ProtocolSkeleton] = {
    ReactionClass.AMIDE_FORMATION: ProtocolSkeleton(
        reaction_class=ReactionClass.AMIDE_FORMATION,
        description="Carboxylic acid + amine → amide via peptide coupling.",
        slots=[
            Slot("carboxylic_acid", "REACTANT",
                 "the carboxylic acid partner",
                 typical_equiv_range=(1.0, 1.0)),
            Slot("amine", "REACTANT",
                 "the amine partner",
                 typical_equiv_range=(1.0, 1.2)),
            Slot("coupling_reagent", "REAGENT",
                 "peptide coupling reagent (EDC, HATU, DCC, T3P, ...)",
                 typical_equiv_range=(1.0, 1.5),
                 fallback_names=("EDC", "HATU")),
            Slot("base", "REAGENT",
                 "non-nucleophilic base (DIPEA, NMM, TEA)",
                 typical_equiv_range=(2.0, 4.0),
                 fallback_names=("DIPEA", "TEA")),
            Slot("solvent", "SOLVENT",
                 "polar aprotic solvent (DMF, DCM, NMP, MeCN)",
                 fallback_names=("DMF", "DCM")),
        ],
        workup_sequence=[
            "EXTRACTION", "WASH", "DRY_WITH_MATERIAL",
            "FILTRATION", "CONCENTRATION", "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(20.0, 30.0),
        typical_duration_minutes=(60.0, 720.0),
        inert_atmosphere_required=False,
    ),

    ReactionClass.SUZUKI_COUPLING: ProtocolSkeleton(
        reaction_class=ReactionClass.SUZUKI_COUPLING,
        description="Pd-catalysed C–C coupling: aryl halide + boronic acid.",
        slots=[
            Slot("aryl_halide", "REACTANT",
                 "aryl/heteroaryl halide partner",
                 typical_equiv_range=(1.0, 1.0)),
            Slot("boronic_partner", "REACTANT",
                 "boronic acid / ester / trifluoroborate",
                 typical_equiv_range=(1.1, 1.5)),
            Slot("pd_source", "CATALYST",
                 "Pd precatalyst (Pd(PPh3)4, Pd(OAc)2, Pd2(dba)3, PEPPSI)",
                 typical_equiv_range=(0.02, 0.1),
                 fallback_names=("Pd(PPh3)4",)),
            Slot("ligand", "REAGENT",
                 "phosphine ligand (XPhos, SPhos, BINAP, ...); required for Pd(OAc)2-style precats",
                 typical_equiv_range=(0.04, 0.2),
                 required=False),
            Slot("base", "REAGENT",
                 "inorganic base (K2CO3, Cs2CO3, K3PO4)",
                 typical_equiv_range=(2.0, 3.0),
                 fallback_names=("K2CO3",)),
            Slot("solvent", "SOLVENT",
                 "ether or dipolar aprotic (dioxane, THF, DMF, toluene; often + water)",
                 fallback_names=("1,4-dioxane",)),
        ],
        workup_sequence=[
            "EXTRACTION", "WASH", "DRY_WITH_MATERIAL",
            "FILTRATION", "CONCENTRATION", "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(80.0, 110.0),
        typical_duration_minutes=(240.0, 1440.0),
        inert_atmosphere_required=True,
    ),

    ReactionClass.BUCHWALD_HARTWIG: ProtocolSkeleton(
        reaction_class=ReactionClass.BUCHWALD_HARTWIG,
        description="Pd-catalysed C–N coupling: aryl halide + amine.",
        slots=[
            Slot("aryl_halide", "REACTANT",
                 "aryl/heteroaryl halide", typical_equiv_range=(1.0, 1.0)),
            Slot("amine", "REACTANT",
                 "primary or secondary amine partner",
                 typical_equiv_range=(1.0, 1.5)),
            Slot("pd_source", "CATALYST",
                 "Pd(OAc)2, Pd2(dba)3, Pd G3/G4 precatalyst",
                 typical_equiv_range=(0.01, 0.1),
                 fallback_names=("Pd(OAc)2",)),
            Slot("ligand", "REAGENT",
                 "biarylphosphine ligand (XPhos, BrettPhos, RuPhos, BINAP)",
                 typical_equiv_range=(0.02, 0.2),
                 fallback_names=("XPhos",)),
            Slot("base", "REAGENT",
                 "strong base (NaOtBu, Cs2CO3, K3PO4, LiHMDS)",
                 typical_equiv_range=(1.5, 3.0),
                 fallback_names=("Cs2CO3",)),
            Slot("solvent", "SOLVENT",
                 "anhydrous ether (dioxane, THF) or toluene",
                 fallback_names=("1,4-dioxane",)),
        ],
        workup_sequence=[
            "EXTRACTION", "WASH", "DRY_WITH_MATERIAL",
            "FILTRATION", "CONCENTRATION", "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(80.0, 110.0),
        typical_duration_minutes=(720.0, 1440.0),
        inert_atmosphere_required=True,
    ),

    ReactionClass.REDUCTIVE_AMINATION: ProtocolSkeleton(
        reaction_class=ReactionClass.REDUCTIVE_AMINATION,
        description="Aldehyde/ketone + amine → secondary amine via hydride reduction.",
        slots=[
            Slot("carbonyl", "REACTANT",
                 "aldehyde or ketone", typical_equiv_range=(1.0, 1.0)),
            Slot("amine", "REACTANT",
                 "primary or secondary amine partner",
                 typical_equiv_range=(1.0, 1.5)),
            Slot("reductant", "REAGENT",
                 "STAB or NaBH3CN (red-am-specific)",
                 typical_equiv_range=(1.2, 2.0),
                 fallback_names=("sodium triacetoxyborohydride",)),
            Slot("acid_catalyst", "REAGENT",
                 "weak acid catalyst (acetic acid)",
                 typical_equiv_range=(0.0, 1.0),
                 required=False),
            Slot("solvent", "SOLVENT",
                 "DCE, DCM, MeOH, or THF",
                 fallback_names=("dichloroethane",)),
        ],
        workup_sequence=[
            "ADDITION", "EXTRACTION", "WASH",
            "DRY_WITH_MATERIAL", "FILTRATION", "CONCENTRATION",
            "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(20.0, 30.0),
        typical_duration_minutes=(720.0, 1440.0),
    ),

    ReactionClass.GRIGNARD: ProtocolSkeleton(
        reaction_class=ReactionClass.GRIGNARD,
        description="In-situ Grignard formation + addition to electrophile.",
        slots=[
            Slot("alkyl_aryl_halide", "REACTANT",
                 "the alkyl or aryl halide", typical_equiv_range=(1.0, 2.0)),
            Slot("magnesium", "REACTANT",
                 "magnesium turnings",
                 typical_equiv_range=(1.0, 2.0),
                 fallback_names=("magnesium",)),
            Slot("electrophile", "REACTANT",
                 "carbonyl electrophile (aldehyde, ketone, ester, nitrile, ...)",
                 typical_equiv_range=(1.0, 1.0)),
            Slot("solvent", "SOLVENT",
                 "anhydrous ether (THF, Et2O, 2-MeTHF)",
                 fallback_names=("THF",)),
        ],
        workup_sequence=[
            "ADDITION", "EXTRACTION", "WASH",
            "DRY_WITH_MATERIAL", "FILTRATION", "CONCENTRATION",
            "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(-10.0, 66.0),
        typical_duration_minutes=(60.0, 240.0),
        inert_atmosphere_required=True,
        extra_notes=(
            "Use REFLUX control_type when the paragraph implies "
            "ether refluxing during Mg insertion."
        ),
    ),

    ReactionClass.REDUCTION: ProtocolSkeleton(
        reaction_class=ReactionClass.REDUCTION,
        description="Functional-group reduction (carbonyl, nitrile, nitro, ...).",
        slots=[
            Slot("substrate", "REACTANT",
                 "the substrate being reduced",
                 typical_equiv_range=(1.0, 1.0)),
            Slot("reductant", "REAGENT",
                 "NaBH4 / LiAlH4 / DIBAL / H2-Pd",
                 typical_equiv_range=(1.0, 2.0),
                 fallback_names=("NaBH4",)),
            Slot("solvent", "SOLVENT",
                 "MeOH/EtOH for NaBH4; THF/Et2O for LiAlH4/DIBAL",
                 fallback_names=("methanol",)),
        ],
        workup_sequence=[
            "ADDITION", "EXTRACTION", "WASH",
            "DRY_WITH_MATERIAL", "FILTRATION", "CONCENTRATION",
        ],
        typical_temperature_c=(-78.0, 30.0),
        typical_duration_minutes=(30.0, 240.0),
        inert_atmosphere_required=False,
    ),

    ReactionClass.OXIDATION: ProtocolSkeleton(
        reaction_class=ReactionClass.OXIDATION,
        description="Functional-group oxidation (alcohol → carbonyl, sulfide → sulfoxide, ...).",
        slots=[
            Slot("substrate", "REACTANT",
                 "the substrate being oxidised",
                 typical_equiv_range=(1.0, 1.0)),
            Slot("oxidant", "REAGENT",
                 "DMP, Swern reagent pair, PCC, mCPBA, KMnO4, TEMPO",
                 typical_equiv_range=(1.1, 2.0),
                 fallback_names=("Dess-Martin periodinane",)),
            Slot("buffer", "REAGENT",
                 "NaHCO3 buffer often used with DMP",
                 typical_equiv_range=(0.0, 3.0),
                 required=False),
            Slot("solvent", "SOLVENT",
                 "DCM is the most common; CHCl3 / MeCN for some reagents",
                 fallback_names=("DCM",)),
        ],
        workup_sequence=[
            "ADDITION", "EXTRACTION", "WASH",
            "DRY_WITH_MATERIAL", "FILTRATION", "CONCENTRATION",
            "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(-78.0, 30.0),
        typical_duration_minutes=(30.0, 240.0),
    ),

    ReactionClass.HALOGENATION: ProtocolSkeleton(
        reaction_class=ReactionClass.HALOGENATION,
        description="C–H halogenation or alpha-halogenation via NBS/NCS/Selectfluor/DAST.",
        slots=[
            Slot("substrate", "REACTANT",
                 "the substrate being halogenated",
                 typical_equiv_range=(1.0, 1.0)),
            Slot("halogenating_agent", "REAGENT",
                 "NBS, NCS, NIS, Selectfluor, DAST, Deoxofluor",
                 typical_equiv_range=(1.0, 1.5),
                 fallback_names=("NBS",)),
            Slot("initiator", "REAGENT",
                 "radical initiator (AIBN, light) for NBS benzylic brominations",
                 required=False),
            Slot("solvent", "SOLVENT",
                 "CCl4 (legacy) / DCM / MeCN; THF for Selectfluor",
                 fallback_names=("DCM",)),
        ],
        workup_sequence=[
            "FILTRATION", "EXTRACTION", "WASH",
            "DRY_WITH_MATERIAL", "CONCENTRATION", "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(0.0, 80.0),
        typical_duration_minutes=(60.0, 480.0),
    ),

    ReactionClass.BOC_DEPROTECTION: ProtocolSkeleton(
        reaction_class=ReactionClass.BOC_DEPROTECTION,
        description="Acidic removal of Boc / Cbz / Fmoc protecting groups.",
        slots=[
            Slot("protected_substrate", "REACTANT",
                 "the Boc/Cbz/Fmoc protected substrate",
                 typical_equiv_range=(1.0, 1.0)),
            Slot("deprotection_acid", "REAGENT",
                 "TFA, HCl in dioxane, or HCl in EtOAc",
                 typical_equiv_range=(5.0, 50.0),
                 fallback_names=("TFA",)),
            Slot("solvent", "SOLVENT",
                 "DCM (TFA), dioxane / EtOAc (HCl); MeOH for HCl",
                 fallback_names=("DCM",)),
        ],
        workup_sequence=[
            "CONCENTRATION", "ADDITION", "EXTRACTION",
            "DRY_WITH_MATERIAL", "FILTRATION", "CONCENTRATION",
        ],
        typical_temperature_c=(0.0, 30.0),
        typical_duration_minutes=(60.0, 240.0),
    ),

    ReactionClass.ESTERIFICATION: ProtocolSkeleton(
        reaction_class=ReactionClass.ESTERIFICATION,
        description="Carboxylic acid + alcohol → ester (Fischer, Steglich, ...).",
        slots=[
            Slot("carboxylic_acid", "REACTANT",
                 "the carboxylic acid partner",
                 typical_equiv_range=(1.0, 1.0)),
            Slot("alcohol", "REACTANT",
                 "the alcohol partner",
                 typical_equiv_range=(1.0, 5.0)),
            Slot("activator", "REAGENT",
                 "Steglich (DCC + DMAP), Fischer (H2SO4 / p-TsOH), or thionyl chloride",
                 typical_equiv_range=(0.05, 1.5),
                 fallback_names=("DCC",)),
            Slot("solvent", "SOLVENT",
                 "DCM for Steglich, toluene for Fischer (often neat in excess alcohol)",
                 fallback_names=("DCM",)),
        ],
        workup_sequence=[
            "FILTRATION", "EXTRACTION", "WASH",
            "DRY_WITH_MATERIAL", "CONCENTRATION", "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(20.0, 110.0),
        typical_duration_minutes=(240.0, 1440.0),
    ),

    ReactionClass.MITSUNOBU: ProtocolSkeleton(
        reaction_class=ReactionClass.MITSUNOBU,
        description="Mitsunobu coupling: nucleophile + alcohol → inversion product via DIAD/PPh3.",
        slots=[
            Slot("nucleophile", "REACTANT",
                 "acidic nucleophile (phenol, imide, sulfonamide, CO2H)",
                 typical_equiv_range=(1.0, 1.5)),
            Slot("alcohol", "REACTANT",
                 "the alcohol whose configuration is inverted",
                 typical_equiv_range=(1.0, 1.0)),
            Slot("diad", "REAGENT",
                 "DIAD or DEAD",
                 typical_equiv_range=(1.1, 1.5),
                 fallback_names=("DIAD",)),
            Slot("phosphine", "REAGENT",
                 "triphenylphosphine",
                 typical_equiv_range=(1.1, 1.5),
                 fallback_names=("PPh3",)),
            Slot("solvent", "SOLVENT",
                 "THF or toluene",
                 fallback_names=("THF",)),
        ],
        workup_sequence=[
            "CONCENTRATION", "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(0.0, 30.0),
        typical_duration_minutes=(60.0, 1440.0),
    ),

    ReactionClass.WITTIG: ProtocolSkeleton(
        reaction_class=ReactionClass.WITTIG,
        description="Wittig / HWE / Julia olefination: phosphonium ylide + carbonyl → alkene.",
        slots=[
            Slot("carbonyl", "REACTANT",
                 "aldehyde or ketone partner",
                 typical_equiv_range=(1.0, 1.0)),
            Slot("phosphonium_partner", "REACTANT",
                 "phosphonium salt / ylide / HWE phosphonate",
                 typical_equiv_range=(1.0, 1.5)),
            Slot("base", "REAGENT",
                 "n-BuLi / NaH / NaHMDS / LDA",
                 typical_equiv_range=(1.0, 1.5),
                 fallback_names=("n-BuLi",)),
            Slot("solvent", "SOLVENT",
                 "anhydrous THF or Et2O",
                 fallback_names=("THF",)),
        ],
        workup_sequence=[
            "ADDITION", "EXTRACTION", "WASH",
            "DRY_WITH_MATERIAL", "FILTRATION", "CONCENTRATION",
            "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(-78.0, 30.0),
        typical_duration_minutes=(60.0, 480.0),
        inert_atmosphere_required=True,
    ),

    ReactionClass.N_ALKYLATION: ProtocolSkeleton(
        reaction_class=ReactionClass.N_ALKYLATION,
        description="N-alkylation of an amine via alkyl halide / mesylate / tosylate.",
        slots=[
            Slot("amine", "REACTANT",
                 "the amine partner", typical_equiv_range=(1.0, 1.0)),
            Slot("electrophile", "REACTANT",
                 "alkyl halide / mesylate / tosylate / epoxide",
                 typical_equiv_range=(1.0, 2.0)),
            Slot("base", "REAGENT",
                 "K2CO3 / Cs2CO3 / TEA / DIPEA",
                 typical_equiv_range=(2.0, 3.0),
                 fallback_names=("K2CO3",)),
            Slot("solvent", "SOLVENT",
                 "MeCN, DMF, or DMSO",
                 fallback_names=("DMF",)),
        ],
        workup_sequence=[
            "EXTRACTION", "WASH", "DRY_WITH_MATERIAL",
            "FILTRATION", "CONCENTRATION", "FLASH_CHROMATOGRAPHY",
        ],
        typical_temperature_c=(20.0, 100.0),
        typical_duration_minutes=(240.0, 1440.0),
    ),
}


def get_skeleton(rcls: ReactionClass) -> ProtocolSkeleton | None:
    """Look up the skeleton for a reaction class; None if not defined."""
    return _SKELETONS.get(rcls)


def all_skeletons() -> list[ProtocolSkeleton]:
    """Return every defined skeleton — used when the classifier is unsure."""
    return list(_SKELETONS.values())


def known_classes() -> list[ReactionClass]:
    return list(_SKELETONS.keys())


__all__ = [
    "Slot",
    "ProtocolSkeleton",
    "get_skeleton",
    "all_skeletons",
    "known_classes",
]
