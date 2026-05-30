"""Top-level predictor entry point.

Wires every Tier 1-4 primitive into a single deterministic call:

    propose_protocol(rxn_smiles, *, constraints, corpus, weights) -> Output

The function takes a target reaction SMILES and optional user
constraints / corpus / weights, runs the appropriate channels (exact
match, K-NN, class skeleton), composes one candidate per matching
skeleton, screens for safety, and returns them ranked. The agent loop
(Tier 5b in the plan; see ``agent_predict.py``) wraps this with LLM
critic + repair; this module stays LLM-free for cheap, fast testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eln_structurer.predict.composition import (
    ProposalResult,
    compose_protocol,
)
from eln_structurer.predict.corpus import LocalCorpus
from eln_structurer.predict.hte_corpus import default_seed_corpus
from eln_structurer.predict.ranker import RankedProposal, Weights, rank_proposals
from eln_structurer.predict.retrieval import (
    Hit,
    retrieve_exact,
    retrieve_knn,
)
from eln_structurer.predict.risks import (
    SafetyVerdict,
    hard_constraint_filter,
    recency_summary,
    safety_screen,
)
from eln_structurer.predict.skeleton import (
    ProtocolSkeleton,
    all_skeletons,
    get_skeleton,
)
from eln_structurer.reaction_class import (
    ClassificationResult,
    ReactionClass,
    classify_reaction,
)


@dataclass
class PredictorOutput:
    """Container the predictor returns to its caller.

    ``ranked_proposals`` is best-first. Each carries the full reasoning
    trail (channel reports, slot provenance, score breakdown). The
    ``warnings`` list aggregates corpus-wide concerns (staleness, no
    classifier confidence, etc.). ``safety_blocked`` is True iff any
    candidate hit the BLOCKED verdict; those candidates are excluded
    from the ranking.
    """
    target_reaction_smiles: str
    classification: ClassificationResult | None
    ranked_proposals: list[RankedProposal]
    warnings: list[str] = field(default_factory=list)
    safety_blocked_proposals: list[ProposalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: classify the target reaction without a draft
# ---------------------------------------------------------------------------


def _classify_target(rxn_smiles: str) -> ClassificationResult | None:
    """Classify a *target* reaction by composing a stub draft.

    We don't have a full ReactionDraft for the target — we only have the
    SMILES. Building a minimal draft just to feed the classifier is
    cheap and the classifier's name-pattern matchers don't care about
    the rest of the ORD structure. When the target reaction has no
    extracted compound names we return None and the caller falls back
    to running every skeleton.
    """
    try:
        from eln_structurer.schema import (
            AmountModel,
            CompoundIdentifierModel,
            CompoundModel,
            ConditionsModel,
            OutcomeModel,
            ProductModel,
            ReactionDraft,
            ReactionInputModel,
            TemperatureModel,
        )
    except Exception:                  # pragma: no cover - defensive
        return None

    if rxn_smiles.count(">") != 2:
        return None
    left, _middle, right = rxn_smiles.split(">")
    if not left or not right:
        return None

    inputs = []
    for i, smi in enumerate([s for s in left.split(".") if s]):
        inputs.append(ReactionInputModel(
            name=f"reactant_{i}",
            components=[CompoundModel(
                identifiers=[CompoundIdentifierModel(type="SMILES", value=smi),
                             CompoundIdentifierModel(type="NAME", value=smi)],
                amount=AmountModel(value=1.0, units="mmol"),
                reaction_role="REACTANT",
                is_limiting=(i == 0),
            )],
        ))
    if not inputs:
        return None
    products = [
        ProductModel(compound=CompoundModel(
            identifiers=[CompoundIdentifierModel(type="SMILES", value=s)],
            reaction_role="PRODUCT",
        ))
        for s in right.split(".") if s
    ]
    draft = ReactionDraft(
        identifiers=[CompoundIdentifierModel(type="SMILES", value=rxn_smiles)],
        inputs=inputs,
        conditions=ConditionsModel(temperature=TemperatureModel(control_type="AMBIENT")),
        outcomes=[OutcomeModel(products=products)],
        notes="(classifier stub)",
        source_paragraph="(predictor target)",
    )
    return classify_reaction(draft)


# ---------------------------------------------------------------------------
# Per-skeleton candidate construction
# ---------------------------------------------------------------------------


def _propose_with_skeleton(
    *,
    rxn_smiles: str,
    skeleton: ProtocolSkeleton,
    corpus: LocalCorpus,
    constraints: dict[str, Any] | None,
) -> tuple[ProposalResult, dict[str, list[Hit]]]:
    filters = hard_constraint_filter(constraints)
    # Apply hard constraints to exact matches too — an "exact same
    # reaction known" record that uses a banned solvent is still
    # banned. Risk-table rationale: filter at query time, not rank time.
    exact = retrieve_exact(corpus, rxn_smiles)
    if filters:
        exact = [h for h in exact if all(f(h.record) for f in filters)]
    knn = retrieve_knn(corpus, rxn_smiles, k=5, filters=filters)
    hits = {"exact": exact, "knn": knn}
    proposal = compose_protocol(
        target_reaction_smiles=rxn_smiles,
        skeleton=skeleton,
        hits_by_channel=hits,
        user_constraints=constraints,
    )
    return proposal, hits


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def propose_protocol(
    rxn_smiles: str,
    *,
    constraints: dict[str, Any] | None = None,
    corpus: LocalCorpus | None = None,
    weights: Weights | None = None,
    reference_year: int = 2025,
) -> PredictorOutput:
    """Build, screen, and rank candidate protocols for a target reaction.

    When the classifier is confident (≥ 0.7) we compose against the
    single matched skeleton. Below threshold we run **all** skeletons in
    parallel so the ranker arbitrates — this is the "classifier
    misclassification cascade" mitigation from the plan.

    Safety-blocked candidates are removed from the ranking (and
    surfaced separately) before scoring, so the chemist sees what was
    excluded.
    """
    # Explicit None check — an empty user-supplied corpus is falsy via
    # __len__, so ``or default_seed_corpus()`` would silently swap it
    # out. The user passed an empty corpus on purpose.
    if corpus is None:
        corpus = default_seed_corpus()
    warnings: list[str] = []
    classification = _classify_target(rxn_smiles)

    # --- decide which skeletons to try ----------------------------------
    if classification is not None and classification.cls is not ReactionClass.UNKNOWN \
            and classification.confidence >= 0.7:
        skeleton = get_skeleton(classification.cls)
        skeletons_to_try: list[ProtocolSkeleton] = [skeleton] if skeleton else []
    else:
        skeletons_to_try = all_skeletons()
        if classification is not None:
            warnings.append(
                f"classifier confidence {classification.confidence:.2f} "
                f"below 0.7 — trying all {len(skeletons_to_try)} skeletons"
            )
        else:
            warnings.append("classifier returned no class — trying all skeletons")

    if not skeletons_to_try:
        return PredictorOutput(
            target_reaction_smiles=rxn_smiles,
            classification=classification,
            ranked_proposals=[],
            warnings=warnings + ["no skeleton available for this reaction class"],
        )

    # --- build candidates -----------------------------------------------
    proposals: list[ProposalResult] = []
    hits_per_proposal: list[dict[str, list[Hit]]] = []
    blocked: list[ProposalResult] = []

    for sk in skeletons_to_try:
        proposal, hits = _propose_with_skeleton(
            rxn_smiles=rxn_smiles,
            skeleton=sk,
            corpus=corpus,
            constraints=constraints,
        )
        report = safety_screen(proposal.draft)
        if report.verdict is SafetyVerdict.BLOCKED:
            blocked.append(proposal)
            warnings.append(
                f"safety-blocked candidate for {sk.reaction_class.value}: "
                f"{'; '.join(report.flags)}"
            )
            continue
        if report.verdict is SafetyVerdict.WARN:
            proposal.warnings.append(
                f"safety screen: {'; '.join(report.flags)}"
            )
        proposals.append(proposal)
        hits_per_proposal.append(hits)

    # --- corpus-wide warnings -------------------------------------------
    all_hits: list[Hit] = []
    for h_dict in hits_per_proposal:
        for hits in h_dict.values():
            all_hits.extend(hits)
    if all_hits:
        rec = recency_summary(all_hits, reference_year=reference_year)
        if rec.warning:
            warnings.append(rec.warning)

    # --- rank ------------------------------------------------------------
    ranked = rank_proposals(
        proposals,
        hits_by_proposal=hits_per_proposal,
        constraints=constraints,
        weights=weights,
    )

    return PredictorOutput(
        target_reaction_smiles=rxn_smiles,
        classification=classification,
        ranked_proposals=ranked,
        warnings=warnings,
        safety_blocked_proposals=blocked,
    )


__all__ = [
    "PredictorOutput",
    "propose_protocol",
]
