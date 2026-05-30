"""Multi-channel protocol predictor.

Tier 1 surface: corpus + fingerprint + retrieval + risk primitives.
The agentic composition layer arrives in Tier 2; this tier ships the
deterministic, offline-callable building blocks that the agent will
delegate to.

Public surface:

- :class:`ReactionRecord`, :class:`CorpusSource`, :class:`LocalCorpus`
- :func:`reaction_fingerprint`, :func:`tanimoto`
- :func:`retrieve_exact`, :func:`retrieve_knn`, :class:`Hit`
- :class:`ConfidenceLevel`, :class:`SlotProposal`, :class:`ConsensusResult`,
  :func:`multi_source_vote`
- :func:`safety_screen`, :func:`recency_summary`,
  :func:`classifier_must_be_confident`, :func:`hard_constraint_filter`
"""

from __future__ import annotations

from eln_structurer.predict.composition import (
    ProposalResult,
    compose_protocol,
)
from eln_structurer.predict.confidence import (
    ChannelReport,
    ConfidenceLevel,
    ConsensusResult,
    SlotProposal,
    multi_source_vote,
    summarise_channels,
)
from eln_structurer.predict.corpus import (
    CorpusSource,
    LocalCorpus,
    ReactionRecord,
)
from eln_structurer.predict.fingerprint import reaction_fingerprint, tanimoto
from eln_structurer.predict.hte_corpus import default_seed_corpus
from eln_structurer.predict.retrieval import (
    Hit,
    RecordFilter,
    min_year,
    no_halogenated_solvents,
    retrieve_exact,
    retrieve_knn,
    source_in,
)
from eln_structurer.predict.risks import (
    RecencySummary,
    SafetyReport,
    SafetyVerdict,
    classifier_must_be_confident,
    hard_constraint_filter,
    recency_summary,
    safety_screen,
)
from eln_structurer.predict.skeleton import (
    ProtocolSkeleton,
    Slot,
    all_skeletons,
    get_skeleton,
    known_classes,
)

__all__ = [
    # corpus
    "CorpusSource",
    "LocalCorpus",
    "ReactionRecord",
    "default_seed_corpus",
    # fingerprint
    "reaction_fingerprint",
    "tanimoto",
    # retrieval
    "Hit",
    "RecordFilter",
    "retrieve_exact",
    "retrieve_knn",
    "no_halogenated_solvents",
    "min_year",
    "source_in",
    # confidence
    "ConfidenceLevel",
    "SlotProposal",
    "ConsensusResult",
    "ChannelReport",
    "multi_source_vote",
    "summarise_channels",
    # risk primitives
    "RecencySummary",
    "SafetyReport",
    "SafetyVerdict",
    "classifier_must_be_confident",
    "hard_constraint_filter",
    "recency_summary",
    "safety_screen",
    # skeletons + composition
    "Slot",
    "ProtocolSkeleton",
    "all_skeletons",
    "get_skeleton",
    "known_classes",
    "ProposalResult",
    "compose_protocol",
]
