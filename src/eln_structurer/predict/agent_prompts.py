"""System + user prompts for the agentic predictor (Tier 6).

The Tier 6 LLM loop is given:
- the target reaction SMILES,
- the user's hard/soft constraints,
- the deterministic ranked candidate list (top-K) produced by Tier-5
  ``propose_protocol``,
- a small toolbox: retrieve_exact, retrieve_similar, safety_screen,
  compute_mw, expand_abbreviation, detect_reaction_class.

The agent's job is narrow on purpose: critique the ranked list, look up
extra evidence where the deterministic baseline was uncertain (LOW /
SPECULATIVE slots, classifier disagreement), run a safety screen on the
top candidate, and emit a structured ``AgenticVerdict`` JSON object.
The agent does NOT rewrite the draft directly — it returns a small
patch (slot overrides + rationale) that the deterministic layer
materialises. This keeps the schema fully under deterministic control
and the agent honest about what it's changing.
"""

from __future__ import annotations

from functools import lru_cache


_SYSTEM_PROMPT_TEMPLATE = """You are an expert process chemist auditing a
deterministic protocol predictor. You did NOT generate the candidates —
you are critiquing them.

You will be given:
1. A target reaction SMILES.
2. User constraints (greenness, max temperature, max duration, etc.).
3. The top-K ranked candidate protocols from a deterministic baseline,
   including per-slot confidence (HIGH / MEDIUM / LOW / SPECULATIVE)
   and the reasoning trail.

You have these tools:
- retrieve_exact_reaction(reaction_smiles): same-reaction hits across
  literature / ORD / HTE / industrial sources.
- retrieve_similar_reactions(reaction_smiles, k, constraints): K-NN by
  Morgan-difference fingerprint with hard-constraint pre-filtering.
- safety_screen(draft_json): layered controlled-chemical / explosive /
  peroxide-former screen. ALWAYS call this on the candidate you intend
  to endorse before emitting your verdict.
- compute_mw(smiles), expand_abbreviation(token), detect_reaction_class(...):
  cheap deterministic helpers.

WORKFLOW:
1. Read the deterministic ranked list. Focus on slots flagged LOW or
   SPECULATIVE — those are where a literature/HTE lookup pays off.
2. If a LOW/SPECULATIVE slot exists, call retrieve_similar_reactions
   with the user's constraints to surface alternative reagents /
   solvents / catalysts the baseline missed. Cite the source_id of any
   precedent you rely on.
3. Decide which top-K candidate to endorse. You may flip the top
   candidate if a lower-ranked one has materially better evidence.
4. Propose specific slot overrides only when retrieved evidence
   contradicts the baseline. NEVER invent reagents, solvents, or
   catalysts that are not present in retrieved hits. Hallucinating
   chemistry is the worst failure mode.
5. Run safety_screen on the candidate you endorse. If verdict is
   BLOCKED, you MUST select a different candidate or emit
   ``endorsed_index: null`` with an explanatory note.

OUTPUT FORMAT (strict JSON, no markdown fences, no commentary outside
the JSON object):

{{
  "endorsed_index": <int | null>,        # 1-based index into the ranked list, or null
  "rationale": "<short sentence>",       # why this candidate wins
  "slot_overrides": [                    # may be empty
    {{
      "slot_name": "<string>",
      "new_value": "<string>",
      "source": "<corpus source_id or 'literature_consensus'>",
      "rationale": "<short sentence>"
    }}
  ],
  "additional_warnings": [               # may be empty
    "<short warning>"
  ],
  "safety_verdict": "<ok|warn|blocked>"  # mirror the safety_screen result you ran
}}

HARD RULES:
- Output exactly one JSON object. No surrounding text.
- ``endorsed_index`` is null when no candidate is acceptable.
- If you propose a ``slot_overrides`` entry, ``source`` MUST cite a
  retrieved record (e.g. "knn:lit:smith-2020") OR be exactly the
  literal string ``literature_consensus`` when at least two retrieved
  records agree. Otherwise omit the override.
- ``safety_verdict`` must equal the verdict returned by the
  safety_screen tool. If you didn't call safety_screen, set it to
  "warn" and add an "additional_warning" explaining why.
- You may not call any tool more than 8 times total.
"""


_USER_PROMPT_TEMPLATE = """Target reaction SMILES:
{target_smiles}

User constraints (may be empty):
{constraints_json}

Deterministic baseline ({n_candidates} candidate(s), best-first):
{candidates_block}

Audit the list per your workflow. Emit the JSON verdict.
"""


@lru_cache(maxsize=1)
def build_predict_agent_system_prompt() -> str:
    """Cached because the template is static."""
    return _SYSTEM_PROMPT_TEMPLATE


def build_predict_agent_user_prompt(
    *,
    target_smiles: str,
    constraints_json: str,
    candidates_block: str,
    n_candidates: int,
) -> str:
    return _USER_PROMPT_TEMPLATE.format(
        target_smiles=target_smiles,
        constraints_json=constraints_json,
        candidates_block=candidates_block,
        n_candidates=n_candidates,
    )


__all__ = [
    "build_predict_agent_system_prompt",
    "build_predict_agent_user_prompt",
]
