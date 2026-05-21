"""Tests for source-grounded numerics + missing-value tracking."""

from __future__ import annotations

from eln_structurer.rules.numeric_grounding import (
    NumericValueGrounded,
    UnitMatchesQuote,
    UnspecifiedFieldsAreValid,
)
from eln_structurer.schema import (
    AmountModel,
    CompoundIdentifierModel,
    CompoundModel,
    ConditionsModel,
    OutcomeModel,
    ProductMeasurementModel,
    ProductModel,
    ReactionDraft,
    ReactionInputModel,
    TemperatureModel,
)
from eln_structurer.tools import verify_quote_against
from tests.conftest import rule_ids as _ids


def _draft(
    *,
    source_paragraph: str,
    amount: AmountModel | None = None,
    measurement: ProductMeasurementModel | None = None,
    unspecified: list[str] | None = None,
) -> ReactionDraft:
    components: list[CompoundModel] = []
    components.append(
        CompoundModel(
            identifiers=[CompoundIdentifierModel(type="NAME", value="substrate")],
            amount=amount or AmountModel(value=1.0, units="mmol"),
            reaction_role="REACTANT",
            is_limiting=True,
        )
    )
    products = [
        ProductModel(
            compound=CompoundModel(
                identifiers=[CompoundIdentifierModel(type="NAME", value="product")],
                reaction_role="PRODUCT",
            ),
            measurements=([measurement] if measurement else []),
        )
    ]
    return ReactionDraft(
        identifiers=[],
        inputs=[ReactionInputModel(name="r", components=components)],
        conditions=ConditionsModel(temperature=TemperatureModel(control_type="AMBIENT")),
        outcomes=[OutcomeModel(products=products)],
        notes="n",
        source_paragraph=source_paragraph,
        unspecified_fields=unspecified or [],
    )


# NUM-001 ---------------------------------------------------------------------


def test_num001_passes_when_quote_present() -> None:
    draft = _draft(
        source_paragraph="To 1.38 g, 10.0 mmol salicylic acid was added...",
        amount=AmountModel(value=1.38, units="g", source_quote="1.38 g, 10.0 mmol"),
    )
    assert NumericValueGrounded().check(draft) == []


def test_num001_fires_when_quote_missing() -> None:
    draft = _draft(
        source_paragraph="To 1.38 g salicylic acid was added...",
        amount=AmountModel(value=1.38, units="g", source_quote="2.50 g"),
    )
    violations = NumericValueGrounded().check(draft)
    assert "NUM-001" in _ids(violations)


def test_num001_silent_when_inferred() -> None:
    """inferred=True means no source_quote check, even if quote is set."""
    draft = _draft(
        source_paragraph="...",
        amount=AmountModel(
            value=1.2,
            units="equiv",
            source_quote="1.2 equiv (derived)",
            inferred=True,
        ),
    )
    assert NumericValueGrounded().check(draft) == []


def test_num001_silent_when_no_quote() -> None:
    """Drafts without source_quote skip NUM-001 entirely (optional)."""
    draft = _draft(
        source_paragraph="trivial",
        amount=AmountModel(value=1.0, units="mmol"),
    )
    assert NumericValueGrounded().check(draft) == []


def test_num001_normalises_whitespace() -> None:
    """Quote and paragraph differing only in whitespace should still match."""
    draft = _draft(
        source_paragraph="To  1.38  g,\n10.0  mmol salicylic",
        amount=AmountModel(value=1.38, units="g", source_quote="1.38 g, 10.0 mmol"),
    )
    assert NumericValueGrounded().check(draft) == []


def test_num001_fires_on_measurement_too() -> None:
    """Same rule applies to ProductMeasurement.source_quote."""
    draft = _draft(
        source_paragraph="afforded the product (181 mg, 92%)",
        measurement=ProductMeasurementModel(
            type="YIELD",
            value=85.0,           # paragraph says 92, draft says 85
            units="%",
            source_quote="85%",   # this quote does not appear in the paragraph
        ),
    )
    violations = NumericValueGrounded().check(draft)
    assert "NUM-001" in _ids(violations)


# NUM-002 ---------------------------------------------------------------------


def test_num002_passes_when_quote_contains_unit() -> None:
    draft = _draft(
        source_paragraph="To 1.38 g salicylic acid",
        amount=AmountModel(value=1.38, units="g", source_quote="1.38 g"),
    )
    assert UnitMatchesQuote().check(draft) == []


def test_num002_passes_for_unit_alias() -> None:
    """'equiv' should accept 'equivalents' in the quote."""
    draft = _draft(
        source_paragraph="1.2 equivalents of base",
        amount=AmountModel(value=1.2, units="equiv", source_quote="1.2 equivalents"),
    )
    assert UnitMatchesQuote().check(draft) == []


def test_num002_warns_when_unit_absent_from_quote() -> None:
    draft = _draft(
        source_paragraph="To 1.38 g salicylic acid",
        amount=AmountModel(value=1.38, units="g", source_quote="1.38"),  # no 'g'
    )
    violations = UnitMatchesQuote().check(draft)
    assert "NUM-002" in _ids(violations)


# NUM-003 ---------------------------------------------------------------------


def test_num003_passes_for_well_formed_paths() -> None:
    draft = _draft(
        source_paragraph="p",
        unspecified=["conditions.duration_minutes", "conditions.atmosphere"],
    )
    assert UnspecifiedFieldsAreValid().check(draft) == []


def test_num003_warns_on_bad_top_level_key() -> None:
    draft = _draft(source_paragraph="p", unspecified=["foo.bar"])
    violations = UnspecifiedFieldsAreValid().check(draft)
    assert "NUM-003" in _ids(violations)


def test_num003_warns_when_field_is_actually_populated() -> None:
    """Agent says notes is unspecified but it's set — flag the contradiction."""
    draft = _draft(source_paragraph="p", unspecified=["notes"])
    # notes default is "n" in our _draft helper; non-empty.
    violations = UnspecifiedFieldsAreValid().check(draft)
    assert "NUM-003" in _ids(violations)


def test_num003_silent_for_bracketed_paths() -> None:
    """List-indexed paths are skipped (we don't introspect them)."""
    draft = _draft(
        source_paragraph="p",
        unspecified=["outcomes[0].reaction_time_minutes"],
    )
    assert UnspecifiedFieldsAreValid().check(draft) == []


# verify_quote helper ---------------------------------------------------------


def test_verify_quote_helper_exact_match() -> None:
    r = verify_quote_against("1.38 g", "added 1.38 g of substrate")
    assert r.ok is True


def test_verify_quote_helper_whitespace_normalised() -> None:
    r = verify_quote_against("1.38 g", "added  1.38  g of substrate")
    assert r.ok is True


def test_verify_quote_helper_reports_prefix() -> None:
    r = verify_quote_against("1.38 g of phenol", "added 1.38 g of substrate")
    assert r.ok is False
    assert r.nearest_match is not None
    assert "1.38 g" in r.nearest_match.lower()


def test_verify_quote_helper_empty_quote() -> None:
    r = verify_quote_against("", "anything")
    assert r.ok is False
