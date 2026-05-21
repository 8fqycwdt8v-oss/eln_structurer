"""Tests for paragraph normalization."""

from __future__ import annotations

from eln_structurer.preprocess import normalize_paragraph


def test_unicode_micro_sign_becomes_u() -> None:
    result = normalize_paragraph("Added 5 µL of solution.")
    assert "uL" in result.normalized
    assert "µ" not in result.normalized
    assert result.changed


def test_multiplication_sign_becomes_x() -> None:
    result = normalize_paragraph("washed with brine (3 × 10 mL)")
    assert "(3 x 10 mL)" in result.normalized
    assert "×" not in result.normalized


def test_smart_quotes_become_ascii() -> None:
    result = normalize_paragraph("“standard” workup")  # curly double quotes
    assert "“" not in result.normalized
    assert "”" not in result.normalized


def test_rt_expansion() -> None:
    result = normalize_paragraph("stirred at rt for 1 h")
    assert "room temperature" in result.normalized
    assert " rt " not in result.normalized


def test_aqueous_expansion() -> None:
    result = normalize_paragraph("washed with sat. NaHCO3")
    assert "saturated NaHCO3" in result.normalized


def test_no_change_returns_changed_false() -> None:
    """Plain ASCII paragraphs untouched by abbreviations should report changed=False."""
    text = "The reaction was conducted at standard conditions."
    result = normalize_paragraph(text)
    assert result.changed is False
    assert result.normalized == text.strip()


def test_original_is_preserved() -> None:
    text = "Heated to 80 °C × 2 h at rt"
    result = normalize_paragraph(text)
    assert result.original == text  # original untouched
    assert result.original != result.normalized  # but normalization happened


def test_non_breaking_space_collapsed() -> None:
    result = normalize_paragraph("benzaldehyde (1.0 equiv)")
    assert " " not in result.normalized
    assert "benzaldehyde (1.0 equiv)" in result.normalized
