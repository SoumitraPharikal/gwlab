"""Unit tests for GWLab scientific formatting utilities."""

import math
from gwlab.format import (
    format_float,
    format_scientific,
    format_with_unit,
    format_uncertainty,
    to_superscript,
    MINUS,
    THIN_SPACE,
)


def test_to_superscript():
    assert to_superscript(0) == "⁰"
    assert to_superscript(123) == "¹²³"
    assert to_superscript(-19) == "⁻¹⁹"


def test_format_float():
    assert format_float(3.14159, 2) == "3.14"
    assert format_float(-5.2, 1) == f"{MINUS}5.2"
    assert format_float(float("nan")) == "NaN"
    assert format_float(float("inf")) == "+∞"
    assert format_float(float("-inf")) == f"{MINUS}∞"


def test_format_scientific():
    # Negative exponent
    res = format_scientific(1.234e-19, sigfigs=4)
    assert res == f"1.234 \u00d7 10⁻¹⁹"

    # Negative number
    res_neg = format_scientific(-4.5e6, sigfigs=2)
    assert res_neg.startswith(MINUS)

    # Zero
    assert format_scientific(0.0) == "0"


def test_format_with_unit():
    res = format_with_unit(28.0956, "M_sun", decimals=2)
    assert res == f"28.10{THIN_SPACE}M_sun"


def test_format_uncertainty():
    res = format_uncertainty(201646.1413, 0.2374, decimals=4, unit="nats")
    assert res == f"201646.1413 ± 0.2374{THIN_SPACE}nats"
