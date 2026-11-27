"""Scientific number formatting utilities for GWLab.

Follows the editorial and typographical conventions of Kerr Geodesics Lab:
- True typographical minus (U+2212 '−') instead of ASCII hyphen
- Unicode superscripts for scientific exponent notation (e.g. 10⁻¹⁶)
- Thin space (U+2009) between values and units
- Explicit handling of NaN, ±inf, and zero
"""

from __future__ import annotations

import math
from typing import Union

# Typographical constants
MINUS = "\u2212"
THIN_SPACE = "\u2009"
TIMES = "\u00d7"

_SUPERSCRIPTS: dict[str, str] = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "-": "⁻",
    "+": "⁺",
}


def to_superscript(val: Union[int, str]) -> str:
    """Convert an integer or string of digits/signs to unicode superscripts."""
    return "".join(_SUPERSCRIPTS.get(c, c) for c in str(val))


def _replace_minus(s: str) -> str:
    """Replace standard hyphen-minus with true unicode minus."""
    return s.replace("-", MINUS)


def format_float(
    val: float,
    decimals: int = 4,
    use_unicode_minus: bool = True,
) -> str:
    """Format a floating point value with specified decimal places.
    
    Uses true unicode minus if use_unicode_minus is True.
    """
    if math.isnan(val):
        return "NaN"
    if math.isinf(val):
        return f"{MINUS}\u221e" if val < 0 and use_unicode_minus else ("-\u221e" if val < 0 else "+\u221e")
    
    formatted = f"{val:.{decimals}f}"
    if use_unicode_minus:
        formatted = _replace_minus(formatted)
    return formatted


def format_scientific(
    val: float,
    sigfigs: int = 4,
    use_unicode_minus: bool = True,
) -> str:
    """Format a floating point value in editorial scientific notation.
    
    Example:
        1.234e-19 -> "1.234 × 10⁻¹⁹"
    """
    if math.isnan(val):
        return "NaN"
    if math.isinf(val):
        return f"{MINUS}\u221e" if val < 0 and use_unicode_minus else ("-\u221e" if val < 0 else "+\u221e")
    if val == 0.0:
        return "0"

    exp = int(math.floor(math.log10(abs(val))))
    mantissa = val / (10 ** exp)
    
    # If rounding pushes mantissa to 10
    mantissa_str = f"{mantissa:.{sigfigs - 1}f}"
    if float(mantissa_str) >= 10.0:
        mantissa /= 10.0
        exp += 1
        mantissa_str = f"{mantissa:.{sigfigs - 1}f}"
        
    if use_unicode_minus:
        mantissa_str = _replace_minus(mantissa_str)
        exp_str = to_superscript(exp)
    else:
        exp_str = str(exp)
        return f"{mantissa_str}e{exp_str}"
        
    return f"{mantissa_str} {TIMES} 10{exp_str}"


def format_with_unit(
    val: float,
    unit: str,
    decimals: int = 2,
    scientific: bool = False,
    use_unicode_minus: bool = True,
) -> str:
    """Format a value with a thin-space separated unit.
    
    Example:
        format_with_unit(30.0, "M_sun") -> "30.00 M_sun" (with thin space)
    """
    if scientific or (abs(val) > 0 and (abs(val) >= 1e4 or abs(val) < 1e-2)):
        formatted_val = format_scientific(val, sigfigs=decimals + 1, use_unicode_minus=use_unicode_minus)
    else:
        formatted_val = format_float(val, decimals=decimals, use_unicode_minus=use_unicode_minus)
        
    if not unit:
        return formatted_val
    return f"{formatted_val}{THIN_SPACE}{unit}"


def format_uncertainty(
    val: float,
    unc: float,
    decimals: int = 4,
    unit: str = "",
    use_unicode_minus: bool = True,
) -> str:
    """Format a measurement with symmetric uncertainty.
    
    Example:
        format_uncertainty(201646.1413, 0.2374, decimals=4, unit="nats")
        -> "201646.1413 ± 0.2374 nats"
    """
    val_str = format_float(val, decimals=decimals, use_unicode_minus=use_unicode_minus)
    unc_str = format_float(unc, decimals=decimals, use_unicode_minus=use_unicode_minus)
    res = f"{val_str} \u00b1 {unc_str}"
    if unit:
        res = f"{res}{THIN_SPACE}{unit}"
    return res
