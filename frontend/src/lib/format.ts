/**
 * Scientific number formatting utility for GWLab.
 *
 * Implements editorial formatting rules:
 * - Proper typographic minus sign (\u2212)
 * - Multiplication sign (\u00D7) and true unicode superscripts for scientific notation
 * - Thin space (\u2009) separating numerical values from physical units
 * - Elimination of floating-point artifacts
 * - Clean tabular display formatting without altering underlying raw floats
 */

const SUPERSCRIPTS: Record<string, string> = {
  "-": "\u207B",
  "0": "\u2070",
  "1": "\u00B9",
  "2": "\u00B2",
  "3": "\u00B3",
  "4": "\u2074",
  "5": "\u2075",
  "6": "\u2076",
  "7": "\u2077",
  "8": "\u2078",
  "9": "\u2079",
};

/**
 * Converts an exponent integer to Unicode superscripts.
 * e.g., -16 -> "⁻¹⁶", 3 -> "³"
 */
export function toSuperscript(exp: number): string {
  const str = String(exp);
  return str
    .split("")
    .map((ch) => SUPERSCRIPTS[ch] || ch)
    .join("");
}

/**
 * Formats a number with true minus sign and rounded to specified decimal places.
 */
export function formatFloat(
  val: number,
  decimals: number = 4,
  fixed: boolean = false
): string {
  if (isNaN(val)) return "NaN";
  if (!isFinite(val)) return val > 0 ? "+∞" : "\u2212∞";

  if (Math.abs(val) < 1e-15) {
    return (0).toFixed(fixed ? decimals : 0);
  }

  const isNeg = val < 0;
  const absVal = Math.abs(val);

  let formatted = fixed ? absVal.toFixed(decimals) : String(Number(absVal.toFixed(decimals)));
  if (!fixed && formatted.includes(".")) {
    formatted = formatted.replace(/\.?0+$/, "");
  }

  return (isNeg ? "\u2212" : "") + formatted;
}

/**
 * Formats a floating-point value with an attached physical unit, separated by a thin space.
 * e.g., formatWithUnit(28.0955, "M_sun") -> "28.0955 M_sun"
 */
export function formatWithUnit(
  val: number,
  unit: string = "",
  decimals: number = 4
): string {
  const numStr = formatFloat(val, decimals, true);
  return unit ? `${numStr}\u2009${unit}` : numStr;
}

/**
 * Formats scientific numbers with \u00D7 10^exp unicode superscripts.
 * e.g., 2.34e-21 -> "2.34 × 10⁻²¹"
 */
export function formatScientific(
  val: number,
  options?: {
    sigFigs?: number;
    forceScientificThreshold?: number;
    upperThreshold?: number;
  }
): string {
  if (val === 0) return "0";
  if (isNaN(val)) return "NaN";
  if (!isFinite(val)) return val > 0 ? "+∞" : "\u2212∞";

  const sigFigs = options?.sigFigs ?? 3;
  const forceScientificThreshold = options?.forceScientificThreshold ?? 1e-4;
  const upperThreshold = options?.upperThreshold ?? 1e5;

  const isNeg = val < 0;
  const absVal = Math.abs(val);

  if (absVal >= forceScientificThreshold && absVal < upperThreshold) {
    return (isNeg ? "\u2212" : "") + absVal.toFixed(sigFigs);
  }

  const expStr = absVal.toExponential(sigFigs);
  const [mantissa, exponent] = expStr.split("e");
  const expInt = parseInt(exponent, 10);

  return `${isNeg ? "\u2212" : ""}${mantissa}\u2009\u00D7\u200910${toSuperscript(expInt)}`;
}

/**
 * Formats value with uncertainty: val ± err [unit]
 */
export function formatUncertainty(
  val: number,
  err: number,
  decimals: number = 4,
  unit?: string
): string {
  const vStr = formatFloat(val, decimals, true);
  const eStr = formatFloat(err, decimals, true);
  const base = `${vStr}\u2009\u00B1\u2009${eStr}`;
  return unit ? `${base}\u2009${unit}` : base;
}
