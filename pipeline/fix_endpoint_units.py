"""Correct the unit handling when converting ClinicalTrials.gov outcome rows
into beta-binomial (responders, denominator) pairs.

THE BUG
-------
``pipeline/oncology_trial_similarity_pipeline.py`` computes

    proportion = count / denominator

unconditionally. But ``count`` is the value in whatever unit the registry
reports the outcome in. ClinicalTrials.gov reports objective response rate
sometimes as a participant count and sometimes as a percentage. When the unit is
"Percentage of Participants", a row reading ``26.0`` with denominator 31 means
an ORR of 26%, not 26 responders out of 31 (which the buggy formula turns into
0.839).

Across the ORR corpus this affects every row whose unit is a percentage or a
proportion. 24% of arm-level ORR rows end up with a "proportion" above 1, which
is self-evidently impossible; the remainder land inside (0, 1) and look
plausible, which is what makes the bug dangerous rather than merely obvious.

THE FIX
-------
Dispatch on the reported unit, and refuse to guess when the unit is unreadable.
"""

from __future__ import annotations

import re

# Units where the reported value is a headcount of responders.
_COUNT_UNIT = re.compile(r"^\s*(participants?|patients?|subjects?|"
                         r"number of (participants?|patients?|subjects?)|"
                         r"responders?)\s*$", re.I)
# Units where the reported value is already a percentage (0-100).
_PERCENT_UNIT = re.compile(r"percent|%", re.I)
# Units where the reported value is already a proportion (0-1).
_PROPORTION_UNIT = re.compile(r"proportion|fraction|probability", re.I)


class UnitError(ValueError):
    """Raised when an outcome row cannot be converted safely."""


def rate_from_row(unit, value, denominator):
    """Return the response RATE implied by one arm-level outcome row.

    Raises UnitError when the unit is missing or unrecognised, so that callers
    drop the row instead of silently propagating a wrong number.
    """
    if value is None or denominator in (None, 0):
        raise UnitError("missing value or denominator")
    value = float(value)
    denominator = float(denominator)
    u = (unit or "").strip()
    if not u:
        raise UnitError("missing unit")

    if _COUNT_UNIT.match(u):
        rate = value / denominator
    elif _PERCENT_UNIT.search(u):
        rate = value / 100.0
    elif _PROPORTION_UNIT.search(u):
        rate = value
    else:
        raise UnitError(f"unrecognised unit: {unit!r}")

    if not (0.0 <= rate <= 1.0):
        raise UnitError(f"implied rate {rate:.4f} outside [0,1] for unit {unit!r}")
    return rate


def responders_from_row(unit, value, denominator):
    """Return integer (responders, n) for a beta-binomial component.

    Percentage-reported outcomes do not carry the exact numerator, so the
    responder count is reconstructed as round(rate * n). The rounding error is
    at most half a patient and is recorded by the caller as a data-quality flag.
    """
    rate = rate_from_row(unit, value, denominator)
    n = int(round(float(denominator)))
    y = int(round(rate * n))
    y = max(0, min(y, n))
    return y, n, (not _COUNT_UNIT.match((unit or "").strip()))


def audit_rows(rows):
    """Summarise how a batch of (unit, value, denominator) rows converts.

    Returns (converted, failures) where converted entries carry both the old
    buggy rate and the corrected rate so the impact can be quantified.
    """
    converted, failures = [], []
    for unit, value, denom in rows:
        try:
            rate = rate_from_row(unit, value, denom)
        except UnitError as exc:
            failures.append({"unit": unit, "value": value,
                             "denominator": denom, "error": str(exc)})
            continue
        buggy = (float(value) / float(denom)) if denom else float("nan")
        converted.append({"unit": unit, "value": value, "denominator": denom,
                          "corrected_rate": rate, "buggy_rate": buggy,
                          "abs_error": abs(buggy - rate)})
    return converted, failures
