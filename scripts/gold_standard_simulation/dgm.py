"""Gold-standard data-generating mechanism for the borrowability simulation.

Design principles (see 手稿修改方案 v3, section 3.3):

1.  The truth is generated FIRST, from latent clinical attributes, using a
    non-linear mechanism that is deliberately NOT the two-head DeepSets model.
    The borrowing methods only ever see noisy observed features, never the
    latent attributes. This avoids a self-fulfilling simulation.

2.  The gold standard is defined at the PARAMETER level (decision-relevant
    exchangeability), not by surface similarity.

3.  Traps and hidden gems arise from real clinical structure rather than
    artificial injection: textual/surface similarity is driven by disease and
    regimen, whereas the true response rate is driven mostly by LINE OF THERAPY
    and PD-L1 status. Two trials can therefore look alike and be
    non-exchangeable, or look different and be exchangeable.

Parameters are anchored to advanced non-small-cell lung cancer (NSCLC):
first-line chemo-immunotherapy ORR ~0.50-0.60, first-line IO monotherapy in
PD-L1-high ~0.45, second-line IO monotherapy ~0.20, second-line chemotherapy
~0.12, third-line and beyond ~0.10, with per-arm sample sizes concentrated
between 20 and 60 patients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# NSCLC-anchored latent structure
# ---------------------------------------------------------------------------
LINES = ("1L", "2L", "3L+")
HISTOLOGY = ("nonsq", "sq")
REGIMENS = ("chemo", "io_mono", "chemo_io", "targeted")
PDL1 = ("low", "high")

# Baseline logit contributions. Chosen so that the induced ORRs match the
# published NSCLC ranges cited above.
_INTERCEPT = -2.05
_LINE_EFFECT = {"1L": 1.35, "2L": 0.10, "3L+": -0.35}
_REGIMEN_EFFECT = {"chemo": 0.00, "io_mono": 0.15, "chemo_io": 0.85, "targeted": 1.10}
_PDL1_EFFECT = {"low": -0.25, "high": 0.55}
# Non-linear interaction: immunotherapy only works well when PD-L1 is high;
# targeted therapy only works in the biomarker-enriched subgroup.
_IO_PDL1_SYNERGY = 0.80
_TARGETED_BIOMARKER_SYNERGY = 1.30

FEATURE_NAMES = (
    "s_overall",
    "disease_match",
    "regimen_match",
    "endpoint_match",
    "followup_match",
    "eligibility_match",
    "result_quality",
    "neg_redflag",
    "log_n",
)


@dataclass
class SimConfig:
    """Configuration for one simulation scenario."""

    name: str = "base"
    n_candidates: int = 12
    # Probability that a candidate shares the query's line of therapy.
    p_same_line: float = 0.45
    # Probability that a candidate shares the query's disease/regimen surface.
    p_same_surface: float = 0.55
    # Systematic historical-vs-current logit shift (prior-data conflict).
    conflict_shift: float = 0.0
    # Between-trial heterogeneity on the logit scale.
    tau: float = 0.25
    # Probability a candidate has an incompatible endpoint definition, which
    # biases its OBSERVED rate even when its true rate is close.
    p_endpoint_incompatible: float = 0.18
    p_poor_result_quality: float = 0.12
    # Exchangeability tolerance defining the gold standard (absolute ORR).
    epsilon: float = 0.05
    # Query arm size.
    query_n: tuple[int, int] = (20, 45)
    feature_noise: float = 0.55
    # Design-based operating characteristics. When set, the query's true
    # response rate is pinned to this value by shifting the WHOLE system (query
    # and donors) on the logit scale by a common offset. Recentring rather than
    # overwriting keeps the exchangeability / trap / hidden-gem structure of the
    # donor pool intact, so that the same generative mechanism can be evaluated
    # under a point null (theta_query_fixed = theta_null, giving type I error)
    # and under an alternative (theta_query_fixed = theta_null + delta, giving
    # power). See 手稿修改方案 v3 section 1.1.
    theta_query_fixed: float | None = None
    extras: dict = field(default_factory=dict)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _true_logit(line, regimen, pdl1, biomarker, rng_noise):
    """Non-linear map from latent attributes to the true ORR logit."""
    logit = (
        _INTERCEPT
        + _LINE_EFFECT[line]
        + _REGIMEN_EFFECT[regimen]
        + _PDL1_EFFECT[pdl1]
        + rng_noise
    )
    if regimen in ("io_mono", "chemo_io") and pdl1 == "high":
        logit += _IO_PDL1_SYNERGY
    if regimen == "targeted":
        logit += _TARGETED_BIOMARKER_SYNERGY if biomarker else -0.90
    return logit


def _draw_profile(rng):
    return {
        "line": LINES[rng.integers(0, 3)],
        "histology": HISTOLOGY[rng.integers(0, 2)],
        "regimen": REGIMENS[rng.integers(0, 4)],
        "pdl1": PDL1[rng.integers(0, 2)],
        "biomarker": bool(rng.random() < 0.35),
    }


def _score5(match: bool, rng, noise, high=4.3, low=1.4):
    """Map a latent boolean match onto a noisy 0-5 reranker dimension score."""
    base = high if match else low
    return float(np.clip(base + rng.normal(0.0, noise), 0.0, 5.0))


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def simulate_query(cfg: SimConfig, rng: np.random.Generator) -> dict:
    """Simulate one query trial and its candidate donor set with ground truth."""
    q = _draw_profile(rng)
    q_logit = _true_logit(q["line"], q["regimen"], q["pdl1"], q["biomarker"], rng.normal(0, cfg.tau))

    # Design-based recentring: shift the entire system so that the query's true
    # rate equals the design value exactly. The offset is applied to the query
    # AND to every donor, so all relative distances -- and therefore every
    # exchangeability label -- are preserved.
    offset = 0.0
    if cfg.theta_query_fixed is not None:
        offset = _logit(cfg.theta_query_fixed) - q_logit
        q_logit += offset

    theta_query = float(_sigmoid(q_logit))
    n_query = int(rng.integers(cfg.query_n[0], cfg.query_n[1] + 1))

    candidates = []
    for _ in range(cfg.n_candidates):
        c = _draw_profile(rng)
        # Surface similarity: share disease/regimen with the query.
        if rng.random() < cfg.p_same_surface:
            c["histology"] = q["histology"]
            c["regimen"] = q["regimen"]
        # Line of therapy is what actually drives the response rate.
        if rng.random() < cfg.p_same_line:
            c["line"] = q["line"]
            c["pdl1"] = q["pdl1"]

        c_logit = _true_logit(
            c["line"], c["regimen"], c["pdl1"], c["biomarker"], rng.normal(0, cfg.tau)
        )
        c_logit += offset + cfg.conflict_shift
        theta_true = float(_sigmoid(c_logit))

        # Endpoint incompatibility / poor result quality bias the OBSERVED rate.
        endpoint_ok = rng.random() >= cfg.p_endpoint_incompatible
        result_ok = rng.random() >= cfg.p_poor_result_quality
        obs_bias = 0.0
        if not endpoint_ok:
            obs_bias += rng.normal(0.0, 0.10) + 0.10
        if not result_ok:
            obs_bias += rng.normal(0.0, 0.07)
        theta_obs = float(np.clip(theta_true + obs_bias, 0.01, 0.99))

        n_c = int(np.clip(rng.lognormal(np.log(38), 0.55), 8, 400))
        y_c = int(rng.binomial(n_c, theta_obs))

        # ---- observed features (noisy, decoupled from the truth) ----
        same_hist = c["histology"] == q["histology"]
        same_reg = c["regimen"] == q["regimen"]
        same_line = c["line"] == q["line"]
        nz = cfg.feature_noise

        disease_match = _score5(same_hist, rng, nz)
        regimen_match = _score5(same_reg, rng, nz)
        endpoint_match = _score5(endpoint_ok, rng, nz, high=4.4, low=1.0)
        followup_match = _score5(endpoint_ok and rng.random() < 0.85, rng, nz)
        # Eligibility partially (and noisily) reflects line of therapy.
        eligibility_match = _score5(same_line if rng.random() < 0.6 else bool(rng.random() < 0.5), rng, nz)
        result_quality = _score5(result_ok, rng, nz, high=4.5, low=0.6)
        redflag = float(np.clip((0.0 if endpoint_ok else 0.45) + (0.0 if result_ok else 0.35)
                                + max(0.0, rng.normal(0.05, 0.12)), 0.0, 1.0))
        # Surface similarity: driven by disease + regimen only. This is exactly
        # what a text retriever sees, and it is blind to line of therapy.
        s_overall = float(np.clip(
            0.5 * (disease_match / 5.0) + 0.4 * (regimen_match / 5.0)
            + 0.1 * (endpoint_match / 5.0) + rng.normal(0.0, 0.05), 0.0, 1.0))

        features = np.array([
            s_overall,
            disease_match / 5.0,
            regimen_match / 5.0,
            endpoint_match / 5.0,
            followup_match / 5.0,
            eligibility_match / 5.0,
            result_quality / 5.0,
            -redflag,
            float(np.log1p(n_c)),
        ], dtype=np.float64)

        # ---- GOLD STANDARD: parameter-level exchangeability on the scale
        # that borrowing actually uses (the observed rate). ----
        borrowable = bool(abs(theta_obs - theta_query) <= cfg.epsilon)

        candidates.append({
            "features": features,
            "y": y_c,
            "n": n_c,
            "theta_true": theta_true,
            "theta_obs": theta_obs,
            "borrowable": borrowable,
            "same_surface": bool(same_hist and same_reg),
            "same_line": same_line,
            "endpoint_ok": endpoint_ok,
            "result_ok": result_ok,
            "dim": {
                "disease": disease_match,
                "regimen": regimen_match,
                "endpoint": endpoint_match,
                "followup": followup_match,
                "eligibility": eligibility_match,
                "result_quality": result_quality,
                "redflag": redflag,
                "overall100": s_overall * 100.0,
            },
        })

    y_query = int(rng.binomial(n_query, theta_query))
    return {
        "theta_query": theta_query,
        "y_query": y_query,
        "n_query": n_query,
        "query_profile": q,
        "candidates": candidates,
    }


def simulate_dataset(cfg: SimConfig, n_queries: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [simulate_query(cfg, rng) for _ in range(n_queries)]


# ---------------------------------------------------------------------------
# External / synthetic control arm scenario (手稿修改方案 v3, section 4.1)
# ---------------------------------------------------------------------------
@dataclass
class ExternalControlConfig:
    """Configuration for the hybrid-control (external control arm) scenario.

    The current trial randomises to a treatment arm and a SMALL internal
    control arm. Candidate donors are historical control arms retrieved from the
    literature. Borrowing augments the internal control; the estimand is the
    treatment effect theta_trt - theta_ctl.

    The regulatory failure mode this is built to expose: if borrowed external
    controls are systematically WORSE than the internal control (a downward
    drift), the hybrid control prior drags the control estimate down and the
    treatment looks better than it is, inflating type I error.
    """

    name: str = "EC_base"
    n_candidates: int = 12
    # Internal arm sizes: the control arm is deliberately small, which is what
    # motivates borrowing in the first place.
    n_treatment: tuple[int, int] = (40, 70)
    n_control: tuple[int, int] = (15, 30)
    # True control-arm response rate of the current trial.
    theta_control: float = 0.20
    # True treatment effect on the ABSOLUTE rate scale. 0.0 = point null, which
    # is where type I error is evaluated.
    treatment_effect: float = 0.0
    # Probability that an external control is drawn from the same population as
    # the internal control (i.e. genuinely exchangeable).
    p_comparable: float = 0.50
    # Systematic logit drift applied to NON-comparable external controls.
    # Negative = external controls look worse than the internal control, which
    # biases the treatment effect upward and inflates type I error.
    drift_shift: float = -0.80
    tau: float = 0.25
    p_endpoint_incompatible: float = 0.15
    p_poor_result_quality: float = 0.12
    epsilon: float = 0.05
    feature_noise: float = 0.55
    extras: dict = field(default_factory=dict)


def simulate_external_control_query(cfg: ExternalControlConfig,
                                    rng: np.random.Generator) -> dict:
    """Simulate one hybrid-control trial with retrievable external controls."""
    q = _draw_profile(rng)
    # The internal control arm is by construction a standard-of-care arm.
    q["regimen"] = "chemo"

    theta_ctl = float(np.clip(cfg.theta_control, 0.01, 0.99))
    theta_trt = float(np.clip(theta_ctl + cfg.treatment_effect, 0.01, 0.99))
    ctl_logit = _logit(theta_ctl)

    n_trt = int(rng.integers(cfg.n_treatment[0], cfg.n_treatment[1] + 1))
    n_ctl = int(rng.integers(cfg.n_control[0], cfg.n_control[1] + 1))
    y_trt = int(rng.binomial(n_trt, theta_trt))
    y_ctl = int(rng.binomial(n_ctl, theta_ctl))

    candidates = []
    for _ in range(cfg.n_candidates):
        c = _draw_profile(rng)
        c["regimen"] = "chemo"  # historical control arms are also SOC arms

        comparable = bool(rng.random() < cfg.p_comparable)
        if comparable:
            c["line"] = q["line"]
            c["pdl1"] = q["pdl1"]
            c["histology"] = q["histology"]
            c_logit = ctl_logit + rng.normal(0.0, cfg.tau)
        else:
            # Population / temporal drift: the external control comes from a
            # different era or a differently selected population.
            c_logit = ctl_logit + cfg.drift_shift + rng.normal(0.0, cfg.tau)

        theta_true = float(_sigmoid(c_logit))

        endpoint_ok = rng.random() >= cfg.p_endpoint_incompatible
        result_ok = rng.random() >= cfg.p_poor_result_quality
        obs_bias = 0.0
        if not endpoint_ok:
            obs_bias += rng.normal(0.0, 0.10) + 0.10
        if not result_ok:
            obs_bias += rng.normal(0.0, 0.07)
        theta_obs = float(np.clip(theta_true + obs_bias, 0.01, 0.99))

        n_c = int(np.clip(rng.lognormal(np.log(55), 0.55), 10, 500))
        y_c = int(rng.binomial(n_c, theta_obs))

        # ---- observed features: surface similarity is driven by disease and
        # regimen, and is therefore blind to the population drift that actually
        # determines comparability. ----
        same_hist = c["histology"] == q["histology"]
        same_reg = True  # both are SOC control arms
        nz = cfg.feature_noise

        disease_match = _score5(same_hist, rng, nz)
        regimen_match = _score5(same_reg, rng, nz)
        endpoint_match = _score5(endpoint_ok, rng, nz, high=4.4, low=1.0)
        followup_match = _score5(endpoint_ok and rng.random() < 0.85, rng, nz)
        eligibility_match = _score5(
            (c["line"] == q["line"]) if rng.random() < 0.6 else bool(rng.random() < 0.5),
            rng, nz)
        result_quality = _score5(result_ok, rng, nz, high=4.5, low=0.6)
        redflag = float(np.clip((0.0 if endpoint_ok else 0.45) + (0.0 if result_ok else 0.35)
                                + max(0.0, rng.normal(0.05, 0.12)), 0.0, 1.0))
        s_overall = float(np.clip(
            0.5 * (disease_match / 5.0) + 0.4 * (regimen_match / 5.0)
            + 0.1 * (endpoint_match / 5.0) + rng.normal(0.0, 0.05), 0.0, 1.0))

        features = np.array([
            s_overall, disease_match / 5.0, regimen_match / 5.0,
            endpoint_match / 5.0, followup_match / 5.0, eligibility_match / 5.0,
            result_quality / 5.0, -redflag, float(np.log1p(n_c)),
        ], dtype=np.float64)

        borrowable = bool(abs(theta_obs - theta_ctl) <= cfg.epsilon)

        candidates.append({
            "features": features, "y": y_c, "n": n_c,
            "theta_true": theta_true, "theta_obs": theta_obs,
            "borrowable": borrowable, "comparable": comparable,
            "same_surface": bool(same_hist), "same_line": c["line"] == q["line"],
            "endpoint_ok": endpoint_ok, "result_ok": result_ok,
            "dim": {"disease": disease_match, "regimen": regimen_match,
                    "endpoint": endpoint_match, "followup": followup_match,
                    "eligibility": eligibility_match, "result_quality": result_quality,
                    "redflag": redflag, "overall100": s_overall * 100.0},
        })

    return {
        # The borrowing target is the CONTROL arm rate.
        "theta_query": theta_ctl,
        "y_query": y_ctl,
        "n_query": n_ctl,
        "theta_control": theta_ctl,
        "theta_treatment": theta_trt,
        "y_control": y_ctl, "n_control": n_ctl,
        "y_treatment": y_trt, "n_treatment": n_trt,
        "true_effect": theta_trt - theta_ctl,
        "query_profile": q,
        "candidates": candidates,
    }


def simulate_external_control_dataset(cfg: ExternalControlConfig, n_queries: int,
                                      seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [simulate_external_control_query(cfg, rng) for _ in range(n_queries)]


def label_summary(dataset: list[dict]) -> dict:
    """Diagnostics: how many trap / hidden-gem donors the mechanism produced."""
    total = borrow = trap = gem = 0
    for q in dataset:
        for c in q["candidates"]:
            total += 1
            borrow += c["borrowable"]
            # Trap: looks similar on the surface but is NOT borrowable.
            trap += c["same_surface"] and not c["borrowable"]
            # Hidden gem: looks dissimilar but IS borrowable.
            gem += (not c["same_surface"]) and c["borrowable"]
    return {
        "n_candidates": total,
        "prop_borrowable": borrow / max(total, 1),
        "prop_trap": trap / max(total, 1),
        "prop_hidden_gem": gem / max(total, 1),
    }
