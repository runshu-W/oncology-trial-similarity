"""Performance measures for the gold-standard borrowing simulation.

Two families, kept deliberately separate (see 手稿修改方案 v3, section 1.5):

*   Operating characteristics for the estimand theta_0 -- bias, empirical SE,
    RMSE, credible-interval coverage, type I error and power. These are the
    frequentist quantities a biostatistics journal expects.
*   Borrowing-decision discrimination -- sensitivity/specificity/FPR/PPV/NPV,
    ROC-AUC and PR-AUC of the per-candidate weights against the parameter-level
    gold standard. These are only computable because the simulation has a known
    truth.

Every reported quantity carries a Monte Carlo standard error (MCSE).
"""

from __future__ import annotations

import numpy as np

# --- Design constants for the operating-characteristic evaluation -----------
# A single-arm oncology design testing H0: theta <= THETA_NULL against
# H1: theta > THETA_NULL, declaring success when the posterior probability of
# H1 exceeds SUCCESS_THRESHOLD. THETA_NULL is a FIXED design benchmark: it does
# not depend on the replicate's true value. Type I error is therefore evaluated
# in a null world where theta_0 = THETA_NULL exactly, and power in an
# alternative world where theta_0 = THETA_NULL + DESIGN_DELTA.
#
# The earlier version of this simulation set the null threshold to the
# replicate's own true theta_0, which yields a one-sided posterior calibration
# rate rather than a frequentist type I error. That quantity is still useful as
# a diagnostic and is reported separately as `calibration_rate`.
THETA_NULL = 0.20
DESIGN_DELTA = 0.15
SUCCESS_THRESHOLD = 0.975
POWER_MARGIN = 0.15


def mcse_mean(values):
    values = np.asarray(values, dtype=float)
    n = values.size
    return float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def mcse_prop(p, n):
    return float(np.sqrt(max(p * (1 - p), 0.0) / n)) if n > 0 else float("nan")


def roc_auc(scores, labels):
    """Rank-based ROC-AUC (ties handled by mid-ranks)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(scores, labels):
    """Average precision (area under the precision-recall curve)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    tp = np.cumsum(lab)
    precision = tp / np.arange(1, len(lab) + 1)
    return float(np.sum(precision * lab) / labels.sum())


def _rank(x):
    """Mid-ranks, used for Spearman correlation."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    xs = x[order]
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def oracle_weights(cands, theta0, epsilon):
    """Bayes-optimal borrowing allocation under known truth (辅金标准).

    Given the truth, the allocation that minimises the mean squared error of the
    borrowed estimate puts mass only on genuinely exchangeable donors, in
    proportion to the information each carries. Donors outside the
    exchangeability band contribute bias and get zero mass.

    This is a strictly stronger benchmark than the binary borrowable/not label:
    it grades not just WHICH donors a method picks but HOW it splits mass among
    them. Returns a normalised weight vector (all zeros if nothing is
    exchangeable).
    """
    w = np.array([
        (c["n"] if abs(c["theta_obs"] - theta0) <= epsilon else 0.0)
        for c in cands
    ], dtype=float)
    total = w.sum()
    return w / total if total > 0 else w


def weight_agreement(learned, oracle, eps=1e-9):
    """Agreement between a method's allocation and the oracle allocation.

    Reports Spearman correlation (rank agreement, threshold-free) and
    KL(oracle || learned) after normalising and smoothing both, which penalises
    putting mass where the oracle puts none.
    """
    learned = np.asarray(learned, dtype=float)
    oracle = np.asarray(oracle, dtype=float)
    if oracle.sum() <= 0 or learned.sum() <= 0 or len(learned) < 2:
        return float("nan"), float("nan")
    p = oracle / oracle.sum()
    q = learned / learned.sum()
    q = (q + eps) / (1.0 + eps * len(q))
    mask = p > 0
    kl = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return spearman(learned, oracle), kl


def ess_response(discounts, cands, theta0, edges=(0.0, 0.05, 0.10, 0.20, 1.0)):
    """Learned per-source ESS binned by true distance from the query rate.

    Evidence for 手稿修改方案 v3 section 2.5: a discount head that genuinely
    adapts to comparability should give monotonically less weight to donors that
    are further from the truth. Returns (bin_sums, bin_counts) so that many
    replicates can be pooled before taking the ratio.
    """
    nb = len(edges) - 1
    sums, counts = np.zeros(nb), np.zeros(nb)
    if discounts is None:
        return sums, counts
    for a, c in zip(discounts, cands):
        d = abs(c["theta_obs"] - theta0)
        for b in range(nb):
            if edges[b] <= d < edges[b + 1]:
                sums[b] += a * c["n"]
                counts[b] += 1
                break
    return sums, counts


def classification_counts(scores, labels, n_cand):
    """Binarise weights into 'borrowed' vs 'not borrowed'.

    A candidate counts as effectively borrowed when it receives at least half of
    an equal share of the historical mass, i.e. normalised weight >= 1/(2K).
    This threshold is scale-free and does not depend on the method.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    total = scores.sum()
    share = scores / total if total > 0 else np.zeros_like(scores)
    predicted = share >= (1.0 / (2.0 * max(n_cand, 1)))
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    return tp, fp, fn, tn


def summarise_replicates(rows, n_candidates_nominal):
    """Aggregate per-replicate records into a scenario-level summary."""
    bias = np.array([r["bias"] for r in rows])
    cover = np.array([r["covered"] for r in rows], dtype=float)
    width = np.array([r["width"] for r in rows])
    t1 = np.array([r["reject_null"] for r in rows], dtype=float)
    pw = np.array([r["reject_power"] for r in rows], dtype=float)
    ess = np.array([r["prior_ess"] for r in rows])
    mass = np.array([r["hist_mass"] for r in rows])
    nll = np.array([r["nll"] for r in rows])
    n = len(rows)

    aucs = np.array([r["auc"] for r in rows if not np.isnan(r["auc"])])
    praucs = np.array([r["prauc"] for r in rows if not np.isnan(r["prauc"])])

    # Design-based decision rate against the FIXED null, plus the older
    # calibration diagnostic against the replicate's own truth.
    decide = np.array([r.get("reject_design", np.nan) for r in rows], dtype=float)
    decide = decide[~np.isnan(decide)]
    calib = np.array([r.get("calibration_hit", np.nan) for r in rows], dtype=float)
    calib = calib[~np.isnan(calib)]

    # Oracle-allocation agreement (auxiliary gold standard).
    osp = np.array([r["oracle_spearman"] for r in rows
                    if not np.isnan(r.get("oracle_spearman", np.nan))])
    okl = np.array([r["oracle_kl"] for r in rows
                    if not np.isnan(r.get("oracle_kl", np.nan))])

    # Learned-ESS response curve, pooled across replicates.
    ess_bins = np.zeros(4)
    ess_counts = np.zeros(4)
    for r in rows:
        if r.get("ess_sums") is not None:
            ess_bins += np.asarray(r["ess_sums"])
            ess_counts += np.asarray(r["ess_counts"])
    ess_curve = [float(ess_bins[i] / ess_counts[i]) if ess_counts[i] > 0 else float("nan")
                 for i in range(4)]

    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    fpr = 1 - spec if not np.isnan(spec) else float("nan")
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else float("nan")

    return {
        "n_replicates": n,
        # --- operating characteristics ---
        "bias": float(bias.mean()), "bias_mcse": mcse_mean(bias),
        "empirical_se": float(bias.std(ddof=1)),
        "rmse": float(np.sqrt((bias ** 2).mean())),
        "rmse_mcse": mcse_mean(bias ** 2) / (2 * max(np.sqrt((bias ** 2).mean()), 1e-9)),
        "coverage95": float(cover.mean()), "coverage95_mcse": mcse_prop(cover.mean(), n),
        "interval_width": float(width.mean()), "interval_width_mcse": mcse_mean(width),
        # Rejection rate against the FIXED design null. Interpreted as type I
        # error in a null world and as power in an alternative world; the driver
        # records which world produced the row.
        "reject_rate": float(decide.mean()) if decide.size else float("nan"),
        "reject_rate_mcse": mcse_prop(decide.mean(), decide.size) if decide.size else float("nan"),
        # Legacy diagnostic: one-sided posterior calibration against the
        # replicate's own true value. NOT a frequentist type I error.
        "calibration_rate": float(calib.mean()) if calib.size else float("nan"),
        "calibration_rate_mcse": mcse_prop(calib.mean(), calib.size) if calib.size else float("nan"),
        "type1": float(t1.mean()), "type1_mcse": mcse_prop(t1.mean(), n),
        "power": float(pw.mean()), "power_mcse": mcse_prop(pw.mean(), n),
        # --- auxiliary gold standard: agreement with the oracle allocation ---
        "oracle_spearman": float(osp.mean()) if osp.size else float("nan"),
        "oracle_spearman_mcse": mcse_mean(osp) if osp.size > 1 else float("nan"),
        "oracle_kl": float(okl.mean()) if okl.size else float("nan"),
        "oracle_kl_mcse": mcse_mean(okl) if okl.size > 1 else float("nan"),
        # --- learned ESS by true distance from the query rate ---
        "ess_d000_005": ess_curve[0], "ess_d005_010": ess_curve[1],
        "ess_d010_020": ess_curve[2], "ess_d020_plus": ess_curve[3],
        "prior_ess": float(ess.mean()), "prior_ess_mcse": mcse_mean(ess),
        "hist_mass": float(mass.mean()), "hist_mass_mcse": mcse_mean(mass),
        "mean_nll": float(nll.mean()), "mean_nll_mcse": mcse_mean(nll),
        # --- borrowing-decision discrimination ---
        "roc_auc": float(aucs.mean()) if aucs.size else float("nan"),
        "roc_auc_mcse": mcse_mean(aucs) if aucs.size > 1 else float("nan"),
        "pr_auc": float(praucs.mean()) if praucs.size else float("nan"),
        "pr_auc_mcse": mcse_mean(praucs) if praucs.size > 1 else float("nan"),
        "sensitivity": sens, "specificity": spec, "fpr": fpr,
        "ppv": ppv, "npv": npv, "f1": f1,
        "sensitivity_mcse": mcse_prop(sens, tp + fn) if (tp + fn) else float("nan"),
        "specificity_mcse": mcse_prop(spec, tn + fp) if (tn + fp) else float("nan"),
        "fpr_mcse": mcse_prop(fpr, tn + fp) if (tn + fp) else float("nan"),
        "ppv_mcse": mcse_prop(ppv, tp + fp) if (tp + fp) else float("nan"),
    }
