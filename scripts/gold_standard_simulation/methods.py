"""Borrowing priors compared in the gold-standard simulation.

Every method maps a query's candidate donor set to a beta mixture prior
    p(theta) = lambda_0 * Beta(1,1) + sum_i lambda_i * Beta(alpha_i, beta_i)
so that all methods are scored on exactly the same footing.
"""

from __future__ import annotations

import math

import numpy as np

from autodiff import (Tensor, concat, digamma, gammaln, lgamma, logsumexp,
                      mean0, relu, sigmoid)

LAMBDA0_DEFAULT = 0.20
# Grid for posterior summaries. 800 points give a resolution of 0.00125 on the
# probability scale, which is far finer than any quantity we report, and the
# density evaluation is the dominant cost of the whole simulation.
_GRID = np.linspace(1e-4, 1 - 1e-4, 800)
_LOG_GRID = np.log(_GRID)
_LOG1M_GRID = np.log1p(-_GRID)


# ---------------------------------------------------------------------------
# Core mixture maths
# ---------------------------------------------------------------------------
def log_beta_binomial(y, n, alpha, beta):
    return (
        gammaln(np.array([n + 1.0]))[0]
        - gammaln(np.array([y + 1.0]))[0]
        - gammaln(np.array([n - y + 1.0]))[0]
        + gammaln(np.array([y + alpha]))[0]
        + gammaln(np.array([n - y + beta]))[0]
        - gammaln(np.array([n + alpha + beta]))[0]
        - (gammaln(np.array([alpha]))[0] + gammaln(np.array([beta]))[0]
           - gammaln(np.array([alpha + beta]))[0])
    )


class MixturePrior:
    """A beta mixture prior with a weak component plus historical components."""

    def __init__(self, lambda0, weights, alphas, betas, ess_terms=None, scores=None):
        self.lambda0 = float(lambda0)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.betas = np.asarray(betas, dtype=np.float64)
        self.ess_terms = np.zeros_like(self.weights) if ess_terms is None else np.asarray(ess_terms)
        # Per-candidate score used for the borrowing-discrimination analysis.
        self.scores = self.weights if scores is None else np.asarray(scores, dtype=np.float64)
        # Per-source ESS discount, populated by the two-head model only.
        self.discounts = None

    # -- full component list including the weak component -------------------
    def _all(self):
        w = np.concatenate([[self.lambda0], self.weights])
        a = np.concatenate([[1.0], self.alphas])
        b = np.concatenate([[1.0], self.betas])
        return w, a, b

    def prior_ess(self):
        """Effective sample size contributed by the historical components."""
        return float(np.sum(self.weights * self.ess_terms))

    def historical_mass(self):
        return float(np.sum(self.weights))

    def log_predictive(self, y, n):
        w, a, b = self._all()
        keep = w > 0
        if not keep.any():
            return log_beta_binomial(y, n, 1.0, 1.0)
        terms = [math.log(wi) + log_beta_binomial(y, n, ai, bi)
                 for wi, ai, bi in zip(w[keep], a[keep], b[keep])]
        m = max(terms)
        return m + math.log(sum(math.exp(t - m) for t in terms))

    def posterior(self, y, n):
        """Return posterior mixture weights and updated beta parameters."""
        w, a, b = self._all()
        keep = w > 0
        w, a, b = w[keep], a[keep], b[keep]
        logw = np.array([math.log(wi) + log_beta_binomial(y, n, ai, bi)
                         for wi, ai, bi in zip(w, a, b)])
        logw -= logw.max()
        post_w = np.exp(logw)
        post_w /= post_w.sum()
        return post_w, a + y, b + (n - y)

    def posterior_mean(self, y, n):
        pw, pa, pb = self.posterior(y, n)
        return float(np.sum(pw * pa / (pa + pb)))

    def _posterior_density(self, y, n):
        """Posterior density on the shared grid (computed once, reused)."""
        pw, pa, pb = self.posterior(y, n)
        logpdf = ((pa[:, None] - 1) * _LOG_GRID[None, :]
                  + (pb[:, None] - 1) * _LOG1M_GRID[None, :]
                  - (gammaln(pa) + gammaln(pb) - gammaln(pa + pb))[:, None])
        dens = np.sum(pw[:, None] * np.exp(logpdf), axis=0)
        total = dens.sum()
        return dens / total if total > 0 else dens

    def posterior_summary(self, y, n, null_threshold, power_threshold, level=0.95):
        """Posterior mean, credible interval and both one-sided tail
        probabilities from a single density evaluation."""
        pw, pa, pb = self.posterior(y, n)
        mean = float(np.sum(pw * pa / (pa + pb)))
        dens = self._posterior_density(y, n)
        cdf = np.cumsum(dens)
        lo = float(_GRID[np.searchsorted(cdf, (1 - level) / 2)])
        hi = float(_GRID[min(np.searchsorted(cdf, 1 - (1 - level) / 2), len(_GRID) - 1)])
        p_null = float(dens[_GRID > null_threshold].sum())
        p_power = float(dens[_GRID > power_threshold].sum())
        return mean, lo, hi, p_null, p_power

    def posterior_interval(self, y, n, level=0.95):
        dens = self._posterior_density(y, n)
        cdf = np.cumsum(dens)
        lo = float(_GRID[np.searchsorted(cdf, (1 - level) / 2)])
        hi = float(_GRID[min(np.searchsorted(cdf, 1 - (1 - level) / 2), len(_GRID) - 1)])
        return lo, hi

    def posterior_prob_greater(self, y, n, threshold):
        dens = self._posterior_density(y, n)
        return float(dens[_GRID > threshold].sum())

    def treatment_effect(self, y_ctl, n_ctl, y_trt, n_trt, level=0.95):
        """Hybrid-control inference for the treatment effect.

        The control rate uses THIS prior updated with the internal control arm;
        the treatment rate uses a flat Beta(1,1) updated with the treatment arm.
        Both posteriors live on the shared grid, so

            P(theta_trt > theta_ctl) = sum_c p_ctl(c) * P(theta_trt > c)

        is an exact discrete convolution rather than a Monte Carlo estimate.
        Returns the posterior mean effect, its credible interval and the
        posterior probability of superiority.
        """
        dens_ctl = self._posterior_density(y_ctl, n_ctl)
        a_t, b_t = 1.0 + y_trt, 1.0 + (n_trt - y_trt)
        log_t = ((a_t - 1) * _LOG_GRID + (b_t - 1) * _LOG1M_GRID
                 - (gammaln(np.array([a_t]))[0] + gammaln(np.array([b_t]))[0]
                    - gammaln(np.array([a_t + b_t]))[0]))
        dens_trt = np.exp(log_t - log_t.max())
        dens_trt /= dens_trt.sum()

        # P(theta_trt > c) for every grid point c, via the survival function.
        surv_trt = np.concatenate([np.cumsum(dens_trt[::-1])[::-1][1:], [0.0]])
        p_sup = float(np.sum(dens_ctl * surv_trt))

        mean_ctl = float(np.sum(dens_ctl * _GRID))
        mean_trt = float(np.sum(dens_trt * _GRID))

        # Exact distribution of the difference. Both posteriors live on the same
        # uniform grid, so the law of theta_trt - theta_ctl is the discrete
        # cross-correlation of the two densities: the probability that the
        # difference equals k grid steps is sum_j dens_trt[j+k] dens_ctl[j].
        # np.correlate evaluates this in C, which is orders of magnitude faster
        # than binning an explicit outer product and involves no approximation.
        cc = np.correlate(dens_trt, dens_ctl, mode="full")
        total = cc.sum()
        if total > 0:
            cc = cc / total
        n_grid = len(_GRID)
        step = _GRID[1] - _GRID[0]
        diff_vals = (np.arange(2 * n_grid - 1) - (n_grid - 1)) * step
        cdf = np.cumsum(cc)
        lo = float(diff_vals[np.searchsorted(cdf, (1 - level) / 2)])
        hi = float(diff_vals[min(np.searchsorted(cdf, 1 - (1 - level) / 2),
                                 len(diff_vals) - 1)])
        return mean_trt - mean_ctl, lo, hi, p_sup, mean_ctl


# ---------------------------------------------------------------------------
# Rule-based reranker weights (the manuscript's Stage-2 rule)
# ---------------------------------------------------------------------------
def conservative_gate(dim):
    if dim["endpoint"] < 1.5 or dim["result_quality"] <= 0.5:
        return 0.0
    gate = 1.0
    if dim["disease"] < 1.5:
        gate *= 0.2
    elif dim["disease"] < 2.5:
        gate *= 0.6
    if dim["redflag"] > 0.5:
        gate *= 0.5
    return gate


def rule_components(cands, lambda0=LAMBDA0_DEFAULT, discount=0.35):
    raw, alphas, betas, ess = [], [], [], []
    for c in cands:
        gate = conservative_gate(c["dim"])
        w = gate * discount * max(0.0, c["dim"]["overall100"]) / 100.0 * math.log1p(c["n"])
        raw.append(w)
        alphas.append(1.0 + discount * c["y"])
        betas.append(1.0 + discount * (c["n"] - c["y"]))
        ess.append(discount * c["n"])
    raw = np.array(raw)
    total = raw.sum()
    weights = np.zeros_like(raw) if total <= 0 else (1 - lambda0) * raw / total
    l0 = 1.0 if total <= 0 else lambda0
    return MixturePrior(l0, weights, alphas, betas, ess, scores=raw)


def weak_only(cands, **_):
    return MixturePrior(1.0, np.zeros(len(cands)), np.ones(len(cands)),
                        np.ones(len(cands)), np.zeros(len(cands)),
                        scores=np.zeros(len(cands)))


# ---------------------------------------------------------------------------
# Prospective set-level conflict feature (contribution 2)
# ---------------------------------------------------------------------------
def _weighted_median(values, weights):
    """Weighted median of ``values`` with non-negative ``weights``."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    total = w.sum()
    if total <= 0:
        return float(np.median(v))
    c = np.cumsum(w) / total
    return float(v[np.searchsorted(c, 0.5)])


def prospective_conflict(cands, max_z=8.0):
    """Set-level conflict signal g_i for each candidate.

    g_i measures how far candidate i's observed rate sits from the CONSENSUS of
    the remaining candidates, standardised by the robust spread of that
    consensus:

        m_{-i} = information-weighted median of {rate_j : j != i}
        s_{-i} = 1.4826 * weighted MAD of {rate_j : j != i} about m_{-i}
        g_i    = |rate_i - m_{-i}| / s_{-i}

    Crucially this uses ONLY the donors' own already-published outcomes; the
    query trial's held-out outcome never enters. The signal is therefore
    available at design time, which is what makes the resulting allocation
    prospective in the sense of 手稿修改方案 v3, section 2.4. This is the
    property that separates it from outcome-dependent adapters such as SAM, and
    from UIP-JS, whose weights are computed against the current dataset.

    Leave-one-out construction matters: including candidate i in its own
    consensus would let a single dominant donor define the consensus it is then
    measured against, masking exactly the outlier we want to catch.
    """
    K = len(cands)
    g = np.zeros(K)
    if K < 3:
        return g
    rates = np.array([c["y"] / max(c["n"], 1) for c in cands], dtype=float)
    ns = np.array([float(c["n"]) for c in cands])
    for i in range(K):
        mask = np.ones(K, dtype=bool)
        mask[i] = False
        m = _weighted_median(rates[mask], ns[mask])
        mad = _weighted_median(np.abs(rates[mask] - m), ns[mask])
        # Floor the scale so that a near-degenerate donor pool cannot produce
        # arbitrarily large z-scores.
        scale = max(1.4826 * mad, 0.02)
        g[i] = min(abs(rates[i] - m) / scale, max_z)
    return g


def candidate_features(cands, prospective=False):
    """Assemble the model input matrix.

    With ``prospective=True`` the nine borrowability dimensions are augmented by
    the scaled set-level conflict signal, giving a 10-dimensional input.

    The conflict signal depends only on the donors' observed outcomes, never on
    the model parameters, so it is computed once per candidate set and cached on
    the candidate dicts. Without the cache it would be recomputed on every
    forward pass of every training epoch, which dominates the runtime.
    """
    X = np.stack([c["features"] for c in cands])
    if not prospective:
        return X
    if "_g_cached" not in cands[0]:
        g_all = prospective_conflict(cands)
        for c, gi in zip(cands, g_all):
            c["_g_cached"] = float(gi)
    g = np.array([c["_g_cached"] for c in cands]).reshape(-1, 1) / 8.0
    return np.concatenate([X, g], axis=1)


def power_prior(cands, a0=0.30, lambda0=0.0, **_):
    ysum = sum(c["y"] for c in cands)
    nsum = sum(c["n"] for c in cands)
    alpha = 1.0 + a0 * ysum
    beta = 1.0 + a0 * (nsum - ysum)
    prior = MixturePrior(lambda0, [1.0 - lambda0], [alpha], [beta], [a0 * nsum],
                         scores=np.full(len(cands), 1.0 / max(len(cands), 1)))
    prior.scores = np.full(len(cands), 1.0)
    return prior


def pooling(cands, **_):
    return power_prior(cands, a0=1.0, lambda0=0.0)


def robust_map(cands, w=0.5, **_):
    """Robustified pooled meta-analytic-predictive prior (RBesT style)."""
    rates = np.array([c["y"] / max(c["n"], 1) for c in cands])
    ns = np.array([c["n"] for c in cands], dtype=float)
    if len(rates) == 0:
        return weak_only(cands)
    mean_rate = float(np.average(rates, weights=ns))
    # Between-trial heterogeneity inflates the MAP variance -> discounted ESS.
    between_var = float(np.var(rates)) if len(rates) > 1 else 0.0
    within = mean_rate * (1 - mean_rate)
    ess_map = within / max(between_var + within / max(ns.sum(), 1.0), 1e-6)
    ess_map = float(np.clip(ess_map, 1.0, 200.0))
    alpha = 1.0 + mean_rate * ess_map
    beta = 1.0 + (1 - mean_rate) * ess_map
    prior = MixturePrior(1.0 - w, [w], [alpha], [beta], [ess_map])
    prior.scores = np.full(len(cands), w)
    return prior


# ---------------------------------------------------------------------------
# Unit information prior (Jin & Yin 2021, Stat Med 40:5657-5672), binary case
# ---------------------------------------------------------------------------
def _log_beta_fn(a, b):
    return gammaln(np.asarray(a, dtype=float)) + gammaln(np.asarray(b, dtype=float)) \
        - gammaln(np.asarray(a, dtype=float) + np.asarray(b, dtype=float))


def _kl_beta(a1, b1, a2, b2):
    """KL( Beta(a1,b1) || Beta(a2,b2) )."""
    return float(
        _log_beta_fn(a2, b2) - _log_beta_fn(a1, b1)
        + (a1 - a2) * digamma(np.array([a1]))[0]
        + (b1 - b2) * digamma(np.array([b1]))[0]
        + (a2 - a1 + b2 - b1) * digamma(np.array([a1 + b1]))[0]
    )


def _uip_from_weights(cands, w, n_current, grid=12):
    """Build the UIP of Jin & Yin (2021) section 3.2 for binary data.

    Unit information  I_U(theta_k) = 1 / (theta_k (1 - theta_k))
    Prior mean        mu   = sum_k w_k theta_k
    Prior variance    eta2 = { M sum_k w_k I_U(theta_k) }^{-1}
    Beta parameters   alpha = mu {mu(1-mu)/eta2 - 1},  beta = (1-mu){...}

    M is the amount parameter. The paper places a Uniform(0, n) hyper-prior on
    it; rather than running MCMC we marginalise over M on a grid, which turns
    the UIP into a finite beta mixture and lets it be scored on exactly the same
    footing as every other prior here. The grid is restricted to the region
    where the beta parameters are positive (M > 1 / (mu(1-mu) S), which is
    essentially M > 1), and the uniform prior is renormalised over that region.
    """
    if len(cands) == 0 or np.sum(w) <= 0:
        return weak_only(cands)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    # The paper evaluates the unit information at the MLE. With the small arms
    # typical of oncology, y_k = 0 is common, and the raw MLE then sends
    # I_U = 1/(theta(1-theta)) to infinity, letting a single zero-response donor
    # dominate the total information and inflating the borrowed ESS by an order
    # of magnitude. We therefore evaluate at the Jeffreys posterior mean
    # (y_k + 1/2)/(n_k + 1) -- the same Jeffreys prior the paper already uses to
    # build the UIP-JS weights, so this is internally consistent with the
    # method rather than an ad hoc patch. It restores the paper's own stated
    # property that the borrowed ESS is approximately M - 1.
    theta_hat = np.array([(c["y"] + 0.5) / (c["n"] + 1.0) for c in cands], dtype=float)
    theta_hat = np.clip(theta_hat, 1e-3, 1 - 1e-3)

    mu = float(np.sum(w * theta_hat))
    mu = float(np.clip(mu, 1e-3, 1 - 1e-3))
    info = 1.0 / (theta_hat * (1.0 - theta_hat))      # I_U(theta_hat_k)
    s = float(np.sum(w * info))
    if s <= 0:
        return weak_only(cands)

    m_lo = 1.0 / (mu * (1.0 - mu) * s)                # properness threshold
    m_hi = float(max(n_current, m_lo * 1.5))
    if m_hi <= m_lo:
        return weak_only(cands)
    ms = np.linspace(m_lo * 1.02, m_hi, grid)

    alphas, betas, ess = [], [], []
    for m in ms:
        eta2 = 1.0 / (m * s)
        common = mu * (1.0 - mu) / eta2 - 1.0
        if common <= 0:
            continue
        a, b = mu * common, (1.0 - mu) * common
        alphas.append(a)
        betas.append(b)
        ess.append(a + b - 2.0)                        # borrowed sample size
    if not alphas:
        return weak_only(cands)

    k = len(alphas)
    weights = np.full(k, 1.0 / k)                      # uniform over M
    prior = MixturePrior(0.0, weights, alphas, betas, ess)
    # Per-source scores for the borrowing-discrimination analysis are the UIP
    # weights themselves, which is what the method uses to allocate information.
    prior.scores = w
    return prior


def uip_dirichlet(cands, y0=None, n0=None, **_):
    """UIP-Dirichlet: weights from the Dirichlet(min(1, n_k/n)) prior mean.

    Depends only on sample sizes, so -- like our prospective allocation -- it
    never looks at the current trial's outcome.
    """
    if len(cands) == 0:
        return weak_only(cands)
    n_cur = float(n0) if n0 else float(np.mean([c["n"] for c in cands]))
    gamma = np.array([min(1.0, c["n"] / max(n_cur, 1.0)) for c in cands], dtype=float)
    if gamma.sum() <= 0:
        return weak_only(cands)
    return _uip_from_weights(cands, gamma, n_cur)


def uip_js(cands, y0, n0, **_):
    """UIP-JS: weights inversely proportional to the JS divergence between the
    Jeffreys-prior posteriors of the current and each historical dataset.

    NOTE this method is OUTCOME-DEPENDENT: d_k is computed against the current
    trial's data. It is therefore a retrospective comparator, and is not
    available at design time. We report it because it is the strongest published
    closed-form competitor, and we flag the distinction explicitly rather than
    comparing it against our prospective allocation as if they were equivalent.
    """
    if len(cands) == 0:
        return weak_only(cands)
    n_cur = float(n0)
    a_cur, b_cur = 0.5 + float(y0), 0.5 + float(n0 - y0)
    d = np.zeros(len(cands))
    for i, c in enumerate(cands):
        yk, nk = float(c["y"]), float(c["n"])
        # The paper subsamples D_k down to n when n_k > n. For summary-level
        # data we use the deterministic analogue of that subsample: rescale the
        # counts to the current sample size, preserving the observed rate.
        if nk > n_cur and nk > 0:
            scale = n_cur / nk
            yk, nk = yk * scale, n_cur
        a_k, b_k = 0.5 + yk, 0.5 + (nk - yk)
        d[i] = 0.5 * (_kl_beta(a_cur, b_cur, a_k, b_k)
                      + _kl_beta(a_k, b_k, a_cur, b_cur)) + 1e-6
    w = 1.0 / np.maximum(d, 1e-9)
    return _uip_from_weights(cands, w, n_cur)


def apply_sam(prior: MixturePrior, y, n, temperature=1.0):
    """SAM-style conflict adapter (Yang et al. 2023)."""
    mass = prior.historical_mass()
    if mass <= 0:
        return prior, 0.0
    log_weak = log_beta_binomial(y, n, 1.0, 1.0)
    terms = [math.log(w) + log_beta_binomial(y, n, a, b)
             for w, a, b in zip(prior.weights, prior.alphas, prior.betas) if w > 0]
    if not terms:
        return prior, 0.0
    m = max(terms)
    log_hist = m + math.log(sum(math.exp(t - m) for t in terms)) - math.log(mass)
    ratio = math.exp(min(log_hist - log_weak, 50.0))
    mult = 1.0 if ratio >= 1.0 else max(0.0, ratio) ** temperature
    new_w = prior.weights * mult
    out = MixturePrior(1.0 - float(new_w.sum()), new_w, prior.alphas, prior.betas,
                       prior.ess_terms, scores=prior.scores)
    return out, (0.0 if mult >= 1.0 else 1.0)


# ---------------------------------------------------------------------------
# Two-head DeepSets (NumPy implementation of the manuscript's core model)
# ---------------------------------------------------------------------------
class TwoHeadDeepSets:
    """phi -> mean-pool -> [rho: allocation head, discount head]."""

    def __init__(self, input_dim=9, hidden=16, seed=20260603):
        rng = np.random.default_rng(seed)

        def lin(i, o):
            return (Tensor(rng.normal(0, math.sqrt(2.0 / i), (i, o)), requires_grad=True),
                    Tensor(np.zeros((1, o)), requires_grad=True))

        self.W1, self.b1 = lin(input_dim, hidden)
        self.W2, self.b2 = lin(hidden, hidden)
        self.W3, self.b3 = lin(hidden * 2, hidden)
        self.W4, self.b4 = lin(hidden, 1)
        self.D1, self.d1 = lin(hidden * 2, hidden)
        self.D2, self.d2 = lin(hidden, 1)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
                       self.W4, self.b4, self.D1, self.d1, self.D2, self.d2]

    def _context(self, X):
        """Per-candidate embedding concatenated with the mean-pooled set context.

        Mean pooling is what makes the model permutation invariant over the
        candidate set (Zaheer et al. 2017); concatenating the pooled context back
        onto each embedding is what lets a candidate's allocation depend on the
        rest of the set rather than on its own features alone.
        """
        e = relu(relu(X.matmul(self.W1) + self.b1).matmul(self.W2) + self.b2)
        c = mean0(e)
        return concat([e, _expand(c, e.data.shape[0])], axis=1)

    def forward(self, X):
        ctx = self._context(X)
        z = (relu(ctx.matmul(self.W3) + self.b3).matmul(self.W4) + self.b4).reshape(-1)
        a = sigmoid((relu(ctx.matmul(self.D1) + self.d1).matmul(self.D2) + self.d2).reshape(-1))
        return z, a

    def state(self):
        return [p.data.copy() for p in self.params]

    def load(self, state):
        for p, s in zip(self.params, state):
            p.data = s.copy()

    def save_npz(self, path):
        np.savez(path, **{f"p{i}": p.data for i, p in enumerate(self.params)})

    @classmethod
    def load_npz(cls, path):
        model = cls()
        z = np.load(path)
        model.load([z[f"p{i}"] for i in range(len(model.params))])
        return model


def _expand(c: Tensor, rows: int) -> Tensor:
    """Broadcast a (1,h) pooled context to (rows,h) inside the autodiff graph."""
    ones = Tensor(np.ones((rows, 1)))
    return ones.matmul(c)


def _example_tensors(q, lambda0=LAMBDA0_DEFAULT, rule_discount=0.35, prospective=False):
    cands = q["candidates"]
    X = candidate_features(cands, prospective=prospective)
    gates = np.array([conservative_gate(c["dim"]) for c in cands])
    y = np.array([float(c["y"]) for c in cands])
    n = np.array([float(c["n"]) for c in cands])
    rule = rule_components(cands, lambda0=lambda0, discount=rule_discount)
    rule_w = rule.weights / max(rule.weights.sum(), 1e-12)
    return X, gates, y, n, rule_w


def _loss_for_query(model, q, lambda0=LAMBDA0_DEFAULT, rho=0.1, ess_cap=100.0,
                    prospective=False, fixed_discount=None):
    X, gates, y, n, rule_w = _example_tensors(q, lambda0, prospective=prospective)
    if gates.sum() <= 0:
        return None
    Xt = Tensor(X)
    z, a = model.forward(Xt)
    if fixed_discount is not None:
        # Ablation for contribution 3: keep the learned allocation head but
        # replace the learned per-source ESS discount with a constant.
        a = Tensor(np.full(len(gates), float(fixed_discount)))

    log_gate = Tensor(np.where(gates > 0, np.log(np.maximum(gates, 1e-12)), -1e9))
    zg = z + log_gate
    log_norm = logsumexp(zg)
    log_lam = zg - log_norm + Tensor(math.log(1.0 - lambda0))

    alpha = Tensor(1.0) + a * Tensor(y)
    beta = Tensor(1.0) + a * Tensor(n - y)

    y0, n0 = float(q["y_query"]), float(q["n_query"])
    lc = (gammaln(np.array([n0 + 1.0]))[0] - gammaln(np.array([y0 + 1.0]))[0]
          - gammaln(np.array([n0 - y0 + 1.0]))[0])
    log_bb = (Tensor(lc) + lgamma(alpha + Tensor(y0)) + lgamma(beta + Tensor(n0 - y0))
              - lgamma(alpha + beta + Tensor(n0))
              - (lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)))
    weak = Tensor(math.log(lambda0) + log_beta_binomial(y0, n0, 1.0, 1.0))
    all_terms = concat([weak.reshape(1), (log_lam + log_bb).reshape(-1)], axis=0)
    loss = -logsumexp(all_terms)

    # Mild KL pull toward the rule allocation (regulariser).
    if rho > 0:
        lam = (log_lam - Tensor(math.log(1.0 - lambda0))).exp()
        rw = Tensor(np.maximum(rule_w, 1e-9))
        kl = (rw * (rw.log() - (lam + Tensor(1e-9)).log())).sum()
        loss = loss + Tensor(rho) * kl

    # ESS cap penalty.
    lam2 = (log_lam).exp()
    ess = (lam2 * a * Tensor(n)).sum()
    over = ess - Tensor(ess_cap)
    if float(over.data) > 0:
        loss = loss + Tensor(1e-4) * over * over
    return loss


def train_two_head(train_queries, epochs=60, lr=0.01, seed=20260603, verbose=False,
                   prospective=False, fixed_discount=None):
    model = TwoHeadDeepSets(input_dim=10 if prospective else 9, seed=seed)
    m = [np.zeros_like(p.data) for p in model.params]
    v = [np.zeros_like(p.data) for p in model.params]
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    for epoch in range(epochs):
        total, cnt = 0.0, 0
        for q in train_queries:
            loss = _loss_for_query(model, q, prospective=prospective,
                                   fixed_discount=fixed_discount)
            if loss is None:
                continue
            for p in model.params:
                p.grad = None
            loss.backward()
            step += 1
            for i, p in enumerate(model.params):
                g = p.grad if p.grad is not None else np.zeros_like(p.data)
                g = np.clip(g, -5.0, 5.0)
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                mh = m[i] / (1 - b1 ** step)
                vh = v[i] / (1 - b2 ** step)
                p.data = p.data - lr * mh / (np.sqrt(vh) + eps)
            total += float(loss.data)
            cnt += 1
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  mean loss {total / max(cnt,1):.4f}", flush=True)
    return model


def two_head_prior(model, cands, lambda0=LAMBDA0_DEFAULT, prospective=False,
                   fixed_discount=None):
    X = candidate_features(cands, prospective=prospective)
    gates = np.array([conservative_gate(c["dim"]) for c in cands])
    y = np.array([float(c["y"]) for c in cands])
    n = np.array([float(c["n"]) for c in cands])
    if gates.sum() <= 0:
        return weak_only(cands)
    z, a = model.forward(Tensor(X))
    zg = z.data + np.where(gates > 0, np.log(np.maximum(gates, 1e-12)), -1e9)
    zg -= zg.max()
    w = np.exp(zg)
    w = (1 - lambda0) * w / w.sum()
    disc = np.full(len(cands), float(fixed_discount)) if fixed_discount is not None else a.data
    alphas = 1.0 + disc * y
    betas = 1.0 + disc * (n - y)
    prior = MixturePrior(lambda0, w, alphas, betas, disc * n, scores=w)
    # Retain the per-source discount so that the learned-ESS response curve
    # (手稿修改方案 v3, section 2.5) can be reconstructed downstream.
    prior.discounts = disc
    return prior
