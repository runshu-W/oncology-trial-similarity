"""Selective two-head model inside the gold-standard simulation (revision 2).

Ports the manuscript's revised core method — learnable per-query total
borrowing via a set-level weak-component head joining the candidate softmax,
plus an empirical-Bayes Beta(a0, b0) anchor — into the pure-NumPy simulation
framework, so the selective prior receives the same operating-characteristic
evaluation as every other method (review round 2, Major point 2).

Everything mirrors pipeline/two_head_selective.py: joint softmax
lambda = softmax([w0, z_i + log gate_i]), EB anchor by beta-binomial MLE
(grid + coordinate refinement), KL-to-rule on the NORMALISED candidate
allocation, borrowed-information penalty 1e-4 * max(sum lam_i a_i n_i - 100, 0)^2.
"""
from __future__ import annotations

import math

import numpy as np

from autodiff import Tensor, concat, lgamma, logsumexp, relu, sigmoid
from methods import (
    LAMBDA0_DEFAULT, MixturePrior, TwoHeadDeepSets, _example_tensors,
    candidate_features, conservative_gate, gammaln, log_beta_binomial,
)


# ---------------------------------------------------------------------------
# Mixture prior with a general Beta(a0, b0) weak component
# ---------------------------------------------------------------------------
class AnchoredMixturePrior(MixturePrior):
    """MixturePrior whose weak component is Beta(a0, b0) instead of Beta(1, 1)."""

    def __init__(self, lambda0, weights, alphas, betas, ess_terms=None,
                 scores=None, weak_ab=(1.0, 1.0)):
        super().__init__(lambda0, weights, alphas, betas, ess_terms, scores)
        self.weak_ab = (float(weak_ab[0]), float(weak_ab[1]))

    def _all(self):
        w = np.concatenate([[self.lambda0], self.weights])
        a = np.concatenate([[self.weak_ab[0]], self.alphas])
        b = np.concatenate([[self.weak_ab[1]], self.betas])
        return w, a, b

    def log_predictive(self, y, n):
        w, a, b = self._all()
        keep = w > 0
        if not keep.any():
            return log_beta_binomial(y, n, *self.weak_ab)
        terms = [math.log(wi) + log_beta_binomial(y, n, ai, bi)
                 for wi, ai, bi in zip(w[keep], a[keep], b[keep])]
        m = max(terms)
        return m + math.log(sum(math.exp(t - m) for t in terms))


def eb_only(cands, anchor):
    """Intercept-only EB reference: all mass on the anchored weak component."""
    k = len(cands)
    return AnchoredMixturePrior(1.0, np.zeros(k), np.ones(k), np.ones(k),
                                np.zeros(k), scores=np.zeros(k), weak_ab=anchor)


def apply_sam_anchored(prior: AnchoredMixturePrior, y, n, temperature=1.0):
    """SAM adapter computed against the prior's OWN weak component.

    The conflict ratio compares the historical-mixture predictive with the
    predictive of the prior's weak component (the EB anchor here); mass removed
    from the historical components returns to that same weak component. This is
    the anchored analogue of methods.apply_sam, which hardcodes Beta(1, 1).
    """
    mass = prior.historical_mass()
    if mass <= 0:
        return prior, 0.0
    log_weak = log_beta_binomial(y, n, *prior.weak_ab)
    terms = [math.log(w) + log_beta_binomial(y, n, a, b)
             for w, a, b in zip(prior.weights, prior.alphas, prior.betas) if w > 0]
    if not terms:
        return prior, 0.0
    m = max(terms)
    log_hist = m + math.log(sum(math.exp(t - m) for t in terms)) - math.log(mass)
    ratio = math.exp(min(log_hist - log_weak, 50.0))
    mult = 1.0 if ratio >= 1.0 else max(0.0, ratio) ** temperature
    new_w = prior.weights * mult
    out = AnchoredMixturePrior(1.0 - float(new_w.sum()), new_w, prior.alphas,
                               prior.betas, prior.ess_terms, scores=prior.scores,
                               weak_ab=prior.weak_ab)
    out.discounts = prior.discounts
    return out, (0.0 if mult >= 1.0 else 1.0)


# ---------------------------------------------------------------------------
# Empirical-Bayes anchor (verbatim port of pipeline/two_head_selective.py)
# ---------------------------------------------------------------------------
def fit_eb_anchor(data: list[tuple[float, float]]) -> tuple[float, float]:
    def mnll(a: float, b: float) -> float:
        total = 0.0
        for y, m in data:
            total -= (
                math.lgamma(m + 1) - math.lgamma(y + 1) - math.lgamma(m - y + 1)
                + math.lgamma(y + a) + math.lgamma(m - y + b) - math.lgamma(m + a + b)
                + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            )
        return total / len(data)

    best = (1.0, 1.0, mnll(1.0, 1.0))
    for mu in [x / 100 for x in range(5, 61, 2)]:
        for k in [0.5, 1, 2, 3, 5, 7, 10, 15, 25, 50]:
            a, b = mu * k, (1 - mu) * k
            v = mnll(a, b)
            if v < best[2]:
                best = (a, b, v)
    a0, b0 = best[0], best[1]
    for _ in range(80):
        improved = False
        for da, db in [(1.02, 1), (0.98, 1), (1, 1.02), (1, 0.98),
                       (1.004, 1), (0.996, 1), (1, 1.004), (1, 0.996)]:
            a, b = a0 * da, b0 * db
            v = mnll(a, b)
            if v < best[2]:
                best = (a, b, v)
                a0, b0 = a, b
                improved = True
        if not improved:
            break
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Selective architecture: two heads + set-level weak-component head
# ---------------------------------------------------------------------------
class SelectiveTwoHead(TwoHeadDeepSets):
    """TwoHeadDeepSets plus psi: pooled context -> weak-component logit."""

    def __init__(self, input_dim=9, hidden=16, seed=20260603):
        super().__init__(input_dim=input_dim, hidden=hidden, seed=seed)
        rng = np.random.default_rng(seed + 1)

        def lin(i, o):
            return (Tensor(rng.normal(0, math.sqrt(2.0 / i), (i, o)), requires_grad=True),
                    Tensor(np.zeros((1, o)), requires_grad=True))

        self.P1, self.p1 = lin(hidden, hidden)
        self.P2, self.p2 = lin(hidden, 1)
        self.params = self.params + [self.P1, self.p1, self.P2, self.p2]

    def _pooled(self, X):
        from autodiff import mean0
        e = relu(relu(X.matmul(self.W1) + self.b1).matmul(self.W2) + self.b2)
        return e, mean0(e)

    def forward_selective(self, X):
        from methods import _expand
        e, c = self._pooled(X)
        ctx = concat([e, _expand(c, e.data.shape[0])], axis=1)
        z = (relu(ctx.matmul(self.W3) + self.b3).matmul(self.W4) + self.b4).reshape(-1)
        a = sigmoid((relu(ctx.matmul(self.D1) + self.d1).matmul(self.D2) + self.d2).reshape(-1))
        w0 = (relu(c.matmul(self.P1) + self.p1).matmul(self.P2) + self.p2).reshape(-1)
        return z, a, w0


def _loss_selective(model, q, anchor, rho=0.1, ess_cap=100.0,
                    prospective=False, fixed_discount=None):
    X, gates, y, n, rule_w = _example_tensors(q, LAMBDA0_DEFAULT,
                                              prospective=prospective)
    if gates.sum() <= 0:
        return None
    Xt = Tensor(X)
    z, a, w0 = model.forward_selective(Xt)
    if fixed_discount is not None:
        a = Tensor(np.full(len(gates), float(fixed_discount)))

    log_gate = Tensor(np.where(gates > 0, np.log(np.maximum(gates, 1e-12)), -1e9))
    zg = z + log_gate
    joint = concat([w0.reshape(1), zg.reshape(-1)], axis=0)
    lse = logsumexp(joint)
    log_lam0 = w0.reshape(1) - lse
    log_lam = zg - lse

    alpha = Tensor(1.0) + a * Tensor(y)
    beta = Tensor(1.0) + a * Tensor(n - y)
    y0, n0 = float(q["y_query"]), float(q["n_query"])
    lc = (gammaln(np.array([n0 + 1.0]))[0] - gammaln(np.array([y0 + 1.0]))[0]
          - gammaln(np.array([n0 - y0 + 1.0]))[0])
    log_bb = (Tensor(lc) + lgamma(alpha + Tensor(y0)) + lgamma(beta + Tensor(n0 - y0))
              - lgamma(alpha + beta + Tensor(n0))
              - (lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)))
    weak = log_lam0 + Tensor(log_beta_binomial(y0, n0, anchor[0], anchor[1]))
    all_terms = concat([weak.reshape(1), (log_lam + log_bb).reshape(-1)], axis=0)
    loss = -logsumexp(all_terms)

    # KL toward the rule allocation over the NORMALISED candidate allocation
    # (leaves the learned total borrowing unpenalised).
    if rho > 0:
        lam = log_lam.exp()
        s = lam.sum()
        if float(s.data) > 1e-12:
            log_cand_norm = log_lam - s.log()
            rw = Tensor(np.maximum(rule_w, 1e-9))
            kl = (rw * (rw.log() - log_cand_norm)).sum()
            loss = loss + Tensor(rho) * kl

    lam2 = log_lam.exp()
    ess = (lam2 * a * Tensor(n)).sum()
    over = ess - Tensor(ess_cap)
    if float(over.data) > 0:
        loss = loss + Tensor(1e-4) * over * over
    return loss


def train_selective(train_queries, anchor, epochs=60, lr=0.01, seed=20260603,
                    verbose=False, prospective=False, fixed_discount=None):
    model = SelectiveTwoHead(input_dim=10 if prospective else 9, seed=seed)
    m = [np.zeros_like(p.data) for p in model.params]
    v = [np.zeros_like(p.data) for p in model.params]
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    for epoch in range(epochs):
        total, cnt = 0.0, 0
        for q in train_queries:
            loss = _loss_selective(model, q, anchor, prospective=prospective,
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
            print(f"  epoch {epoch:3d}  mean loss {total / max(cnt, 1):.4f}", flush=True)
    return model


def selective_prior(model, cands, anchor, prospective=False, fixed_discount=None):
    X = candidate_features(cands, prospective=prospective)
    gates = np.array([conservative_gate(c["dim"]) for c in cands])
    y = np.array([float(c["y"]) for c in cands])
    n = np.array([float(c["n"]) for c in cands])
    if gates.sum() <= 0:
        return eb_only(cands, anchor)
    z, a, w0 = model.forward_selective(Tensor(X))
    zg = z.data + np.where(gates > 0, np.log(np.maximum(gates, 1e-12)), -1e9)
    joint = np.concatenate([w0.data.reshape(-1), zg])
    joint -= joint.max()
    lam = np.exp(joint)
    lam /= lam.sum()
    lam0, w = float(lam[0]), lam[1:]
    disc = (np.full(len(cands), float(fixed_discount))
            if fixed_discount is not None else a.data)
    alphas = 1.0 + disc * y
    betas = 1.0 + disc * (n - y)
    prior = AnchoredMixturePrior(lam0, w, alphas, betas, disc * n,
                                 scores=w, weak_ab=anchor)
    prior.discounts = disc
    return prior


def save_selective_npz(model, path):
    np.savez(path, **{f"p{i}": p.data for i, p in enumerate(model.params)})


def load_selective_npz(path, input_dim):
    model = SelectiveTwoHead(input_dim=input_dim)
    z = np.load(path)
    model.load([z[f"p{i}"] for i in range(len(model.params))])
    return model
