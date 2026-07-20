"""Central-finite-difference gradient checks for the minimal autodiff engine."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autodiff import Tensor, concat, lgamma, logsumexp, mean0, relu, sigmoid  # noqa: E402


def _numeric_grad(fn, x, eps=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        up, dn = x.copy(), x.copy()
        up[idx] += eps
        dn[idx] -= eps
        grad[idx] = (fn(up) - fn(dn)) / (2 * eps)
        it.iternext()
    return grad


def _check(name, build, x0, tol=1e-5):
    def scalar(xv):
        return float(build(Tensor(xv)).data)

    t = Tensor(x0, requires_grad=True)
    out = build(t)
    out.backward()
    analytic = t.grad
    numeric = _numeric_grad(scalar, x0)
    err = np.max(np.abs(analytic - numeric))
    status = "OK " if err < tol else "FAIL"
    print(f"[{status}] {name:28s} max|analytic-numeric| = {err:.3e}")
    return err < tol


def main() -> int:
    rng = np.random.default_rng(20260720)
    ok = True

    x = rng.normal(size=(4, 3))
    W = Tensor(rng.normal(size=(3, 2)), requires_grad=True)
    b = Tensor(rng.normal(size=(1, 2)), requires_grad=True)

    ok &= _check("relu(x@W+b).sum()", lambda t: relu(t.matmul(W) + b).sum(), x)
    ok &= _check("sigmoid(x).sum()", lambda t: sigmoid(t).sum(), x)
    ok &= _check("(x*x).mean()", lambda t: (t * t).mean(), x)
    ok &= _check("log(exp(x)+1).sum()", lambda t: (t.exp() + 1.0).log().sum(), x)
    ok &= _check("logsumexp(flat x)", lambda t: logsumexp(t.reshape(-1)), x)
    ok &= _check("mean0 pooling", lambda t: (mean0(t) * mean0(t)).sum(), x)
    ok &= _check("concat([x,x],axis=1)", lambda t: (concat([t, t], axis=1) ** 1).sum()
                 if False else concat([t, t], axis=1).sum(), x)

    pos = np.abs(rng.normal(size=(5,))) + 1.5
    ok &= _check("lgamma(x).sum()", lambda t: lgamma(t).sum(), pos)

    # A composite resembling the real beta-binomial predictive term.
    y = np.array([3.0, 7.0, 2.0])
    n = np.array([20.0, 30.0, 15.0])

    def bb(disc):
        a = Tensor(1.0) + disc * Tensor(y)
        bt = Tensor(1.0) + disc * Tensor(n - y)
        return (lgamma(a + Tensor(2.0)) + lgamma(bt) - lgamma(a + bt)).sum()

    ok &= _check("beta-binomial-like term", bb, np.array([0.3, 0.6, 0.9]))

    print("\nALL GRADIENT CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
