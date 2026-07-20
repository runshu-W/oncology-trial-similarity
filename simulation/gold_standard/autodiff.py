"""Minimal reverse-mode automatic differentiation on NumPy arrays.

Written because the analysis environment has no PyTorch. Supports exactly the
operations needed by the two-head DeepSets borrowing model:
linear layers, ReLU, sigmoid, mean-pooling, concatenation, log/exp,
log-sum-exp and lgamma (whose derivative is digamma).

All gradients are verified against central finite differences in
``test_autodiff.py``.
"""

from __future__ import annotations

import math as _math

import numpy as np

__all__ = ["Tensor", "relu", "sigmoid", "concat", "mean0", "lgamma", "logsumexp", "digamma", "gammaln"]


def digamma(x: np.ndarray) -> np.ndarray:
    """Vectorised digamma (psi) function.

    Uses upward recurrence psi(x) = psi(x+1) - 1/x until x >= 6, then the
    standard asymptotic expansion.
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    y = x.copy()
    # Recurrence to push the argument into the asymptotic regime.
    for _ in range(24):
        mask = y < 6.0
        if not mask.any():
            break
        result[mask] -= 1.0 / y[mask]
        y[mask] += 1.0
    inv = 1.0 / y
    inv2 = inv * inv
    series = (
        -1.0 / 12.0
        + inv2 * (1.0 / 120.0 + inv2 * (-1.0 / 252.0 + inv2 * (1.0 / 240.0 + inv2 * (-1.0 / 132.0))))
    )
    result += np.log(y) - 0.5 * inv + inv2 * series
    return result


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum a gradient back down to ``shape`` after NumPy broadcasting."""
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the autodiff graph."""

    __slots__ = ("data", "grad", "_backward", "_parents", "requires_grad")

    def __init__(self, data, requires_grad: bool = False, _parents=(), _backward=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad) or any(p.requires_grad for p in _parents)
        self.grad = None
        self._parents = tuple(_parents)
        self._backward = _backward

    # -- construction helpers -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape})"

    def _make(self, data, parents, backward):
        return Tensor(data, _parents=parents, _backward=backward)

    # -- arithmetic -----------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data + other.data

        def backward(g):
            return (_unbroadcast(g, self.data.shape), _unbroadcast(g, other.data.shape))

        return self._make(out_data, (self, other), backward)

    __radd__ = __add__

    def __neg__(self):
        return self._make(-self.data, (self,), lambda g: (-g,))

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data * other.data

        def backward(g):
            return (
                _unbroadcast(g * other.data, self.data.shape),
                _unbroadcast(g * self.data, other.data.shape),
            )

        return self._make(out_data, (self, other), backward)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data / other.data

        def backward(g):
            return (
                _unbroadcast(g / other.data, self.data.shape),
                _unbroadcast(-g * self.data / (other.data ** 2), other.data.shape),
            )

        return self._make(out_data, (self, other), backward)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data @ other.data

        def backward(g):
            return (g @ other.data.T, self.data.T @ g)

        return self._make(out_data, (self, other), backward)

    __matmul__ = matmul

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)

        def backward(g):
            grad = np.asarray(g, dtype=np.float64)
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            return (np.broadcast_to(grad, self.data.shape).copy(),)

        return self._make(out_data, (self,), backward)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def log(self):
        out_data = np.log(self.data)
        return self._make(out_data, (self,), lambda g: (g / self.data,))

    def exp(self):
        out_data = np.exp(self.data)
        return self._make(out_data, (self,), lambda g: (g * out_data,))

    def reshape(self, *shape):
        original = self.data.shape
        out_data = self.data.reshape(*shape)
        return self._make(out_data, (self,), lambda g: (g.reshape(original),))

    # -- backward pass --------------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                build(parent)
            topo.append(node)

        build(self)
        for node in topo:
            node.grad = None
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node._backward is None or node.grad is None:
                continue
            grads = node._backward(node.grad)
            for parent, grad in zip(node._parents, grads):
                if grad is None or not parent.requires_grad:
                    continue
                parent.grad = grad.copy() if parent.grad is None else parent.grad + grad


# -- free functions -----------------------------------------------------------
def relu(t: Tensor) -> Tensor:
    out_data = np.maximum(t.data, 0.0)
    mask = (t.data > 0.0).astype(np.float64)
    return Tensor(out_data, _parents=(t,), _backward=lambda g: (g * mask,))


def sigmoid(t: Tensor) -> Tensor:
    out_data = 1.0 / (1.0 + np.exp(-t.data))
    return Tensor(out_data, _parents=(t,), _backward=lambda g: (g * out_data * (1.0 - out_data),))


_lgamma_vec = np.vectorize(_math.lgamma, otypes=[np.float64])


def gammaln(x: np.ndarray) -> np.ndarray:
    """Vectorised log-gamma built on the standard library (no SciPy needed)."""
    return _lgamma_vec(np.asarray(x, dtype=np.float64))


def lgamma(t: Tensor) -> Tensor:
    out_data = gammaln(t.data)
    psi = digamma(t.data)
    return Tensor(out_data, _parents=(t,), _backward=lambda g: (g * psi,))


def concat(tensors, axis=1) -> Tensor:
    datas = [t.data for t in tensors]
    out_data = np.concatenate(datas, axis=axis)
    sizes = [d.shape[axis] for d in datas]

    def backward(g):
        splits, start = [], 0
        for size in sizes:
            idx = [slice(None)] * g.ndim
            idx[axis] = slice(start, start + size)
            splits.append(g[tuple(idx)].copy())
            start += size
        return tuple(splits)

    return Tensor(out_data, _parents=tuple(tensors), _backward=backward)


def mean0(t: Tensor) -> Tensor:
    """Mean over axis 0, keeping dims (the DeepSets pooling operation)."""
    return t.mean(axis=0, keepdims=True)


def logsumexp(t: Tensor) -> Tensor:
    """Log-sum-exp over a flat vector."""
    shift = float(np.max(t.data))
    out_data = shift + np.log(np.exp(t.data - shift).sum())
    softmax = np.exp(t.data - out_data)
    return Tensor(out_data, _parents=(t,), _backward=lambda g: (g * softmax,))
