"""Verify fixed teaching calculations for lesson 00.04.

The vectors and scalar values are dimensionless teaching quantities. This
script verifies prediction, residual, loss, one-variable derivative, both
partial derivatives, and the gradient against frozen hand-derived results.

It deliberately does not use autodiff, symbolic algebra, finite differences,
an optimizer, or any library beyond NumPy.
"""

from __future__ import annotations

import numpy as np


# ── frozen inputs ─────────────────────────────────────────────────────

X = np.array([1.0, 2.0])
W = np.array([1.0, 1.0])
Y = np.float64(5.0)


# ── shape guards ──────────────────────────────────────────────────────


def require_vector(array: np.ndarray, length: int, name: str) -> np.ndarray:
    """Return a float64 rank-1 array of exact length, or raise."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.shape != (length,):
        raise ValueError(
            f"{name} must have shape ({length},); received shape {arr.shape}"
        )
    return arr


def require_scalar(value: np.floating | float, name: str) -> np.floating:
    """Return a float64 scalar (shape ()) or raise."""
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise ValueError(
            f"{name} must be a scalar with shape (); received shape {arr.shape}"
        )
    return np.float64(arr)


# ── componentwise gradient from expanded polynomial ───────────────────


def partial_w1_expanded(w1: np.floating, w2: np.floating) -> np.floating:
    """∂L/∂w₁ = w₁ + 2w₂ − 5  (from term-by-term expansion)."""
    return w1 + np.float64(2.0) * w2 - np.float64(5.0)


def partial_w2_expanded(w1: np.floating, w2: np.floating) -> np.floating:
    """∂L/∂w₂ = 2w₁ + 4w₂ − 10  (from term-by-term expansion)."""
    return np.float64(2.0) * w1 + np.float64(4.0) * w2 - np.float64(10.0)


# ── verification ──────────────────────────────────────────────────────


def main() -> None:
    """Run deterministic shape, arithmetic, gradient, and cross-check tests."""

    x = require_vector(X, 2, "x")
    w = require_vector(W, 2, "w")
    y = require_scalar(Y, "y")

    # Shapes
    assert x.shape == (2,), f"x shape: {x.shape}"
    assert w.shape == (2,), f"w shape: {w.shape}"
    assert np.ndim(y) == 0, f"y ndim: {np.ndim(y)}"

    # Forward pass
    y_hat = w @ x
    r = y_hat - y
    L = np.float64(0.5) * r * r

    assert np.ndim(y_hat) == 0
    assert np.ndim(r) == 0
    assert np.ndim(L) == 0

    np.testing.assert_allclose(y_hat, 3.0)
    np.testing.assert_allclose(r, -2.0)
    np.testing.assert_allclose(L, 2.0)

    # One-variable warm-up
    t = np.float64(3.0)
    q_val = np.float64(0.5) * (t - np.float64(5.0)) ** 2
    qp_val = t - np.float64(5.0)

    np.testing.assert_allclose(q_val, 2.0)
    np.testing.assert_allclose(qp_val, -2.0)

    # Componentwise gradient (expanded polynomial)
    dL_dw1 = partial_w1_expanded(w[0], w[1])
    dL_dw2 = partial_w2_expanded(w[0], w[1])
    grad_expanded = np.array([dL_dw1, dL_dw2])

    assert grad_expanded.shape == (2,)
    np.testing.assert_allclose(dL_dw1, -2.0)
    np.testing.assert_allclose(dL_dw2, -4.0)

    # Vector-formula gradient (independent cross-check)
    grad_vector = (w @ x - y) * x

    assert grad_vector.shape == (2,)
    np.testing.assert_allclose(grad_vector, np.array([-2.0, -4.0]))

    # Both paths agree
    np.testing.assert_allclose(grad_expanded, grad_vector)

    print("PASS")


if __name__ == "__main__":
    main()
