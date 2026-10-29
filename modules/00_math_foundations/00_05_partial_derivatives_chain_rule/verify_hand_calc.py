"""Verify fixed teaching calculations for lesson 00.05."""

from __future__ import annotations

import numpy as np


RTOL = 1e-12
ATOL = 1e-12


def main() -> None:
    x = np.float64(3.0)
    y = np.float64(2.0)
    u = x + 2.0 * y
    f = u * u
    outer_rate = 2.0 * u
    inner_gradient = np.array([1.0, 2.0], dtype=np.float64)
    chain_gradient = outer_rate * inner_gradient

    expanded_output = x * x + 4.0 * x * y + 4.0 * y * y
    expanded_gradient = np.array(
        [2.0 * x + 4.0 * y, 4.0 * x + 8.0 * y], dtype=np.float64
    )

    assert np.asarray(u).shape == ()
    assert np.asarray(f).shape == ()
    assert np.asarray(outer_rate).shape == ()
    assert inner_gradient.shape == (2,)
    assert chain_gradient.shape == (2,)
    assert chain_gradient.dtype == np.float64

    np.testing.assert_allclose(u, 7.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(f, 49.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(outer_rate, 14.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        chain_gradient, [14.0, 28.0], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        expanded_output, 49.0, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        expanded_gradient, [14.0, 28.0], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        chain_gradient, expanded_gradient, rtol=RTOL, atol=ATOL
    )

    bridge_x = np.array([1.0, 2.0], dtype=np.float64)
    weights = np.array([1.0, 1.0], dtype=np.float64)
    prediction = weights @ bridge_x
    residual = prediction - np.float64(5.0)
    loss = 0.5 * residual * residual
    loss_gradient = residual * bridge_x
    np.testing.assert_allclose(prediction, 3.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(residual, -2.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, 2.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        loss_gradient, [-2.0, -4.0], rtol=RTOL, atol=ATOL
    )

    print("PASS")


if __name__ == "__main__":
    main()
