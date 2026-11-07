"""Verify fixed teaching calculations for Lesson 01.05."""

import numpy as np


RTOL = 0.0
ATOL = 1e-12


def main() -> None:
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([2.0, 4.0], dtype=np.float64)
    eta = np.float64(0.1)
    parameter = np.float64(0.0)

    expected_losses = np.array([10.0, 2.5, 0.625], dtype=np.float64)
    expected_gradients = np.array([-10.0, -5.0, -2.5], dtype=np.float64)
    expected_after = np.array([1.0, 1.5, 1.75], dtype=np.float64)
    observed_losses = []
    observed_gradients = []
    observed_after = []

    for _ in range(3):
        residuals = y - parameter * x
        loss = np.float64(0.5 * np.sum(residuals**2, dtype=np.float64))
        slope = np.float64(-np.sum(x * residuals, dtype=np.float64))
        parameter = np.float64(parameter - eta * slope)
        observed_losses.append(loss)
        observed_gradients.append(slope)
        observed_after.append(parameter)

    np.testing.assert_allclose(observed_losses, expected_losses, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        observed_gradients, expected_gradients, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(observed_after, expected_after, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        0.5 * ((2.0 - 1.75) ** 2 + (4.0 - 3.5) ** 2),
        0.15625,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(10.0 / 5.0, 2.0, rtol=RTOL, atol=ATOL)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
