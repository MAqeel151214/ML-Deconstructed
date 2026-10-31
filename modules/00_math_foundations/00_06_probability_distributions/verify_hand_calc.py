"""Verify fixed teaching calculations for lesson 00.06."""

import numpy as np


RTOL = 1e-12
ATOL = 1e-12


def main() -> None:
    outcomes = np.array([0.0, 1.0], dtype=np.float64)
    probabilities = np.array([0.4, 0.6], dtype=np.float64)

    expectation = np.sum(outcomes * probabilities)
    second_moment = np.sum(outcomes**2 * probabilities)
    variance = second_moment - expectation**2
    std_deviation = np.sqrt(variance)
    direct_variance = np.sum((outcomes - expectation) ** 2 * probabilities)
    interval_probability = (0.75 - 0.25) * 1.0

    assert outcomes.shape == (2,)
    assert probabilities.shape == (2,)
    assert outcomes.dtype == np.float64
    assert probabilities.dtype == np.float64
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(), 1.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(expectation, 0.6, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(second_moment, 0.6, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(variance, 0.24, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        std_deviation, 0.4898979485566356, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(direct_variance, 0.24, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(interval_probability, 0.5, rtol=RTOL, atol=ATOL)

    print("PASS")


if __name__ == "__main__":
    main()
