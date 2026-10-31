"""Verify fixed teaching calculations for lesson 00.07."""

from fractions import Fraction

import numpy as np


RTOL = 1e-12
ATOL = 0.0


def main() -> None:
    priors = np.array([0.01, 0.99], dtype=np.float64)
    likelihood = np.array([[0.90, 0.10], [0.05, 0.95]], dtype=np.float64)
    counts = np.array([[180, 20], [990, 18810]], dtype=np.int64)

    joint = priors[:, None] * likelihood
    marginal = joint.sum(axis=0)
    assert np.all(marginal > 0)
    posterior = joint / marginal[None, :]

    assert priors.shape == (2,)
    assert likelihood.shape == (2, 2)
    assert joint.shape == (2, 2)
    assert marginal.shape == (2,)
    assert posterior.shape == (2, 2)
    np.testing.assert_allclose(joint, [[0.009, 0.001], [0.0495, 0.9405]], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(marginal, [0.0585, 0.9415], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(posterior[:, 0], [2 / 13, 11 / 13], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        posterior[:, 1], [2 / 1883, 1881 / 1883], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(posterior.sum(axis=0), [1.0, 1.0], rtol=RTOL, atol=ATOL)
    assert Fraction(9, 1000) / Fraction(117, 2000) == Fraction(2, 13)
    assert counts.sum() == 20_000
    assert Fraction(int(counts[0, 0]), int(counts[:, 0].sum())) == Fraction(2, 13)

    print("PASS")


if __name__ == "__main__":
    main()
