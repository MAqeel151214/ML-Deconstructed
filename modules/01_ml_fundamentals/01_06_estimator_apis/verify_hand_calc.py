"""Verify fixed teaching calculations for Lesson 01.06."""

import numpy as np
from sklearn.linear_model import LinearRegression


RTOL = 0.0
ATOL = 1e-12


def main() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    y = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float64)
    X_new = np.array([[5.0], [6.0]], dtype=np.float64)

    assert X.shape == (4, 1)
    assert y.shape == (4,)
    assert X_new.shape == (2, 1)
    assert X.dtype == np.float64
    assert y.dtype == np.float64
    assert X_new.dtype == np.float64

    estimator = LinearRegression(fit_intercept=True)
    assert estimator.get_params()["fit_intercept"] is True
    fitted = estimator.fit(X, y)
    assert fitted is estimator
    assert fitted.n_features_in_ == 1
    np.testing.assert_allclose(fitted.coef_, [2.0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fitted.intercept_, 1.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        fitted.predict(X_new), [11.0, 13.0], rtol=RTOL, atol=ATOL
    )

    slope = (9.0 - 3.0) / (4.0 - 1.0)
    intercept = 3.0 - slope * 1.0
    np.testing.assert_allclose(slope, 2.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(intercept, 1.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        [slope * 5.0 + intercept, slope * 6.0 + intercept],
        [11.0, 13.0],
        rtol=RTOL,
        atol=ATOL,
    )
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
