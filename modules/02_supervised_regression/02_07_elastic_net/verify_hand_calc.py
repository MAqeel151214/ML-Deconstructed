"""Independent verifier for Lesson 02.07 frozen arithmetic."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge


RTOL = 0.0
ATOL = 1e-8
SKLEARN_ATOL = 1e-6

X = np.array(
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
    dtype=np.float64,
)
y = np.array([5.0, 1.0, 4.0, 8.0, 3.0], dtype=np.float64)

EXPECTED = {
    "OLS": np.array([1.7, 3.5, -1.0], dtype=np.float64),
    "Ridge": np.array([3.2, 1.4, -0.4], dtype=np.float64),
    "Lasso": np.array([2.7, 1.5, 0.0], dtype=np.float64),
    "ENet-A": np.array([3.2, 1.2, -0.2], dtype=np.float64),
    "ENet-B": np.array([3.2, 1.0, 0.0], dtype=np.float64),
}

EXPECTED_PREDICTIONS = {
    "OLS": np.array([5.2, 0.7, 4.2, 7.7, 3.2], dtype=np.float64),
    "Ridge": np.array([4.6, 2.8, 4.2, 5.6, 3.8], dtype=np.float64),
    "Lasso": np.array([4.2, 2.7, 4.2, 5.7, 4.2], dtype=np.float64),
    "ENet-A": np.array([4.4, 3.0, 4.2, 5.4, 4.0], dtype=np.float64),
    "ENet-B": np.array([4.2, 3.2, 4.2, 5.2, 4.2], dtype=np.float64),
}

SETTINGS = {
    "OLS": (0.0, 0.0),
    "Ridge": (0.0, 0.3),
    "Lasso": (0.8, 0.0),
    "ENet-A": (0.2, 0.3),
    "ENet-B": (0.8, 0.1),
}


def soft_threshold(z: float, gamma: float) -> float:
    return float(np.sign(z) * max(abs(z) - gamma, 0.0))


def closed_form(lambda_1: float, lambda_2: float) -> np.ndarray:
    coefficients = np.array(
        [soft_threshold(1.4, lambda_1), soft_threshold(-0.4, lambda_1)],
        dtype=np.float64,
    )
    coefficients /= 0.4 + 2.0 * lambda_2
    return np.r_[4.2 - coefficients.sum(), coefficients]


def objective(theta: np.ndarray, lambda_1: float, lambda_2: float) -> tuple[float, ...]:
    residuals = y - (theta[0] + X @ theta[1:])
    rss = float(np.sum(residuals**2))
    mse = rss / y.size
    l1 = lambda_1 * float(np.sum(np.abs(theta[1:])))
    l2 = lambda_2 * float(np.sum(theta[1:] ** 2))
    return rss, mse, 0.5 * mse, l1, l2, 0.5 * mse + l1 + l2


def main() -> None:
    centered_x = X - X.mean(axis=0)
    centered_y = y - y.mean()
    np.testing.assert_allclose(
        centered_x.T @ centered_x, [[2.0, 0.0], [0.0, 2.0]], rtol=RTOL, atol=1e-12
    )
    np.testing.assert_allclose(
        centered_x.T @ centered_y, [7.0, -2.0], rtol=RTOL, atol=1e-12
    )

    expected_objectives = {
        "OLS": (0.30, 0.06, 0.03),
        "Ridge": (9.84, 1.968, 1.62),
        "Lasso": (10.30, 2.06, 2.23),
        "ENet-A": (12.16, 2.432, 1.94),
        "ENet-B": (14.80, 2.96, 2.38),
    }

    for name, (lambda_1, lambda_2) in SETTINGS.items():
        theta = closed_form(lambda_1, lambda_2)
        np.testing.assert_allclose(theta, EXPECTED[name], rtol=RTOL, atol=ATOL)
        predictions = theta[0] + X @ theta[1:]
        residuals = y - predictions
        np.testing.assert_allclose(
            predictions, EXPECTED_PREDICTIONS[name], rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            residuals, y - EXPECTED_PREDICTIONS[name], rtol=RTOL, atol=ATOL
        )
        values = objective(theta, lambda_1, lambda_2)
        np.testing.assert_allclose(
            [values[0], values[1], values[-1]],
            expected_objectives[name],
            rtol=RTOL,
            atol=ATOL,
        )

        if name != "OLS":
            alpha = lambda_1 + 2.0 * lambda_2
            ratio = lambda_1 / alpha
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=ratio,
                    fit_intercept=True,
                    max_iter=100_000,
                    tol=1e-10,
                ).fit(X, y)
            np.testing.assert_allclose(
                np.r_[model.intercept_, model.coef_],
                theta,
                rtol=RTOL,
                atol=SKLEARN_ATOL,
            )

    ridge = Ridge(alpha=3.0, fit_intercept=True).fit(X, y)
    np.testing.assert_allclose(
        np.r_[ridge.intercept_, ridge.coef_],
        EXPECTED["Ridge"],
        rtol=RTOL,
        atol=SKLEARN_ATOL,
    )
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
