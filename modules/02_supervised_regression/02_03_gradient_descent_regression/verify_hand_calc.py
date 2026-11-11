"""Verify the fixed gradient-descent teaching calculations."""

import numpy as np


RTOL = 0.0
ATOL = 1e-12


def main() -> None:
    X = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
        dtype=np.float64,
    )
    y = np.array([5.0, 1.0, 4.0, 8.0, 3.0], dtype=np.float64)
    A = np.column_stack((np.ones(5, dtype=np.float64), X))
    theta = np.zeros(3, dtype=np.float64)
    eta = np.float64(0.1)
    observed_theta = [theta.copy()]
    observed_gradients = []
    observed_mse = []

    for _ in range(5):
        predictions = np.asarray(A @ theta, dtype=np.float64)
        residuals = np.asarray(predictions - y, dtype=np.float64)
        mse = np.float64(np.mean(residuals**2, dtype=np.float64))
        gradient = np.asarray(
            (np.float64(2.0) / y.size) * (A.T @ residuals),
            dtype=np.float64,
        )
        observed_gradients.append(gradient)
        observed_mse.append(mse)
        theta = np.asarray(theta - eta * gradient, dtype=np.float64)
        observed_theta.append(theta.copy())

    predictions = np.asarray(A @ theta, dtype=np.float64)
    residuals = np.asarray(predictions - y, dtype=np.float64)
    observed_mse.append(np.float64(np.mean(residuals**2, dtype=np.float64)))

    np.testing.assert_allclose(
        observed_theta,
        [
            [0.0, 0.0, 0.0],
            [0.84, 1.12, 0.76],
            [1.136, 1.6064, 0.9152],
            [1.24448, 1.866368, 0.870464],
            [1.2882176, 2.04079616, 0.76456448],
            [1.309501952, 2.1788168192, 0.6446836736],
        ],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        observed_gradients,
        [
            [-8.4, -11.2, -7.6],
            [-2.96, -4.864, -1.552],
            [-1.0848, -2.59968, 0.44736],
            [-0.437376, -1.7442816, 1.0589952],
            [-0.21284352, -1.380206592, 1.198808064],
        ],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        observed_mse,
        [23.0, 5.7552, 3.25568256, 2.574780076032,
         2.168511051300863, 1.8446924257768553],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        predictions, [3.4883187712, 1.9541856256, 4.1330024448,
                      6.311819264, 4.7776861184],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        residuals, [-1.5116812288, 0.9541856256, 0.1330024448,
                     -1.688180736, 1.7776861184],
        rtol=RTOL,
        atol=ATOL,
    )
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
