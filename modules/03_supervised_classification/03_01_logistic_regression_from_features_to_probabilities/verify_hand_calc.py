"""Deterministic verification for 03.01 logistic regression.

This script checks the frozen forward-pass arithmetic for the lesson
"Logistic Regression: From Features to Probabilities". It uses NumPy for the
primary implementation and pure-Python math.exp as an independent cross-check.
"""

from __future__ import annotations

import math

import numpy as np

RTOL = 0.0
ATOL = 1e-12


X = np.array(
    [[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
    dtype=np.float64,
)
y = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
b = -1.5
w = np.array([1.0], dtype=np.float64)

EXPECTED_Z = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=np.float64)
EXPECTED_EXP_NEG_Z = np.array(
    [
        33.1154519586923,
        12.1824939607035,
        4.48168907033806,
        1.64872127070013,
        0.606530659712633,
        0.22313016014843,
        0.0820849986238988,
        0.0301973834223185,
    ],
    dtype=np.float64,
)
EXPECTED_DENOMINATOR = np.array(
    [
        34.1154519586923,
        13.1824939607035,
        5.48168907033806,
        2.64872127070013,
        1.60653065971263,
        1.22313016014843,
        1.0820849986239,
        1.03019738342232,
    ],
    dtype=np.float64,
)
EXPECTED_P = np.array(
    [
        0.0293122307513563,
        0.0758581800212435,
        0.182425523806356,
        0.377540668798145,
        0.622459331201855,
        0.817574476193644,
        0.924141819978757,
        0.970687769248644,
    ],
    dtype=np.float64,
)
EXPECTED_Y_HAT = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
EXPECTED_CORRECT = np.array([True, True, True, False, False, True, True, True])


def sigmoid(z: np.ndarray | np.floating | float) -> np.ndarray | np.floating:
    """Logistic sigmoid: sigma(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def validate_binary_classification_inputs(
    X_value: np.ndarray,
    y_value: np.ndarray,
    b_value: float,
    w_value: np.ndarray,
) -> None:
    """Validate lesson input shapes and values in a deterministic order."""
    X_arr = np.asarray(X_value)
    y_arr = np.asarray(y_value)
    w_arr = np.asarray(w_value)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (N, p)")
    if w_arr.ndim != 1 or w_arr.shape[0] != X_arr.shape[1]:
        raise ValueError("w must have shape (p,) matching X's feature count")
    if y_arr.ndim != 1 or y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("y must have shape (N,) matching X's example count")
    if X_arr.shape[0] < 1 or X_arr.shape[1] < 1:
        raise ValueError("inputs must be non-empty")
    if not (
        np.all(np.isfinite(X_arr))
        and np.all(np.isfinite(y_arr))
        and np.isfinite(b_value)
        and np.all(np.isfinite(w_arr))
    ):
        raise ValueError("inputs must all be finite")
    if not np.all((y_arr == 0.0) | (y_arr == 1.0)):
        raise ValueError("target y must contain only 0 and 1")


def forward_logistic(
    X_value: np.ndarray,
    y_value: np.ndarray,
    b_value: float,
    w_value: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray | np.floating]:
    """Evaluate the fixed logistic-regression model; this function does not train."""
    validate_binary_classification_inputs(X_value, y_value, b_value, w_value)
    z_value = b_value + X_value @ w_value
    exp_neg_z_value = np.exp(-z_value)
    denominator_value = 1.0 + exp_neg_z_value
    p_value = 1.0 / denominator_value
    y_hat_value = (p_value >= threshold).astype(int)
    accuracy_value = np.mean(y_hat_value == y_value)
    return {
        "z": z_value,
        "exp_neg_z": exp_neg_z_value,
        "denominator": denominator_value,
        "p": p_value,
        "y_hat": y_hat_value,
        "accuracy": accuracy_value,
    }


def pure_python_probabilities() -> list[float]:
    """Independent math.exp path that does not call sigmoid or forward_logistic."""
    probabilities: list[float] = []
    for row in X:
        z_i = b + w[0] * float(row[0])
        probabilities.append(1.0 / (1.0 + math.exp(-z_i)))
    return probabilities


def assert_raises_message(func, expected_message: str) -> None:
    try:
        func()
    except ValueError as exc:
        assert expected_message in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected ValueError containing {expected_message!r}")


def test_primary_forward_pass() -> None:
    result = forward_logistic(X, y, b, w)
    z_value = result["z"]
    exp_neg_z_value = result["exp_neg_z"]
    denominator_value = result["denominator"]
    p_value = result["p"]
    y_hat_value = result["y_hat"]
    accuracy_value = result["accuracy"]

    assert X.shape == (8, 1)
    assert y.shape == (8,)
    assert w.shape == (1,)
    assert X.dtype == np.float64
    assert y.dtype == np.float64
    assert w.dtype == np.float64
    assert b == -1.5

    np.testing.assert_array_equal(z_value, EXPECTED_Z)
    np.testing.assert_allclose(exp_neg_z_value, EXPECTED_EXP_NEG_Z, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(denominator_value, EXPECTED_DENOMINATOR, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(p_value, EXPECTED_P, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(y_hat_value, EXPECTED_Y_HAT)
    np.testing.assert_array_equal(y_hat_value == y, EXPECTED_CORRECT)
    assert accuracy_value == 0.75


def test_sigmoid_properties() -> None:
    assert sigmoid(0.0) == 0.5
    grid = np.arange(-6.0, 6.0 + 0.25, 0.25, dtype=np.float64)
    grid_p = sigmoid(grid)
    assert np.all((grid_p > 0.0) & (grid_p < 1.0))
    assert np.all(np.diff(grid_p) > 0.0)
    for value in [0.5, 1.5, 2.5, 3.5]:
        np.testing.assert_allclose(sigmoid(-value) + sigmoid(value), 1.0, rtol=RTOL, atol=ATOL)


def test_boundary_and_new_inputs() -> None:
    assert b + w[0] * 1.5 == 0.0
    assert sigmoid(0.0) == 0.5

    new_X = np.array([[1.5], [-4.0]], dtype=np.float64)
    new_z = b + new_X @ w
    new_p = sigmoid(new_z)
    new_y_hat = (new_p >= 0.5).astype(int)

    np.testing.assert_array_equal(new_z, np.array([0.0, -5.5], dtype=np.float64))
    np.testing.assert_allclose(
        new_p,
        np.array([0.5, 0.004070137715896128], dtype=np.float64),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_array_equal(new_y_hat, np.array([1, 0], dtype=int))


def test_pure_python_cross_check() -> None:
    np.testing.assert_allclose(pure_python_probabilities(), EXPECTED_P, rtol=RTOL, atol=ATOL)


def test_threshold_demo() -> None:
    result = forward_logistic(X, y, b, w, threshold=0.7)
    np.testing.assert_array_equal(result["y_hat"], np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int))
    assert result["accuracy"] == 0.875


def test_exercises() -> None:
    # E1
    z_e1 = -1.5 + 1.0 * 0.5
    p_e1 = 1.0 / (1.0 + math.exp(-z_e1))
    assert z_e1 == -1.0
    np.testing.assert_allclose(p_e1, 0.2689414213699951, rtol=RTOL, atol=ATOL)
    assert int(p_e1 >= 0.5) == 0

    # E2
    b_e2 = 2.0
    w_e2 = -0.5
    boundary_e2 = -b_e2 / w_e2
    z_e2 = b_e2 + w_e2 * 8.0
    p_e2 = 1.0 / (1.0 + math.exp(-z_e2))
    assert boundary_e2 == 4.0
    assert z_e2 == -2.0
    np.testing.assert_allclose(p_e2, 0.11920292202211755, rtol=RTOL, atol=ATOL)
    assert int(p_e2 >= 0.5) == 0

    # E3
    X_ex = np.array(
        [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [3.0, 3.0], [4.0, 3.0], [3.0, 4.0], [4.0, 4.0]],
        dtype=np.float64,
    )
    y_ex = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    z_ex = -4.0 + X_ex @ np.array([1.0, 1.0], dtype=np.float64)
    p_ex = sigmoid(z_ex)
    y_hat_ex = (p_ex >= 0.5).astype(int)
    expected_z_ex = np.array([-2.0, -1.0, -1.0, 0.0, 2.0, 3.0, 3.0, 4.0], dtype=np.float64)
    expected_p_ex = np.array(
        [
            0.119202922022118,
            0.268941421369995,
            0.268941421369995,
            0.5,
            0.880797077977882,
            0.952574126822433,
            0.952574126822433,
            0.982013790037908,
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(z_ex, expected_z_ex)
    np.testing.assert_allclose(p_ex, expected_p_ex, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(y_hat_ex, np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=int))
    assert np.mean(y_hat_ex == y_ex) == 0.875
    assert -4.0 + 3.0 + 1.0 == 0.0


def test_validation_errors() -> None:
    assert_raises_message(
        lambda: forward_logistic(X.reshape(8), y, b, w),
        "X must be a 2-D array of shape (N, p)",
    )
    assert_raises_message(
        lambda: forward_logistic(X, y, b, np.array([[1.0]], dtype=np.float64)),
        "w must have shape (p,) matching X's feature count",
    )
    assert_raises_message(
        lambda: forward_logistic(X, y[:-1], b, w),
        "y must have shape (N,) matching X's example count",
    )
    assert_raises_message(
        lambda: forward_logistic(np.empty((0, 1), dtype=np.float64), np.empty((0,), dtype=np.float64), b, w),
        "inputs must be non-empty",
    )
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    assert_raises_message(
        lambda: forward_logistic(X_nan, y, b, w),
        "inputs must all be finite",
    )
    y_bad = y.copy()
    y_bad[0] = 2.0
    assert_raises_message(
        lambda: forward_logistic(X, y_bad, b, w),
        "target y must contain only 0 and 1",
    )


def test_float64_edge_facts() -> None:
    rounding_cutover = 53.0 * math.log(2.0)
    np.testing.assert_allclose(rounding_cutover, 36.7368005696771, rtol=RTOL, atol=ATOL)
    assert sigmoid(36.73) < 1.0
    assert sigmoid(36.74) == 1.0
    overflow_edge = math.log(np.finfo(np.float64).max)
    np.testing.assert_allclose(overflow_edge, 709.782712893384, rtol=RTOL, atol=ATOL)


def main() -> None:
    test_primary_forward_pass()
    test_sigmoid_properties()
    test_boundary_and_new_inputs()
    test_pure_python_cross_check()
    test_threshold_demo()
    test_exercises()
    test_validation_errors()
    test_float64_edge_facts()
    print("PASS: 03.01 logistic-regression hand calculations verified.")


if __name__ == "__main__":
    main()
