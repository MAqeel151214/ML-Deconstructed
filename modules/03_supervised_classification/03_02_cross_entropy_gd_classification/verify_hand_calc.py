"""Deterministic verification for 03.02 Cross-Entropy & GD for Classification.

This script checks the frozen arithmetic and gradient descent optimization
trajectory for the lesson "Cross-Entropy & GD for Classification".
It uses NumPy for the primary vectorized implementation and pure-Python math
functions (math.exp, math.log) as an independent cross-check.

Environment: Python 3.10+, numpy 1.26+, scikit-learn 1.5+
Tolerances: RTOL = 0.0, ATOL = 1e-12 (matching project standard)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

RTOL = 0.0
ATOL = 1e-12

# ---------------------------------------------------------------------------
# Frozen 8-point dataset (linearly separable: max x|y=0 = 1.0 < min x|y=1 = 2.0)
# ---------------------------------------------------------------------------
x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
N = len(x)
A = np.column_stack([np.ones(N, dtype=np.float64), x])

# ---------------------------------------------------------------------------
# Frozen Reference Tables T1 through T6 (W0 / W1 / W2 Canon)
# ---------------------------------------------------------------------------

# Table T1: States 0 through 10
EXPECTED_T1_THETA = np.array(
    [
        [0.000000, 0.000000],
        [-0.012500, 0.087500],
        [-0.025773, 0.163199],
        [-0.039622, 0.229048],
        [-0.053874, 0.286796],
        [-0.068397, 0.337895],
        [-0.083089, 0.383510],
        [-0.097877, 0.424563],
        [-0.112707, 0.461783],
        [-0.127538, 0.495753],
        [-0.142342, 0.526940],
    ],
    dtype=np.float64,
)

EXPECTED_T1_LOSS = np.array(
    [
        0.693147,
        0.620153,
        0.564805,
        0.522192,
        0.488733,
        0.461917,
        0.439996,
        0.421748,
        0.406306,
        0.393046,
        0.381510,
    ],
    dtype=np.float64,
)

EXPECTED_T1_BOUNDARY = np.array(
    [
        np.nan,
        0.142857,
        0.157927,
        0.172985,
        0.187850,
        0.202421,
        0.216655,
        0.230537,
        0.244069,
        0.257261,
        0.270129,
    ],
    dtype=np.float64,
)

EXPECTED_T1_GRADIENT = np.array(
    [
        [0.125000, -0.875000],
        [0.132734, -0.756989],
        [0.138485, -0.658488],
        [0.142526, -0.577482],
        [0.145227, -0.510994],
        [0.146923, -0.456149],
        [0.147879, -0.410525],
        [0.148294, -0.372205],
        [0.148312, -0.339702],
        [0.148038, -0.311870],
        [0.147547, -0.287826],
    ],
    dtype=np.float64,
)

EXPECTED_T1_ACCURACY = np.array(
    [0.375, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875],
    dtype=np.float64,
)

# Table T2: State 0 Per-Example Audit
EXPECTED_T2_P = np.full(8, 0.5, dtype=np.float64)
EXPECTED_T2_RESIDUAL = np.array([0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5], dtype=np.float64)
EXPECTED_T2_RESIDUAL_X = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, -1.0, -1.5, -2.0], dtype=np.float64)
EXPECTED_T2_LOSS_I = np.full(8, np.log(2.0), dtype=np.float64)

# Table T3: State 1 Per-Example Audit (theta = [-0.0125, 0.0875])
EXPECTED_T3_Z = np.array([-0.275000, -0.187500, -0.100000, -0.012500, 0.075000, 0.162500, 0.250000, 0.337500], dtype=np.float64)
EXPECTED_T3_P = np.array([0.431680, 0.453262, 0.475021, 0.496875, 0.518741, 0.540536, 0.562177, 0.583583], dtype=np.float64)
EXPECTED_T3_RESIDUAL = np.array([0.431680, 0.453262, 0.475021, 0.496875, 0.518741, -0.459464, -0.437823, -0.416417], dtype=np.float64)
EXPECTED_T3_RESIDUAL_X = np.array([-1.295040, -0.906524, -0.475021, 0.000000, 0.518741, -0.918928, -1.313470, -1.665668], dtype=np.float64)
EXPECTED_T3_LOSS_I = np.array([0.565071, 0.603785, 0.644397, 0.686917, 0.731350, 0.615194, 0.575939, 0.538568], dtype=np.float64)

# Table T4: State 2 Per-Example Audit (theta = [-0.025773, 0.163199])
EXPECTED_T4_Z = np.array([-0.515370, -0.352171, -0.188972, -0.025773, 0.137425, 0.300624, 0.463823, 0.627022], dtype=np.float64)
EXPECTED_T4_P = np.array([0.373936, 0.412856, 0.452897, 0.493557, 0.534302, 0.574595, 0.613921, 0.651814], dtype=np.float64)
EXPECTED_T4_RESIDUAL = np.array([0.373936, 0.412856, 0.452897, 0.493557, 0.534302, -0.425405, -0.386079, -0.348186], dtype=np.float64)
EXPECTED_T4_RESIDUAL_X = np.array([-1.121807, -0.825712, -0.452897, 0.000000, 0.534302, -0.850810, -1.158238, -1.392744], dtype=np.float64)
EXPECTED_T4_LOSS_I = np.array([0.468302, 0.532485, 0.603118, 0.680343, 0.764219, 0.554090, 0.487889, 0.427996], dtype=np.float64)

# Long-Run Milestones (Table T5)
EXPECTED_K139_THETA = np.array([-1.389114, 1.388957], dtype=np.float64)
EXPECTED_K139_BOUNDARY = 1.000113
EXPECTED_K139_LOSS = 0.161692

EXPECTED_K5000_THETA = np.array([-7.004738, 4.844064], dtype=np.float64)
EXPECTED_K5000_BOUNDARY = 1.446046
EXPECTED_K5000_LOSS = 0.022079


# ===================================================================
# Core Teaching Functions
# ===================================================================

def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Vectorized logistic sigmoid: sigma(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy_loss(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Mean binary cross-entropy loss with numerical guardrail."""
    p_clipped = np.clip(p_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped)))


def cross_entropy_gradient(A_mat: np.ndarray, y_true: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    """Batch gradient: (1 / N) * A.T @ (p - y)."""
    return (1.0 / A_mat.shape[0]) * (A_mat.T @ (p_pred - y_true))


def validate_logistic_inputs(
    A_mat: np.ndarray,
    y_true: np.ndarray,
    eta_val: float,
    n_iter_val: int,
) -> None:
    """Validate shapes, values, and hyperparameters deterministically."""
    A_arr = np.asarray(A_mat, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.float64)

    if A_arr.ndim != 2 or A_arr.shape[1] != 2:
        raise ValueError("Design matrix A must have shape (N, 2)")
    if y_arr.ndim != 1 or y_arr.shape[0] != A_arr.shape[0]:
        raise ValueError("Target y must have shape (N,) matching A")
    if not np.all((y_arr == 0.0) | (y_arr == 1.0)):
        raise ValueError("Target y must contain only binary labels 0 and 1")
    if not (np.all(np.isfinite(A_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("Inputs must contain only finite numbers")
    if not np.isfinite(eta_val) or eta_val <= 0:
        raise ValueError("Learning rate eta must be positive and finite")
    if n_iter_val < 1:
        raise ValueError("Number of iterations must be at least 1")


def train_logistic_gd(
    A_mat: np.ndarray,
    y_true: np.ndarray,
    eta_val: float,
    n_iter_val: int,
    theta0: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Train logistic regression with full-batch gradient descent."""
    validate_logistic_inputs(A_mat, y_true, eta_val, n_iter_val)
    A_arr = np.asarray(A_mat, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.float64)

    theta = np.zeros(2, dtype=np.float64) if theta0 is None else np.asarray(theta0, dtype=np.float64).copy()

    history: dict[str, np.ndarray] = {
        "theta": np.zeros((n_iter_val + 1, 2), dtype=np.float64),
        "loss": np.zeros(n_iter_val + 1, dtype=np.float64),
        "gradient": np.zeros((n_iter_val + 1, 2), dtype=np.float64),
        "accuracy": np.zeros(n_iter_val + 1, dtype=np.float64),
        "boundary": np.zeros(n_iter_val + 1, dtype=np.float64),
    }

    for k in range(n_iter_val + 1):
        z = A_arr @ theta
        p = sigmoid(z)
        loss = cross_entropy_loss(y_arr, p)
        grad = cross_entropy_gradient(A_arr, y_arr, p)
        y_hat = (p >= 0.5).astype(int)
        acc = float(np.mean(y_hat == y_arr))
        boundary = -theta[0] / theta[1] if theta[1] != 0.0 else np.nan

        history["theta"][k] = theta
        history["loss"][k] = loss
        history["gradient"][k] = grad
        history["accuracy"][k] = acc
        history["boundary"][k] = boundary

        if k < n_iter_val:
            theta = theta - eta_val * grad

    return history


def assert_raises_message(func: Any, expected_fragment: str) -> None:
    """Assert that func() raises ValueError containing expected_fragment."""
    try:
        func()
    except ValueError as exc:
        assert expected_fragment in str(exc), f"Expected {expected_fragment!r} in {exc!r}"
    else:
        raise AssertionError(f"Expected ValueError containing {expected_fragment!r}, but no error was raised.")


# ===================================================================
# Test Functions
# ===================================================================

def test_primary_trace() -> None:
    """Test full trace across states 0 through 10 against Table T1."""
    history = train_logistic_gd(A, y, eta_val=0.1, n_iter_val=10)

    assert history["theta"].shape == (11, 2)
    assert history["loss"].shape == (11,)
    assert history["gradient"].shape == (11, 2)
    assert history["accuracy"].shape == (11,)
    assert history["boundary"].shape == (11,)

    np.testing.assert_allclose(history["theta"], EXPECTED_T1_THETA, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(history["loss"], EXPECTED_T1_LOSS, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(history["gradient"], EXPECTED_T1_GRADIENT, rtol=RTOL, atol=1e-5)
    np.testing.assert_array_equal(history["accuracy"], EXPECTED_T1_ACCURACY)

    # State 0 boundary is NaN; states 1 to 10 match T1
    assert np.isnan(history["boundary"][0])
    np.testing.assert_allclose(history["boundary"][1:], EXPECTED_T1_BOUNDARY[1:], rtol=RTOL, atol=1e-5)


def test_table_t2_state_0_audit() -> None:
    """Verify Table T2 per-example audit at State 0."""
    theta0 = np.array([0.0, 0.0], dtype=np.float64)
    z0 = A @ theta0
    p0 = sigmoid(z0)
    res0 = p0 - y
    res_x0 = res0 * x
    loss_i0 = -(y * np.log(p0) + (1.0 - y) * np.log(1.0 - p0))

    np.testing.assert_array_equal(z0, np.zeros(8, dtype=np.float64))
    np.testing.assert_allclose(p0, EXPECTED_T2_P, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(res0, EXPECTED_T2_RESIDUAL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(res_x0, EXPECTED_T2_RESIDUAL_X, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss_i0, EXPECTED_T2_LOSS_I, rtol=RTOL, atol=ATOL)

    assert np.sum(res0) == 1.0
    assert np.sum(res_x0) == -7.0
    np.testing.assert_allclose(np.mean(loss_i0), np.log(2.0), rtol=RTOL, atol=ATOL)
    assert float(np.mean((p0 >= 0.5) == y)) == 0.375


def test_table_t3_state_1_audit() -> None:
    """Verify Table T3 per-example audit at State 1."""
    theta1 = np.array([-0.012500, 0.087500], dtype=np.float64)
    z1 = A @ theta1
    p1 = sigmoid(z1)
    res1 = p1 - y
    res_x1 = res1 * x
    loss_i1 = -(y * np.log(p1) + (1.0 - y) * np.log(1.0 - p1))

    np.testing.assert_allclose(z1, EXPECTED_T3_Z, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(p1, EXPECTED_T3_P, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(res1, EXPECTED_T3_RESIDUAL, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(res_x1, EXPECTED_T3_RESIDUAL_X, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(loss_i1, EXPECTED_T3_LOSS_I, rtol=RTOL, atol=1e-5)

    np.testing.assert_allclose(np.mean(res1), 0.132734, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(np.mean(res_x1), -0.756989, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(np.mean(loss_i1), 0.620153, rtol=RTOL, atol=1e-5)
    assert float(np.mean((p1 >= 0.5) == y)) == 0.875
    np.testing.assert_allclose(-theta1[0] / theta1[1], 1.0 / 7.0, rtol=RTOL, atol=ATOL)


def test_table_t4_state_2_audit() -> None:
    """Verify Table T4 per-example audit at State 2."""
    theta2 = np.array([-0.025773, 0.163199], dtype=np.float64)
    z2 = A @ theta2
    p2 = sigmoid(z2)
    res2 = p2 - y
    res_x2 = res2 * x
    loss_i2 = -(y * np.log(p2) + (1.0 - y) * np.log(1.0 - p2))

    np.testing.assert_allclose(z2, EXPECTED_T4_Z, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(p2, EXPECTED_T4_P, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(res2, EXPECTED_T4_RESIDUAL, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(res_x2, EXPECTED_T4_RESIDUAL_X, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(loss_i2, EXPECTED_T4_LOSS_I, rtol=RTOL, atol=1e-5)

    np.testing.assert_allclose(np.mean(res2), 0.138485, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(np.mean(res_x2), -0.658488, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(np.mean(loss_i2), 0.564805, rtol=RTOL, atol=1e-5)
    assert float(np.mean((p2 >= 0.5) == y)) == 0.875
    np.testing.assert_allclose(-theta2[0] / theta2[1], 0.157927, rtol=RTOL, atol=1e-5)


def test_sigmoid_properties_and_derivative() -> None:
    """Verify sigmoid symmetry, monotonicity, range, and analytical derivative."""
    assert sigmoid(0.0) == 0.5
    grid_z = np.linspace(-6.0, 6.0, 121, dtype=np.float64)
    grid_p = sigmoid(grid_z)

    # Range and monotonicity
    assert np.all((grid_p > 0.0) & (grid_p < 1.0))
    assert np.all(np.diff(grid_p) > 0.0)

    # Symmetry: sigma(-z) + sigma(z) == 1
    np.testing.assert_allclose(sigmoid(-grid_z) + sigmoid(grid_z), 1.0, rtol=RTOL, atol=ATOL)

    # Central difference derivative check vs sigma*(1-sigma)
    h = 1e-6
    numerical_deriv = (sigmoid(grid_z + h) - sigmoid(grid_z - h)) / (2 * h)
    analytical_deriv = grid_p * (1.0 - grid_p)
    np.testing.assert_allclose(numerical_deriv, analytical_deriv, rtol=1e-5, atol=1e-5)


def test_loss_monotonicity_and_step_invariants() -> None:
    """Verify strict decrease of CE loss and invariant accuracy behavior."""
    history = train_logistic_gd(A, y, eta_val=0.1, n_iter_val=10)
    losses = history["loss"]

    # Strict monotonicity across 10 steps
    for k in range(10):
        assert losses[k + 1] < losses[k], f"Loss did not decrease at step {k}"

    # Accuracy jumps from 0.375 to 0.875, then stays flat through state 10
    assert history["accuracy"][0] == 0.375
    for k in range(1, 11):
        assert history["accuracy"][k] == 0.875

    # First update boundary invariant: x* = 1/7 regardless of eta
    for test_eta in [0.01, 0.05, 0.1, 0.2, 0.5]:
        h_eta = train_logistic_gd(A, y, eta_val=test_eta, n_iter_val=1)
        b1 = h_eta["theta"][1, 0]
        w1 = h_eta["theta"][1, 1]
        np.testing.assert_allclose(-b1 / w1, 1.0 / 7.0, rtol=RTOL, atol=ATOL)


def test_separable_milestones() -> None:
    """Verify k*=139 100% accuracy milestone and k=5000 visual display cap."""
    history_150 = train_logistic_gd(A, y, eta_val=0.1, n_iter_val=140)
    assert history_150["accuracy"][138] == 0.875
    assert history_150["accuracy"][139] == 1.0
    assert history_150["boundary"][138] < 1.0
    assert history_150["boundary"][139] > 1.0

    np.testing.assert_allclose(history_150["theta"][139], EXPECTED_K139_THETA, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(history_150["boundary"][139], EXPECTED_K139_BOUNDARY, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(history_150["loss"][139], EXPECTED_K139_LOSS, rtol=RTOL, atol=1e-5)

    history_5000 = train_logistic_gd(A, y, eta_val=0.1, n_iter_val=5000)
    np.testing.assert_allclose(history_5000["theta"][5000], EXPECTED_K5000_THETA, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(history_5000["boundary"][5000], EXPECTED_K5000_BOUNDARY, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(history_5000["loss"][5000], EXPECTED_K5000_LOSS, rtol=RTOL, atol=1e-5)
    assert history_5000["accuracy"][5000] == 1.0


def test_pure_python_cross_check() -> None:
    """Independent math.exp and math.log cross-check (Pathway A)."""
    pure_theta = [0.0, 0.0]
    x_list = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    y_list = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    for _ in range(10):
        grad_b = 0.0
        grad_w = 0.0
        for xi, yi in zip(x_list, y_list):
            zi = pure_theta[0] + pure_theta[1] * xi
            pi = 1.0 / (1.0 + math.exp(-zi))
            diff = pi - yi
            grad_b += diff
            grad_w += diff * xi
        pure_theta[0] -= 0.1 * (grad_b / 8.0)
        pure_theta[1] -= 0.1 * (grad_w / 8.0)

    history = train_logistic_gd(A, y, eta_val=0.1, n_iter_val=10)
    np.testing.assert_allclose(pure_theta, history["theta"][10], rtol=RTOL, atol=ATOL)


def test_exercises() -> None:
    """Verify numerical solutions to graded exercises E1, E2, E3, E4, E6."""
    # E1: -ln(p) penalty profile
    np.testing.assert_allclose(-math.log(0.999), 0.0010005, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(-math.log(0.500), math.log(2.0), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(-math.log(0.010), 4.605170, rtol=RTOL, atol=1e-5)
    np.testing.assert_allclose(-math.log(1e-6), 13.815511, rtol=RTOL, atol=1e-5)

    # E2: State 1 update with eta = 0.2
    b1_e2 = 0.0 - 0.2 * 0.125
    w1_e2 = 0.0 - 0.2 * (-0.875)
    np.testing.assert_allclose(b1_e2, -0.025, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w1_e2, 0.175, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(-b1_e2 / w1_e2, 1.0 / 7.0, rtol=RTOL, atol=ATOL)

    # E3: Boundaries stay in (0.0, 1.0)
    assert 0.0 < 1.0 / 7.0 < 1.0
    assert 0.0 < 0.157927 < 1.0

    # E4: Early stopping at 100% accuracy
    def train_until_accuracy(target_acc: float = 1.0) -> int:
        theta_curr = np.zeros(2, dtype=np.float64)
        for it in range(1, 1000):
            p_curr = sigmoid(A @ theta_curr)
            y_hat_curr = (p_curr >= 0.5).astype(int)
            if float(np.mean(y_hat_curr == y)) >= target_acc:
                return it - 1
            theta_curr -= 0.1 * cross_entropy_gradient(A, y, p_curr)
        return -1

    assert train_until_accuracy(1.0) == 139

    # E6: Overlapping non-separable data from 03.01
    y_overlap = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    x_overlap = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    A_overlap = np.column_stack([np.ones(8, dtype=np.float64), x_overlap])
    h_overlap = train_logistic_gd(A_overlap, y_overlap, eta_val=0.1, n_iter_val=5000)
    # Converges to finite solution theta ~ [-1.923440, 1.282293]
    np.testing.assert_allclose(h_overlap["theta"][5000], [-1.923440, 1.282293], rtol=RTOL, atol=1e-4)


def test_validation_errors() -> None:
    """Verify ValueError is raised for invalid inputs."""
    assert_raises_message(lambda: train_logistic_gd(A[:, :1], y, 0.1, 10), "shape (N, 2)")
    assert_raises_message(lambda: train_logistic_gd(A, y[:-1], 0.1, 10), "matching A")
    assert_raises_message(lambda: train_logistic_gd(A, np.full(8, 2.0), 0.1, 10), "binary labels")
    assert_raises_message(lambda: train_logistic_gd(A, y, -0.1, 10), "positive and finite")
    assert_raises_message(lambda: train_logistic_gd(A, y, 0.0, 10), "positive and finite")
    assert_raises_message(lambda: train_logistic_gd(A, y, 0.1, 0), "at least 1")

    A_nan = A.copy()
    A_nan[0, 0] = np.nan
    assert_raises_message(lambda: train_logistic_gd(A_nan, y, 0.1, 10), "finite")


def main() -> None:
    test_primary_trace()
    test_table_t2_state_0_audit()
    test_table_t3_state_1_audit()
    test_table_t4_state_2_audit()
    test_sigmoid_properties_and_derivative()
    test_loss_monotonicity_and_step_invariants()
    test_separable_milestones()
    test_pure_python_cross_check()
    test_exercises()
    test_validation_errors()
    print("PASS: 03.02 cross-entropy and gradient descent hand calculations verified.")


if __name__ == "__main__":
    main()
