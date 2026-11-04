"""Verify the frozen Lesson 01.02 loss-function calculations."""

import numpy as np


RTOL = 0.0
ATOL = 1e-12

RESIDUALS_A = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
RESIDUALS_B = np.array([0.5, 1.0, 1.5, 5.0], dtype=np.float64)
RESIDUALS_C = np.array([-0.5, 1.5, -2.0, 1.0], dtype=np.float64)


def validate_residuals(residuals: np.ndarray) -> None:
    if not isinstance(residuals, np.ndarray):
        raise TypeError("residuals must be a numpy.ndarray")
    if residuals.dtype != np.float64:
        raise ValueError("residuals must have dtype float64")
    if residuals.ndim != 1:
        raise ValueError("residuals must be a 1-D array")
    if residuals.size == 0:
        raise ValueError("residuals must be non-empty")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must contain only finite values")


def pointwise_absolute_loss(residuals: np.ndarray) -> np.ndarray:
    validate_residuals(residuals)
    absolute_losses = np.abs(residuals)
    return absolute_losses


def pointwise_squared_loss(residuals: np.ndarray) -> np.ndarray:
    validate_residuals(residuals)
    squared_losses = residuals ** 2
    return squared_losses


def mean_absolute_error(residuals: np.ndarray) -> np.float64:
    absolute_losses = pointwise_absolute_loss(residuals)
    mae = np.mean(absolute_losses)
    return mae


def mean_squared_error(residuals: np.ndarray) -> np.float64:
    squared_losses = pointwise_squared_loss(residuals)
    mse = np.mean(squared_losses)
    return mse


def expect_error(function, value, error_type, message):
    try:
        function(value)
    except error_type as error:
        assert str(error) == message
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message}")


def main() -> None:
    expected = {
        "A": ([0.5, 1.0, 1.5, 2.0], [0.25, 1.0, 2.25, 4.0], 1.25, 1.875),
        "B": ([0.5, 1.0, 1.5, 5.0], [0.25, 1.0, 2.25, 25.0], 2.0, 7.125),
        "C": ([0.5, 1.5, 2.0, 1.0], [0.25, 2.25, 4.0, 1.0], 1.25, 1.875),
    }
    vectors = {"A": RESIDUALS_A, "B": RESIDUALS_B, "C": RESIDUALS_C}
    for name, residuals in vectors.items():
        assert residuals.shape == (4,)
        assert residuals.dtype == np.float64
        absolute_expected, squared_expected, mae_expected, mse_expected = expected[name]
        absolute_losses = pointwise_absolute_loss(residuals)
        squared_losses = pointwise_squared_loss(residuals)
        np.testing.assert_allclose(absolute_losses, absolute_expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(squared_losses, squared_expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(mean_absolute_error(residuals), mae_expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(mean_squared_error(residuals), mse_expected, rtol=RTOL, atol=ATOL)
        assert isinstance(mean_absolute_error(residuals), np.float64)
        assert isinstance(mean_squared_error(residuals), np.float64)
        assert mean_absolute_error(residuals).ndim == 0
        assert mean_squared_error(residuals).ndim == 0
        assert np.all(absolute_losses >= 0.0)
        assert np.all(squared_losses >= 0.0)
        np.testing.assert_allclose(
            mean_absolute_error(residuals), np.mean(np.abs(residuals)), rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            mean_squared_error(residuals), np.mean(residuals ** 2), rtol=RTOL, atol=ATOL
        )
        assert np.all(np.isfinite(absolute_losses))
        assert np.all(np.isfinite(squared_losses))

    assert np.isclose(np.mean(RESIDUALS_C), 0.0, rtol=RTOL, atol=ATOL)
    assert np.isclose(mean_absolute_error(RESIDUALS_C), 1.25, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(mean_absolute_error(RESIDUALS_A), mean_absolute_error(RESIDUALS_C), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(mean_squared_error(RESIDUALS_A), mean_squared_error(RESIDUALS_C), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose((5.0 - 2.0) / 4, 0.75, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose((25.0 - 4.0) / 4, 5.25, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(1.25**2, 1.5625, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(1.875 - 1.25**2, 0.3125, rtol=RTOL, atol=ATOL)

    expect_error(mean_absolute_error, np.array([], dtype=np.float64), ValueError, "residuals must be non-empty")
    expect_error(mean_absolute_error, np.array([np.nan, 1.0], dtype=np.float64), ValueError, "residuals must contain only finite values")
    expect_error(mean_absolute_error, np.array([np.inf, 1.0], dtype=np.float64), ValueError, "residuals must contain only finite values")
    expect_error(mean_absolute_error, [0.5, 1.0], TypeError, "residuals must be a numpy.ndarray")
    expect_error(mean_absolute_error, np.array([0.5, 1.0], dtype=np.float32), ValueError, "residuals must have dtype float64")
    expect_error(mean_absolute_error, np.array([[0.5, 1.0]], dtype=np.float64), ValueError, "residuals must be a 1-D array")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
