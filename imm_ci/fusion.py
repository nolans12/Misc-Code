from __future__ import annotations

import numpy as np
from scipy import optimize
from numpy.linalg import inv


def det_of_fused_covariance(
    omega: np.ndarray, cov1: np.ndarray, cov2: np.ndarray
) -> float:
    """
    Determinant of the fused covariance matrix for a given CI weight omega.

    This matches the pattern:
        P = inv(omega * inv(cov1) + (1 - omega) * inv(cov2))
        return det(P)
    """
    # Ensure omega is a scalar in [0, 1]
    omega = float(np.clip(np.atleast_1d(omega)[0], 0.0, 1.0))

    cov1_inv = np.linalg.inv(cov1)
    cov2_inv = np.linalg.inv(cov2)

    P_inv = omega * cov1_inv + (1.0 - omega) * cov2_inv
    P = np.linalg.inv(P_inv)

    return float(np.linalg.det(P))

def covariance_intersection_single(
    x1: np.ndarray,
    P1: np.ndarray,
    x2: np.ndarray,
    P2: np.ndarray,
    block_ci: bool = False,
) -> tuple[np.ndarray, np.ndarray, float | np.ndarray]:
    """
    Perform covariance intersection (CI) fusion.

    If block_ci=True:
        - Perform CI independently on position, velocity, acceleration
        - Enforce block-diagonal covariance
        - Cross-correlations are zeroed
    """

    x1 = np.atleast_1d(x1).reshape(-1)
    x2 = np.atleast_1d(x2).reshape(-1)
    P1 = np.atleast_2d(P1)
    P2 = np.atleast_2d(P2)

    assert x1.shape == x2.shape
    assert P1.shape == P2.shape
    n = x1.shape[0]

    if not block_ci:
        # ---------- Full CI (original) ----------
        res = optimize.minimize(
            det_of_fused_covariance,
            x0=np.array([0.5]),
            args=(P1, P2),
            bounds=[(0.0, 1.0)],
        )
        omega = float(np.clip(res.x[0], 0.0, 1.0))

        P_fused = inv(omega * inv(P1) + (1.0 - omega) * inv(P2))
        x_fused = P_fused @ (
            omega * (inv(P1) @ x1) + (1.0 - omega) * (inv(P2) @ x2)
        )

        return x_fused, P_fused, omega

    # ---------- Block CI ----------
    assert n == 9, "block_ci assumes [pos, vel, accel] with 3D each"

    blocks = [
        slice(0, 3),   # position
        slice(3, 6),   # velocity
        slice(6, 9),   # acceleration
    ]

    x_fused = np.zeros_like(x1)
    P_fused = np.zeros_like(P1)
    omegas = np.zeros(3)

    for i, sl in enumerate(blocks):
        x1_b = x1[sl]
        x2_b = x2[sl]
        P1_b = P1[sl, sl]
        P2_b = P2[sl, sl]

        res = optimize.minimize(
            det_of_fused_covariance,
            x0=np.array([0.5]),
            args=(P1_b, P2_b),
            bounds=[(0.0, 1.0)],
        )
        omega_b = float(np.clip(res.x[0], 0.0, 1.0))
        omegas[i] = omega_b

        P_b = inv(omega_b * inv(P1_b) + (1.0 - omega_b) * inv(P2_b))
        x_b = P_b @ (
            omega_b * (inv(P1_b) @ x1_b)
            + (1.0 - omega_b) * (inv(P2_b) @ x2_b)
        )

        x_fused[sl] = x_b
        P_fused[sl, sl] = P_b

    return x_fused, P_fused, omegas



def fuse_ci_series(
    X1: np.ndarray,
    P1: np.ndarray,
    X2: np.ndarray,
    P2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply covariance intersection fusion over a time series of estimates.

    Parameters
    ----------
    X1, X2 : (T, n) ndarray
        Sequences of state means.
    P1, P2 : (T, n, n) ndarray
        Sequences of covariance matrices.

    Returns
    -------
    X_fused : (T, n) ndarray
        Fused state means.
    P_fused : (T, n, n) ndarray
        Fused covariances.
    omegas : (T,) ndarray
        CI weights applied to (X1, P1) at each time.
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)

    if X1.shape != X2.shape:
        raise ValueError(f"X1 and X2 must have the same shape; got {X1.shape} vs {X2.shape}.")
    if P1.shape != P2.shape:
        raise ValueError(f"P1 and P2 must have the same shape; got {P1.shape} vs {P2.shape}.")

    T, _ = X1.shape

    X_fused = np.zeros_like(X1)
    P_fused = np.zeros_like(P1)
    omegas = np.zeros(T, dtype=float)

    for k in range(T):
        # If we don't have a valid second covariance here, fall back to first estimate.
        if not np.any(P2[k]):
            X_fused[k] = X1[k]
            P_fused[k] = P1[k]
            omegas[k] = 1.0
            continue

        x_fused_k, P_fused_k, w_k = covariance_intersection_single(
            X1[k], P1[k], X2[k], P2[k], block_ci=False
        )
        X_fused[k] = x_fused_k
        P_fused[k] = P_fused_k
        omegas[k] = w_k #[0] # just use pos

    return X_fused, P_fused, omegas
