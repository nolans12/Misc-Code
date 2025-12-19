# target.py
from __future__ import annotations

import numpy as np


class Target3D:
    """
    Simple 3D target simulator:
      - Boost phase: net upward acceleration +5g in Z (user request).
      - Ballistic phase: acceleration -g in Z.

    State is "truth" CA-like:
      [x y z vx vy vz ax ay az]^T
    """

    def __init__(
        self,
        dt: float,
        boost_end_time: float,
        g: float = 9.80665,
        accel_boost_g: float = 5.0,
        process_sigma_acc: float = 0.0,
        seed: int | None = None,
    ):
        self.dt = float(dt)
        self.boost_end_time = float(boost_end_time)
        self.g = float(g)
        self.accel_boost = float(accel_boost_g) * self.g  # +Z acceleration during boost (net)
        self.process_sigma_acc = float(process_sigma_acc)

        self.rng = np.random.default_rng(seed)
        self.t = 0.0

        # truth state (9x1)
        self.x = np.zeros((9, 1), dtype=float)

    def set_initial(
        self,
        pos_xyz: np.ndarray,
        vel_xyz: np.ndarray,
    ) -> None:
        pos_xyz = np.asarray(pos_xyz, dtype=float).reshape(3)
        vel_xyz = np.asarray(vel_xyz, dtype=float).reshape(3)

        self.x[:] = 0.0
        self.x[0:3, 0] = pos_xyz
        self.x[3:6, 0] = vel_xyz

    def _truth_accel(self) -> np.ndarray:
        """
        Returns acceleration vector [ax, ay, az] for current time self.t.
        """
        ax, ay = 0.0, 0.0
        if self.t < self.boost_end_time:
            az = self.accel_boost  # net upward +5g
        else:
            az = -self.g  # ballistic
        return np.array([ax, ay, az], dtype=float)

    def step(self) -> np.ndarray:
        """
        Advance truth by one time step using constant acceleration over dt.
        Adds optional accel noise (process_sigma_acc) to mimic modeling error.
        Returns new truth state (9x1).
        """
        dt = self.dt

        a = self._truth_accel()
        if self.process_sigma_acc > 0:
            a = a + self.rng.normal(0.0, self.process_sigma_acc, size=3)

        # Write accel into truth state (CA truth)
        self.x[6:9, 0] = a

        # Kinematics
        p = self.x[0:3, 0]
        v = self.x[3:6, 0]

        p_new = p + v * dt + 0.5 * a * dt * dt
        v_new = v + a * dt

        self.x[0:3, 0] = p_new
        self.x[3:6, 0] = v_new

        self.t += dt
        return self.x.copy()

    def measure_position(self, R: np.ndarray) -> np.ndarray:
        """
        Position measurement z = [x y z] + noise, noise ~ N(0, R)
        """
        R = np.asarray(R, dtype=float)
        z_true = self.x[0:3, 0]

        # Sample measurement noise with covariance R
        try:
            L = np.linalg.cholesky(R)
            noise = L @ self.rng.normal(size=3)
        except np.linalg.LinAlgError:
            noise = self.rng.normal(size=3) * np.sqrt(np.maximum(np.diag(R), 0.0))

        return (z_true + noise).reshape(3)
