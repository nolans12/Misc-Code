# target.py
from __future__ import annotations

import numpy as np


class Motion:
    """
    Accel motion
    Args:
        start_t: Start time of the motion.
        end_t: End time of the motion.
        accel: Acceleration of the motion.
        noise_sigma: Noise sigma of the motion.
    """
    def __init__(self, start_t, end_t, accel, noise_sigma):
        self.start_t = start_t
        self.end_t = end_t
        self.accel = accel * np.array([1, -1, 1], dtype=float)
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(1)

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
        motions: list[Motion],
    ):
        self.t = 0.0
        self.motions = motions
        self.x = np.zeros((9, 1), dtype=float)
        self.rng = np.random.default_rng(1)

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
        self.x[6:9, 0] = self._truth_accel()

    def _truth_accel(self) -> np.ndarray:
        """
        Returns acceleration vector [ax, ay, az] for current time self.t.
        If current time is within the start and end time of any motion, use that acceleration; else, [0, 0, 0].
        """
        for motion in self.motions:
            if motion.start_t <= self.t < motion.end_t:
                return np.array(motion.accel, dtype=float) + motion.noise_sigma * self.rng.normal(size=3)
        return np.zeros(3, dtype=float)

    def step(self, dt: float) -> np.ndarray:
        """
        Advance truth by one time step using constant acceleration over dt.
        Adds optional accel noise (process_sigma_acc) to mimic modeling error.
        Returns new truth state (9x1).
        """

        # dt time step
        self.t += dt

        # Kinematics
        p = self.x[0:3, 0]
        v = self.x[3:6, 0]
        a = self._truth_accel()

        p_new = p + v * dt + 0.5 * a * dt * dt
        v_new = v + a * dt

        self.x[0:3, 0] = p_new
        self.x[3:6, 0] = v_new
        self.x[6:9, 0] = a 

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
