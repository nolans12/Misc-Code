"""Simulate GEO az/el measurements + per-satellite CA Kalman tracking."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse DCT's target/sensor/plotting implementations.
THIS_DIR = Path(__file__).resolve().parent
DCT_DIR = THIS_DIR.parent / "DCT"
if str(DCT_DIR) not in sys.path:
    sys.path.insert(0, str(DCT_DIR))

from plot_sim import LivePlotter  # noqa: E402
from sensing import create_geo_satellites, measurement_model  # noqa: E402
from target import BallisticTarget  # noqa: E402


SEED = 2
NOISE_SIGMA = 5e-4  # [rad]
SIM_DURATION_S = 10 * 60
DT_S = 10.0
NUM_MISSILES = 2

# CA KF tuning inspired by RTS/rts_monotracker.py (without RTS smoother).
EST_CA_SIGMA = 5e-4  # [rad/s^2]
INIT_ANGLE_SIGMA = 4e-3  # [rad]
INIT_RATE_SIGMA = 2e-3  # [rad/s]
INIT_ACCEL_SIGMA = 1e-3  # [rad/s^2]

MISSILE_CONFIGS = [
    [-10.0, -40.0, 0.0, 20.0, 0.0, 0.0, 250.0],
    [-6.0, -50.5, 0.0, 15.0, 15.0, 0.0, 250.0],
    [-15.0, -45.0, 0.0, 25.0, -20.0, 0.0, 250.0],
][:NUM_MISSILES]

DATA_DIR = THIS_DIR / "data"


def _build_zoom_extent(configs: list[list[float]], margin_deg: float = 15.0) -> list[float]:
    lats = [v for cfg in configs for v in (cfg[0], cfg[3])]
    lons = [v for cfg in configs for v in (cfg[1], cfg[4])]
    return [min(lats) - margin_deg, min(lons) - margin_deg, max(lats) + margin_deg, max(lons) + margin_deg]


class ConstantAccelerationKalman2D:
    """Linear CA Kalman filter in az/el space.

    State ordering:
    [az, el, az_rate, el_rate, az_acc, el_acc]
    """

    def __init__(self, dt: float, ca_sigma: float, meas_sigma: float):
        self.dt = dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        self.F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        self.R = np.diag([meas_sigma * meas_sigma, meas_sigma * meas_sigma])

        q = ca_sigma * ca_sigma
        self.Q = np.zeros((6, 6))
        for p, v, a in [(0, 2, 4), (1, 3, 5)]:
            self.Q[p, p] = dt4 / 4 * q
            self.Q[p, v] = dt3 / 2 * q
            self.Q[v, p] = dt3 / 2 * q
            self.Q[p, a] = dt2 / 2 * q
            self.Q[a, p] = dt2 / 2 * q
            self.Q[v, v] = dt2 * q
            self.Q[v, a] = dt * q
            self.Q[a, v] = dt * q
            self.Q[a, a] = q

        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.initialized = False
        self._last_meas: np.ndarray | None = None

    def initialize(self, az: float, el: float) -> None:
        self.x = np.array([az, el, 0.0, 0.0, 0.0, 0.0])
        self.P = np.diag(
            [
                INIT_ANGLE_SIGMA**2,
                INIT_ANGLE_SIGMA**2,
                INIT_RATE_SIGMA**2,
                INIT_RATE_SIGMA**2,
                INIT_ACCEL_SIGMA**2,
                INIT_ACCEL_SIGMA**2,
            ]
        )
        self._last_meas = np.array([az, el])
        self.initialized = True

    def step(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.initialized:
            self.initialize(z[0], z[1])
            return self.x.copy(), self.P.copy(), np.zeros(2)

        if self._last_meas is not None and self.dt > 0.0:
            self.x[2] = (z[0] - self._last_meas[0]) / self.dt
            self.x[3] = (z[1] - self._last_meas[1]) / self.dt
        self._last_meas = z.copy()

        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        innov = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ innov
        self.P = (np.eye(6) - K @ self.H) @ P_pred
        return self.x.copy(), self.P.copy(), innov


@dataclass(slots=True)
class SimulationOutputs:
    npz_path: Path
    meas_csv_path: Path
    kf_csv_path: Path
    kf_perf_plot_path: Path


def _create_targets() -> list[BallisticTarget]:
    targets: list[BallisticTarget] = []
    for cfg in MISSILE_CONFIGS:
        targets.append(
            BallisticTarget(
                launch_lat_deg=cfg[0],
                launch_lon_deg=cfg[1],
                launch_alt_km=cfg[2],
                impact_lat_deg=cfg[3],
                impact_lon_deg=cfg[4],
                impact_alt_km=cfg[5],
                flight_duration_s=SIM_DURATION_S,
                max_altitude_km=cfg[6],
            )
        )
    return targets


def _create_satellites(rng: np.random.Generator):
    seed_locs = [
        [MISSILE_CONFIGS[0][0], MISSILE_CONFIGS[0][1]],
        [MISSILE_CONFIGS[0][3], MISSILE_CONFIGS[0][4]],
        [
            0.5 * (MISSILE_CONFIGS[0][0] + MISSILE_CONFIGS[0][3]),
            0.5 * (MISSILE_CONFIGS[0][1] + MISSILE_CONFIGS[0][4]),
        ],
    ]
    jittered = [[lat + rng.uniform(-10, 10), lon + rng.uniform(-10, 10)] for lat, lon in seed_locs]
    return create_geo_satellites(positions=jittered, noise_sigma=NOISE_SIGMA, rng=rng)


def _plot_kf_performance(df_kf: pd.DataFrame) -> Path:
    sat_names = sorted(df_kf["satellite"].unique())
    target_ids = sorted(df_kf["target_id"].unique())

    nrows = len(sat_names)
    ncols = max(1, len(target_ids))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    fig.suptitle("Per-Satellite CA KF Tracking in Az/El", fontsize=14, weight="bold")

    for r, sat in enumerate(sat_names):
        for c, target_id in enumerate(target_ids):
            ax = axes[r][c]
            sel = df_kf[(df_kf["satellite"] == sat) & (df_kf["target_id"] == target_id)]
            if sel.empty:
                ax.set_axis_off()
                continue

            t = sel["time_s"].to_numpy()
            az_true = sel["true_az_rad"].to_numpy()
            az_est = sel["kf_az_rad"].to_numpy()
            el_true = sel["true_el_rad"].to_numpy()
            el_est = sel["kf_el_rad"].to_numpy()

            ax.plot(t, (az_true - az_est) * 1e3, "-", lw=1.6, label="Az err (mrad)")
            ax.plot(t, (el_true - el_est) * 1e3, "-", lw=1.6, label="El err (mrad)")
            ax.plot(t, (az_true - sel["meas_az_rad"].to_numpy()) * 1e3, "--", lw=1.0, alpha=0.7, label="Az meas err")
            ax.plot(t, (el_true - sel["meas_el_rad"].to_numpy()) * 1e3, "--", lw=1.0, alpha=0.7, label="El meas err")

            az_rms = np.sqrt(np.mean((az_true - az_est) ** 2)) * 1e3
            el_rms = np.sqrt(np.mean((el_true - el_est) ** 2)) * 1e3
            ax.set_title(f"{sat} / Target {target_id} | RMS: az={az_rms:.2f} mrad, el={el_rms:.2f} mrad")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Error [mrad]")
            ax.grid(alpha=0.35)
            ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    out = DATA_DIR / "kf_tracking_errors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    return out


def run_simulation(seed: int = SEED, live_plot: bool = True) -> SimulationOutputs:
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    targets = _create_targets()
    satellites = _create_satellites(rng)
    zoom = _build_zoom_extent(MISSILE_CONFIGS)

    all_truth: list[list[np.ndarray]] = [[] for _ in range(NUM_MISSILES)]
    all_measurements = [[] for _ in range(NUM_MISSILES)]

    # One CA filter for each (satellite, target) pairing.
    kf_bank: dict[tuple[int, int], ConstantAccelerationKalman2D] = {}
    for sat_idx in range(len(satellites)):
        for target_idx in range(NUM_MISSILES):
            kf_bank[(sat_idx, target_idx)] = ConstantAccelerationKalman2D(
                dt=DT_S, ca_sigma=EST_CA_SIGMA, meas_sigma=NOISE_SIGMA
            )

    plotter = LivePlotter(zoom_extent=zoom, num_missiles=NUM_MISSILES) if live_plot else None
    times = np.arange(0.0, SIM_DURATION_S + DT_S, DT_S)

    meas_rows: list[dict[str, float | int | str]] = []
    kf_rows: list[dict[str, float | int | str]] = []

    for idx, t in enumerate(times):
        for target_idx, target in enumerate(targets):
            pos_ecef, vel_ecef = target.state_at(t)
            truth_state = np.concatenate([pos_ecef, vel_ecef])
            all_truth[target_idx].append(truth_state)

            measurements = [sat.measure(t, pos_ecef, rng=rng) for sat in satellites]
            all_measurements[target_idx].extend(measurements)

            for sat_idx, meas in enumerate(measurements):
                true_az, true_el = measurement_model(pos_ecef, meas.sat.position)
                z = np.array([meas.az, meas.el])
                x_kf, P_kf, innov = kf_bank[(sat_idx, target_idx)].step(z)

                meas_rows.append(
                    {
                        "time_s": t,
                        "satellite": meas.sat.name,
                        "target_id": target_idx + 1,
                        "meas_az_rad": meas.az,
                        "meas_el_rad": meas.el,
                        "true_az_rad": true_az,
                        "true_el_rad": true_el,
                        "meas_sigma_rad": np.sqrt(meas.R[0, 0]),
                        "sat_x_km": meas.sat.position[0],
                        "sat_y_km": meas.sat.position[1],
                        "sat_z_km": meas.sat.position[2],
                    }
                )

                kf_rows.append(
                    {
                        "time_s": t,
                        "satellite": meas.sat.name,
                        "target_id": target_idx + 1,
                        "true_az_rad": true_az,
                        "true_el_rad": true_el,
                        "meas_az_rad": meas.az,
                        "meas_el_rad": meas.el,
                        "kf_az_rad": x_kf[0],
                        "kf_el_rad": x_kf[1],
                        "kf_az_rate_rad_s": x_kf[2],
                        "kf_el_rate_rad_s": x_kf[3],
                        "kf_az_acc_rad_s2": x_kf[4],
                        "kf_el_acc_rad_s2": x_kf[5],
                        "innov_az_rad": innov[0],
                        "innov_el_rad": innov[1],
                        "kf_var_az": P_kf[0, 0],
                        "kf_var_el": P_kf[1, 1],
                    }
                )

        if plotter:
            plotter.update(
                time=t,
                times_history=times[: idx + 1].tolist(),
                all_truth=all_truth,
                all_measurements=all_measurements,
                satellites=satellites,
            )

    if plotter:
        plotter.close()

    df_meas = pd.DataFrame(meas_rows)
    df_kf = pd.DataFrame(kf_rows)

    meas_csv = DATA_DIR / "measurements.csv"
    kf_csv = DATA_DIR / "kf_updates.csv"
    df_meas.to_csv(meas_csv, index=False)
    df_kf.to_csv(kf_csv, index=False)

    npz_path = DATA_DIR / "simulation_run.npz"
    save_dict: dict[str, np.ndarray | int] = {"times": times, "num_missiles": NUM_MISSILES}
    for i in range(NUM_MISSILES):
        save_dict[f"truth_{i}"] = np.stack(all_truth[i])
        meas_i = all_measurements[i]
        save_dict[f"meas_times_{i}"] = np.array([m.time for m in meas_i])
        save_dict[f"meas_sat_positions_{i}"] = np.array([m.sat.position for m in meas_i])
        save_dict[f"meas_azs_{i}"] = np.array([m.az for m in meas_i])
        save_dict[f"meas_els_{i}"] = np.array([m.el for m in meas_i])
    np.savez(npz_path, **save_dict)

    kf_plot = _plot_kf_performance(df_kf)
    return SimulationOutputs(npz_path=npz_path, meas_csv_path=meas_csv, kf_csv_path=kf_csv, kf_perf_plot_path=kf_plot)


if __name__ == "__main__":
    outputs = run_simulation()
    print(f"Saved trajectory data to: {outputs.npz_path}")
    print(f"Saved measurements dataframe to: {outputs.meas_csv_path}")
    print(f"Saved KF update dataframe to: {outputs.kf_csv_path}")
    print(f"Saved KF performance plot to: {outputs.kf_perf_plot_path}")
