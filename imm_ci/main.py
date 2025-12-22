# main.py
from __future__ import annotations

import numpy as np
import copy

from imm import IMM
from target import Target3D
from ca import CAFilter
from cv import CVFilter
from measurement import Measurement


# ======================= CONFIG =======================

TOTAL_TIME = 20.0
BOOST_END_TIME = 10.0
DT = 0.1

CA_SIGMA = 10.0
CV_SIGMA = 10.0
MIXING_SIGMA = CA_SIGMA

EST_MEAS_SIGMA = 15.0
TRUE_MEAS_SIGMA = 3.0

TARG_GS = 5.0
TARG_SIGMA = 0.1

# =====================================================


def main():
    steps = int(TOTAL_TIME / DT)

    # ---------------- Truth target ----------------
    target = Target3D(
        dt=DT,
        boost_end_time=BOOST_END_TIME,
        g=9.80665,
        accel_boost_g=TARG_GS,
        process_sigma_acc=TARG_SIGMA,
        seed=2,
    )
    target.set_initial(
        pos_xyz=np.array([0.0, 0.0, 0.0]),
        vel_xyz=np.array([150.0, 30.0, 0.0]),
    )

    # ---------------- IMM parameters ----------------
    PI = np.array([[0.99, 0.01],
                   [0.01, 0.99]])

    mu0 = np.array([0.9, 0.1])
    R_est = EST_MEAS_SIGMA**2 * np.eye(3)
    R_true = TRUE_MEAS_SIGMA**2 * np.eye(3)

    # First measurement at t=0 for initialization only
    z0 = target.measure_position(R_true)

    # ---------------- Initial states ----------------
    x0_ca = np.zeros((9, 1))
    x0_ca[0:3, 0] = z0
    x0_ca[3:6, 0] = [140.0, 20.0, 10.0]
    P0_ca = np.diag([EST_MEAS_SIGMA**2]*3 + [100.0**2]*3 + [50.0**2]*3)

    x0_cv = np.zeros((6, 1))
    x0_cv[0:3, 0] = z0
    x0_cv[3:6, 0] = [140.0, 20.0, 10.0]
    P0_cv = np.diag([EST_MEAS_SIGMA**2]*3 + [100.0**2]*3)

    # ---------------- Forward IMM ----------------
    ca_fwd = CAFilter(DT, CA_SIGMA, R_est, x0_ca, P0_ca)
    cv_fwd = CVFilter(DT, CV_SIGMA, R_est, x0_cv, P0_cv, embed_sigma=MIXING_SIGMA)

    imm_fwd = IMM(
        models=[ca_fwd, cv_fwd],
        PI=PI,
        mu0=mu0,
        dt=DT,
        time_weighted_likelihood=False,
    )

    truth_hist = np.zeros((steps, 9))
    meas_hist = np.zeros((steps, 3))
    x_fwd = np.zeros((steps, 9))
    P_fwd = np.zeros((steps, 9, 9))
    mu_fwd = np.zeros((steps, 2))
    measurements: list[Measurement] = []

    for k in range(steps):
        truth = target.step()
        z_true = target.measure_position(R_true)

        # Measurement time is the current target time after stepping.
        meas = Measurement(t=target.t, z=z_true, R=R_est)
        measurements.append(meas)

        out = imm_fwd.step(meas)

        truth_hist[k] = truth[:, 0]
        meas_hist[k] = z_true
        x_fwd[k] = out["x_common"][:, 0]
        P_fwd[k] = out["P_common"][:, :]
        mu_fwd[k] = out["mu"]

    # ---------------- Backward IMM ----------------
    ca_back = copy.deepcopy(ca_fwd)
    cv_back = copy.deepcopy(cv_fwd)

    # Fresh backward IMM (already instantiated)
    # Initialize at the last measurement time so that the first backward step
    # uses a negative dt.
    imm_back = IMM(
        models=[ca_back, cv_back],
        PI=PI.T,
        mu0=np.ones(2) / 2,
        dt=DT,
        time_weighted_likelihood=False,
        t0=measurements[-1].t,
    )

    # RESET IMM STATE EXPLICITLY
    imm_back.set_state(
        est=x_fwd[-1],  # or a neutral prior (see note below)
        cov=P_fwd[-1],
        mu=mu_fwd[-1],
    )

    x_back = np.zeros_like(x_fwd)
    mu_back = np.zeros_like(mu_fwd)

    # Run backward over measurements z[K-1] ... z[1] so dt < 0
    for k, meas in enumerate(measurements[:0:-1]):
        # skip first meas
        if k == 0:
            continue
        out = imm_back.step(meas)
        x_back[k] = out["x_common"][:, 0]
        mu_back[k] = out["mu"]

    # Reverse backward results to forward time
    x_back = x_back[::-1]
    mu_back = mu_back[::-1]

    # ---------------- Plots ----------------
    try:
        import matplotlib.pyplot as plt

        # Time axis from actual measurement timestamps
        t = np.array([m.t for m in measurements])

        # ---- Altitude ----
        plt.figure(figsize=(9, 4))
        plt.plot(t, truth_hist[:, 2], label="Truth")
        plt.plot(t, x_fwd[:, 2], label="Forward IMM")
        plt.plot(t, x_back[:, 2], "--", label="Backward IMM")
        plt.axvline(BOOST_END_TIME, linestyle="--", color="k")
        plt.xlabel("Time (s)")
        plt.ylabel("Altitude (m)")
        plt.legend()
        plt.grid()

        # ---- Velocity magnitude ----
        plt.figure(figsize=(9, 4))
        plt.plot(t, np.linalg.norm(truth_hist[:, 3:6], axis=1), label="Truth")
        plt.plot(t, np.linalg.norm(x_fwd[:, 3:6], axis=1), label="Forward IMM")
        plt.plot(t, np.linalg.norm(x_back[:, 3:6], axis=1), "--", label="Backward IMM")
        plt.axvline(BOOST_END_TIME, linestyle="--", color="k")
        plt.xlabel("Time (s)")
        plt.ylabel("|v| (m/s)")
        plt.legend()
        plt.grid()

        # ---- Mode probabilities ----
        plt.figure(figsize=(9, 4))
        plt.plot(t, mu_fwd[:, 0], label="P(CA) forward")
        plt.plot(t, mu_fwd[:, 1], label="P(CV) forward")
        plt.plot(t, mu_back[:, 0], "--", label="P(CA) backward")
        plt.plot(t, mu_back[:, 1], "--", label="P(CV) backward")
        plt.axvline(BOOST_END_TIME, linestyle="--", color="k")
        plt.xlabel("Time (s)")
        plt.ylabel("Mode probability")
        plt.legend()
        plt.grid()

        plt.show()

    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()
