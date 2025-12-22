# main.py
from __future__ import annotations

import numpy as np

from imm import IMM
from target import Target3D


TOTAL_TIME = 20.0
BOOST_END_TIME = 10.0
DT = 0.01

CA_SIGMA = 5.0 # m/s^2
CV_SIGMA = 5.0 # m/s
MIXING_SIGMA = CA_SIGMA 

EST_MEAS_SIGMA = 15.0 # m, R
TRUE_MEAS_SIGMA = 5.0 # m, R

TIME_WEIGHTED_LIKELIHOOD = False
LIKELIHOOD_TAU = 10.0

### True target
targ_gs = 5.0 # gs
targ_sigma = 0.1 # m/s^2 noise

def main():
    # ---------------- Simulation settings ----------------
    steps = int(TOTAL_TIME / DT)

    # Target truth
    target = Target3D(
        dt=DT,
        boost_end_time=BOOST_END_TIME,
        g=9.80665,
        accel_boost_g=targ_gs,        # +5g net upward during boost
        process_sigma_acc=targ_sigma,    # optional truth accel noise (m/s^2)
        seed=2,
    )
    target.set_initial(
        pos_xyz=np.array([0.0, 0.0, 0.0]),
        vel_xyz=np.array([150.0, 30.0, 0.0]),
    )

    # ---------------- IMM settings ----------------
    # Model switch probabilities:
    #   - prefer CA early, CV later, but IMM will infer from likelihoods.
    # PI[i,j]=P(j | i) with model order [CA, CV]
    PI = np.array([
        [0.95, 0.05],
        [0.05, 0.95],
    ], dtype=float)

    mu0 = np.array([0.9, 0.1], dtype=float)  # start believing boost is more likely

    # Initial filter states
    # We'll initialize from the first measurement with rough guesses.
    z0 = target.measure_position(TRUE_MEAS_SIGMA**2 * np.eye(3))

    # CA init (9)
    x0_ca = np.zeros((9, 1))
    x0_ca[0:3, 0] = z0
    x0_ca[3:6, 0] = np.array([140.0, 20.0, 10.0])  # rough guess
    x0_ca[6:9, 0] = np.array([0.0, 0.0, 0.0])

    P0_ca = np.diag([EST_MEAS_SIGMA**2]*3 + [100.0**2]*3 + [50.0**2]*3)

    # CV init (6)
    x0_cv = np.zeros((6, 1))
    x0_cv[0:3, 0] = z0
    x0_cv[3:6, 0] = np.array([140.0, 20.0, 10.0])
    P0_cv = np.diag([EST_MEAS_SIGMA**2]*3 + [100.0**2]*3)

    # Process noise tuning:
    # CA uses jerk noise (sigma_acc_ca parameter in imm.py is treated like sigma_jerk)
    # CV uses accel noise
    sigma_jerk_ca = CA_SIGMA  # (m/s^3) effective
    sigma_acc_cv = CV_SIGMA   # (m/s^2) effective

    imm = IMM(
        dt=DT,
        R=EST_MEAS_SIGMA**2 * np.eye(3),
        sigma_acc_ca=sigma_jerk_ca,
        sigma_acc_cv=sigma_acc_cv,
        PI=PI,
        mu0=mu0,
        x0_ca=x0_ca,
        P0_ca=P0_ca,
        x0_cv=x0_cv,
        P0_cv=P0_cv,
        embed_accel_sigma=MIXING_SIGMA**2,
        time_weighted_likelihood=TIME_WEIGHTED_LIKELIHOOD,
        likelihood_tau=LIKELIHOOD_TAU,
    )

    # ---------------- Run simulation + tracking ----------------
    truth_hist = np.zeros((steps, 9))
    meas_hist = np.zeros((steps, 3))
    est_hist = np.zeros((steps, 9))
    mu_hist = np.zeros((steps, 2))

    for k in range(steps):
        x_true = target.step()
        z = target.measure_position(TRUE_MEAS_SIGMA**2 * np.eye(3))

        out = imm.step(z)

        truth_hist[k, :] = x_true[:, 0]
        meas_hist[k, :] = z
        est_hist[k, :] = out["x_combined_common"][:, 0]
        mu_hist[k, :] = out["mu"]

    # ---------------- Print quick summary ----------------
    print("Final model probabilities [CA, CV]:", mu_hist[-1])
    print("Final truth pos:", truth_hist[-1, 0:3])
    print("Final est   pos:", est_hist[-1, 0:3])
    print("Final truth vel:", truth_hist[-1, 3:6])
    print("Final est   vel:", est_hist[-1, 3:6])

    # Optional plot (uncomment if you want)
    try:
        import matplotlib.pyplot as plt

        t = np.arange(steps) * DT

        # ---------------- Extract quantities ----------------
        # Altitude
        z_true = truth_hist[:, 2]
        z_est  = est_hist[:, 2]

        # Velocity magnitude
        v_true = np.linalg.norm(truth_hist[:, 3:6], axis=1)
        v_est  = np.linalg.norm(est_hist[:, 3:6], axis=1)

        # Acceleration magnitude
        a_true = np.linalg.norm(truth_hist[:, 6:9], axis=1)
        a_est  = np.linalg.norm(est_hist[:, 6:9], axis=1)

        # IMM mode probabilities
        p_ca = mu_hist[:, 0]
        p_cv = mu_hist[:, 1]

        # ---------------- Figure 1: Alt / Vel / Accel ----------------
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 8))

        # Altitude
        axs[0].plot(t, z_true, label="Truth")
        axs[0].plot(t, z_est, label="IMM est")
        axs[0].axvline(BOOST_END_TIME, linestyle="--", color="k")
        axs[0].set_ylabel("Altitude z (m)")
        axs[0].set_title("Altitude / Velocity / Acceleration vs Time")
        axs[0].legend()
        axs[0].grid(True)

        # Velocity magnitude
        axs[1].plot(t, v_true, label="Truth")
        axs[1].plot(t, v_est, label="IMM est")
        axs[1].axvline(BOOST_END_TIME, linestyle="--", color="k")
        axs[1].set_ylabel("|v| (m/s)")
        axs[1].legend()
        axs[1].grid(True)

        # Acceleration magnitude
        axs[2].plot(t, a_true, label="Truth")
        axs[2].plot(t, a_est, label="IMM est")
        axs[2].axvline(BOOST_END_TIME, linestyle="--", color="k")
        axs[2].set_ylabel("|a| (m/s²)")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()

        # ---------------- Figure 2: IMM mode probabilities ----------------
        plt.figure(figsize=(8, 4))
        plt.plot(t, p_ca, label="P(CA)")
        plt.plot(t, p_cv, label="P(CV)")
        plt.axvline(BOOST_END_TIME, linestyle="--", color="k", label="Boost end")
        plt.xlabel("Time (s)")
        plt.ylabel("Mode probability")
        plt.title("IMM Mode Probabilities")
        plt.legend()
        plt.grid(True)

        plt.show()

    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()
