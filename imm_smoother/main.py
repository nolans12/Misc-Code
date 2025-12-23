# main.py
from __future__ import annotations

import numpy as np
from imm import IMM
from target import Target3D
from ca import CAFilter
from cv import CVFilter
from measurement import Measurement
from smoother_old import IMMSmootherRTS
# from smoother import IMMSmootherRTS

# ======================= CONFIG =======================

IMM_SMOOTHER = True

TOTAL_TIME = 20.0
FIRST_ACCEL = 10.0
SECOND_ACCEL = 20.0
DT = 0.1

CA_SIGMA = 10.0
CV_SIGMA = 10.0
MIXING_SIGMA = CA_SIGMA * 0

EST_MEAS_SIGMA = 15.0
TRUE_MEAS_SIGMA = 3.0

TARG_GS = 5.0
TARG_SIGMA = 0.1

# =====================================================


def plot_with_sigma(ax, t, mean, P, idx, label, color, linestyle="-", alpha=0.25):
    sigma = np.sqrt(P[:, idx, idx])
    ax.plot(t, mean[:, idx], label=label, color=color, linestyle=linestyle)
    ax.fill_between(
        t,
        mean[:, idx] - sigma,
        mean[:, idx] + sigma,
        color=color,
        alpha=alpha,
    )


def main():
    steps = int(TOTAL_TIME / DT)

    # ---------------- Truth target ----------------
    target = Target3D(
        dt=DT,
        FIRST_ACCEL=FIRST_ACCEL,
        SECOND_ACCEL=SECOND_ACCEL,
        g=9.80665,
        accel_boost_g=TARG_GS,
        process_sigma_acc=TARG_SIGMA,
        seed=59,
    )
    target.set_initial(
        pos_xyz=np.array([0.0, 0.0, 0.0]),
        vel_xyz=np.array([150.0, 30.0, 0.0]),
    )

    # ---------------- IMM parameters ----------------
    PI = np.array([[0.99, 0.01],
                   [0.01, 0.99]])

    mu0 = np.array([0.9, 0.1])
    R_est = EST_MEAS_SIGMA ** 2 * np.eye(3)
    R_true = TRUE_MEAS_SIGMA ** 2 * np.eye(3)

    # First measurement at t=0 for initialization only
    z0 = target.measure_position(R_true)

    # ---------------- Initial states ----------------
    x0_ca = np.zeros((9, 1))
    x0_ca[0:3, 0] = z0
    x0_ca[3:6, 0] = [140.0, 20.0, 10.0]
    P0_ca = np.diag([EST_MEAS_SIGMA ** 2] * 3 + [100.0 ** 2] * 3 + [50.0 ** 2] * 3)

    x0_cv = np.zeros((6, 1))
    x0_cv[0:3, 0] = z0
    x0_cv[3:6, 0] = [140.0, 20.0, 10.0]
    P0_cv = np.diag([EST_MEAS_SIGMA ** 2] * 3 + [100.0 ** 2] * 3)

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
    t_fwd = np.zeros(steps)
    measurements: list[Measurement] = []

    for k in range(steps):
        truth = target.step()
        z_true = target.measure_position(R_true)

        meas = Measurement(t=target.t, z=z_true, R=R_est)
        measurements.append(meas)

        out = imm_fwd.step(meas)

        truth_hist[k] = truth[:, 0]
        meas_hist[k] = z_true
        x_fwd[k] = out["x_common"][:, 0]
        P_fwd[k] = out["P_common"]
        mu_fwd[k] = out["mu"]
        t_fwd[k] = meas.t

    # ---------------- IMM RTS Smoother ----------------
    if IMM_SMOOTHER:
        smoother = IMMSmootherRTS(PI)
        sm = smoother.smooth(imm_fwd, measurements)
        x_smooth = sm["x_s"][:, :, 0]
        P_smooth = sm["P_s"]
        mu_smooth = sm["mu"]
    else:
        x_smooth = None
        P_smooth = None

    # ---------------- Plots ----------------
    try:
        import matplotlib.pyplot as plt

        t = np.array([m.t for m in measurements])

        # ---- Position with ±1σ ----
        fig_pos, axs_pos = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        labels_pos = ["x (m)", "y (m)", "z (m)"]

        for i in range(3):
            axs_pos[i].plot(t, truth_hist[:, i], "k", label="Truth")

            plot_with_sigma(
                axs_pos[i], t, x_fwd, P_fwd, i,
                label="Forward IMM", color="tab:blue", linestyle="--", alpha=0.2
            )

            if IMM_SMOOTHER:
                plot_with_sigma(
                    axs_pos[i], t, x_smooth, P_smooth, i,
                    label="Smoothed IMM", color="tab:orange", linestyle="-", alpha=0.3
                )

            axs_pos[i].set_ylabel(labels_pos[i])
            axs_pos[i].grid(True)
            axs_pos[i].axvline(FIRST_ACCEL, linestyle="--", color="k")
            axs_pos[i].axvline(SECOND_ACCEL, linestyle="--", color="k")

        axs_pos[0].legend()
        axs_pos[-1].set_xlabel("Time (s)")
        fig_pos.tight_layout()

        # ---- Velocity with ±1σ ----
        fig_vel, axs_vel = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        labels_vel = ["vx (m/s)", "vy (m/s)", "vz (m/s)"]

        for i in range(3):
            idx = 3 + i
            axs_vel[i].plot(t, truth_hist[:, idx], "k", label="Truth")

            plot_with_sigma(
                axs_vel[i], t, x_fwd, P_fwd, idx,
                label="Forward IMM", color="tab:blue", linestyle="--", alpha=0.2
            )

            if IMM_SMOOTHER:
                plot_with_sigma(
                    axs_vel[i], t, x_smooth, P_smooth, idx,
                    label="Smoothed IMM", color="tab:orange", linestyle="-", alpha=0.3
                )

            axs_vel[i].set_ylabel(labels_vel[i])
            axs_vel[i].grid(True)
            axs_vel[i].axvline(FIRST_ACCEL, linestyle="--", color="k")
            axs_vel[i].axvline(SECOND_ACCEL, linestyle="--", color="k")

        axs_vel[0].legend()
        axs_vel[-1].set_xlabel("Time (s)")
        fig_vel.tight_layout()

        # ---- Acceleration with ±1σ ----
        fig_acc, axs_acc = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        labels_acc = ["$a_x$ (m/s²)", "$a_y$ (m/s²)", "$a_z$ (m/s²)"]

        for i in range(3):
            idx = 6 + i
            axs_acc[i].plot(t, truth_hist[:, idx], "k", label="Truth")

            plot_with_sigma(
                axs_acc[i], t, x_fwd, P_fwd, idx,
                label="Forward IMM", color="tab:blue", linestyle="--", alpha=0.2
            )

            if IMM_SMOOTHER:
                plot_with_sigma(
                    axs_acc[i], t, x_smooth, P_smooth, idx,
                    label="Smoothed IMM", color="tab:orange", linestyle="-", alpha=0.3
                )

            axs_acc[i].set_ylabel(labels_acc[i])
            axs_acc[i].grid(True)
            axs_acc[i].axvline(FIRST_ACCEL, linestyle="--", color="k")
            axs_acc[i].axvline(SECOND_ACCEL, linestyle="--", color="k")

        axs_acc[0].legend()
        axs_acc[-1].set_xlabel("Time (s)")
        fig_acc.tight_layout()

        # ---- Mode probabilities ----
        plt.figure(figsize=(9, 4))
        plt.plot(t, mu_fwd[:, 0], label="P(CA) forward")
        plt.plot(t, mu_fwd[:, 1], label="P(CV) forward")
        if IMM_SMOOTHER:
            plt.plot(t, mu_smooth[:, 0], "--", label="P(CA) smooth")
            plt.plot(t, mu_smooth[:, 1], "--", label="P(CV) smooth")
        plt.axvline(FIRST_ACCEL, linestyle="--", color="k")
        plt.axvline(SECOND_ACCEL, linestyle="--", color="k")
        plt.xlabel("Time (s)")
        plt.ylabel("Mode probability")
        plt.legend()
        plt.grid()

        plt.show()

    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()
