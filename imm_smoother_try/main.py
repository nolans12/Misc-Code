# main.py
from __future__ import annotations

import numpy as np
from scipy.linalg import fractional_matrix_power

from imm import IMM
from target import Motion, Target3D
from ca import CAFilter
from measurement import Measurement
from smoother import IMMSmootherRTS

# ======================= CONFIG =======================

IMM_SMOOTHER = True

TOTAL_TIME = 30.0
DT = 0.1

MOTIONS = [
    Motion(start_t=0.0, end_t=15, accel=5, noise_sigma=10),
    Motion(start_t=15, end_t=30, accel=5.0, noise_sigma=1),
]

CA_1_SIGMA = 1.0
CA_2_SIGMA = 10.0

EST_MEAS_SIGMA = 15.0
TRUE_MEAS_SIGMA = 5.0

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
    steps = int(TOTAL_TIME / DT) + 1
    
    
    # INITIAL EVERYTHING

    # ---------------- Target ----------------
    target = Target3D(
        motions=MOTIONS,
    )
    target.set_initial(
        pos_xyz=np.array([0.0, 0.0, 0.0]),
        vel_xyz=np.array([100.0, -50.0, 0.0]),
    )

    # ---------------- IMM parameters ----------------
    PI = np.array([[0.99, 0.01],
                   [0.01, 0.99]])

    mu0 = np.array([0.50, 0.50])
    R_est = EST_MEAS_SIGMA ** 2 * np.eye(3)
    R_true = TRUE_MEAS_SIGMA ** 2 * np.eye(3)
    
    # First measurement at t=0 for initialization only
    z0 = target.measure_position(R_true)
    
     # ---------------- Initial states ----------------
    x0_ca = np.zeros((9, 1))
    x0_ca[0:3, 0] = z0
    x0_ca[3:6, 0] = [120.0, -52.0, 2.0]
    P0_ca_1 = np.diag([EST_MEAS_SIGMA ** 2] * 3 + [25.0 ** 2] * 3 + [CA_1_SIGMA ** 2] * 3)
    P0_ca_2 = np.diag([EST_MEAS_SIGMA ** 2] * 3 + [25.0 ** 2] * 3 + [CA_1_SIGMA ** 2] * 3)

    # ---------------- Forward IMM ----------------
    ca_1 = CAFilter(DT, CA_1_SIGMA, R_est, x0_ca, P0_ca_1)
    ca_2 = CAFilter(DT, CA_2_SIGMA, R_est, x0_ca, P0_ca_2)

    imm_fwd = IMM(
        models=[ca_1, ca_2],
        PI=PI,
        mu0=mu0,
        dt=DT,
    )

    truth_hist = np.zeros((steps, 9))
    meas_hist = np.zeros((steps, 3))
    x_fwd = np.zeros((steps, 9))
    P_fwd = np.zeros((steps, 9, 9))
    mu_fwd = np.zeros((steps, 2))
    t_fwd = np.zeros(steps)
    measurements: list[Measurement] = []

    # Give k = 0
    truth_hist[0] = target.x.reshape(9)
    out = imm_fwd.step()
    meas_hist[0] = [0, 0, 0]
    x_fwd[0] = out["x_common"][:, 0]
    P_fwd[0] = out["P_common"]
    mu_fwd[0] = out["mu"]
    t_fwd[0] = 0.0


    for k in range(steps):
        if k == 0:
            continue

        truth = target.step(DT)
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


    # # Print forward data
    # for snap in imm_fwd._history:
    #     print(f"Forward state: time: {snap['time']}, \n state: {snap['x_common']}\n p_mode: {snap['mu']}")

    # ---------------- IMM RTS Smoother ----------------
    if IMM_SMOOTHER:
        smoother = IMMSmootherRTS(fractional_matrix_power(PI, DT))
        sm = smoother.smooth(imm_fwd)
        x_smooth = sm["x_s"][:, :, 0]
        P_smooth = sm["P_s"]
        mu_smooth = sm["mu"]
    else:
        x_smooth = None
        P_smooth = None

    # ---------------- Plots ----------------
    try:
        import matplotlib.pyplot as plt

        # t = np.array([m.t for m in measurements])
        t = t_fwd

        # ---- Position with ±1σ ----
        fig_pos, axs_pos = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        labels_pos = ["x (m)", "y (m)", "z (m)"]

        for i in range(3):
            axs_pos[i].plot(t, truth_hist[:, i], "k", label="Truth")
            
            # Plot measurements as red dots
            label_meas = "Measurements" if i == 0 else None
            axs_pos[i].scatter(t, meas_hist[:, i], color="red", s=10, label=label_meas, zorder=5)

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
            for motion in MOTIONS:
                axs_pos[i].axvline(motion.start_t, linestyle="--", color="k")
                axs_pos[i].axvline(motion.end_t, linestyle="--", color="k")

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
            for motion in MOTIONS:
                axs_vel[i].axvline(motion.start_t, linestyle="--", color="k")
                axs_vel[i].axvline(motion.end_t, linestyle="--", color="k")

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
            for motion in MOTIONS:
                axs_acc[i].axvline(motion.start_t, linestyle="--", color="k")
                axs_acc[i].axvline(motion.end_t, linestyle="--", color="k")

        axs_acc[0].legend()
        axs_acc[-1].set_xlabel("Time (s)")
        fig_acc.tight_layout()

        # ---- Mode probabilities ----
        plt.figure(figsize=(9, 4))
        plt.plot(t, mu_fwd[:, 0], label="P(CA_1) forward")
        plt.plot(t, mu_fwd[:, 1], label="P(CA_2) forward")
        if IMM_SMOOTHER:
            plt.plot(t, mu_smooth[:, 0], "--", label="P(CA_1) smooth")
            plt.plot(t, mu_smooth[:, 1], "--", label="P(CA_2) smooth")
        for motion in MOTIONS:
            plt.axvline(motion.start_t, linestyle="--", color="k")
            plt.axvline(motion.end_t, linestyle="--", color="k")
        plt.xlabel("Time (s)")
        plt.ylabel("Mode probability")
        plt.legend()
        plt.grid()

        plt.show()

    except Exception as e:
        print("Plot skipped:", e)

    # # At the end print all measuremnet times and states
    # for meas in measurements:
    #     t = float(meas.t)
    #     z = meas.z  # length-3
    #     print(
    #         f'vector.push_back(make_meas('
    #         f'{t:.6f}, '
    #         f'(Eigen::VectorXd(3) << {z[0]:.6f}, {z[1]:.6f}, {z[2]:.6f}).finished(), '
    #         f'noise_sigma));'
    #     )


if __name__ == "__main__":
    main()
