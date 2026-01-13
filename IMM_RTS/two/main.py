from types import MethodType

import numpy as np
from imm import IMM
from models import make_ca_3d, make_cv_3d, to_external_ca, to_internal_ca, to_external_cv, to_internal_cv
from measurement import Measurement
from target import Target3D
from smoother import IMMSmootherRTS


# ======================= CONFIG =======================

IMM_SMOOTHER = True

TOTAL_TIME = 60.0
FIRST_ACCEL = 20.0
SECOND_ACCEL = 40.0
DT = 1.0

CA_SIGMA = 10.0
CV_SIGMA = 10.0
MIXING_SIGMA = CA_SIGMA * 0

EST_MEAS_SIGMA = 15.0
TRUE_MEAS_SIGMA = 5.0

TARG_GS = 5.0
TARG_SIGMA = 1.0

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

    # ---------------- Truth target ----------------
    target = Target3D(
        dt=DT,
        FIRST_ACCEL=FIRST_ACCEL,
        SECOND_ACCEL=SECOND_ACCEL,
        g=9.80665,
        accel_boost_g=TARG_GS,
        process_sigma_acc=TARG_SIGMA,
        seed=100,
    )
    target.set_initial(
        pos_xyz=np.array([0.0, 0.0, 0.0]),
        vel_xyz=np.array([100.0, 100.0, 0.0]),
    )

    R = np.diag([EST_MEAS_SIGMA ** 2, EST_MEAS_SIGMA ** 2, EST_MEAS_SIGMA ** 2])
    R_true = np.diag([TRUE_MEAS_SIGMA ** 2, TRUE_MEAS_SIGMA ** 2, TRUE_MEAS_SIGMA ** 2])

    ca = make_ca_3d(DT, CA_SIGMA, R)
    ca.to_external = MethodType(to_external_ca, ca)
    ca.to_internal = MethodType(to_internal_ca, ca)
    cv = make_cv_3d(DT, CV_SIGMA, R)
    cv.to_external = MethodType(to_external_cv, cv)
    cv.to_internal = MethodType(to_internal_cv, cv)

    models = [ca, cv]

    PI = np.array([
        [0.95, 0.05],
        [0.05, 0.95],
    ])

    mu0 = np.array([0.9, 0.1])

    imm = IMM(
        models=models,
        PI=PI,
        mu0=mu0,
        dt=DT,
        t0=0.0,
    )

    z0 = target.measure_position(R_true)

    x0 = np.zeros((9))
    x0[0:3] = z0
    x0[3:6] = [0, 0, 0]
    x0[6:9] = [0, 0, 0]
    P0 = np.diag([EST_MEAS_SIGMA ** 2] * 3 + [CV_SIGMA ** 2] * 3 + [CA_SIGMA ** 2] * 3)

    imm.set_state(x0, P0, mu0)


    ## CONTAINERS
    steps = int(TOTAL_TIME / DT) + 1
    truth_hist = np.zeros((steps, 9))
    meas_hist = np.zeros((steps, 3))
    x_fwd = np.zeros((steps, 9))
    P_fwd = np.zeros((steps, 9, 9))
    mu_fwd = np.zeros((steps, 2))
    t_fwd = np.zeros(steps)
    measurements: list[Measurement] = []

    # Give k = 0
    truth_hist[0] = target.x.reshape(9)
    meas = Measurement(t=target.t, z=z0, R=R)
    measurements.append(meas)
    # out = imm.step(meas)
    meas_hist[0] = [0, 0, 0]
    x_fwd[0] = x0
    P_fwd[0] = P0
    mu_fwd[0] = mu0
    t_fwd[0] = 0.0

    t = 0.0
    for k in range(steps):
        t += DT

        truth = target.step()
        z_true = target.measure_position(R_true)
        meas = Measurement(t=target.t, z=z_true, R=R)
        out = imm.step(meas)

        x_fwd[k] = out["x_common"]
        P_fwd[k] = out["P_common"]
        mu_fwd[k] = out["mu"]
        t_fwd[k] = meas.t
        measurements.append(meas)
        truth_hist[k] = truth[:, 0]
        meas_hist[k] = z_true

    # ---------------- IMM RTS Smoother ----------------
    if IMM_SMOOTHER:
        smoother = IMMSmootherRTS(PI)
        sm = smoother.smooth(imm)
        x_smooth = sm["x_s"][:, :]
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
