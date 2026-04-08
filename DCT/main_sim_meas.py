"""ASEN 6055 Final Project for Ballistic Tracking - Multi-Missile Version"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sensing import create_geo_satellites
from target import BallisticTarget
from plot_sim import LivePlotter

# RNG Seed
SEED = 2  
rng = np.random.default_rng(SEED)

# Meas Noise (sigma) for EKF
NOISE_SIGMA = 5e-4  # radians

# Sim Parameters
SIM_DURATION_S = 10 * 60  # sec
DT_S = 10.0  # sec, how often to take a meas and update ekf

# Multi-Missile Configuration
NUM_MISSILES = 1  # Number of missiles to simulate

# Missile Trajectories: list of [launch_lat, launch_lon, launch_alt, impact_lat, impact_lon, impact_alt, max_altitude]

# ###### CSO FORK:
# MISSILE_CONFIGS = [
#     [-10.0, -40.0, 0.0, 20.0, 0.0, 0.0, 250.0],   # Missile 1
#     [-12.5, -42.5, 0.0, 15.0, 5.0, 0.0, 250.0],    # Missile 2
#     [-15.0, -45.0, 0.0, 25.0, -5.0, 0.0, 250.0],  # Missile 3
# ]

# ###### CSO CROSS TRACK:
# MISSILE_CONFIGS = [
#     [-10.0, -40.0, 0.0, 20.0, 0.0, 0.0, 250.0],   # Missile 1
#     [0.0, 0.0, 0.0, 10.0, -40.0, 0.0, 250.0],    # Missile 2
# ]

###### Standard MSLs:
MISSILE_CONFIGS = [
    [-10.0, -40.0, 0.0, 20.0, 0.0, 0.0, 250.0],  # Missile 1
    [-6.0, -50.5, 0.0, 15.0, 15.0, 0.0, 250.0],  # Missile 2
    [-15.0, -45.0, 0.0, 25.0, -20.0, 0.0, 250.0],  # Missile 3
]

# Use only the first NUM_MISSILES from the config
MISSILE_CONFIGS = MISSILE_CONFIGS[:NUM_MISSILES]

# Compute zoom extent from all missiles
all_lats = []
all_lons = []
for config in MISSILE_CONFIGS:
    all_lats.extend([config[0], config[3]])
    all_lons.extend([config[1], config[4]])

ZOOM_MAG = 15.0
ZOOM = [min(all_lats) - ZOOM_MAG, min(all_lons) - ZOOM_MAG,
        max(all_lats) + ZOOM_MAG, max(all_lons) + ZOOM_MAG]

DATA_DIR = Path(__file__).resolve().parent / "data"


def run_simulation(seed: int = 0, live_plot: bool = True) -> Path:
    print(f"=== Multi-Missile Simulation ===")
    print(f"Number of missiles: {NUM_MISSILES}")
    print(f"Simulation duration: {SIM_DURATION_S}s")

    #### CREATE TARGETS ####
    targets = []
    for i, config in enumerate(MISSILE_CONFIGS):
        launch_lat, launch_lon, launch_alt, impact_lat, impact_lon, impact_alt, max_alt = config
        print(f"\nMissile {i + 1}:")
        print(f"  LAUNCH: lat={launch_lat}°, lon={launch_lon}°, alt={launch_alt} km")
        print(f"  IMPACT: lat={impact_lat}°, lon={impact_lon}°, alt={impact_alt} km")
        print(f"  MAX ALTITUDE: {max_alt} km")

        target = BallisticTarget(
            launch_lat_deg=launch_lat,
            launch_lon_deg=launch_lon,
            launch_alt_km=launch_alt,
            impact_lat_deg=impact_lat,
            impact_lon_deg=impact_lon,
            impact_alt_km=impact_alt,
            flight_duration_s=SIM_DURATION_S,
            max_altitude_km=max_alt,
        )
        targets.append(target)

    #### CREATE SATELLITES ####
    # One near launch, one near impact, one in between
    # w/ random +- 5 degree jitter
    jittered_positions = []
    for lat, lon in [
        [MISSILE_CONFIGS[0][0], MISSILE_CONFIGS[0][1]],  # Launch
        [MISSILE_CONFIGS[0][3], MISSILE_CONFIGS[0][4]],  # Impact
        # [(MISSILE_CONFIGS[0][0] + MISSILE_CONFIGS[0][3]) / 2, (MISSILE_CONFIGS[0][1] + MISSILE_CONFIGS[0][4]) / 2],
        # Midpoint
    ]:
        jitter_lat = lat + rng.uniform(-10, 10)
        jitter_lon = lon + rng.uniform(-10, 10)
        jittered_positions.append([jitter_lat, jitter_lon])
    satellites = create_geo_satellites(
        positions=jittered_positions,
        noise_sigma=NOISE_SIGMA,
        rng=rng
    )
    print(f"\nCreated {len(satellites)} satellites")

    #### RUN SIMULATION ####
    # Store results for each missile
    all_truth = [[] for _ in range(NUM_MISSILES)]
    all_measurements = [[] for _ in range(NUM_MISSILES)]

    #### PLOT SIMULATION ####
    plotter = LivePlotter(zoom_extent=ZOOM, num_missiles=NUM_MISSILES) if live_plot else None

    # Times to take measurements at, every dt
    times = np.arange(0.0, SIM_DURATION_S + DT_S, DT_S)

    meas_rows = []
    for idx, t in enumerate(times):
        # Process each missile
        for missile_idx, target in enumerate(targets):
            pos, vel = target.state_at(t)
            truth_state = np.concatenate([pos, vel])
            all_truth[missile_idx].append(truth_state)

            # Sample measurements for this missile
            measurements = [sat.measure(t, pos, rng=rng) for sat in satellites]
            all_measurements[missile_idx].extend(measurements)

            # Create dataframe row
            meas_rows.extend([{"sat": meas.sat.name, "target": missile_idx + 1, "time": meas.time, "az": meas.az,
                              "el": meas.el, "R": np.sqrt(meas.R[0, 0]), "sat_x": meas.sat.position[0], "sat_y": meas.sat.position[1],  "sat_z": meas.sat.position[2], "FoR": meas.sat.field_of_regard} for meas in measurements])

            print(
                f"For satellite 1, target {missile_idx}:\ntime:\n{measurements[1].time}\nmeas:\n{measurements[1].az}, {measurements[1].el}\n")

        # Update live plot
        if plotter:
            plotter.update(
                time=t,
                times_history=times[:idx + 1].tolist(),
                all_truth=all_truth,
                all_measurements=all_measurements,
                satellites=satellites,
            )

    if plotter:
        print("\nSimulation complete.")
        plotter.close()

    #### SAVE RESULTS ####
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "simulation_run.npz"

    df_meas = pd.DataFrame(meas_rows)
    df_meas.to_csv(f'{DATA_DIR}/measurements.csv', index=False)

    # Convert to arrays
    save_dict = {
        'times': times,
        'num_missiles': NUM_MISSILES,
    }

    for i in range(NUM_MISSILES):
        save_dict[f'truth_{i}'] = np.stack(all_truth[i])

        # Save measurements
        meas = all_measurements[i]
        save_dict[f'meas_times_{i}'] = np.array([m.time for m in meas])
        save_dict[f'meas_sat_positions_{i}'] = np.array([m.sat.position for m in meas])
        save_dict[f'meas_azs_{i}'] = np.array([m.az for m in meas])
        save_dict[f'meas_els_{i}'] = np.array([m.el for m in meas])

    np.savez(output_path, **save_dict)
    return output_path


if __name__ == "__main__":
    result_path = run_simulation()
    print(f"Saved ballistic tracking results to {result_path}")
