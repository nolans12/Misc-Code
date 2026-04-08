"""Plotting file - Multi-Missile Support"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import chi2
from common import ecef_to_lla
from sensing import GeoSatellite, Measurement

EARTH_RADIUS_KM = 6378.0
DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "data" / "simulation_run.npz"

# Color scheme for multiple missiles
MISSILE_COLORS = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']


class LivePlotter:
    """Real-time ground track visualization during simulation - Multi-Missile Support."""

    def __init__(self, zoom_extent: list[float] | None = None, save_frames: bool = True, num_missiles: int = 1):
        """
        Args:
            zoom_extent: [min_lat, min_lon, max_lat, max_lon] for plot extent
            save_frames: If True, save frames for creating animation
            num_missiles: Number of missiles to track
        """
        plt.ion()
        self.num_missiles = num_missiles
        self.fig = plt.figure(figsize=(14, 8))
        self.ax_ground = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        if zoom_extent:
            self.ax_ground.set_extent(
                [zoom_extent[1], zoom_extent[3], zoom_extent[0], zoom_extent[2]],
                crs=ccrs.PlateCarree()
            )
        else:
            self.ax_ground.set_global()

        self.ax_ground.coastlines(linewidth=0.5)
        self.ax_ground.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        self.ax_ground.add_feature(cfeature.LAND, facecolor="#f4f2ec")
        self.ax_ground.add_feature(cfeature.OCEAN, facecolor="#c6dbef")

        gl = self.ax_ground.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )
        gl.top_labels = False
        gl.right_labels = False

        self.ax_ground.set_title("Live Missile Trajectory Ground Track", fontsize=14, weight="bold")

        # Storage for each missile's track
        self.truth_tracks = [None] * num_missiles
        self.est_tracks = [None] * num_missiles
        self.truth_markers = [None] * num_missiles
        self.est_markers = [None] * num_missiles

        # Storage for satellites and measurements
        self.sat_markers = []
        self.meas_scatters = [None] * num_missiles

        # Frame saving for animation
        self.save_frames = save_frames
        self.frame_dir = Path(__file__).resolve().parent / "data" / "frames"
        self.frame_count = 0

        if self.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            for old_frame in self.frame_dir.glob("frame_*.png"):
                old_frame.unlink()

        plt.tight_layout()
        plt.pause(0.001)

    def update(
        self,
        time: float,
        times_history: list[float],
        all_truth: list[list[np.ndarray]],
        all_measurements: list[list[Measurement]],
        satellites: list[GeoSatellite] | None = None,
    ) -> None:
        """Update ground track with new data for all missiles."""

        # Plot each missile
        for missile_idx in range(self.num_missiles):
            if len(all_truth[missile_idx]) == 0:
                continue

            color = MISSILE_COLORS[missile_idx % len(MISSILE_COLORS)]

            # Current state
            truth_state = all_truth[missile_idx][-1]

            # Convert ECEF to LLA
            truth_lla = ecef_to_lla(truth_state[0], truth_state[1], truth_state[2])

            # Ground track update
            if len(all_truth[missile_idx]) > 1:
                truth_llas = np.array([ecef_to_lla(s[0], s[1], s[2]) for s in all_truth[missile_idx]])

                if self.truth_tracks[missile_idx]:
                    self.truth_tracks[missile_idx].remove()
                if self.est_tracks[missile_idx]:
                    self.est_tracks[missile_idx].remove()

                # Truth track (solid line)
                (self.truth_tracks[missile_idx],) = self.ax_ground.plot(
                    truth_llas[:, 1], truth_llas[:, 0],
                    '-', color=color, linewidth=2.5, alpha=0.4,
                    label=f"Missile {missile_idx+1} Truth" if missile_idx == 0 else "",
                    transform=ccrs.PlateCarree(), zorder=3
                )

            # Current position markers
            if self.truth_markers[missile_idx]:
                self.truth_markers[missile_idx].remove()

            (self.truth_markers[missile_idx],) = self.ax_ground.plot(
                truth_lla[1], truth_lla[0],
                'o', color=color, markersize=10, alpha=0.6,
                transform=ccrs.PlateCarree(), zorder=4
            )

            # Plot measurements as X's
            measurements = all_measurements[missile_idx]
            if measurements and len(measurements) > 0:
                if self.meas_scatters[missile_idx]:
                    self.meas_scatters[missile_idx].remove()

                meas_colors = [m.sat.color for m in measurements]
                meas_lat = [m.lat for m in measurements]
                meas_lon = [m.lon for m in measurements]

                self.meas_scatters[missile_idx] = self.ax_ground.scatter(
                    meas_lon, meas_lat,
                    marker='x', s=30, c=meas_colors, alpha=0.6,
                    transform=ccrs.PlateCarree(), zorder=7
                )

        # Plot satellite positions (once)
        if satellites and not self.sat_markers:
            if not hasattr(self, "sat_legend_marker"):
                self.sat_legend_marker, = self.ax_ground.plot(
                    [None], [None], "^", markersize=12, color="black",
                    label="GEO Satellites", transform=ccrs.PlateCarree(), zorder=6
                )
                self.sat_markers.append(self.sat_legend_marker)

            for sat in satellites:
                sat_pos = sat.position
                sat_lla = ecef_to_lla(sat_pos[0], sat_pos[1], sat_pos[2])
                (marker,) = self.ax_ground.plot(
                    sat_lla[1], sat_lla[0], "^", markersize=12,
                    transform=ccrs.PlateCarree(), zorder=6, color=sat.color
                )
                self.sat_markers.append(marker)

            # Measurement legend marker (once)
            if not hasattr(self, "meas_legend_marker"):
                self.meas_legend_marker, = self.ax_ground.plot(
                    [None], [None], "x", markersize=8, color="black",
                    label="Measurements", transform=ccrs.PlateCarree(), zorder=7
                )

        # Update legend on first frame
        if len(times_history) == 1:
            self.ax_ground.legend(loc="best", fontsize=10)

        # Update title
        time_min = time / 60.0
        self.ax_ground.set_title(
            f"Live Missile Trajectory Ground Track (t={time_min:.2f} min)",
            fontsize=14, weight="bold"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

        # Save frame
        if self.save_frames:
            frame_path = self.frame_dir / f"frame_{self.frame_count:04d}.png"
            self.fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            self.frame_count += 1

    def close(self) -> None:
        plt.ioff()

        # Create animation from saved frames
        if self.save_frames and self.frame_count > 0:
            print(f"\nCreating animation from {self.frame_count} frames...")
            output_dir = self.frame_dir.parent
            try:
                import imageio

                gif_path = output_dir / "live_plot_animation.gif"
                frames = []
                for i in range(self.frame_count):
                    frame_path = self.frame_dir / f"frame_{i:04d}.png"
                    if frame_path.exists():
                        frames.append(imageio.imread(frame_path))

                if frames:
                    imageio.mimsave(gif_path, frames, fps=10, loop=0)
                    print(f"Saved animation GIF to {gif_path}")

                try:
                    mp4_path = output_dir / "live_plot_animation.mp4"
                    imageio.mimsave(mp4_path, frames, fps=10, codec='libx264')
                    print(f"Saved animation MP4 to {mp4_path}")
                except Exception as e:
                    print(f"Could not create MP4: {e}")

            except ImportError:
                print("imageio not installed. Install with 'pip install imageio' to create animations.")
            except Exception as e:
                print(f"Error creating animation: {e}")

        plt.show()


def plot_final_results(data_path: Path | str | None = None, zoom_extent: list[float] | None = None) -> None:
    """Create final results plot for multi-missile simulation."""
    if data_path is None:
        data_path = DEFAULT_DATA_FILE
    else:
        data_path = Path(data_path)
        if data_path.is_dir():
            data_path = data_path / "simulation_run.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"No simulation results found at {data_path}")

    data = np.load(data_path)
    times = data["times"]
    num_missiles = int(data.get("num_missiles", 1))

    print(f"Plotting results for {num_missiles} missiles")

    # Create figure with subplots: altitude, velocity, ground track
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.3)

    # 1. Altitude Profile
    ax_alt = fig.add_subplot(gs[0, 0])
    for i in range(num_missiles):
        truth = data[f'truth_{i}']
        estimates = data[f'estimates_{i}']
        cov_diagonals = data[f'cov_diagonals_{i}']

        truth_lla = np.array([ecef_to_lla(s[0], s[1], s[2]) for s in truth])
        est_lla = np.array([ecef_to_lla(s[0], s[1], s[2]) for s in estimates])

        color = MISSILE_COLORS[i % len(MISSILE_COLORS)]

        ax_alt.plot(times, truth_lla[:, 2], '-', color=color, linewidth=2.5,
                   alpha=0.4, label=f"Missile {i+1} Truth", zorder=2)
        ax_alt.plot(times, est_lla[:, 2], '--', color=color, linewidth=2,
                   label=f"Missile {i+1} Est", zorder=3)

        alt_std = np.sqrt(cov_diagonals[:, 2])
        ax_alt.fill_between(times, est_lla[:, 2] - alt_std, est_lla[:, 2] + alt_std,
                           color=color, alpha=0.1, zorder=1)

    ax_alt.set_title("Altitude Profiles", fontsize=14, weight="bold")
    ax_alt.set_xlabel("Time [s]", fontsize=12)
    ax_alt.set_ylabel("Altitude [km]", fontsize=12)
    ax_alt.legend(loc="best", fontsize=9, ncol=2)
    ax_alt.grid(True, alpha=0.5, linestyle='--', linewidth=0.7)

    # 2. Velocity Magnitude Profile
    ax_vel = fig.add_subplot(gs[1, 0])
    for i in range(num_missiles):
        truth = data[f'truth_{i}']
        estimates = data[f'estimates_{i}']

        truth_vel_mag = np.linalg.norm(truth[:, 3:6], axis=1)
        est_vel_mag = np.linalg.norm(estimates[:, 3:6], axis=1)

        color = MISSILE_COLORS[i % len(MISSILE_COLORS)]

        ax_vel.plot(times, truth_vel_mag, '-', color=color, linewidth=2.5,
                   alpha=0.4, label=f"Missile {i+1} Truth", zorder=2)
        ax_vel.plot(times, est_vel_mag, '--', color=color, linewidth=2,
                   label=f"Missile {i+1} Est", zorder=3)

    ax_vel.set_title("Velocity Magnitude Profiles", fontsize=14, weight="bold")
    ax_vel.set_xlabel("Time [s]", fontsize=12)
    ax_vel.set_ylabel("Velocity [km/s]", fontsize=12)
    ax_vel.legend(loc="best", fontsize=9, ncol=2)
    ax_vel.grid(True, alpha=0.5, linestyle='--', linewidth=0.7)

    # 3. Ground Track
    ax_ground = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())

    if zoom_extent:
        ax_ground.set_extent([zoom_extent[1], zoom_extent[3], zoom_extent[0], zoom_extent[2]],
                            crs=ccrs.PlateCarree())
    else:
        ax_ground.set_global()

    ax_ground.coastlines(linewidth=0.5)
    ax_ground.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax_ground.add_feature(cfeature.LAND, facecolor="#f4f2ec")
    ax_ground.add_feature(cfeature.OCEAN, facecolor="#c6dbef")

    gl = ax_ground.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    for i in range(num_missiles):
        truth = data[f'truth_{i}']
        estimates = data[f'estimates_{i}']

        truth_lla = np.array([ecef_to_lla(s[0], s[1], s[2]) for s in truth])
        est_lla = np.array([ecef_to_lla(s[0], s[1], s[2]) for s in estimates])

        color = MISSILE_COLORS[i % len(MISSILE_COLORS)]

        ax_ground.plot(truth_lla[:, 1], truth_lla[:, 0], '-', color=color, linewidth=2.5,
                      alpha=0.4, label=f"Missile {i+1} Truth", transform=ccrs.PlateCarree(), zorder=3)
        ax_ground.plot(est_lla[:, 1], est_lla[:, 0], '--', color=color, linewidth=2,
                      label=f"Missile {i+1} Est", transform=ccrs.PlateCarree(), zorder=2)

        # Mark launch and impact
        ax_ground.plot(truth_lla[0, 1], truth_lla[0, 0], 'o', color=color, markersize=8,
                      transform=ccrs.PlateCarree(), zorder=5)
        ax_ground.plot(truth_lla[-1, 1], truth_lla[-1, 0], 's', color=color, markersize=8,
                      transform=ccrs.PlateCarree(), zorder=5)

    ax_ground.set_title("Multi-Missile Ground Tracks", fontsize=14, weight="bold")
    ax_ground.legend(loc="best", fontsize=9, ncol=2)

    plt.tight_layout()

    # Save figure
    output_dir = data_path.parent
    output_path_png = output_dir / "tracking_results.png"
    output_path_pdf = output_dir / "tracking_results.pdf"
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Saved tracking results to {output_path_png} and {output_path_pdf}")

    plt.show(block=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot multi-missile simulation results")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to data directory or .npz file")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else None

    try:
        from main_sim_meas import ZOOM
        zoom_extent = ZOOM
    except ImportError:
        zoom_extent = None

    plot_final_results(data_path=data_path, zoom_extent=zoom_extent)
    plt.show()