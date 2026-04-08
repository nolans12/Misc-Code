"""Simplified sensing utilities for a single ballistic target observed by GEO sats."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from common import ecef_to_lla, lla_to_ecef, sphere_line_intersection
colors = [
    "darkred",
    "darkgreen",
    "darkblue",
    "indigo",
    "darkorange",
    "darkmagenta",
    "maroon",
    "navy",
    "teal",
    "brown",
]

EARTH_RADIUS_KM = 6378.0
GEO_ALTITUDE_KM = 36000.0

def measurement_model(target_pos: NDArray[np.float64], sat_pos: NDArray[np.float64]) -> tuple[float, float]:
    """Compute azimuth and elevation from satellite to target.
    
    Args:
        target_pos: Target position in ECEF [km]
        sat_pos: Satellite position in ECEF [km]
        
    Returns:
        (azimuth, elevation) in radians
    """
    rel = target_pos - sat_pos
    x, y, z = rel
    az = np.arctan2(y, x)
    rng_norm = np.linalg.norm(rel)
    elev = np.arcsin(z / rng_norm)
    return az, elev

@dataclass(slots=True)
class Measurement:
    """Container for az/el measurements."""
    time: float
    sat: GeoSatellite
    az: float  # azimuth [rad]
    el: float  # elevation [rad]
    R: NDArray[np.float64]  # 2x2 covariance [rad^2]
    lat: float # latitude [deg]
    lon: float # longitude [deg]

class GeoSatellite:
    """Fixed GEO satellite with a simple bearings-only sensor."""

    def __init__(self, name: str, position: NDArray[np.float64], noise_sigma: float, rng):
        self.name = name
        self.position = position
        self._R = np.diag([noise_sigma**2, noise_sigma**2])
        self.color = rng.choice(colors)
        self.field_of_regard = 30

    def measure(
        self,
        time_s: float,
        target_pos: NDArray[np.float64],
        rng: np.random.Generator
    ) -> Measurement:

        az, elev = measurement_model(target_pos, self.position)
        noise = rng.multivariate_normal(mean=[0.0, 0.0], cov=self._R)
        az += noise[0]
        elev += noise[1]

        # Az, El, R to ECEF
        true_range = np.linalg.norm(target_pos - self.position)
        rel_x = np.cos(az) * np.cos(elev)
        rel_y = np.sin(az) * np.cos(elev)
        rel_z = np.sin(elev)
        meas_ecef = self.position + np.array([rel_x, rel_y, rel_z]) * true_range
        lla = ecef_to_lla(meas_ecef[0], meas_ecef[1], meas_ecef[2])

        return Measurement(
            time=time_s,
            sat=self,
            az=az + noise[0],
            el=elev + noise[1],
            R=self._R.copy() * 3,
            lat=lla[0],
            lon=lla[1]
        )

def create_geo_satellites(positions: list[list[float]], noise_sigma: float, rng) -> list['GeoSatellite']:
    """Create GEO satellites at specified lat/lon positions.
    
    Args:
        positions: List of [lat, lon] in degrees
        noise_sigma: Measurement noise sigma [rad]
    """
    satellites: list[GeoSatellite] = []
    radius = EARTH_RADIUS_KM + GEO_ALTITUDE_KM
    
    for idx, (lat, lon) in enumerate(positions):
        # Init Pos in ECEF from Lat Lon and GEO Altitude
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        pos = np.array([
            radius * np.cos(lat_rad) * np.cos(lon_rad),
            radius * np.cos(lat_rad) * np.sin(lon_rad),
            radius * np.sin(lat_rad),
        ])
        satellites.append(GeoSatellite(name=f"GEO-{idx+1}", position=pos, noise_sigma=noise_sigma, rng=rng))
    return satellites

