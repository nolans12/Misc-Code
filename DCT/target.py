"""Ballistic target dynamics and trajectory simulation."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from common import lla_to_ecef

class BallisticTarget:
    """Ballistic trajectory with parabolic altitude profile.
    
    Works in LLA (Lat/Lon/Alt) frame:
    - Lat/Lon: Linear interpolation (constant horizontal velocity)
    - Altitude: Parabolic arc (climbs to max_altitude at midpoint, then descends)
    - Converts to ECEF only when returning position/velocity
    """

    def __init__(
        self,
        launch_lat_deg: float,
        launch_lon_deg: float,
        launch_alt_km: float,
        impact_lat_deg: float,
        impact_lon_deg: float,
        impact_alt_km: float,
        flight_duration_s: float,
        max_altitude_km: float,
    ):
        """Initialize a ballistic target.
        
        Args:
            launch_lat_deg: Launch latitude in degrees
            launch_lon_deg: Launch longitude in degrees
            launch_alt_km: Launch altitude in km
            impact_lat_deg: Impact latitude in degrees
            impact_lon_deg: Impact longitude in degrees
            impact_alt_km: Impact altitude in km
            flight_duration_s: Total flight time in seconds
            max_altitude_km: Maximum altitude at trajectory midpoint in km
        """
        self.lat0 = launch_lat_deg
        self.lon0 = launch_lon_deg
        self.alt0 = launch_alt_km
        
        self.lat1 = impact_lat_deg
        self.lon1 = impact_lon_deg
        self.alt1 = impact_alt_km
        
        self.duration = flight_duration_s
        self.max_alt = max_altitude_km

        self.alt_c = self.alt0
        self.alt_a = -4.0 * (self.max_alt - 0.5 * self.alt0 - 0.5 * self.alt1)
        self.alt_b = self.alt1 - self.alt0 - self.alt_a

    def state_at(self, time_s: float) -> NDArray[np.float64]:
        """Get position and velocity at a given time.
        
        Computes trajectory in LLA frame, then converts to ECEF.
        
        Args:
            time_s: Time in seconds since launch
            
        Returns:
            position: ECEF position (km)
        """
        t_norm = np.clip(time_s / self.duration, 0.0, 1.0)
        
        # Linear interpolation for lat/lon
        lat_deg = self.lat0 + (self.lat1 - self.lat0) * t_norm
        lon_deg = self.lon0 + (self.lon1 - self.lon0) * t_norm
        
        # Parabolic altitude profile
        alt_km = self.alt_a * t_norm**2 + self.alt_b * t_norm + self.alt_c
        
        # return ECEF
        pos_ecef_km = lla_to_ecef(lat_deg, lon_deg, alt_km)
        
        # Compute velocity via finite difference b/w two points and tiny dt
        dt = 0.1 # small dt
        if time_s < self.duration - dt:
            t_next = time_s + dt
            t_norm_next = t_next / self.duration
            
            lat_next = self.lat0 + (self.lat1 - self.lat0) * t_norm_next
            lon_next = self.lon0 + (self.lon1 - self.lon0) * t_norm_next
            alt_next = self.alt_a * t_norm_next**2 + self.alt_b * t_norm_next + self.alt_c
            
            pos_next_km = lla_to_ecef(lat_next, lon_next, alt_next)
            vel_ecef_km_s = (pos_next_km - pos_ecef_km) / dt
        elif time_s > dt:
            t_prev = time_s - dt
            t_norm_prev = t_prev / self.duration
            
            lat_prev = self.lat0 + (self.lat1 - self.lat0) * t_norm_prev
            lon_prev = self.lon0 + (self.lon1 - self.lon0) * t_norm_prev
            alt_prev = self.alt_a * t_norm_prev**2 + self.alt_b * t_norm_prev + self.alt_c
            
            pos_prev_km = lla_to_ecef(lat_prev, lon_prev, alt_prev)
            
            vel_ecef_km_s = (pos_ecef_km - pos_prev_km) / dt
        else:
            # at t=0, use forward difference
            vel_ecef_km_s = np.zeros(3)
        
        return pos_ecef_km, vel_ecef_km_s
