import numpy as np
from numpy import typing as npt

def ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    Assumed given in km, so convert to meters

    Output is: lat (deg), lon (deg), alt (km)
    """
    x = x * 1000
    y = y * 1000
    z = z * 1000

    # WGS84 ellipsoid constants
    a = 6378137.0  # Semi-major axis (meters)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f**2  # Square of eccentricity

    # Computation
    b = a * (1 - f)  # Semi-minor axis
    ep2 = (a**2 - b**2) / b**2  # Second eccentricity squared
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lat = np.arctan2(z + ep2 * b * sin_theta**3, p - e2 * a * cos_theta**3)
    lon = np.arctan2(y, x)

    # Iterative calculation for altitude
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert latitude and longitude from radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt / 1000


def lla_to_ecef(lat: float, lon: float, alt: float) -> npt.NDArray:
    """Conversion of lat (deg), lon (deg), and alt (km) to ECEF coordinates in km"""
    # Define the WGS84 datum
    a = 6378137.0  # Semi-major axis (meters)
    b = 6356752.31424518  # Semi-minor axis (meters)
    e_squared = (a**2 - b**2) / a**2  # First eccentricity squared
    
    lat = np.radians(lat)
    lon = np.radians(lon)

    # Calculate the ECEF coordinates
    N = a / np.sqrt(1 - e_squared * np.sin(lat) ** 2)
    x = (N + alt * 1000) * np.cos(lat) * np.cos(lon)
    y = (N + alt * 1000) * np.cos(lat) * np.sin(lon)
    z = ((b**2 / a**2) * N + alt * 1000) * np.sin(lat)
    return np.array([x / 1000, y / 1000, z / 1000])



def sphere_line_intersection(
    line_point: npt.NDArray,
    line_direction: npt.NDArray,
    sphere_center: npt.NDArray | None = None,
    sphere_radius: float = 6378.0,
) -> npt.NDArray | None:
    """Compute the intersection point of a line and a sphere.

    When not provided, the sphere defaults to the Earth in
    the ECI frame (origin).

    Args:
        line_point: Point on the line.
        line_direction: Direction of the line.
        sphere_center: Coordinates of the sphere center.
        sphere_radius: Radius of the sphere.

    Returns:
        array or None: Intersection point(s) or None if no intersection.
    """
    if sphere_center is None:
        # Default to Earth center
        sphere_center = np.array((0, 0, 0))

    # Unpack sphere parameters
    x0, y0, z0 = sphere_center
    r = sphere_radius

    # Unpack line parameters
    x1, y1, z1 = line_point
    dx, dy, dz = line_direction

    # Compute coefficients for the quadratic equation
    a = dx**2 + dy**2 + dz**2
    b = 2 * (dx * (x1 - x0) + dy * (y1 - y0) + dz * (z1 - z0))
    c = (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2 - r**2

    # Compute discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return None
    elif discriminant == 0:
        # One intersection
        t = -b / (2 * a)
        intersection_point = np.array([x1 + t * dx, y1 + t * dy, z1 + t * dz])
        return intersection_point
    else:
        # Two intersections
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        intersection_point1 = np.array([x1 + t1 * dx, y1 + t1 * dy, z1 + t1 * dz])
        intersection_point2 = np.array([x1 + t2 * dx, y1 + t2 * dy, z1 + t2 * dz])

        # Calculate distances
        dist1 = np.linalg.norm(intersection_point1 - line_point)
        dist2 = np.linalg.norm(intersection_point2 - line_point)

        if dist1 < dist2:
            return intersection_point1
        else:
            return intersection_point2