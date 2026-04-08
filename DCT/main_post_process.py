from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TIME_TO_FREEZE = 500 # sec

DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_TARG_ALT = 1000
R_EARTH = 6_371  # km
COLORS = ["red", "cyan", "lime", "orange", "magenta"]

df = pd.read_csv(f'{DATA_DIR}/measurements.csv')

## First - make the monotrack plots for each sensor

df['target_plot'] = df['target'].astype(str)
fig = px.scatter(
    df,
    x="az",
    y="el",
    color="target_plot",             # color by target within each sat
    facet_col="sat",            # one subplot per satellite
    facet_col_wrap=3,           # max 3 columns before wrapping
    hover_data=["time", "target"],
    title="Az-El Space per Satellite",
)
fig.update_xaxes(matches=None)
fig.update_yaxes(matches=None)
fig.show()

df_frozen = df[df['time'] == TIME_TO_FREEZE]

## Plot the cones

def earth_sphere():
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 80)
    x = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "steelblue"], [1, "steelblue"]],
        showscale=False, opacity=0.7,
        hoverinfo="skip", name="Earth",
    )


def cone_traces(sat_pos_km, for_deg, color, name, n=60, height_scale=2.0):
    p    = np.array(sat_pos_km, dtype=float)
    half = np.radians(for_deg / 2)

    nadir = -p / np.linalg.norm(p)
    ref   = np.array([0, 0, 1]) if abs(nadir[2]) < 0.9 else np.array([1, 0, 0])
    u     = np.cross(nadir, ref); u /= np.linalg.norm(u)
    v     = np.cross(nadir, u)

    height = (np.linalg.norm(p) - R_EARTH) * height_scale  # ← scaled
    radius = height * np.tan(half)

    phis = np.linspace(0, 2 * np.pi, n)
    base = p + nadir * height
    ring = np.array([base + radius * (np.cos(phi) * u + np.sin(phi) * v) for phi in phis])

    n_ring = len(ring)
    cone_mesh = go.Mesh3d(
        x=[p[0]] + ring[:, 0].tolist(),
        y=[p[1]] + ring[:, 1].tolist(),
        z=[p[2]] + ring[:, 2].tolist(),
        i=[0] * n_ring,
        j=list(range(1, n_ring + 1)),
        k=list(range(2, n_ring + 1)) + [1],
        color=color, opacity=0.25,
        name=name, showlegend=True,
    )

    return [cone_mesh]


def plot_fov(fig, df_frozen):
    df_plot        = df_frozen.copy()
    df_plot["sat"] = df_plot["sat"].astype(str)

    color_map = {}
    for i, (sat_name, group) in enumerate(df_plot.groupby("sat")):
        color = COLORS[i % len(COLORS)]
        color_map[sat_name] = color

        fig.add_trace(go.Scatter3d(
            x=group["sat_x"],
            y=group["sat_y"],
            z=group["sat_z"],
            mode="markers",
            marker=dict(size=5, color=color),
            name=sat_name,
        ))

    for _, row in df_frozen.iterrows():
        p_km  = np.array([row["sat_x"], row["sat_y"], row["sat_z"]])
        color = color_map.get(str(row["sat"]), "gray")
        for trace in cone_traces(p_km, row["FoR"], color, str(row["sat"])):
            fig.add_trace(trace)

    fig.add_trace(earth_sphere())

    fig.update_layout(
        title=f"Satellite Fields of Regard at {int(df_frozen['time'].iloc[0])} sec",
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X (km)"),
            yaxis=dict(title="Y (km)"),
            zaxis=dict(title="Z (km)"),
        ),
        legend_title="Satellite",
    )
    return fig

def plot_los(fig, df_frozen, ray_length=80000):
    """
    Plot lines of sight as dashed lines from each satellite in the az/el direction.

    Args:
        fig:        existing go.Figure
        df_frozen:  DataFrame with sat_x, sat_y, sat_z (km), az (az rad), el (el rad), sat
        ray_length: how long to draw the ray in km
    """
    for _, row in df_frozen.iterrows():
        sat_pos = np.array([row["sat_x"], row["sat_y"], row["sat_z"]])

        direction = az_el_to_los(row["az"], row["el"])
        end_point = sat_pos + direction * ray_length

        fig.add_trace(go.Scatter3d(
            x=[sat_pos[0], end_point[0]],
            y=[sat_pos[1], end_point[1]],
            z=[sat_pos[2], end_point[2]],
            mode="lines",
            line=dict(color='black', width=5, dash="dot"),
            showlegend=False,
            hovertemplate=(
                f"SAT {row['sat']}<br>"
                f"az: {np.degrees(row['az']):.2f}°<br>"
                f"el: {np.degrees(row['el']):.2f}°"
                "<extra></extra>"
            ),
        ))

    return fig

def plot_intersects(fig, intersection_pts, color="white", size=6):
    """
    Plot intersection points on an existing figure.

    Args:
        fig:              existing go.Figure
        intersection_pts: list of np.ndarray points, or empty list
        color:            marker color
        size:             marker size
    """
    if not intersection_pts:
        return fig

    pts = np.array(intersection_pts)

    fig.add_trace(go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker=dict(size=size, color=color, symbol="diamond"),
        name="Intersections",
        hovertemplate=(
            "x: %{x:.1f} km<br>"
            "y: %{y:.1f} km<br>"
            "z: %{z:.1f} km<br>"
            "<extra>Intersection</extra>"
        ),
    ))

    return fig


def az_el_to_los(az, el):
    """Inverse of measurement_model — az/el in radians to unit direction vector."""
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.array([x, y, z])

def los_to_az_el(los: np.ndarray) -> tuple[float, float]:
    """
    Project a 3D ECEF point into a satellite's az/el measurement frame.
    Inverse of az_el_to_direction.

    Args:
        los: direction vector in ECEF 

    Returns:
        (az, el) in radians
    """
    los_unit = los / np.linalg.norm(los)

    az = np.arctan2(los_unit[1], los_unit[0])        # atan2(y, x)
    el = np.arcsin(np.clip(los_unit[2], -1.0, 1.0))  # arcsin(z), clipped for safety

    return az, el




def ray_sphere_intersect_closest(ray_pos, ray_dir, radius):
    """
    Returns the closest positive intersection of a ray with a sphere at origin.
    Returns None if no intersection.
    """
    p = np.array(ray_pos, dtype=float)
    d = np.array(ray_dir, dtype=float); d /= np.linalg.norm(d)

    b    = 2 * np.dot(p, d)
    c    = np.dot(p, p) - radius ** 2
    disc = b ** 2 - 4 * c

    if disc < 0:
        return None

    t1 = (-b - np.sqrt(disc)) / 2
    t2 = (-b + np.sqrt(disc)) / 2

    # Take smallest positive t
    ts = [t for t in (t1, t2) if t > 0]
    if not ts:
        return None

    return p + min(ts) * d

def point_in_cone(pt, apex, axis, half_angle_deg):
    """Check if a point is inside the cone."""
    v    = np.array(pt) - np.array(apex)
    proj = np.dot(v, axis)
    if proj < 0:
        return False  # behind apex
    if np.linalg.norm(v) < 1e-10:
        return True   # at apex
    cos_angle = proj / np.linalg.norm(v)
    return cos_angle >= np.cos(np.radians(half_angle_deg))


def segment_cone_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    cone_apex: np.ndarray,
    cone_axis: np.ndarray,
    cone_half_angle_deg: float,
) -> list[np.ndarray]:
    """
    Find intersection of a line segment [p1, p2] with a cone.

    Rules:
    - If segment is fully inside cone  → return [p1, p2]
    - If segment partially intersects  → return [intersection, endpoint] or [p1/p2, intersection]
    - If segment passes through cone   → return [entry, exit] intersection points
    - If no intersection               → return []

    Args:
        p1, p2:              endpoints of the line segment
        cone_apex:           apex of the cone
        cone_axis:           axis direction (unit or not)
        cone_half_angle_deg: half-angle in degrees

    Returns:
        List of 0, 1, or 2 points
    """
    p1   = np.array(p1,        dtype=float)
    p2   = np.array(p2,        dtype=float)
    apex = np.array(cone_apex, dtype=float)
    a    = np.array(cone_axis, dtype=float)
    a /= np.linalg.norm(a)

    p1_in = point_in_cone(p1, apex, a, cone_half_angle_deg)
    p2_in = point_in_cone(p2, apex, a, cone_half_angle_deg)

    # Both inside — segment is entirely within cone
    if p1_in and p2_in:
        return [p1, p2]

    # Find where the infinite line intersects the cone
    d    = p2 - p1
    d_len = np.linalg.norm(d)
    if d_len < 1e-10:
        return []
    d_unit = d / d_len

    cos2 = np.cos(np.radians(cone_half_angle_deg)) ** 2
    w    = p1 - apex

    da = np.dot(d_unit, a)
    wa = np.dot(w, a)
    dw = np.dot(d_unit, w)
    ww = np.dot(w, w)

    A = da ** 2 - cos2
    B = 2 * (da * wa - cos2 * dw)
    C = wa ** 2 - cos2 * ww

    disc = B ** 2 - 4 * A * C

    if disc < 0:
        return []

    if abs(A) < 1e-10:
        if abs(B) < 1e-10:
            return []
        t_hit = -C / B / d_len  # convert from unit-dir t to [0,1] param
        ts = [t_hit]
    else:
        sqrt_disc = np.sqrt(disc)
        # t here is in units of d_unit, convert to [0,1] param
        t1 = (-B - sqrt_disc) / (2 * A) / d_len
        t2 = (-B + sqrt_disc) / (2 * A) / d_len
        ts = [t1, t2]

    # Keep only t in [0, 1] and on the correct nadir half of cone
    valid_pts = []
    for t in ts:
        if 0.0 <= t <= 1.0:
            pt = p1 + t * d
            if np.dot(pt - apex, a) >= 0:  # correct nadir side
                valid_pts.append(pt)

    if not valid_pts:
        # One end inside, one outside but no valid crossing found — return inside endpoint
        if p1_in:
            return [p1]
        if p2_in:
            return [p2]
        return []

    # One endpoint inside + one crossing → return inside endpoint + crossing
    if p1_in:
        return [p1] + valid_pts[:1]
    if p2_in:
        return valid_pts[:1] + [p2]

    # Both outside — return the crossing points (entry + exit)
    return valid_pts


# Go index 1 to index 2
idx1 = df_frozen.iloc[1]
idx2 = df_frozen.iloc[0]

print(f'We are projecting {idx1.sat}s measurement into {idx2.sat}s monotrack frame')

idx1_pos = np.array([idx1.sat_x, idx1.sat_y, idx1.sat_z])
ray1_dir = az_el_to_los(idx1.az, idx1.el)

# Get measurement to earth intersection point
meas1_intersect_w_earth = ray_sphere_intersect_closest(idx1_pos, ray1_dir, R_EARTH)

# Now get the intersection points between line cone intersection,
# line defined by idx1_pos to meas1_intersect
# cone defined by sat2 meas FoR

idx2_pos = np.array([idx2.sat_x, idx2.sat_y, idx2.sat_z])

intersect_pts = segment_cone_intersect(
    p1 = meas1_intersect_w_earth, # earth intersect
    p2 = idx1_pos, # sat pos
    cone_apex = idx2_pos, # other cone parameters
    cone_axis = -idx2_pos / np.linalg.norm(idx2_pos),
    cone_half_angle_deg = idx2.FoR / 2
)

# Show cone and LOS and intersects
fig = go.Figure()
fig = plot_fov(fig, df_frozen)
fig = plot_los(fig, df_frozen)
fig = plot_intersects(fig, intersect_pts)
fig.show()


#### Now, project intersect points into idx2's FOV.

projected = []
for pt in intersect_pts:
    # Get the az el
    az, el = los_to_az_el(pt - idx2_pos)
    # Now get the range so we can convert the R from idx1 to idx2 frame
    r_pt_1 = np.linalg.norm(pt - idx1_pos)
    r_pt_2 = np.linalg.norm(pt - idx2_pos)
    
    # So R from idx1 into ECEF cov.
    R_in_idx2_fov = r_pt_1 * idx1.R / r_pt_2
    
    projected.append({"time": TIME_TO_FREEZE, "az_proj": az, "el_proj": el, "R_proj": R_in_idx2_fov, "label": "Projected"})

df_projected = pd.DataFrame(projected)


# --- Build the 2D az/el plot ---
fig = px.scatter(
    df[df['sat'] == idx2.sat],
    x="az",
    y="el",
    color="sat",
    hover_data=df.columns,  # Show all columns on hover
)

proj_scatter = px.scatter(
    df_projected,
    x="az_proj",
    y="el_proj",
    color_discrete_sequence=["red"],
    hover_data=df_projected.columns,  # Show all columns on hover
)
for trace in proj_scatter.data:
    fig.add_trace(trace)
    
# --- Draw 1-sigma circles for each projected point ---
theta = np.linspace(0, 2 * np.pi, 100)
for _, row in df_projected.iterrows():
    r = row["R_proj"]
    circle_az = row["az_proj"] + r * np.cos(theta)
    circle_el = row["el_proj"] + r * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=circle_az,
        y=circle_el,
        mode="lines",
        line=dict(color="red", width=1.5, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))

fig.add_trace(go.Scatter(
    x=df_projected["az_proj"],
    y=df_projected["el_proj"],
    mode="lines",
    line=dict(color="red", width=1.5, dash="dash"),
    showlegend=False,
    hoverinfo="skip",
))

# Now that we ahve the line from df_projected we can get closest approach point from df_frozen to line.

real_meas_at_time = df_frozen[df_frozen['sat'] == idx2.sat][["az", "el"]].iloc[0].values

# Line defined by projected point 1 and 2 in az/el space
p1 = df_projected[["az_proj", "el_proj"]].iloc[0].values
p2 = df_projected[["az_proj", "el_proj"]].iloc[1].values
line_vec = p2 - p1
line_len = np.linalg.norm(line_vec)
line_unit = line_vec / line_len


# Project pt onto the line, clamped to [0, 1]
along_line = np.clip(np.dot(real_meas_at_time - p1, line_unit) / line_len, 0, 1)
closest_pt_on_line = p1 + along_line * line_len * line_unit
dist = np.linalg.norm(real_meas_at_time - closest_pt_on_line)

fig.add_trace(go.Scatter(
        x=[closest_pt_on_line[0]],
        y=[closest_pt_on_line[1]],
        mode="markers",
        marker=dict(color="red", size=10, symbol="square"),
        showlegend=False,
        hovertemplate=(
            "<b>Closest Point on Line</b><br>"
            "az: %{x}<br>"
            "el: %{y}<br>"
            "TIME_TO_FREEZE: " + str(TIME_TO_FREEZE)
        ),
    ))


# Now interpoalte from R_p1 to R_p2 to get the R to use on the closest approach point.
R_p1 = df_projected[["R_proj"]].iloc[0].values
R_p2 = df_projected[["R_proj"]].iloc[1].values
R_closest = R_p1 + (R_p2 - R_p1) * along_line

fig.show()


# Now finally take the mahal between the the real measurement at TIME_TO_FREEZE and the projected line
mahal = np.sqrt(dist * dist / (R_closest * R_closest + idx2.R * idx2.R))
print("Mahalanobis distance:", mahal)


