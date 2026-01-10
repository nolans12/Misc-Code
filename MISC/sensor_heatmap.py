import pandas as pd
import numpy as np
import plotly.express as px

# --- Step 1: Create dataframe ---
np.random.seed(42)

n_sensors = 10   # number of sensors
n_points = 1_000_000  # measurements per sensor

data = {
    "Sensor": np.repeat([f"Sensor_{i}" for i in range(n_sensors)], n_points),
    "Row": np.random.uniform(0, 600, n_sensors * n_points),
    "Col": np.random.uniform(0, 600, n_sensors * n_points),
}

df = pd.DataFrame(data)

for sensor in df["Sensor"].unique():
    sub_df = df[df["Sensor"] == sensor].copy()
    
    # Use fractional pixel positions
    sub_df["Row_frac"] = sub_df["Row"] % 1
    sub_df["Col_frac"] = sub_df["Col"] % 1
    
    fig = px.density_heatmap(
        sub_df,
        x="Row_frac",
        y="Col_frac",
        nbinsx=20,
        nbinsy=20,
        range_x=[0, 1],   # Force x-axis 0–1
        range_y=[0, 1],   # Force y-axis 0–1
        title=f"Pixel Heatmap for {sensor}",
        color_continuous_scale="Viridis"
    )
    
    # Make axes square
    fig.update_yaxes(scaleanchor="x", scaleratio=1, tickformat=".2f", title="Fractional Col Position")
    fig.update_xaxes(tickformat=".2f", title="Fractional Row Position")
    
    # Optional: set gridlines for readability
    fig.update_layout(
        xaxis=dict(tick0=0, dtick=0.05),
        yaxis=dict(tick0=0, dtick=0.05),
        font=dict(size=12)
    )
    
    fig.write_html(f"heatmap_{sensor}.html")
    fig.show()



df.head()
