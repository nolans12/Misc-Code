# GEO Tracker Demo

A simplified simulation environment for tracking a single ballistic missile using 3 GEO satellites with Extended Kalman Filter (EKF) estimation.

## Overview

This demo is a streamlined version of the full Satellite-DDF system, focusing on:
- **Single ballistic target** with simple constant-velocity dynamics + gravity
- **3 GEO satellites** positioned 36,000 km above Earth's equator
- **Instant communication** between satellites (no network delays)
- **Perfect data association** (no false alarms or missed detections)
- **Centralized EKF** fusing bearings-only measurements

## Files

- `sensing.py` - Satellite and target models, measurement generation
- `tracking.py` - EKF implementation for state estimation
- `main.py` - Simulation entry point with live visualization
- `plot.py` - Enhanced plotting utilities
- `data/` - Simulation results (auto-generated)

## Installation

### Minimal Installation (without map features)
```bash
pip install numpy matplotlib
```

### Full Installation (with cartopy for ground track maps)
```bash
pip install numpy matplotlib cartopy
```

**Note**: Cartopy provides realistic world maps with coastlines and borders for the ground track. Without it, the ground track will display as a simple lat/lon plot. On Windows, you may need to install GDAL and other dependencies first. See [Cartopy installation guide](https://scitools.org.uk/cartopy/docs/latest/installing.html) for details.

Alternatively, install all requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Run Simulation with Live Plotting

```bash
python main_sim_meas.py
```

This will:
1. Simulate a 600-second ballistic trajectory
2. Display **live updates** showing:
   - Ground track on world map (with coastlines)
   - Position and velocity errors
   - Altitude profile
   - XY overhead view
   - Covariance evolution
3. Save results to `data/simulation_run.npz`

### View Saved Results

```bash
python plot_sim.py
```

Generates comprehensive static plots from the most recent simulation, including:
- Ground track with start/end markers
- Error norms (position & velocity)
- Altitude comparison
- XY overhead projection
- Uncertainty bounds (1σ)

## Parameters

Edit `main.py` to customize:
- `SIM_DURATION_S` - simulation length (default: 600s)
- `DT_S` - time step (default: 5s)
- `seed` - random seed for reproducibility
- `live_plot` - enable/disable live visualization

Edit `sensing.py` to adjust:
- Target initial position/velocity
- Satellite noise levels (default: 0.05°)
- Number of GEO satellites (default: 3)

Edit `tracking.py` to modify:
- Process noise `process_sigma` (default: 0.02)
- Initial covariance

## Example Output

The live plotter shows real-time tracking performance:
- **Blue** = ground truth
- **Red** = EKF estimate
- **Green triangles** = GEO satellite positions

The ground track uses Cartopy's PlateCarree projection with coastlines and borders for geographic context.

## Differences from Full Satellite-DDF

This demo **removes**:
- Complex configuration system
- Database storage
- Distributed estimation & planning
- Network delays and routing
- Track management (initiation, deletion, fusion)
- Multiple targets and data association
- High-fidelity orbit propagation

This demo **keeps**:
- EKF prediction/update logic
- Bearings-only measurement model
- Covariance propagation
- Core state estimation algorithms

## Extending the Demo

To add features:
1. **Multiple targets**: Extend `BallisticTarget` list in `main.py`
2. **Data association**: Implement gating in `tracking.py`
3. **Network delays**: Add message queues in `main.py`
4. **Orbit propagation**: Replace fixed GEO positions with time-varying orbits

See the full `Satellite-DDF` codebase for reference implementations.

