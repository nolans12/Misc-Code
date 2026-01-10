import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import math


TRUE_JITTER = 1.0  # m
EST_JITTER = 5.0   # m

# Initial velocities
X_VEL_0 = -15.0   # m/s
Y_VEL_0 = 10.0    # m/s

# Missile dynamics
BOOST_ACCEL = 70.0      # m/s² - strong upward acceleration during boost phase
BALLISTIC_ACCEL = -9.8  # m/s² - gravity during ballistic phase (1g downward)
BOOST_DURATION = 10.0    # seconds - how long the boost phase lasts

TRUE_CA_SIGMA = 2.0    # m/s² - process noise during boost
EST_CA_SIGMA = 8.0     # m/s² - our estimate (higher uncertainty)

SHOW_COVARIANCES = True

# Sim
dt = 0.5  # Time step (s) - smaller for better resolution of transition
duration = 20 # Simulation duration (s)
x0 = -100.0  # Initial x position (m)
y0 = 200.0   # Initial y position (m)

class ConstantAccelerationModel:
    """Constant acceleration motion model for 2D tracking"""
    
    def __init__(self, dt: float, ca_sigma: float):
        self.dt = dt
        self.ca_sigma = ca_sigma
        
        # State transition matrix for constant acceleration model
        # State: [x, y, x_dot, y_dot, x_acc, y_acc]
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Process noise matrix for constant acceleration model
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        
        self.Q = np.zeros((6, 6))
        
        # X components (indices 0, 2, 4)
        self.Q[0, 0] = dt4 / 4 * ca_sigma**2
        self.Q[0, 2] = dt3 / 2 * ca_sigma**2
        self.Q[2, 0] = dt3 / 2 * ca_sigma**2
        self.Q[0, 4] = dt2 / 2 * ca_sigma**2
        self.Q[4, 0] = dt2 / 2 * ca_sigma**2
        self.Q[2, 2] = dt2 * ca_sigma**2
        self.Q[2, 4] = dt * ca_sigma**2
        self.Q[4, 2] = dt * ca_sigma**2
        self.Q[4, 4] = 1 * ca_sigma**2
        
        # Y components (indices 1, 3, 5)
        self.Q[1, 1] = dt4 / 4 * ca_sigma**2
        self.Q[1, 3] = dt3 / 2 * ca_sigma**2
        self.Q[3, 1] = dt3 / 2 * ca_sigma**2
        self.Q[1, 5] = dt2 / 2 * ca_sigma**2
        self.Q[5, 1] = dt2 / 2 * ca_sigma**2
        self.Q[3, 3] = dt2 * ca_sigma**2
        self.Q[3, 5] = dt * ca_sigma**2
        self.Q[5, 3] = dt * ca_sigma**2
        self.Q[5, 5] = 1 * ca_sigma**2
        
    def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state and covariance"""
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

class MeasurementModel:
    """Measurement model for x and y position"""
    
    def __init__(self, measurement_noise_std: float = 1.0):
        # Measurement matrix - we only measure x and y
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # x measurement
            [0, 1, 0, 0, 0, 0]   # y measurement
        ])
        
        # Measurement noise covariance
        self.R = np.diag([measurement_noise_std**2, measurement_noise_std**2])
        
    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, 
               z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state and covariance with measurement"""
        # Innovation
        y = z - self.H @ x_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Update
        x_update = x_pred + K @ y
        P_update = (np.eye(6) - K @ self.H) @ P_pred
        
        return x_update, P_update

class KalmanFilter:
    """Forward Kalman filter"""
    
    def __init__(self, dt: float, ca_sigma: float, measurement_noise_std: float = 1.0):
        self.motion_model = ConstantAccelerationModel(dt, ca_sigma)
        self.measurement_model = MeasurementModel(measurement_noise_std)
        
    def filter(self, measurements: np.ndarray, x0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run forward Kalman filter"""
        n_steps = len(measurements)
        n_states = 6
        
        # Storage for filtered states and covariances
        x_filtered = np.zeros((n_steps, n_states))
        P_filtered = np.zeros((n_steps, n_states, n_states))
        
        # Initialize
        x = x0.copy()
        P = P0.copy()
        
        for i in range(n_steps):
            # Predict
            x_pred, P_pred = self.motion_model.predict(x, P)
            
            # Update
            x, P = self.measurement_model.update(x_pred, P_pred, measurements[i])
            
            # Store results
            x_filtered[i] = x
            P_filtered[i] = P
            
        return x_filtered, P_filtered

class RTSSmoother:
    """Rauch-Tung-Striebel smoother"""
    
    def __init__(self, dt: float, ca_sigma: float):
        self.motion_model = ConstantAccelerationModel(dt, ca_sigma)
        
    def smooth(self, x_filtered: np.ndarray, P_filtered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run RTS smoother"""
        n_steps = len(x_filtered)
        n_states = 6
        
        # Storage for smoothed states and covariances
        x_smoothed = np.zeros((n_steps, n_states))
        P_smoothed = np.zeros((n_steps, n_states, n_states))
        
        # Initialize with last filtered state
        x_smoothed[-1] = x_filtered[-1]
        P_smoothed[-1] = P_filtered[-1]
        
        # Backward pass
        for i in range(n_steps-2, -1, -1):
            # Smoothing gain
            F = self.motion_model.F
            Q = self.motion_model.Q
            
            # Predict next state and covariance
            x_pred = F @ x_filtered[i]
            P_pred = F @ P_filtered[i] @ F.T + Q
            
            # Smoothing gain
            C = P_filtered[i] @ F.T @ np.linalg.inv(P_pred)
            
            # Smooth
            x_smoothed[i] = x_filtered[i] + C @ (x_smoothed[i+1] - x_pred)
            P_smoothed[i] = P_filtered[i] + C @ (P_smoothed[i+1] - P_pred) @ C.T
            
        return x_smoothed, P_smoothed

def generate_missile_trajectory(dt: float, duration: float, 
                               x0: float, y0: float, 
                               x_dot0: float = X_VEL_0, y_dot0: float = Y_VEL_0,
                               noise_std: float = TRUE_JITTER) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a boosting missile trajectory with boost phase followed by ballistic phase"""
    
    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)
    
    # Initialize arrays
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    x_dot = np.zeros(n_steps)
    y_dot = np.zeros(n_steps)
    x_acc = np.zeros(n_steps)
    y_acc = np.zeros(n_steps)
    
    # Initial conditions
    x[0] = x0
    y[0] = y0
    x_dot[0] = x_dot0
    y_dot[0] = y_dot0
    
    # Determine boost transition step
    boost_transition_step = int(BOOST_DURATION / dt)
    
    # Generate trajectory with boost and ballistic phases
    for i in range(n_steps):
        # Determine acceleration based on phase
        if t[i] < BOOST_DURATION:
            # Boost phase - strong upward acceleration in y direction
            base_x_acc = 0.0  # No acceleration in x during boost
            base_y_acc = BOOST_ACCEL  # Strong upward acceleration
            process_noise = TRUE_CA_SIGMA
        else:
            # Ballistic phase - gravity only
            base_x_acc = 0.0
            base_y_acc = BALLISTIC_ACCEL  # Gravity pulls down
            process_noise = TRUE_CA_SIGMA * 0.5  # Less process noise during ballistic
        
        # Add process noise to acceleration
        x_acc[i] = base_x_acc + np.random.normal(0, process_noise)
        y_acc[i] = base_y_acc + np.random.normal(0, process_noise)
        
        # Integrate to get velocity and position
        if i > 0:
            # Update velocity (Euler integration)
            x_dot[i] = x_dot[i-1] + x_acc[i-1] * dt
            y_dot[i] = y_dot[i-1] + y_acc[i-1] * dt
            
            # Update position
            x[i] = x[i-1] + x_dot[i-1] * dt + 0.5 * x_acc[i-1] * dt**2
            y[i] = y[i-1] + y_dot[i-1] * dt + 0.5 * y_acc[i-1] * dt**2
    
    # True state matrix
    x_true = np.column_stack([x, y, x_dot, y_dot, x_acc, y_acc])
    
    # Add noise to measurements
    measurements = np.column_stack([
        x + np.random.normal(0, noise_std, n_steps),
        y + np.random.normal(0, noise_std, n_steps)
    ])
    
    return x_true, measurements, t

def plot_missile_results(t: np.ndarray, x_true: np.ndarray, x_filtered: np.ndarray, 
                        x_smoothed: np.ndarray, measurements: np.ndarray,
                        P_filtered: np.ndarray, P_smoothed: np.ndarray):
    """Plot the missile tracking results with emphasis on the boost-to-ballistic transition"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('RTS Smoother for Boosting Missile Tracking\n(Boost Phase → Ballistic Phase Transition)', fontsize=16)
    
    # State names for titles
    state_names = ['X Position', 'Y Position', 'X Velocity', 'Y Velocity', 'X Acceleration', 'Y Acceleration']
    state_units = ['(m)', '(m)', '(m/s)', '(m/s)', '(m/s²)', '(m/s²)']
    
    # Plot each state component
    for i in range(6):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Scale data (no scaling needed for meters)
        scale_factor = 1.0  # Already in meters
        
        # Add vertical line to show boost-to-ballistic transition
        ax.axvline(x=BOOST_DURATION, color='orange', linestyle='--', alpha=0.7, 
                  linewidth=2, label='Boost → Ballistic')
        
        # Plot true, filtered, and smoothed states
        ax.plot(t, x_true[:, i] * scale_factor, 'b-', label='True', linewidth=2)
        ax.plot(t, x_filtered[:, i] * scale_factor, 'r--', label='Filtered (KF)', linewidth=1.5)
        ax.plot(t, x_smoothed[:, i] * scale_factor, 'g:', label='Smoothed (RTS)', linewidth=2)
        
        # Add measurements for position states (x and y)
        if i < 2:
            ax.scatter(t, measurements[:, i] * scale_factor, c='k', alpha=0.5, s=8, label='Measurements')
        
        # Add covariance shading for filtered (KF) estimates
        if SHOW_COVARIANCES:
            std_filtered = np.sqrt(P_filtered[:, i, i])
            ax.fill_between(t, 
                           (x_filtered[:, i] - 2*std_filtered) * scale_factor, 
                           (x_filtered[:, i] + 2*std_filtered) * scale_factor, 
                           alpha=0.2, color='red', label='Filtered ±2σ')
            
            # Add covariance shading for smoothed estimates
            std_smoothed = np.sqrt(P_smoothed[:, i, i])
            ax.fill_between(t, 
                           (x_smoothed[:, i] - 2*std_smoothed) * scale_factor, 
                           (x_smoothed[:, i] + 2*std_smoothed) * scale_factor, 
                           alpha=0.2, color='green', label='Smoothed ±2σ')
        
        # Set labels and title
        ax.set_ylabel(f'{state_names[i]} {state_units[i]}')
        ax.set_title(f'{state_names[i]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add phase annotations for y acceleration plot (most interesting)
        if i == 5:  # Y acceleration
            ax.text(BOOST_DURATION/2, ax.get_ylim()[1]*0.8, f'BOOST\n+{BOOST_ACCEL} m/s²', 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            ax.text(BOOST_DURATION + (duration-BOOST_DURATION)/2, ax.get_ylim()[0]*0.8, 
                   f'BALLISTIC\n{BALLISTIC_ACCEL} m/s²', ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
        
        # Add x-label for bottom row
        if row == 2:
            ax.set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

def plot_tracking_errors(t: np.ndarray, x_true: np.ndarray, x_filtered: np.ndarray, 
                        x_smoothed: np.ndarray):
    """Plot tracking errors to highlight KF lag vs RTS performance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tracking Errors: Kalman Filter Lag vs RTS Smoother Performance', fontsize=14)
    
    # Calculate errors
    error_filtered = (x_filtered - x_true)  # Already in meters
    error_smoothed = (x_smoothed - x_true)
    
    # Plot y position and velocity errors (most interesting for missile)
    states_to_plot = [1, 3]  # Y position and velocity
    state_names = ['Y Position', 'Y Velocity'] 
    state_units = ['(m)', '(m/s)']
    
    for idx, i in enumerate(states_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Add vertical line to show boost-to-ballistic transition
        ax.axvline(x=BOOST_DURATION, color='orange', linestyle='--', alpha=0.7, 
                  linewidth=2, label='Boost → Ballistic')
        
        # Plot errors
        ax.plot(t, error_filtered[:, i], 'r-', label='Filtered Error', linewidth=2)
        ax.plot(t, error_smoothed[:, i], 'g-', label='Smoothed Error', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        ax.set_ylabel(f'Error {state_units[idx]}')
        ax.set_title(f'{state_names[idx]} Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)')
    
    # Plot acceleration errors for both x and y
    for idx, i in enumerate([4, 5]):  # X and Y acceleration
        row = (idx + 2) // 2
        col = (idx + 2) % 2
        ax = axes[1, idx % 2]
        
        # Add vertical line to show boost-to-ballistic transition
        ax.axvline(x=BOOST_DURATION, color='orange', linestyle='--', alpha=0.7, 
                  linewidth=2, label='Boost → Ballistic')
        
        # Plot errors
        ax.plot(t, error_filtered[:, i], 'r-', label='Filtered Error', linewidth=2)
        ax.plot(t, error_smoothed[:, i], 'g-', label='Smoothed Error', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        accel_name = 'X' if i == 4 else 'Y'
        ax.set_ylabel('Error (m/s²)')
        ax.set_title(f'{accel_name} Acceleration Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

def calculate_rmse(x_true: np.ndarray, x_estimated: np.ndarray) -> np.ndarray:
    """Calculate RMSE for each state component"""
    return np.sqrt(np.mean((x_true - x_estimated)**2, axis=0))

def calculate_phase_rmse(x_true: np.ndarray, x_estimated: np.ndarray, 
                        t: np.ndarray, phase_duration: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate RMSE for boost and ballistic phases separately"""
    boost_mask = t < phase_duration
    ballistic_mask = t >= phase_duration
    
    rmse_boost = np.sqrt(np.mean((x_true[boost_mask] - x_estimated[boost_mask])**2, axis=0))
    rmse_ballistic = np.sqrt(np.mean((x_true[ballistic_mask] - x_estimated[ballistic_mask])**2, axis=0))
    
    return rmse_boost, rmse_ballistic

def main():
    """Main function to run the missile tracking simulation"""
    
    # Generate missile trajectory
    print("Generating boosting missile trajectory...")
    print(f"Boost phase: 0 to {BOOST_DURATION}s (+{BOOST_ACCEL:.1f} m/s² in Y)")
    print(f"Ballistic phase: {BOOST_DURATION}s to {duration}s ({BALLISTIC_ACCEL:.1f} m/s² in Y)")
    
    x_true, measurements, t = generate_missile_trajectory(
        dt, duration, x0, y0
    )
    
    # Initialize with velocity estimate from first few measurements
    x_vel_0 = (measurements[1][0] - measurements[0][0]) / dt
    y_vel_0 = (measurements[1][1] - measurements[0][1]) / dt
    
    # Initial conditions - start with reasonable uncertainty
    x0_init = np.array([x0, y0, x_vel_0, y_vel_0, 0, 0])
    P0 = np.diag([
        EST_JITTER**2, 
        EST_JITTER**2, 
        (2*EST_JITTER)**2, 
        (2*EST_JITTER)**2, 
        EST_CA_SIGMA**2, 
        EST_CA_SIGMA**2
    ]) 
    
    # Run forward Kalman filter
    print("Running forward Kalman filter...")
    kf = KalmanFilter(dt, EST_CA_SIGMA, EST_JITTER)
    x_filtered, P_filtered = kf.filter(measurements, x0_init, P0)
    
    # Run RTS smoother
    print("Running RTS smoother...")
    smoother = RTSSmoother(dt, EST_CA_SIGMA)
    x_smoothed, P_smoothed = smoother.smooth(x_filtered, P_filtered)
    
    # Calculate overall RMSE
    rmse_filtered = calculate_rmse(x_true, x_filtered)
    rmse_smoothed = calculate_rmse(x_true, x_smoothed)
    
    # Calculate phase-specific RMSE
    rmse_boost_filt, rmse_ballistic_filt = calculate_phase_rmse(x_true, x_filtered, t, BOOST_DURATION)
    rmse_boost_smooth, rmse_ballistic_smooth = calculate_phase_rmse(x_true, x_smoothed, t, BOOST_DURATION)
    
    print("\n" + "="*60)
    print("OVERALL RMSE RESULTS:")
    print("="*60)
    state_names = ['X Pos', 'Y Pos', 'X Vel', 'Y Vel', 'X Acc', 'Y Acc']
    for i, name in enumerate(state_names):
        print(f"{name:8}: Filtered: {rmse_filtered[i]:6.3f}, Smoothed: {rmse_smoothed[i]:6.3f}")
    
    print("\n" + "="*60)
    print("BOOST PHASE RMSE (0-8s):")
    print("="*60)
    for i, name in enumerate(state_names):
        print(f"{name:8}: Filtered: {rmse_boost_filt[i]:6.3f}, Smoothed: {rmse_boost_smooth[i]:6.3f}")
    
    print("\n" + "="*60)
    print("BALLISTIC PHASE RMSE (8-20s):")
    print("="*60)
    for i, name in enumerate(state_names):
        print(f"{name:8}: Filtered: {rmse_ballistic_filt[i]:6.3f}, Smoothed: {rmse_ballistic_smooth[i]:6.3f}")
    
    # Plot results
    print(f"\nPlotting results...")
    plot_missile_results(t, x_true, x_filtered, x_smoothed, measurements, P_filtered, P_smoothed)
    plot_tracking_errors(t, x_true, x_filtered, x_smoothed)
    

if __name__ == "__main__":
    main()