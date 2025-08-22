import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import math


TRUE_JITTER = 1e-3  # mrad
EST_JITTER = 5e-3   # mrad

AZ_VEL_0 = -15e-3   # mrad/s
EL_VEL_0 = 10e-3    # mrad/s

TRUE_ACCEL = 5e-3   # mrad/s²
TRUE_CA_SIGMA = 1e-3  # mrad/s²
EST_CA_SIGMA = 5e-3     # mrad/s² 

SHOW_COVARIANCES = True

# Sim
dt = 5.0  # Time step (s)
duration = 20 # Simulation duration (s)
az0 = -1.62  # Initial azimuth (rad)
el0 = 2.67  # Initial elevation (rad)

class ConstantAccelerationModel:
    """Constant acceleration motion model for 2D tracking"""
    
    def __init__(self, dt: float, ca_sigma: float):
        self.dt = dt
        self.ca_sigma = ca_sigma
        
        # State transition matrix for constant acceleration model
        # State: [az, el, az_dot, el_dot, az_acc, el_acc]
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Fixed: Proper process noise matrix for constant acceleration model
        # State order: [az, el, az_dot, el_dot, az_acc, el_acc]
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        
        # Build Q matrix with proper indexing for [az, el, az_dot, el_dot, az_acc, el_acc]
        self.Q = np.zeros((6, 6))
        
        # Azimuth components (indices 0, 2, 4)
        # Position-position
        self.Q[0, 0] = dt4 / 4 * ca_sigma**2
        # Position-velocity
        self.Q[0, 2] = dt3 / 2 * ca_sigma**2
        self.Q[2, 0] = dt3 / 2 * ca_sigma**2
        # Position-acceleration
        self.Q[0, 4] = dt2 / 2 * ca_sigma**2
        self.Q[4, 0] = dt2 / 2 * ca_sigma**2
        # Velocity-velocity
        self.Q[2, 2] = dt2 * ca_sigma**2
        # Velocity-acceleration
        self.Q[2, 4] = dt * ca_sigma**2
        self.Q[4, 2] = dt * ca_sigma**2
        # Acceleration-acceleration
        self.Q[4, 4] = 1 * ca_sigma**2
        
        # Elevation components (indices 1, 3, 5)
        # Position-position
        self.Q[1, 1] = dt4 / 4 * ca_sigma**2
        # Position-velocity
        self.Q[1, 3] = dt3 / 2 * ca_sigma**2
        self.Q[3, 1] = dt3 / 2 * ca_sigma**2
        # Position-acceleration
        self.Q[1, 5] = dt2 / 2 * ca_sigma**2
        self.Q[5, 1] = dt2 / 2 * ca_sigma**2
        # Velocity-velocity
        self.Q[3, 3] = dt2 * ca_sigma**2
        # Velocity-acceleration
        self.Q[3, 5] = dt * ca_sigma**2
        self.Q[5, 3] = dt * ca_sigma**2
        # Acceleration-acceleration
        self.Q[5, 5] = 1 * ca_sigma**2
        
    def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state and covariance"""
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

class MeasurementModel:
    """Measurement model for azimuth and elevation"""
    
    def __init__(self, measurement_noise_std: float = 1e-6):  # Fixed: parameterized noise
        # Measurement matrix - we only measure az and el
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # az measurement
            [0, 1, 0, 0, 0, 0]   # el measurement
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
    
    def __init__(self, dt: float, ca_sigma: float, measurement_noise_std: float = 1e-6):
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

def generate_trajectory(dt: float, duration: float, 
                       az0: float, el0: float, 
                       az_dot0: float = AZ_VEL_0, el_dot0: float = EL_VEL_0,
                       az_acc_mean: float = TRUE_ACCEL, el_acc_mean: float = TRUE_ACCEL,
                       noise_std: float = TRUE_JITTER) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a trajectory following constant acceleration model with noise"""
    
    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)
    
    # Fixed: Generate true trajectory with proper physics
    # Initialize arrays
    az = np.zeros(n_steps)
    el = np.zeros(n_steps)
    az_dot = np.zeros(n_steps)
    el_dot = np.zeros(n_steps)
    az_acc = np.zeros(n_steps)
    el_acc = np.zeros(n_steps)
    
    # Initial conditions
    az[0] = az0
    el[0] = el0
    az_dot[0] = az_dot0
    el_dot[0] = el_dot0
    az_acc[0] = az_acc_mean + np.random.normal(0, TRUE_CA_SIGMA)
    el_acc[0] = el_acc_mean + np.random.normal(0, TRUE_CA_SIGMA)
    
    # Integrate forward with proper physics
    for i in range(1, n_steps):
        # Add process noise to acceleration
        az_acc[i] = az_acc_mean + np.random.normal(0, TRUE_CA_SIGMA)
        el_acc[i] = el_acc_mean + np.random.normal(0, TRUE_CA_SIGMA)
        
        # Update velocity (Euler integration)
        az_dot[i] = az_dot[i-1] + az_acc[i-1] * dt
        el_dot[i] = el_dot[i-1] + el_acc[i-1] * dt
        
        # Update position
        az[i] = az[i-1] + az_dot[i-1] * dt + 0.5 * az_acc[i-1] * dt**2
        el[i] = el[i-1] + el_dot[i-1] * dt + 0.5 * el_acc[i-1] * dt**2
    
    # True state matrix
    x_true = np.column_stack([az, el, az_dot, el_dot, az_acc, el_acc])
    
    # Add noise to measurements
    measurements = np.column_stack([
        az + np.random.normal(0, noise_std, n_steps),
        el + np.random.normal(0, noise_std, n_steps)
    ])
    
    return x_true, measurements, t

def plot_results(t: np.ndarray, x_true: np.ndarray, x_filtered: np.ndarray, 
                x_smoothed: np.ndarray, measurements: np.ndarray,
                P_filtered: np.ndarray, P_smoothed: np.ndarray):
    """Plot the results with covariance visualization"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('RTS Smoother Results for Constant Acceleration Model', fontsize=16)
    
    # State names for titles
    state_names = ['Azimuth', 'Elevation', 'Azimuth Rate', 'Elevation Rate', 'Azimuth Acceleration', 'Elevation Acceleration']
    state_units = ['(mrad)', '(mrad)', '(mrad/s)', '(mrad/s)', '(mrad/s²)', '(mrad/s²)']
    
    # Plot each state component
    for i in range(6):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Scale data for all plots to millirad
        scale_factor = 1e3  # Convert from rad to mrad for all states
        
        # Plot true, filtered, and smoothed states
        ax.plot(t, x_true[:, i] * scale_factor, 'b-', label='True', linewidth=2)
        ax.plot(t, x_filtered[:, i] * scale_factor, 'r--', label='Filtered (KF)', linewidth=1.5)
        ax.plot(t, x_smoothed[:, i] * scale_factor, 'g:', label='Smoothed (RTS)', linewidth=2)
        
        # Add measurements for position states (az and el)
        if i < 2:
            ax.scatter(t, measurements[:, i] * scale_factor, c='k', alpha=0.5, s=10, label='Measurements')
        
        # Add covariance shading for filtered (KF) estimates
        if SHOW_COVARIANCES:
            std_filtered = np.sqrt(P_filtered[:, i, i])
            ax.fill_between(t, 
                           (x_filtered[:, i] - 2*std_filtered) * scale_factor, 
                           (x_filtered[:, i] + 2*std_filtered) * scale_factor, 
                           alpha=0.3, color='red', label='Filtered ±2σ')
            
            # Add covariance shading for smoothed estimates
            std_smoothed = np.sqrt(P_smoothed[:, i, i])
            ax.fill_between(t, 
                           (x_smoothed[:, i] - 2*std_smoothed) * scale_factor, 
                           (x_smoothed[:, i] + 2*std_smoothed) * scale_factor, 
                           alpha=0.3, color='green', label='Smoothed ±2σ')
        
        # Set labels and title
        ax.set_ylabel(f'{state_names[i]} {state_units[i]}')
        ax.set_title(state_names[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add x-label for bottom row
        if row == 2:
            ax.set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

def calculate_rmse(x_true: np.ndarray, x_estimated: np.ndarray) -> np.ndarray:
    """Calculate RMSE for each state component"""
    return np.sqrt(np.mean((x_true - x_estimated)**2, axis=0))

def main():
    """Main function to run the RTS smoother"""
    
    # Generate trajectory
    print("Generating trajectory...")
    x_true, measurements, t = generate_trajectory(
        dt, duration, az0, el0
    )
    
    # Get init diff over dt for vel
    az_vel_0 = (measurements[1][0] - measurements[0][0]) / dt
    el_vel_0 = (measurements[1][1] - measurements[0][1]) / dt
    
    # Init conditions
    # x0 = np.array([az0, el0, 0, 0, 0, 0])
    x0 = np.array([az0, el0, az_vel_0, el_vel_0, 0, 0])
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
    kf = KalmanFilter(dt, EST_CA_SIGMA, EST_JITTER)  # Fixed: pass measurement noise
    x_filtered, P_filtered = kf.filter(measurements, x0, P0)
    
    # Run RTS smoother
    print("Running RTS smoother...")
    smoother = RTSSmoother(dt, EST_CA_SIGMA)
    x_smoothed, P_smoothed = smoother.smooth(x_filtered, P_filtered)
    
    # Calculate RMSE
    rmse_filtered = calculate_rmse(x_true, x_filtered)
    rmse_smoothed = calculate_rmse(x_true, x_smoothed)
    
    print("\nRMSE Results:")
    state_names = ['Azimuth', 'Elevation', 'Az Rate', 'El Rate', 'Az Acc', 'El Acc']
    for i, name in enumerate(state_names):
        scale = 1e3  # Convert all to millirad units
        print(f"{name:12}: Filtered: {rmse_filtered[i] * scale:.3f}, Smoothed: {rmse_smoothed[i] * scale:.3f}")
    
    # Plot results
    print("\nPlotting results...")
    plot_results(t, x_true, x_filtered, x_smoothed, measurements, P_filtered, P_smoothed)
    
    # Print final state estimates (in millirad)
    print(f"\nFinal State Estimates (mrad):")
    print(f"True:      [{', '.join([f'{x*1e3:.3f}' for x in x_true[-1]])}]")
    print(f"Filtered:  [{', '.join([f'{x*1e3:.3f}' for x in x_filtered[-1]])}]")
    print(f"Smoothed:  [{', '.join([f'{x*1e3:.3f}' for x in x_smoothed[-1]])}]")

if __name__ == "__main__":
    main()