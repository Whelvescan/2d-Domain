import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & GEOMETRY
# ==========================================
L = 0.5          # Length in z-direction (m)
W = 0.2          # Width in y-direction (m)
k = 50           # Thermal conductivity (W/mK)

# Material Properties (Assuming Structural Carbon Steel)
rho = 7850       # Density (kg/m^3)
Cp = 460         # Specific heat capacity (J/kg*K)
alpha = k / (rho * Cp)  # Thermal diffusivity (m^2/s)

# Boundary Values
T0 = 80          # Constant temp at z=0 (°C)
h = 15           # Natural Convection coeff at z=L (W/m^2K)
T_inf = 27       # Ambient Air temp at z=L (°C)
q_in = 500      # Constant heat flux entering at y=0 (W/m^2)

# Grid Resolution
Nz, Ny = 60, 30  
z = np.linspace(0, L, Nz)
y = np.linspace(0, W, Ny)
dz = L / (Nz - 1)
dy = W / (Ny - 1)

# Time Settings
total_time = 8 * 3600  # Total simulation time: 8 hours in seconds

# Stability Criterion for Explicit Time Stepping (von Neumann)
dt_max = 0.5 / (alpha * (1/dz**2 + 1/dy**2))
dt = dt_max * 0.95      # Set dt safely below the limit (approx 0.75 seconds)
total_steps = int(total_time / dt)

# Specific time intervals we want to capture and plot (in seconds)
plot_times = [15*60, 1*3600, 4*3600, 8*3600]
plot_labels = ["15 Minutes", "1 Hour", "4 Hours", "8 Hours (Steady State)"]
captured_profiles = {}

# Initialize Temperature Matrix uniformly at an initial state (e.g., Room Temp)
T = np.ones((Nz, Ny)) * 27.0  
# Instantly apply the fixed base temperature at z=0
T[0, :] = T0

# ==========================================
# 2. TRANSIENT NUMERICAL SOLVER
# ==========================================
print(f"Starting explicit transient simulation (Time step dt = {dt:.3f} s)...")

current_time = 0.0
for step in range(1, total_steps + 1):
    T_old = T.copy()
    current_time += dt
    
    # 2.1 Explicit Update Equation for Internal Nodes
    for i in range(1, Nz - 1):
        for j in range(1, Ny - 1):
            d2T_dz2 = (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) / dz**2
            d2T_dy2 = (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1]) / dy**2
            T[i, j] = T_old[i, j] + alpha * dt * (d2T_dz2 + d2T_dy2)
            
    # 2.2 Enforce Boundary Conditions
    T[0, :] = T0                                                       # Left (z=0): Fixed Temp
    T[-1, :] = (T[-2, :] + (h * dz / k) * T_inf) / (1 + (h * dz / k))  # Right (z=L): Convection Cooling
    T[:, 0] = T[:, 1] + (q_in * dy / k)                                # Bottom (y=0): Constant Flux Input
    T[:, -1] = T[:, -2]                                                # Top (y=W): Insulated

    # Save a snapshot if we hit a milestone time target
    for pt, pl in zip(plot_times, plot_labels):
        if abs(current_time - pt) < (dt / 2):
            captured_profiles[pl] = T.copy()

print("Simulation Complete.")

# ==========================================
# 3. VISUALIZATION (Time Evolution Grid)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
Z, Y = np.meshgrid(z, y, indexing='ij')
axes_flat = axes.flatten()

# Establish a uniform global color limits matching final steady values
vmin, vmax = 27.0, 134.0 

for idx, label in enumerate(plot_labels):
    ax = axes_flat[idx]
    T_snapshot = captured_profiles[label]
    
    # Generate contour maps
    contour = ax.contourf(Z, Y, T_snapshot, levels=100, cmap='plasma', vmin=vmin, vmax=vmax)
    
    # Identify local extrema for this time step
    max_idx = np.unravel_index(np.argmax(T_snapshot), T_snapshot.shape)
    z_max, y_max = z[max_idx[0]], y[max_idx[1]]
    T_max = T_snapshot[max_idx]
    
    # Plot tracking markers
    ax.plot(z_max, y_max, 'w*', markersize=12, label=f'Max: {T_max:.1f}°C')
    
    ax.set_title(f'Thermal Profile at t = {label}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)

# Add shared global labels
fig.text(0.5, 0.02, 'Axial Distance z (m)', ha='center', fontsize=14)
fig.text(0.02, 0.5, 'Vertical Distance y (m)', va='center', rotation='vertical', fontsize=14)

# Put a common synchronized color bar on the right side
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
cbar = fig.colorbar(contour, cax=cbar_ax)
cbar.set_label('Temperature (°C)', fontsize=12)

plt.suptitle('Transient 2D Heat Propagation Over Time (Starting from 27°C)', fontsize=16, fontweight='bold')
plt.show()