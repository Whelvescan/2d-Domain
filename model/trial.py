import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & GEOMETRY
# ==========================================
L = 0.5          # Length in z-direction (m)
W = 0.2          # Width in y-direction (m)
k = 50           # Thermal conductivity (W/mK)

# Boundary Values
T0 = 80          # Constant temp at z=0 (°C)
h = 15           # Natural Convection coeff at z=L (W/m^2K)
T_inf = 27       # Ambient Air temp at z=L (°C)
q_in = 5000      # Constant heat flux entering at y=0 (W/m^2)

# Grid Resolution
Nz, Ny = 60, 30  
z = np.linspace(0, L, Nz)
y = np.linspace(0, W, Ny)
dz = L / (Nz - 1)
dy = W / (Ny - 1)

# Initialize Temperature Matrix
T = np.ones((Nz, Ny)) * T0  # Initial guess

# ==========================================
# 2. NUMERICAL SOLVER (FDM Laplace with Convection)
# ==========================================
max_error = 1e-6
error = 1.0
iterations = 0

while error > max_error and iterations < 10000:
    T_old = T.copy()
    
    # 2.1 Update Internal Nodes (Pure Diffusion: q_dot = 0)
    for i in range(1, Nz - 1):
        for j in range(1, Ny - 1):
            term_z = (T[i+1, j] + T[i-1, j]) / dz**2
            term_y = (T[i, j+1] + T[i, j-1]) / dy**2
            denom = 2/dz**2 + 2/dy**2
            T[i, j] = (term_z + term_y) / denom
            
    # 2.2 Enforce Boundary Conditions
    T[0, :] = T0                                                       # Left (z=0): Fixed Temp
    T[-1, :] = (T[-2, :] + (h * dz / k) * T_inf) / (1 + (h * dz / k))  # Right (z=L): Active Convection Cooling
    T[:, 0] = T[:, 1] + (q_in * dy / k)                                # Bottom (y=0): Constant Flux Input
    T[:, -1] = T[:, -2]                                                # Top (y=W): Insulated

    error = np.max(np.abs(T - T_old))
    iterations += 1

# ==========================================
# 3. CRITICAL POINT EXTRACTION
# ==========================================
# Find Absolute Maxima
max_idx = np.unravel_index(np.argmax(T), T.shape)
z_max, y_max = z[max_idx[0]], y[max_idx[1]]
T_max = T[max_idx]

# Find Geometric Midpoint
mid_z_idx, mid_y_idx = Nz // 2, Ny // 2
z_mid, y_mid = z[mid_z_idx], y[mid_y_idx]
T_mid = T[mid_z_idx, mid_y_idx]

print(f"Convergence achieved in {iterations} iterations.")
print(f"Maximum Temperature: {T_max:.2f}°C at z={z_max:.3f}m, y={y_max:.3f}m")
print(f"Geometric Midpoint Temp: {T_mid:.2f}°C at z={z_mid:.3f}m, y={y_mid:.3f}m")

# ==========================================
# 4. 2D VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 6))
Z, Y = np.meshgrid(z, y, indexing='ij')

contour = plt.contourf(Z, Y, T, levels=100, cmap='plasma')
cbar = plt.colorbar(contour)
cbar.set_label('Temperature (°C)', fontsize=12)

# Mark the Correct Maxima and Midpoint
plt.plot(z_max, y_max, 'w*', markersize=14, label=f'Max Temp: {T_max:.1f}°C')
plt.plot(z_mid, y_mid, 'gD', markersize=8, label=f'Midpoint: {T_mid:.1f}°C')

plt.title('2D Temperature Profile with Active Convection Cooling at z=0.5', fontsize=14)
plt.xlabel('Axial Distance z (m)', fontsize=12)
plt.ylabel('Vertical Distance y (m)', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()