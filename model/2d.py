import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & GEOMETRY (1x1 Square)
# ==========================================
T_max = 80.0     # Hot boundary temperature (°C)
T_min = 27.0     # Cold boundary temperature (°C)

# Mesh Resolution for the Normalized Domain
Nz, Ny = 100, 100
z = np.linspace(0, 1, Nz)
y = np.linspace(0, 1, Ny)
Z, Y = np.meshgrid(z, y, indexing='ij')

# Initialize the dimensionless temperature matrix (theta) to zero
theta = np.zeros((Nz, Ny))

# ==========================================
# 2. EVALUATE THE FOURIER SERIES EQUATION
# ==========================================
max_terms = 100  # Number of terms to guarantee strict analytical convergence

for i in range(max_terms):
    n = 2 * i + 1  # Generates only odd modes: 1, 3, 5, 7...
    
    # Exact Fourier coefficient derived via orthogonality filtering
    Cn = 4 / (n * np.pi * np.sinh(n * np.pi))
    
    # Spatial structural wave modes
    sin_term = np.sin(n * np.pi * Z)
    sinh_term = np.sinh(n * np.pi * (1 - Y))
    
    # Accumulate current mode contribution
    theta += Cn * sin_term * sinh_term

# Explicitly override the hot baseline boundary to remove mathematical truncation ripples
theta[:, 0] = 1.0

# ==========================================
# 3. BACK-CONVERSION TO PHYSICAL TEMPERATURE (T)
# ==========================================
# Transformation Formula: T = theta * (T_max - T_min) + T_min
T_physical = theta * (T_max - T_min) + T_min

# ==========================================
# 4. EXTRACTION OF CRITICAL DATA & SLICES
# ==========================================
mid_z_idx = Nz // 2
mid_y_idx = Ny // 2

# Extract individual coordinate points for center assessment
theta_mid = theta[mid_z_idx, mid_y_idx]
T_mid = T_physical[mid_z_idx, mid_y_idx]

# Slice 1: Temperature along the z-direction crossing the midline (y = 0.5)
T_slice_z = T_physical[:, mid_y_idx]

# Slice 2: Temperature along the y-direction crossing the midline (z = 0.5)
T_slice_y = T_physical[mid_z_idx, :]

# ==========================================
# 5. TERMINAL METRICS MONITOR
# ==========================================
print("=" * 60)
print(f"{'CONSOLIDATED ANALYTICAL PROFILE OUTPUT':^60}")
print("=" * 60)
print(f"Evaluated Geometry Matrix : {Nz} x {Ny} Points")
print(f"Fourier Series Extent     : Computed up to {2*max_terms-1} expansion modes")
print("-" * 60)
print(f"Dimensionless Center Value (theta)  : {theta_mid:.4f}")
print(f"Actual Physical Core Temp (T_mid)  : {T_mid:.2f}°C")
print("=" * 60)

# ==========================================
# 6. COMPREHENSIVE GRAPH VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(14, 10))

# 6.1 Main 2D Contour Map Panel (Spans across the entire top half)
ax_map = plt.subplot2grid((2, 2), (0, 0), colspan=2)
contour = ax_map.contourf(Z, Y, T_physical, levels=100, cmap='plasma')
cbar = fig.colorbar(contour, ax=ax_map)
cbar.set_label('Actual Temperature (°C)', fontsize=11)

# Overlay positional tracking cut lines on the map
ax_map.axhline(0.5, color='white', linestyle='--', alpha=0.6, label='Horizontal Track (y=0.5)')
ax_map.axvline(0.5, color='cyan', linestyle='--', alpha=0.6, label='Vertical Track (z=0.5)')
ax_map.plot(0.5, 0.5, 'gD', markersize=10, label=f'Center Midpoint: {T_mid:.2f}°C')

ax_map.set_title('2D Analytical Temperature Profile Map with Track Splines', fontsize=13, fontweight='bold')
ax_map.set_xlabel('Dimensionless Position z', fontsize=11)
ax_map.set_ylabel('Dimensionless Position y', fontsize=11)
ax_map.legend(loc='upper right')
ax_map.set_aspect('equal')

# 6.2 Horizontal Profile Line Plot: Temperature vs z-axis
ax_z = plt.subplot2grid((2, 2), (1, 0))
ax_z.plot(z, T_slice_z, color='firebrick', linewidth=2.5, label='Midline Slice Profile')
ax_z.axvline(0.5, color='gray', linestyle=':', alpha=0.7)
ax_z.axhline(T_mid, color='gray', linestyle=':', alpha=0.7)
ax_z.plot(0.5, T_mid, 'gD', markersize=8)

ax_z.set_title('Temperature Variation in z-direction (at y=0.5)', fontsize=11, fontweight='bold')
ax_z.set_xlabel('Dimensionless Position z', fontsize=10)
ax_z.set_ylabel('Actual Physical Temperature (°C)', fontsize=10)
ax_z.set_xlim(0, 1)
ax_z.set_ylim(T_min - 2, T_max + 2)
ax_z.grid(True, linestyle=':', alpha=0.6)
ax_z.legend(loc='lower center')

# 6.3 Vertical Profile Line Plot: Temperature vs y-axis
ax_y = plt.subplot2grid((2, 2), (1, 1))
ax_y.plot(y, T_slice_y, color='royalblue', linewidth=2.5, label='Midline Slice Profile')
ax_y.axvline(0.5, color='gray', linestyle=':', alpha=0.7)
ax_y.axhline(T_mid, color='gray', linestyle=':', alpha=0.7)
ax_y.plot(0.5, T_mid, 'gD', markersize=8)

ax_y.set_title('Temperature Variation in y-direction (at z=0.5)', fontsize=11, fontweight='bold')
ax_y.set_xlabel('Dimensionless Position y', fontsize=10)
ax_y.set_ylabel('Actual Physical Temperature (°C)', fontsize=10)
ax_y.set_xlim(0, 1)
ax_y.set_ylim(T_min - 2, T_max + 2)
ax_y.grid(True, linestyle=':', alpha=0.6)
ax_y.legend(loc='upper right')

plt.suptitle('Unified 2D Space-Variant Analytical Thermal Analysis Matrix', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()