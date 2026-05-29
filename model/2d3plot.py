import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting engine

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
max_terms = 100  

for i in range(max_terms):
    n = 2 * i + 1  
    Cn = 4 / (n * np.pi * np.sinh(n * np.pi))
    theta += Cn * np.sin(n * np.pi * Z) * np.sinh(n * np.pi * (1 - Y))

# Explicitly override the hot baseline boundary to remove mathematical truncation ripples
theta[:, 0] = 1.0

# Back-Conversion to Physical Temperature (T)
T_physical = theta * (T_max - T_min) + T_min

# ==========================================
# 3. EXTRACTION OF CRITICAL DATA & SLICES
# ==========================================
mid_z_idx = Nz // 2
mid_y_idx = Ny // 2
T_mid = T_physical[mid_z_idx, mid_y_idx]
T_slice_z = T_physical[:, mid_y_idx]
T_slice_y = T_physical[mid_z_idx, :]

# ==========================================
# 4. TERMINAL METRICS MONITOR
# ==========================================
print("=" * 60)
print(f"{'CONSOLIDATED ANALYTICAL PROFILE OUTPUT':^60}")
print("=" * 60)
print(f"Dimensionless Center Value (theta)  : {theta[mid_z_idx, mid_y_idx]:.4f}")
print(f"Actual Physical Core Temp (T_mid)  : {T_mid:.2f}°C")
print("=" * 60)

# ==========================================
# 5. SEPARATED GRAPH VISUALIZATION WINDOWS
# ==========================================

# --- WINDOW 1: NEW 3D SURFACE PLOT ---
fig1 = plt.figure(1, figsize=(10, 8))
ax3d = fig1.add_subplot(111, projection='3d')

# Plot the 3D surface
surf = ax3d.plot_surface(Z, Y, T_physical, cmap='plasma', edgecolor='none', alpha=0.9)

# Add a color bar matching the surface profile
cbar = fig1.colorbar(surf, ax=ax3d, shrink=0.6, aspect=10)
cbar.set_label('Actual Temperature (°C)', fontsize=11)

# Highlight the midpoint core temperature directly on the 3D hill
ax3d.scatter(0.5, 0.5, T_mid, color='lime', s=100, marker='D', label=f'Center Core: {T_mid:.2f}°C', zorder=5)

# Set labels and viewport angles
ax3d.set_title('Window 1: 3D Steady-State Temperature Surface Landscape', fontsize=13, fontweight='bold')
ax3d.set_xlabel('Dimensionless Position z', fontsize=10)
ax3d.set_ylabel('Dimensionless Position y', fontsize=10)
ax3d.set_zlabel('Actual Temperature (°C)', fontsize=10)
ax3d.legend(loc='upper left')

# Adjust the camera angle for optimal viewing perspective (elevation, azimuth)
ax3d.view_init(elev=25, azim=-135)
plt.tight_layout()

# --- WINDOW 2: Horizontal z-direction Slice ---
plt.figure(2, figsize=(8, 4.5))
plt.plot(z, T_slice_z, color='firebrick', linewidth=2.5, label='Midline Slice Profile')
plt.axvline(0.5, color='gray', linestyle=':', alpha=0.7)
plt.axhline(T_mid, color='gray', linestyle=':', alpha=0.7)
plt.plot(0.5, T_mid, 'gD', markersize=8)
plt.title('Window 2: Temperature Variation in z-direction (at y=0.5)', fontsize=12, fontweight='bold')
plt.xlabel('Dimensionless Position z', fontsize=11)
plt.ylabel('Actual Physical Temperature (°C)', fontsize=11)
plt.xlim(0, 1)
plt.ylim(T_min - 2, T_max + 2)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower center')
plt.tight_layout()

# --- WINDOW 3: Vertical y-direction Slice ---
plt.figure(3, figsize=(8, 4.5))
plt.plot(y, T_slice_y, color='royalblue', linewidth=2.5, label='Midline Slice Profile')
plt.axvline(0.5, color='gray', linestyle=':', alpha=0.7)
plt.axhline(T_mid, color='gray', linestyle=':', alpha=0.7)
plt.plot(0.5, T_mid, 'gD', markersize=8)
plt.title('Window 3: Temperature Variation in y-direction (at z=0.5)', fontsize=12, fontweight='bold')
plt.xlabel('Dimensionless Position y', fontsize=11)
plt.ylabel('Actual Physical Temperature (°C)', fontsize=11)
plt.xlim(0, 1)
plt.ylim(T_min - 2, T_max + 2)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()

# Synchronously display all opened figures in individual viewports
plt.show()

from stl import mesh

# Create vertices from the grid network
vertices = np.zeros((Nz * Ny, 3))
idx = 0
for i in range(Nz):
    for j in range(Ny):
        vertices[idx] = [z[i], y[j], T_physical[i, j] / 80.0]  # Scale Z down slightly for CAD proportion
        idx += 1

# Connect vertices into triangular face plates
faces = []
for i in range(Nz - 1):
    for j in range(Ny - 1):
        v0 = i * Ny + j
        v1 = i * Ny + (j + 1)
        v2 = (i + 1) * Ny + j
        v3 = (i + 1) * Ny + (j + 1)
        faces.append([v0, v1, v2])  # First Triangle segment
        faces.append([v1, v3, v2])  # Second Triangle segment

faces = np.array(faces)

# Compile into structural STL mesh architecture and save
thermal_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        thermal_mesh.vectors[i][j] = vertices[f[j]]

thermal_mesh.save('thermal_landscape_model.stl')
print("CAD-compatible 'thermal_landscape_model.stl' generated successfully!")