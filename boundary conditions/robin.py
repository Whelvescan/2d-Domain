import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & CONSTANTS
# ==========================================
L = 0.5          # Length of cylinder (m)
k = 50           # Thermal conductivity (W/mK)
T0 = 80          # Fixed temperature at z=0 (Celsius)
h = 15           # Convective heat transfer coefficient (W/m^2K)
T_inf = 27       # Ambient air temperature (Celsius)
nodes = 100      # Number of calculation points

# ==========================================
# 2. ANALYTICAL CONSTANTS (Derived from BCs)
# ==========================================
# C2 is the intercept from the Dirichlet boundary at z=0
C2 = T0

# C1 is the slope derived from the Robin boundary at z=L
# Formula: C1 = -h * (T0 - T_inf) / (k + h*L)
C1 = (-h * (T0 - T_inf)) / (k + h * L)

# ==========================================
# 3. COORDINATES & TEMPERATURE ARRAY
# ==========================================
z = np.linspace(0, L, nodes)
T = C1 * z + C2

# Midpoint calculation for marking
mid_idx = nodes // 2
z_mid = z[mid_idx]
T_mid = T[mid_idx]

# ==========================================
# 4. VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(z, T, color='darkorange', linewidth=2.5, label='Temp Profile (Dirichlet-Robin)')

# Mark the Midpoint
plt.plot(z_mid, T_mid, 'bo', markersize=8, label=f'Midpoint: {T_mid:.2f}°C')

# Add text label for the midpoint
plt.annotate(f'({z_mid:.2f}m, {T_mid:.1f}°C)', 
             xy=(z_mid, T_mid), 
             xytext=(15, 15), 
             textcoords='offset points',
             color='blue',
             arrowprops=dict(arrowstyle='->', color='blue'),
             fontweight='bold')

# Reference line for Ambient Air
plt.axhline(T_inf, color='blue', linestyle='--', alpha=0.6, label=f'Ambient Air ({T_inf}°C)')

# Formatting
plt.title('1D Axial Temperature: Fixed Temp (z=0) & Convection (z=L)', fontsize=14)
plt.xlabel('Distance z (m)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

# Final check: Print results to terminal
print("-" * 40)
print(f"Fixed End Temp (z=0): {T[0]:.2f}°C")
print(f"Midpoint Temp (z={z_mid}): {T_mid:.2f}°C")
print(f"Exposed Surface Temp (z={L}): {T[-1]:.2f}°C")
print("-" * 40)

plt.show()