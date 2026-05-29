import numpy as np
import matplotlib.pyplot as plt

# 1. Parameters
L = 0.5          # Length of the cylinder (m)
k = 50           # Thermal conductivity (W/mK)
T0 = 30          # Temperature at z=0 (Celsius)
TL = 80          # Temperature at z=L (Celsius)
q_dot = 200000    # Internal heat generation (W/m^3) - ASSUMED
nodes = 100      

# 2. Coordinate System
z = np.linspace(0, L, nodes)

# 3. Parabolic Constants (Derived from Dirichlet Boundaries)
C2 = T0
C1 = (TL - T0) / L + (q_dot * L) / (2 * k)

# 4. Temperature Distribution (Analytical Solution)
T = -(q_dot / (2 * k)) * z**2 + C1 * z + C2

# 5. Extract Maxima
# Maxima occurs where dT/dz = 0 => z_max = (C1 * k) / q_dot
z_max = (C1 * k) / q_dot
T_max = -(q_dot / (2 * k)) * z_max**2 + C1 * z_max + C2

# 6. Output Results
print("-" * 30)
print(f"Internal Heat Generation: {q_dot} W/m³")
print(f"Maximum Temp: {T_max:.2f}°C")
print(f"Location of Maxima: z = {z_max:.3f}m")
print("-" * 30)

# 7. Plotting
plt.figure(figsize=(10, 6))
plt.plot(z, T, color='red', linewidth=2.5, label='Parabolic Profile (with q_dot)')

# Mark the Maxima (Hot Spot)
if 0 <= z_max <= L:
    plt.plot(z_max, T_max, 'ko', markersize=8, label=f'Maxima: {T_max:.1f}°C')
    plt.annotate(f'Hot Spot: {T_max:.1f}°C', (z_max, T_max), xytext=(10, 10), 
                 textcoords="offset points", fontweight='bold')

plt.title('1D Axial Temperature Distribution with Heat Generation', fontsize=14)
plt.xlabel('Axial Distance z (m)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()