import numpy as np
import matplotlib.pyplot as plt

# 1. Parameters
L = 0.5          # Length (m)
k = 50           # Thermal conductivity (W/mK)
q_in = 2000      # Applied heat flux at z=0 (W/m^2)
TL = 80          # Fixed temp at z=L (C)
q_dot = 10000   # Internal heat generation (W/m^3)
nodes = 100
z = np.linspace(0, L, nodes)

# 2. Constants Derived from BCs
C1 = -q_in / k
C2 = TL + (q_dot * L**2) / (2 * k) - (C1 * L)

# 3. Temperature Distribution (Parabolic)
T = -(q_dot / (2 * k)) * z**2 + C1 * z + C2

# 4. Maxima Calculation (dT/dz = 0)
# - (q_dot / k) * z + C1 = 0  => z = C1 * k / q_dot
z_max = (C1 * k) / q_dot
T_max = -(q_dot / (2 * k)) * z_max**2 + C1 * z_max + C2

# 5. Output Results
print("-" * 40)
print(f"Internal Heat Generation: {q_dot} W/m³")
print(f"Surface Temp at z=0: {T[0]:.2f}°C")
print(f"Fixed End Temp at z=L: {T[-1]:.2f}°C")
if 0 <= z_max <= L:
    print(f"Peak Temp (Hot Spot): {T_max:.2f}°C at z={z_max:.4f}m")
else:
    print(f"Maximum Temp: {np.max(T):.2f}°C at z={z[np.argmax(T)]:.2f}m")
print("-" * 40)

# 6. Plotting
plt.figure(figsize=(10, 6))
plt.plot(z, T, 'r-', linewidth=2.5, label='Internal Heat Gen (q_dot)')
plt.plot(z[np.argmax(T)], np.max(T), 'ko', label='Maximum Temp')

plt.title('1D Axial Temp: Flux (z=0), Fixed Temp (z=L) with Heat Gen')
plt.xlabel('Distance z (m)')
plt.ylabel('Temperature (°C)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()