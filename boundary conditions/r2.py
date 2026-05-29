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
q_dot = 200000     # Internal heat generation (W/m^3)
nodes = 200      

# ==========================================
# 2. ANALYTICAL CONSTANTS (Derived from BCs)
# ==========================================
C2 = T0
# C1 derived by substituting general solution into convection BC at z=L
numerator = h * (T_inf - T0 + (q_dot * L**2) / (2 * k)) - q_dot * L
denominator = k + h * L
C1 = numerator / denominator

# ==========================================
# 3. COORDINATES & TEMPERATURE ARRAY
# ==========================================
z = np.linspace(0, L, nodes)
T = -(q_dot / (2 * k)) * z**2 + C1 * z + C2

# Find Maximum Temperature and its Location
max_idx = np.argmax(T)
z_max, T_max = z[max_idx], T[max_idx]

# Find Midpoint Temperature
mid_idx = nodes // 2
z_mid, T_mid = z[mid_idx], T[mid_idx]

# ==========================================
# 4. TERMINAL OUTPUT
# ==========================================
print("-" * 45)
print(f"Max Temperature: {T_max:.2f}°C at z = {z_max:.3f}m")
print(f"Midpoint Temp:   {T_mid:.2f}°C")
print(f"Convection End:  {T[-1]:.2f}°C")
print("-" * 45)

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(z, T, color='darkorange', linewidth=2.5, label='Parabolic Temp Profile')

# Mark the Maximum Point (Red Star)
plt.plot(z_max, T_max, 'r*', markersize=12, label=f'Peak: {T_max:.2f}°C')

# Mark the Midpoint (Blue Circle)
plt.plot(z_mid, T_mid, 'bo', markersize=8, label=f'Mid: {T_mid:.2f}°C')

# Annotations
plt.annotate(f'Peak: {T_max:.1f}°C', xy=(z_max, T_max), xytext=(15, 10), 
             textcoords='offset points', color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))

plt.annotate(f'Mid: {T_mid:.1f}°C', xy=(z_mid, T_mid), xytext=(15, -20), 
             textcoords='offset points', color='blue', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='blue'))

plt.axhline(T_inf, color='blue', linestyle='--', alpha=0.5, label=f'Air ({T_inf}°C)')
plt.title('1D Heat Distribution: Fixed Temp, Convection & Heat Gen', fontsize=13)
plt.xlabel('Distance z (m)')
plt.ylabel('Temperature (°C)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()