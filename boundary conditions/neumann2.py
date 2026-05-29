import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & CONSTANTS
# ==========================================
L = 0.5          # Length of cylinder (m)
k = 50           # Thermal conductivity (W/mK)
q_in = 2000      # Applied heat flux at z=0 (W/m^2)
h = 15           # Convective heat transfer coefficient (W/m^2K)
T_inf = 27       # Ambient air temperature (Celsius)
q_dot = 20000    # Internal heat generation (W/m^3)
nodes = 500      

# ==========================================
# 2. MATHEMATICAL DERIVATION (Constants)
# ==========================================
# C1 from Neumann BC: -k(dT/dz) = q_in at z=0
C1 = -q_in / k

# C2 from Robin BC: -k(dT/dz) = h(T - T_inf) at z=L
term_L = -k * (-(q_dot / k) * L + C1)
C2 = (term_L / h) + (q_dot * L**2 / (2 * k)) - (C1 * L) + T_inf

# ==========================================
# 3. COORDINATES & CALCULATIONS
# ==========================================
z = np.linspace(0, L, nodes)
T = -(q_dot / (2 * k)) * z**2 + C1 * z + C2

# Extract Critical Points
max_idx = np.argmax(T)
z_max, T_max = z[max_idx], T[max_idx]

mid_idx = nodes // 2
z_mid, T_mid = z[mid_idx], T[mid_idx]

T_surface_L = T[-1]

# ==========================================
# 4. TERMINAL OUTPUT
# ==========================================
print("="*50)
print(f"{'THERMAL ANALYSIS SYSTEM OUTPUT':^50}")
print("="*50)
print(f"Input Parameters:")
print(f"  - Length (L):         {L} m")
print(f"  - Thermal Cond (k):   {k} W/mK")
print(f"  - Input Flux (q_in):  {q_in} W/m^2")
print(f"  - Heat Gen (q_dot):   {q_dot} W/m^3")
print("-"*50)
print(f"Calculated Results:")
print(f"  - Max Temp (z={z_max:.3f}m):  {T_max:.2f}°C")
print(f"  - Midpoint (z={z_mid:.2f}m):   {T_mid:.2f}°C")
print(f"  - Surface  (z={L:.2f}m):     {T_surface_L:.2f}°C")
print("="*50)

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 7))

# Main Temperature Curve
plt.plot(z, T, color='magenta', linewidth=2.5, label='Temperature Profile')

# Mark Maxima (Red Star) and Midpoint (Blue Dot)
plt.plot(z_max, T_max, 'r*', markersize=15, label=f'Max Temp: {T_max:.2f}°C')
plt.plot(z_mid, T_mid, 'bo', markersize=8, label=f'Midpoint: {T_mid:.2f}°C')

# Annotations
plt.annotate(f'Peak: {T_max:.1f}°C', xy=(z_max, T_max), xytext=(20, 10), 
             textcoords='offset points', color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))

plt.annotate(f'Mid: {T_mid:.1f}°C', xy=(z_mid, T_mid), xytext=(20, -20), 
             textcoords='offset points', color='blue', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='blue'))

# Plot Formatting
plt.title('1D Axial Temperature Distribution (Heat Gen: 20kW/m³)', fontsize=14)
plt.xlabel('Distance z (m)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.axhline(T_inf, color='green', linestyle='--', label=f'Air Temp ({T_inf}°C)')
plt.legend(loc='best')

plt.show()