import numpy as np
import matplotlib.pyplot as plt

# 1. Parameters
L = 0.5          # Length of the cylinder (m)
T0 = 30          # Temperature at left face (z=0) in Celsius
TL = 80          # Temperature at right face (z=L) in Celsius
nodes = 100      # Resolution of the calculation

# 2. Coordinate System
z = np.linspace(0, L, nodes)

# 3. Temperature Distribution Equation (Analytical Solution)
# For 1D steady state, no heat generation: T(z) = ((TL - T0)/L) * z + T0
T = ((TL - T0) / L) * z + T0

# 4. Extract Specific Values
# Midpoint calculation
mid_idx = nodes // 2
z_mid = z[mid_idx]
T_mid = T[mid_idx]

# Maximum temperature and its location
max_temp = np.max(T)
max_loc = z[np.argmax(T)]

# 5. Output Results to Console
print("-" * 30)
print("  THERMAL ANALYSIS RESULTS")
print("-" * 30)
print(f"Governing Eq: T(z) = { (TL-T0)/L } * z + {T0}")
print(f"Midpoint Temp (z={z_mid:.3f}m): {T_mid:.2f}°C")
print(f"Maximum Temp: {max_temp:.2f}°C")
print(f"Max Temp Location: z = {max_loc:.3f}m")
print("-" * 30)

# 6. Plotting the Distribution
plt.figure(figsize=(10, 6))

# Plot the main temperature curve
plt.plot(z, T, color='red', linewidth=2.5, label='Temperature Profile')

# Mark the Midpoint
plt.plot(z_mid, T_mid, 'bo', markersize=8, label=f'Midpoint: {T_mid:.1f}°C')

# Mark the Boundaries
plt.scatter([0, L], [T0, TL], color='black', zorder=5)

# Formatting the Chart
plt.title('1D Axial Temperature Distribution (Dirichlet Conditions)', fontsize=14)
plt.xlabel('Axial Distance z (m)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(T0 - 10, TL + 10) # Add padding to see boundaries clearly

# Annotate points
plt.annotate(f'Left: {T0}°C', (0, T0), textcoords="offset points", xytext=(0,10), ha='center', weight='bold')
plt.annotate(f'Right: {TL}°C', (L, TL), textcoords="offset points", xytext=(0,10), ha='center', weight='bold')
plt.annotate(f'Mid: {T_mid}°C', (z_mid, T_mid), textcoords="offset points", xytext=(0,-20), ha='center', color='blue')

plt.legend()
plt.show()