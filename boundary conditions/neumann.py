import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 0.5          # Length (m)
k = 50           # Thermal conductivity (W/mK)
q_in = 2000      # Applied heat flux at z=0 (W/m^2)
TL = 80          # Fixed temp at z=L (C)
nodes = 100
z = np.linspace(0, L, nodes)

# Analytical Solution: T(z) = (q_in/k) * (L - z) + TL
T = (q_in / k) * (L - z) + TL

# Calculations
max_temp = np.max(T)
mid_temp = T[nodes // 2]

print(f"Max Temp: {max_temp:.2f}°C at z=0")
print(f"Midpoint Temp: {mid_temp:.2f}°C")

# Plotting
plt.plot(z, T, 'g-', label='Mixed Neumann-Dirichlet')
plt.title('1D Axial Temp: Flux (z=0) and Fixed Temp (z=L)')
plt.xlabel('z (m)')
plt.ylabel('Temp (°C)')
plt.grid(True)
plt.legend()
plt.show()