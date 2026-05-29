import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. INPUT PARAMETERS (V and I for Team Zeta)
# ==========================================
V = 12            # Voltage (Volts)
I = 2             # Current (Amps)
L = 0.2           # Cylinder Length (m)
R = 0.05          # Cylinder Radius (m)
k = 15            # Cylinder conductivity (W/m-K)
k_ins = 0.5       # Imperfect insulation conductivity
t_ins = 0.01      # Insulation thickness (m)
T_amb = 300       # Surrounding Temperature (K)

# ==========================================
# 2. CALCULATE HEAT GENERATION
# ==========================================
Power = V * I
Volume = np.pi * (R**2) * L
q_gen = Power / Volume

# ==========================================
# 3. NUMERICAL SETUP
# ==========================================
N = 50            
dr = R / (N - 1)
r = np.linspace(0, R, N)

A = np.zeros((N, N))
B = np.zeros(N)

# Interior Nodes
for i in range(1, N-1):
    ri = r[i]
    A[i, i-1] = 1/(dr**2) - 1/(2 * ri * dr)
    A[i, i]   = -2/(dr**2)
    A[i, i+1] = 1/(dr**2) + 1/(2 * ri * dr)
    B[i] = -q_gen / k

# ==========================================
# 4. BOUNDARY CONDITIONS
# ==========================================
# Center (r=0): Symmetry
A[0, 0] = 1
A[0, 1] = -1
B[0] = 0

# Surface (r=R): The interface between Cylinder and Insulation
# This calculates the temperature before the heat passes through insulation
h_eff = 10
A[N-1, N-2] = -k / dr
A[N-1, N-1] = (k / dr) + h_eff
B[N-1] = h_eff * T_amb

# ==========================================
# 5. SOLVE AND DISPLAY
# ==========================================
T = np.linalg.solve(A, B)

# The last node T[-1] is the Surface of the Cylinder
print("-" * 40)
print(f"RESULTS FOR V={V}V, I={I}A")
print("-" * 40)
print(f"Max Temp (Center):         {max(T):.2f} K")
print(f"Cylinder Surface Temp:    {T[-1]:.2f} K") # This is what you asked for
print(f"Ambient Temperature:      {T_amb}.00 K")
print(f"Temperature Rise Above Ambient: {T[-1] - T_amb:.2f} K")
print("-" * 40)

plt.plot(r, T, 'b-', label='Cylinder Internal Temp')
plt.axhline(y=T_amb, color='r', linestyle='--', label='Ambient (300K)')
plt.xlabel("Radius (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature Profile (Metal to Insulation Interface)")
plt.legend()
plt.grid(True)
plt.show()
