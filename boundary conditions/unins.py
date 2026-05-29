import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. FIXED PARAMETERS
# ==========================================
V_fixed = 120       # Constant Voltage (Volts)
L = 0.2            # Length (m)
R = 0.05           # Radius (m)
k = 15             # Cylinder conductivity (W/m-K)
h = 10             # Convection coefficient (W/m^2-K)
T_amb = 300        # Ambient Temperature (K)

# Simulation Grid
N = 50
dr = R / (N - 1)
r = np.linspace(0, R, N)

def solve_uninsulated_convection(V, I):
    Power = V * I
    Volume = np.pi * (R**2) * L
    q_gen = Power / Volume
    
    A = np.zeros((N, N))
    B = np.zeros(N)
    
    # Interior Nodes
    for i in range(1, N-1):
        ri = r[i]
        A[i, i-1] = 1/(dr**2) - 1/(2 * ri * dr)
        A[i, i]   = -2/(dr**2)
        A[i, i+1] = 1/(dr**2) + 1/(2 * ri * dr)
        B[i] = -q_gen / k

    # Boundary Conditions
    # 1. Center (r=0): Symmetry
    A[0, 0] = 1; A[0, 1] = -1; B[0] = 0 
    
    # 2. Surface (r=R): Direct Convection to Air
    A[N-1, N-2] = -k / dr
    A[N-1, N-1] = (k / dr) + h
    B[N-1] = h * T_amb
    
    return np.linalg.solve(A, B), Power

# ==========================================
# 2. PARAMETRIC STUDY
# ==========================================
current_values = [1, 2, 4, 6, 10, 12]  
plt.figure(figsize=(10, 6))

print("-" * 65)
print(f"{'Current (A)':<12} | {'Power (W)':<10} | {'Max (K)':<10} | {'Surface (K)':<12}")
print("-" * 65)

for I in current_values:
    T_profile, P_val = solve_uninsulated_convection(V_fixed, I)
    
    print(f"{I:<12} | {P_val:<10.1f} | {max(T_profile):<10.2f} | {T_profile[-1]:<12.2f}")
    
    plt.plot(r, T_profile, label=f'I = {I}A ({P_val}W)')

# Graph Formatting
plt.axhline(y=T_amb, color='black', linestyle='--', alpha=0.5, label='Ambient (300K)')
plt.title(f"Uninsulated Cylinder: Temperature vs. Power (h = {h} W/m^2K)")
plt.xlabel("Radius (m)")
plt.ylabel("Temperature (K)")
plt.legend(loc='upper left', fontsize='small')
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()