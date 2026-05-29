import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SHARED PARAMETERS
# ==========================================
V, I = 120, 2      # Electrical Input
L, R = 0.2, 0.05  # Geometry
k_cyl = 15        # Cylinder conductivity
k_ins = 0.5       # Insulation conductivity
t_ins = 0.01      # Insulation thickness
h_air = 10        # Heat Transfer Coefficient for both (W/m^2-K)
T_amb = 300       # Ambient (K)

N = 50
dr = R / (N - 1)
r = np.linspace(0, R, N)

def solve_system(V, I, h_effective):
    Power = V * I
    Volume = np.pi * (R**2) * L
    q_gen = Power / Volume
    
    A = np.zeros((N, N))
    B = np.zeros(N)
    
    for i in range(1, N-1):
        ri = r[i]
        A[i, i-1] = 1/(dr**2) - 1/(2 * ri * dr)
        A[i, i]   = -2/(dr**2)
        A[i, i+1] = 1/(dr**2) + 1/(2 * ri * dr)
        B[i] = -q_gen / k_cyl

    A[0, 0] = 1; A[0, 1] = -1; B[0] = 0 # Symmetry
    
    # Boundary Condition using the provided h
    A[N-1, N-2] = -k_cyl / dr
    A[N-1, N-1] = (k_cyl / dr) + h_effective
    B[N-1] = h_effective * T_amb
    
    return np.linalg.solve(A, B)

# ==========================================
# 2. CALCULATING THE DIFFERENCE
# ==========================================
# Uninsulated: Heat goes straight to air
T_unins = solve_system(V, I, h_air)

# Insulated: Heat must pass through insulation (R_ins) and then air (R_conv)
# Total Resistance R_total = R_insulation + R_convection
# For a thin-film approx: 1/h_total = (t_ins/k_ins) + (1/h_air)
h_total = 1 / ((t_ins / k_ins) + (1 / h_air))

T_ins = solve_system(V, I, h_total)

# ==========================================
# 3. RESULTS
# ==========================================
print("-" * 55)
print(f"{'Scenario':<25} | {'Max (K)':<10} | {'Surface (K)':<12}")
print("-" * 55)
print(f"{'Uninsulated (h=10)':<25} | {max(T_unins):<10.2f} | {T_unins[-1]:<12.2f}")
print(f"{'Insulated (h_total=' + f'{h_total:.1f})':<25} | {max(T_ins):<10.2f} | {T_ins[-1]:<12.2f}")
print("-" * 55)

plt.figure(figsize=(10, 6))
plt.plot(r, T_unins, 'r--', label='Uninsulated (Direct Convection)')
plt.plot(r, T_ins, 'b-', label=f'Insulated (Total h={h_total:.2f})')
plt.axhline(y=300, color='gray', linestyle=':', label='Ambient 300K')
plt.title(f"Thermal Impact of Adding Insulation (h_air = {h_air})")
plt.xlabel("Radius (m)")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()