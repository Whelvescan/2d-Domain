import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & GEOMETRY (1x1 Square)
# ==========================================
T_max = 80.0     # Hot boundary temperature (°C)
T_min = 27.0     # Cold boundary temperature (°C)
k = 50.0         # Thermal conductivity (W/m·K)
L = 1.0          # Normalized side length (m)

Nz, Ny = 100, 100
z = np.linspace(0, 1, Nz)
y = np.linspace(0, 1, Ny)
Z, Y = np.meshgrid(z, y, indexing='ij')

dtheta_dz = np.zeros((Nz, Ny))
dtheta_y = np.zeros((Nz, Ny))
flux_scale = -(k * (T_max - T_min)) / L
max_terms = 100  

# Evaluate global arrays for background plot visualization
for i in range(max_terms):
    n = 2 * i + 1  
    Cn = 4 / (n * np.pi * np.sinh(n * np.pi))
    dtheta_dz += Cn * (n * np.pi * np.cos(n * np.pi * Z)) * np.sinh(n * np.pi * (1 - Y))
    dtheta_y += Cn * np.sin(n * np.pi * Z) * (-n * np.pi * np.cosh(n * np.pi * (1 - Y)))

q_flux_z = flux_scale * dtheta_dz
q_flux_y = flux_scale * dtheta_y
q_flux_magnitude = np.sqrt(q_flux_z**2 + q_flux_y**2)

# ==========================================
# 2. GENERATE STATIC BACKGROUND MAP
# ==========================================
# We turn on interactive plotting mode so the graph window stays open 
# while the terminal waits for your text inputs.
plt.ion() 
fig, ax = plt.subplots(figsize=(8, 7))
contour = ax.contourf(Z, Y, q_flux_magnitude, levels=100, cmap='viridis')
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Heat Flux Magnitude ($W/m^2$)', fontsize=11)

skip = 6
ax.quiver(Z[::skip, ::skip], Y[::skip, ::skip], 
          q_flux_z[::skip, ::skip], q_flux_y[::skip, ::skip], 
          color='white', alpha=0.4, scale=1e5)

ax.plot(0.5, 0.5, 'gD', markersize=6, label='Geometric Center')
ax.set_title('Live Heat Flux Probe Mapping Grid', fontsize=12, fontweight='bold')
ax.set_xlabel('Dimensionless Position z')
ax.set_ylabel('Dimensionless Position y')
ax.legend(loc='upper right')
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
plt.pause(0.1) # Soft pause to render visual window layout layers cleanly

# Initialize an empty plot object for our moving query marker
probe_marker, = ax.plot([], [], 'ro', markersize=10, label='Active Probe Pin')

# ==========================================
# 3. LIVE INTERACTIVE TERMINAL LOOP
# ==========================================
print("\n" + "=" * 60)
print(f"{'LIVE RUNTIME HEAT FLUX QUERY INTERFACE':^60}")
print("=" * 60)
print("Instructions:")
print(" -> Enter coordinates as decimals between 0.0 and 1.0 (e.g., 0.25, 0.75)")
print(" -> Type 'exit' at any time to shut down the console engine.")
print("-" * 60)

while True:
    try:
        user_input = input("\nEnter target coordinates (z, y): ").strip()
        
        if user_input.lower() == 'exit':
            print("\nShutting down calculation engine. Goodbye!")
            break
            
        # Parse text input string split by comma delimiter
        z_str, y_str = user_input.split(",")
        qz_query = float(z_str.strip())
        qy_query = float(y_str.strip())
        
        # Guardrail checking boundary system domain conditions
        if not (0 <= qz_query <= 1) or not (0 <= qy_query <= 1):
            print("🛑 Error: Coordinates must remain bounded strictly within [0.0, 1.0]!")
            continue
            
        # Evaluate analytical calculation series at the unique coordinates requested
        qz_sum = 0.0
        qy_sum = 0.0
        for i in range(max_terms):
            n = 2 * i + 1
            Cn = 4 / (n * np.pi * np.sinh(n * np.pi))
            qz_sum += Cn * (n * np.pi * np.cos(n * np.pi * qz_query)) * np.sinh(n * np.pi * (1 - qy_query))
            qy_sum += Cn * np.sin(n * np.pi * qz_query) * (-n * np.pi * np.cosh(n * np.pi * (1 - qy_query)))
            
        final_qz = flux_scale * qz_sum
        final_qy = flux_scale * qy_sum
        final_mag = np.sqrt(final_qz**2 + final_qy**2)
        
        # Display calculation properties instantly to user screen console
        print("-" * 45)
        print(f"📍 CALCULATED VECTOR METRICS AT ({qz_query:.3f}, {qy_query:.3f}):")
        print(f" -> qz (Horizontal Component) : {final_qz:.2f} W/m²")
        print(f" -> qy (Vertical Component)   : {final_qy:.2f} W/m²")
        print(f" -> Net Vector Magnitude       : {final_mag:.2f} W/m²")
        print("-" * 45)
        
        # Dynamically move the red target dot on your open UI screen map
        probe_marker.set_data([qz_query], [qy_query])
        ax.legend(loc='upper right')
        plt.draw()
        plt.pause(0.1)
        
    except ValueError:
        print("🛑 Error: Invalid format! Please enter values separated by a comma (e.g., 0.5, 0.5)")
    except KeyboardInterrupt:
        print("\nSession broken manually. Exiting engine loop.")
        break