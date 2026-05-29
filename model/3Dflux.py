import numpy as np
import plotly.graph_objects as go
import sys

# ==========================================
# 1. PARAMETERS & GEOMETRY (1x1 Square)
# ==========================================
T_max = 80.0     # Hot boundary temperature (°C)
T_min = 27.0     # Cold boundary temperature (°C)
k = 50.0         # Thermal conductivity (W/m·K)
L = 1.0          # Normalized side length (m)

# Mesh Resolution for the 2D Plane
Nz, Ny = 100, 100
z = np.linspace(0, 1, Nz)
y = np.linspace(0, 1, Ny)
Z, Y = np.meshgrid(z, y, indexing='ij')

theta = np.zeros((Nz, Ny))
dtheta_dz = np.zeros((Nz, Ny))
dtheta_y = np.zeros((Nz, Ny))

# Constant physical scaling multiplier
flux_scale = -(k * (T_max - T_min)) / L
max_terms = 100  

# ==========================================
# 2. EVALUATE COMPLETE FIELDS (GLOBAL ARRAYS)
# ==========================================
print("Evaluating analytical Fourier series equations... Please wait.")
for i in range(max_terms):
    n = 2 * i + 1  
    Cn = 4 / (n * np.pi * np.sinh(n * np.pi))
    
    # 2.1 Complete global temperature array accumulation
    theta += Cn * np.sin(n * np.pi * Z) * np.sinh(n * np.pi * (1 - Y))
    
    # 2.2 Complete global derivative array accumulations
    dtheta_dz += Cn * (n * np.pi * np.cos(n * np.pi * Z)) * np.sinh(n * np.pi * (1 - Y))
    dtheta_y += Cn * np.sin(n * np.pi * Z) * (-n * np.pi * np.cosh(n * np.pi * (1 - Y)))

# Override baseline hot boundary row to suppress truncation ripple noise
theta[:, 0] = 1.0

# Scale metrics back to real physical temperature measurements
T_physical = theta * (T_max - T_min) + T_min

# ==========================================
# 3. INITIALIZE INTERACTIVE 3D VIEWPORT
# ==========================================
# Generate base thermal mountain topology trace
fig = go.Figure(data=[go.Surface(
    z=T_physical.T,  # Transposed to align with Plotly layout indexing
    x=z,
    y=y,
    colorscale='Plasma',
    colorbar=dict(title='Temperature (°C)', len=0.75, x=1.05),
    hoverinfo='text',
    text=[[f"z: {z[i]:.2f}, y: {y[j]:.2f}<br>Temp: {T_physical[i,j]:.2f}°C" 
           for j in range(Ny)] for i in range(Nz)]
)])

# Add empty dynamic tracking dot placeholder trace
fig.add_trace(go.Scatter3d(
    x=[], y=[], z=[],
    mode='markers+text',
    marker=dict(size=12, color='lime', symbol='diamond', line=dict(color='black', width=2)),
    name='Active Probe Target'
))

fig.update_layout(
    title='Interactive 3D Temperature Surface with Live Heat Flux Probing',
    scene=dict(
        xaxis_title='Dimensionless Position z',
        yaxis_title='Dimensionless Position y',
        zaxis_title='Temperature (°C)',
        aspectratio=dict(x=1, y=1, z=0.7)
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

# Launch open graphics view panel automatically in primary internet browser
fig.show()

# ==========================================
# 4. LIVE INTERACTIVE TERMINAL LOOP
# ==========================================
print("\n" + "=" * 60)
print(f"{'3D THERMAL LANDSCAPE ANALYSIS INTERFACE':^60}")
print("=" * 60)
print("Instructions:")
print(" -> Enter testing coordinate frames as decimals: z, y (e.g., 0.25, 0.50)")
print(" -> Type 'exit' to cleanly close down operations.")
print("-" * 60)

while True:
    try:
        user_input = input("\nEnter coordinates for heat flux calculation (z, y): ").strip()
        
        if user_input.lower() == 'exit':
            print("\nShutting down interactive session. Plotting pipeline closed.")
            break
            
        # Parse inputs by comma divider
        z_str, y_str = user_input.split(",")
        qz_query = float(z_str.strip())
        qy_query = float(y_str.strip())
        
        # Grid range boundary validation guardrails
        if not (0 <= qz_query <= 1) or not (0 <= qy_query <= 1):
            print("🛑 Error: Specified targets must reside within boundary spectrum [0.0, 1.0].")
            continue
            
        # 4.1 Evaluate analytical matrix calculations directly at single coordinate point
        t_sum = 0.0
        qz_sum = 0.0
        qy_sum = 0.0
        
        for i in range(max_terms):
            n = 2 * i + 1
            Cn = 4 / (n * np.pi * np.sinh(n * np.pi))
            
            t_sum += Cn * np.sin(n * np.pi * qz_query) * np.sinh(n * np.pi * (1 - qy_query))
            qz_sum += Cn * (n * np.pi * np.cos(n * np.pi * qz_query)) * np.sinh(n * np.pi * (1 - qy_query))
            qy_sum += Cn * np.sin(n * np.pi * qz_query) * (-n * np.pi * np.cosh(n * np.pi * (1 - qy_query)))
            
        if qy_query == 0.0:
            t_sum = 1.0  # Safe hot base override limit condition check
            
        final_temp = t_sum * (T_max - T_min) + T_min
        final_qz = flux_scale * qz_sum
        final_qy = flux_scale * qy_sum
        final_mag = np.sqrt(final_qz**2 + final_qy**2)
        
        # 4.2 DISPLAY METRICS DIRECTLY IN TERMINAL CONSOLE
        print("-" * 50)
        print(f"📍 CORE PROPERTIES TRACKED AT COORDINATES ({qz_query:.3f}, {qy_query:.3f}):")
        print(f" -> Local Temperature      : {final_temp:.2f} °C")
        print(f" -> qz Horizontal Flux     : {final_qz:.2f} W/m²")
        print(f" -> qy Vertical Flux       : {final_qy:.2f} W/m²")
        print(f" -> Net Flux Vector Speed  : {final_mag:.2f} W/m²")
        print("-" * 50)
        
        # 4.3 UPDATE THE DYNAMIC HOVER INJECTS INSIDE THE OPEN 3D PLOTLY CANVAS
        hover_label = (f"PROBE TARGET<br>"
                       f"Position: ({qz_query:.2f}, {qy_query:.2f})<br>"
                       f"Temp: {final_temp:.2f}°C<br>"
                       f"qz Flux: {final_qz:.1f} W/m²<br>"
                       f"qy Flux: {final_qy:.1f} W/m²<br>"
                       f"Net Intensity: {final_mag:.1f} W/m²")
        
        # Append parameters into active point data layer configurations
        fig.data[1].x = [qz_query]
        fig.data[1].y = [qy_query]
        fig.data[1].z = [final_temp]
        fig.data[1].text = [f"Active Pin: {final_mag:.1f} W/m²"]
        fig.data[1].hovertext = [hover_label]
        fig.data[1].hoverinfo = 'text'
        
        # Command active local session channel instance frame context reload
        fig.show()
        
    except ValueError:
        print("🛑 Error: Input execution broken. Separate tracking arguments using a comma (e.g. 0.35, 0.60)")
    except KeyboardInterrupt:
        print("\nPipeline terminal loop broken. Exiting runtime context.")
        break