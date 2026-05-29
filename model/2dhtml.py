import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. PARAMETERS & GEOMETRY (1x1 Square)
# ==========================================
T_max = 80.0     # Hot boundary temperature (°C)
T_min = 27.0     # Cold boundary temperature (°C)

Nz, Ny = 100, 100
z = np.linspace(0, 1, Nz)
y = np.linspace(0, 1, Ny)
Z, Y = np.meshgrid(z, y, indexing='ij')

theta = np.zeros((Nz, Ny))

# ==========================================
# 2. EVALUATE THE FOURIER SERIES EQUATION
# ==========================================
max_terms = 100  
for i in range(max_terms):
    n = 2 * i + 1  
    Cn = 4 / (n * np.pi * np.sinh(n * np.pi))
    theta += Cn * np.sin(n * np.pi * Z) * np.sinh(n * np.pi * (1 - Y))

theta[:, 0] = 1.0  # Force boundary conditions

# Back-Conversion to Physical Temperature (T)
T_physical = theta * (T_max - T_min) + T_min
T_mid = T_physical[Nz // 2, Ny // 2]

# ==========================================
# 3. GENERATE INTERACTIVE 3D SURFACE MODEL
# ==========================================
# Create the main 3D surface
fig = go.Figure(data=[go.Surface(
    z=T_physical.T,  # Transpose needed to match Plotly's axis orientation
    x=z,
    y=y,
    colorscale='Plasma',
    colorbar=dict(title='Temperature (°C)')
)])

# Add a floating 3D marker right on the geometric center core
fig.add_trace(go.Scatter3d(
    x=[0.5],
    y=[0.5],
    z=[T_mid],
    mode='markers',
    marker=dict(size=8, color='lime', symbol='diamond'),
    name=f'Core Midpoint: {T_mid:.2f}°C'
))

# Configure 3D scene architecture and labeling
fig.update_layout(
    title='Interactive 2D Analytical Temperature Landscape',
    scene=dict(
        xaxis_title='Dimensionless Position z',
        yaxis_title='Dimensionless Position y',
        zaxis_title='Actual Temperature (°C)',
        aspectratio=dict(x=1, y=1, z=0.7)  # Slightly flattens the Z scale for cleaner looks
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# ==========================================
# 4. EXPORT AND SAVE TO A 3D FILE
# ==========================================
output_filename = "3d_thermal_landscape.html"
fig.write_html(output_filename)

print("=" * 60)
print(f"Success! Interactive 3D File Compiled Successfully.")
print(f"Saved file asset location: ./{output_filename}")
print("Double-click this file in your folder to view it in your browser!")
print("=" * 60)

# Optional: Also show it immediately on your screen right now
fig.show()