import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context

# ==========================================
# 1. ANALYTICAL EQUATIONS ENGINE
# ==========================================
T_max, T_min, k, L = 80.0, 27.0, 50.0, 1.0
Nz, Ny = 100, 100
z_arr = np.linspace(0, 1, Nz)
y_arr = np.linspace(0, 1, Ny)
Z, Y = np.meshgrid(z_arr, y_arr, indexing='ij')

theta = np.zeros((Nz, Ny))
flux_scale = -(k * (T_max - T_min)) / L
max_terms = 100  

# Pre-calculate global temperature landscape
for i in range(max_terms):
    n = 2 * i + 1  
    Cn = 4 / (n * np.pi * np.sinh(n * np.pi))
    theta += Cn * np.sin(n * np.pi * Z) * np.sinh(n * np.pi * (1 - Y))
theta[:, 0] = 1.0
T_physical = theta * (T_max - T_min) + T_min

def compute_point_metrics(qz_query, qy_query):
    """Evaluates temperature and directional heat flux components at a specific point."""
    t_sum, qz_sum, qy_sum = 0.0, 0.0, 0.0
    for i in range(max_terms):
        n = 2 * i + 1
        Cn = 4 / (n * np.pi * np.sinh(n * np.pi))
        t_sum += Cn * np.sin(n * np.pi * qz_query) * np.sinh(n * np.pi * (1 - qy_query))
        qz_sum += Cn * (n * np.pi * np.cos(n * np.pi * qz_query)) * np.sinh(n * np.pi * (1 - qy_query))
        qy_sum += Cn * np.sin(n * np.pi * qz_query) * (-n * np.pi * np.cosh(n * np.pi * (1 - qy_query)))
    
    if qy_query == 0.0: t_sum = 1.0
    return t_sum * (T_max - T_min) + T_min, flux_scale * qz_sum, flux_scale * qy_sum

# ==========================================
# 2. DASH APPLICATION INTERFACE LAYOUT
# ==========================================
app = Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f4f6f9'}, children=[
    html.H1("Steady-State Thermal Analysis Dashboard", style={'textAlign': 'center', 'color': '#1e293b', 'marginBottom': '30px'}),
    
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'gap': '30px', 'justifyContent': 'center'}, children=[
        
        # LEFT CONTROL PANEL PANEL
        html.Div(style={'width': '350px', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}, children=[
            html.H3("Interactive Probe Position", style={'marginTop': '0', 'color': '#334155'}),
            html.P("Input coordinates between 0.0 and 1.0:", style={'color': '#64748b', 'fontSize': '14px'}),
            
            html.Div(style={'marginBottom': '15px'}, children=[
                html.Label("Dimensionless Coordinate Z:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Input(id='input-z', type='number', value=0.5, min=0, max=1, step=0.01, style={'width': '100%', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #cbd5e1'})
            ]),
            
            html.Div(style={'marginBottom': '25px'}, children=[
                html.Label("Dimensionless Coordinate Y:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Input(id='input-y', type='number', value=0.5, min=0, max=1, step=0.01, style={'width': '100%', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #cbd5e1'})
            ]),
            
            html.Button("Calculate Flux", id='btn-calc', n_clicks=0, style={'width': '100%', 'padding': '12px', 'backgroundColor': '#2563eb', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '15px'}),
            
            html.Hr(style={'margin': '25px 0', 'border': '0', 'borderTop': '1px solid #e2e8f0'}),
            
            # RESULTS CONTAINER BOX
            html.H4("Calculated Point Properties:", style={'color': '#334155', 'marginBottom': '10px'}),
            html.Div(id='output-results-panel', style={'backgroundColor': '#f8fafc', 'padding': '15px', 'borderRadius': '5px', 'borderLeft': '4px solid #2563eb', 'fontSize': '14px', 'lineHeight': '1.6'})
        ]),
        
        # RIGHT GRAPH DISPLAY PANEL
        html.Div(style={'flex': '1', 'maxWidth': '850px', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}, children=[
            dcc.Graph(id='main-3d-surface', style={'height': '600px'})
        ])
    ])
])

# ==========================================
# 3. INTERACTIVE REACTIVE GRAPH CALLBACK (FIXED)
# ==========================================
@app.callback(
    [Output('main-3d-surface', 'figure'),
     Output('output-results-panel', 'children')],
    [Input('btn-calc', 'n_clicks')],
    [State('input-z', 'value'), 
     State('input-y', 'value')]
)
def update_dashboard(n_clicks, val_z, val_y):
    # Safe boundary fallbacks if boxes are cleared or empty
    q_z = 0.5 if val_z is None or not (0 <= val_z <= 1) else val_z
    q_y = 0.5 if val_y is None or not (0 <= val_y <= 1) else val_y
    
    # Calculate local physics properties
    temp, qz, qy = compute_point_metrics(q_z, q_y)
    mag = np.sqrt(qz**2 + qy**2)
    
    # Recompile base 3D canvas topology layout trace
    base_surface = go.Surface(
        z=T_physical.T, x=z_arr, y=y_arr,
        colorscale='Plasma', opacity=0.85, showscale=True,
        colorbar=dict(title='Temp (°C)', len=0.6, y=0.5)
    )
    
    # Place dynamic interactive cursor right on top of the calculated coordinate height
    probe_dot = go.Scatter3d(
        x=[q_z], y=[q_y], z=[temp],
        mode='markers',
        marker=dict(size=10, color='lime', symbol='diamond', line=dict(color='black', width=3)),
        hoverinfo='text',
        text=f"Position: ({q_z}, {q_y})<br>Flux Magnitude: {mag:.1f} W/m²"
    )
    
    fig = go.Figure(data=[base_surface, probe_dot])
    fig.update_layout(
        title='3D Thermal Field Landscape Engine',
        scene=dict(
            xaxis_title='z-axis', yaxis_title='y-axis', zaxis_title='Temp (°C)',
            aspectratio=dict(x=1, y=1, z=0.6)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Build HTML display formatting block text strings
    text_results = [
        html.Div([html.B("Local Temperature: "), f"{temp:.2f} °C"]),
        html.Div([html.B("qz (Horizontal Flux): "), f"{qz:.2f} W/m²"]),
        html.Div([html.B("qy (Vertical Flux): "), f"{qy:.2f} W/m²"]),
        html.Div([html.B("Net Vector Intensity: "), html.Span(f"{mag:.2f} W/m²", style={'color': '#2563eb', 'fontWeight': 'bold'})])
    ]
    
    return fig, text_results