import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ==========================================
# 1. HIGH-SPEED RED-BLACK VECTORIZED ENGINE
# ==========================================
def run_steady_solver(theta, bc_types, bc_params_matrix, N, max_iter, tol, omega, dx):
    """
    Red-Black Gauss-Seidel Over-Relaxation Solver.
    Splits the grid into a checkerboard pattern to vectorize updates safely.
    Eliminates loops, prevents NaN errors, and works flawlessly without Numba.
    """
    # Create index coordinates for the interior domain (N-2 x N-2)
    i_idx, j_idx = np.ogrid[1:N-1, 1:N-1]
    
    # Define checkerboard masks for the interior nodes
    # Red nodes: (i + j) is even | Black nodes: (i + j) is odd
    red_mask = ((i_idx + j_idx) % 2 == 0)
    black_mask = ((i_idx + j_idx) % 2 == 1)
    
    for iteration in range(max_iter):
        old_theta = theta.copy()
        
        # --- PHASE 1: Update Red Interior Nodes ---
        # Calculate finite difference targets for all interior points
        target = 0.25 * (theta[2:, 1:-1] + theta[:-2, 1:-1] + theta[1:-1, 2:] + theta[1:-1, :-2])
        # Apply Successive Over-Relaxation selectively to Red nodes
        theta[1:-1, 1:-1] = np.where(red_mask, (1.0 - omega) * theta[1:-1, 1:-1] + omega * target, theta[1:-1, 1:-1])
        
        # --- PHASE 2: Update Black Interior Nodes (Using newly updated Red nodes) ---
        target = 0.25 * (theta[2:, 1:-1] + theta[:-2, 1:-1] + theta[1:-1, 2:] + theta[1:-1, :-2])
        # Apply Successive Over-Relaxation selectively to Black nodes
        theta[1:-1, 1:-1] = np.where(black_mask, (1.0 - omega) * theta[1:-1, 1:-1] + omega * target, theta[1:-1, 1:-1])
                    
        # --- PHASE 3: Update Boundary Node Slices Safely ---
        
        # LEFT BOUNDARY (i = 0)
        b_type = bc_types[0]
        if b_type == 0:   
            theta[0, 1:-1] = bc_params_matrix[0, 0]
        elif b_type == 1: 
            theta[0, 1:-1] = (1.0-omega)*theta[0, 1:-1] + omega * ((2.0*theta[1, 1:-1] + theta[0, 2:] + theta[0, :-2]) / 4.0)
        elif b_type == 2: 
            theta[0, 1:-1] = (1.0-omega)*theta[0, 1:-1] + omega * ((2.0*theta[1, 1:-1] + 2.0*dx*bc_params_matrix[0, 1] + theta[0, 2:] + theta[0, :-2]) / 4.0)
        elif b_type == 3: 
            factor = 2.0 * dx * bc_params_matrix[0, 2]
            theta[0, 1:-1] = (1.0-omega)*theta[0, 1:-1] + omega * ((2.0*theta[1, 1:-1] + factor*bc_params_matrix[0, 3] + theta[0, 2:] + theta[0, :-2]) / (4.0 + factor))

        # RIGHT BOUNDARY (i = N - 1)
        b_type = bc_types[1]
        if b_type == 0:
            theta[N-1, 1:-1] = bc_params_matrix[1, 0]
        elif b_type == 1:
            theta[N-1, 1:-1] = (1.0-omega)*theta[N-1, 1:-1] + omega * ((2.0*theta[N-2, 1:-1] + theta[N-1, 2:] + theta[N-1, :-2]) / 4.0)
        elif b_type == 2:
            theta[N-1, 1:-1] = (1.0-omega)*theta[N-1, 1:-1] + omega * ((2.0*theta[N-2, 1:-1] + 2.0*dx*bc_params_matrix[1, 1] + theta[N-1, 2:] + theta[N-1, :-2]) / 4.0)
        elif b_type == 3:
            factor = 2.0 * dx * bc_params_matrix[1, 2]
            theta[N-1, 1:-1] = (1.0-omega)*theta[N-1, 1:-1] + omega * ((2.0*theta[N-2, 1:-1] + factor*bc_params_matrix[1, 3] + theta[N-1, 2:] + theta[N-1, :-2]) / (4.0 + factor))

        # BOTTOM BOUNDARY (j = 0)
        b_type = bc_types[2]
        if b_type == 0:
            theta[1:-1, 0] = bc_params_matrix[2, 0]
        elif b_type == 1:
            theta[1:-1, 0] = (1.0-omega)*theta[1:-1, 0] + omega * ((theta[2:, 0] + theta[:-2, 0] + 2.0*theta[1:-1, 1]) / 4.0)
        elif b_type == 2:
            theta[1:-1, 0] = (1.0-omega)*theta[1:-1, 0] + omega * ((theta[2:, 0] + theta[:-2, 0] + 2.0*theta[1:-1, 1] + 2.0*dx*bc_params_matrix[2, 1]) / 4.0)
        elif b_type == 3:
            factor = 2.0 * dx * bc_params_matrix[2, 2]
            theta[1:-1, 0] = (1.0-omega)*theta[1:-1, 0] + omega * ((theta[2:, 0] + theta[:-2, 0] + 2.0*theta[1:-1, 1] + factor*bc_params_matrix[2, 3]) / (4.0 + factor))

        # TOP BOUNDARY (j = N - 1)
        b_type = bc_types[3]
        if b_type == 0:
            theta[1:-1, N-1] = bc_params_matrix[3, 0]
        elif b_type == 1:
            theta[1:-1, N-1] = (1.0-omega)*theta[1:-1, N-1] + omega * ((theta[2:, N-1] + theta[:-2, N-1] + 2.0*theta[1:-1, N-2]) / 4.0)
        elif b_type == 2:
            theta[1:-1, N-1] = (1.0-omega)*theta[1:-1, N-1] + omega * ((theta[2:, N-1] + theta[:-2, N-1] + 2.0*theta[1:-1, N-2] + 2.0*dx*bc_params_matrix[3, 1]) / 4.0)
        elif b_type == 3:
            factor = 2.0 * dx * bc_params_matrix[3, 2]
            theta[1:-1, N-1] = (1.0-omega)*theta[1:-1, N-1] + omega * ((theta[2:, N-1] + theta[:-2, N-1] + 2.0*theta[1:-1, N-2] + factor*bc_params_matrix[3, 3]) / (4.0 + factor))
            
        # --- PHASE 4: Handle 4 Corners ---
        theta[0, 0] = 0.5 * (theta[1, 0] + theta[0, 1])
        theta[N-1, 0] = 0.5 * (theta[N-2, 0] + theta[N-1, 1])
        theta[0, N-1] = 0.5 * (theta[1, N-1] + theta[0, N-2])
        theta[N-1, N-1] = 0.5 * (theta[N-2, N-1] + theta[N-1, N-2])

        # Evaluate absolute global error convergence across array criteria
        if np.max(np.abs(theta - old_theta)) < tol:
            break
            
    return theta

def solve_heat_2d(bc_types, bc_params, N=200, max_iter=100000, tol=1e-7, omega=1.95):
    dx = 1.0 / (N - 1)
    theta = np.full((N, N), 0.5)
    
    edge_list = ["Left", "Right", "Bottom", "Top"]
    opt_list = ["Dirichlet (Fixed Temp)", "Insulated", "Constant Flux", "Convection"]
    
    bc_types_idx = np.zeros(4, dtype=np.int32)
    bc_values_matrix = np.zeros((4, 4))
    
    for idx, edge in enumerate(edge_list):
        bc_types_idx[idx] = opt_list.index(bc_types[edge])
        p = bc_params[edge]
        bc_values_matrix[idx, 0] = p.get("theta", 0.0)
        bc_values_matrix[idx, 1] = p.get("q", 0.0)
        bc_values_matrix[idx, 2] = p.get("Bi", 1.0)
        bc_values_matrix[idx, 3] = p.get("theta_inf", 0.0)
        
        if bc_types_idx[idx] == 0:
            if idx == 0: theta[0, :] = bc_values_matrix[idx, 0]         
            elif idx == 1: theta[N-1, :] = bc_values_matrix[idx, 0]     
            elif idx == 2: theta[:, 0] = bc_values_matrix[idx, 0]       
            elif idx == 3: theta[:, N-1] = bc_values_matrix[idx, 0]     
        
    return run_steady_solver(theta, bc_types_idx, bc_values_matrix, N, max_iter, tol, omega, dx)

# ==========================================
# 2. STREAMLIT INTERACTIVE USER INTERFACE
# ==========================================
st.set_page_config(layout="wide")
st.title("Interactive 2D Dimensionless Heat Conduction Solver")
st.markdown("Configure your 2D domain boundaries on the sidebar, then compute and view the interactive 3D profile.")

bc_types = {}
bc_params = {}

st.sidebar.header("🛠️ Boundary Condition Settings")
edges = ["Left", "Right", "Bottom", "Top"]
options = ["Dirichlet (Fixed Temp)", "Insulated", "Constant Flux", "Convection"]

for edge in edges:
    st.sidebar.subheader(f"📍 {edge} Boundary")
    choice = st.sidebar.selectbox(f"Type for {edge}", options, key=f"type_{edge}")
    bc_types[edge] = choice
    bc_params[edge] = {}
    
    if choice == "Dirichlet (Fixed Temp)":
        bc_params[edge]["theta"] = st.sidebar.slider(f"θ value ({edge})", 0.0, 1.0, 0.5 if edge=="Top" else 0.0, step=0.05)
    elif choice == "Constant Flux":
        bc_params[edge]["q"] = st.sidebar.number_input(f"Dimensionless Flux q* ({edge})", value=1.0, step=0.1)
    elif choice == "Convection":
        bc_params[edge]["Bi"] = st.sidebar.number_input(f"Biot Number Bi ({edge})", value=1.0, min_value=0.01, step=0.5)
        bc_params[edge]["theta_inf"] = st.sidebar.slider(f"Ambient θ_inf ({edge})", 0.0, 1.0, 0.0, step=0.05)

st.sidebar.markdown("---")
N_res = st.sidebar.slider("Grid Resolution (N x N)", 20, 400, 100, step=10)

if 'computed_theta' not in st.session_state:
    st.session_state.computed_theta = None

if st.sidebar.button("🚀 Compute Simulation"):
    with st.spinner("Computing steady-state thermal landscape..."):
        st.session_state.computed_theta = solve_heat_2d(bc_types, bc_params, N=N_res)
    st.success("Computation Complete!")

# Main Visual Display Screen Area
if st.session_state.computed_theta is not None:
    theta_field = st.session_state.computed_theta
    N_current = theta_field.shape[0]
    
    mid_idx = N_current // 2
    mid_theta = theta_field[mid_idx, mid_idx]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📊 Analytical Metrics")
        st.metric(label="Dimensionless Midpoint Temperature (θ_mid)", value=f"{mid_theta:.4f}")

        dx_val = 1.0 / (N_current - 1)
        st.metric(label="Grid Spacing (Δx = Δy)", value=f"{dx_val:.5f}")
        
        st.markdown("---")
        st.subheader("🎯 Query Custom Location")
        st.write("Input coordinates within the domain $X \\in [0,1], Y \\in [0,1]$:")
        
        query_x = st.number_input("Enter X coordinate", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        query_y = st.number_input("Enter Y coordinate", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        idx_x = int(round(query_x * (N_current - 1)))
        idx_y = int(round(query_y * (N_current - 1)))
        
        queried_theta = theta_field[idx_x, idx_y]
        st.metric(label=f"Temperature θ at ({query_x:.2f}, {query_y:.2f})", value=f"{queried_theta:.4f}")

    with col2:
        st.header("🌡️ Interactive 3D Thermal Landscape")
        
        x_line = np.linspace(0, 1, N_current)
        y_line = np.linspace(0, 1, N_current)
        
        fig = go.Figure(data=[go.Surface(
            z=theta_field.T,
            x=x_line,
            y=y_line,
            colorscale='inferno',
            colorbar=dict(title='θ Value'),
            contours=dict(
                x=dict(show=False, start=0, end=1, size=dx_val, color="rgba(255, 255, 255, 0.15)"),
                y=dict(show=False, start=0, end=1, size=dx_val, color="rgba(255, 255, 255, 0.15)"),
                z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
            ),
            hidesurface=False
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title='Dimensionless X',
                yaxis_title='Dimensionless Y',
                zaxis_title='Temperature (θ)',
                aspectratio=dict(x=1, y=1, z=0.7),
                xaxis=dict(showgrid=False, showbackground=False, zeroline=False),
                yaxis=dict(showgrid=False, showbackground=False, zeroline=False),
                zaxis=dict(showgrid=False, showbackground=False, zeroline=False)
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 3. 2D CENTERLINE PROFILES
    # ==========================================
    st.markdown("---")
    st.header("📈 Centerline Temperature Profiles")
    st.write("2D slices taken through the exact middle of the plate.")
    
    horizontal_profile = theta_field[:, mid_idx]
    vertical_profile = theta_field[mid_idx, :]   
    
    fig2d = go.Figure()
    
    fig2d.add_trace(go.Scatter(
        x=x_line, 
        y=horizontal_profile, 
        mode='lines', 
        name='Horizontal Centerline (Y = 0.5)',
        line=dict(color='cyan', width=3)
    ))
    
    fig2d.add_trace(go.Scatter(
        x=y_line, 
        y=vertical_profile, 
        mode='lines', 
        name='Vertical Centerline (X = 0.5)',
        line=dict(color='magenta', width=3)
    ))
    
    fig2d.update_layout(
        xaxis_title="Dimensionless Distance (X or Y)",
        yaxis_title="Temperature (θ)",
        hovermode="x unified",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig2d, use_container_width=True)
else:
    st.info("👈 Click the 'Compute Simulation' button in the sidebar to build your interactive 3D thermal profile map.")