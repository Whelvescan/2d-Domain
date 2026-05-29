import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ==========================================
# 1. THE CORE NUMERICAL SOLVER ENGINE
# ==========================================
def solve_heat_2d(bc_types, bc_params, N=40, max_iter=2000, tol=1e-5, omega=1.5):
    """
    Solves 2D Laplace equation with custom boundary conditions using SOR.
    Domain is X ∈ [0, 1] and Y ∈ [0, 1].
    """
    # Grid spacing
    dx = 1.0 / (N - 1)
    
    # Initialize grid with a reasonable guess (average of Dirichlet boundaries if any)
    theta = np.zeros((N, N))
    
    # Helper to pull boundary inputs safely
    def get_bc_vals(edge):
        b_type = bc_types[edge]
        p = bc_params[edge]
        if b_type == "Dirichlet (Fixed Temp)":
            return p.get("theta", 0.0), 0.0, 0.0
        elif b_type == "Constant Flux":
            return 0.0, p.get("q", 0.0), 0.0
        elif b_type == "Convection":
            return 0.0, 0.0, (p.get("Bi", 1.0), p.get("theta_inf", 0.0))
        return 0.0, 0.0, 0.0 # Insulated default

    # Pre-calculate boundary parameters for speed
    # Order of indexing: Bottom (y=0, j=0), Top (y=1, j=N-1), Left (x=0, i=0), Right (x=1, i=N-1)
    
    # Gauss-Seidel Iteration Loop with SOR
    for iteration in range(max_iter):
        theta_old = theta.copy()
        
        # --- Update Interior Nodes ---
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                theta_new = 0.25 * (theta[i+1, j] + theta[i-1, j] + theta[i, j+1] + theta[i, j-1])
                theta[i, j] = (1 - omega) * theta[i, j] + omega * theta_new
                
        # --- Update Boundary Nodes Dynamically ---
        
        # 1. LEFT BOUNDARY (i = 0, for all j from 1 to N-2)
        b_type = bc_types["Left"]
        p = bc_params["Left"]
        for j in range(1, N - 1):
            if b_type == "Dirichlet (Fixed Temp)":
                theta[0, j] = p["theta"]
            elif b_type == "Insulated":
                theta[0, j] = (1-omega)*theta[0, j] + omega * ((2*theta[1, j] + theta[0, j+1] + theta[0, j-1]) / 4.0)
            elif b_type == "Constant Flux":
                # d_theta/dx = -q -> theta[-1,j] = theta[1,j] + 2*dx*q
                theta[0, j] = (1-omega)*theta[0, j] + omega * ((2*theta[1, j] + 2*dx*p["q"] + theta[0, j+1] + theta[0, j-1]) / 4.0)
            elif b_type == "Convection":
                # -d_theta/dx = Bi*(theta - theta_inf) -> virtual node derivation
                factor = 2 * dx * p["Bi"]
                theta[0, j] = (1-omega)*theta[0, j] + omega * ((2*theta[1, j] + factor*p["theta_inf"] + theta[0, j+1] + theta[0, j-1]) / (4.0 + factor))

        # 2. RIGHT BOUNDARY (i = N - 1, for all j from 1 to N-2)
        b_type = bc_types["Right"]
        p = bc_params["Right"]
        for j in range(1, N - 1):
            if b_type == "Dirichlet (Fixed Temp)":
                theta[N-1, j] = p["theta"]
            elif b_type == "Insulated":
                theta[N-1, j] = (1-omega)*theta[N-1, j] + omega * ((2*theta[N-2, j] + theta[N-1, j+1] + theta[N-1, j-1]) / 4.0)
            elif b_type == "Constant Flux":
                theta[N-1, j] = (1-omega)*theta[N-1, j] + omega * ((2*theta[N-2, j] - 2*dx*p["q"] + theta[N-1, j+1] + theta[N-1, j-1]) / 4.0)
            elif b_type == "Convection":
                factor = 2 * dx * p["Bi"]
                theta[N-1, j] = (1-omega)*theta[N-1, j] + omega * ((2*theta[N-2, j] + factor*p["theta_inf"] + theta[N-1, j+1] + theta[N-1, j-1]) / (4.0 + factor))

        # 3. BOTTOM BOUNDARY (j = 0, for all i from 1 to N-2)
        b_type = bc_types["Bottom"]
        p = bc_params["Bottom"]
        for i in range(1, N - 1):
            if b_type == "Dirichlet (Fixed Temp)":
                theta[i, 0] = p["theta"]
            elif b_type == "Insulated":
                theta[i, 0] = (1-omega)*theta[i, 0] + omega * ((theta[i+1, 0] + theta[i-1, 0] + 2*theta[i, 1]) / 4.0)
            elif b_type == "Constant Flux":
                theta[i, 0] = (1-omega)*theta[i, 0] + omega * ((theta[i+1, 0] + theta[i-1, 0] + 2*theta[i, 1] + 2*dx*p["q"]) / 4.0)
            elif b_type == "Convection":
                factor = 2 * dx * p["Bi"]
                theta[i, 0] = (1-omega)*theta[i, 0] + omega * ((theta[i+1, 0] + theta[i-1, 0] + 2*theta[i, 1] + factor*p["theta_inf"]) / (4.0 + factor))

        # 4. TOP BOUNDARY (j = N - 1, for all i from 1 to N-2)
        b_type = bc_types["Top"]
        p = bc_params["Top"]
        for i in range(1, N - 1):
            if b_type == "Dirichlet (Fixed Temp)":
                theta[i, N-1] = p["theta"]
            elif b_type == "Insulated":
                theta[i, N-1] = (1-omega)*theta[i, N-1] + omega * ((theta[i+1, N-1] + theta[i-1, N-1] + 2*theta[i, N-2]) / 4.0)
            elif b_type == "Constant Flux":
                theta[i, N-1] = (1-omega)*theta[i, N-1] + omega * ((theta[i+1, N-1] + theta[i-1, N-1] + 2*theta[i, N-2] - 2*dx*p["q"]) / 4.0)
            elif b_type == "Convection":
                factor = 2 * dx * p["Bi"]
                theta[i, N-1] = (1-omega)*theta[i, N-1] + omega * ((theta[i+1, N-1] + theta[i-1, N-1] + 2*theta[i, N-2] + factor*p["theta_inf"]) / (4.0 + factor))

        # --- Handle 4 Corners (Averaging adjacent boundaries for stability) ---
        theta[0, 0] = 0.5 * (theta[1, 0] + theta[0, 1])
        theta[N-1, 0] = 0.5 * (theta[N-2, 0] + theta[N-1, 1])
        theta[0, N-1] = 0.5 * (theta[1, N-1] + theta[0, N-2])
        theta[N-1, N-1] = 0.5 * (theta[N-2, N-1] + theta[N-1, N-2])

        # Check convergence criteria
        if np.max(np.abs(theta - theta_old)) < tol:
            break
            
    return theta

# ==========================================
# 2. STREAMLIT INTERACTIVE USER INTERFACE
# ==========================================
st.set_page_config(layout="wide")
st.title("Interactive 2D Dimensionless Heat Conduction Solver")
st.markdown("Configure your 2D domain boundaries on the sidebar, then compute and query parameters instantly.")

# Create container dictionaries for inputs
bc_types = {}
bc_params = {}

st.sidebar.header("🛠️ Boundary Condition Settings")
edges = ["Left", "Right", "Bottom", "Top"]
options = ["Dirichlet (Fixed Temp)", "Insulated", "Constant Flux", "Convection"]

# Populate Sidebar Dropdowns conditionally
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
    # 'Insulated' needs no extra numeric parameters

# Resolution Configuration
st.sidebar.markdown("---")
N_res = st.sidebar.slider("Grid Resolution (N x N)", 20, 60, 40, step=5)

# Trigger numerical execution
if 'computed_theta' not in st.session_state:
    st.session_state.computed_theta = None

if st.sidebar.button("🚀 Compute Simulation"):
    with st.spinner("Computing steady-state distribution..."):
        st.session_state.computed_theta = solve_heat_2d(bc_types, bc_params, N=N_res)
    st.success("Computation Complete!")

# Main Area Visualizations and Spot Queries
if st.session_state.computed_theta is not None:
    theta_field = st.session_state.computed_theta
    N_current = theta_field.shape[0]
    
    # 1. Output Important Quantitative Metrics
    mid_idx = N_current // 2
    mid_theta = theta_field[mid_idx, mid_idx]
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("📊 Analytical Metrics")
        st.metric(label="Dimensionless Midpoint Temperature (θ_mid)", value=f"{mid_theta:.4f}")
        
        st.markdown("---")
        st.subheader("🎯 Query Custom Location")
        st.write("Input coordinates within the normalized domain $X \\in [0,1], Y \\in [0,1]$:")
        
        # User dynamic location input
        query_x = st.number_input("Enter X coordinate", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        query_y = st.number_input("Enter Y coordinate", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        # Map user normalized inputs back to discrete indices
        idx_x = int(round(query_x * (N_current - 1)))
        idx_y = int(round(query_y * (N_current - 1)))
        
        queried_theta = theta_field[idx_x, idx_y]
        st.metric(label=f"Temperature θ at ({query_x:.2f}, {query_y:.2f})", value=f"{queried_theta:.4f}")

    with col2:
        st.header("🌡️ Temperature Distribution (θ Grid)")
        
        # Generate Contourf Plot using Matplotlib
        fig, ax = plt.subplots(figsize=(6, 5))
        X, Y = np.meshgrid(np.linspace(0, 1, N_current), np.linspace(0, 1, N_current))
        
        # Transpose matrix field to match geometric X and Y orientations correctly
        cp = ax.contourf(X, Y, theta_field.T, levels=20, cmap='inferno')
        fig.colorbar(cp, ax=ax, label='Dimensionless Temperature (θ)')
        ax.set_xlabel('Dimensionless X')
        ax.set_ylabel('Dimensionless Y')
        ax.set_title('Steady State Contour Map')
        
        st.pyplot(fig)
else:
    st.info("👈 Click the 'Compute Simulation' button in the sidebar to visualize and calculate the temperature profiles.")