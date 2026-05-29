import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display, clear_output

# 1. COMPUTATIONAL ENGINE (Finite Difference Method)
def solve_heat_equation(left_type, left_val, right_type, right_val, 
                        bottom_type, bottom_val, top_type, top_val, max_tau=0.1):
    Nx, Ny = 25, 25
    dx, dy = 1.0 / (Nx - 1), 1.0 / (Ny - 1)
    
    # Explicit stability check (Fourier condition)
    dt = 0.2 * min(dx, dy)**2
    steps = int(max_tau / dt)
    
    # Initialize grid (Internal initial theta = 0.0)
    theta = np.zeros((Ny, Nx))
    
    for _ in range(steps):
        theta_old = theta.copy()
        
        # Internal nodes calculation
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                d2x = (theta_old[i, j+1] - 2*theta_old[i, j] + theta_old[i, j-1]) / dx**2
                d2y = (theta_old[i+1, j] - 2*theta_old[i, j] + theta_old[i-1, j]) / dy**2
                theta[i, j] = theta_old[i, j] + dt * (d2x + d2y)
        
        # LEFT BOUNDARY (X = 0, j = 0)
        if left_type == 'Dirichlet':
            theta[:, 0] = left_val
        elif left_type == 'Homogeneous Neumann':
            theta[:, 0] = theta[:, 1]
        elif left_type == 'Non-Homogeneous Neumann':
            theta[:, 0] = theta[:, 1] + left_val * dx
        elif left_type == 'Robin':
            theta[:, 0] = theta[:, 1] / (1.0 + left_val * dx)

        # RIGHT BOUNDARY (X = 1, j = Nx-1)
        if right_type == 'Dirichlet':
            theta[:, -1] = right_val
        elif right_type == 'Homogeneous Neumann':
            theta[:, -1] = theta[:, -2]
        elif right_type == 'Non-Homogeneous Neumann':
            theta[:, -1] = theta[:, -2] - right_val * dx
        elif right_type == 'Robin':
            theta[:, -1] = theta[:, -2] / (1.0 + right_val * dx)

        # BOTTOM BOUNDARY (Y = 0, i = 0)
        if bottom_type == 'Dirichlet':
            theta[0, :] = bottom_val
        elif bottom_type == 'Homogeneous Neumann':
            theta[0, :] = theta[1, :]
        elif bottom_type == 'Non-Homogeneous Neumann':
            theta[0, :] = theta[1, :] + bottom_val * dy
        elif bottom_type == 'Robin':
            theta[0, :] = theta[1, :] / (1.0 + bottom_val * dy)

        # TOP BOUNDARY (Y = 1, i = Ny-1)
        if top_type == 'Dirichlet':
            theta[-1, :] = top_val
        elif top_type == 'Homogeneous Neumann':
            theta[-1, :] = theta[-2, :]
        elif top_type == 'Non-Homogeneous Neumann':
            theta[-1, :] = theta[-2, :] - top_val * dy
        elif top_type == 'Robin':
            theta[-1, :] = theta[-2, :] / (1.0 + top_val * dy)
            
    return theta, Nx, Ny

# 2. INTERFACE DEFINITIONS
bc_options = ['Dirichlet', 'Homogeneous Neumann', 'Non-Homogeneous Neumann', 'Robin']

# Generate custom UI controls for each side
def create_side_controls(side_name):
    dropdown = widgets.Dropdown(options=bc_options, value='Dirichlet', description=f'{side_name} BC:')
    value_input = widgets.FloatText(value=1.0, description='Value / Bi:', disabled=False)
    
    # Hide input text window if Homogeneous Neumann is picked
    def handle_bc_change(change):
        if change['new'] == 'Homogeneous Neumann':
            value_input.layout.display = 'none'
        else:
            value_input.layout.display = 'block'
            if change['new'] == 'Robin':
                value_input.description = 'Biot No (Bi):'
            elif change['new'] == 'Non-Homogeneous Neumann':
                value_input.description = 'Flux (q*):'
            else:
                value_input.description = 'Temp (theta):'
                
    dropdown.observe(handle_bc_change, names='value')
    return dropdown, value_input

left_drop, left_val = create_side_controls('Left')
right_drop, right_val = create_side_controls('Right')
bottom_drop, bottom_val = create_side_controls('Bottom')
top_drop, top_val = create_side_controls('Top')

# Simulation & Point Tracking Controls
tau_slider = widgets.FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01, description='Time (tau):')
x_query = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description='Query X:')
y_query = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description='Query Y:')
run_btn = widgets.Button(description='Generate Graphs', button_style='primary')

output_area = widgets.Output()

# 3. GRAPH PLOTTING AND POINT INTERPOLATION
def update_dashboard(b):
    with output_area:
        clear_output(wait=True)
        
        # Calculate solution matrix
        theta, Nx, Ny = solve_heat_equation(
            left_drop.value, left_val.value, right_drop.value, right_val.value,
            bottom_drop.value, bottom_val.value, top_drop.value, top_val.value, tau_slider.value
        )
        
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y)
        
        # Pinpoint query location indices
        i_x = int(x_query.value * (Nx - 1))
        i_y = int(y_query.value * (Ny - 1))
        queried_theta = theta[i_y, i_x]
        
        # Display the specific query value boldly at the top
        display(widgets.HTML(f"<h3>📍 Value at targeted point: theta({x_query.value}, {y_query.value}) = <b>{queried_theta:.4f}</b></h3>"))
        
        # Initialize Figures
        fig = plt.figure(figsize=(15, 5))
        
        # Subplot 1: 2D Cross Sections
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(x, theta[i_y, :], 'r-', label=f'X Profile (at Y={y_query.value})', linewidth=2)
        ax1.plot(y, theta[:, i_x], 'b--', label=f'Y Profile (at X={x_query.value})', linewidth=2)
        ax1.plot(x_query.value, queried_theta, 'go', markersize=10, label='Queried Point')
        ax1.set_title("2D Cross-Sectional Dimensionless Temperature Profiles")
        ax1.set_xlabel("Dimensionless Position (X or Y)")
        ax1.set_ylabel("Dimensionless Temperature (\\theta)")
        ax1.grid(True)
        ax1.legend()
        
        # Subplot 2: 3D Surface Mesh
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax2.plot_surface(X, Y, theta, cmap='plasma', edgecolor='none', alpha=0.8)
        ax2.scatter(x_query.value, y_query.value, queried_theta, color='green', s=100, depthshade=False, label='Target Point')
        ax2.set_title("3D Thermal Distribution Mesh")
        ax2.set_xlabel("X Boundary")
        ax2.set_ylabel("Y Boundary")
        ax2.set_zlabel("\\theta Profile")
        fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Scale (\\theta)')
        
        plt.tight_layout()
        plt.show()

run_btn.on_click(update_dashboard)

# Arrange layouts cleanly
input_panel = widgets.VBox([
    widgets.HTML("<h4>Step 1: Assign Boundary Conditions for Each Wall</h4>"),
    widgets.HBox([widgets.VBox([left_drop, left_val]), widgets.VBox([right_drop, right_val])]),
    widgets.HBox([widgets.VBox([bottom_drop, bottom_val]), widgets.VBox([top_drop, top_val])]),
    widgets.HTML("<h4>Step 2: Simulation Execution Time & Target Query Point</h4>"),
    widgets.HBox([tau_slider, x_query, y_query]),
    widgets.Box(layout=widgets.Layout(height='15px')),  # Replaced widgets.PADDING with a clean spacer box
    run_btn
])

# Deploy application
display(input_panel, output_area)