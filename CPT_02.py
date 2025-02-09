import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html

# =============================================================================
# 1. COMPUTATIONAL BACKEND: ADVANCED ORDERING INDEX CALCULATION
# =============================================================================

def compute_ordering_index(n: int, l: int, z: int) -> float:
    """
    Compute the ordering index "O" considering:
    - Principal quantum number (n)
    - Orbital quantum number (l)
    - Atomic number (Z)
    - Relativistic correction (proportional to Z^2/n^2 for heavy elements)
    - Electron-electron interactions (estimated using Z_eff and shielding)

    Formula (simplified for demonstration):
        O = n + (l / 10) + α * (Z**2 / n**2) - β * np.log(Z + 1)

    where:
        α = 0.001 for relativistic effect,
        β = 0.02 for shielding correction.
    """
    alpha = 0.001  # Relativistic correction factor
    beta = 0.02    # Electron shielding and repulsion correction

    relativistic_effect = alpha * (z ** 2 / n ** 2)
    shielding_correction = beta * np.log(z + 1)
    
    return n + (l / 10.0) + relativistic_effect - shielding_correction

# Define proper electron configurations for accurate period mapping
periods = {
    1: [2],            # Period 1 (s)
    2: [2, 6],         # Period 2 (s, p)
    3: [2, 6],         # Period 3 (s, p)
    4: [2, 6, 10],     # Period 4 (s, p, d)
    5: [2, 6, 10],     # Period 5 (s, p, d)
    6: [2, 6, 10, 14], # Period 6 (s, p, d, f)
    7: [2, 6, 10, 14]  # Period 7 (s, p, d, f)
}

block_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
color_map = {'s': 'blue', 'p': 'green', 'd': 'red', 'f': 'yellow'}

def generate_elements_data():
    """
    Generate a DataFrame with element quantum properties and computed ordering index O.
    """
    data = []
    atomic_number = 1  # Hydrogen starts at 1
    
    for period, sublevels in periods.items():
        l_values = [0, 1, 2, 3]  # s, p, d, f
        
        for i, max_electrons in enumerate(sublevels):
            l = l_values[i]  # Assign block based on level
            for _ in range(max_electrons):
                if atomic_number > 118:  # Limit to known elements
                    break
                O = compute_ordering_index(period, l, atomic_number)
                data.append({
                    'Element': f"E{atomic_number}",
                    'n': period,
                    'l': l,
                    'O': O,
                    'Block': block_map[l],
                    'Z': atomic_number  # Atomic number
                })
                atomic_number += 1
    
    return pd.DataFrame(data)

def compute_spiral_positions(df: pd.DataFrame, R: float = 1.0):
    """
    Compute 3D positions for elements along a spiral on a cylinder.
    """
    k = 2 * np.pi / 2  # Controls spiral density
    df['theta'] = k * df['O']
    df['x'] = R * np.cos(df['theta'])
    df['y'] = R * np.sin(df['theta'])
    df['z'] = df['O']
    df['color'] = df['Block'].apply(lambda b: color_map[b])
    return df

# =============================================================================
# 2. DATA PREPARATION
# =============================================================================

elements_df = generate_elements_data()
elements_df = compute_spiral_positions(elements_df)

# =============================================================================
# 3. 3D VISUALIZATION USING PLOTLY
# =============================================================================

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=elements_df['x'],
    y=elements_df['y'],
    z=elements_df['z'],
    mode='markers+text',
    marker=dict(size=5, color=elements_df['color'], opacity=0.8),
    text=elements_df['Element'],
    textposition="top center",
    hoverinfo='text'
))

fig.update_layout(
    title="Continuum Periodic Table (CPT) - Quantum Enhanced",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Ordering Index (O)"),
    margin=dict(l=0, r=0, b=0, t=50)
)

# =============================================================================
# 4. WEB APPLICATION USING DASH
# =============================================================================

app = Dash(__name__)
app.layout = html.Div(children=[
    html.H1("Continuum Periodic Table (CPT) - Enhanced Quantum Model"),
    html.P("A dynamic 3D visualization incorporating quantum corrections, relativistic effects, and statistical adjustments."),
    dcc.Graph(id='cpt-3d-graph', figure=fig),
    html.Footer("Developed with Python, Plotly, Dash", style={'textAlign': 'center', 'marginTop': '20px'})
])

# =============================================================================
# 5. RUN THE WEB APP
# =============================================================================

if __name__ == '__main__':
    app.run_server(debug=True)
