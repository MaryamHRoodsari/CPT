"""
CPT Visualization Web App
-------------------------
This application computes a continuous ordering index “O” for a simulated set
of elements (based on simplified quantum numbers) and maps them onto a spiral
structure. The 3D visualization is rendered using Plotly and served via Dash.

Requirements:
    - numpy
    - pandas
    - plotly
    - dash

Run this script (e.g., python cpt_app.py) and navigate to the indicated local URL.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html

# =============================================================================
# 1. COMPUTATIONAL BACKEND: GENERATE ELEMENT DATA AND SPIRAL COORDINATES
# =============================================================================

def compute_ordering_index(n: int, l: int) -> float:
    """
    Compute the ordering index "O" for an element.
    
    This simplified function uses:
        O = n + (l / 10)
    where n is the principal quantum number and l is the orbital quantum number.
    
    In a more refined model, additional corrections (electron-electron interactions,
    relativistic effects, etc.) and statistical adjustments would be incorporated.
    """
    return n + l / 10.0

def generate_elements_data(num_elements: int = 118) -> pd.DataFrame:
    """
    Generate a DataFrame of simulated elements with quantum numbers.
    
    For simplicity, we simulate periods with pre-defined lengths:
      - Period lengths (approximate for demonstration): [2, 8, 8, 18, 18, 32, 32]
      
    For each period, we assign:
      - The first element as belonging to the s-block (l = 0)
      - The remaining elements as p-block (l = 1)
      
    (Note: A fully robust model would include d and f blocks and more accurate period
     distributions based on actual electron configurations.)
    """
    period_lengths = [2, 8, 8, 18, 18, 32, 32]
    data = []
    element_counter = 1
    
    for period, length in enumerate(period_lengths, start=1):
        if element_counter > num_elements:
            break

        # s-block element (first element of the period)
        n = period
        l = 0  # s-block
        O = compute_ordering_index(n, l)
        data.append({
            'Element': f"E{element_counter}",
            'n': n,
            'l': l,
            'O': O,
            'Block': 's'
        })
        element_counter += 1

        # p-block elements for the remaining positions in this period
        for i in range(1, length):
            if element_counter > num_elements:
                break
            n = period
            l = 1  # p-block (for demonstration)
            O = compute_ordering_index(n, l)
            data.append({
                'Element': f"E{element_counter}",
                'n': n,
                'l': l,
                'O': O,
                'Block': 'p'
            })
            element_counter += 1

    return pd.DataFrame(data)

def compute_spiral_positions(df: pd.DataFrame, R: float = 1.0) -> pd.DataFrame:
    """
    Compute 3D coordinates (x, y, z) for each element on a spiral wrapped around
    a cylinder. The vertical coordinate (z) is set to the ordering index "O".
    
    The angular coordinate theta is proportional to O, with a scaling factor k.
    """
    # Scaling: set k such that each unit of O gives a full circle if desired.
    # For a smooth spiral, you might set k < 2*pi so that the spiral wraps gradually.
    k = 2 * np.pi / 1.0  # adjust as needed
    x_coords, y_coords, z_coords = [], [], []
    
    for O in df['O']:
        theta = k * O  # angle in radians
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        z = O  # vertical axis represents the ordering index
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        
    df['x'] = x_coords
    df['y'] = y_coords
    df['z'] = z_coords
    return df

# =============================================================================
# 2. DATA PREPARATION
# =============================================================================

# Generate simulated element data (using 118 elements as a demonstration)
elements_df = generate_elements_data(num_elements=118)
elements_df = compute_spiral_positions(elements_df)

# For visualization, assign colors based on the block type.
# For simplicity, we map:
#   s-block: blue, p-block: green, (others default to gray)
block_colors = {'s': 'blue', 'p': 'green', 'd': 'orange', 'f': 'red'}
elements_df['color'] = elements_df['Block'].apply(lambda b: block_colors.get(b, 'gray'))

# =============================================================================
# 3. 3D VISUALIZATION WITH PLOTLY
# =============================================================================

# Create a 3D scatter plot of the elements on the spiral.
fig = go.Figure(data=[
    go.Scatter3d(
        x=elements_df['x'],
        y=elements_df['y'],
        z=elements_df['z'],
        mode='markers+text',
        marker=dict(
            size=5,
            color=elements_df['color'],
            opacity=0.8
        ),
        text=elements_df['Element'],
        textposition="top center",
        hoverinfo='text'
    )
])

# Update the layout with titles and axis labels.
fig.update_layout(
    title="Continuum Periodic Table (CPT) - Spiral Structure",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Ordering Index (O)"
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

# =============================================================================
# 4. WEB APPLICATION USING DASH
# =============================================================================

app = Dash(__name__)
app.layout = html.Div(children=[
    html.H1("Continuum Periodic Table (CPT) - 3D Visualization"),
    html.P("A dynamic, spiral representation of elements based on quantum properties, "
           "group theory, and statistical analysis."),
    dcc.Graph(
        id='cpt-3d-graph',
        figure=fig
    ),
    html.Footer("Developed using Python, Plotly, and Dash", style={'textAlign': 'center', 'marginTop': '20px'})
])

# =============================================================================
# 5. RUN THE WEB APP
# =============================================================================

if __name__ == '__main__':
    # Run the Dash app (by default on http://127.0.0.1:8050/)
    app.run_server(debug=True)
