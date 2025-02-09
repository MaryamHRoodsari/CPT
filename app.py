import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# =============================================================================
# 1. COMPUTATIONAL BACKEND
# =============================================================================

def compute_ordering_index(n: int, l: int, z: int) -> float:
    """
    Computes the ordering index "O" using a simplified model that includes:
      - Principal quantum number (n)
      - Orbital quantum number (l)
      - Atomic number (Z)
      - Relativistic effect: alpha * (Z^2 / n^2)
      - Electron-electron shielding: beta * log(Z+1)
    """
    alpha = 0.001  # Relativistic correction factor
    beta = 0.02    # Shielding/repulsion correction factor

    relativistic_effect = alpha * (z ** 2 / n ** 2)
    shielding_correction = beta * np.log(z + 1)

    O = n + (l / 10.0) + relativistic_effect - shielding_correction
    return O

# Each period's electron sub-shell capacities (simplified distribution)
periods = {
    1: [2],          # Period 1: s
    2: [2, 6],       # Period 2: s, p
    3: [2, 6],       # Period 3: s, p
    4: [2, 6, 10],   # Period 4: s, p, d
    5: [2, 6, 10],   # Period 5: s, p, d
    6: [2, 6, 10, 14],  # Period 6: s, p, d, f
    7: [2, 6, 10, 14]   # Period 7: s, p, d, f
}

# Map orbital quantum number l to block name
block_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}

# Colors for standard blocks and bridging elements
color_map = {
    's': 'blue',
    'p': 'green',
    'd': 'red',
    'f': 'orange',
    'Bridging': 'purple'
}

# Hypothetical bridging elements with approximate (n, l) quantum numbers
bridging_elements = {
    'X₁': (4, 1),
    'X₂': (5, 2),
    'Y₁': (6, 3)
}

def generate_elements_data() -> pd.DataFrame:
    """
    Generates a DataFrame of elements (real + bridging).
      - Real elements (Z=1..118) are distributed according to 'periods'.
      - Bridging elements are added with custom quantum numbers.
    """
    data = []
    atomic_number = 1

    # Fill real elements up to Z=118
    for period, sublevels in periods.items():
        l_values = [0, 1, 2, 3]  # s, p, d, f (only as many as sublevels provided)
        for i, capacity in enumerate(sublevels):
            l = l_values[i]
            for _ in range(capacity):
                if atomic_number > 118:  # Limit to 118 elements
                    break
                O_val = compute_ordering_index(period, l, atomic_number)
                data.append({
                    'Element': f"E{atomic_number}",
                    'Z': atomic_number,
                    'n': period,
                    'l': l,
                    'Block': block_map[l],
                    'O': O_val
                })
                atomic_number += 1

    # Add bridging elements with unique pseudo-atomic numbers
    bridging_base = 200  # Arbitrary base for bridging elements (greater than 118)
    for idx, (name, (n_b, l_b)) in enumerate(bridging_elements.items(), start=1):
        z_est = bridging_base + idx
        O_val = compute_ordering_index(n_b, l_b, z_est)
        data.append({
            'Element': name,
            'Z': z_est,
            'n': n_b,
            'l': l_b,
            'Block': 'Bridging',
            'O': O_val
        })

    df = pd.DataFrame(data)
    return df

# Generate the base dataset
elements_df = generate_elements_data()

# Global spiral parameters (used for positioning)
radius = 1.2
spiral_density = 2.0  # Controls the number of turns
k = 2 * np.pi / spiral_density  # Factor for spiral computation

def compute_dynamic_positions(df: pd.DataFrame, phase_offset: float) -> pd.DataFrame:
    """
    Computes dynamic 3D positions for the elements on a spiral.
      - Uses the ordering index 'O' for the vertical axis.
      - Applies a phase offset (from slider) to animate rotation.
    """
    # Compute new theta values with the additional phase offset
    thetas = k * df['O'] + phase_offset
    df = df.copy()  # Avoid modifying the global DataFrame directly
    df['x'] = radius * np.cos(thetas)
    df['y'] = radius * np.sin(thetas)
    df['z'] = df['O']  # Use the ordering index as the z-axis
    return df

# =============================================================================
# 2. BUILD DASH WEB APPLICATION
# =============================================================================

app = Dash(__name__)
server = app.server  # Expose the WSGI server for production (e.g., Gunicorn)

# Create dropdown options for filtering by block type
unique_blocks = sorted(list(elements_df['Block'].unique()))
block_dropdown_options = [{'label': 'All', 'value': 'All'}] + \
    [{'label': b, 'value': b} for b in unique_blocks]

app.layout = html.Div([
    html.H1("Continuum Periodic Table (CPT) - 3D WebGL Visualization"),
    html.P(
        "This interactive plot shows a continuous, spiral version of the periodic table. "
        "Use the dropdown to filter by block (s, p, d, f, Bridging) and the slider to animate the spiral."
    ),
    html.Div([
        dcc.Dropdown(
            id='block-filter',
            options=block_dropdown_options,
            value='All',
            clearable=False,
            style={'width': '40%', 'margin-bottom': '10px'}
        ),
        dcc.Slider(
            id='animation-slider',
            min=0,
            max=100,
            step=1,
            value=0,
            marks={i: str(i) for i in range(0, 101, 10)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
    dcc.Graph(
        id='cpt-3d-graph',
        style={'height': '70vh'}
    ),
    html.Div(
        id='element-details',
        style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '18px'}
    ),
    html.Footer(
        "Developed with Python, Plotly WebGL, and Dash",
        style={'textAlign': 'center', 'marginTop': '20px', 'color': '#666'}
    )
])

# =============================================================================
# 3. INTERACTIVITY: CALLBACKS
# =============================================================================

@app.callback(
    Output('cpt-3d-graph', 'figure'),
    [Input('block-filter', 'value'),
     Input('animation-slider', 'value')]
)
def update_3d_graph(selected_block, slider_value):
    """
    Update the 3D graph based on the selected block filter and the animation slider.
      - The slider controls a phase offset that rotates the spiral.
      - The dropdown filters the elements by block.
    """
    # Determine the phase offset (e.g., slider_value scaled by 0.1)
    phase_offset = slider_value * 0.1

    # Filter the elements DataFrame based on the selected block
    if selected_block == 'All':
        df_filtered = elements_df.copy()
    else:
        df_filtered = elements_df[elements_df['Block'] == selected_block].copy()

    # Compute dynamic positions using the phase offset
    df_dynamic = compute_dynamic_positions(df_filtered, phase_offset)

    fig = go.Figure()

    # Plot each block as a separate trace for a proper legend
    blocks_in_filtered = df_dynamic['Block'].unique()
    for blk in blocks_in_filtered:
        sub_df = df_dynamic[df_dynamic['Block'] == blk]
        fig.add_trace(go.Scatter3d(
            x=sub_df['x'],
            y=sub_df['y'],
            z=sub_df['z'],
            mode='markers+text',
            text=sub_df['Element'],
            textposition="top center",
            hoverinfo='text',
            name=f"{blk} block",
            marker=dict(
                size=5,
                color=color_map.get(blk, 'gray'),
                opacity=0.8
            )
        ))

    fig.update_layout(
        title="CPT: Spiral Representation with Dynamic Rotation",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Ordering Index (O)"
        ),
        legend=dict(
            x=0, y=1.0,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(0,0,0,0.1)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # For performance, ensure Plotly uses WebGL (via scatter3d)
    fig.update_traces(
        type='scatter3d',
        mode='markers+text',
        marker=dict(symbol='circle'),
        textposition="top center"
    )

    return fig

@app.callback(
    Output('element-details', 'children'),
    [Input('cpt-3d-graph', 'clickData')]
)
def display_element_details(clickData):
    """
    When an element is clicked, display its details:
      - Element symbol, atomic number (Z), quantum numbers, block, and ordering index.
    """
    if not clickData:
        return "Click on an element to see its detailed properties."

    clicked_element = clickData['points'][0]['text']  # Get the 'Element' label
    row = elements_df.loc[elements_df['Element'] == clicked_element]
    if row.empty:
        return "No data found for the selected element."

    r = row.iloc[0]
    details_str = (
        f"Element: {r['Element']} | "
        f"Atomic Number (Z): {r['Z']} | "
        f"n={r['n']}, l={r['l']} ({r['Block']} block) | "
        f"Ordering Index (O): {r['O']:.3f}"
    )
    return details_str

# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == '__main__':
    # In production, disable debug mode.
    app.run_server(host="0.0.0.0", port=8050, debug=True)
