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
    beta = 0.02    # Shielding/repulsion correction

    relativistic_effect = alpha * (z ** 2 / n ** 2)
    shielding_correction = beta * np.log(z + 1)

    O = n + (l / 10.0) + relativistic_effect - shielding_correction
    return O

# Each period's electron sub-shell capacities (simplified real distribution)
periods = {
    1: [2],          # Period 1: 2 e- in total (s)
    2: [2, 6],       # Period 2: s, p
    3: [2, 6],       # Period 3: s, p
    4: [2, 6, 10],   # Period 4: s, p, d
    5: [2, 6, 10],   # Period 5: s, p, d
    6: [2, 6, 10, 14], # Period 6: s, p, d, f
    7: [2, 6, 10, 14]  # Period 7: s, p, d, f
}

# Map orbital quantum number l to block name
block_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}

# Colors for standard blocks
color_map = {'s': 'blue', 'p': 'green', 'd': 'red', 'f': 'yellow', 'Bridging': 'purple'}

# Hypothetical bridging elements with approximate (n, l) quantum numbers
# We'll assign distinct pseudo-atomic numbers so each bridging element is unique.
bridging_elements = {
    'X₁': (4, 1),
    'X₂': (5, 2),
    'Y₁': (6, 3)
}


def generate_elements_data() -> pd.DataFrame:
    """
    Generates a DataFrame of elements (real + bridging).
    - For real elements (Z=1..118), distribute them per 'periods'.
    - Add bridging elements with custom quantum numbers (n,l).
    """
    data = []
    atomic_number = 1

    # Fill real elements up to Z=118
    for period, sublevels in periods.items():
        l_values = [0, 1, 2, 3]  # s, p, d, f
        for i, capacity in enumerate(sublevels):
            l = l_values[i]
            for _ in range(capacity):
                if atomic_number > 118:  # only fill up to 118
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

    # Now add bridging elements with unique pseudo-atomic numbers
    bridging_counter = 0
    bridging_base = 200  # base atomic number for bridging (arbitrary > 118)
    for name, (n_b, l_b) in bridging_elements.items():
        bridging_counter += 1
        z_est = bridging_base + bridging_counter  # each bridging element gets a unique Z
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


def compute_spiral_positions(df: pd.DataFrame, radius=1.2, spiral_density=2.0) -> pd.DataFrame:
    """
    Assign each element to (x,y,z) coordinates on a 3D spiral around a cylinder.
    - radius: radial distance from center
    - spiral_density: controls how tightly the spiral wraps (bigger -> fewer turns)
    """
    # k determines how many "turns" per unit of O
    k = 2 * np.pi / spiral_density
    thetas = k * df['O'].values

    x_vals = radius * np.cos(thetas)
    y_vals = radius * np.sin(thetas)
    z_vals = df['O'].values

    df['x'] = x_vals
    df['y'] = y_vals
    df['z'] = z_vals
    return df


# Prepare main dataset
elements_df = generate_elements_data()
elements_df = compute_spiral_positions(elements_df)


# =============================================================================
# 2. BUILD DASH APP
# =============================================================================

app = Dash(__name__)

# Generate block options dynamically (including 'All')
unique_blocks = sorted(list(elements_df['Block'].unique()))
block_dropdown_options = [{'label': 'All', 'value': 'All'}] + \
                         [{'label': b, 'value': b} for b in unique_blocks]

app.layout = html.Div([
    html.H1("Continuum Periodic Table (CPT) - 3D WebGL Visualization"),
    html.P(
        "This interactive plot shows a spiral, continuous version of the periodic table. "
        "Use the dropdown to filter by block or bridging elements. Click on a point "
        "to see detailed info."
    ),

    dcc.Dropdown(
        id='block-filter',
        options=block_dropdown_options,
        value='All',
        clearable=False,
        style={'width': '40%', 'margin-bottom': '10px'}
    ),

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
    [Input('block-filter', 'value')]
)
def update_3d_graph(selected_block):
    """
    Build a 3D Scatter plot with separate traces per block,
    so we get a proper legend and can keep each block visually distinct.
    """
    # Filter the DataFrame
    if selected_block == 'All':
        df_filtered = elements_df.copy()
    else:
        df_filtered = elements_df[elements_df['Block'] == selected_block].copy()

    fig = go.Figure()

    # We'll plot each block as a separate trace to get a proper legend
    blocks_in_filtered = df_filtered['Block'].unique()

    for blk in blocks_in_filtered:
        sub_df = df_filtered[df_filtered['Block'] == blk]
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
        title="CPT: Spiral Representation",
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

    # Use Plotly's WebGL for performance
    fig.update_traces(
        type='scatter3d',
        mode='markers+text',
        marker=dict(symbol='circle', line=dict(width=0)),
        textposition="top center"
    )

    return fig


@app.callback(
    Output('element-details', 'children'),
    [Input('cpt-3d-graph', 'clickData')]
)
def display_element_details(clickData):
    """
    When an element is clicked, display its quantum info and ordering index.
    """
    if not clickData:
        return "Click on an element to see its detailed properties."

    clicked_element = clickData['points'][0]['text']  # 'Element' label
    row = elements_df.loc[elements_df['Element'] == clicked_element]
    if row.empty:
        return "No data found for selected element."

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
    app.run_server(debug=True)
