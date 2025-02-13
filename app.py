import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate

from mendeleev import element as mendeleev_element

# =============================================================================
# 1. COMPUTATIONAL BACKEND
# =============================================================================

DATA_CACHE_PATH = "elements_data.csv"  # The local cache file to cache results

def compute_ordering_index(n: int, l: int, z: int) -> float:
    """
    Computes a simplified ordering index "O" including:
      - n: principal quantum number
      - l: orbital quantum number
      - z: atomic number
      - relativistic correction, shielding
    """
    alpha = 0.001
    beta = 0.02

    relativistic_effect = alpha * (z ** 2 / n ** 2)
    shielding_correction = beta * np.log(z + 1)
    O = n + (l / 10.0) + relativistic_effect - shielding_correction
    return O

block_to_l = {'s': 0, 'p': 1, 'd': 2, 'f': 3}

marker_shape_map = {
    's': 'circle',
    'p': 'diamond-open',
    'd': 'cross',
    'f': 'square',
    'bridging': 'diamond'
}

def build_elements_data() -> pd.DataFrame:
    """
    Build a DataFrame of real chemical elements (Z=1..118) plus hypothetical bridging.
    
    - 'Element': the long multiline label for hover info
    - 'PlotSymbol': the short text (e.g. 'H', 'He', 'X₁') displayed on the plot
    """
    data = []

    # Real elements
    for Z in range(1, 119):
        try:
            el = mendeleev_element(Z)
        except Exception:
            continue
        
        name = el.name
        symbol = el.symbol
        period = el.period or 1
        block_letter = el.block.lower() if el.block else 's'
        l_val = block_to_l.get(block_letter, 0)
        m_val = 0
        s_val = 0.5

        O_val = compute_ordering_index(period, l_val, Z)

        # Long hover label
        hover_label = f"{name} ({symbol}, {Z})\nO={O_val:.2f}"
        # Short symbol for the actual text on the marker
        plot_symbol = symbol  # e.g. "H", "He", "Fe"

        data.append({
            'Element': hover_label,
            'PlotSymbol': plot_symbol,
            'Name': name,
            'Symbol': symbol,
            'Z': Z,
            'n': period,
            'l': l_val,
            'm': m_val,
            's': s_val,
            'Block': block_letter,
            'O': O_val
        })

    # Hypothetical bridging
    bridging_elements = {
        'X₁': (4, 1),
        'X₂': (5, 2),
        'Y₁': (6, 3)
    }
    bridging_base = 200

    for idx, (b_sym, (n_b, l_b)) in enumerate(bridging_elements.items(), start=1):
        z_est = bridging_base + idx
        O_val = compute_ordering_index(n_b, l_b, z_est)
        hover_label = (f"Hypothetical {b_sym} (n={n_b}, l={l_b})\n"
                       f"O={O_val:.2f} - Unstable/Undiscovered")
        plot_symbol = b_sym  # e.g. "X₁"

        data.append({
            'Element': hover_label,
            'PlotSymbol': plot_symbol,
            'Name': f"Hypothetical {b_sym}",
            'Symbol': b_sym,
            'Z': z_est,
            'n': n_b,
            'l': l_b,
            'm': 0,
            's': 0.5,
            'Block': 'bridging',
            'O': O_val
        })

    return pd.DataFrame(data)

def get_elements_data_cached() -> pd.DataFrame:
    """Use local CSV if present; otherwise build the dataset and cache it."""
    if os.path.exists(DATA_CACHE_PATH):
        df = pd.read_csv(DATA_CACHE_PATH)
    else:
        df = build_elements_data()
        df.to_csv(DATA_CACHE_PATH, index=False)
    return df

elements_df = get_elements_data_cached()

# Spiral config
radius = 3.0
spiral_density = 2.0
k = 2 * np.pi / spiral_density

def compute_dynamic_positions(df: pd.DataFrame, phase_offset: float) -> pd.DataFrame:
    """Compute 3D spiral coords (x,y,z) from ordering index O + offset."""
    thetas = k * df['O'] + phase_offset
    df = df.copy()
    df['x'] = radius * np.cos(thetas)
    df['y'] = radius * np.sin(thetas)
    df['z'] = df['O']
    return df

# =============================================================================
# 2. DASH APP
# =============================================================================

app = Dash(__name__)
server = app.server

unique_blocks = sorted(list(elements_df['Block'].unique()))
block_dropdown_options = [{'label': 'ALL', 'value': 'All'}] + [
    {'label': b.upper(), 'value': b} for b in unique_blocks
]

app.layout = html.Div([
    html.H1("Continuum Periodic Table (CPT) - 3D Quantum Visualization"),
        html.P(
        "Explore the real periodic table reimagined as a dynamic, spiral structure. "
        "Each element is labeled with its real name, symbol, atomic number, and computed ordering index. "
        "Hypothetical bridging elements (unstable/undiscovered) are shown with special labels. "
        "Use the dropdown to filter by block (S, P, D, F, BRIDGING) and the slider to rotate the spiral."
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
    dcc.Graph(id='cpt-3d-graph', style={'height': '70vh'}),
    html.Div(id='element-details', style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '18px'}),
    html.Footer("Developed with Python, Plotly, Dash, and Mendeleev (cached data)", 
                style={'textAlign': 'center', 'marginTop': '20px', 'color': '#666'})
])

# =============================================================================
# 3. CALLBACKS
# =============================================================================

@app.callback(
    Output('cpt-3d-graph', 'figure'),
    [Input('block-filter', 'value'), Input('animation-slider', 'value')]
)
def update_3d_graph(selected_block, slider_value):
    phase_offset = slider_value * 0.1
    if selected_block == 'All':
        df_filtered = elements_df
    else:
        df_filtered = elements_df[elements_df['Block'] == selected_block]

    if df_filtered.empty:
        raise PreventUpdate

    df_dynamic = compute_dynamic_positions(df_filtered, phase_offset)
    fig = go.Figure()

    custom_colorscale = [[0, 'red'], [1, 'blue']]
    blocks_in_filtered = df_dynamic['Block'].unique()
    show_colorbar = True

    for blk in blocks_in_filtered:
        sub_df = df_dynamic[df_dynamic['Block'] == blk]
        symbol = marker_shape_map.get(blk, 'circle')
        fig.add_trace(go.Scatter3d(
            x=sub_df['x'],
            y=sub_df['y'],
            z=sub_df['z'],
            # We only show short symbol on the marker
            text=sub_df['PlotSymbol'],
            mode='markers+text',
            textposition="top center",
            # For the hover, we use the full 'Element' column
            hovertext=sub_df['Element'],
            hoverinfo='text',
            name=f"{blk.upper()} block",
            marker=dict(
                size=6,
                symbol=symbol,
                color=sub_df['O'],
                colorscale=custom_colorscale,
                colorbar=dict(title="Ordering Index (O)") if show_colorbar else None,
                opacity=0.9
            )
        ))
        show_colorbar = False

    fig.update_layout(
        title="CPT: Spiral Representation with Short Marker Labels",
        scene=dict(
            xaxis_title="X (Spiral Projection)",
            yaxis_title="Y (Spiral Projection)",
            zaxis_title="Ordering Index (O)"
        ),
        legend=dict(x=0, y=1.0, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.2)'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.update_traces(type='scatter3d', mode='markers+text', textposition="top center")
    return fig

@app.callback(
    Output('element-details', 'children'),
    [Input('cpt-3d-graph', 'clickData')]
)
def display_element_details(clickData):
    if not clickData:
        return "Click on an element to see its detailed properties."
    
    # The user clicked the marker whose text = sub_df['PlotSymbol'],
    # but the hover is sub_df['Element']. The 'clickData' references the hovertext or text,
    # depending on how Plotly reports it in 3D. We'll handle that carefully.
    # By default, 'clickData' might use the same property as 'hovertext' = sub_df['Element'].
    # Let's see:
    label_clicked = clickData['points'][0].get('hovertext') or clickData['points'][0]['text']

    row = elements_df.loc[elements_df['Element'] == label_clicked]
    if row.empty:
        # maybe it matches 'PlotSymbol' for bridging?
        row = elements_df.loc[elements_df['PlotSymbol'] == label_clicked]
        if row.empty:
            return "No data found for the selected element."

    r = row.iloc[0]
    details_str = (
        f"Name: {r['Name']} ({r['Symbol']}) | Z: {r['Z']} | "
        f"n={r['n']}, l={r['l']}, m={r['m']}, s={r['s']} | "
        f"O={r['O']:.3f}"
    )
    return details_str

# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
