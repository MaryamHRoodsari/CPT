import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# =============================================================================
# 1. COMPUTATIONAL BACKEND: ADVANCED ORDERING INDEX & QUANTUM DATA
# =============================================================================

def compute_ordering_index(n: int, l: int, z: int) -> float:
    """Computes the ordering index 'O' using quantum corrections."""
    alpha = 0.001  # Relativistic effect
    beta = 0.02    # Electron shielding correction
    relativistic_effect = alpha * (z ** 2 / n ** 2)
    shielding_correction = beta * np.log(z + 1)
    return n + (l / 10.0) + relativistic_effect - shielding_correction

# Define electron configurations for proper period mapping
periods = {1: [2], 2: [2, 6], 3: [2, 6], 4: [2, 6, 10], 5: [2, 6, 10], 6: [2, 6, 10, 14], 7: [2, 6, 10, 14]}
block_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
color_map = {'s': 'blue', 'p': 'green', 'd': 'red', 'f': 'yellow'}
bridging_elements = {'X₁': (4, 1), 'X₂': (5, 2), 'Y₁': (6, 3)}

def generate_elements_data():
    """Generates quantum data for the CPT structure, including bridging elements."""
    data = []
    atomic_number = 1  

    for period, sublevels in periods.items():
        l_values = [0, 1, 2, 3]  
        for i, max_electrons in enumerate(sublevels):
            l = l_values[i]
            for _ in range(max_electrons):
                if atomic_number > 118:
                    break
                O = compute_ordering_index(period, l, atomic_number)
                data.append({'Element': f"E{atomic_number}", 'n': period, 'l': l, 'O': O, 'Block': block_map[l], 'Z': atomic_number})
                atomic_number += 1
    
    # Adding hypothetical bridging elements (estimated quantum values)
    for name, (n, l) in bridging_elements.items():
        z_estimated = 120 + len(data)  # Assign a pseudo-atomic number
        O = compute_ordering_index(n, l, z_estimated)
        data.append({'Element': name, 'n': n, 'l': l, 'O': O, 'Block': 'Bridging', 'Z': z_estimated})

    return pd.DataFrame(data)

def compute_spiral_positions(df, R=1.2):
    """Maps elements onto a cylindrical spiral with WebGL acceleration support."""
    k = 2 * np.pi / 2  
    df['theta'] = k * df['O']
    df['x'] = R * np.cos(df['theta'])
    df['y'] = R * np.sin(df['theta'])
    df['z'] = df['O']
    df['color'] = df['Block'].apply(lambda b: color_map.get(b, 'purple'))  # Purple for bridging elements
    return df

# =============================================================================
# 2. DATA PROCESSING & SPIRAL POSITIONING
# =============================================================================

elements_df = generate_elements_data()
elements_df = compute_spiral_positions(elements_df)

# =============================================================================
# 3. DASH WEB APP: INTERACTIVE CPT VISUALIZATION
# =============================================================================

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Continuum Periodic Table (CPT) - Interactive 3D Visualization"),
    html.P("Explore the elements with dynamic filtering and detailed quantum information."),

    # Dropdown filter for element blocks
    dcc.Dropdown(
        id='block-filter',
        options=[{'label': b, 'value': b} for b in ['All', 's', 'p', 'd', 'f', 'Bridging']],
        value='All',
        clearable=False,
        style={'width': '40%', 'margin-bottom': '10px'}
    ),

    # Interactive 3D graph
    dcc.Graph(id='cpt-3d-graph', style={'height': '70vh'}),

    # Element details display
    html.Div(id='element-details', style={'textAlign': 'center', 'margin-top': '20px', 'fontSize': '18px'}),
    
    # Footer
    html.Footer("Developed with Python, Plotly WebGL, and Dash", style={'textAlign': 'center', 'marginTop': '20px'})
])

# =============================================================================
# 4. CALLBACKS FOR INTERACTIVITY
# =============================================================================

@app.callback(
    Output('cpt-3d-graph', 'figure'),
    [Input('block-filter', 'value')]
)
def update_graph(selected_block):
    """Updates 3D graph based on selected block."""
    filtered_df = elements_df if selected_block == 'All' else elements_df[elements_df['Block'] == selected_block]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=filtered_df['x'], y=filtered_df['y'], z=filtered_df['z'],
        mode='markers+text',
        marker=dict(size=5, color=filtered_df['color'], opacity=0.8),
        text=filtered_df['Element'],
        textposition="top center",
        hoverinfo='text'
    ))

    # Update layout
    fig.update_layout(
        title="Continuum Periodic Table (CPT) - Quantum Enhanced",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Ordering Index (O)"),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    return fig

@app.callback(
    Output('element-details', 'children'),
    [Input('cpt-3d-graph', 'clickData')]
)
def display_element_details(clickData):
    """Displays detailed quantum information when an element is clicked."""
    if not clickData:
        return "Click on an element to see detailed quantum properties."

    element_name = clickData['points'][0]['text']
    element_info = elements_df[elements_df['Element'] == element_name].iloc[0]

    return f"Element: {element_info['Element']} | Atomic Number: {element_info['Z']} | " \
           f"Quantum Numbers: n={element_info['n']}, l={element_info['l']} | " \
           f"Ordering Index (O): {round(element_info['O'], 2)} | Block: {element_info['Block']}"

# =============================================================================
# 5. RUN THE WEB APP
# =============================================================================

if __name__ == '__main__':
    app.run_server(debug=True)
