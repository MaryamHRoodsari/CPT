import dash
from dash import html, dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np

# Initialize the Dash app and expose the underlying WSGI server for production (e.g., Gunicorn)
app = dash.Dash(__name__)
server = app.server

def generate_cpt_figure(frame=0):
    """
    Generate a 3D CPT figure that simulates a spiral animation.
    The frame parameter shifts the phase of the sine and cosine functions.
    """
    t = np.linspace(0, 4 * np.pi, 200)
    x = np.sin(t + frame * 0.1)
    y = np.cos(t + frame * 0.1)
    z = t
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='royalblue', width=6)
        )
    ])
    fig.update_layout(
        title="CPT Simulator & Animator (Proprietary)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

app.layout = html.Div([
    html.H1("CPT Simulator and Animator (Proprietary)"),
    dcc.Graph(id="cpt-graph", figure=generate_cpt_figure()),
    html.Label("Animation Frame:"),
    dcc.Slider(
        id="frame-slider",
        min=0,
        max=100,
        step=1,
        value=0,
        marks={i: str(i) for i in range(0, 101, 10)},
    )
])

@app.callback(
    Output("cpt-graph", "figure"),
    Input("frame-slider", "value")
)
def update_figure(frame):
    """Update the 3D CPT figure based on the slider input."""
    return generate_cpt_figure(frame)

if __name__ == '__main__':
    # Note: In production, remove debug=True.
    app.run_server(host="0.0.0.0", port=8050, debug=True)
g