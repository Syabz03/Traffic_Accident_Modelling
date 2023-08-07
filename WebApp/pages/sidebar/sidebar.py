from dash import Dash, html, dcc
from app import app
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

layout = html.Div(
    children = [ 
        html.Img(
                src = app.get_asset_url('SUSS.png'), 
                style = {'width': '100%', 'border-style': 'dashed', 'border': '2px'}
        ),
        html.Br(),
        html.Br(),
        dmc.Divider(variant="solid", color = 'black'),
        html.Br(),
        dbc.Nav(
            [
                dbc.NavLink("Prediction", href="/prediction", active = 'exact'),
                dbc.NavLink("Model Explanation Severity", href="/model_explanation_sev", active = 'exact'),
                dbc.NavLink("Model Explanation Distance", href="/model_explanation_dist", active = 'exact'),
            ],
            vertical = True,
            pills = True,
        ),
    ],
    className= 'SIDEBAR_STYLE'
)