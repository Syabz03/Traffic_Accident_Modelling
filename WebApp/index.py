from dash import Dash, html, dcc, Input, Output, State
from app import app
from pages import model_explanation_dist, model_explanation_sev, prediction

from pages.sidebar import sidebar

app.layout = html.Div(
    children = [ 
        dcc.Location(id = 'url', refresh=True,),
        sidebar.layout,
        html.Div(
            id = 'main_container',
            className = 'CONTENT_STYLE'
        ),
    ]
)

app.validation_layout = html.Div(
    children = [
        app.layout,
        # model_explanation_sev.layout,
        # model_explanation_dist.layout,
    ]
)

@app.callback(
    Output('main_container', 'children'),
    Output('url','pathname',),
    Input('url','pathname',),
    prevent_initial_call = True
)
def display_page(pathname):
    if pathname == '/model_explanation_dist':
        return [model_explanation_dist.layout], pathname
    elif pathname == '/model_explanation_sev':
        return [model_explanation_sev.layout], pathname
    elif pathname == '/prediction':
        return [prediction.layout], pathname
    else:
        return [], '/model_explanation_sev'
