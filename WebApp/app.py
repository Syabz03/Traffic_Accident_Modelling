from dash import Dash
import dash_bootstrap_components as dbc
import flask
import dash_auth
import os

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

server = flask.Flask(__name__)

app = Dash(
	__name__,
    server = server,
    title = 'US Traffic Accidents Visualization',
	external_stylesheets = [dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

import index
if __name__ == '__main__':
    index.app.run_server(debug=True, dev_tools_props_check=False)