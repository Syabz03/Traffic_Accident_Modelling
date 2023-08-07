from azure.cosmos.partition_key import PartitionKey
from dash import html, dcc, State, Input, Output
from resources import config
from app import app

import dash
import pickle
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import azure.cosmos.documents as documents
import azure.cosmos.exceptions as exceptions
import azure.cosmos.cosmos_client as cosmos_client

resources_dir = 'resources/data/'
state_options, counties_options, cities_options, weather_conditions_options, wind_direction_options = {}, {}, {}, {}, {}
day_night_options = {1: 'Night', 2: 'Day'}
boolean_options = {1: 'True', 0: 'False'}

### For Cosmos DB
HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
DATABASE_ID = config.settings['database_id']
ACCIDENTS_CONTAINER_ID = config.settings['accidents_container_id']
DEFAULTS_CONTAINER_ID = config.settings['defaults_container_id']

client = cosmos_client.CosmosClient(HOST, {'masterKey': MASTER_KEY}, user_agent="CosmosDBPythonQuickstart", user_agent_overwrite=True)
db = client.get_database_client(DATABASE_ID)
accidents_container = db.get_container_client(ACCIDENTS_CONTAINER_ID)
defaults_container = db.get_container_client(DEFAULTS_CONTAINER_ID)

def initialise_dropdown_values():
    states_df = pd.read_pickle(resources_dir + 'states.pkl').to_dict('records')
    for state in states_df:
        state_options[state['id']] = state['State_Name']

    counties_df = pd.read_pickle(resources_dir + 'counties.pkl').to_dict('records')
    for counties in counties_df:
        counties_options[counties['id']] = counties['County']

    cities_df = pd.read_pickle(resources_dir + 'cities.pkl').to_dict('records')
    for cities in cities_df:
        cities_options[cities['id']] = cities['City']

    weather_condition_df = pd.read_pickle(resources_dir + 'weather_condition.pkl').to_dict('records')
    for weather_condition in weather_condition_df:
        weather_conditions_options[weather_condition['id']] = weather_condition['Weather']

    wind_direction_df = pd.read_pickle(resources_dir + 'wind_direction.pkl').to_dict('records')
    for wind_direction in wind_direction_df:
        wind_direction_options[wind_direction['id']] = wind_direction['Wind_Direction']
    
initialise_dropdown_values()

@app.callback(
    [
        Output('info_div', 'style'),
        Output('prediction_div', 'style'),
        Output('Severity_Pred', 'children'),
        Output('Severity_Label', 'children'),
        Output('Distance_Pred', 'children'),
        Output('Distance_Label', 'children'),
        Output('Severity_Fig', 'children'), 
        Output('Distance_Fig', 'children'),
        Output('start_time_input', 'style'),
        Output('end_time_input', 'style'),
        Output('city_input', 'style'),
        Output('county_input', 'style'),
        Output('state_input', 'style'),
        Output('timezone_input', 'style'),
        Output('weather_timestamp_input', 'style'),
        Output('temperature_input', 'style'),
        Output('wind_chill_input', 'style'),
        Output('humidity_input', 'style'),
        Output('pressure_input', 'style'),
        Output('visibility_input', 'style'),
        Output('wind_direction_input', 'style'),
        Output('wind_speed_input', 'style'),
        Output('precipitation_input', 'style'),
        Output('weather_condition_input', 'style'),
        Output('amenity_input', 'style'),
        Output('bump_input', 'style'),
        Output('crossing_input', 'style'),
        Output('give_way_input', 'style'),
        Output('junction_input', 'style'),
        Output('no_exit_input', 'style'),
        Output('railway_input', 'style'),
        Output('roundabout_input', 'style'),
        Output('station_input', 'style'),
        Output('stop_input', 'style'),
        Output('traffic_calming_input', 'style'),
        Output('traffic_signal_input', 'style'),
        Output('turning_loop_input', 'style'),
        Output('day_night_input', 'style')
    ],
    Input('submit-val', 'n_clicks'),
    State('start_time_input', 'value'),
    State('end_time_input', 'value'),
    State('city_input', 'value'),
    State('county_input', 'value'),
    State('state_input', 'value'),
    State('timezone_input', 'value'),
    State('weather_timestamp_input', 'value'),
    State('temperature_input', 'value'),
    State('wind_chill_input', 'value'),
    State('humidity_input', 'value'),
    State('pressure_input', 'value'),
    State('visibility_input', 'value'),
    State('wind_direction_input', 'value'),
    State('wind_speed_input', 'value'),
    State('precipitation_input', 'value'),
    State('weather_condition_input', 'value'),
    State('amenity_input', 'value'),
    State('bump_input', 'value'),
    State('crossing_input', 'value'),
    State('give_way_input', 'value'),
    State('junction_input', 'value'),
    State('no_exit_input', 'value'),
    State('railway_input', 'value'),
    State('roundabout_input', 'value'),
    State('station_input', 'value'),
    State('stop_input', 'value'),
    State('traffic_calming_input', 'value'),
    State('traffic_signal_input', 'value'),
    State('turning_loop_input', 'value'),
    State('day_night_input', 'value')
)
def update_prediction(n_clicks, start_time, end_time, city, county, state, timezone, weather_timestamp, temperature, wind_chill, humidity, 
                  pressure, visibility, wind_direction, wind_speed, precipitation, weather_condition, 
                  amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop, 
                  traffic_calming, traffic_signal, turning_loop, day_night):

    values = [start_time, end_time, city, county, state, timezone, weather_timestamp, temperature, wind_chill, humidity, 
                pressure, visibility, wind_direction, wind_speed, precipitation, weather_condition, 
                amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop, 
                traffic_calming, traffic_signal, turning_loop, day_night]

    pred_severity, pred_distance, pred_severity_fig, pred_distance_fig = "", "", "", ""
    sev_label =  "" if any(v is None or v == "" for v in values) else "Predicted Severity"
    dist_label = "" if any(v is None or v == "" for v in values) else "Predicted Distance (mi)"

    if (start_time is not None and end_time is not None):
        if len(start_time) > 16:
            start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
        else:
            start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")

        if len(end_time) > 16:
            end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
        else:
            end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")

        if len(weather_timestamp) > 16:
            weather_timestamp = datetime.datetime.strptime(weather_timestamp, "%Y-%m-%dT%H:%M:%S")
        else:
            weather_timestamp = datetime.datetime.strptime(weather_timestamp, "%Y-%m-%dT%H:%M")
        
        duration = end_time - start_time
        time_duration = int(duration.total_seconds() / 60)
    
    info_div, prediction_div = {'display': 'none'}, {'display': 'none'}
    valid_input_style, invalid_input_style = {'width': '100%'}, {'outline': '1px solid red', 'width': '100%'}
    input_style_list = [valid_input_style] * len(values)
    if n_clicks and any(v is None or v == "" for v in values):
        info_div = {'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto',
                    'textAlign': 'center'}
        
        ## Check if any value in values is empty, replace with invalid input style, else, keep to default style
        input_style_list = [invalid_input_style if input_value is None or input_value == "" else valid_input_style for input_value in values]
    
    if n_clicks and all(v is not None and v != "" for v in values):
        user_input_list_pred = [
            city, county, state,
            temperature, wind_chill, humidity, pressure, visibility, wind_direction,
            wind_speed, precipitation, weather_condition, amenity, bump, crossing, give_way, junction,	
            no_exit, railway, roundabout, station, stop, traffic_calming, traffic_signal, turning_loop,
            day_night
        ]

        pred_severity, pred_severity_fig, pred_distance, pred_distance_fig = prediction(user_input_list_pred)
        
        user_input_list = [
            pred_severity, str(start_time), str(end_time), pred_distance, city, county, state, timezone,
            str(weather_timestamp), temperature, wind_chill, humidity, pressure, visibility, wind_direction,
            wind_speed, precipitation, weather_condition, str(bool(amenity)), str(bool(bump)), str(bool(crossing)), str(bool(give_way)), str(bool(junction)),	
            str(bool(no_exit)), str(bool(railway)), str(bool(roundabout)), str(bool(station)), str(bool(stop)), str(bool(traffic_calming)), str(bool(traffic_signal)), str(bool(turning_loop)),
            day_night, time_duration,
        ]
        insert_to_db(user_input_list)

        prediction_div = {'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'textAlign': 'center'}
    
    return_array = [info_div, prediction_div, pred_severity, sev_label, pred_distance, dist_label, pred_severity_fig, pred_distance_fig]
    return_array.extend(input_style_list)

    return return_array

def insert_to_db(input):
    last_id_query = "SELECT TOP 1 c.id FROM c ORDER BY c._ts DESC"
    last_id = list(accidents_container.query_items(
                    query = last_id_query,
                    enable_cross_partition_query = True
                ))[0]
    
    new_id = str(int(last_id['id']) + 1)
    input.insert(0, new_id)

    keys = ['id', 'Severity', 'Start_Time', 'End_Time', 'Distance(mi)', 'City',
            'County', 'State', 'Timezone', 'Weather_Timestamp', 'Temperature(F)',
            'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
            'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)',
            'Weather_Condition', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
            'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
            'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset',
            'Time_Duration(min)'
            ]
    new_record = {keys[i]: input[i] for i in range(len(keys))}
    accidents_container.create_item(body = new_record)

def prediction(input):
    # Import Best model trained
    severity_pkl_model_filename = "resources/severity/severity_lr_best_model.pkl"
    with open(severity_pkl_model_filename, 'rb') as file:
        severity_lr_model = pickle.load(file)

    severity_pkl_model_filename = "resources/severity/severity_rfc_best_model.pkl"
    with open(severity_pkl_model_filename, 'rb') as file:
        severity_rf_model = pickle.load(file)

    severity_pkl_model_filename = "resources/severity/severity_knn_best_model.pkl"
    with open(severity_pkl_model_filename, 'rb') as file:
        severity_knn_model = pickle.load(file)
    
    severity_pkl_model_filename = "resources/severity/severity_dt_best_model.pkl"
    with open(severity_pkl_model_filename, 'rb') as file:
        severity_dt_model = pickle.load(file)

    distance_pkl_model_filename = "resources/distance/distance_lr_best_model.pkl"
    with open(distance_pkl_model_filename, 'rb') as file:
        distance_lr_model = pickle.load(file)

    distance_pkl_model_filename = "resources/distance/distance_rfc_best_model.pkl"
    with open(distance_pkl_model_filename, 'rb') as file:
        distance_rf_model = pickle.load(file)
    
    distance_pkl_model_filename = "resources/distance/distance_knn_best_model.pkl"
    with open(distance_pkl_model_filename, 'rb') as file:
        distance_knn_model = pickle.load(file)
    
    distance_pkl_model_filename = "resources/distance/distance_dt_best_model.pkl"
    with open(distance_pkl_model_filename, 'rb') as file:
        distance_dt_model = pickle.load(file)
    
    column_names = [
        'City', 'County', 'State', 
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction',
        'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
        'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 
        'Sunrise_Sunset']
    user_input_dframe = pd.DataFrame([input], columns=column_names)

    # pred_severity_lr = severity_lr_model.predict(user_input_dframe)[0]
    pred_severity_rf = severity_rf_model.predict(user_input_dframe)[0]
    # pred_severity_knn = severity_knn_model.predict(user_input_dframe)[0]
    # pred_severity_dt = severity_dt_model.predict(user_input_dframe)[0]
    
    lr_ynew_prob = severity_lr_model.predict_proba(user_input_dframe)
    rf_ynew_prob = severity_rf_model.predict_proba(user_input_dframe)
    knn_ynew_prob = severity_knn_model.predict_proba(user_input_dframe)
    dt_ynew_prob = severity_dt_model.predict_proba(user_input_dframe)
    severity_fig = get_prediction_stats('Severity', lr_ynew_prob, rf_ynew_prob, knn_ynew_prob, dt_ynew_prob)

    # pred_distance_lr = distance_lr_model.predict(user_input_dframe)[0]
    pred_distance_rf = distance_rf_model.predict(user_input_dframe)[0]
    # pred_distance_knn = distance_knn_model.predict(user_input_dframe)[0]
    # pred_distance_dt = distance_dt_model.predict(user_input_dframe)[0]
    
    lr_ynew_prob = distance_lr_model.predict_proba(user_input_dframe)
    rf_ynew_prob = distance_rf_model.predict_proba(user_input_dframe)
    knn_ynew_prob = distance_knn_model.predict_proba(user_input_dframe)
    dt_ynew_prob = distance_dt_model.predict_proba(user_input_dframe)
    distance_fig = get_prediction_stats('Distance', lr_ynew_prob, rf_ynew_prob, knn_ynew_prob, dt_ynew_prob)

    if (pred_distance_rf == 0):
        pred_distance_rf = "0 - 1"
    elif (pred_distance_rf == 1):
        pred_distance_rf = "1 - 2"
    elif (pred_distance_rf == 2):
        pred_distance_rf = "2 - 3"
    elif (pred_distance_rf == 3):
        pred_distance_rf = "3 - 4"
    elif (pred_distance_rf == 4):
        pred_distance_rf = "4 - 5"
    elif (pred_distance_rf == 5):
        pred_distance_rf = "5 - 6"
    elif (pred_distance_rf == 6):
        pred_distance_rf = "6 - 7"
    elif (pred_distance_rf == 7):
        pred_distance_rf = "7 - 8"

    return pred_severity_rf, severity_fig, pred_distance_rf, distance_fig

def get_prediction_stats(feature, lr_ynew_prob, rf_ynew_prob, knn_ynew_prob, dt_ynew_prob):
    algo_list = ['Logistic Regression', 'Random Forest', 'K-Nearest Neighbour', 'Decision Tree']

    feature_data_range = [*range(1,5)] if feature == 'Severity' else [*range(0,8)]

    lr_ynew_prob = lr_ynew_prob[0].tolist()
    lr_repeated = [algo_list[0]] * len(lr_ynew_prob)
    lr_combined_df = pd.DataFrame({'model': lr_repeated, feature: feature_data_range, 'probability': lr_ynew_prob})

    rf_ynew_prob = rf_ynew_prob[0].tolist()
    rf_repeated = [algo_list[1]] * len(rf_ynew_prob)
    rf_combined_df = pd.DataFrame({'model': rf_repeated, feature: feature_data_range,  'probability': rf_ynew_prob})

    knn_ynew_prob = knn_ynew_prob[0].tolist()
    knn_repeated = [algo_list[2]] * len(knn_ynew_prob)
    knn_combined_df = pd.DataFrame({'model': knn_repeated, feature: feature_data_range, 'probability': knn_ynew_prob})

    dt_ynew_prob = dt_ynew_prob[0].tolist()
    dt_repeated = [algo_list[3]] * len(dt_ynew_prob)
    dt_combined_df = pd.DataFrame({'model': dt_repeated, feature: feature_data_range, 'probability': dt_ynew_prob})

    probability_df = pd.concat([lr_combined_df, rf_combined_df, knn_combined_df, dt_combined_df], ignore_index = True, sort = False)

    fig = px.bar(probability_df, x = 'model', y = 'probability', color = feature, text = feature,
             labels = {
                 'model': 'ML Model',
                 'probability': 'Probability (%)',
                 'prediction': 'Prediction'
             })
    fig.layout.yaxis.tickformat = ',.0%'
    fig.update_layout(yaxis_range = [0, 1.1], plot_bgcolor = 'white', coloraxis = {"colorbar": {"dtick": 1}})
    fig.update_traces(textfont_size = 14)
    # new = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'KNN': 'K-Nearest Neighbour', 'DT': 'Decision Trees'}
    # fig.for_each_trace(lambda t: t.update(name = new[t.name]))
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        # gridcolor='lightgrey'
    )

    subtitle = "Shows the predicted " + feature + " and probability of the prediction between the 4 models trained"
    card_fig_element = dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H3(f"Prediction Probability"),
                html.H6(subtitle, className = "card-subtitle")
            ]),
        ]),
        dbc.CardBody(
            dcc.Graph(id = "fig_fi_scores", figure = fig,
                        responsive = True,
                        style = {
                            "width": "100%",
                            "height": "100%"
                        }
                    )
        )
    ]),

    return card_fig_element

def query_default_values(query, value='default'):
    # querying to the database
    df_defaults = list(defaults_container.query_items(
                        query = query,
                        enable_cross_partition_query=True
                    ))[0]

    timezone_val = df_defaults['Timezone']
    temp_val = df_defaults['Temperature(F)']
    wind_chill_val = df_defaults['Wind_Chill(F)']
    humidity_val = df_defaults['Humidity(%)']
    pressure_val = df_defaults['Pressure(in)']
    visibility_val = df_defaults['Visibility(mi)']
    wind_direction_val = df_defaults['Wind_Direction']
    wind_speed_val = df_defaults['Wind_Speed(mph)']
    precipitation_val = df_defaults['Precipitation(in)']
    weather_condition_val = df_defaults['Weather_Condition']
    amenity_val = df_defaults['Amenity']
    bump_val = df_defaults['Bump']
    crossing_val = df_defaults['Crossing']
    give_way_val = df_defaults['Give_Way']
    junction_val = df_defaults['Junction']
    no_exit_val = df_defaults['No_Exit']
    railway_val = df_defaults['Railway']
    roundabout_val = df_defaults['Roundabout']
    station_val = df_defaults['Station']
    stop_val = df_defaults['Stop']
    traffic_calming_val = df_defaults['Traffic_Calming']
    traffic_signal_val = df_defaults['Traffic_Signal']
    turning_loop_val = df_defaults['Turning_Loop']
    sunrise_sunset_val = df_defaults['Sunrise_Sunset']

    return_array = [temp_val, wind_chill_val, humidity_val, pressure_val, visibility_val, wind_direction_val, wind_speed_val, precipitation_val, weather_condition_val, amenity_val, bump_val, crossing_val, give_way_val, junction_val, no_exit_val, railway_val, roundabout_val, station_val, stop_val, traffic_calming_val, traffic_signal_val, turning_loop_val, sunrise_sunset_val, timezone_val]
    return return_array

@app.callback(
    [
        # Output('start_time_input', 'value'),
        # Output('end_time_input', 'value'),
        # Output('city_input', 'value'),
        # Output('county_input', 'value'),
        # Output('state_input', 'value'),
        Output('default-val', 'n_clicks'),
        # Output('weather_timestamp_input', 'value'),
        Output('temperature_input', 'value'),
        Output('wind_chill_input', 'value'),
        Output('humidity_input', 'value'),
        Output('pressure_input', 'value'),
        Output('visibility_input', 'value'),
        Output('wind_direction_input', 'value'),
        Output('wind_speed_input', 'value'),
        Output('precipitation_input', 'value'),
        Output('weather_condition_input', 'value'),
        Output('amenity_input', 'value'),
        Output('bump_input', 'value'),
        Output('crossing_input', 'value'),
        Output('give_way_input', 'value'),
        Output('junction_input', 'value'),
        Output('no_exit_input', 'value'),
        Output('railway_input', 'value'),
        Output('roundabout_input', 'value'),
        Output('station_input', 'value'),
        Output('stop_input', 'value'),
        Output('traffic_calming_input', 'value'),
        Output('traffic_signal_input', 'value'),
        Output('turning_loop_input', 'value'),
        Output('day_night_input', 'value'),
        Output('timezone_input', 'value')
    ],
    Input('default-val', 'n_clicks'),
    Input("weather_condition_input", "value"),
    State('weather_condition_input', 'value'),
    State('timezone_input', 'value'),
    State('temperature_input', 'value'),
    State('wind_chill_input', 'value'),
    State('humidity_input', 'value'),
    State('pressure_input', 'value'),
    State('visibility_input', 'value'),
    State('wind_direction_input', 'value'),
    State('wind_speed_input', 'value'),
    State('precipitation_input', 'value'),
    State('amenity_input', 'value'),
    State('bump_input', 'value'),
    State('crossing_input', 'value'),
    State('give_way_input', 'value'),
    State('junction_input', 'value'),
    State('no_exit_input', 'value'),
    State('railway_input', 'value'),
    State('roundabout_input', 'value'),
    State('station_input', 'value'),
    State('stop_input', 'value'),
    State('traffic_calming_input', 'value'),
    State('traffic_signal_input', 'value'),
    State('turning_loop_input', 'value'),
    State('day_night_input', 'value'),
)
def get_default_values(n_clicks, weather_condition_new, weather_condition_old, timezone, temperature, wind_chill, humidity, pressure, 
                  visibility, wind_direction, wind_speed, precipitation, amenity, bump, crossing, 
                  give_way, junction, no_exit, railway, roundabout, station, stop, 
                  traffic_calming, traffic_signal, turning_loop, day_night):
    
    values = [temperature, wind_chill, humidity, pressure, 
              visibility, wind_direction, wind_speed, precipitation, weather_condition_new,
              amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop, 
              traffic_calming, traffic_signal, turning_loop, day_night, timezone]

    ## DropDown Boolean 0: False, 1: True
    temp_val, wind_chill_val, humidity_val, pressure_val = "", "", "", ""
    visibility_val, wind_speed_val, precipitation_val = "", "", ""
    wind_direction_val, weather_condition_val, amenity_val, bump_val, crossing_val = None, None, 0, 0, 0
    give_way_val, junction_val, no_exit_val, railway_val, roundabout_val = 0, 0, 0, 0, 0
    station_val, stop_val, traffic_calming_val, traffic_signal_val, turning_loop_val, sunrise_sunset_val, timezone_val = 0, 0, 0, 0, 0, None, None

    default_vals = [temp_val, wind_chill_val, humidity_val, pressure_val, 
                    visibility_val, wind_direction_val, wind_speed_val, precipitation_val, weather_condition_val, 
                    amenity_val, bump_val, crossing_val, give_way_val, junction_val, no_exit_val, railway_val, 
                    roundabout_val, station_val, stop_val, traffic_calming_val, traffic_signal_val, turning_loop_val, 
                    sunrise_sunset_val, timezone_val]
    
    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if input_id == "weather_condition_input" and weather_condition_new != None:
            query = " SELECT * FROM c where c.Purpose = '" + str(float(weather_condition_new)) + "' "
            default_vals = query_default_values(query)
        
    if n_clicks:
        query = " SELECT * FROM c where c.Purpose = 'default' "
        default_vals = query_default_values(query)

    output_list = [values[i] if values[i] else default_vals[i] for i in range(len(values))]
    return_array = [0]
    return_array.extend(output_list)
    return return_array

## Layout of Page
layout = html.Div([
    html.Br(),
    html.H1(children = 'Prediction', style = {'margin-left': '12px'}),
    html.Br(),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'city_label', children = 'City'),
                className = 'pred-top-left-subdiv'
        ),
            html.Div(
                dcc.Dropdown(
                    id = 'city_input', 
                    options = [{'label': v, 'value': k} for k, v in cities_options.items()]
                ),
                className = 'pred-top-right-subdiv'
            )
        ],
        className = 'pred-top-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'county_label', children = 'County'),
                className = 'pred-top-left-subdiv'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'county_input', 
                    options = [{'label': v, 'value': k} for k, v in counties_options.items()]
                ),
                className = 'pred-top-right-subdiv'
            )
        ],
        className = 'pred-top-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'state_label', children = 'State'),
                className = 'pred-top-left-subdiv'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'state_input',
                    options = [{'label': v, 'value': k} for k, v in state_options.items()]
                ),
                className = 'pred-top-right-subdiv'
            )
        ],
        className = 'pred-top-div'
    ),
    html.Br(),
    html.Br(),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'start_time_label', children = 'Start Time'),
                className = 'pred-top-left-subdiv'
            ),
            html.Div(
                dcc.Input(id = 'start_time_input', type = 'datetime-local', step = "1"),
                className = 'pred-mid-right-subdiv'
            )
        ],
        className = 'pred-top-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'end_time_label', children = 'End Time'),
                className = 'pred-top-left-subdiv'
            ),
            html.Div(
                dcc.Input(id = 'end_time_input', type = 'datetime-local', step = "1"),
                className = 'pred-mid-right-subdiv'
            )
        ],
        className = 'pred-top-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'weather_timestamp_label', children = 'Weather Timestamp'),
                className = 'pred-top-left-subdiv'
            ),
            html.Div(
                dcc.Input(id = 'weather_timestamp_input', type = 'datetime-local', step = "1"),
                className = 'pred-mid-right-subdiv'
            )
        ],
        className = 'pred-top-div'
    ),
    html.Br(),
    html.Br(),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'temperature_label', children = 'Temperature (F)'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Input(id = 'temperature_input', type = 'number'),
                className = 'input-div'
            )
        ],
        className = 'pred-left-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'wind_chill_label', children = 'Wind_Chill (F)'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Input(id = 'wind_chill_input', type = 'number'),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'humidity_label', children = 'Humidity (%)'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Input(id = 'humidity_input', type = 'number', min = 0),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'pressure_label', children = 'Pressure (in)'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Input(id = 'pressure_input', type = 'number'),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div mr-0'
    ),
    html.Br(),
    html.Br(),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'visibility_label', children = 'Visibility (mi)'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Input(id = 'visibility_input', type = 'number'),
                className = 'input-div'
            )
        ],
        className = 'pred-left-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'wind_speed_label', children = 'Wind Speed (mph)'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Input(id='wind_speed_input', type='number'),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'precipitation_label', children = 'Precipitation (in)'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Input(id='precipitation_input', type='number', min=0),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'wind_direction_label', children = 'Wind Direction'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(id='wind_direction_input', options = [{'label': v, 'value': k} for k, v in wind_direction_options.items()]),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div mr-0'
    ),
    html.Br(),
    html.Br(),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'weather_condition_label', children = 'Weather Condition'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(id='weather_condition_input', options = [{'label': v, 'value': k} for k, v in weather_conditions_options.items()]),
                className = 'input-div'
            )
        ],
        className = 'pred-left-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'amenity_label', children = 'Amenity'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'amenity_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'bump_label', children = 'Bump'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'bump_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'crossing_label', children = 'Crossing'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id='crossing_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div mr-0'
    ),
    html.Br(),
    html.Br(),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'give_way_label', children = 'Give_Way'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'give_way_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-left-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'junction_label', children = 'Junction'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'junction_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'no_exit_label', children = 'No Exit'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'no_exit_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div')
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'railway_label', children = 'Railway'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'railway_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div mr-0'
    ),
    html.Br(),
    html.Br(),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'roundabout_label', children = 'Roundabout'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'roundabout_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-left-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'station_label', children = 'Station'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'station_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'stop_label', children = 'Stop'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'stop_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'traffic_calming_label', children = 'Traffic Calming'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'traffic_calming_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div mr-0'
    ),
    html.Br(),
    html.Br(),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'traffic_signal_label', children = 'Traffic Signal'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'traffic_signal_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-left-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'turning_loop_label', children = 'Turning Loop'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'turning_loop_input', 
                    options = [{'label': v, 'value': k} for k, v in boolean_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'day_night_label', children = 'Day/Night'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id = 'day_night_input', 
                    options = [{'label': v, 'value': k} for k, v in day_night_options.items()]
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div'
    ),
    html.Div( 
        children = [
            html.Div(
                html.H6(id = 'timezone_label', children = 'Timezone'),
                className = 'label-div'
            ),
            html.Div(
                dcc.Dropdown(
                    id='timezone_input', 
                    options = ['US/Eastern', 'US/Pacific', 'US/Central', 'US/Mountain']
                ),
                className = 'input-div'
            )
        ],
        className = 'pred-right-div mr-0'
    ),
    html.Br(),
    html.Br(),
    html.Div(
        children = [
            html.Div(
                dbc.Button('Use Default Values', id = 'default-val', n_clicks = 0, color = 'secondary')
                , style={'textAlign':'left', 'width': '50%', 'display': 'inline-block'}
            ),
            html.Div(
                dbc.Button('Predict', id = 'submit-val', n_clicks = 0, color = 'primary')
                , style={'textAlign':'right', 'width': '48%', 'display': 'inline-block', 'margin-right': '20px'}
            )
        ]
        , style={'textAlign':'center', 'width': '100%'}
    ),
    html.Br(),
    html.Br(),
    dcc.Loading(
        id = "loading-pred",
        type = "default",
        children = [
            html.Div(
                id = 'info_div',
                children = [
                    html.H1(id = 'info_label', children = 'Please ensure all fields are filled')
                ]
            ),
            html.Div(
                id = 'prediction_div',
                children = [
                    html.Div( 
                        html.H1(id = 'Severity_Label'),
                        className = 'pred-labels'
                    ),
                    html.Div( 
                        html.H1(id = 'Distance_Label'),
                        className = 'pred-labels'
                    ),
                    html.Div( 
                        html.H2(id = 'Severity_Pred'),
                        className = 'pred-labels'
                    ),
                    html.Div( 
                        html.H2(id = 'Distance_Pred'),
                        className = 'pred-labels'
                    ),
                    html.Br(),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(
                            id = 'Severity_Fig',
                            children = []
                        ),
                        dbc.Col(
                            id = 'Distance_Fig', 
                            children = []
                        )
                    ]),
                    html.Br(),
                ]
            ),
        ]
    ),
    html.Br(),
    html.Br(),
], className = 'layout')