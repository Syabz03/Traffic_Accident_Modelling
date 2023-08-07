import bz2
import json
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

from app import app
from dash import html, dash_table
from explainerdashboard.custom import *
# from explainerdashboard import ClassifierExplainer

res_sev_dir = 'resources/severity/'

pkl_X_res_filename = res_sev_dir + "X_res.pkl"
pkl_y_res_filename = res_sev_dir + "y_res.pkl"
with open(pkl_X_res_filename, 'rb') as file:
    X = pickle.load(file)
with open(pkl_y_res_filename, 'rb') as file:
    y = pickle.load(file)

###
# Classes for different Explainability tabs
###
class CustomClassificationTab(ExplainerComponent):
    def __init__(self, explainer, name = None):
        super().__init__(explainer, title = "Classification Stats")

        self.index = ClassifierModelSummaryComponent(explainer, hide_selector = True,
                                                     hide_cutoff = False, hide_footer = False
                                                    )

        self.contributions = ClassificationComponent(explainer,
                                                     hide_selector = True, hide_cutoff = False,
                                                     hide_footer = False, hide_percentage = True
                                                    )

    def layout(self):
        accuracy_box_graph, error_box_graph, accuracy_bar_graph, error_bar_graph = build_classification_graph()
        return dbc.Container([
            html.Br(),
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.H3(f"Classification Summary", id = "fi-label"),
                    ]),
                ]),
                dbc.CardBody(
                    html.Div([
                        # html.Br(),
                        html.H6(f'''
                                    Through the plots shown below, we can see that the Machine Learning (ML) model 
                                    trained with the highest Accuracy and Lowest error is the Random Forest model.
                                    Thus, the following explainability portion(s) will explain the Random Forest model in detail.
                                '''
                        ),
                        # html.Br()
                    ])
                )
            ]),
            html.Br(),
            # html.Br(),
            # dbc.Row([
            #     dbc.Col([
            #         dbc.Card([
            #             dbc.CardHeader([
            #                 html.Div([
            #                     html.H3(f"Accuracy Scores", id = "fi-label"),
            #                     html.H6(f"4 different Machine Learning models that we have trained with the Traffic Accident dataset yielded the following accuracy scores.", className = "card-subtitle")
            #                 ]),
            #             ]),
            #             dbc.CardBody(
            #                 html.Div([
            #                     # html.Br(),
            #                     accuracy_box_graph,
            #                     # html.Br()
            #                 ])
            #             )
            #         ])
            #     ]),
            #     dbc.Col([
            #         dbc.Card([
            #             dbc.CardHeader([
            #                 html.Div([
            #                     html.H3(f"Error Scores", id = "fi-label"),
            #                     html.H6(f"Mean Squared Error was used to measure this metric against the 4 different Machine Learning models that we have trained with the Traffic Accident dataset.", className = "card-subtitle")
            #                 ]),
            #             ]),
            #             dbc.CardBody(
            #                 html.Div([
            #                     # html.Br(),
            #                     error_box_graph,
            #                     # html.Br()
            #                 ])
            #             )
            #         ])
            #     ]),
            # ]),
            # html.Br(),
            # html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.H3(f"Accuracy Scores", id = "fi-label"),
                                html.H6(f"4 different Machine Learning models that we have trained with the Traffic Accident dataset yielded the following accuracy scores.", className = "card-subtitle")
                            ]),
                        ]),
                        dbc.CardBody(
                            html.Div([
                                # html.Br(),
                                accuracy_bar_graph,
                                # html.Br()
                            ])
                        )
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.H3(f"Error Scores", id = "fi-label"),
                                html.H6(f"Mean Squared Error was used to measure this metric against the 4 different Machine Learning models that we have trained with the Traffic Accident dataset.", className = "card-subtitle")
                            ]),
                        ]),
                        dbc.CardBody(
                            html.Div([
                                # html.Br(),
                                error_bar_graph,
                                # html.Br()
                            ])
                        )
                    ])
                ]),
            ]),
            html.Br()
        ])
    
class CustomFeatureImportanceTab(ExplainerComponent):
    def __init__(self, explainer, name = None):
        super().__init__(explainer, title = "Feature Importance")
    
    def make_fi_graph(self):
        pkl_fi_filename = res_sev_dir + "severity_feature_importance.pkl"
        with open(pkl_fi_filename, 'rb') as file:
            fi_scores = pickle.load(file)
        
        fi_scores_fig = px.bar(fi_scores, x = 'Importance_Score', y = 'Feature', color = 'Category',
                                # title = "Feature Importance for Random Forest model",
                                labels = {
                                    "Importance_Score": "Importance Score (%)"
                                }
                            )
        fi_scores_fig.update_layout(
            plot_bgcolor = 'white',
            margin_l = 200,
            dragmode = False,
            yaxis = dict(
                titlefont = dict(size = 16),
                tickfont = dict(size = 14),
                title_standoff = 20,
                categoryorder = 'array',
                categoryarray = fi_scores['Feature']
            ),
            xaxis = dict(
                titlefont = dict(size = 16),
                tickfont = dict(size = 14),
                tickformat = ',.0%'
            )
        )
        fi_scores_fig.update_xaxes(
            mirror = True,
            ticks = 'outside',
            showline = True,
            linecolor = 'black',
            gridcolor = 'lightgrey'
        )
        fi_scores_fig.update_yaxes(
            mirror = True,
            ticks = 'outside',
            showline = True,
            linecolor = 'black',
            gridcolor = 'lightgrey'
        )

        return fi_scores_fig

    def layout(self):
        fi_graph = self.make_fi_graph()

        return dbc.Container([
            html.Br(),
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.H3(f"Feature Importance", id = "fi-label"),
                        html.H6(f"Which features had the biggest impact?", className = "card-subtitle")
                    ]),
                ]),
                dbc.CardBody(
                    dcc.Graph(id = "fig_fi_scores", figure = fi_graph,
                                responsive = True,
                                style = {
                                    "width": "100%",
                                    "height": "100%"
                                }
                            )
                    , id = 'sev-contrib_card'
                )
            ], style={'height': '750px'}),
            html.Br()
        ])
    
class CustomFeatureDependenceTab(ExplainerComponent):
    def __init__(self, explainer, name = None):
        super().__init__(explainer, title = "Feature Dependence")
        
        self.shap_summary = ShapSummaryComponent(explainer,
                                hide_type = True, summary_type = 'aggregate',
                                hide_selector = True, depth = 10
                                )
        self.shap_dependence = ShapDependenceComponent(explainer,
                                hide_selector = True, hide_index = True,
                                col = 'Pressure(in)', color_col = 'Humidity(%)',
                                )
        self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)

    def layout(self):
        feature_list = sorted(X.columns.to_list())

        return dbc.Container([
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.H3(f"Partial Dependence Plot", id = "fi-label"),
                                html.H6(f"Correlation between feature and the impact to the Traffic Accident Severity", className = "card-subtitle")
                            ]),
                        ]),
                        dbc.CardBody(
                            html.Div([
                                html.Label('Feature:', className = 'form-label'),
                                dcc.Dropdown(feature_list, 'Pressure(in)', id = 'sev-feature-dropdown'),
                                dcc.Loading(
                                    id = "loading-component",
                                    type = "default",
                                    children = html.Div(
                                                        id = "sev-pdp-output",
                                                        children = [
                                                            html.Br(),
                                                            html.Br(),
                                                            html.Br()
                                                        ]
                                                    )
                                ),
                                html.Br(),
                                html.Br(),
                                html.Br()
                            ])
                        )
                    ]),
                ]),
                dbc.Col([
                    self.shap_dependence.layout()
                ])
            ]),
            html.Br()
        ])

    @app.callback(
        Output('sev-pdp-output', 'children'),
        Input('sev-feature-dropdown', 'value')
    )
    def update_output(value):
        graph_layout = []

        if value is not None and value != "":
            feature_delta = get_pdp_values(value)
            pdp_graph = build_pdp_graph(feature_delta, value)
            pdp_feature_explaination = {
                'Pressure(in)' : "We can see that there is a negative trend for the feature Pressure. As the surrounding Pressure rises, the Severity to an accident decreases.",
                'Humidity(%)' : "We can see that there is a positive trend for the feature Humidity. As the surrounding Humidity rises, the Severity to an accident also increases.",
                'Precipitation(in)' : "", # To check
                'Temperature(F)' : "We can see that there is a positive trend for the feature Temperature. As the surrounding Temperature increases, there is an also an increase in Severity to an accident.",
                'Visibility(mi)' : "", # To check
                'Wind_Chill(F)': "We can see that there is a slight positive trend for the feature Wind_Chill. As the surrounding Wind_Chill increases, there is an also an increase in Severity to an accident.",
                'Wind_Speed(mph)': "There is no visible trend seen for the feature Wind_Speed as the Severity impact seems to be at random.",
                'Weather_Condition': "Since this feature value is integer encoded, there will be no clear trend to be seen in a PDP plot",
                'Wind_Direction': "Since this feature value is integer encoded, there will be no clear trend to be seen in a PDP plot",
                'City': "Since this feature value is integer encoded, there will be no clear trend to be seen in a PDP plot",
                'County': "Since this feature value is integer encoded, there will be no clear trend to be seen in a PDP plot",
                'State': "Since this feature value is integer encoded, there will be no clear trend to be seen in a PDP plot",
                'Amenity': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Bump': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Crossing': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Give_Way': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Junction': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'No_Exit': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Railway': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Roundabout': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Station': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Stop': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Sunrise_Sunset': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Traffic_Calming': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Traffic_Signal': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
                'Turning_Loop': "Since this feature value is of either 1 (True) or 0 (False), there is no conceivable PDP to be plot",
            }
            graph_layout = [
                html.Br(),
                html.Br(),
                html.H6(pdp_feature_explaination[value], className = "card-subtitle"),
                dcc.Graph(id = "fig_severity", figure = pdp_graph)
            ]
        
        return graph_layout

class CustomIndividualPredictionTab(ExplainerComponent):
    def __init__(self, explainer, name = "SevIndividualPrediction"):
        super().__init__(explainer, title = "Individual Predictions")
        
        self.current_index = 1
        self.index = ClassifierRandomIndexComponent(explainer, name = "SevClassRandIndex",
                                                    subtitle = "Select which record to view its details",
                                                    hide_selector = True, index = self.current_index,
                                                    hide_pred_or_perc = True, hide_slider = True, pos_label = None
                                                    )

        self.prediction_summary = ClassifierPredictionSummaryComponent(explainer,
                                                    hide_selector = True, index = self.current_index,
                                                    hide_index = True, pos_label = None
                                                    )
        
        self.connector = IndexConnector(self.index, [self.prediction_summary])
        self.persistent_contrib_dict = {}

    def layout(self):
        pos_label = int(self.explainer.preds[self.current_index - 1]) - 1
        dataframe = self.explainer.get_contrib_summary_df(str(self.current_index), topx = 26, sort = "abs", pos_label = pos_label)
        self.persistent_contrib_dict = parse_contribution_record(dataframe)
        self.contrib_fig = build_contribution_graph(dataframe)
        
        return dbc.Container([
            html.Br(), 
            dbc.Row([
                dbc.Col([
                    self.index.layout()
                ]),
                dbc.Col([
                    self.prediction_summary.layout()
                ]),
            ]),
            html.Br(),
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.H3(f"Chosen Record", id = "random-index-record"),
                        html.H6(f"Record's Details", className = "card-subtitle")
                    ]),
                ]),
                dbc.CardBody(
                    id = 'sev-contrib_card',
                    children = [
                        dash_table.DataTable(
                            self.persistent_contrib_dict.to_dict('records'),
                            [{"name": i, "id": i} for i in self.persistent_contrib_dict.columns],
                            cell_selectable = False,
                            fixed_columns = {'headers': True, 'data': 1},
                            style_header = {
                                'backgroundColor': 'whitesmoke',
                                'fontWeight': 'bold'
                            },
                            style_cell = {
                                'textAlign': 'left',
                                'font-family':'sans-serif',
                                'minWidth': '100%'},
                            style_table = {'overflowX': 'auto', 'minWidth': '100%'},
                            style_data_conditional = [{
                                    'if': {"column_id": "Feature"},
                                    'width': '80px'
                            }],
                            id = 'sev-contrib_tbl'
                        ),
                        html.Br(),
                        self.contrib_fig,
                        html.Br()
                    ],
                    style = {'minHeight':'650px'}
                ),
            ], class_name = "h-100"),
            html.Br(),
            html.Br()
        ])

###
# Main Functions for Severity - Explainability
###
@app.callback(Output('sev-tab-output', 'children'),
              Input('sev-tabs-options', 'value'))
def render_content(tab):
    if tab == 'classification-stats':
        return classification_stats_tab.layout()
    elif tab == 'feature-importance':
        return feature_importance_tab.layout()
    elif tab == 'feature-dependence':
        return feature_dependence_tab.layout()
    elif tab == 'individual-predictions':
        return indiv_pred_tab.layout()

### Functions for CustomClassificationTab
def read_list_from_txt(path):
    file = open(path, "r")
    
    content = file.read()
    content_list = content.split(", ")
    content_list = [float(x) for x in content_list]
    return content_list

def build_classification_graph():
    algo_list = ['LR', 'RF', 'KNN', 'DT']
    basePath = res_sev_dir + 'model_data/'

    lr_accuracy_path = basePath + 'lr_accuracy.txt'
    # lr_error_path = basePath + 'lr_error.txt'
    lr_mse_path = basePath + 'lr_mse.txt'

    rf_accuracy_path = basePath + 'rf_accuracy.txt'
    # rf_error_path = basePath + 'rf_error.txt'
    rf_mse_path = basePath + 'rf_mse.txt'

    knn_accuracy_path = basePath + 'knn_accuracy.txt'
    # knn_error_path = basePath + 'knn_error.txt'
    knn_mse_path = basePath + 'knn_mse.txt'

    dt_accuracy_path = basePath + 'dt_accuracy.txt'
    # dt_error_path = basePath + 'dt_error.txt'
    dt_mse_path = basePath + 'dt_mse.txt'

    ## Load Metrics List
    lr_accuracy_list = read_list_from_txt(lr_accuracy_path)
    rf_accuracy_list = read_list_from_txt(rf_accuracy_path)
    knn_accuracy_list = read_list_from_txt(knn_accuracy_path)
    dt_accuracy_list = read_list_from_txt(dt_accuracy_path)

    lr_error_list = read_list_from_txt(lr_mse_path)
    rf_error_list = read_list_from_txt(rf_mse_path)
    knn_error_list = read_list_from_txt(knn_mse_path)
    dt_error_list = read_list_from_txt(dt_mse_path)

    ml_accuracy_list = [lr_accuracy_list, rf_accuracy_list, knn_accuracy_list, dt_accuracy_list]
    accuracy_list = []
    for mlModels in ml_accuracy_list:
        accuracy_list.append(np.mean(mlModels) * 100)
    
    ml_error_list = [lr_error_list, rf_error_list, knn_error_list, dt_error_list]
    error_list = []
    for mlModels in ml_error_list:
        error_list.append(np.mean(mlModels) * 100)

    accuracy_box_graph = build_accuracy_box_graph(algo_list, ml_accuracy_list)
    accuracy_bar_graph = build_accuracy_bar_graph(algo_list, accuracy_list)
    error_box_graph = build_error_box_graph(algo_list, ml_error_list)
    error_bar_graph = build_error_bar_graph(algo_list, error_list)

    return accuracy_box_graph, error_box_graph, accuracy_bar_graph, error_bar_graph

def build_accuracy_box_graph(algo_list, accuracy_list):
    lr_repeated = [algo_list[0]] * len(accuracy_list[0])
    lr_combined_df = pd.DataFrame({'model': lr_repeated, 'accuracy': accuracy_list[0]})

    rf_repeated = [algo_list[1]] * len(accuracy_list[1])
    rf_combined_df = pd.DataFrame({'model': rf_repeated, 'accuracy': accuracy_list[1]})

    knn_repeated = [algo_list[2]] * len(accuracy_list[2])
    knn_combined_df = pd.DataFrame({'model': knn_repeated, 'accuracy': accuracy_list[2]})

    dt_repeated = [algo_list[3]] * len(accuracy_list[3])
    dt_combined_df = pd.DataFrame({'model': dt_repeated, 'accuracy': accuracy_list[3]})

    combined_df = pd.concat([lr_combined_df, rf_combined_df, knn_combined_df, dt_combined_df], ignore_index=True, sort=False)
    combined_df['accuracy'] = combined_df['accuracy'] * 100
    fig = px.box(combined_df, x = 'model', y = 'accuracy', color = 'model',
                    labels = {
                        "model": "ML Model",
                        "accuracy": "Accuracy (%)"
                    }
                )
    
    new = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'KNN': 'K-Nearest Neighbour', 'DT': 'Decision Trees'}
    fig.for_each_trace(lambda t: t.update(name = new[t.name]))
    fig.update_yaxes(
        mirror = True,
        ticks = 'outside',
        showline = True,
        linecolor = 'black',
        gridcolor = 'lightgrey',
        title_standoff = 15
    )
    fig.update_xaxes(
        mirror = True,
        ticks = 'outside',
        showline = True,
        linecolor = 'black',
        # gridcolor='lightgrey'
    )
    acc_graph = dcc.Graph(
                        figure = fig,
                        responsive = True,
                        style = {
                            "width": "100%",
                            "height": "100%"
                        }
                )
    return acc_graph

def build_error_box_graph(algo_list, error_list):
    lr_repeated = [algo_list[0]] * len(error_list[0])
    lr_combined_df = pd.DataFrame({'model': lr_repeated, 'error': error_list[0]})

    rf_repeated = [algo_list[1]] * len(error_list[1])
    rf_combined_df = pd.DataFrame({'model': rf_repeated, 'error': error_list[1]})

    knn_repeated = [algo_list[2]] * len(error_list[2])
    knn_combined_df = pd.DataFrame({'model': knn_repeated, 'error': error_list[2]})

    dt_repeated = [algo_list[3]] * len(error_list[3])
    dt_combined_df = pd.DataFrame({'model': dt_repeated, 'error': error_list[3]})

    combined_df = pd.concat([lr_combined_df, rf_combined_df, knn_combined_df, dt_combined_df], ignore_index=True, sort=False)
    fig = px.box(combined_df, x = 'model', y = 'error', color = 'model',
                    labels = {
                        "model": "ML Model",
                        "error": "Error"
                    }
                )
    fig.update_layout(yaxis_range = [0.13, 0.19], plot_bgcolor = 'white')
    new = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'KNN': 'K-Nearest Neighbour', 'DT': 'Decision Trees'}
    fig.for_each_trace(lambda t: t.update(name = new[t.name]))
    fig.update_yaxes(
        mirror = True,
        ticks = 'outside',
        showline = True,
        linecolor = 'black',
        gridcolor = 'lightgrey',
        title_standoff = 15
    )
    fig.update_xaxes(
        mirror = True,
        ticks = 'outside',
        showline = True,
        linecolor = 'black'
    )
    err_graph = dcc.Graph(
                        figure = fig,
                        responsive = True,
                        style = {
                            "width": "100%",
                            "height": "100%"
                        }
                )
    return err_graph

def build_accuracy_bar_graph(algo_list, accuracy_list):
    percentile_list = pd.DataFrame({'model': algo_list, 'accuracy': accuracy_list})
    fig = px.bar(percentile_list, x='model', y='accuracy', color='model', text_auto = True,
                labels = {
                    'model': 'ML Model',
                    'accuracy': 'Accuracy (%)'
                })
    fig.update_layout(yaxis_range=[92, 94], plot_bgcolor='white')
    fig.update_traces(textfont_size = 14, textangle = 0, textposition = "outside")
    new = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'KNN': 'K-Nearest Neighbour', 'DT': 'Decision Trees'}
    fig.for_each_trace(lambda t: t.update(name = new[t.name]))
    fig.update_yaxes(
        mirror = True,
        ticks = 'outside',
        showline = True,
        linecolor = 'black',
        gridcolor = 'lightgrey'
    )
    fig.update_xaxes(
        mirror = True,
        ticks = 'outside',
        showline = True,
        linecolor = 'black',
        # gridcolor='lightgrey'
    )
    acc_graph = dcc.Graph(
                        figure = fig,
                        responsive = True,
                        style = {
                            "width": "100%",
                            "height": "400px"
                        }
                )
    return acc_graph

def build_error_bar_graph(algo_list, error_list):
    percentile_list = pd.DataFrame({'model': algo_list, 'error': error_list})
    fig = px.bar(percentile_list, x='model', y='error', color='model', text_auto = True,
                labels = {
                    'model': 'ML Model',
                    'error': 'Mean Squared Error'
                })
    fig.update_layout(yaxis_range=[14, 15.5], plot_bgcolor='white')
    fig.update_traces(textfont_size = 14, textangle = 0, textposition = "outside")
    new = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'KNN': 'K-Nearest Neighbour', 'DT': 'Decision Trees'}
    fig.for_each_trace(lambda t: t.update(name = new[t.name]))
    fig.update_yaxes(
        mirror = True,
        ticks = 'outside',
        showline = True,
        linecolor = 'black',
        gridcolor = 'lightgrey'
    )
    fig.update_xaxes(
        mirror = True,
        ticks = 'outside',
        showline = True,
        linecolor = 'black',
        # gridcolor='lightgrey'
    )
    err_graph = dcc.Graph(
                        figure = fig,
                        responsive = True,
                        style = {
                            "width": "100%",
                            "height": "400px"
                        }
                )
    return err_graph

### Functions for CustomFeatureDependenceTab
def get_pdp_values(feature):
    ## Creating Empty DataFrame and Storing it in variable _df
    pred_avg_sev_delta_df = pd.DataFrame()

    severity_pkl_model_filename = res_sev_dir + "severity_rfc_best_model_sampled.pkl"
    with open(severity_pkl_model_filename, 'rb') as file:
        model = pickle.load(file)
    
    X_copy = X.copy()
    y_copy = y.copy()

    list_of_values = sorted(X_copy[feature].tolist())
    
    list_of_feature_values, pred_avg_sev_delta_ls = [], []

    Q3 = np.quantile(X_copy[feature], 0.75)
    Q1 = np.quantile(X_copy[feature], 0.25)
    print('Feature: ', feature, ', Q1: ', Q1, ', Q3:', Q3)

    sev_range_delta = np.sum(y_copy.unique()) / y_copy.nunique()
    print('Severity Delta: ', sev_range_delta)

    # Iterate through values in selected feature and predict its Severity
    for i in list_of_values: 
        if i >= Q1 and i <= Q3:
            X_copy[feature] = i
            y_pred = model.predict(X_copy)
            
            # Predicted Severity (Average)
            y_pred_avg = sum(y_pred) / len(y_pred)
            sev_delta = y_pred_avg - sev_range_delta
            pred_avg_sev_delta_ls.append(sev_delta)

            list_of_feature_values.append(i)

    pred_avg_sev_delta_df['Severity'] = pred_avg_sev_delta_ls
    pred_avg_sev_delta_df[feature] = list_of_feature_values

    return pred_avg_sev_delta_df

def build_pdp_graph(feature_delta, feature_name):
    pdp_fig = px.line(feature_delta, x = feature_name, y = 'Severity',
                        labels = {
                            "Severity": "Severity Impact",
                        }
                    )
    pdp_fig.add_hline(y = 0)
    pdp_fig.update_layout(
        plot_bgcolor = 'white',
        dragmode = False,
        yaxis = dict(
            titlefont = dict(size = 16),
            tickfont = dict(size = 14),
            title_standoff = 40
        ),
        xaxis = dict(
            titlefont = dict(size = 16),
            tickfont = dict(size = 14)
        )
    )
    pdp_fig.update_xaxes(
        mirror = True,
        ticks = 'outside',
        showline = True,
        linecolor = 'black',
        gridcolor = 'lightgrey'
    )
    pdp_fig.update_yaxes(
        mirror = True,
        ticks = 'outside',
        showline = True,
        linecolor = 'black',
        gridcolor = 'lightgrey'
    )

    return pdp_fig
    
### Functions for CustomIndividualPredictionTab
def parse_contribution_record(records):
    true_false_list = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]
    records_dict = str(records.to_dict('records'))
    records_dict = records_dict.replace("'", "\"")
    records_list = json.loads(records_dict)

    states, counties, cities, weather_conditions, wind_directions, day_night = get_ordinal_values()

    new_records_list = {}
    for record_dict in records_list[1:-2]:
        new_record_dict = {}
        for key, value in record_dict.items():
            if key == 'Reason' and '=' in value:
                feature, feature_value = [word.strip() for word in value.split('=')]
                if feature in true_false_list:
                    new_record_dict['Value'] = "True" if feature_value == "1.0" else "False"
                elif feature == 'City':
                    new_record_dict['Value'] = cities.get(float(feature_value))
                elif feature == 'State':
                    new_record_dict['Value'] = states.get(float(feature_value))
                elif feature == 'County':
                    new_record_dict['Value'] = counties.get(float(feature_value))
                elif feature == 'Wind_Direction':
                    new_record_dict['Value'] = wind_directions.get(float(feature_value))
                elif feature == 'Weather_Condition':
                    new_record_dict['Value'] = weather_conditions.get(float(feature_value))
                elif feature == 'Sunrise_Sunset':
                    new_record_dict['Value'] = day_night.get(float(feature_value))
                else:
                    new_record_dict['Value'] = feature_value
            elif key == 'Effect':
                new_record_dict[key] = value
            
        new_records_list[feature] = new_record_dict
    
    new_records_df = pd.DataFrame(new_records_list)
    return new_records_df.rename_axis("Feature").reset_index()

def build_contribution_graph(dataframe):
    contrib_df = dataframe[1:-2] # Remove summary rows
    
    contrib_df[['Reason', 'Value']] = contrib_df['Reason'].str.split('=', expand = True)
    contrib_df.loc[:, 'Effect'] = contrib_df.loc[:, 'Effect'].str[:-1]
    contrib_df.loc[:, 'Effect'] = contrib_df.loc[:, 'Effect'].astype(float)
    contrib_df = contrib_df.reindex(contrib_df['Effect'].abs().sort_values(ascending = False).index).head(10)

    fig = px.bar(contrib_df, x = "Reason", y = "Effect",
        color = contrib_df.loc[:, 'Effect'] > 0,
        category_orders = dict(Reason = contrib_df.Reason.to_list()),
        color_discrete_map = {True: "green", False: "red"},
        text_auto = True,
        labels = {
            "Effect": "SHAP Effect (%)"
        }
    ).update_layout(
        xaxis_dtick = "M1", legend_title_text = "Effect Legend", showlegend = True,
        xaxis = dict(type = 'category')
    )

    new = {'True': 'Positive', 'False': 'Negative'}
    fig.for_each_trace(lambda t: t.update(name = new[t.name]))
    fig.update_traces(hovertemplate = None, hoverinfo = 'skip')

    fig_graph_element = dcc.Graph(
                             figure = fig,
                            responsive = True,
                                style = {
                                    "width": "100%",
                                    "height": "100%"
                                }
                        )

    return fig_graph_element

def get_ordinal_values():
    resources_dir = 'resources/data/'
    states, counties, cities, weather_conditions, wind_directions = {}, {}, {}, {}, {}

    states_df = pd.read_pickle(resources_dir + 'states.pkl').to_dict('records')
    for state in states_df:
        states[state['id']] = state['State_Name']

    counties_df = pd.read_pickle(resources_dir + 'counties.pkl').to_dict('records')
    for county in counties_df:
        counties[county['id']] = county['County']

    cities_df = pd.read_pickle(resources_dir + 'cities.pkl').to_dict('records')
    for city in cities_df:
        cities[city['id']] = city['City']

    weather_condition_df = pd.read_pickle(resources_dir + 'weather_condition.pkl').to_dict('records')
    for weather_condition in weather_condition_df:
        weather_conditions[weather_condition['id']] = weather_condition['Weather']

    wind_direction_df = pd.read_pickle(resources_dir + 'wind_direction.pkl').to_dict('records')
    for wind_direction in wind_direction_df:
        wind_directions[wind_direction['id']] = wind_direction['Wind_Direction']

    day_night = {1: 'Night', 2: 'Day'}
    
    return states, counties, cities, weather_conditions, wind_directions, day_night
    
@app.callback(
    Output("sev-contrib_card", "children"),
    Input("random-index-clas-index-SevClassRandIndex", "value")
)
def get_updated_index(index):
    return_array = []

    if index is not None and index != "":
        pos_label = int(classifier_explainer.preds[int(index) - 1]) - 1
        contrib_df = classifier_explainer.get_contrib_summary_df(str(index), topx = 26, sort = "abs", pos_label = pos_label)
        persistent_contrib_dict = parse_contribution_record(contrib_df)
        contrib_fig = build_contribution_graph(contrib_df)

        return_array = [
            dash_table.DataTable(
                persistent_contrib_dict.to_dict('records'),
                [{"name": i, "id": i} for i in persistent_contrib_dict.columns],
                cell_selectable = False,
                fixed_columns = {'headers': True, 'data': 1},
                style_header = {
                    'backgroundColor': 'whitesmoke',
                    'fontWeight': 'bold'
                },
                style_cell = {
                    'textAlign': 'left',
                    'font-family':'sans-serif',
                    'minWidth': '100%'},
                style_table = {'overflowX': 'auto', 'minWidth': '100%'},
                style_data_conditional=[{
                        'if': {"column_id": "Feature"},
                        'width': '80px'
                }],
                id = 'sev-contrib_tbl'
            ),
            html.Br(),
            contrib_fig,
            html.Br()
        ]
    
    return return_array

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

classifier_explainer = decompress_pickle(res_sev_dir + 'severity_model_explainer_records.pbz2') 
# classifier_explainer = ClassifierExplainer.from_file(res_sev_dir + 'severity_model_explainer_records.joblib')
indiv_pred_tab = CustomIndividualPredictionTab(classifier_explainer)
indiv_pred_tab.register_callbacks(app)
classification_stats_tab = CustomClassificationTab(classifier_explainer)
classification_stats_tab.register_callbacks(app)
feature_importance_tab = CustomFeatureImportanceTab(classifier_explainer)
feature_importance_tab.register_callbacks(app)
feature_dependence_tab = CustomFeatureDependenceTab(classifier_explainer)
feature_dependence_tab.register_callbacks(app)

server = app.server
layout = html.Div([
    html.Br(),
    html.Div([
        html.H1(children = 'Model Explainer: Severity', 
            style = {'margin-left':'10px', 'margin-bottom':'1.5rem'})
    ], style = {"margin-left": "auto", "margin-right": "auto", "width": "90%"}),
    html.Div([
        dcc.Tabs(id = "sev-tabs-options", value = 'classification-stats', children = [
            dcc.Tab(label = 'Classification Stats', value = 'classification-stats'),
            dcc.Tab(label = 'Feature Importance', value = 'feature-importance'),
            dcc.Tab(label = 'Feature Dependence', value = 'feature-dependence'),
            dcc.Tab(label = 'Individual Predictions', value = 'individual-predictions'),
        ]),
    ], className = 'layout'),
    html.Div(id = 'sev-tab-output')
])