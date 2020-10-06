import visualization

import datetime
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash('baltic-wave-map')
server = app.server

wind_data = np.load('data/visualization_example/wind_dat.npy')

app.layout = html.Div([
    html.H1('Baltic Wave Map', style={'text-align': 'center'}),
    html.Div(id='controls', children=[
        dcc.Dropdown(id='sel_plot',
            options=[{'label': 'Waves', 'value': 'waves'}, {'label': 'Wind', 'value': 'wind'}],
            multi=False, value='waves',
            style={'width': '40%'}
        ),
        html.Div(id='time_sel', children=[
            dcc.DatePickerSingle(id='sel_date',
                min_date_allowed=datetime.date(2020, 8, 1),
                max_date_allowed=datetime.date(2020, 8, 7),
                initial_visible_month=datetime.date(2020, 8, 5),
                date=datetime.date(2020, 8, 1),
                display_format='DD.MM.YYYY'
            ),
            dcc.Slider(id='sel_time',
                min=0,
                max=23,
                value=0,
                step=None,
                marks={i:'{}:00'.format(i) for i in range(0,24,4)}
            ),
        ]),
        dcc.Checklist(id='sel_layers',
            options=[
                {'label': 'Map Outline', 'value': 'map'},
                {'label': 'Coloured Contour', 'value': 'contour'},
                {'label': 'Direction Arrows', 'value': 'direction'},
                {'label': 'Buoy Locations', 'value': 'buoys'},
            ],
            value=['map', 'contour', 'direction', 'buoys']
        ),
    ]),
    html.Div(id='display', children=[
        dcc.Graph(id='map_plot')
    ]),
])

@app.callback(
    Output(component_id='map_plot', component_property='figure'),
    [
        Input(component_id='sel_plot', component_property='value'),
        Input(component_id='sel_date', component_property='date'),
        Input(component_id='sel_time', component_property='value'),
        Input(component_id='sel_layers', component_property='value'),
    ]
)
def update_graph(plot_sel, date_sel, time_sel, layer_sel):
    """Dash callback function"""
    if plot_sel == 'waves':
        title = 'Wave Prediction'
    elif plot_sel == 'wind':
        title = 'Wind Data'

    return visualization.create_vis(wind_data, title=title, layers=layer_sel)

if __name__ == '__main__':
    app.run_server(debug=True)