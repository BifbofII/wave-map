import visualization

import datetime
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash('baltic-wave-map')
server = app.server

wave_data = np.load('predictions/wave_data.npy')
wind_data = np.load('predictions/wind_data.npy')
times = np.load('predictions/times.npy', allow_pickle=True)

colours = dict(
    bg_1 = '#061233',
    bg_2 = '#030b1e',
    text = '#ffffff',
    border = '#ffffff',
)
padding = '30px'
border_width = '4px'

app.layout = html.Div([
    html.Header(style={
            'width': '100%',
            'padding': padding,
            'background-color': colours['bg_2'],
            'border': border_width,
            'border-color': colours['border'],
            'border-bottom-style': 'solid',
        },
        children=[
            html.H1('Baltic Wave Map'),
            html.A('View Source Code', href='https://github.com/bifbofii/wave-map'),
        ]
    ),
    html.Div(id='container',
        style={
            'position': 'relative',
            'width': '100%',
            'height': '100%',
            'max-width': '1400px',
            'margin': '0 auto',
            'padding': '0 20px',
            'box-sizing': 'border-box',
            'margin-top': padding,
        },
        children=[
            html.Div(id='cols', className='row', children=[
                html.Div(id='controls', className='four columns',
                    style={
                        'background-color': colours['bg_2'],
                        'border': border_width,
                        'border-color': colours['border'],
                        'border-style': 'solid',
                        'padding': padding,
                    },
                    children=[
                        html.H2('Controls'),
                        html.Div(id='plot_sel', children=[
                            html.H3('Plot Selection'),
                            dcc.Dropdown(id='sel_plot',
                                options=[
                                    {'label': 'Waves', 'value': 'waves'},
                                    {'label': 'Wind', 'value': 'wind'},
                                ],
                                multi=False, value='waves',
                            ),
                        ]),
                        html.Div(id='time_sel', children=[
                            html.H3('Time Selection'),
                            dcc.DatePickerSingle(id='sel_date',
                                min_date_allowed=times.min().date(),
                                max_date_allowed=times.max().date(),
                                initial_visible_month=times.min().date(),
                                date=times.min().date(),
                                display_format='DD.MM.YYYY'
                            ),
                            html.Div(
                                style={'margin-top': '15px'},
                                children=[
                                    dcc.Slider(id='sel_time',
                                        min=0,
                                        max=23,
                                        value=0,
                                        step=None,
                                        marks={int(i): '{}:00'.format(i)
                                            for i in np.unique([d.hour for d in times])},
                                    ),
                                ]
                            ),
                        ]),
                        html.Div(id='layer_sel', children=[
                            html.H3('Layer Selection'),
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
                    ]
                ),
                html.Div(id='display', className='eight columns',
                    style={
                        'background-color': colours['bg_2'],
                        'border': border_width,
                        'border-color': colours['border'],
                        'border-style': 'solid',
                        'padding': padding,
                    },
                    children=[
                        html.H2(id='output_title', children='Output'),
                        dcc.Graph(id='map_plot'),
                    ]
                ),
            ]),
    ]),
])

@app.callback(
    [
        Output(component_id='map_plot', component_property='figure'),
        Output(component_id='output_title', component_property='children'),
    ],
    [
        Input(component_id='sel_plot', component_property='value'),
        Input(component_id='sel_date', component_property='date'),
        Input(component_id='sel_time', component_property='value'),
        Input(component_id='sel_layers', component_property='value'),
    ]
)
def update_graph(plot_sel, date_sel, time_sel, layer_sel):
    """Dash callback function"""
    time = datetime.datetime.strptime(date_sel, '%Y-%m-%d') + datetime.timedelta(hours=time_sel)
    time_ind = np.argmin([np.abs(t.total_seconds()) for t in times - time])
    if plot_sel == 'waves':
        title = 'Wave Prediction - ' + time.strftime('%d.%m.%Y %H:%M')
        data = wave_data[:,:,time_ind,:]
        mask = True
        cname = 'Wave height / m'
    elif plot_sel == 'wind':
        title = 'Wind Data - ' + time.strftime('%d.%m.%Y %H:%M')
        data = wind_data[:,:,time_ind,:]
        mask = False
        cname = 'Wind speed / m/s'

    fig = visualization.create_vis(data, layers=layer_sel, mask_land=mask, contour_name=cname)
    fig.update_layout(paper_bgcolor=colours['bg_2'], font_color=colours['text'])

    return fig, title

if __name__ == '__main__':
    app.run_server(debug=True)