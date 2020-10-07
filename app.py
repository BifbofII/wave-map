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
    modal_bg = 'rgba(0,0,0,0.4)',
)
padding = '30px'
border_width = '4px'

modal_style = {
    'display': 'none',
    'position': 'fixed',
    'z-index': 1005,
    'left': 0,
    'top': 0,
    'width': '100vw',
    'height': '100vh',
    'overflow': 'auto',
    'background-color': colours['modal_bg'],
}

wave_info = """
    On this map, a prediction of the wave height and direction can be seen.
    The prediction is based on the wind data from within the last 6 hours.
    Marked as the green dots are the locations of five wave buoys.
    The prediction model was trained based on data of these five buoys.

    Since this is only a proof on concept, only predictions for the time between the 01.08.2020 and the 07.08.2020 (in four hour steps) are available.

    The map is interactive, can be zoomed in and more information is shown when hovering with the mouse over the map.
"""

wind_info = """
    Shown on this map is wind data from the [Copernicus Project](https://www.copernicus.eu/en).
    This data was used as the predictive variables for generating the wave model.

    The map is interactive, can be zoomed in and more information is shown when hovering with the mouse over the map.
"""

control_info = """
    Here you can select what data you want to see.

    First you can select wether you want to see precicted data about waves in the baltic sea or wind data in the region.

    Next you can select the time for which you want to see the data.
    First select a date, then the time.

    Lastly you can select which layers should be shown:
    * Map outline: The outline of the countries around the baltic sea
    * Coloured contour: A contour plot of either the wave height or the wind speed
    * Direction arrows: Arrows pointing in the direction of the waves or the wind
    * Buoy locations: The locations of the wave buoys used for trainging the data model

    Since this is only a proof on concept, only predictions for the time between the 01.08.2020 and the 07.08.2020 (in four hour steps) are available.
"""

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
                        html.Div(className='row', children=[
                            html.H2('Controls', className='eight columns'),
                            html.Button('Info', id='ctl_modal_open', className='four columns')
                        ]),
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
                        html.Div(className='row', children=[
                            html.Div(className='column eight columns', children=[
                                html.H2('Output', id='output_title'),
                                html.H3('Date', id='output_date'),
                            ]),
                            html.Button('Info', id='map_modal_open', className='four columns'),
                        ]),
                        dcc.Graph(id='map_plot'),
                    ]
                ),
            ]),
    ]),
    html.Div(id='map_modal', style=modal_style, children=[
        html.Div(
            style={
                'width': '60vw',
                'margin': '10% auto',
                'padding': '10px 15px',
                'background-color': colours['bg_2'],
                'border': border_width,
                'border-style': 'solid',
                'border-color': colours['border'],
            },
            children=[
                html.Div(className='row', children=[
                    html.H2('Information about the Data', className='ten columns'),
                    html.Button('Close', id='map_modal_close', className='two columns'),
                ]),
                dcc.Markdown(id='map_modal_content'),
            ],
        ),
    ]),
    html.Div(id='ctl_modal', style=modal_style, children=[
        html.Div(
            style={
                'width': '60vw',
                'margin': '10% auto',
                'padding': '10px 15px',
                'background-color': colours['bg_2'],
                'border': border_width,
                'border-style': 'solid',
                'border-color': colours['border'],
            },
            children=[
                html.Div(className='row', children=[
                    html.H2('Information about the Controls', className='ten columns'),
                    html.Button('Close', id='ctl_modal_close', className='two columns'),
                ]),
                dcc.Markdown(id='ctl_modal_content', children=control_info),
            ],
        ),
    ]),
])

@app.callback(
    [
        Output(component_id='map_plot', component_property='figure'),
        Output(component_id='output_title', component_property='children'),
        Output(component_id='output_date', component_property='children'),
        Output(component_id='map_modal_content', component_property='children'),
    ],
    [
        Input(component_id='sel_plot', component_property='value'),
        Input(component_id='sel_date', component_property='date'),
        Input(component_id='sel_time', component_property='value'),
        Input(component_id='sel_layers', component_property='value'),
    ]
)
def update_graph(plot_sel, date_sel, time_sel, layer_sel):
    """Update displayed graph"""
    time = datetime.datetime.strptime(date_sel, '%Y-%m-%d') + datetime.timedelta(hours=time_sel)
    time_ind = np.argmin([np.abs(t.total_seconds()) for t in times - time])
    if plot_sel == 'waves':
        title = 'Wave Prediction'
        data = wave_data[:,:,time_ind,:]
        mask = True
        cname = 'Wave height / m'
        info = wave_info
    elif plot_sel == 'wind':
        title = 'Wind Data'
        data = wind_data[:,:,time_ind,:]
        mask = False
        cname = 'Wind speed / m/s'
        info = wind_info

    fig = visualization.create_vis(data, layers=layer_sel, mask_land=mask, contour_name=cname)
    fig.update_layout(paper_bgcolor=colours['bg_2'], font_color=colours['text'])

    return fig, title, time.strftime('%d.%m.%Y %H:%M'), info

map_modal_callback = app.callback(
    Output(component_id='map_modal', component_property='style'),
    [
        Input(component_id='map_modal_open', component_property='n_clicks'),
        Input(component_id='map_modal_close', component_property='n_clicks'),
    ]
    )(lambda open_cl, close_cl: open_close_modal('map_modal_open', open_cl, close_cl))

ctl_modal_callback = app.callback(
    Output(component_id='ctl_modal', component_property='style'),
    [
        Input(component_id='ctl_modal_open', component_property='n_clicks'),
        Input(component_id='ctl_modal_close', component_property='n_clicks'),
    ]
    )(lambda open_cl, close_cl: open_close_modal('ctl_modal_open', open_cl, close_cl))

def open_close_modal(open_name, open_cl, close_cl):
    """Open or close the modal info box"""
    ctx = dash.callback_context
    modal_style['display'] = 'none'

    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id == open_name:
            modal_style['display'] = 'block'

    return modal_style

if __name__ == '__main__':
    app.run_server(debug=True)