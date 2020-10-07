import pickle
import numpy as np
import pandas as pd
from memoization import cached

import plotly.graph_objects as go
import plotly.figure_factory as ff

buoy_data = pd.read_csv('data/wave_stations.csv', index_col=0)
lats = np.load('predictions/lats.npy')
lons = np.load('predictions/lons.npy')

mask = np.load('data/vis_mask.npy')
with open('data/map_traces.pkl', 'rb') as f:
    map_traces = pickle.load(f)

@cached
def create_vis(wave_data, title=None, layers=None, mask_land=True, contour_name=None):
    """
    Create a plotly figure displaying the predicted wave data

    :param wave_data: the wave data as a (lat x lon x (u/v)) numpy array
    :param layers: a list of the layers to include.
        Possible values are: 'map', 'contour', 'direction', 'buoys'
    :param mask_land: whether to mask the land area or not
    :returns: a plotly figure
    """
    if mask_land:
        wave_data[mask,:] = np.nan
    wave_height = np.sqrt(np.sum(wave_data**2, axis=2))
    buoy_trace = buoy_scatter(buoy_data)
    contour_trace = wave_contour(wave_height, lats, lons, contour_name)
    quiver_trace = wave_quivers(wave_data / np.concatenate([wave_height[:,:,np.newaxis], wave_height[:,:,np.newaxis]], axis=2) / 2,
        lats, lons)

    if layers is None:
        traces = map_traces + [contour_trace, quiver_trace, buoy_trace]
    else:
        traces = []
        for layer in layers:
            if layer == 'map':
                traces.extend(map_traces)
            elif layer == 'contour':
                traces.append(contour_trace)
            elif layer == 'direction':
                traces.append(quiver_trace)
            elif layer == 'buoys':
                traces.append(buoy_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(title=title, showlegend=False, margin=dict(l=0, r=0, b=0, t=0 if title is None else 50))
    fig.update_traces(marker=dict(size=20, color='MediumSeaGreen'), selector=dict(name='buoys'))
    fig.update_traces(line=dict(color='MediumSlateBlue'), selector=dict(name='directions'))
    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, visible=False, scaleanchor='x', scaleratio=1)
    return fig

def buoy_scatter(data):
    """
    Create a plotly trace of the buoy locations on a map

    :param data: a pandas dataframe with information about the buoys
    :returns: a plotly trace instance
    """
    lat = data['Latitude (decimals)']
    lon = data['Longitude (decimals)']
    hover_text = data['Observation station']
    return go.Scatter(x=lon, y=lat, hovertext=hover_text, name='buoys', mode='markers')

def wave_contour(data, lats, lons, contour_name):
    """
    Create a plotly contour trace of the wave height

    :param data: a 2D numpy array of the wave height at the locations
    :param lats: a 1D numpy array of latitude values
    :param lons: a 1D numpy array of longitude values
    :returns a plotly trace instance
    """
    return go.Contour(x=lons, y=lats, z=data, opacity=0.5, name='wave-contour',
        colorbar={'title': contour_name})

def wave_quivers(data, lats, lons):
    """
    Create a plotly quiver trace showing the wave direction

    :param data: a 3D numpy array of the wave direction data
    :param lats: a 1D numpy array of latitude values
    :param lons: a 1D numpy array of longitude values
    :returns a plotly trace instance
    """
    latmesh, lonmesh = np.meshgrid(lats, lons)
    u = data[:,:,0].T
    v = data[:,:,1].T
    quiver_obj = ff._quiver._Quiver(lonmesh, latmesh, u, v,
        scale=.15, arrow_scale=0.3, angle=np.pi/9)
    barb_x, barb_y = quiver_obj.get_barbs()
    arrow_x, arrow_y = quiver_obj.get_quiver_arrows()
    return go.Scatter(x=barb_x+arrow_x, y=barb_y+arrow_y, mode='lines', hoverinfo='skip',
        name='directions')

if __name__ == '__main__':
    wind_data = np.load('data/visualization_example/wind_dat.npy')
    fig = create_vis(wind_data)
    fig.show()