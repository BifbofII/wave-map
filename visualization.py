import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.figure_factory as ff

from mpl_toolkits.basemap import Basemap

buoy_data = pd.read_csv('data/wave_stations.csv', index_col=0)
lats = np.load('data/visualization_example/lat.npy')
lons = np.load('data/visualization_example/lon.npy')

def create_vis(wave_data):
    """
    Create a plotly figure displaying the predicted wave data

    :param wave_data: the wave data as a (lat x lon x (u/v)) numpy array
    :returns: a plotly figure
    """
    wave_height = np.sqrt(np.sum(wave_data**2, axis=2))
    buoy_trace = buoy_scatter(buoy_data)
    contour_trace = wave_contour(wave_height, lats, lons)
    quiver_trace = wave_quivers(wave_data / np.concatenate([wave_height[:,:,np.newaxis], wave_height[:,:,np.newaxis]], axis=2) / 2,
        lats, lons)
    map_traces = get_map_traces(resolution='i')
    fig = go.Figure(data=map_traces+[contour_trace, quiver_trace, buoy_trace])
    fig.update_layout(title='Wave Prediction', mapbox_style='open-street-map')
    fig.update_traces(marker=dict(size=20, color='MediumSeaGreen'), selector=dict(name='buoys'))
    fig.update_traces(line=dict(color='MediumSlateBlue'), selector=dict(name='directions'))
    fig.update_xaxes(showgrid=False, visible=False, range=(16,34))
    fig.update_yaxes(showgrid=False, visible=False, range=(59,66))
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

def wave_contour(data, lats, lons):
    """
    Create a plotly contour trace of the wave height

    :param data: a 2D numpy array of the wave height at the locations
    :param lats: a 1D numpy array of latitude values
    :param lons: a 1D numpy array of longitude values
    :returns a plotly trace instance
    """
    return go.Contour(x=lons, y=lats, z=data, opacity=0.5, name='wave-contour')

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

def polygons_to_traces(poly_paths, N_poly, m):
    """ 
    Turn polygons into traces

    Credit: https://chart-studio.plotly.com/~Dreamshot/9144/import-plotly-plotly-version-/#/

    :param poly_paths: paths to polygons
    :param N_poly: number of polygon to convert
    """
    # init. plotting list
    data = dict(
        x=[],
        y=[],
        mode='lines',
        line=go.scatter.Line(color='black'),
        name='map',
        hoverinfo='skip'
    )

    for i_poly in range(N_poly):
        poly_path = poly_paths[i_poly]
        
        # get the Basemap coordinates of each segment
        coords_cc = np.array(
            [(vertex[0],vertex[1]) 
             for (vertex,code) in poly_path.iter_segments(simplify=False)]
        )
        
        # convert coordinates to lon/lat by 'inverting' the Basemap projection
        lon_cc, lat_cc = m(coords_cc[:,0],coords_cc[:,1], inverse=True)
    
        
        # add plot.ly plotting options
        data['x'] = data['x'] + lon_cc.tolist() + [np.nan]
        data['y'] = data['y'] + lat_cc.tolist() + [np.nan]
     
    return [data]

def get_coastline_traces(m):
    """
    Credit: https://chart-studio.plotly.com/~Dreamshot/9144/import-plotly-plotly-version-/#/

    :param m: map
    """
    poly_paths = m.drawcoastlines().get_paths() # coastline polygon paths
    N_poly = 91  # use only the 91st biggest coastlines (i.e. no rivers)
    return polygons_to_traces(poly_paths, N_poly, m)

def get_country_traces(m):
    """
    Credit: https://chart-studio.plotly.com/~Dreamshot/9144/import-plotly-plotly-version-/#/

    :param m: map
    """
    poly_paths = m.drawcountries().get_paths() # country polygon paths
    N_poly = len(poly_paths)  # use all countries
    return polygons_to_traces(poly_paths, N_poly, m)

def get_map_traces(**kwargs):
    m = Basemap(**kwargs)
    return get_coastline_traces(m) + get_country_traces(m)

if __name__ == '__main__':
    wind_data = np.load('data/visualization_example/wind_dat.npy')
    fig = create_vis(wind_data)
    fig.show()