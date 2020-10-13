"""
Script to save the basemap into a file

This is used so that the mpl_toolkit.basemap module does not need to be installed in Heroku.
"""

import pickle
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.basemap import Basemap

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
    map_traces = get_map_traces(resolution='i', llcrnrlon=14, llcrnrlat=57, urcrnrlon=36, urcrnrlat=68) 
    with open('data/map_traces.pkl', 'wb') as f:
        pickle.dump(map_traces, f)