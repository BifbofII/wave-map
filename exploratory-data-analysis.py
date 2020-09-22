# %% [markdown]
# # Wave Map Exploratory Data Analysis

# %%
import numpy as np
import pandas as pd
import pygrib
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
pd.options.plotting.backend = 'plotly'

# %% [markdown]
# ## Wave Data

# %%
wave_data = pd.read_csv('data/wave_data.csv', header=[0,1], index_col=0, parse_dates=True)
wave_data.columns.names = ['Station', 'Signal']
station_data = pd.read_csv('data/wave_stations.csv', index_col=0)

# %%
wave_data

# %%
wave_data.describe()

# %%
print('Percentages of missing data:')
wave_data.isnull().mean()

# %%
station_data

# %%
# Divide wave height and direction
wave_height = wave_data.iloc[:,1::2]
wave_height.columns = wave_height.columns.droplevel(1)
wave_dir = wave_data.iloc[:,::2]
wave_dir.columns = wave_dir.columns.droplevel(1)

# %%
# Plot wave height
fig = wave_height.plot()
fig.update_layout(title='Wave height')
fig.show()

# %%
# Plot one month of wave height
fig = wave_height.loc['2020-01',:].plot()
fig.update_layout(title='Wave height')
fig.show()

# %%
# Plot wave height and direction
# One arrow per day
subsampling = 48
x,y = np.meshgrid(np.arange(0, wave_height.shape[0]/subsampling), np.arange(0, wave_height.shape[1] * 100, 100))
d = wave_dir.iloc[::subsampling,:].values.T
h = wave_height.iloc[::subsampling,:].values.T
u = np.sin(np.radians(d)) * h
v = np.cos(np.radians(d)) * h

x_axis = list(enumerate(wave_height.iloc[::subsampling,:].index))[::100]

fig = ff.create_quiver(x, y, u, v, scale=10)
fig.update_layout(title='Wave height and direction',
    yaxis=dict(tickmode='array', tickvals=[0,100,200,300,400], ticktext=list(wave_height.columns)),
    xaxis=dict(tickmode='array', tickvals=list(map(lambda x: x[0], x_axis)), ticktext=list(map(lambda x: str(x[1]), x_axis))))
fig.show()

# %%
# Plot missing values
x = np.array(wave_height.index).astype('datetime64[m]')
y = wave_height.columns
z = np.invert(wave_height.isnull().values).astype(int).T
colorsc = [[0, 'rgb(194,59,34)'],
            [0.5, 'rgb(194,59,34)'], 
            [0.5, 'rgb(0,179,30)'],
            [1, 'rgb(0,179,30)']]
fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, colorscale=colorsc,
    colorbar=dict(tickmode='array', tickvals=[0,1], ticktext=['null', 'value'])))
fig.update_layout(title='Missing values over time', xaxis=dict(side='top'))
fig.show()

# %% [markdown]
# ## Weather Data

# %%
grbs = pygrib.open('data/weather_data.grib')

# %%
# Plot wind data on map
u_msg = grbs.select(shortName='10u')[0]
v_msg = grbs.select(shortName='10v')[0]
x, y = u_msg.latlons()
u = u_msg.values
v = v_msg.values

quiver_obj = plotly.figure_factory._quiver._Quiver(x, y, u, v,
    scale=.03, arrow_scale=0.3, angle=np.pi/9)
barb_x, barb_y = quiver_obj.get_barbs()
arrow_x, arrow_y = quiver_obj.get_quiver_arrows()

fig = go.Figure(data=go.Scattermapbox(lat=barb_x+arrow_x, lon=barb_y+arrow_y, mode='lines'))
fig.update_layout(title='Wind direction', mapbox_style='open-street-map',
    mapbox=dict(zoom=4, center=dict(lat=x.mean(), lon=y.mean())),
    margin=dict(l=0, r=0, t=50, b=0))
fig.show()

# %%
grbs.close()

# %% [markdown]
# ### Weather Subset for Buoys

# %%
weather_data = pd.read_csv('data/weather_data.csv', header=[0,1], index_col=0, parse_dates=True)