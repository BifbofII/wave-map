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
from plotly.subplots import make_subplots
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
wave_height = wave_data.loc[:,(slice(None),'Wave height (m)')]
wave_height.columns = wave_height.columns.droplevel(1)
wave_dir = wave_data.loc[:,(slice(None),'Direction of waves (deg)')]
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

# %%
# Divide wind speed and direction
wind_speed = weather_data.loc[:,(slice(None),'wind_speed')]
wind_speed.columns = wind_speed.columns.droplevel(1)
wind_dir = weather_data.loc[:,(slice(None),'direction')]
wind_dir.columns = wind_dir.columns.droplevel(1)

# %%
# Plot wind speed
fig = wind_speed.plot()
fig.update_layout(title='Wind speed')
fig.show()

# %%
# Combine data of one buoy
suomenlahti = pd.concat(
    [wind_speed['suomenlahti'], wind_dir['suomenlahti'],
    wave_height['suomenlahti'], wave_dir['suomenlahti']], axis=1, join='inner')
suomenlahti.columns = ['wind_speed', 'wind_dir', 'wave_height', 'wave_dir']

# %%
# Plot wind speed and wave height of one buoy
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=suomenlahti.index, y=suomenlahti['wind_speed'], name='wind speed'),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=suomenlahti.index, y=suomenlahti['wave_height'], name='wave height'),
    secondary_y=True,
)
fig.update_layout(title='Wind speed and wave height')
fig.show()

# %% [markdown]
# A strong correlation between the wind speed and the wave height can be seen.
# The correlation seems to be delayed at most by a few hours.

# %%
# Plot wind direction and wave direction of one buoy
fig = suomenlahti[['wind_dir', 'wave_dir']].plot()
fig.update_layout(title='Wind and wave direction')
fig.show()

# %% [markdown]
# Wave and wind direction does also seem to be strongly correlated.

# %%
# Plot wave height vs. wind speed
fig = suomenlahti.plot(x='wind_speed', y='wave_height', kind='scatter')
fig.update_layout(title='Correlation Wind Speed and Wave Height (Suomenlahti)')
fig.show()

# %%
# Plot wave dir vs. wind dir
fig = suomenlahti.plot(x='wind_dir', y='wave_dir', kind='scatter')
fig.update_layout(title='Correlation Wind Direction and Wave Direction (Suomenlahti)')
fig.show()

# %%
# Plot wave height vs. wind speed (all buoys)
join = pd.concat([wind_speed, wave_height], axis=1, join='inner')
join = join.dropna()
x = join.iloc[:,:5].values.flatten()
y = join.iloc[:,5:].values.flatten()
fig = px.scatter(x=x, y=y)
fig.update_layout(title='Correlation Wind Speed and Wave Height')
fig.update_xaxes(title='wind speed / m/s')
fig.update_yaxes(title='wave height / m')
fig.show()

# %%
# Plot wave dir vs. wind dir (all buoys)
join = pd.concat([wind_dir, wave_dir], axis=1, join='inner')
join = join.dropna()
x = join.iloc[:,:5].values.flatten()
y = join.iloc[:,5:].values.flatten()
fig = px.scatter(x=x, y=y)
fig.update_layout(title='Correlation Wind Direction and Wave Direction')
fig.update_xaxes(title='wind direction / deg')
fig.update_yaxes(title='wave direction / deg')
fig.show()

# %%
# Plot wave direction distribution
no_nan_wave_dir = wave_dir.dropna()
fig = ff.create_distplot([no_nan_wave_dir[c].values for c in wave_dir], wave_dir.columns, bin_size=2)
fig.update_layout(title='Distribution of wave direction per buoy')
fig.update_xaxes(title='wave direction / deg')
fig.show()

# %%
# Plot wind direction distribution
no_nan_wind_dir = wind_dir.dropna()
fig = ff.create_distplot([no_nan_wind_dir[c].values for c in wind_dir], wind_dir.columns, bin_size=2)
fig.update_layout(title='Distribution of wind direction per buoy')
fig.update_xaxes(title='wind direction / deg')
fig.show()