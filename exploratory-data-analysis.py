# %% [markdown]
# # Wave Map Exploratory Data Analysis

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
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
x = np.array(wave_height.index).astype('datetime64[m]')
y = wave_height.columns
z = np.invert(wave_height.isnull().values).astype(int).T
colorsc = [[0, 'rgb(194,59,34)'],
            [0.5, 'rgb(194,59,34)'], 
            [0.5, 'rgb(0,179,30)'],
            [1, 'rgb(0,179,30)']]
fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, colorscale=colorsc, colorbar=dict(tickmode='array', tickvals=[0,1], ticktext=['null', 'value'])))
fig.update_layout(title='Missing values over time', xaxis=dict(side='top'))
fig.show()