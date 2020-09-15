# %% [markdown]
# # Wave Map Exploratory Data Analysis

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
station_data

# %%
# Indexing helpers
idx = pd.IndexSlice
wave_height = idx[:, 'Wave height (m)']
wave_dir = idx[:, 'Direction of waves (deg)']

# %%
# Plot wave height
wave_data.loc[:,wave_height].plot()
plt.show()

# %%
# Plot one month of wave height
wave_data.loc['2020-01',wave_height].plot()
plt.gca().set_xlim('2020-01-01', '2020-02-01')
plt.show()

# %%
# Plot missing values
sns.heatmap(wave_data.loc[:,wave_height].isnull().values)
plt.show()