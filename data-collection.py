# %% [markdown]
# # Wave Map Data Collection

# %%
import datetime
import tempfile

import pygrib
import numpy as np
import pandas as pd
from functools import reduce

from owslib.wfs import WebFeatureService
from wfs2df import parse_wfs

# %% [markdown]
# ## Wave Data - Open Data Interface
# Download wave height and direction data from the FMI Open Data service, clean it up and save as CSV files.
# 
# Unfortunitely the open data interface seems to report a huge number of missing values.

# %%
def download_long_time_interval(wfs, query, start, end=datetime.datetime.now(), params=None):
    """
    Helper function to download data for long time intervals

    The open data service allows only downloads for up to 7 days at once.
    This function splits a longer time interval into shorter ones, downloads the segments of data and joins them

    :param wfs: a owslib wfs server
    :param query: stored query id
    :param start: the start time as a datetime
    :param end: the end time as a datetime (optional)
    :param params: query parameters (other than times) (optional)
    :returns: the joined data and station data
    """
    data = []
    meta = []
    with tempfile.TemporaryFile() as station_tmp:
        # Download station info
        resp = wfs.getfeature(storedQueryID='fmi::ef::stations')
        station_tmp.write(resp.read())

        # Download data
        while start + datetime.timedelta(7) < end:
            params.update(starttime=start.isoformat(), endtime=(start + datetime.timedelta(7)).isoformat())
            resp = wfs.getfeature(storedQueryID=query, storedQueryParams=params)
            with tempfile.TemporaryFile() as data_tmp:
                data_tmp.write(resp.read())
                data_tmp.seek(0)
                station_tmp.seek(0)
                it_data, it_meta = parse_wfs(data_tmp, station_tmp)
                data.append(it_data)
                meta.append(it_meta)
                start = start + datetime.timedelta(7)
        params.update(starttime=start.isoformat(), endtime=end.isoformat())
        resp = wfs.getfeature(storedQueryID=query, storedQueryParams=params)
        with tempfile.TemporaryFile() as data_tmp:
            data_tmp.write(resp.read())
            data_tmp.seek(0)
            station_tmp.seek(0)
            it_data, it_meta = parse_wfs(data_tmp, station_tmp)
            data.append(it_data)
            meta.append(it_meta)
    joined_data = pd.concat(data)
    stations = reduce(lambda a,b: pd.merge(a, b, how='right'), map(lambda m: m['stations'], meta))
    return joined_data, stations

# %%
fmi_wfs = WebFeatureService(url='https://opendata.fmi.fi/wfs', version='2.0.0')
stored_query_id = 'fmi::observations::wave::timevaluepair'
query_params = {'parameters': 'WaveHs,ModalWDi'}
start_time = datetime.date(2020, 1, 1)
end_time = datetime.date(2020, 7, 1)

data, stations = download_long_time_interval(fmi_wfs, stored_query_id, start=start_time, end=end_time, params=query_params)

data.describe()

# %%
data.to_csv('data/wave_data_od.csv')
stations.to_csv('data/wave_stations_od.csv')
 
# %% [markdown]
# ## Wave Data - Downloaded CSV Files
# The FMI does also provide a interface for downloading data as CSV files.
# We downloaded data for wave height and direction from there from 01.01.2019 - 14.09.2020, this data is joined here and exported as a combined CSV File.

# %%
buoys = ['helsinki-suomenlinna', 'peraemeri', 'pohjois-itaemeri', 'selkaemeri', 'suomenlahti']
data = []
meta = []
for buoy in buoys:
    buoy_data = pd.read_csv('data/fmi-download/' + buoy + '.csv')
    buoy_meta = pd.read_csv('data/fmi-download/' + buoy + '-meta.csv')
    buoy_data['datetime'] = pd.to_datetime(buoy_data['Year'].astype(str) + '-' + buoy_data['m'].astype(str) + '-' + buoy_data['d'].astype(str) + 'T' + buoy_data['Time'])
    buoy_data = buoy_data.set_index('datetime')
    buoy_data = buoy_data.drop(['Year', 'm', 'd', 'Time', 'Time zone'], axis=1)
    buoy_data.columns = pd.MultiIndex.from_tuples([(buoy, c) for c in buoy_data.columns])
    buoy_data.columns.names = ['buoy', 'parameter']
    buoy_meta['location'] = buoy
    data.append(buoy_data)
    meta.append(buoy_meta)
data = pd.concat(data, axis=1)
meta = pd.concat(meta)
meta = meta.set_index('Station ID')
# sort index
data = data.sort_index()

# %%
# Compute wave data as vector (u/v)
# http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
u = -np.sin(np.radians(data.loc[:,(slice(None),'Direction of waves (deg)')].values)) \
    * data.loc[:,(slice(None),'Wave height (m)')].values
v = -np.cos(np.radians(data.loc[:,(slice(None),'Direction of waves (deg)')].values)) \
    * data.loc[:,(slice(None),'Wave height (m)')].values
cols = [(buoy, param) for param in ['wave_u', 'wave_v'] for buoy in buoys]
uv_df = pd.DataFrame(np.concatenate([u,v], axis=1), index=data.index,
    columns=pd.MultiIndex.from_tuples(cols))
data = pd.concat([data, uv_df], axis=1)

# %%
data.to_csv('data/wave_data.csv')
meta.to_csv('data/wave_stations.csv')

# %% [markdown]
# ## Weather Data - Downloaded Grib File
# Extract the weather information corresponding to the buoys from the grib file downloaded from Copernicus.

# %%
grbs = pygrib.open('data/weather_data.grib')

# %%
stations = pd.read_csv('data/wave_stations.csv', index_col=0)

# %%
ex_msg = grbs.message(1)
lat, lon = ex_msg.latlons()
lats = lat[:,0].flatten()
lons = lon[0,:].flatten()

# %%
# Find closest point in grib file to buoys
stations['lat_ind'] = -1
stations['lon_ind'] = -1

for ind, buoy in stations.iterrows():
    lat_diff = np.abs(lats - buoy['Latitude (decimals)'])
    lon_diff = np.abs(lons - buoy['Longitude (decimals)'])
    stations.loc[ind, 'lat_ind'] = np.argmin(lat_diff)
    stations.loc[ind, 'lon_ind'] = np.argmin(lon_diff)

# %%
weather_params = ['10u', '10v', '2t', 'sp']

data = dict()
for param in weather_params:
    param_grbs = grbs.select(shortName=param)
    index = [datetime.datetime(grb.year, grb.month, grb.day, grb.hour) for grb in param_grbs]
    for _, buoy in stations.iterrows():
        data[(buoy['location'],param)] = pd.Series(index=index,
            data=[grb.values[buoy['lat_ind'],buoy['lon_ind']] for grb in param_grbs], name=param)

data = pd.DataFrame(data)
data.columns.names = ['buoy', 'parameter']

# %%
# Compute wind data as speed and direction
# http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
ws = np.sqrt(data.loc[:,(slice(None),'10u')].values**2 + data.loc[:,(slice(None),'10v')].values**2)
d = np.degrees(np.arctan2(data.loc[:,(slice(None),'10u')].values,
    data.loc[:,(slice(None),'10v')].values)) + 180
cols = [(buoy, param) for param in ['wind_speed', 'direction'] for buoy in buoys]
wsd_df = pd.DataFrame(np.concatenate([ws,d], axis=1), index=data.index,
    columns=pd.MultiIndex.from_tuples(cols))
data = pd.concat([data, wsd_df], axis=1)

# %%
# Example data point
data.loc['2020-07-01 11:00', 'helsinki-suomenlinna']

# %%
data.to_csv('data/weather_data.csv')

# %%
u_msg = grbs.select(shortName='10u')[0]
v_msg = grbs.select(shortName='10v')[0]
x, y = u_msg.latlons()
u = u_msg.values
v = v_msg.values

# %%
lat = x[:,0]
lon = y[0,:]
dat = np.concatenate([u[:,:,np.newaxis], v[:,:,np.newaxis]], axis=2)

# %%
np.save('data/visualization_example/lat.npy', lat)
np.save('data/visualization_example/lon.npy', lon)
np.save('data/visualization_example/wind_dat.npy', dat)

# %%
# Save wind data for vis
time = [datetime.datetime(2020, 8, d, h) for d in range(1,8) for h in range(0,24,4)]
wind_data = np.zeros((len(lat), len(lon), len(time), 2))
for i, t in enumerate(time):
    u_msg = grbs.select(shortName='10u', year=t.year, month=t.month, day=t.day, hour=t.hour)[0]
    v_msg = grbs.select(shortName='10v', year=t.year, month=t.month, day=t.day, hour=t.hour)[0]
    wind_data[:,:,i,0] = u_msg.values
    wind_data[:,:,i,1] = v_msg.values

# %%
np.save('predictions/wind_data.npy', wind_data)

# %%
grbs.close()