# %% [markdown]
# # Wave Map Data Collection

# %%
import datetime
import tempfile

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
data.to_csv('data/wave_data.csv')
meta.to_csv('data/wave_stations.csv')