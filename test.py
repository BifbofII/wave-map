# %% [markdown]
# # Interactive Test Script

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
from owslib.wfs import WebFeatureService
wfs = WebFeatureService(url='https://opendata.fmi.fi/wfs', version='2.0.0')
wfs.identification.title

# %%
[(i, q.id) for i, q in enumerate(wfs.storedqueries)]

# %%
[p.name for p in wfs.storedqueries[85].parameters]

# %%
resp = wfs.getfeature(storedQueryID='fmi::ef::stations')
with open('data/stations.gml', 'wb') as f:
    f.write(resp.read())

# %%
resp = wfs.getfeature(storedQueryID='fmi::observations::wave::simple')
with open('data/simple.gml', 'wb') as f:
    f.write(resp.read())

# %%
resp = wfs.getfeature(storedQueryID='fmi::observations::wave::timevaluepair')
with open('data/tvp.gml', 'wb') as f:
    f.write(resp.read())
    
# %%
resp = wfs.getfeature(storedQueryID='fmi::observations::wave::multipointcoverage')
with open('data/mpc.gml', 'wb') as f:
    f.write(resp.read())

# %%
from wfs2df import parse_wfs
df, met = parse_wfs('data/tvp.gml', 'data/stations.gml')
print(met)
df.describe()

# %%
wh = df.iloc[:,::5]

# %%
resp = wfs.getfeature(storedQueryID='fmi::observations::wave::timevaluepair', storedQueryParams={'starttime': '2020-09-01T00:00:00Z', 'endtime':'2020-09-03T23:00:00Z'})
with open('data/sept1-3.gml', 'wb') as f:
    f.write(resp.read())

# %%
df, met = parse_wfs('data/sept1-3.gml')
wh = df.iloc[:,::5]
wh = wh.dropna(axis=1, how='all')
wh = wh.dropna(axis=0, how='all')
wh.plot()
plt.show()

# %%
b = wh.iloc[:,0]
b.dropna().plot()
plt.show()