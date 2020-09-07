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
[storedquery.title for storedquery in wfs.storedqueries]

# %%
resp = wfs.getfeature(storedQueryID='fmi::observations::wave::simple')
with open('data/simple.gml', 'wb') as f:
    f.write(resp.read())

# %%
resp = wfs.getfeature(storedQueryID='fmi::observations::wave::timevaluepair')
with open('data/tvp.gml', 'wb') as f:
    f.write(resp.read())

# %%
from wfs2df import parse_wfs
df, met = parse_wfs('data/tvp.gml')
print(met)
df.describe()