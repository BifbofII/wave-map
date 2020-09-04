# %% [markdown]
# # Interactive Test Script

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fmiod

# %%
dl = fmiod.FmiDownloader()
ql = dl.get_stored_query_list()
ql.head()

# %%
wave_ql = dl.find_queries('wave')
wave_ql

# %%
wave_ql.loc['fmi::observations::wave::simple',:].Abstract