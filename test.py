# %% [markdown]
# # Interactive Test Script

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fmiod

# %%
dl = fmiod.FmiDownloader()
dl.get_stored_query_list().head()