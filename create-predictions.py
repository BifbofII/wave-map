# %% [markdown]
# # Create Wave Height Predictions
# This script will create wave height predictions for set time points and save them as numpy arrays.
# The saved numpy arrays are then used in the visualization component.

# %%
import datetime
import numpy as np
from predmod import JoinedWaveModel

# %%
model = JoinedWaveModel('models/wave-height-model.pkl', 'models/wave-dir-model.pkl',
    'data/weather_data.grib')

# %%
location = None
time = [datetime.datetime(2020, 8, d, h) for d in range(1,8) for h in range(0,24,4)]

# %%
wave_data, lats, lons, times = model.predict(location=location, time=time)

# %%
np.save('predictions/wave_data.npy', wave_data)
np.save('predictions/lats.npy', lats)
np.save('predictions/lons.npy', lons)
np.save('predictions/times.npy', times)