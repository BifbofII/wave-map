# %% [markdown]
# # Wave Map Model Building

# %%
import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

pd.options.plotting.backend = 'plotly'
np.random.seed(42)

# %%
# Load data
wave_data = pd.read_csv('data/wave_data.csv', header=[0,1], index_col=0, parse_dates=True)
weather_data = pd.read_csv('data/weather_data.csv', header=[0,1], index_col=0, parse_dates=True)

# %%
# Join data
data = pd.concat([wave_data, weather_data], axis=1, join='inner')

# %%
# Add lagging values
lagging_vars = ['10u', '10v', '2t', 'sp']
lagging_times = [datetime.timedelta(hours=h) for h in range(1, 4)]
dat = [data]
for var in lagging_vars:
    for i, t in enumerate(lagging_times):
        var_dat = data.loc[:,(slice(None),var)]
        var_dat.index = var_dat.index + t
        var_dat.columns = pd.MultiIndex.from_tuples([(b, p+'_lag'+str(i+1)) for b, p in var_dat.columns])
        dat.append(var_dat)
data = pd.concat(dat, axis=1)

# %%
# Helper function
def describe_regression(pipe, coef_desc, target_desc):
    model = pipe.named_steps['regression']
    return pd.DataFrame(
        np.append(model.coef_, model.intercept_[:,np.newaxis], axis=1).T,
        index=np.append(coef_desc, 'intercept'), columns=target_desc)

# %% [markdown]
# ## Train Test Split
# Use a randomly chosen one of the buoys as the test dataset.

# %%
num_buoys = data.columns.levshape[0]
test_buoy = data.columns.get_level_values(0)[np.random.randint(0, num_buoys)]

# %%
test = data[test_buoy]
train = data.drop(test_buoy, axis=1)

# %% [markdown]
# ## Simple Linear Regression
# Linear regression from Wind Vector, Temperature and Pressure to Wave Vector.
# No Temporal Information.
#
# Inputs:
# * 10u
# * 10v
# * 2t
# * sp
#
#
# Outputs:
# * wave_u
# * wave_v

# %%
# Build input and output matrices
input_vars = ['10u', '10v', '2t', 'sp']
output_vars = ['wave_u', 'wave_v']

test_dropped = test.dropna()
X_test = test_dropped[input_vars].values
Y_test = test_dropped[output_vars].values

X_train = None
Y_train = None
for i, buoy in enumerate(train.columns.levels[0].drop(test_buoy)):
    buoy_dat = train[buoy].dropna()
    if X_train is None:
        X_train = buoy_dat[input_vars].values
        Y_train = buoy_dat[output_vars].values
    else:
        X_train = np.concatenate([X_train, buoy_dat[input_vars].values])
        Y_train = np.concatenate([Y_train, buoy_dat[output_vars].values])

# %%
# Train model
pipe = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression())])
pipe.fit(X_train, Y_train)
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))

# %%
# Model coefficiens
describe_regression(pipe, input_vars, output_vars)

# %% [markdown]
# The linear regression gives the expected result: The are strong correlations between 10u-wave_u and 10v-wave_v.
# However, the accuracy of the model is not good.
# For some reason the train accuracy is lower than the test accuracy.

# %% [markdown]
# ## Linear Regression with older inputs
# Linear regression from Wind Vector, Temperature and Pressure to Wave Vector.
# Each input variable is used four times, once with the current value, once lagging one hour and so on.
#
# Inputs:
# * 10u
# * 10v
# * 2t
# * sp
#
#
# Outputs:
# * wave_u
# * wave_v

# %%
# Build input and output matrices
base_vars = ['10u', '10v', '2t', 'sp']
input_vars = base_vars.copy()
for l in range(3):
    input_vars.extend([v + '_lag' + str(l+1) for v in base_vars])
input_vars.sort()
output_vars = ['wave_u', 'wave_v']

test_dropped = test.dropna()
X_test = test_dropped[input_vars].values
Y_test = test_dropped[output_vars].values

X_train = None
Y_train = None
for i, buoy in enumerate(train.columns.levels[0].drop(test_buoy)):
    buoy_dat = train[buoy].dropna()
    if X_train is None:
        X_train = buoy_dat[input_vars].values
        Y_train = buoy_dat[output_vars].values
    else:
        X_train = np.concatenate([X_train, buoy_dat[input_vars].values])
        Y_train = np.concatenate([Y_train, buoy_dat[output_vars].values])

# %%
# Train model
pipe = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression())])
pipe.fit(X_train, Y_train)
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))

# %%
# Model coefficiens
describe_regression(pipe, input_vars, output_vars)

# %% [markdown]
# The regression model with lagging input variables shows increased accuracy.
# The largest influence on the model comes from the variables lagging the most, therefore longer lagging variables should be included.