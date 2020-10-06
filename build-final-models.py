# %% [markdown]
# # Wave Map Build and Export Final Models
# ## Model Descriptions
# There are two distinct models, one predicting the wave height and one predicting the direction.
# This approach is chosen because the wave height has a stronger correlation and this way the height prediction is better since it is not influenced by the direction.
#
# ### Wave Height Model
# **SGDRegressor with 'huber' loss**
#
# Inputs:
# * 10u
# * 10v
# * wind_speed
# * wind_speed_sq
# * 10u_lag2
# * 10v_lag2
# * wind_speed_lag2
# * wind_speed_sq_lag2
# * 10u_lag4
# * 10v_lag4
# * wind_speed_lag4
# * wind_speed_sq_lag4
#
#
# Outputs:
# * wave_height
#
#
# ### Wave Direction Model
# **RidgeRegression**
# Ridge is used because it can predict the entire direction vector as one value.
#
# Inputs:
# * 10u
# * 10v
# * wind_speed
# * wind_speed_sq
# * 10u_lag2
# * 10v_lag2
# * wind_speed_lag2
# * wind_speed_sq_lag2
# * 10u_lag4
# * 10v_lag4
# * wind_speed_lag4
# * wind_speed_sq_lag4
#
#
# Outputs:
# * wave_u
# * wave_v


# %%
import datetime
import pickle
import numpy as np
import pandas as pd

import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

np.random.seed(42)

# %%
# Load data
wave_data = pd.read_csv('data/wave_data.csv', header=[0,1], index_col=0, parse_dates=True)
weather_data = pd.read_csv('data/weather_data.csv', header=[0,1], index_col=0, parse_dates=True)

# %%
# Join data
data = pd.concat([wave_data, weather_data], axis=1, join='inner')

# %%
# Add squared wind speed
sq_dat = data.loc[:,(slice(None),'wind_speed')]**2
sq_dat.columns = pd.MultiIndex.from_tuples([(b, p+'_sq') for b, p in sq_dat.columns])
data = pd.concat([data, sq_dat], axis=1)

# %%
# Add lagging values
lagging_vars = ['10u', '10v', '2t', 'sp', 'wind_speed', 'wind_speed_sq']
lagging_times = [datetime.timedelta(hours=h) for h in range(1, 20)]
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
    if len(model.coef_.shape) <= 1:
        return pd.DataFrame(
            np.append(model.coef_, model.intercept_).T,
            index=np.append(coef_desc, 'intercept'), columns=target_desc)
    else:
        return pd.DataFrame(
            np.append(model.coef_, model.intercept_[:,np.newaxis], axis=1).T,
            index=np.append(coef_desc, 'intercept'), columns=target_desc)

# %%
# Another helper function
def plot_prediction(pipe, X, y, func=lambda x: x.flatten()):
    y_pred = pipe.predict(X)
    fig = px.scatter(x=func(y), y=func(y_pred))
    fig.update_layout(title='Prediction vs true value')
    fig.update_xaxes(title='true value')
    fig.update_yaxes(title='prediction')
    fig.show()

# %% [markdown]
# ## Train Test Split
# Use a randomly chosen one of the buoys as the test dataset.

# %%
num_buoys = data.columns.levshape[0]
test_buoy = 'pohjois-itaemeri'

# %%
test = data[test_buoy]
train = data.drop(test_buoy, axis=1)

# %% [markdown]
# ## Wave Height Model

# %%
# Build input and output matrices
base_vars = ['10u', '10v', 'wind_speed', 'wind_speed_sq']
input_vars = base_vars.copy()
for l in [2, 4]:
    input_vars.extend([v + '_lag' + str(l) for v in base_vars])
input_vars.sort()
output_vars = ['Wave height (m)']

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
height_pipe = Pipeline([('scaler', StandardScaler()), ('regression', SGDRegressor(loss='huber'))])
height_pipe.fit(X_train, Y_train)
print('Train accuracy of the model: {:.3f}'.format(height_pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(height_pipe.score(X_test, Y_test)))
plot_prediction(height_pipe, X_test, Y_test)

# %%
describe_regression(height_pipe, input_vars, output_vars)

# %%
# Save model
with open('models/wave-height-model.pkl', 'wb') as f:
    pickle.dump(height_pipe, f)

# %% [markdown]
# ## Wave Direction Model

# %%
# Build input and output matrices
base_vars = ['10u', '10v', 'wind_speed', 'wind_speed_sq']
input_vars = base_vars.copy()
for l in [2, 4]:
    input_vars.extend([v + '_lag' + str(l) for v in base_vars])
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
alphas = np.logspace(-2, 5, num=20)
dir_pipe = Pipeline([('scaler', StandardScaler()), ('regression', RidgeCV())])
dir_pipe.fit(X_train, Y_train)
print('Chosen regularization parameter: alpha={:.3f}'.format(dir_pipe.named_steps['regression'].alpha_))
print('Train accuracy of the model: {:.3f}'.format(dir_pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(dir_pipe.score(X_test, Y_test)))
plot_prediction(dir_pipe, X_test, Y_test, func=lambda x: np.sqrt(np.sum(x**2, axis=1)))

# %%
# Model coefficiens
describe_regression(dir_pipe, input_vars, output_vars)

# %%
# Save model
with open('models/wave-dir-model.pkl', 'wb') as f:
    pickle.dump(dir_pipe, f)

# %% [markdown]
# ## Test Performance of Combined Model

# %%
# Build input and output matrices
base_vars = ['10u', '10v', 'wind_speed', 'wind_speed_sq']
input_vars = base_vars.copy()
for l in [2, 4]:
    input_vars.extend([v + '_lag' + str(l) for v in base_vars])
input_vars.sort()
output_vars = ['wave_u', 'wave_v']

test_dropped = test.dropna()
X_test = test_dropped[input_vars].values
Y_test = test_dropped[output_vars].values

# %%
# Combine predictions
height = height_pipe.predict(X_test)
dir = dir_pipe.predict(X_test)

scale_factor = height / np.sqrt((dir**2).sum(axis=1))

pred = dir * np.concatenate([scale_factor[:,np.newaxis], scale_factor[:,np.newaxis]], axis=1)

# %%
# Test performance
print('Test accuracy of the combined models: {:.3f}'.format(r2_score(Y_test, pred)))

fig = px.scatter(x=np.sqrt(np.sum(Y_test**2, axis=1)), y=np.sqrt(np.sum(pred**2, axis=1)))
fig.update_layout(title='Prediction vs true value')
fig.update_xaxes(title='true value')
fig.update_yaxes(title='prediction')
fig.show()