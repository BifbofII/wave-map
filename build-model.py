# %% [markdown]
# # Wave Map Model Experiments

# %%
import datetime
import numpy as np
import pandas as pd

import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskLassoCV, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV

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
# * repeating from '_lag1' to '_lag3'
#
#
# Outputs:
# * wave_u
# * wave_v

# %%
# Build input and output matrices
base_vars = ['10u', '10v', '2t', 'sp']
input_vars = base_vars.copy()
for l in range(1, 4):
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

# %% [markdown]
# ## Linear Regression with even older inputs
# Linear regression from Wind Vector, Temperature and Pressure to Wave Vector.
# Each input variable is used nine times, once with the current value, once lagging one hour and so on.
# The goal here is to see how old the variables with the strongest correlation are.
#
# Inputs:
# * 10u
# * 10v
# * 2t
# * sp
# * repeating from '_lag1' to '_lag9'
#
#
# Outputs:
# * wave_u
# * wave_v

# %%
# Build input and output matrices
base_vars = ['10u', '10v', '2t', 'sp']
input_vars = base_vars.copy()
for l in range(1, 10):
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
# The model yields basically the same accuracy as when using less lagging variables.
# Of the lagging variables again, the oldest is the one with the strongest correlation.
# It is not clear from this result how much lag should be included.
#
# Next we try introducing regularization.

# %% [markdown]
# ## Ridge Regression
# Ridge regression from Wind Vector, Temperature and Pressure to Wave Vector.
# Each input variable is used nine times, once with the current value, once lagging one hour and so on.
# The goal here is to see how old the variables with the strongest correlation are.
#
# Inputs:
# * 10u
# * 10v
# * 2t
# * sp
# * repeating from '_lag1' to '_lag9'
#
#
# Outputs:
# * wave_u
# * wave_v

# %%
# Build input and output matrices
base_vars = ['10u', '10v', '2t', 'sp']
input_vars = base_vars.copy()
for l in range(1, 10):
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
pipe = Pipeline([('scaler', StandardScaler()), ('regression', RidgeCV(alphas=alphas))])
pipe.fit(X_train, Y_train)
print('Chosen regularization parameter: alpha={:.3f}'.format(pipe.named_steps['regression'].alpha_))
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))

# %%
# Model coefficiens
describe_regression(pipe, input_vars, output_vars)

# %% [markdown]
# The regularization does not seem to influence the performance much.

# %% [markdown]
# ## Lasso Regression
# Lasso Regression for selecting important variables.
#
# Inputs:
# * 10u
# * 10v
# * 2t
# * sp
# * repeating from '_lag1' to '_lag9'
#
#
# Outputs:
# * wave_u
# * wave_v

# %%
# Build input and output matrices
base_vars = ['10u', '10v', '2t', 'sp']
input_vars = base_vars.copy()
for l in range(1, 10):
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
pipe = Pipeline([('scaler', StandardScaler()), ('regression', MultiTaskLassoCV(alphas=alphas))])
pipe.fit(X_train, Y_train)
print('Chosen regularization parameter: alpha={:.3f}'.format(pipe.named_steps['regression'].alpha_))
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))

# %%
# Model coefficiens
coef = describe_regression(pipe, input_vars, output_vars)
coef

# %%
# Unneeded variables
coef[(coef['wave_u'] == 0) & (coef['wave_v'] == 0)].index

# %% [markdown]
# The lasso regression shows that Surface Pressure and Temperature can be dropped all together.
# Next we try lasso with only the wind components.

# %% [markdown]
# ## Lasso Regression - Wind Only
# Lasso Regression for selecting which wind lag is important.
#
# Inputs:
# * 10u
# * 10v
# * repeating from '_lag1' to '_lag9'
#
#
# Outputs:
# * wave_u
# * wave_v

# %%
# Build input and output matrices
base_vars = ['10u', '10v']
input_vars = base_vars.copy()
for l in range(1, 10):
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
pipe = Pipeline([('scaler', StandardScaler()), ('regression', MultiTaskLassoCV(alphas=alphas))])
pipe.fit(X_train, Y_train)
print('Chosen regularization parameter: alpha={:.3f}'.format(pipe.named_steps['regression'].alpha_))
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))

# %%
# Model coefficiens
coef = describe_regression(pipe, input_vars, output_vars)
coef

# %%
# Unneeded variables
coef[(coef['wave_u'] == 0) & (coef['wave_v'] == 0)].index

# %% [markdown]
# Not that much useful information.
# Why is the coefficient of the longest lag always that high?

# %% [markdown]
# ## Lasso Regression - Wind Only - More Lag
# Lasso Regression for selecting which wind lag is important.
#
# Inputs:
# * 10u
# * 10v
# * repeating from '_lag2' '_lag4' to '_lag18'
#
#
# Outputs:
# * wave_u
# * wave_v

# %%
# Build input and output matrices
base_vars = ['10u', '10v']
input_vars = base_vars.copy()
for l in range(1, 10):
    input_vars.extend([v + '_lag' + str(l*2) for v in base_vars])
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
pipe = Pipeline([('scaler', StandardScaler()), ('regression', MultiTaskLassoCV(alphas=alphas))])
pipe.fit(X_train, Y_train)
print('Chosen regularization parameter: alpha={:.3f}'.format(pipe.named_steps['regression'].alpha_))
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))

# %%
# Model coefficiens
coef = describe_regression(pipe, input_vars, output_vars)
coef

# %%
# Unneeded variables
coef[(coef['wave_u'] == 0) & (coef['wave_v'] == 0)].index

# %% [markdown]
# Here the strongest influence seems to be from the parameters lagging by 2 and 4 hours.
# Will keep 2 and 4 hour lag.

# %% [markdown]
# ## Ridge Regression - Wind only
# Ridge regression from Wind Vector, Wind Speed and Wind Speed Squared
# Lag for 2 hours and 4 hours.
#
# Inputs:
# * 10u
# * 10v
# * 10u_lag2
# * 10v_lag2
# * 10u_lag4
# * 10v_lag4
#
#
# Outputs:
# * wave_u
# * wave_v

# %%
# Build input and output matrices
base_vars = ['10u', '10v']
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
pipe = Pipeline([('scaler', StandardScaler()), ('regression', RidgeCV())])
pipe.fit(X_train, Y_train)
print('Chosen regularization parameter: alpha={:.3f}'.format(pipe.named_steps['regression'].alpha_))
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))

# %%
# Model coefficiens
describe_regression(pipe, input_vars, output_vars)

# %% [markdown]
# ## Ridge Regression - Wind only - More Features
# Ridge regression from Wind Vector, Wind Speed and Wind Speed Squared
# Lag for 2 hours and 4 hours.
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
pipe = Pipeline([('scaler', StandardScaler()), ('regression', RidgeCV())])
pipe.fit(X_train, Y_train)
print('Chosen regularization parameter: alpha={:.3f}'.format(pipe.named_steps['regression'].alpha_))
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))
plot_prediction(pipe, X_test, Y_test, func=lambda x: np.sqrt(np.sum(x**2, axis=1)))

# %%
# Model coefficiens
describe_regression(pipe, input_vars, output_vars)

# %% [markdown]
# Including the wind speed and the squared wind speed does improve the performance.

# %% [markdown]
# ## Support Vector Machine
# Support vector regression from Wind Vector, Wind Speed and Wind Speed Squared
# Lag for 2 hours and 4 hours.
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
pipe_u = Pipeline([('scaler', StandardScaler()), ('regression', LinearSVR())])
pipe_v = Pipeline([('scaler', StandardScaler()), ('regression', LinearSVR())])
pipe_u.fit(X_train, Y_train[:,0])
pipe_v.fit(X_train, Y_train[:,1])
print('Chosen regularization parameter (u): C={:.3f}'.format(pipe_u.named_steps['regression'].C))
print('Chosen regularization parameter (v): C={:.3f}'.format(pipe_v.named_steps['regression'].C))
print('Train accuracy of the model (u): {:.3f}'.format(pipe_u.score(X_train, Y_train[:,0])))
print('Train accuracy of the model (v): {:.3f}'.format(pipe_v.score(X_train, Y_train[:,1])))

# %%
# Test model
print('Test accuracy of the model (u): {:.3f}'.format(pipe_u.score(X_test, Y_test[:,0])))
print('Test accuracy of the model (v): {:.3f}'.format(pipe_v.score(X_test, Y_test[:,1])))
coef = describe_regression(pipe, input_vars, output_vars)
coef

# %%
# Unneeded variables
coef[(coef['wave_u'] == 0) & (coef['wave_v'] == 0)].index

# %% [markdown]
# Here the strongest influence seems to be from the parameters lagging by 2 and 4 hours.
# Will keep 2 and 4 hour lag.

# %% [markdown]
# ## Kernel Ridge Regression (Linear)
# **This model does not run, therefore it is commented out**
#
# It is unclear if the correlation between wind and wave height ist linear.
# Before experimenting with hand crafter features, we try kernelized ridge regression.
#
# Inputs:
# * 10u
# * 10v
# * 10u_lag2
# * 10v_lag2
# * 10u_lag4
# * 10v_lag4
#
#
# Outputs:
# * wave_u
# * wave_v

# %%
# Build input and output matrices
base_vars = ['10u', '10v']
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
# pipe = Pipeline([('scaler', StandardScaler()), ('regression', KernelRidge(kernel='linear'))])
# pipe.fit(X_train, Y_train)
# print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
# print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))
# plot_prediction(pipe, X_test, Y_test, func=lambda x: np.sqrt(np.sum(x**2, axis=1)))

# %% [markdown]
# ## Ridge Regression - Wave height only
# The correlation between speed and height seems to be stronger than the direction correlation.
# Try predicting the height independently of the direction.
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
alphas = np.logspace(-2, 5, num=20)
pipe = Pipeline([('scaler', StandardScaler()), ('regression', RidgeCV())])
pipe.fit(X_train, Y_train)
print('Chosen regularization parameter: alpha={:.3f}'.format(pipe.named_steps['regression'].alpha_))
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))
plot_prediction(pipe, X_test, Y_test)

# %%
describe_regression(pipe, input_vars, output_vars)

# %% [markdown]
# ## SGD Regressor - Huber Loss - Wave height only
# Try another loss function.
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
pipe = Pipeline([('scaler', StandardScaler()), ('regression', SGDRegressor(loss='huber'))])
pipe.fit(X_train, Y_train)
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))
plot_prediction(pipe, X_test, Y_test)

# %%
describe_regression(pipe, input_vars, output_vars)

# %% [markdown]
# ## SGD Regressor - Huber Loss - Wave U
# First part of SGD model for direction
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

# %%
# Build input and output matrices
base_vars = ['10u', '10v', 'wind_speed', 'wind_speed_sq']
input_vars = base_vars.copy()
for l in [2, 4]:
    input_vars.extend([v + '_lag' + str(l) for v in base_vars])
input_vars.sort()
output_vars = ['wave_u']

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
pipe = Pipeline([('scaler', StandardScaler()), ('regression', SGDRegressor(loss='huber'))])
pipe.fit(X_train, Y_train)
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))
plot_prediction(pipe, X_test, Y_test)

# %%
describe_regression(pipe, input_vars, output_vars)

# %% [markdown]
# ## SGD Regressor - Huber Loss - Wave V
# First part of SGD model for direction
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
# * wave_v

# %%
# Build input and output matrices
base_vars = ['10u', '10v', 'wind_speed', 'wind_speed_sq']
input_vars = base_vars.copy()
for l in [2, 4]:
    input_vars.extend([v + '_lag' + str(l) for v in base_vars])
input_vars.sort()
output_vars = ['wave_v']

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
pipe = Pipeline([('scaler', StandardScaler()), ('regression', SGDRegressor(loss='huber'))])
pipe.fit(X_train, Y_train)
print('Train accuracy of the model: {:.3f}'.format(pipe.score(X_train, Y_train)))

# %%
# Test model
print('Test accuracy of the model: {:.3f}'.format(pipe.score(X_test, Y_test)))
plot_prediction(pipe, X_test, Y_test)

# %%
describe_regression(pipe, input_vars, output_vars)