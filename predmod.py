import pickle
import pygrib
import datetime
import numpy as np
import pandas as pd
from memoization import cached
from abc import ABC, abstractmethod


class WavePredictor(ABC):
    """An abstract base class for wrapper classes around a wave model"""

    def __init__(self, model_path, grib_data=None):
        """
        Abstract function for initializing a predictor

        :param model_path: the path to a pickled model to load
        :param grib_data: all input data (as a grib file) that should be used for the model
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        if grib_data is not None:
            self.grib_data = pygrib.open(grib_data)
        else:
            self.grib_data = None

    def __del__(self):
        """Destructor closing the grib file"""
        if self.grib_data is not None:
            self.grib_data.close()

    @cached
    def extract_grib_data(self, location, time, short_names=None):
        """
        Extract data from the grib file for a specific location and or time

        :param location: the location as a (lat,lon) tuple
        :param time: the requested time as a datetime
        :param short_names: the short names of the parameters to return (as a list) (None for all)
        :returns: a pandas dataframe with the parameters as columns and the times as indices.
            If no location is specified, each cell contains a numpy 2D array, else a scalar.
            Also the lats, lons and times are also returned.
        :raises ValueError: if the location or time are out of the range in the data
        """
        if self.grib_data is None:
            raise ValueError('No data file was specified on initialization')
        if time is None:
            times = np.unique([datetime.datetime(grb.year, grb.month, grb.day, grb.hour) for grb in self.grib_data])
            self.grib_data.seek(0)
        else:
            if type(time) is list or type(time) is np.array:
                times = time
            else:
                times = np.array([time])

        if short_names is None:
            short_names = np.unique([grb.shortName for grb in self.grib_data])
            self.grib_data.seek(0)

        ex_msg = self.grib_data.message(1)
        lat, lon = ex_msg.latlons()
        lats = lat[:,0].flatten()
        lons = lon[0,:].flatten()
        if location is not None:
            lat_diff = np.abs(lats - location[0])
            lon_diff = np.abs(lons - location[1])
            lats = np.array([location[0]])
            lons = np.array([location[1]])
            location = (np.argmin(lat_diff), np.argmin(lon_diff))
        
        data = []
        for p in short_names:
            series = []
            for t in times:
                try:
                    grb = self.grib_data.select(shortName=p,
                        year=t.year, month=t.month, day=t.day, hour=t.hour)[0]
                except IndexError:
                    raise ValueError('The time or parameter was not found in the grib file')
                if location is None:
                    series.append(grb.values)
                else:
                    series.append(grb.values[location[0], location[1]])
            data.append(pd.Series(series, index=times, name=p))

        return pd.concat(data, axis=1), lats, lons, times

    @abstractmethod
    def predict(self, location=None, time=None):
        """
        Predict the wave height at a location and time

        :param location: an (optional) location to do the prediction at as a (lat,lon) tuple
        :param time: an (optional) time to do the prediction for as datetime.
            A list or array of times is also possible.
        :returns: a numpy array with different dimensions, depending on the call
            If location was specified, a array of (time x params) is returned.
            If location was not specified, but time was specified, a (lat x lon x params) grid is returned
            If neither location nor time are specified, a (lat x lon x time x params) array is returned.
            Also the lats, lons and times are also returned
        :raises ValueError: if the location or time are out of the range in the data
        """
        pass


class WindWaveModel(WavePredictor):
    """
    A wrapper for a wave model with the following inputs:
    
    * 10u
    * 10v
    * wind_speed
    * wind_speed_sq
    * 10u_lag2
    * 10v_lag2
    * wind_speed_lag2
    * wind_speed_sq_lag2
    * 10u_lag4
    * 10v_lag4
    * wind_speed_lag4
    * wind_speed_sq_lag4
    """

    def predict(self, location=None, time=None):
        if time is not None and type(time) is not list and type(time) is not np.array:
            time = [time]
        # Load data
        now, lats, lons, times = self.extract_grib_data(location=location, time=time, short_names=['10u', '10v'])
        now = now.copy()
        lag_2, _, _, _ = self.extract_grib_data(location=location,
            time=[t-datetime.timedelta(hours=2) for t in time],
            short_names=['10u', '10v'])
        lag_2 = lag_2.copy()
        lag_2.index = lag_2.index + datetime.timedelta(hours=2)
        lag_2.columns = [c + '_lag2' for c in lag_2.columns]
        lag_4, _, _, _ = self.extract_grib_data(location=location,
            time=[t-datetime.timedelta(hours=4) for t in time],
            short_names=['10u', '10v'])
        lag_4 = lag_4.copy()
        lag_4.index = lag_4.index + datetime.timedelta(hours=4)
        lag_4.columns = [c + '_lag4' for c in lag_4.columns]
        # Join data
        data = pd.concat([now, lag_2, lag_4], axis=1)
        # Compute wind speed and square
        for l in ['', '_lag2', '_lag4']:
            data['wind_speed_sq'+l] = data['10u'+l]**2 + data['10v'+l]**2
            data['wind_speed'+l] = data['wind_speed_sq'+l].map(np.sqrt)
        # Build input array
        base_vars = ['10u', '10v', 'wind_speed', 'wind_speed_sq']
        input_vars = base_vars.copy()
        for l in [2, 4]:
            input_vars.extend([v + '_lag' + str(l) for v in base_vars])
        input_vars.sort()
        if location is None:
            shape = data.iloc[0,0].shape
            pred = []
            for i in range(shape[0]):
                row = []
                for j in range(shape[1]):
                    X = data.applymap(lambda c: c[i,j])[input_vars].values
                    row.append(self.model.predict(X)[np.newaxis,np.newaxis,:,np.newaxis])
                pred.append(np.concatenate(row, axis=1))
            pred = np.concatenate(pred, axis=0)
            if pred.shape[2] == 1:
                pred = pred[:,:,0,:]
                if pred.shape[2] == 1:
                    pred = pred[:,:,0]
            elif pred.shape[3] == 1:
                pred = pred[:,:,:,0]
            return pred, lats, lons, times
        else:
            X = data[input_vars].values
            return self.model.predict(X), lats, lons, times


class JoinedWaveModel(WavePredictor):
    """A model joining the wave height and direction prediction"""

    def __init__(self, height_model, dir_model, grib_data):
        """Initialize the predictor by loading all models and the grib data"""
        super().__init__(height_model, grib_data)
        del self.model

        self.height_model = WindWaveModel(height_model)
        self.height_model.extract_grib_data = self.extract_grib_data

        self.dir_model = WindWaveModel(dir_model)
        self.dir_model.extract_grib_data = self.extract_grib_data

    def predict(self, location=None, time=None):
        # Get predictions of different models
        height, lats, lons, times = self.height_model.predict(location, time)
        dir, _, _, _ = self.dir_model.predict(location, time)

        # Scale factor for u and v components to include predicted wave height
        scale_factor = height / np.sqrt(np.sum(dir**2, axis=len(dir.shape)-1))

        # Calc returns
        dir = dir * np.concatenate(
            [np.expand_dims(scale_factor, len(scale_factor.shape)),
            np.expand_dims(scale_factor, len(scale_factor.shape))],
            axis=len(scale_factor.shape))

        return dir, lats, lons, times

# Main for debugging
if __name__ == '__main__':
    wm = JoinedWaveModel('models/wave-height-model.pkl', 'models/wave-dir-model.pkl', 'data/weather_data.grib')
    print(wm.predict(
        time=[datetime.datetime(2019, 2, 15, 6), datetime.datetime(2019, 2, 15, 10)]))