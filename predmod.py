import pickle
import pygrib
import datetime
import numpy as np
import pandas as pd
from memoization import cached
from abc import ABC, abstractmethod


class WavePredictor(ABC):
    """An abstract base class for wrapper classes around a wave model"""

    def __init__(self, model_path, grib_data):
        """
        Abstract function for initializing a predictor

        :param model_path: the path to a pickled model to load
        :param grib_data: all input data (as a grib file) that should be used for the model
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.grib_data = pygrib.open(grib_data)

    def __del__(self):
        """Destructor closing the grib file"""
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
        if time is None:
            times = np.unique([datetime.datetime(grb.year, grb.month, grb.day, grb.hour) for grb in self.grib_data])
            self.grib_data.seek(0)
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
                grb = self.grib_data.select(shortName=p, year=t.year, month=t.month, day=t.day, hour=t.hour)[0]
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
        :param time: an (optional) time to do the prediction for as datetime
        :returns: a numpy array with different dimensions, depending on the call
            If location was specified, a array of (time x params) is returned.
            If location was not specified, but time was specified, a (lat x lon x params) grid is returned
            If neither location nor time are specified, a (lat x lon x time x params) array is returned.
            Also the lats, lons and times are also returned
        :raises ValueError: if the location or time are out of the range in the data
        """
        pass


class SGDWaveModel(WavePredictor):
    """A wrapper for a SGDRegressor wave model"""

    def predict(self, location=None, time=None):
        # Load data
        now, lats, lons, times = self.extract_grib_data(location=location, time=time, short_names=['10u', '10v'])
        lag_2, _, _, _ = self.extract_grib_data(location=location, time=time-datetime.timedelta(hours=2),
            short_names=['10u', '10v'])
        lag_2.index = lag_2.index + datetime.timedelta(hours=2)
        lag_2.columns = [c + '_lag2' for c in lag_2.columns]
        lag_4, _, _, _ = self.extract_grib_data(location=location, time=time-datetime.timedelta(hours=4),
            short_names=['10u', '10v'])
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
            if time is not None:
                pred = pred[:,:,0,:]
            return pred, lats, lons, times
        else:
            X = data[input_vars].values
            return self.model.predict(X)[:,np.newaxis].T, lats, lons, times


# Main for debugging
if __name__ == '__main__':
    wh = SGDWaveHeight('models/wave-height-model.pkl', 'data/weather_data.grib')
    print(wh.predict(time=datetime.datetime(2019, 2, 15, 6)))
    print(wh.predict(location=(60.123,24.972)))