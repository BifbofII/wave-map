# Overview of Data Files

## Wave Data - Open Data API
The files `wave_data_od.csv` and `wave_stations_od.csv` contain data downloaded from the FMI WFS API.
The data is not correctly parsed (yet) and should not be used.

## Wave Data - Manually Downloaded
The `fmi-download` folder contains CSV files that were downloaded from the web interface to FMIs API [here](https://en.ilmatieteenlaitos.fi/download-observations).
The CSV files were then cleaned up and joined together and exported as `wave_data.csv`.
Additional metadata about the wave buoys can be found in `wave_stations.csv`.

The data contains information about the wave height and direction at the five buoys from the 01.01.2019 to the 14.09.2020 in intervals of 30 minutes.

## Weather Data
The `weather_data.grib` file was downloaded from [Copernicus](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form) and contains wind speed and direction, temperature (2m) and surface pressure data for the are from 16-34 degrees longitude and 59-66 degrees latitude.
The data is hourly for a timeframe of 01.01.2019 to the 16.09.2020.

### Subset of Weather Data at Buoy Locations
The `weather_data.csv` file contains weather data at the buoy locations.
This data was extracted from the Grib file.