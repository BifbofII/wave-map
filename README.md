# Wave Prediction Map

A mini-project for the lecture _Introduction to Data Science_.

## TOC
  * [Development Setup](#development-setup)
  * [Data Download](#data-download)
  * [Project Proposal](#project-proposal)
  * [Directories](#directories)
  * [Explatory data analysis](#explatory-data-analysis)

## Development Setup

Run the following commands from the root of the project for setting up a python virtual environment, enabling it and installing all dependencies:
```
python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

Alternatively, tun the following commands to setup an anaconda environment for the project:
```
conda create -n wave-map python=3.6 anaconda
conda activate wave-map
conda install -n wave-map --file requirements.txt
```

### Data Download

The big data file is not in Git LFS anymore because the Git LFS capacity of GitHub ran out.
The Grib file can now be downloaded from [here](https://jabsserver.net/downloads/weather_data.grib) or with one of the following commands:

```
curl https://jabsserver.net/downloads/weather_data.grib > data/weather_data.grib
wget --show-progress -P data https://jabsserver.net/downloads/weather_data.grib
```
##### weather_data.grib

It contains wind speed and direction, temperature (2m) and surface pressure data for the are from 16-34 degrees longitude and 59-66 degrees latitude.
The data is hourly for a timeframe of 01.01.2019 to the 16.09.2020.

##### weather_data.csv

The file contains weather data at the buoy locations.
This data was extracted from the Grib file.

##### wave_data.csv

The `fmi-download` folder contains CSV files that were downloaded from the web interface to FMIs API [here](https://en.ilmatieteenlaitos.fi/download-observations).
The CSV files were then cleaned up and joined together and exported as `wave_data.csv`.
Additional metadata about the wave buoys can be found in `wave_stations.csv`.




## Project Proposal

### Pitch

Sailing or boating generally is heavily affected by the weather elements.
Nowadays the weather forecast are generally quite good and you can get live weather data from quite a few measuring station.
Regardless all this the information/prediction about wave hight is still quite poor due to the lack of measuring buoys (only four in the coast of Finland).
Our proposal is to make predictions about the wave hight in different parts of Finland from weather data and visualize it more intuitively for the end user.

### Data

The data could be gathered from weather models like GFS, ECMWF, ICON etc.

### Data Analysis

We could use some machine learning tools and the wave buoys data to teach our model and to make predictions about the wave hight.

### Communication of results

Preferably we could have a map of coast of Finland and representing the wave hight in the different parts of the coast

### Operationalization

The added value would ne from users point, that the sites of Ilmatieteenlaitos has only wave prediction from the four buoys, but not from the whole coast.
This way we could provide useful data for much wider audience of sailors and other seafarers.

## Directories

| Direcotry     |  Description                                                         |
|---------------|----------------------------------------------------------------------|
| assets        | css file for the webpage                                             |
| data          | data files used in for the models, visualization                     |
| documentation | plots and figures from the project                                   |
| models        | the resulting model from the data analysis used for the predictions  |
| predictions   | predicted data from the model used for the visualization             |

## Explatory data analysis

A large amount of time for any data analysist goes into digging through the data and getting the grip of it. The explatory data analysis has been done in `Explatory-data-analysis.ipynb`. The data we collected/used are the weather data from the grib files and the data from wave bouys in `wave_data.csv`. First thing with the wave data was it that a large part of the data was missing of it. 

![](https://github.com/BifbofII/wave-map/blob/master/documentation/plots/missing_values.png)
