# Wave Prediction Map

A mini-project for the lecture _Introduction to Data Science_.

Final result can be found from here: http://baltic-wave-map.herokuapp.com/

## TOC
* [Development Setup](#development-setup)
  - [Data Download](#data-download)
* [Project Proposal](#project-proposal)
* [Directories](#directories)
* [Exploratory data analysis](#explatory-data-analysis)
* [Model building](#model-building)
* [Final model](#final-model)
* [Visualization](#visualization)

## Development Setup

Run the following commands from the root of the project for setting up a python virtual environment, enabling it and installing all dependencies:
```
python3 -m venv .venv
source ./.venv/bin/activate
pip install -r dev-requirements.txt
```

Alternatively, tun the following commands to setup an anaconda environment for the project:
```
conda create -n wave-map python=3.6 anaconda
conda activate wave-map
conda install -n wave-map --file dev-requirements.txt
```

### Data Download

The big data file is not in Git LFS any more because the Git LFS capacity of GitHub ran out.
The Grib file can now be downloaded from [here](https://jabsserver.net/downloads/weather_data.grib) or with one of the following commands:

```
curl https://jabsserver.net/downloads/weather_data.grib > data/weather_data.grib
wget --show-progress -P data https://jabsserver.net/downloads/weather_data.grib
```

The following gives an overview of the most important data files.
More detailed information can be found in `data/README.md`.

#### weather_data.grib

It contains wind speed and direction, temperature (2m) and surface pressure data for the area from 16-34 degrees longitude and 59-66 degrees latitude.
The data is hourly for a timeframe of 01.01.2019 to the 16.09.2020.

#### weather_data.csv

The file contains weather data at the buoy locations.
This data was extracted from the Grib file.

#### wave_data.csv

The `fmi-download` folder contains CSV files that were downloaded from the web interface to FMIs API [here](https://en.ilmatieteenlaitos.fi/download-observations).
The CSV files were then cleaned up and joined together and exported as `wave_data.csv`.
Additional metadata about the wave buoys can be found in `wave_stations.csv`.

## Project Proposal

### Pitch

Sailing or boating generally is heavily affected by the weather elements.
Nowadays the weather forecast are generally quite good and you can get live weather data from quite a few measuring station.
Regardless all this the information/prediction about wave height is still quite poor due to the lack of measuring buoys (only four in the coast of Finland).
Our proposal is to make predictions about the wave height in different parts of Finland from weather data and visualize it more intuitively for the end user.

### Data

The data could be gathered from weather models like GFS, ECMWF, ICON etc.

### Data Analysis

We could use some machine learning tools and the wave buoys data to teach our model and to make predictions about the wave height.

### Communication of results

Preferably we could have a map of coast of Finland and representing the wave height in the different parts of the coast

### Operationalization

The added value would be from users point, that the sites of Ilmatieteenlaitos has only wave prediction from the four buoys, but not from the whole coast.
This way we could provide useful data for much wider audience of sailors and other seafarers.

## Directories

| Directory     |  Description                                                         |
|---------------|----------------------------------------------------------------------|
| assets        | css file for the webpage                                             |
| data          | data files used in for the models, visualization                     |
| documentation | plots and figures from the project                                   |
| models        | the resulting model from the data analysis used for the predictions  |
| predictions   | predicted data from the model used for the visualization             |

## Exploratory data analysis

### Wave data

A large amount of time for any data analyst goes into digging through the data and getting the grip of it. The exploratory data analysis has been done in `exploratory-data-analysis.ipynb`. The data we collected/used are the weather data from the grib files and the data from wave buoys in `wave_data.csv`. First thing with the wave data was it that a large part of the data was missing of it. 

![](https://github.com/BifbofII/wave-map/blob/master/documentation/plots/missing_values.png)
![](https://github.com/BifbofII/wave-map/blob/master/documentation/plots/kaikki.png)

(Wave buoys shown as orange dots and coast weather stations shown as red dots)

Large parts of buoy-data from Perämeri and Helsinki-Suomenlinna was missing.
Perämeri is the north-most buoy and Helsinki-Suomenlinna is the buoy in waters of Helsinki closer to the coast. One can see that the data is missing during winters.
This is due to the fact that the buoys are picked up if there is risk for the to get caught in the freezing sea.
In the north, where the water is frozen the longest the largest amount of data is missing (Perämeri) and Helsinki-Suomenlinna which is the buoy closest to the coast has the second longest period of missing data.

Some exploratory analysis of the wave height and direction was also made.

![](https://github.com/BifbofII/wave-map/blob/master/documentation/plots/wave_hight_and_dir1_9-20.png)

To reduce the amount of data in this figure, the plot contains one arrow per day.
The arrow gives the direction and the wave height recorded.
More of the exploratory plots can be found from `documentation/plots`.

### Wind/weather data

Quick quiver plot for the wind data for the selected region to see how to parse the grib file.
The grib file doesn't have any missing values since its data is already re-analysed by the copernicus project.

![](https://github.com/BifbofII/wave-map/blob/master/documentation/plots/quiver_plot.png)

To see if the data suggests any correlation between the wind and the wave data, the following plots were generated.

![](https://github.com/BifbofII/wave-map/blob/master/documentation/plots/wind_speed_wave_hight_19-20.png)
![](https://github.com/BifbofII/wave-map/blob/master/documentation/plots/Correlation_Wind_Speed_and_Wave_Height_(Suomenlahti).png)

So from these plots one can see that there is probably correlation between the wind speed and wave height. 

## Model building

Multiple test were ran in order to find the best model.
Here is the list of tested models used in `build-model.ipynb`.
More detailed explanation of the reasoning behind the models can be found from the mentioned file `build-model.ipynb`. 

All models were trained on data from four of the buoys and tested on the fifth (`Pohjois-itämeri`).

### Models
* Linear Regression
* Ridge Regression
* Lasso Regression (For variable selection)
* Support vector Machine
* SGD Regression

All models were tested with different parameters (if applicable) and with different variables as inputs.
The R2-score for the models ranges between 0.535 to 0.738.
 
## Final model
There are two distinct models, one predicting the wave height and one predicting the direction.
This approach is chosen because the wave height has a stronger correlation and this way the height prediction is better since it is not influenced by the direction.

All direction data is handled as vectors instead of degree values to get continuous data without a jump from 360 to 0 degrees.

#### Wave Height Model
* **`SGDRegression` with 'huber' loss** was used for predicting wave height
* Output: wave height
* Inputs: wind direction (as vector components), wind speed and squared wind speed of the last 4 hours
 
#### Wave Direction Model
* `RidgeRegression` was used for predicting wind direction
* Ridge is used because it can predict both vector components at once
* Cross validation is used to fix the regularization parameter
* Output: wave direction (as vector components)
* Inputs: wind direction, wind speed and squared wind speed of the last 4 hours

More detailed descriptions of the inputs and how the model is used in can be found in file `build-final-model.ipynb`.
 
## Visualization

The data visualization is done with the help of Plotly for generating the plots and Plotly Dash for creating a web application.
Since this is a proof of concept project, the web application does only show precomputed data for the week from the 01.08.2020 to the 08.08.2020.

The web app is build in `app.py`.
It includes controls for selecting the date and the data that should be shown.
Based on that, the data is selected, passed to the `create_vis` function from the `visualization.py` module and the returned plot is included in the Dash application.

The finished application is hosted on Heroku to create a deliverable demo application.