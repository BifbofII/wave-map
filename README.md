# Wave Prediction Map

A mini-project for the lecture _Introduction to Data Science_.

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