# Project Plan

## ToDo

* [x] Download wave data
* [x] Download weather data
  * [x] Check format
* [ ] Build model
  * [ ] Decide on model
* [ ] Visualize data
  * [ ] Coastal map
  * [ ] Visualize Buoys
  * [ ] Visualize predicted data

## Project Stages

1. Data Collection
2. Model Construction
3. Visualization

### Data Collection

#### Wave Data

Collection from the FMI through the [manual interface](https://en.ilmatieteenlaitos.fi/download-observations).
Join the data together.

#### Weather Data

Where to download the weather data from?

* [Reanalysed Observation Data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form)
* [Data from the Swedish Meteorological and Hydrological Institute](https://opendata.smhi.se/apidocs/metfcst/get-forecast.html)
* [Marine Prediction API](https://docs.tidetech.org/data-api/?shell#introduction)]

### Build Model

What model do we choose?

* Linear Regression
* LSTM

Start with simple models and try to improve them. From LR with current weather data to more complex models including older data.

### Visualize the Data

Map of coast of Finland with Buoy locations marked, wave height as colour and wave direction as arrows. Alpha mask around the buoys to indicate decreasing accuracy of the model.

[Windy.com](https://api.windy.com/map-forecast) seems to have an API for visualizing data on a map. (Probably the free version isn't of much use)

[Plotly](https://plotly.com) provides [map layers](https://plotly.com/python/mapbox-layers/) e.g. from open street maps where plots can be created on.

## Timeline

* Week 38: Finish data collection
* Week 39: Data pre-processing
* Week 40: Finished model and graphical interface
* Week 41: Merging of interface and model
* Week 42: Final tweaks to the almost finished project
* Week 43: Presentation and delivery