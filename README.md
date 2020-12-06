# AccountReporting
This package  can be used to aggregate bank account information from different sources and prepare the data for analysis.

## Installation

pip install git+https://github.com/DeadEntropy/AccountReporting

## Configuration

The package is driven by a configuration file which is used to specify where to load the data from as well as pass in 
additional information about the bank accounts. By default the package will look for the configuration in "config\config.ini"

## High level Use

The package is provided with an easy to use UI available in 
- `ui`: load using `from bkanalysis.ui import ui`
    - `ui.load()`: loads the data from the folder specified in the config
    - `ui.get_current()`: generate a simple aggragated view of all the accounts
    - `ui.plot_wealth()`: plot a time series of the total wealth
    - `ui.project()`: generate projection of expected wealth
    - `ui.internal_flows()`: plot flows between accounts
    - `ui.plot_sunburst`: plot a sunburst graph

## Detailed Use

The package is designed to work in 3 steps:
1. Load the data using the `transforms.master_transform` module
2. Process the Data using the `process.process` model
3. Analyse the Data
    1. the processed data can be used to visualise the aggregated bank accounts cashflows
    2. projections can be performed via the `projection.projection` module 
