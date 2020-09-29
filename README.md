# AccountReporting
This package  can be used to aggregate bank account information from different sources and prepare the data for analysis.

## Installation

pip install git+https://github.com/DeadEntropy/AccountReporting

## Configuration

The package is driven by a configuration file which is used to specify where to load the data from as well as pass in 
additional information about the bank accounts. By default the package will look for the configuration in "config\config.ini"

## Use

The package is designed to work in 3 steps:
1. Load the data using the `transforms.master_transform` module
2. Process the Data using the `process.process` model
3. Analyse the Data
    1. the processed data can be used to visualise the aggregated bank accounts cashflows
    2. projections can be performed via the `projection.projection` module 
