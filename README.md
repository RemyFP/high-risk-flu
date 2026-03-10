# Influenza high risk population

This repo is used to calculate the population at high-risk of complications following influenza infection at the zip code, county, state, and national level in the United States. The methodology is described in eAppendix 1 of the paper [Estimated Association of Construction Work With Risks of COVID-19 Infection and Hospitalization in Texas](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2772346).

## Contents

Outputs contains estimates of population at high-risk per age group at different geographical granularity levels.
Parameters contains inputs and parameters.
The folder scripts contains the main script `proportion_population_high_risk.py` that combines inputs and estimates high-risk population. The other scripts are used to get or parse inputs:
* `get_block_population.R`: Downloads population per census block from census.
* `load_block_to_zcta_txt_file.py`: Parses raw data from the census website mapping census blocks to ZCTAs.
* `pregnancy_county_level_calc.py`: Estimates county-level pregnancy rates per age group using the data in the subfolder /Parameters/Pregnancy.
* `zcta_to_county.py`: Creates crosswalk from ZCTA to county, with population overlap and creates, using data in folder /Parameters/ZCTA to County.


## Requirements

The code was run using Python 3.11.14, and the list of required packages is provided in the `requirements.txt` file.
