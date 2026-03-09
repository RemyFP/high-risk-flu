"""
This script loads the raw data from the census website mapping census blocks to ZCTAs,
extracts the relevant columns (GEOID of both), and saves it as a csv file
for use in the proportion_population_high_risk script.
"""

import pandas as pd
import os

# File paths
BASE_DIR   = os.getcwd()
PARAMS_DIR = os.path.join(BASE_DIR, 'Parameters')
input_path = os.path.join(PARAMS_DIR, 'ZCTA to County', 'tab20_zcta520_tabblock20_natl.txt')
output_path = os.path.join(PARAMS_DIR, 'ZCTA to County', 'block_to_zcta.csv')

# Load data
df = pd.read_csv(input_path, sep='|', usecols=['GEOID_ZCTA5_20','GEOID_TABBLOCK_20'])

# Save data
df.to_csv(output_path, index=False)