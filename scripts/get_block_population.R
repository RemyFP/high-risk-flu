library(tidycensus)
library(tidyverse)
library(dplyr)

# All states plus territories
state_list <- c(
  'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
  'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
  'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
  'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
  'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI',
  'WY', 'DC', 'PR'#, 'VI', 'GU', 'MP'
)

# List of geo values available at: https://walker-data.com/tidycensus/articles/basic-usage.html
geo_level <- "block"

# Request a key here: https://api.census.gov/data/key_signup.html
my_census_api_key <- "my_census_api_key"
census_api_key(my_census_api_key, install=TRUE, overwrite = TRUE)
readRenviron("~/.Renviron")

# To get the list of available variables
var_list <- load_variables(2020, "pl", cache = TRUE)

# Total population only
population_variable <- c(total_population = "P1_001N")

# Load population per block for all states
number_states_fetched <- 0
for (state in state_list) {
  df_pop_details_state <- get_decennial(
    geography = geo_level, 
    variables = population_variable, 
    state = state, 
    year = 2020)
  
  if (number_states_fetched == 0) {
    df_pop_details <- df_pop_details_state
  } else {
    df_pop_details <- rbind(df_pop_details, df_pop_details_state)
  }
  
  number_states_fetched <- number_states_fetched + 1
}

df_pop_details <- df_pop_details %>%
  select(block_GEOID = GEOID, total_population = value)

filename_out <- paste0("population_per_", geo_level, ".csv")
output_path <- paste0("./Parameters/ZCTA to County/", filename_out)
write.csv(df_pop_details, output_path)

