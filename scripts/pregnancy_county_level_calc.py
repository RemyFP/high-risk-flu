# -*- coding: utf-8 -*-
"""
Estimates county-level pregnancy rates using:
  - County-level fertility rates by age group
  - State-level abortion rates
  - National fetal-loss rates
Ages covered: 10 to 54 years old
"""
import os

import numpy as np
import pandas as pd


# ── Helper functions ───────────────────────────────────────────────────────────

def filter_df(df, conditions):
    """Filter a DataFrame by a list of (column, operation, values) conditions.

    Each condition is a tuple of (column_name, '==' or '!=', list_of_values).
    Conditions are combined with AND logic.
    """
    col, operation, elts = conditions[0]
    if operation == '==':
        fltr = df[col].isin(elts)
    elif operation == '!=':
        fltr = ~df[col].isin(elts)
    else:
        raise ValueError(f"Unsupported filter operation: '{operation}'. Use '==' or '!='.")

    for col, operation, elts in conditions[1:]:
        if operation == '==':
            fltr = fltr & df[col].isin(elts)
        elif operation == '!=':
            fltr = fltr & ~df[col].isin(elts)
        else:
            raise ValueError(f"Unsupported filter operation: '{operation}'. Use '==' or '!='.")

    return df.loc[fltr]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_params(paths):
    """Load and prepare all inputs needed for county-level pregnancy estimation.

    Returns:
        Dict of cleaned DataFrames, lists, and scalar parameters.
    """
    xl_file = pd.ExcelFile(paths['pregnancy_inputs'])
    state_data         = pd.read_excel(xl_file, 'State_Abortions_FetalLosses')
    age_groups_df      = pd.read_excel(xl_file, 'AgeGroups')
    county_data        = pd.read_excel(xl_file, 'CountyFertility')
    county_population  = pd.read_excel(xl_file, 'CountyPopulation_FemaleMale')
    county_list        = pd.read_excel(xl_file, 'CountyList')

    # --- Rename for convenience ---
    county_data.rename(columns={
        'Final Fertility Rate':       'Fertility',
        'County of Residence Code':   'CountyID',
    }, inplace=True)

    # --- Age group mappings ---
    age_groups_df['AgeCode']   = age_groups_df['AgeCode'].apply(str)
    age_groups_df['AgeGroup']  = age_groups_df['AgeGroup'].apply(str)
    age_groups_list  = age_groups_df['AgeGroup'].tolist()
    ages_for_avg     = age_groups_df.loc[age_groups_df['ForAverage'] > 0, 'AgeGroup'].tolist()

    # --- Format ID columns ---
    county_data['CountyID']       = county_data['CountyID'].apply(str)
    county_data['AgeCode']        = county_data['AgeCode'].apply(str)
    county_population['AgeCode']  = county_population['AgeCode'].apply(str)

    county_list['CountyID'] = county_list.apply(
        lambda x: str(x['StateNb']) + '{:03d}'.format(x['CountyNb']), axis=1)
    county_population['CountyID'] = county_population.apply(
        lambda x: str(x['StateNb']) + '{:03d}'.format(x['CountyNb']), axis=1)

    return {
        'state_data':        state_data,
        'age_groups_df':     age_groups_df,
        'age_groups_list':   age_groups_list,
        'ages_for_avg':      ages_for_avg,
        'county_data':       county_data,
        'county_population': county_population,
        'county_list':       county_list,
    }


# ── Analysis functions ─────────────────────────────────────────────────────────

def run_county_pregnancy(paths, p):
    """Compute and save pregnancy rates and counts per county and age group.

    Strategy:
      1. Counties not explicitly listed in the fertility data are mapped to
         their state's catch-all "unidentified" county (county code 999).
      2. A weighted-average fertility rate across ages 15–44 is computed for
         each county using the female population as weights.
      3. The overall pregnancy rate (at the 15–44 level) combines births,
         abortions, and fetal losses, each weighted by how long a woman is
         at risk during the year.
      4. Age-specific rates are obtained by scaling the overall rate by each
         age group's share of the average fertility rate.
      5. Both rates and counts (rate × female population) are exported.

    Args:
        paths: dict
            Must contain:
              - 'pregnancy_inputs': path to the input Excel workbook.
              - 'out_pregnancy':    path for the output CSV.
        p: dict
            Shared parameters from load_params.
    """
    # Proportion of the year a woman is at risk, by pregnancy outcome
    p_birth    = 9 / 12
    p_abortion = 2 / 12
    p_loss     = 3 / 12

    # ── Map each county to its fertility reference county ──────────────────────
    county_fertility_id_list = pd.unique(p['county_data']['CountyID'])
    county_id_list           = p['county_list']['CountyID'].tolist()

    not_in_fertility = [x for x in county_id_list if x not in county_fertility_id_list]
    county_list_unidentified = filter_df(p['county_list'], [['CountyID', '==', not_in_fertility]])
    county_list_unidentified = county_list_unidentified.copy()
    county_list_unidentified['RefCountyID'] = county_list_unidentified['StateNb'].apply(str) + '999'

    county_list = pd.merge(p['county_list'],
                           county_list_unidentified[['CountyID', 'RefCountyID']],
                           on='CountyID', how='left')
    county_list['RefCountyID'] = county_list.apply(
        lambda x: x['CountyID'] if pd.isnull(x['RefCountyID']) else x['RefCountyID'], axis=1)

    # ── Pivot fertility data from long to wide ─────────────────────────────────
    county_data = pd.merge(p['county_data'], p['age_groups_df'][['AgeCode', 'AgeGroup']],
                           on='AgeCode', how='left')
    county_fertility_wide = (county_data[['CountyID', 'AgeGroup', 'Fertility']]
                             .pivot(index='CountyID', columns='AgeGroup', values='Fertility')
                             .reset_index()
                             .rename(columns={'CountyID': 'RefCountyID'}))

    county_list = pd.merge(county_list, county_fertility_wide, on='RefCountyID', how='left')

    # ── Attach female population per county and age group ─────────────────────
    county_population_long = filter_df(p['county_population'],
                                       [['AgeGroup', '==', p['age_groups_list']]])
    county_population_wide = (county_population_long
                              .pivot(index='CountyID', columns='AgeGroup', values='Female')
                              .reset_index())
    new_colnames = ['popF_' + x for x in p['age_groups_list']]
    county_population_wide.rename(
        columns=dict(zip(p['age_groups_list'], new_colnames)), inplace=True)

    county_list = pd.merge(county_list, county_population_wide, on='CountyID', how='left')

    # ── Average fertility rate (ages 15–44, female-population weighted) ────────
    county_list['avg_fertility'] = county_list.apply(
        lambda x: (sum(x[a] * x['popF_' + a] for a in p['ages_for_avg']) /
                   sum(x['popF_' + a] for a in p['ages_for_avg'])),
        axis=1)

    # ── Overall pregnancy rate (15–44 average) ─────────────────────────────────
    county_list = pd.merge(
        county_list,
        p['state_data'][['StateCode', 'AbortionRate', 'FetalLoss_Births_prop']],
        left_on='State', right_on='StateCode', how='left')

    county_list['FetalLosses'] = county_list['avg_fertility'] * county_list['FetalLoss_Births_prop']
    county_list['Pregnancy'] = (
        county_list['avg_fertility'] * p_birth +
        county_list['AbortionRate']  * p_abortion +
        county_list['FetalLosses']   * p_loss
    ) / 1000

    # ── Scale overall rate to each age group ──────────────────────────────────
    counties_pregnancy = county_list.copy()
    for ag in p['age_groups_list']:
        counties_pregnancy[ag] = (counties_pregnancy['Pregnancy'] *
                                  counties_pregnancy[ag] /
                                  counties_pregnancy['avg_fertility'])

    # ── Compute counts (rate × female population) ──────────────────────────────
    counties_pregnancy['Count_Rate'] = 'Rate'
    counties_pregnancy_count = counties_pregnancy.copy()
    counties_pregnancy_count['Count_Rate'] = 'Count'
    for ag in p['age_groups_list']:
        counties_pregnancy_count[ag] = (counties_pregnancy_count[ag] *
                                        counties_pregnancy_count['popF_' + ag])

    counties_pregnancy = pd.concat([counties_pregnancy, counties_pregnancy_count],
                                   ignore_index=True)

    # ── Export ─────────────────────────────────────────────────────────────────
    out_cols = (['StateNb', 'CountyNb', 'State', 'StateLong', 'CountyName',
                 'CountyID', 'Count_Rate'] + p['age_groups_list'])
    counties_pregnancy[out_cols].to_csv(paths['out_pregnancy'], index=False)
    print(f"County pregnancy results saved to: {paths['out_pregnancy']}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    np.set_printoptions(linewidth=125)
    pd.set_option('display.width', 125)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    DATA_DIR = os.path.join(os.getcwd(), 'Parameters', 'Pregnancy')

    paths = {
        'pregnancy_inputs': os.path.join(DATA_DIR, 'Pregnancy Data - Inputs - 2021 02.xlsx'),
        'out_pregnancy':    os.path.join(DATA_DIR,
                                         'Pregnancy rates and count per county and age group.csv'),
    }

    p = load_params(paths)
    run_county_pregnancy(paths, p)


if __name__ == '__main__':
    main()
