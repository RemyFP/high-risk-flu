# -*- coding: utf-8 -*-
"""
Calculates the proportion of individuals at high risk of severe influenza
outcomes per age group, for:
  - Each zip code (ZCTA) in the US
  - Each county in the US
  - The US as a whole (population-weighted average across zip codes)
"""
import os
import copy
import itertools

import numpy as np
import pandas as pd
import us


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


def compute_proportion_n(n, prevalence_dict, high_risk_conditions_list,
                         chronic_conditions_list):
    """Compute the proportion of the population with at least one high-risk
    flu condition, given that they have exactly n chronic conditions. 
    Expressed as a float between 0 and 1.
    Approach: the function estimates the probability of each possible combination
    of n conditions, and sums the probabilities of the combinations that include
    at least one high risk condition.
    """
    permutations = list(itertools.permutations(chronic_conditions_list, n))

    probs_high_risk = []
    for permut in permutations:
        p = 1
        prevalence_p = copy.deepcopy(prevalence_dict)
        for condition in permut:
            p = p * prevalence_p[condition] / sum(prevalence_p.values())
            del prevalence_p[condition]

        if any(condition in permut for condition in high_risk_conditions_list):
            probs_high_risk.append(p)

    return sum(probs_high_risk)


def adjust_mcc(mcc_df, risk_score, number_mcc_list):
    """Scale multiple chronic condition (MCC) prevalence by a location's risk
    score, then renormalize if the total exceeds 1.

    Args:
        mcc_df: MCC prevalence DataFrame at the national level.
        risk_score: Ratio of local raw prevalence sum over the national average.
        number_mcc_list: Possible counts of chronic conditions (e.g. [1, 2, 3]).

    Returns:
        Adjusted MCC DataFrame for the location.
        Rows are age groups of adults (18_44, 45_64, 65+), columns are number of
        chronic conditions (0, 1, 2, 3). Values are proportions of the
        population in each age group with that number of chronic conditions
        expressed as floats (0-1).
    """
    mcc_ztca = mcc_df.copy()
    for n_mcc in number_mcc_list:
        mcc_ztca[n_mcc] = mcc_ztca[n_mcc] * risk_score

    # Renormalize rows where MCC proportions sum above 1
    for i_mcc, mcc_row in mcc_ztca.iterrows():
        if sum(mcc_row.loc[number_mcc_list]) > 1:
            mcc_ztca.loc[i_mcc, number_mcc_list] = (
                mcc_row.loc[number_mcc_list] / sum(mcc_row.loc[number_mcc_list])
            )

    mcc_ztca[0] = 1 - mcc_ztca[number_mcc_list].sum(axis=1)
    return mcc_ztca


def get_zcta_high_risk_inputs(zcta, zcta_row, chronic_conditions_list,
                               places_obesity, zcta_to_county, pregnancy_counties,
                               pregnancy_age_groups, pregnancy_counties_list,
                               missing_counties, renamed_county_codes,
                               county_population):
    """Retrieve all location-specific inputs needed to calculate the high-risk
    population proportion for a given ZCTA.
    
    Args:
        zcta: int
            The ZCTA for which to retrieve inputs.
        zcta_row: pd DataFrame 
            Row of the places_data DataFrame corresponding to the ZCTA. Contains
            prevalence of chronic conditions and risk score for the ZCTA.
        chronic_conditions_list: list
            List of chronic conditions considered in the analysis.
        places_obesity: pandas DataFrame
            DataFrame containing obesity prevalence by ZCTA (columns: ZCTA5, OBESITY, Population).
        zcta_to_county: pandas DataFrame
            DataFrame mapping ZCTAs to counties and their populations. For each ZCTA, 
            contains one or more rows with the overlapping counties and their population counts.
        pregnancy_counties: pandas DataFrame
            DataFrame containing pregnancy rates by county and age group. Each row corresponds
            to a county, with columns giving pregnancy rate in various age groups.
        pregnancy_age_groups: list of str
            List of age groups for which pregnancy rates are available.
        pregnancy_counties_list: list of str
            List of counties for which pregnancy data is available.
        missing_counties: dict
            keys are int of zcta, values are list of str of county codes that overlap
            with the zcta but are missing pregnancy data.
            Dict to store any counties missing pregnancy data, keyed by ZCTA. Filled
            in-place by this function when missing counties are encountered.
        renamed_county_codes: dict
            Keys are county codes as string, values are county codes as string.
            Dict mapping old county codes to new ones for counties that were renamed.
        county_population: pandas DataFrame
            DataFrame containing population counts by county and age group.

    Returns:
        risk_score: float
            Risk score for the ZCTA. Calculated as the sum of chronic condition 
            prevalences in the ZCTA divided by the national weighted average prevalence sum.
        prevalence_dict: dict
            Dictionary mapping chronic condition names to their prevalence in the ZCTA.
            Prevalence expressed as a percentage (0-100).
        obesity: float
            Obesity prevalence in the ZCTA.
            Prevalence expressed as a percentage (0-100).
        pregnancy_dict: dict
            Dictionary mapping age groups to pregnancy rates in the ZCTA.
            Keys are age groups as string (e.g. '18-24'), values are pregnancy rates as 
            floats (0-1).
        female_prop: float
            Proportion of females in the ZCTA's population
            Keys are age groups as string (e.g. '18-24'), values are proportiong of 
            women in population in age group, rates as floats (0-1).
    """
    prevalence_dict = {x: zcta_row[x].values[0] for x in chronic_conditions_list}
    risk_score = zcta_row['risk_score'].values[0]
    obesity = filter_df(places_obesity, [['ZCTA5', '==', [zcta]]])['OBESITY'].values[0]

    # Identify counties overlapping this ZCTA
    zcta_i_counties = filter_df(zcta_to_county, [['ZCTA5', '==', [zcta]]])
    zcta_county_list_updated = [
        renamed_county_codes.get(x,x)
        if renamed_county_codes.get(x,x) in county_population['CountyID'].values
        else x
        for x in zcta_i_counties['CountyID'].tolist()
    ]

    # Drop counties with no pregnancy data; raise if none remain
    zcta_missing_county = [x for x in zcta_i_counties['CountyID']
                           if x not in pregnancy_counties_list]
    if len(zcta_missing_county) > 0:
        missing_counties[zcta] = zcta_missing_county

        if len(zcta_missing_county) < len(zcta_i_counties):
            zcta_i_counties = filter_df(zcta_i_counties,
                                        [['CountyID', '!=', zcta_missing_county]])
        else:
            to_replace = list(renamed_county_codes.keys())
            if any(x in zcta_missing_county for x in to_replace):
                # e.g. South Dakota: Shannon county renamed to Oglala county
                zcta_i_counties = zcta_i_counties.replace(
                    {'CountyID': renamed_county_codes})
                zcta_i_counties = filter_df(zcta_i_counties,
                                            [['CountyID', '!=', zcta_missing_county]])
            else:
                raise ValueError('Could not find a county for ZCTA', zcta)

    zcta_i_counties = zcta_i_counties.copy()
    zcta_i_counties['PopulationProp'] = (zcta_i_counties['Population'] /
                                         zcta_i_counties['Population'].sum())
    pop_county_prop = dict(zip(zcta_i_counties['CountyID'],
                               zcta_i_counties['PopulationProp']))

    # Population-weighted average pregnancy rates across counties
    zcta_pregnancy_rows = filter_df(
        pregnancy_counties,
        [['CountyID', '==', zcta_i_counties['CountyID'].tolist()],
         ['Count_Rate', '==', ['Rate']]])
    zcta_pregnancy_rows = pd.merge(zcta_pregnancy_rows,
                                   zcta_i_counties[['CountyID', 'Population']],
                                   on='CountyID', how='left')
    pregnancy_dict = {
        x: (sum(zcta_pregnancy_rows[x] * zcta_pregnancy_rows['Population']) /
            zcta_pregnancy_rows['Population'].sum())
        for x in pregnancy_age_groups
    }

    # Proportion of women in each age group, averaged over counties
    zcta_county_pop = filter_df(
        county_population,
        [['CountyID', '==', zcta_county_list_updated],
         ['AgeGroup', '==', pregnancy_age_groups]])
    zcta_county_pop = zcta_county_pop.copy()
    zcta_county_pop['FemaleProp'] = zcta_county_pop.apply(
        lambda x: x['TOT_FEMALE'] / x['TOT_POP'] if x['TOT_POP'] > 0 else 0.,
        axis=1)
    print(f"ZCTA {zcta}")
    female_prop_counties = {
        county: {
            ag: filter_df(zcta_county_pop,
                          [['CountyID', '==', [county]],
                           ['AgeGroup', '==', [ag]]])['FemaleProp'].values[0]
            for ag in pregnancy_age_groups
        }
        for county in zcta_county_list_updated
    }

    for county in list(pop_county_prop.keys()):
        if county in renamed_county_codes.keys():
            pop_county_prop[renamed_county_codes[county]] = pop_county_prop.pop(county)
    
    if zcta in [99566, 99573, 99586]:
        pop_county_prop['2066'] = pop_county_prop.pop('2063')
    
    female_prop = {
            ag: sum(pop_county_prop[county] * female_prop_counties[county][ag]
                    for county in pop_county_prop)
            for ag in pregnancy_age_groups
        }

    return risk_score, prevalence_dict, obesity, pregnancy_dict, female_prop


def calculate_high_risk_prop(mcc_ztca, number_mcc_list, age_groups_mcc,
                              prevalence_dict, high_risk_conditions_list,
                              chronic_conditions_list, obesity,
                              obese_with_condition, w_avg_obesity,
                              children_obesity_dict, age_groups_df,
                              detail_age_groups, pregnancy_dict, female_prop,
                              pregnancy_age_groups):
    """Calculate the proportion of high-risk individuals in each age group.

    Combines four risk factors in sequence, each adding individuals not already
    counted as high-risk:
      1. Chronic conditions — via the MCC framework (probability of having at
         least one high-risk condition given n total chronic conditions).
      2. Obesity — adds obese individuals without a pre-existing high-risk
         condition. For adults, uses the local obesity prevalence scaled by the
         share of obese people who do not already have a chronic condition
         (1 - obese_with_condition). For children, scales national age-specific
         obesity risk by the local-to-national obesity ratio.
      3. Age-group mapping — converts MCC-level age groups (e.g. '18_44') into
         the finer detail_age_groups (e.g. '20_24') using weighted averages
         defined in age_groups_df.
      4. Pregnancy — adds pregnant women not already counted, calculated as the
         pregnancy rate times the female proportion of each age group.

    Args:
        mcc_ztca: pandas DataFrame
            Location-adjusted MCC prevalence. Rows are adult age groups
            (AgeGroupMCC column), columns include 1/2/3 (number of chronic
            conditions) and 0 (no condition). Values are proportions (0–1).
        number_mcc_list: list of int
            Possible counts of chronic conditions, typically [1, 2, 3].
        age_groups_mcc: list of str
            Adult age group labels used in mcc_ztca (e.g. ['18_44', '45_64', '65+']).
        prevalence_dict: dict
            Maps each condition name (str) to its local prevalence as a
            percentage (0–100).
        high_risk_conditions_list: list of str
            Subset of chronic conditions that are considered high-risk for flu.
        chronic_conditions_list: list of str
            All chronic conditions included in the MCC analysis.
        obesity: float
            Local obesity prevalence as a percentage (0–100).
        obese_with_condition: float
            National proportion of obese individuals who already have at least
            one chronic condition (0–1). Used to avoid double-counting.
        w_avg_obesity: float
            National population-weighted average obesity prevalence (0–100),
            used to scale children's obesity risk to the local level.
        children_obesity_dict: dict
            Maps child age group labels (str) to their national high-risk
            proportion due to obesity (float, 0–1).
        age_groups_df: pandas DataFrame
            Lookup table mapping each detailed age group to one or two MCC-level
            base groups and their weights (columns: AgeGroup, AgeGroup_1,
            AgeGroup_1_weight, AgeGroup_2, AgeGroup_2_weight).
        detail_age_groups: list of str
            Fine-grained age group labels for the final output
            (e.g. ['0_0.5', '0.5_4', '5_9', ..., '75_84', '85+']).
        pregnancy_dict: dict
            Maps pregnancy-eligible age group labels (str) to the local
            pregnancy rate (float, 0–1).
        female_prop: dict
            Maps pregnancy-eligible age group labels (str) to the proportion
            of females in the local population for that age group (float, 0–1).
        pregnancy_age_groups: list of str
            Age group labels for which pregnancy rates are available.

    Returns:
        dict mapping each label in detail_age_groups (str) to the proportion
        of the local population in that age group who are high-risk (float, 0–1).
    """
    # Proportion at risk given exactly n chronic conditions (location-level)
    high_risk_location = {
        n_mcc: compute_proportion_n(n_mcc, prevalence_dict,
                                    high_risk_conditions_list,
                                    chronic_conditions_list)
        for n_mcc in number_mcc_list
    }

    # Aggregate across MCC counts per adult age group
    high_risk_ages = {}
    for ag in age_groups_mcc:
        mcc_a_g = filter_df(mcc_ztca, [['AgeGroupMCC', '==', [ag]]])
        high_risk_ages[ag] = sum(
            mcc_a_g.loc[:, n_mcc].values[0] * high_risk_location[n_mcc]
            for n_mcc in number_mcc_list
        )

    # Add obese individuals not already counted in the high-risk population
    high_risk_adj_obesity = {}
    for ag in age_groups_mcc:
        new_high_risk_obese = (
            (1 - max(high_risk_ages[ag], obese_with_condition)) * obesity / 100
        )
        high_risk_adj_obesity[ag] = high_risk_ages[ag] + new_high_risk_obese

    # Children: scale national obesity rate by local obesity ratio
    obesity_ratio = obesity / w_avg_obesity
    for ag, ag_risk in children_obesity_dict.items():
        high_risk_adj_obesity[ag] = ag_risk * obesity_ratio

    # Map detailed age groups to their MCC-based age group(s) using weights
    high_risk_all = {}
    for ag in detail_age_groups:
        ag_df = filter_df(age_groups_df, [['AgeGroup', '==', [ag]]])
        base_groups = [ag_df[x].values[0] for x in ['AgeGroup_1', 'AgeGroup_2']
                       if pd.notnull(ag_df[x].values[0])]
        base_weights = [ag_df[x].values[0]
                        for x in ['AgeGroup_1_weight', 'AgeGroup_2_weight']
                        if pd.notnull(ag_df[x].values[0])]
        high_risk_all[ag] = sum(
            high_risk_adj_obesity[base_groups[i]] * base_weights[i]
            for i in range(len(base_groups))
        )

    # Add pregnant women not already in the high-risk population
    prop_pregnant = {
        ag: pregnancy_dict[ag] * female_prop[ag] if ag in pregnancy_age_groups else 0
        for ag in detail_age_groups
    }
    high_risk_final = {
        ag: high_risk_all[ag] + prop_pregnant[ag] * (1 - high_risk_all[ag])
        for ag in detail_age_groups
    }

    return high_risk_final


def df_calc_weighted_avg(df, metric_col, weight_col):
    """Return the weighted average of metric_col using weight_col as weights."""
    return (df[weight_col] * df[metric_col]).sum() / df[weight_col].sum()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_shared_params(paths):
    """Load and prepare all parameters shared across US ZCTA, France, and US
    total analyses.

    Returns:
        Dict of cleaned DataFrames, lists, and scalar parameters.
    """
    # --- Parameter Excel file ---
    xl_file = pd.ExcelFile(paths['params'])
    conditions_df          = pd.read_excel(xl_file, 'ConditionsList')
    mcc_df                 = pd.read_excel(xl_file, 'MultipleConditionsAdults')
    obesity_param          = pd.read_excel(xl_file, 'Obesity')
    age_groups_df          = pd.read_excel(xl_file, 'AgeGroups')
    children_obesity_df    = pd.read_excel(xl_file, 'ChildrenObesity')

    # --- External CSVs / Excel files ---
    places_raw         = pd.read_csv(paths['places'])
    pregnancy_counties = pd.read_csv(paths['pregnancy'])
    county_population         = pd.read_csv(paths['county_pop_csv'])
    zcta_to_county            = pd.read_csv(paths['zcta_to_county'])
    county_rename_CT         = pd.read_csv(paths['county_rename_CT'])
    county_population_age_map = pd.read_csv(paths['county_age_map'])
    
    county_population = pd.merge(county_population, county_population_age_map,
                                 on='AGEGRP', how='left')
    zcta_age_pop = pd.read_csv(paths['zcta_age_pop'])

    # --- Condition lists ---
    chronic_conditions_list   = filter_df(conditions_df,
        [['ChronicCondition',   '==', [1]]])['MeasureId'].tolist()
    high_risk_conditions_list = filter_df(conditions_df,
        [['HighRiskCondition',  '==', [1]]])['MeasureId'].tolist()

    # --- PLACES data: keep crude prevalence columns, rename for convenience ---
    org_cols    = ['ZCTA5'] + [x + '_CrudePrev' for x in chronic_conditions_list]
    clean_cols  = ['ZCTA5'] + chronic_conditions_list
    places_raw['TotalPopulation'] = places_raw['TotalPopulation'].str.replace(',', '').astype(int)
    places_data = places_raw[org_cols].copy()
    places_data.rename(columns=dict(zip(org_cols, clean_cols)), inplace=True)

    places_population = places_raw[['ZCTA5', 'TotalPopulation']]

    places_obesity = places_raw[['ZCTA5', 'OBESITY_CrudePrev', 'TotalPopulation']].copy()
    places_obesity.columns = ['ZCTA5', 'OBESITY', 'Population']

    # --- Risk score: local prevalence sum relative to national weighted average ---
    places_data['PrevalenceSum'] = places_data[chronic_conditions_list].sum(axis=1)
    weighted_avg_df = pd.merge(places_population,
                               places_data[['ZCTA5', 'PrevalenceSum']],
                               on='ZCTA5', how='left')
    weighted_avg_sum = df_calc_weighted_avg(weighted_avg_df,
                                            'PrevalenceSum', 'TotalPopulation')
    places_data['risk_score'] = places_data['PrevalenceSum'] / weighted_avg_sum

    # --- Obesity parameters ---
    children_obesity_dict = dict(zip(children_obesity_df['AgeGroupObesity'],
                                     children_obesity_df['Obesity']))
    obese_with_condition = obesity_param['ObeseWithCondition'].values[0]
    w_avg_obesity = df_calc_weighted_avg(places_obesity, 'OBESITY', 'Population')

    # --- ZCTA-to-county mapping ---
    zcta_to_county = zcta_to_county[['ZCTA5', 'StateNb', 'CountyNb', 'Population']].copy()
    zcta_to_county.columns = ['ZCTA5', 'StateNb', 'CountyNb', 'Population']
    zcta_to_county['CountyID'] = zcta_to_county.apply(
        lambda x: str(int(x['StateNb'])) + '{:03d}'.format(int(x['CountyNb'])), axis=1)

    # --- Format county / age-code ID columns ---
    pregnancy_counties['CountyID'] = pregnancy_counties['CountyID'].apply(str)
    county_population['CountyID'] = county_population.apply(
        lambda x: str(x['STATE']) + '{:03d}'.format(x['COUNTY']), axis=1)

    # --- Age group lists ---
    pregnancy_age_groups = filter_df(age_groups_df,
        [['Pregnancy', '==', [1]]])['AgeGroup'].tolist()
    detail_age_groups = age_groups_df['AgeGroup'].tolist()
    age_groups_mcc    = mcc_df['AgeGroupMCC'].tolist()
    
    # --- ZCTA population per age group ---
    zcta_age_pop['ZCTA5'] = zcta_age_pop['Geographic Area Name'].apply(
        lambda x: str(x).replace('ZCTA5 ', ''))

    # --- Historical county code renames ---
    # https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes/2010.html
    renamed_county_codes = {
        '46113': '46102',   # South Dakota: Shannon county → Oglala county
        '2270':  '2158',    # Alaska: Wade Hampton → Kusilvak Census Area
        '2066': '2261',     # Alaska: Copper River Census Area → Valdez-Cordova Census Area
        '2261': '2063',     # Alaska: Valdez-Cordova Census Area → Chugach Census Area
        '2063': '2261',
        '2066': '2261',
    }
    county_rename_CT['old_county'] = county_rename_CT.apply(
        lambda x: str(int(x['State'])) + '{:03d}'.format(int(x['old_county_fips'])), axis=1)
    county_rename_CT['new_county'] = county_rename_CT.apply(
        lambda x: str(int(x['State'])) + '{:03d}'.format(int(x['new_county_fips'])), axis=1)
    connecticut_rename = dict(zip(county_rename_CT['old_county'], county_rename_CT['new_county']))
    renamed_county_codes.update(connecticut_rename)

    return {
        'chronic_conditions_list':   chronic_conditions_list,
        'high_risk_conditions_list': high_risk_conditions_list,
        'mcc_df':                    mcc_df,
        'age_groups_df':             age_groups_df,
        'age_groups_mcc':            age_groups_mcc,
        'detail_age_groups':         detail_age_groups,
        'pregnancy_age_groups':      pregnancy_age_groups,
        'number_mcc_list':           [1, 2, 3],
        'places_data':               places_data,
        'places_obesity':            places_obesity,
        'children_obesity_dict':     children_obesity_dict,
        'obese_with_condition':      obese_with_condition,
        'w_avg_obesity':             w_avg_obesity,
        'zcta_to_county':            zcta_to_county,
        'pregnancy_counties':        pregnancy_counties,
        'county_population':         county_population,
        'renamed_county_codes':      renamed_county_codes,
        'zcta_age_pop':              zcta_age_pop,
    }


# ── Analysis functions ─────────────────────────────────────────────────────────

def run_zcta(paths, p):
    """Compute and save high-risk proportions per age group for every US ZCTA."""
    list_zcta              = pd.unique(p['places_data']['ZCTA5'])
    pregnancy_counties_list = pd.unique(p['pregnancy_counties']['CountyID'])
    missing_counties       = {}
    high_risk_zcta         = {}

    for zcta in list_zcta:
        print(f'ZCTA: {zcta})')
        zcta_row = filter_df(p['places_data'], [['ZCTA5', '==', [zcta]]])

        risk_score, prevalence_dict, obesity, pregnancy_dict, female_prop = \
            get_zcta_high_risk_inputs(
                zcta, 
                zcta_row, 
                p['chronic_conditions_list'],
                p['places_obesity'], 
                p['zcta_to_county'],
                p['pregnancy_counties'],
                p['pregnancy_age_groups'],
                pregnancy_counties_list, 
                missing_counties,
                p['renamed_county_codes'], 
                p['county_population'])

        mcc_ztca = adjust_mcc(
            p['mcc_df'], 
            risk_score, 
            p['number_mcc_list'])

        high_risk_final = calculate_high_risk_prop(
            mcc_ztca, 
            p['number_mcc_list'], 
            p['age_groups_mcc'],
            prevalence_dict, 
            p['high_risk_conditions_list'],
            p['chronic_conditions_list'], 
            obesity,
            p['obese_with_condition'], 
            p['w_avg_obesity'],
            p['children_obesity_dict'], 
            p['age_groups_df'],
            p['detail_age_groups'], 
            pregnancy_dict, 
            female_prop,
            p['pregnancy_age_groups'])

        high_risk_zcta[zcta] = copy.deepcopy(high_risk_final)

    result_df = pd.DataFrame(high_risk_zcta).T.reset_index()
    result_df.rename(columns={'index': 'ZCTA5'}, inplace=True)
    result_df.to_csv(paths['out_us_zcta'], index=False)
    print(f"US ZCTA results saved to: {paths['out_us_zcta']}")


def run_zcta_counts(paths, p):
    """Compute high-risk and low-risk population counts per age group for every US ZCTA.

    Reads the high-risk proportions produced by run_zcta and multiplies them by
    the ZCTA population in each age group to obtain counts. The population data
    in p['zcta_age_pop'] uses slightly different age-group boundaries, so two
    adjustments are made before merging:
      - '0_4' is split into '0_0.5' (10 %) and '0.5_4' (90 %), matching the
        infant / toddler split used elsewhere in the pipeline.
      - '75_79', '80_84', and '85+' are collapsed into a single '75+' group.

    Args:
        paths: dict
            Must contain:
              - 'out_us_zcta'       : path to the CSV written by run_zcta (input).
              - 'out_zcta_counts'   : path for the output counts CSV.
        p: dict
            Shared parameters from load_shared_params. Uses 'zcta_age_pop'
            (DataFrame with columns ZCTA5, 0_4, 5_9, …, 75_79, 80_84, 85+).
    """
    # ── Load high-risk proportions ─────────────────────────────────────────────
    high_risk_df = pd.read_csv(paths['out_us_zcta'])
    high_risk_df['ZCTA5'] = high_risk_df['ZCTA5'].apply(str)
    age_groups = [c for c in high_risk_df.columns if c != 'ZCTA5']

    # ── Align zcta_age_pop to the same age groups ──────────────────────────────
    pop_df = p['zcta_age_pop'].copy()
    pop_df['0_0.5'] = pop_df['0_4'] / 10
    pop_df['0.5_4'] = pop_df['0_4'] * 9 / 10
    pop_df['75+']   = pop_df[['75_79', '80_84', '85+']].sum(axis=1)
    pop_df = pop_df[['ZCTA5'] + age_groups]

    # ── Compute counts ─────────────────────────────────────────────────────────
    prop_vals = high_risk_df.set_index('ZCTA5')[age_groups]
    pop_vals  = pop_df.set_index('ZCTA5')[age_groups]

    common_zctas = prop_vals.index.intersection(pop_vals.index)
    prop_vals = prop_vals.loc[common_zctas]
    pop_vals  = pop_vals.loc[common_zctas]

    high_risk_counts = (prop_vals * pop_vals).round().astype(int)
    low_risk_counts  = ((1 - prop_vals) * pop_vals).round().astype(int)

    # ── Build tidy output ──────────────────────────────────────────────────────
    high_long = (high_risk_counts.reset_index()
                 .melt(id_vars='ZCTA5', var_name='AgeGroup', value_name='HighRisk'))
    low_long  = (low_risk_counts.reset_index()
                 .melt(id_vars='ZCTA5', var_name='AgeGroup', value_name='LowRisk'))

    result_df = pd.merge(high_long, low_long, on=['ZCTA5', 'AgeGroup'])
    result_df['Total'] = result_df['HighRisk'] + result_df['LowRisk']
    result_df = result_df[['ZCTA5', 'AgeGroup', 'HighRisk', 'LowRisk', 'Total']]

    result_df.to_csv(paths['out_zcta_counts'], index=False)
    print(f"ZCTA counts saved to: {paths['out_zcta_counts']}")


def run_zip_to_county(paths, p):
    """Aggregate high-risk population proportions and counts from ZCTA to county level.

    Reads the ZCTA-level high-risk rates produced by run_us_zcta, weights each
    zip code's contribution to its county(ies) by the share of county population it
    covers, and outputs a CSV with high-risk rates and counts (plus low-risk and
    total population rows) for every county.

    Args:
        paths: dict
            Must contain:
              - 'out_us_zcta'   : path to the ZCTA CSV written by run_us_zcta (input).
              - 'county_pop_csv': path to Census county population CSV
                                  (columns: STATE, COUNTY, AGEGRP, TOT_POP).
              - 'county_age_map': path to CSV mapping AGEGRP codes to age-group labels
                                  (columns: AGEGRP, AgeGroup).
              - 'out_county_ZCTA_base' : path for the output county CSV.
        p: dict
            Shared parameters from load_shared_params. Uses 'zcta_to_county'
            (ZCTA5, CountyID, Population) and 'renamed_county_codes'.
    """
    # ── Load inputs ────────────────────────────────────────────────────────────
    high_risk_zcta_df = pd.read_csv(paths['out_us_zcta'])
    high_risk_zcta_df['ZCTA5'] = high_risk_zcta_df['ZCTA5'].apply(str)
    age_groups = [x for x in high_risk_zcta_df.columns if x != 'ZCTA5']

    # ── County population: build wide table (CountyID × age group) ─────────────
    county_population_raw = pd.read_csv(paths['county_pop_csv'])
    county_population_age_map = pd.read_csv(paths['county_age_map'])

    county_population_long = pd.merge(county_population_raw, county_population_age_map,
                                      on='AGEGRP', how='left')
    county_population_long['AgeGroup'] = county_population_long['AgeGroup'].apply(str)
    county_population_long = filter_df(county_population_long,
                                       [['AgeGroup', '!=', ['Total']]])

    county_population_long['CountyID'] = county_population_long.apply(
        lambda x: str(x['STATE']) + '{:03d}'.format(x['COUNTY']), axis=1)
    county_population_long.rename(columns={'TOT_POP': 'Population'}, inplace=True)
    county_population_long = county_population_long[['CountyID', 'AgeGroup', 'Population']]

    county_population = pd.pivot_table(county_population_long,
                                       values='Population',
                                       index='CountyID',
                                       columns='AgeGroup').reset_index()

    # Split 0–4 into infants / toddlers; collapse 75+ groups to match ZCTA output
    county_population['0_0.5'] = county_population['0_4'] / 10
    county_population['0.5_4'] = county_population['0_4'] * 9 / 10
    county_population['75+']   = county_population[['75_79', '80_84', '85+']].sum(axis=1)
    for col in ['0_4', '75_79', '80_84', '85+']:
        if col in county_population.columns:
            del county_population[col]

    new_pop_col_names = ['pop_' + x for x in age_groups]
    county_population.rename(columns=dict(zip(age_groups, new_pop_col_names)), inplace=True)

    # ── ZCTA-to-county weights ─────────────────────────────────────────────────
    zcta_to_county = p['zcta_to_county'][['ZCTA5', 'CountyID', 'Population']].copy()
    zcta_to_county['ZCTA5'] = zcta_to_county['ZCTA5'].apply(str)
    zcta_to_county.replace({'CountyID': p['renamed_county_codes']}, inplace=True)

    county_pop_from_zips = (zcta_to_county.groupby('CountyID')['Population']
                            .sum().reset_index()
                            .rename(columns={'Population': 'CountyPopulation'}))
    zcta_to_county = pd.merge(zcta_to_county, county_pop_from_zips, on='CountyID', how='left')
    zcta_to_county['zip_weight'] = (zcta_to_county['Population'] /
                                    zcta_to_county['CountyPopulation'])

    # ── Map high-risk rates to counties ───────────────────────────────────────
    high_risk_mapping = pd.merge(high_risk_zcta_df,
                                 zcta_to_county[['ZCTA5', 'CountyID', 'zip_weight']],
                                 on='ZCTA5', how='left')

    # Scale weights by coverage so counties with missing ZCTAs sum to 1
    county_coverage = (high_risk_mapping[['CountyID', 'zip_weight']]
                       .groupby('CountyID').sum().reset_index()
                       .rename(columns={'zip_weight': 'Coverage'}))
    high_risk_mapping = pd.merge(high_risk_mapping, county_coverage, on='CountyID', how='left')
    high_risk_mapping['zip_weight'] = (high_risk_mapping['zip_weight'] /
                                       high_risk_mapping['Coverage'])

    for ag in age_groups:
        high_risk_mapping[ag] = high_risk_mapping[ag] * high_risk_mapping['zip_weight']

    high_risk_rate_counties = (high_risk_mapping.groupby('CountyID')[age_groups]
                               .sum().reset_index())
    high_risk_rate_counties['Rate_Count'] = 'Rate'
    high_risk_rate_counties['RiskGroup']  = 'High'

    low_risk_rate_counties = high_risk_rate_counties.copy()
    low_risk_rate_counties['RiskGroup'] = 'Low'
    for ag in age_groups:
        low_risk_rate_counties[ag] = 1 - high_risk_rate_counties[ag]

    risk_rate_counties = pd.concat([high_risk_rate_counties, low_risk_rate_counties])

    # ── Counts = rate × county population per age group ───────────────────────
    risk_count_counties = risk_rate_counties.copy()
    risk_count_counties['Rate_Count'] = 'Count'
    new_rate_col_names = ['rate_' + x for x in age_groups]
    risk_count_counties.rename(columns=dict(zip(age_groups, new_rate_col_names)), inplace=True)
    risk_count_counties = pd.merge(risk_count_counties, county_population,
                                   on='CountyID', how='left')
    for ag in age_groups:
        risk_count_counties[ag] = (risk_count_counties['rate_' + ag] *
                                   risk_count_counties['pop_' + ag])
    risk_count_counties = risk_count_counties[['CountyID', 'Rate_Count', 'RiskGroup'] + age_groups]

    # ── Total population rows ──────────────────────────────────────────────────
    county_pop_df = county_population.copy()
    county_pop_df.rename(columns=dict(zip(new_pop_col_names, age_groups)), inplace=True)
    county_pop_df['Rate_Count'] = 'Count'
    county_pop_df['RiskGroup']  = 'Total'

    # ── Combine and export ─────────────────────────────────────────────────────
    county_risk_df = pd.concat([risk_rate_counties, risk_count_counties, county_pop_df])
    county_risk_df = county_risk_df[['CountyID', 'Rate_Count', 'RiskGroup'] + age_groups]

    # Remove city of Bedford, Virginia (51515); it merged into Bedford County
    county_risk_df = filter_df(county_risk_df, [['CountyID', '!=', ['51515']]])

    county_risk_df.to_csv(paths['out_county_ZCTA_base'], index=False)
    print(f"County results saved to: {paths['out_county_ZCTA_base']}")


def run_county_to_national(paths, p):
    """Aggregate high-risk population proportions and counts from county to national level.

    Reads the county-level high-risk rates produced by run_zip_to_county, weights each
    county's contribution to the national total by its population per age group, and
    outputs a CSV with high-risk rates and counts (plus low-risk and total population
    rows) for the US as a whole.

    Args:
        paths: dict
            Must contain:
              - 'out_county_ZCTA_base': path to the county CSV written by run_zip_to_county
                                        (input).
              - 'county_pop_csv': path to Census county population CSV
                                  (columns: STATE, COUNTY, AGEGRP, TOT_POP).
              - 'county_age_map': path to CSV mapping AGEGRP codes to age-group labels
                                  (columns: AGEGRP, AgeGroup).
              - 'out_national_ZCTA_base': path for the output national CSV.
        p: dict
            Shared parameters from load_shared_params. Uses 'renamed_county_codes'.
    """
    # ── Load inputs ────────────────────────────────────────────────────────────
    high_risk_county_df = pd.read_csv(paths['out_county'])
    # high_risk_rate_df = filter_df(high_risk_county_df,
    #                               [['Rate_Count', '==', ['Rate']],
    #                                ['RiskGroup',  '==', ['High']]])
    high_risk_rate_df = high_risk_county_df.copy()
    age_groups = [x for x in high_risk_rate_df.columns
                  if x not in ('CountyID', 'Rate_Count', 'RiskGroup')]

    # ── County population: build wide table (CountyID × age group) ─────────────
    county_population_raw     = pd.read_csv(paths['county_pop_csv'])
    county_population_age_map = pd.read_csv(paths['county_age_map'])

    county_population_long = pd.merge(county_population_raw, county_population_age_map,
                                      on='AGEGRP', how='left')
    county_population_long['AgeGroup'] = county_population_long['AgeGroup'].apply(str)
    county_population_long = filter_df(county_population_long,
                                       [['AgeGroup', '!=', ['Total']]])

    county_population_long['CountyID'] = county_population_long.apply(
        lambda x: str(x['STATE']) + '{:03d}'.format(x['COUNTY']), axis=1)
    county_population_long.rename(columns={'TOT_POP': 'Population'}, inplace=True)
    county_population_long = county_population_long[['CountyID', 'AgeGroup', 'Population']]

    county_population = pd.pivot_table(county_population_long,
                                       values='Population',
                                       index='CountyID',
                                       columns='AgeGroup').reset_index()

    # Split 0–4 into infants / toddlers; collapse 75+ groups to match county output
    county_population['0_0.5'] = county_population['0_4'] / 10
    county_population['0.5_4'] = county_population['0_4'] * 9 / 10
    county_population['75+']   = county_population[['75_79', '80_84', '85+']].sum(axis=1)
    for col in ['0_4', '75_79', '80_84', '85+']:
        if col in county_population.columns:
            del county_population[col]

    new_pop_col_names = ['pop_' + x for x in age_groups]
    county_population.rename(columns=dict(zip(age_groups, new_pop_col_names)), inplace=True)
    county_population.replace({'CountyID': p['renamed_county_codes']}, inplace=True)

    # ── Population-weighted national rates ────────────────────────────────────
    county_population['CountyID_int'] = county_population['CountyID'].astype(int)
    high_risk_mapping = pd.merge(high_risk_rate_df, county_population,
                                 left_on='CountyID', right_on='CountyID_int', how='left')

    national_pop = {ag: high_risk_mapping['pop_' + ag].sum() for ag in age_groups}
    national_rate = {
        ag: (high_risk_mapping[ag] * high_risk_mapping['pop_' + ag]).sum() / national_pop[ag]
        for ag in age_groups
    }

    high_risk_rate_national = pd.DataFrame([national_rate])
    high_risk_rate_national['Rate_Count'] = 'Rate'
    high_risk_rate_national['RiskGroup']  = 'High'

    low_risk_rate_national = high_risk_rate_national.copy()
    low_risk_rate_national['RiskGroup'] = 'Low'
    for ag in age_groups:
        low_risk_rate_national[ag] = 1 - high_risk_rate_national[ag]

    risk_rate_national = pd.concat([high_risk_rate_national, low_risk_rate_national])

    # ── Counts = rate × national population per age group ─────────────────────
    risk_count_national = risk_rate_national.copy()
    risk_count_national['Rate_Count'] = 'Count'
    for ag in age_groups:
        risk_count_national[ag] = risk_count_national[ag] * national_pop[ag]

    # ── Total population row ───────────────────────────────────────────────────
    national_pop_df = pd.DataFrame([national_pop])
    national_pop_df['Rate_Count'] = 'Count'
    national_pop_df['RiskGroup']  = 'Total'

    # ── Combine and export ─────────────────────────────────────────────────────
    national_risk_df = pd.concat([risk_rate_national, risk_count_national, national_pop_df])
    national_risk_df = national_risk_df[['Rate_Count', 'RiskGroup'] + age_groups]

    national_risk_df.to_csv(paths['out_national_county_base'], index=False)
    print(f"National results saved to: {paths['out_national_county_base']}")


def run_county_to_state(paths, p):
    """Aggregate high-risk population proportions and counts from county to state level.

    Reads the county-level high-risk rates produced by run_us_county, weights each
    county's contribution to its state by its population per age group, and outputs
    a CSV with high-risk rates and counts (plus low-risk and total population rows)
    for each US state.

    Args:
        paths: dict
            Must contain:
              - 'out_county': path to the county CSV written by run_us_county (input).
              - 'county_pop_csv': path to Census county population CSV
                                  (columns: STATE, COUNTY, AGEGRP, TOT_POP).
              - 'county_age_map': path to CSV mapping AGEGRP codes to age-group labels
                                  (columns: AGEGRP, AgeGroup).
              - 'out_state_county_base': path for the output state CSV.
        p: dict
            Shared parameters from load_shared_params. Uses 'renamed_county_codes'.
    """
    # ── Load inputs ────────────────────────────────────────────────────────────
    high_risk_county_df = pd.read_csv(paths['out_county'])
    high_risk_rate_df = high_risk_county_df.copy()
    age_groups = [x for x in high_risk_rate_df.columns
                  if x not in ('CountyID', 'Rate_Count', 'RiskGroup')]
    
    # State lookup table from us library (FIPS code, name, abbreviation)
    states_and_territories = us.states.STATES_AND_TERRITORIES + [us.states.DC]

    state_names = [s.name for s in states_and_territories]
    state_fips = [s.fips for s in states_and_territories]
    state_abbr = [s.abbr for s in states_and_territories]
    state_lookup_df = pd.DataFrame({
        'StateName': state_names,
        'StateID': state_fips,
        'StateAbbreviation': state_abbr,
    })

    # ── County population: build wide table (CountyID × age group) ─────────────
    county_population_raw     = pd.read_csv(paths['county_pop_csv'])
    county_population_age_map = pd.read_csv(paths['county_age_map'])

    county_population_long = pd.merge(county_population_raw, county_population_age_map,
                                      on='AGEGRP', how='left')
    county_population_long['AgeGroup'] = county_population_long['AgeGroup'].apply(str)
    county_population_long = filter_df(county_population_long,
                                       [['AgeGroup', '!=', ['Total']]])

    county_population_long['CountyID'] = county_population_long.apply(
        lambda x: str(x['STATE']) + '{:03d}'.format(x['COUNTY']), axis=1)
    county_population_long.rename(columns={'TOT_POP': 'Population'}, inplace=True)
    county_population_long = county_population_long[['CountyID', 'AgeGroup', 'Population']]

    county_population = pd.pivot_table(county_population_long,
                                       values='Population',
                                       index='CountyID',
                                       columns='AgeGroup').reset_index()

    # Split 0–4 into infants / toddlers; collapse 75+ groups to match county output
    county_population['0_0.5'] = county_population['0_4'] / 10
    county_population['0.5_4'] = county_population['0_4'] * 9 / 10
    county_population['75+']   = county_population[['75_79', '80_84', '85+']].sum(axis=1)
    for col in ['0_4', '75_79', '80_84', '85+']:
        if col in county_population.columns:
            del county_population[col]

    new_pop_col_names = ['pop_' + x for x in age_groups]
    county_population.rename(columns=dict(zip(age_groups, new_pop_col_names)), inplace=True)
    county_population.replace({'CountyID': p['renamed_county_codes']}, inplace=True)

    # StateID = CountyID with the 3-digit county suffix removed
    county_population['StateID'] = county_population['CountyID'].apply(lambda x: str(x)[:-3])

    # ── Population-weighted state rates ────────────────────────────────────────
    county_population['CountyID_int'] = county_population['CountyID'].astype(int)
    high_risk_mapping = pd.merge(high_risk_rate_df, county_population,
                                 left_on='CountyID', right_on='CountyID_int', how='left')

    state_dfs = []
    for state_id, state_group in high_risk_mapping.groupby('StateID'):
        state_pop = {ag: state_group['pop_' + ag].sum() for ag in age_groups}
        state_rate = {
            ag: ((state_group[ag] * state_group['pop_' + ag]).sum() / state_pop[ag]
                 if state_pop[ag] > 0 else np.nan)
            for ag in age_groups
        }

        high_risk_rate_state = pd.DataFrame([state_rate])
        high_risk_rate_state['StateID']    = state_id
        high_risk_rate_state['Rate_Count'] = 'Rate'
        high_risk_rate_state['RiskGroup']  = 'High'

        low_risk_rate_state = high_risk_rate_state.copy()
        low_risk_rate_state['RiskGroup'] = 'Low'
        for ag in age_groups:
            low_risk_rate_state[ag] = 1 - high_risk_rate_state[ag]

        risk_rate_state = pd.concat([high_risk_rate_state, low_risk_rate_state])

        # Counts = rate × state population per age group
        risk_count_state = risk_rate_state.copy()
        risk_count_state['Rate_Count'] = 'Count'
        for ag in age_groups:
            risk_count_state[ag] = risk_count_state[ag] * state_pop[ag]

        # Total population row
        state_pop_df = pd.DataFrame([state_pop])
        state_pop_df['StateID']    = state_id
        state_pop_df['Rate_Count'] = 'Count'
        state_pop_df['RiskGroup']  = 'Total'

        state_dfs.append(pd.concat([risk_rate_state, risk_count_state, state_pop_df]))

    state_risk_df = pd.concat(state_dfs)

    # Normalize state_lookup_df StateID: us library uses zero-padded FIPS (e.g. '01'),
    # but StateID here is derived without padding (e.g. '1')
    state_lookup_df['StateID'] = state_lookup_df['StateID'].apply(lambda x: str(int(x)))
    state_risk_df = pd.merge(state_risk_df, state_lookup_df, on='StateID', how='left')
    state_risk_df = state_risk_df[['StateID', 'StateName', 'StateAbbreviation',
                                   'Rate_Count', 'RiskGroup'] + age_groups]

    state_risk_df.to_csv(paths['out_state_county_base'], index=False)
    print(f"State results saved to: {paths['out_state_county_base']}")


def run_county(paths, p):
    """Compute and save high-risk proportions per age group for every US county.

    Uses county-level CDC PLACES data to obtain local chronic condition and
    obesity prevalences, then calls the same adjust_mcc / calculate_high_risk_prop
    pipeline used for ZCTAs. Pregnancy and female-proportion inputs are looked up
    directly from the county_population and pregnancy_counties tables already in p.

    Args:
        paths: dict
            Must contain:
              - 'places_county' : path to the county-level CDC PLACES CSV
                                  (columns: CountyFIPS, <cond>_CrudePrev,
                                  OBESITY_CrudePrev, TotalPopulation).
              - 'out_county'    : path for the output CSV.
        p: dict
            Shared parameters from load_shared_params.
    """
    # ── Load county-level PLACES data ─────────────────────────────────────────
    places_county_raw = pd.read_csv(paths['places_county'])
    county_population_age_map = pd.read_csv(paths['county_age_map'])

    # Condition prevalences — same column naming convention as the ZCTA file
    org_cols   = ['CountyFIPS'] + [x + '_CrudePrev' for x in p['chronic_conditions_list']]
    clean_cols = ['CountyID']   + p['chronic_conditions_list']
    places_county_raw['TotalPopulation'] = places_county_raw['TotalPopulation'].str.replace(',', '').astype(int)
    places_county = places_county_raw[org_cols].copy()
    places_county.columns = clean_cols
    # Convert standard 5-digit FIPS (e.g. '01001') to internal format (e.g. '1001')
    places_county['CountyID'] = places_county['CountyID'].apply(lambda x: str(int(x)))
    
    places_county_obesity = places_county_raw[
        ['CountyFIPS', 'OBESITY_CrudePrev', 'TotalPopulation']].copy()
    places_county_obesity.columns = ['CountyID', 'OBESITY', 'Population']
    places_county_obesity['CountyID'] = places_county_obesity['CountyID'].apply(
        lambda x: str(int(x)))

    # ── Risk score: prevalence sum / national county-population-weighted average
    places_county['PrevalenceSum'] = places_county[p['chronic_conditions_list']].sum(axis=1)
    county_pop_for_avg = places_county_raw[['CountyFIPS', 'TotalPopulation']].copy()
    county_pop_for_avg.columns = ['CountyID', 'TotalPopulation']
    county_pop_for_avg['CountyID'] = county_pop_for_avg['CountyID'].apply(lambda x: str(int(x)))
    weighted_avg_df  = pd.merge(county_pop_for_avg,
                                places_county[['CountyID', 'PrevalenceSum']],
                                on='CountyID', how='left')
    weighted_avg_sum = df_calc_weighted_avg(weighted_avg_df, 'PrevalenceSum', 'TotalPopulation')
    places_county['risk_score'] = places_county['PrevalenceSum'] / weighted_avg_sum

    # National county-population-weighted average obesity (used for child obesity scaling)
    w_avg_obesity_county = df_calc_weighted_avg(places_county_obesity, 'OBESITY', 'Population')

    # ── Precompute female proportions per county / age group ──────────────────
    # county_pop = pd.merge(
    #     p['county_population'], county_population_age_map,
    #     on='AGEGRP', how='left')
    county_pop = p['county_population'].copy()
    county_pop_female = filter_df(county_pop,
                                  [['AgeGroup', '==', p['pregnancy_age_groups']]]).copy()
    county_pop_female['FemaleProp'] = county_pop_female.apply(
        lambda x: x['TOT_FEMALE'] / x['TOT_POP'] if x['TOT_POP'] > 0 else 0., axis=1)

    pregnancy_counties_list = pd.unique(p['pregnancy_counties']['CountyID'])
    missing_counties = {}
    high_risk_county = {}

    # ── Main loop ─────────────────────────────────────────────────────────────
    for county in pd.unique(places_county['CountyID']):
        county_row       = filter_df(places_county, [['CountyID', '==', [county]]])
        prevalence_dict  = {x: county_row[x].values[0] for x in p['chronic_conditions_list']}
        risk_score       = county_row['risk_score'].values[0]

        obesity_row = filter_df(places_county_obesity, [['CountyID', '==', [county]]])
        if len(obesity_row) == 0:
            missing_counties[county] = 'no obesity data'
            continue
        obesity = obesity_row['OBESITY'].values[0]

        # Pregnancy: direct county lookup; fall back to renamed code if needed
        county_for_pregnancy = county
        if county not in pregnancy_counties_list:
            renamed = p['renamed_county_codes'].get(county)
            if renamed and renamed in pregnancy_counties_list:
                county_for_pregnancy = renamed
            else:
                missing_counties[county] = 'no pregnancy data'
                continue

        pregnancy_rows = filter_df(p['pregnancy_counties'],
                                   [['CountyID',   '==', [county_for_pregnancy]],
                                    ['Count_Rate', '==', ['Rate']]])
        pregnancy_dict = {ag: pregnancy_rows[ag].values[0] for ag in p['pregnancy_age_groups']}

        # Female proportion: direct county lookup
        county_female_rows = filter_df(county_pop_female, [['CountyID', '==', [county]]])
        if len(county_female_rows) < len(p['pregnancy_age_groups']):
            missing_counties[county] = 'incomplete population data'
            continue
        female_prop = {
            ag: filter_df(county_female_rows,
                          [['AgeGroup', '==', [ag]]])['FemaleProp'].values[0]
            for ag in p['pregnancy_age_groups']
        }

        mcc_county = adjust_mcc(p['mcc_df'], risk_score, p['number_mcc_list'])

        high_risk_final = calculate_high_risk_prop(
            mcc_county,
            p['number_mcc_list'],
            p['age_groups_mcc'],
            prevalence_dict,
            p['high_risk_conditions_list'],
            p['chronic_conditions_list'],
            obesity,
            p['obese_with_condition'],
            w_avg_obesity_county,
            p['children_obesity_dict'],
            p['age_groups_df'],
            p['detail_age_groups'],
            pregnancy_dict,
            female_prop,
            p['pregnancy_age_groups'])

        high_risk_county[county] = copy.deepcopy(high_risk_final)

    result_df = pd.DataFrame(high_risk_county).T.reset_index()
    result_df.rename(columns={'index': 'CountyID'}, inplace=True)
    result_df.to_csv(paths['out_county'], index=False)
    print(f"County direct results saved to: {paths['out_county']}")


def run_us_total(paths, p):
    """Compute and save population-weighted high-risk proportions for the US."""
    places_avg_calc = pd.merge(p['places_data'], p['places_obesity'],
                               on='ZCTA5', how='outer')
    prevalence_dict_US = {
        x: df_calc_weighted_avg(places_avg_calc, x, 'Population')
        for x in p['chronic_conditions_list']
    }
    obesity_US = df_calc_weighted_avg(places_avg_calc, 'OBESITY', 'Population')

    # National-level female proportion and pregnancy rates per age group
    female_prop_US    = {}
    pregnancy_dict_US = {}
    for ag in p['pregnancy_age_groups']:
        pop_ag          = filter_df(p['county_population'], [['AgeGroup', '==', [ag]]])
        pregnancy_rates = filter_df(p['pregnancy_counties'],
                                    [['Count_Rate', '==', ['Rate']]])
        pregnancy_ag = pd.merge(pop_ag[['CountyID', 'TOT_FEMALE']],
                                pregnancy_rates[['CountyID', ag]],
                                on='CountyID', how='left')
        pregnancy_dict_US[ag] = df_calc_weighted_avg(pregnancy_ag, ag, 'TOT_FEMALE')
        female_prop_US[ag]    = (pop_ag['TOT_FEMALE'].sum() /
                                 (pop_ag['TOT_FEMALE'].sum() + pop_ag['TOT_MALE'].sum()))

    high_risk_US = calculate_high_risk_prop(
        p['mcc_df'], p['number_mcc_list'], p['age_groups_mcc'],
        prevalence_dict_US, p['high_risk_conditions_list'],
        p['chronic_conditions_list'], obesity_US,
        p['obese_with_condition'], p['w_avg_obesity'],
        p['children_obesity_dict'], p['age_groups_df'],
        p['detail_age_groups'], pregnancy_dict_US,
        female_prop_US, p['pregnancy_age_groups'])

    pd.DataFrame(high_risk_US, index=['US']).to_csv(
        paths['out_us_total'], index=True)
    print(f"US total results saved to: {paths['out_us_total']}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    np.set_printoptions(linewidth=125)
    pd.set_option('display.width', 125)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    BASE_DIR   = os.getcwd()
    PARAMS_DIR = os.path.join(BASE_DIR, 'Parameters')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'Outputs')

    paths = {
        'params':       os.path.join(PARAMS_DIR, 'High Risk - Parameters.xlsx'),
        'places':       os.path.join(PARAMS_DIR, 'CDC PLACES', 'PLACES_ZCTA_2025.csv'),
        'pregnancy':    os.path.join(PARAMS_DIR, 'Pregnancy',
                                     'Pregnancy rates and count per county and age group.csv'),
        'zcta_to_county': os.path.join(PARAMS_DIR, 'ZCTA to County', 'ZCTA to County.csv'),
        'county_rename_CT': os.path.join(PARAMS_DIR, 'ZCTA to County', 'County renames - Connecticut.csv'),
        'county_pop_csv': os.path.join(PARAMS_DIR, 'Population data', 'Counties',
                                       'county_population_2023_only.csv'),
        'county_age_map': os.path.join(PARAMS_DIR, 'Population data', 'Counties',
                                       'county_population_age_group_map.csv'),
        'zcta_age_pop': os.path.join(PARAMS_DIR, 'Population data', 'ZCTA',
                                       'ZCTA Population per age group.csv'),
        'places_county':  os.path.join(PARAMS_DIR, 'CDC PLACES', 'PLACES_County_2025.csv'),
        'out_us_zcta':    os.path.join(OUTPUT_DIR,
                                     'High risk population rates per age group per zip code.csv'),
        'out_zcta_counts': os.path.join(OUTPUT_DIR,
                                     'High risk counts per age group per zip code.csv'),
        'out_us_total':   os.path.join(OUTPUT_DIR,
                                     'High risk population per age group in the US.csv'),
        'out_county_ZCTA_base': os.path.join(OUTPUT_DIR,
                                       'High risk population per age group per county - aggregate from ZCTA.csv'),
        'out_county':     os.path.join(OUTPUT_DIR,
                                     'High risk population per age group per county.csv'),
        'out_national_county_base': os.path.join(OUTPUT_DIR,
                                     'High risk population per age group in the US - aggregate from county.csv'),
        'out_state_county_base':    os.path.join(OUTPUT_DIR,
                                     'High risk population per age group per state - aggregate from county.csv'),
    }

    p = load_shared_params(paths)
    run_zcta(paths, p)
    run_zcta_counts(paths, p)
    # run_zip_to_county(paths, p)
    run_county(paths, p)
    run_county_to_state(paths, p)
    run_county_to_national(paths, p)
    run_us_total(paths, p)

if __name__ == '__main__':
    main()

