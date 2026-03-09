import os
import pandas as pd

# ── Load data ──────────────────────────────────────────────────────────────────
def load_params(paths):
    block_to_zcta = pd.read_csv(
        paths['block_to_zcta'],
        dtype={
            'GEOID_TABBLOCK_20': str,
            'GEOID_ZCTA5_20': str
            })\
        .rename(columns={
            'GEOID_TABBLOCK_20': 'block',
            'GEOID_ZCTA5_20': 'ZCTA5'
            })
    block_population = pd.read_csv(
        paths['block_population'],
        dtype={
            'population': int,
            'block_GEOID': str
            }
        ).rename(columns={
            'block_GEOID': 'block',
            'total_population': 'block_population'
            }
        ).drop(columns=['Unnamed: 0'])
        
    # Ensure that the ZCTA and block codes are zero-padded to the correct length
    block_to_zcta.dropna(subset=['ZCTA5'], inplace=True)
    block_to_zcta['ZCTA5'] = block_to_zcta['ZCTA5'].apply(
        lambda x: str(x).split('.')[0].zfill(5)
    )
    
    block_to_zcta['block'] = block_to_zcta['block'].apply(lambda x: x.zfill(15))
    block_population['block'] = block_population['block'].apply(lambda x: x.zfill(15))
            
    return block_to_zcta, block_population


# ── Main function ──────────────────────────────────────────────────────────────
def create_zcta_to_county(
    block_to_zcta, 
    block_population,
    paths):
    zcta_to_county_block = block_to_zcta.merge(block_population, on='block', how='left')
    
    # Extract county FIPS code from block code (first 5 digits of the block code)
    zcta_to_county_block['CountyNb'] = zcta_to_county_block['block'].apply(lambda x: x[2:5])
    zcta_to_county_block['StateNb'] = zcta_to_county_block['block'].apply(lambda x: x[:2])
    
    zcta_to_county = zcta_to_county_block\
        .groupby(['ZCTA5', 'StateNb', 'CountyNb'], as_index=False)\
        ['block_population'].sum()\
        .rename(columns={'block_population': 'Population'})
    
    zcta_to_county.to_csv(paths['out_us_zcta'], index=False)
    print(f"US ZCTA results saved to: {paths['out_us_zcta']}")
    

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    BASE_DIR   = os.getcwd()
    PARAMS_DIR = os.path.join(BASE_DIR, 'Parameters')
    zcta_to_block_filename = os.path.join(BASE_DIR, 'inputs', 'proportion_population_high_risk_input.csv')

    paths = {
        'block_to_zcta':    os.path.join(PARAMS_DIR, 'ZCTA to County', 'block_to_zcta.csv'),
        'block_population': os.path.join(PARAMS_DIR, 'ZCTA to County', 'population_per_block.csv'),
        'out_us_zcta':      os.path.join(PARAMS_DIR, 'ZCTA to County', 'ZCTA to County.csv'), 
    }
    
    block_to_zcta, block_population = load_params(paths)
    
    create_zcta_to_county(
        block_to_zcta, 
        block_population,
        paths
        )

if __name__ == '__main__':
    main()