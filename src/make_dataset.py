import pandas as pd
import numpy as np
from pathlib import Path

import logging
logging.basicConfig()
log = logging.getLogger(__name__)

from config import MakeDataSetConfig

def _impute_macrob_score_for_imperfect_matches(df: pd.DataFrame):
    dPerfect = df[df['perfectMatch'] == True]
    dNonPerfect = df[df['perfectMatch'] == False]
    log.debug(f"dPerfect shape {dPerfect.shape}")
    log.debug(f"dNonPerfect shape {dNonPerfect.shape}")

    df['Q1'] = dNonPerfect['RobMacQ1']
    df['Q2'] = dNonPerfect['RobMacQ2']
    df['Q3'] = dNonPerfect['RobMacQ3']
    df['QUAL'] = dNonPerfect['RobMacQualScore']

    # now fill the blanks from the original ratings.
    # it does not matter getting P1 or P2 score as they are perfect match
    df['Q1'] = df['Q1'].fillna(dPerfect['q1p1T'])
    df['Q2'] = df['Q2'].fillna(dPerfect['q2p1T'])
    df['Q3'] = df['Q3'].fillna(dPerfect['q3p1T'])
    df['QUAL'] = df['QUAL'].fillna(dPerfect['P1QualScore'])

    #calculate sum of qual scores and compare with the previous manually summed values to determine if they check out.
    df['summedQs'] = df['Q1']+df['Q2']+df['Q3']
    comparison_QUALScore_columns = np.where(df['summedQs'] == df['QUAL'], True, False)
    df["isQUALequal"] = comparison_QUALScore_columns
    log.info(f'Number of manually summed columns that equal auto-summed QuAL scores:\n{df["isQUALequal"].value_counts()}')
    log.info('Unequal will be replaced by auto-summed QuAL scores.')
    # for those that don't replace with the calculated qual scores
    df.loc[(df.isQUALequal == False),'QUAL'] = df['summedQs']
    # get rid of helper columns
    df.drop(['summedQs', 'isQUALequal'], axis=1, inplace=True)

    return df 

def _add_demog_cols(masterdb: pd.DataFrame, mac_path: Path, sas_path: Path):

    idx = ['Survey N', 'Question N']

    df = masterdb.set_index(idx).sort_index()
    log.info(f'Loading Mac demoographics from {mac_path}')
    mac = pd.read_excel(mac_path).set_index(idx).sort_index()
    log.info(f'Loading Sask demographics from {sas_path}')
    sas = pd.read_excel(sas_path).set_index(idx).sort_index()

    mac = mac[['GenderRes','GenderFac','Type','Unnamed: 6','EPA','PGY']]
    mac['EPA'] = mac['EPA'].str.split(':').apply(lambda x: x[0])
    mac.rename({'Unnamed: 6': 'ObserverType'}, inplace=True, axis=1)

    sas = sas[['Resident Name', 'Observer Name', 'Observer Type', 'EM/PEM vs off-service', 'EPA']]
    sas.rename({'Resident Name': 'GenderRes', 'Observer Name': 'GenderFac', 'Observer Type': 'ObserverType', 'EM/PEM vs off-service': 'Type'}, inplace=True, axis=1)

    mac_sas = pd.concat([mac, sas])

    mac_sas['GenderFac'] = mac_sas['GenderFac'].replace({'M': 'Male', 'F': 'Female'})
    mac_sas.loc[~np.isin(mac_sas['GenderFac'], ['Male','Female']), 'GenderFac'] = pd.NA
    mac_sas['GenderRes'] = mac_sas['GenderRes'].replace({'M': 'Male', 'F': 'Female'})
    mac_sas['Type'] = mac_sas['Type'].replace({'Off Service Faculty': 'Off Service', 'Off-service': 'Off Service', 'EM Regina': 'EM', 'EM (Regina)': 'EM', 'Emergency (BC)': 'EM'})
    # mac_sas['Type'].fillna('Unknown', inplace=True)
    mac_sas['ObserverType'] = mac_sas['ObserverType'].str.lower()
    mac_sas['ObserverType'] = mac_sas['ObserverType'].replace({'facutly': 'faculty'})
    # mac_sas['PGY'].fillna('Unknown', inplace=True)

    df = df.join(mac_sas).reset_index()

    return df

def main(cfg : MakeDataSetConfig):

    log.info("Generating final dataset...")
    log.debug(f"Parameters:\n{cfg.model_dump()}")
    
    # load the raw data
    log.info(f"Loading masterdb from {cfg.masterdb_path}")
    masterdb = pd.read_excel(cfg.masterdb_path,  index_col=0)
    log.debug(f'Shape is {masterdb.shape}')

    # impute the corrected QuAL scores from above if they don't match
    log.info('Imputing corrected scores where necessary...')
    masterdb = _impute_macrob_score_for_imperfect_matches(masterdb)

    # add in demographic columns
    log.info('Merging in demographic columns from raw data...')
    masterdb = _add_demog_cols(masterdb, cfg.mac_path, cfg.sas_path)

    # create modified targets
    log.info('Modifying targets...')
    if cfg.q1_condense:
        masterdb[cfg.q1_condense_col_name] = masterdb['Q1'].replace({
            0: 0,
            1: 0,
            2: 0,
            3: 1
        })
    if cfg.q2_invert:
        masterdb[cfg.q2_invert_col_name] = masterdb['Q2'] * -1 + 1
    if cfg.q3_invert:
        masterdb[cfg.q3_invert_col_name] = masterdb['Q3'] * -1 + 1
    if cfg.qual_condense:
        # if q1 <= 2, 0
        # if q1 >= 2, q2 = 0, 1
        # if q1 >= 2, q2 = 1, 2
        masterdb[cfg.qual_condense_col_name] = (masterdb['Q1'] >= 3).astype('int') # 0 if <= 2, 1 otherwise
        q1_gt2 = masterdb[cfg.qual_condense_col_name]
        masterdb.loc[q1_gt2, cfg.qual_condense_col_name] = (
            masterdb.loc[q1_gt2, cfg.qual_condense_col_name] +
            masterdb.loc[q1_gt2, 'Q2']
        )
        masterdb[cfg.qual_condense_col_name] = masterdb[cfg.qual_condense_col_name].astype('int')

    log.info(f'Changing nan comments ({masterdb[cfg.text_var].isna().sum()}) to blanks...')
    masterdb[cfg.text_var] = masterdb[cfg.text_var].fillna('blank')

    # output the result
    log.info(f'Saving to {cfg.output_path}')
    masterdb.to_csv(cfg.output_path, index=False)

if __name__ == '__main__':
    cfg = MakeDataSetConfig()
    log.setLevel(cfg.log_level)
    main(cfg)
