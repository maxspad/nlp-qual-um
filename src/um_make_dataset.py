import pandas as pd
from numpy import nan
from config import UMMakeDatasetConfig

import logging
logging.basicConfig()
log = logging.getLogger(__name__)

def main(cfg : UMMakeDatasetConfig): 
    log.setLevel(cfg.log_level)

    log.debug(f'Config:\n{cfg}')
    
    # Read mapping file
    log.info(f'Reading mapping file from {cfg.mapping_file}')
    mapping = pd.read_excel(cfg.mapping_file)
    log.debug(f'mapping head:\n{mapping.head()}')

    dfs = []
    for i, r in mapping.iterrows():
        fn = cfg.data_dir + r['folder'] + '/' + r['file']
        
        log.info(f'Loading {fn}')
        header = (0 if r['has_headers'] else None)
        if r['extension'] == 'xlsx':
            df = pd.read_excel(fn + '.xlsx', header=header)
        else:
            df = pd.read_csv(fn + '.csv', header=header)
        
        simple_cols = ['assessor', 'assessor_role','learner','date_assigned',
                        'date_completed','approx_date',
                        'qual','evidence','suggestion','connection']
        for c in simple_cols:
            if pd.notna(r[c]):
                df[c] = df.iloc[:, int(r[c])]
            else:
                df[c] = nan

        concat_cols = ['text']
        for c in concat_cols:
            if type(r[c]) == int:
                df[c] = df.iloc[:,r[c]]
            else:
                cols_to_concat = [int(x) for x in r[c].split(cfg.text_col_split_char)]
                df_to_concat = df.iloc[:, cols_to_concat].fillna(cfg.text_blank_repl_str).astype('str')
                df[c] = df_to_concat.agg(cfg.text_col_join_char.join, axis=1)
        
        final_cols = simple_cols + concat_cols
        final_df = df.filter(final_cols, axis=1)
        final_df['clerkship'] = r['folder']
        final_df['from_file'] = r['file'] + '.' + r['extension']
        dfs.append(final_df)

    dataset = pd.concat(dfs, axis=0)
    log.info(f'Writing dataset to {cfg.output_file}')
    log.info(f'Dataset size: {dataset.shape}')
    dataset.to_csv(cfg.output_file)

if __name__ == '__main__':
    cfg = UMMakeDatasetConfig()
    main(cfg)

