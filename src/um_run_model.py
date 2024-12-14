from config import UMRunModelConfig
import pandas as pd
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

import logging
logging.basicConfig()
log = logging.getLogger(__name__)

def main(cfg : UMRunModelConfig):
    log.setLevel(cfg.log_level)
    log.debug(cfg)

    # Load data
    log.info(f'Loading data from {cfg.data_file}')
    df = pd.read_csv(cfg.data_file)
    df_orig_shape = df.shape
    log.info(f'Data is shape {df_orig_shape}')

    # If smoke test, use a tiny subset of the data
    if cfg.smoke_test:
        log.warning('SMOKE TEST')
        df = df.sample(n=30, random_state=cfg.random_seed)
    

    # Clean data
    log.info(f'Text column {cfg.text_col}, target column {cfg.target_col}')
    log.info(f'Removing rows with NaN in target/text column')
    # df = df.loc[not_na_rows, :]
    df = df.dropna(subset=[cfg.text_col, cfg.target_col])
    not_na_index = df.index
    df_new_shape = df.shape
    log.info(f'New data is shape {df_new_shape}')
    reduction = df_orig_shape[0] - df_new_shape[0]
    pct_reduction = reduction / df_orig_shape[0] * 100
    log.info(f'This is a reduction of {reduction} ({pct_reduction:.2f}%)')

    # Load model
    log.info(f'Loading tokenizer from {cfg.hf_model_name}')
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_name, model_max_length=cfg.hf_model_max_length)
    log.info(f'Loading model from {cfg.hf_model_name}')
    model = AutoModelForSequenceClassification.from_pretrained(cfg.hf_model_name)
    pipe = pipeline(task='text-classification', model=model, tokenizer=tokenizer, device=cfg.hf_model_device)
    log.info('Pipeline created')

    # Run prediction
    log.info('Evaluating, this may take some time...')
    res = pipe(df[cfg.text_col].tolist(), batch_size=cfg.hf_eval_batch_size, truncation=True)
    res = pd.json_normalize(res)
    res['label'] = res['label'].str.split('_').apply(lambda x: int(x[1]))
    label_col = cfg.target_col + '_pred_label'
    score_col = cfg.target_col + '_pred_score'
    res = res.rename({'label': label_col, 'score': score_col}, axis=1)
    res.index = not_na_index

    # Save result
    output_file_name = cfg.output_file_prefix + cfg.target_col + '.csv'
    log.info(f'Saving result to {output_file_name}')
    res.to_csv(output_file_name)



if __name__ == '__main__':
    cfg = UMRunModelConfig()
    main(cfg)