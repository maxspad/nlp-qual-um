from config import TrainConfig

import pandas as pd
import numpy as np
import datasets
import evaluate
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          DataCollatorWithPadding)

import logging
import evaluate
logging.basicConfig()
log = logging.getLogger(__name__)


def main(cfg : TrainConfig):
    log.info('Traning model...')
    log.debug(f'Parameters:\n{cfg.model_dump()}')

    log.info(f'Loading dataset from {cfg.dataset_path}')
    ds = datasets.load_from_disk(cfg.dataset_path)

    log.info(f'Text column: {cfg.text_col}')
    log.info(f'Target col: {cfg.target_col}')
    ds = ds.select_columns([cfg.text_col, cfg.target_col])
    ds = ds.rename_columns({cfg.text_col: 'texts', cfg.target_col: 'labels'})

    log.info(f'Tokenizing')
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
    ds_tok = ds.map(lambda x: tokenizer(x['texts'], padding='max_length', truncation=True, max_length=50))
    log.info(f'Loading model')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased')
    
    training_args = TrainingArguments(
        output_dir='test_output_dir',
        eval_strategy='epoch'
    )

    metric = evaluate.combine(['accuracy', 'f1', 'hyperml/balanced_accuracy','matthews_correlation'])
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        res = metric.compute(predictions=predictions, references=labels)
        print('result is:', res)
        return res

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok['train'],# .shuffle(seed=cfg.random_seed).select(range(200)),
        eval_dataset=ds_tok['test'],#.shuffle(seed=cfg.random_seed).select(range(100)),
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == '__main__':
    cfg = TrainConfig()
    log.setLevel(cfg.log_level)
    main(cfg)