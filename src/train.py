from config import TrainConfig

import pandas as pd
import numpy as np
import datasets
import evaluate
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          pipeline)
import mlflow

import logging
logging.basicConfig()
log = logging.getLogger(__name__)


def main(cfg : TrainConfig):
    log.info('Training model...')
    log.debug(f'Parameters:\n{cfg.model_dump()}')

    log.info(f'Loading dataset from {cfg.dataset_path}')
    ds = datasets.load_from_disk(cfg.dataset_path)

    log.info(f'Text column: {cfg.text_col}')
    log.info(f'Target col: {cfg.target_col}')
    ds = ds.select_columns([cfg.text_col, cfg.target_col])
    ds = ds.rename_columns({cfg.text_col: 'texts', cfg.target_col: 'labels'})

    model_str = cfg.hf_model_family + '/' + cfg.hf_model_name
    log.info(f'Model is: {model_str}')

    log.info(f'Tokenizing')
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    # figure out max length
    if cfg.smoke_test:
        log.warning('--SMOKE TEST--')
        max_length = 25 # small number for rapid testing
    elif cfg.max_length == 'model_max_length':
        max_length = tokenizer.model_max_length
    else:
        max_length = cfg.max_length
    log.info(f'Max length will be: {max_length}')

    ds_tok = ds.map(lambda x: tokenizer(x['texts'], padding='max_length', truncation=True, max_length=max_length))

    log.info(f'Loading model')
    model = AutoModelForSequenceClassification.from_pretrained(model_str)
    
    training_args = TrainingArguments(**cfg.trainer_args.model_dump())
    log.info(f'HF Output directory: {cfg.trainer_args.output_dir}')
    log.debug(f'Training args:\n{training_args}')

    metric = evaluate.combine(['accuracy', 'f1', 'hyperml/balanced_accuracy','matthews_correlation'])
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        res = metric.compute(predictions=predictions, references=labels)
        return res
    
    ds_tok_train = ds_tok['train']
    ds_tok_test = ds_tok['test']
    if cfg.smoke_test:
        ds_tok_train = ds_tok_train.shuffle(seed=cfg.random_seed).select(range(100)) # small number for rapid testing
        ds_tok_test = ds_tok_test.shuffle(seed=cfg.random_seed).select(range(25))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok_train,
        eval_dataset=ds_tok_test,
        compute_metrics=compute_metrics
    )

    log.info(f'Using mlflow experiment {cfg.mlflow_experiment_name}')
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_text(cfg.model_dump_json(), 'model_cfg.json')
        trainer.train()

    if cfg.log_mlflow_model:
        tuned_pipeline = pipeline(
            task='text-classification',
            model=trainer.model,
            batch_size=cfg.trainer_args.per_device_eval_batch_size,
            tokenizer=tokenizer,
        )
        signature = mlflow.models.infer_signature(
            ['this is text1', 'this is text2'],
            mlflow.transformers.generate_signature_output(
                tuned_pipeline, ['This is a response','so is this']
            ),
            params={}
        )
        model_info = mlflow.transformers.log_model(
            transformers_model=tuned_pipeline,
            artifact_path=cfg.trainer_args.output_dir,
            signature=signature,
            input_example=['pass in a string','no really, do it'],
            model_config={}
        )

    if cfg.validate_mlflow_model:
        loaded = mlflow.transformers.load_model(model_uri=model_info.model_uri)
        validation_text = 'this is a test'
        print(loaded(validation_text))

if __name__ == '__main__':
    cfg = TrainConfig()
    log.setLevel(cfg.log_level)
    main(cfg)