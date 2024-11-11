
# Logging setup
import logging
logging.basicConfig()
log = logging.getLogger(__name__)

# Configuration types
from .config import TrainConfig, TrainConfigNoCLI

# Datascience imports
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
import mlflow.types
from ray import train as raytrain

# Typing and utils
from dataclasses import dataclass
from typing import Union, Callable, Any

def load_train_dataset(cfg : TrainConfig) -> datasets.Dataset:
    log.info(f'Loading dataset from {cfg.dataset_path}')
    # this script is only for training / evaluation, not for testing
    ds = datasets.load_from_disk(cfg.dataset_path)['train']
    return ds

def preprocess_dataset(cfg : TrainConfig, ds : datasets.Dataset) -> datasets.Dataset:
    log.info(f'Text column: {cfg.text_col}')
    log.info(f'Target col: {cfg.target_col}')
    ds = ds.select_columns([cfg.text_col, cfg.target_col])
    ds = ds.rename_columns({cfg.text_col: 'texts', cfg.target_col: 'labels'})
    return ds

def get_model_str(cfg : TrainConfig) -> str:
    # the model we will be using
    model_str = cfg.hf_model_family + '/' + cfg.hf_model_name
    log.info(f'Model is: {model_str}')
    return model_str

def get_tokenizer(model_str : str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_str)

def make_tok_func(cfg : TrainConfig, tokenizer : AutoTokenizer) -> Callable[[Any], dict]:
    # figure out max length
    if cfg.smoke_test:
        log.warning('--SMOKE TEST--')
        max_length = 25 # small number for rapid testing
    elif cfg.tokenizer_max_length == 'model_max_length':
        max_length = tokenizer.model_max_length
    else:
        max_length = cfg.tokenizer_max_length
    log.info(f'Max length will be: {max_length}')
    return lambda x: tokenizer(x['texts'], padding='max_length', 
                                truncation=True, max_length=max_length)

def tokenize(ds : datasets.Dataset, tok_func : Callable[[Any], dict]) -> datasets.Dataset:
    return ds.map(tok_func)

def get_model(model_str : str) -> AutoModelForSequenceClassification: 
    log.info(f'Loading model {model_str}')
    return AutoModelForSequenceClassification.from_pretrained(model_str)

def make_compute_metrics_func(metrics : list[str]) -> Callable[[dict], dict]:
    metric = evaluate.combine(metrics)
    def compute_metrics(eval_pred : dict) -> dict:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        res = metric.compute(predictions=predictions, references=labels)
        return res
    return compute_metrics

def split_train_eval(train_ds : datasets.Dataset, eval_size : float, random_seed : int) -> datasets.DatasetDict:
    split = train_ds.train_test_split(test_size=eval_size, seed=random_seed)
    return split['train'], split['test']

@dataclass
class TrainResult:
    mlflow_run : Any = None # TODO: need a type for this
    trainer : Trainer = None

def train_model(
    cfg : TrainConfig,
    model : AutoModelForSequenceClassification,
    training_args : TrainingArguments,
    train_ds : datasets.Dataset,
    eval_ds : datasets.Dataset,
    compute_metrics_func : Callable[[dict], dict],
) -> TrainResult:

    log.info(f'Using mlflow experiment {cfg.mlflow_experiment_name}')
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_func
    )
    with mlflow.start_run() as run:
        log.info(f'Mlflow run name: {run.info.run_name}')
        log.info(f'Mlflow run id: {run.info.run_id}')

        # Log all the things
        mlflow.log_param('TrainConfig', cfg.model_dump())
        mlflow.log_param('TrainerArgs', cfg.trainer_args.model_dump())
        mlflow.log_text(cfg.model_dump_json(), 'TrainConfig.json')
        mlflow.log_text(cfg.trainer_args.model_dump_json(), 'TrainerArgs.json')

        # train
        trainer.train()    
    return TrainResult(mlflow_run=run, trainer=trainer)

def log_mlflow_model(res : TrainResult, tok : AutoTokenizer):
    tuned_pipeline = pipeline(
        task='text-classification',
        model=res.trainer.model,
        batch_size=cfg.trainer_args.per_device_eval_batch_size,
        tokenizer=tok,
    )
    signature = mlflow.models.infer_signature(
        ['this is text1', 'this is text2'],
        mlflow.transformers.generate_signature_output(
            tuned_pipeline, ['This is a response','so is this']
        ),
        params={}
    )
    with mlflow.start_run(run_id=res.mlflow_run.info.run_id):
        model_info = mlflow.transformers.log_model(
            transformers_model=tuned_pipeline,
            artifact_path=cfg.trainer_args.output_dir,
            signature=signature,
            input_example=['pass in a string','no really, do it'],
            model_config={}
        )
    log.info(f'Logged mlflow model to {model_info.model_uri}')
    return model_info

def verify_mlflow_model(model_info):
    log.info(f'Validating model saved to {model_info.model_uri}')
    loaded = mlflow.transformers.load_model(model_uri=model_info.model_uri)
    validation_text = 'this is a test'
    log.info(f'Model output: {loaded(validation_text)}')

def train(cfg : Union[TrainConfig,dict[str, Any]]):
    if type(cfg) == dict:
        # convert this back to a TrainConfig
        # probably coming from Ray
        cfg = TrainConfigNoCLI.model_validate(cfg)

    log.setLevel(cfg.log_level)
    log.info('Training model...')
    log.debug(f'Parameters:\n{cfg.model_dump()}')

    # load and preprocess dataset
    ds_train = load_train_dataset(cfg)
    ds_train = preprocess_dataset(cfg, ds_train)

    # get the model string
    model_str = get_model_str(cfg)

    # get the tokenizer and tokenize
    tokenizer = get_tokenizer(model_str)
    tok_func = make_tok_func(cfg, tokenizer)
    ds_train_tok = tokenize(ds_train, tok_func)

    # get the model
    model = get_model(model_str)

    # Set up arguments from config for HuggingFace Trainer
    training_args = TrainingArguments(**cfg.trainer_args.model_dump())
    log.debug(f'Training args:\n{training_args}')

    # Create a function that takes logits and returns metrics
    compute_metrics = make_compute_metrics_func(cfg.metrics)

    # Split the "train" dataset into a train and an eval dataset
    # Do not use the test dataset to avoid data leakage 
    # TODO: could do cross-validation here
    ds_tok_train, ds_tok_eval = split_train_eval(ds_train_tok, cfg.eval_size, cfg.random_seed)

    # If running quickly, shrink the dataset sizes
    if cfg.smoke_test:
        ds_tok_train = ds_tok_train.shuffle(seed=cfg.random_seed).select(range(cfg.smoke_test_train_size))
        ds_tok_eval = ds_tok_eval.shuffle(seed=cfg.random_seed).select(range(cfg.smoke_test_eval_size))

    # Train the actual model, including mlflow logging of params
    training_res = train_model(cfg, model, training_args, ds_tok_train, ds_tok_eval, compute_metrics)

    # Optionally log the trained model as an mlflow model
    # and verify that it can be loaded
    if cfg.log_mlflow_model:
        model_info = log_mlflow_model(training_res, tokenizer)
        if cfg.validate_mlflow_model:
            verify_mlflow_model(model_info=model_info)

    return training_res

if __name__ == '__main__':
    cfg = TrainConfig()
    train(cfg)