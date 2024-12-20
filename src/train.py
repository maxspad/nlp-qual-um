print('train: starting imports...')
print('config')
from config import TrainConfig, TrainConfigNoCLI

print('numpy')
import numpy as np

print('hf datasets')
import datasets
print('hf evaluate')
import evaluate
print('hf transformers')
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          pipeline)
print('mlflow')
import mlflow

print('dataclasses')
from dataclasses import asdict, dataclass

print('logging')
print('typing')
import logging
from typing import Union, Callable, Any

logging.basicConfig()
log = logging.getLogger(__name__)

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

def make_compute_metrics_func(metrics : list[str], use_ray : bool = False) -> Callable[[dict], dict]:
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
class TrainingResult:
    trainer : Trainer
    mlflow_run : Any

def train_model(
    cfg : TrainConfig,
    model : AutoModelForSequenceClassification,
    training_args : TrainingArguments,
    train_ds : datasets.Dataset,
    eval_ds : datasets.Dataset,
    compute_metrics_func : Callable[[dict], dict],
) -> None:

    log.info(f'Using mlflow experiment {cfg.mlflow_experiment_name}')
    log.info(f'Using mlflow tracking uri {cfg.mlflow_tracking_uri}')
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_func
    )

    with mlflow.start_run(run_name=cfg.mlflow_run_name) as run:
        # mlflow.log_params(cfg.model_dump())
        to_log = cfg.model_dump()
        del to_log['trainer_args']
        mlflow.log_params(to_log)
        # mlflow.log_param('TrainConfig', cfg.model_dump())
        # mlflow.log_param('TrainerArgs', asdict(cfg.trainer_args))
        mlflow.log_text(cfg.model_dump_json(), 'TrainConfig.json')

        # train
        trainer.train()   
    return TrainingResult(trainer=trainer, mlflow_run=run)

def log_mlflow_model(res, tok : AutoTokenizer):
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
            # artifact_path=cfg.trainer_args.output_dir,
            artifact_path='model',
            signature=signature,
            input_example=['pass in a string','no really, do it'],
            model_config={}
        )
    log.info(f'Logged mlflow model to {model_info.model_uri}')
    return model_info

def validate_mlflow_model(model_info):
    log.info(f'Validating model saved to {model_info.model_uri}')
    loaded = mlflow.transformers.load_model(model_uri=model_info.model_uri)
    validation_text = 'this is a test'
    log.info(f'Model output: {loaded(validation_text)}')

def train(cfg : Union[TrainConfig,dict[str, Any]]):
    if type(cfg) == dict:
        # convert this back to a TrainConfig
        # probably coming from Ray
        cfg = TrainConfig.model_validate(cfg)

    log.setLevel(cfg.script_log_level)
    log.info('Training model...')
    log.debug(f'Parameters:\n{cfg.model_dump()}')

    # load dataset
    ds_train = load_train_dataset(cfg)
    
    # preprocess dataset
    ds_train = preprocess_dataset(cfg, ds_train)

    # the model we will be using
    model_str = cfg.hf_model_family + '/' + cfg.hf_model_name
    log.info(f'Model is: {model_str}')

    # get the tokenizer and tokenize
    log.info(f'Tokenizing')
    tokenizer = get_tokenizer(model_str)
    tok_func = make_tok_func(cfg, tokenizer)
    ds_train_tok = tokenize(ds_train, tok_func)

    # get the model
    model = get_model(model_str)

    training_args = TrainingArguments(**cfg.trainer_args.model_dump())
    log.info(f'HF Output directory: {cfg.trainer_args.output_dir}')
    log.debug(f'Training args:\n{training_args}')

    compute_metrics = make_compute_metrics_func(cfg.metrics)

    ds_tok_train, ds_tok_eval = split_train_eval(ds_train_tok, cfg.eval_size, cfg.random_seed)
    if cfg.smoke_test:
        ds_tok_train = ds_tok_train.shuffle(seed=cfg.random_seed).select(range(cfg.smoke_test_train_size)) # small number for rapid testing
        ds_tok_eval = ds_tok_eval.shuffle(seed=cfg.random_seed).select(range(cfg.smoke_test_eval_size))

    training_res = train_model(cfg, model, training_args, ds_tok_train, ds_tok_eval, compute_metrics)
    if cfg.log_mlflow_model:
        model_info = log_mlflow_model(training_res, tokenizer)
        if cfg.validate_mlflow_model:
            validate_mlflow_model(model_info=model_info)

if __name__ == '__main__':
    cfg = TrainConfig(_env_file='.env')
    train(cfg)
