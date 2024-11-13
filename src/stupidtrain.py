# Ray
import ray
from ray import tune, train

# mlflow
import mlflow

from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
from transformers.integrations import MLflowCallback
import logging

from config import RayTrainerConfig, TrainConfig
import train as tr
import train_helpers as th

logging.basicConfig()
log = logging.getLogger(__name__)

def setup_mlflow(cfg : TrainConfig):
    log.debug('Setting up mlflow')
    log.debug(f'Setting up mlflow: tracking URI: {cfg.mlflow_tracking_uri}')
    log.debug(f'Setting up mlflow: experiment name: {cfg.mlflow_experiment_name}')
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

def train_mlflow(config : dict):
    if type(config) == dict:
        cfg = TrainConfig.model_validate(config)
    else:
        cfg = config
    log.setLevel(cfg.log_level)

    log.debug('train_mlflow')
    log.debug(f'train_mlflow: parameters:\n{cfg.model_dump()}')
    
    setup_mlflow(cfg)

    # Load and preprocess datasets
    ds_train = tr.preprocess_dataset(
        cfg,
        tr.load_train_dataset(cfg)
    )

    model_str = cfg.hf_model_family + '/' + cfg.hf_model_name
    log.info(f'Model is {model_str}')

    log.info('Tokenizing...')
    tokenizer = tr.get_tokenizer(model_str)
    tok_func = tr.make_tok_func(cfg, tokenizer)
    ds_train_tok = tr.tokenize(ds_train, tok_func)

    log.info('Getting model...')
    model = tr.get_model(model_str)

    training_args = TrainingArguments(**cfg.trainer_args.model_dump())

    compute_metrics = tr.make_compute_metrics_func(cfg.metrics, use_ray=False)

    ds_tok_train, ds_tok_eval = tr.split_train_eval(ds_train_tok, cfg.eval_size, cfg.random_seed)
    if cfg.smoke_test:
        ds_tok_train = ds_tok_train.shuffle(seed=cfg.random_seed).select(range(100)) # small number for rapid testing
        ds_tok_eval = ds_tok_eval.shuffle(seed=cfg.random_seed).select(range(25))

    log.info('Setting up trainer...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok_train,
        eval_dataset=ds_tok_eval,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(th.RayMLFlowCallback())
    # Remove the huggingface native mlflow callback, it interferes
    # with our custom logging
    print(trainer.pop_callback(MLflowCallback))

    log.info('Starting mlflow run...')
    log.debug(f'MLFlow tracking URI: {mlflow.get_tracking_uri()}')
    with mlflow.start_run() as run:
        log.debug(f'MLFlow run info: {run.info}')
        mlflow.log_params(cfg.model_dump())
        prefix_trainer_args = {'trainer_args.'+k : v for k,v in cfg.trainer_args.model_dump().items()}
        mlflow.log_params(prefix_trainer_args)
        mlflow.log_text(cfg.model_dump_json(), 'TrainConfig.json')
        mlflow.log_text(cfg.trainer_args.model_dump_json(), 'TrainerArgs.json')

        log.info('Starting training...')
        trainer.train()

        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

if __name__ == '__main__':
    cfg = RayTrainerConfig()
    print(cfg.run_config) 

    ray.init()

    train_config = cfg.train_config
    
    mlflow.set_tracking_uri(train_config.mlflow_tracking_uri)
    mlflow.set_experiment(train_config.mlflow_experiment_name)

    tuner = tune.Tuner(
        train_mlflow,
        param_space=train_config.model_dump(),
        tune_config=tune.TuneConfig(**cfg.tune_config.model_dump()),
        run_config=train.RunConfig(**cfg.run_config.model_dump())
    )
    results = tuner.fit()

