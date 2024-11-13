from ray import tune, train
import ray
import mlflow
from config import RayTrainerConfig, TrainConfig
import train as tr
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
from transformers.integrations import MLflowCallback
from ray.train.huggingface.transformers import RayTrainReportCallback
import transformers.trainer
import logging
import shutil
from ray.train import Checkpoint
from pathlib import Path
from tempfile import TemporaryDirectory
logging.basicConfig()
log = logging.getLogger(__name__)

class MyCallback(TrainerCallback):
    CHECKPOINT_NAME = "checkpoint"

    def on_evaluate(self, args: TrainingArguments, state: tr.TrainerState, control: tr.TrainerControl, **kwargs):
        log.debug('MyCallback: on_evaluate')
        log.debug(f'MyCallback: state.log_history:\n{state.log_history}')
        with TemporaryDirectory() as tmpdir:
            metrics = {}
            for blah in state.log_history:
                metrics.update(blah)

            log.debug(f'MyCallback: checkpoint tmpdir: {tmpdir}')
            # Copy ckpt files and construct a Ray Train Checkpoint
            source_ckpt_path = transformers.trainer.get_last_checkpoint(args.output_dir)
            if source_ckpt_path is not None:
                target_ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
                shutil.copytree(source_ckpt_path, target_ckpt_path)
                checkpoint = Checkpoint.from_directory(tmpdir)
            else:
                checkpoint = None
            mlflow.log_metrics(metrics=metrics, step=state.global_step)
            train.report(metrics=metrics, checkpoint=checkpoint)

    def on_save(self, args: TrainingArguments, state: tr.TrainerState, control: tr.TrainerControl, **kwargs):
        log.debug('MyCallback: on_save')

def setup_mlflow(cfg : TrainConfig):
    log.debug('Setting up mlflow')
    log.debug(f'Setting up mlflow: tracking URI: {cfg.mlflow_dir}')
    log.debug(f'Setting up mlflow: experiment name: {cfg.mlflow_experiment_name}')
    mlflow.set_tracking_uri(cfg.mlflow_dir)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

def train_mlflow(config : dict):
    if type(config) == dict:
        cfg = TrainConfig.model_validate(config)
    else:
        cfg = config
    log.setLevel(cfg.log_level)

    # log.debug(f'Parameters:\n{cfg.model_dump()}')
    
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
    # log.debug(f'Training args:\n{training_args}')

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
    trainer.add_callback(MyCallback())
    print(trainer.pop_callback(MLflowCallback))
    log.info('Starting mlflow run...')
    log.debug(f'MLFlow tracking URI: {mlflow.get_tracking_uri()}')
    with mlflow.start_run() as run:
        log.debug(f'Run info: {run.info}')
        mlflow.log_params(cfg.model_dump())
        prefix_trainer_args = {'trainer_args.'+k : v for k,v in cfg.trainer_args.model_dump().items()}
        mlflow.log_params(prefix_trainer_args)
        mlflow.log_text(cfg.model_dump_json(), 'TrainConfig.json')
        mlflow.log_text(cfg.trainer_args.model_dump_json(), 'TrainerArgs.json')

        log.info('Starting training...')
        trainer.train()

        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
    # train.report(metrics)
    # print(f'***METRICS: {metrics}')
    # return metrics

if __name__ == '__main__':
    cfg = RayTrainerConfig()

    ray.init()
    
    param_space = cfg.train_config
    param_space.log_level = 'DEBUG'
    # param_space.trainer_args.num_train_epochs = tune.randint(1,3)
    param_space.trainer_args.num_train_epochs = 1
    param_space.smoke_test = True
    param_space.mlflow_dir = 'http://127.0.0.1:5000'
    param_space.dataset_path = '/Users/maxspad/proj/nlp_qual/nlp-qual-um/data/processed/hf_dataset/'
    param_space.trainer_args.log_level='debug'
    # param_space.trainer_args.disable_tqdm=False
    mlflow.set_tracking_uri(param_space.mlflow_dir)
    mlflow.set_experiment(cfg.train_config.mlflow_experiment_name)
    # train_mlflow(param_space)
    tuner = tune.Tuner(
        train_mlflow,
        param_space=param_space.model_dump(),
        tune_config=tune.TuneConfig(
            mode='max',
            metric='eval_accuracy',
            num_samples=3
        ),
        run_config=train.RunConfig(
            name='stupidtrain',
            storage_path='~/proj/nlp_qual/nlp-qual-um/stupid-ray-results',
            verbose=0
        )
    )
    results = tuner.fit()
    print(results)