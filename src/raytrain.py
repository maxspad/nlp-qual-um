
from .config import TrainerArgs, TrainConfig, TrainConfigNoCLI
from . import train as tr

from transformers import TrainingArguments, Trainer
import ray.train.huggingface.transformers
from ray import tune, train
from ray.air.integrations.mlflow import MLflowLoggerCallback

import logging
logging.basicConfig()
log = logging.getLogger(__name__)

def raytrain(cfg : TrainConfig):
    if type(cfg) == dict:
        cfg = TrainConfigNoCLI.model_validate(cfg)

    log.setLevel(cfg.log_level)

    ds_train = tr.load_train_dataset(cfg)
    ds_train = tr.preprocess_dataset(cfg, ds_train)
    model_str = cfg.hf_model_family + '/' + cfg.hf_model_name

    tokenizer = tr.get_tokenizer(model_str)
    tok_func = tr.make_tok_func(cfg, tokenizer)
    ds_train_tok = tr.tokenize(ds_train, tok_func)

    model = tr.get_model(model_str)

    training_args = TrainingArguments(**cfg.trainer_args.model_dump())

    compute_metrics = tr.make_compute_metrics_func(cfg.metrics, use_ray=False)
    ds_tok_train, ds_tok_eval = tr.split_train_eval(ds_train_tok, cfg.eval_size, cfg.random_seed)
    if cfg.smoke_test:
        ds_tok_train = ds_tok_train.shuffle(seed=cfg.random_seed).select(range(100)) # small number for rapid testing
        ds_tok_eval = ds_tok_eval.shuffle(seed=cfg.random_seed).select(range(25))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok_train,
        eval_dataset=ds_tok_eval,
        compute_metrics=compute_metrics
    )
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    trainer.train()

if __name__ == '__main__':
    cfg = TrainConfig()
    cfg_dict = cfg.model_dump()
    cfg_dict['trainer_args']['num_train_epochs'] = tune.randint(1,3)
    tuner = tune.Tuner(
        raytrain,
        param_space=cfg.model_dump(),
        tune_config=tune.TuneConfig(
            mode='max',
            metric='eval_accuracy',
            num_samples=2
        ),
        run_config=train.RunConfig(
            storage_path='~/proj/nlp_qual/nlp-qual-um/ray_results/',
            callbacks=[
                MLflowLoggerCallback(
                    experiment_name=cfg.mlflow_experiment_name,
                    save_artifact=True
                )
            ],
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=1
            ),
            verbose=1
        )
    )
    results = tuner.fit()
    best = results.get_best_result('eval_accuracy', mode='max')
