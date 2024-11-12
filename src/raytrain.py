
from transformers.trainer_callback import TrainerControl, TrainerState
from config import TrainerArgs, RayTrainerConfig, TrainConfig, TrainConfigNoCLI
import train as tr

from transformers import TrainingArguments, Trainer
import ray.train.huggingface.transformers
from ray import tune, train
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from transformers.integrations import MLflowCallback
from transformers import TrainerCallback
import mlflow

import logging
logging.basicConfig()
log = logging.getLogger(__name__)

class MyMLFlowLogger(TrainerCallback):
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print('on_train_begin')
        import pdb; pdb.set_trace()
        mlflow = setup_mlflow(config={'bananas': 'yes'},
                     tracking_uri='~/proj/nlp_qual/nlp-qual-um/mlruns',
                     create_experiment_if_not_exists=True,
                     experiment_name='bananas',
                     rank_zero_only=False)
        mlflow.log_param('banana', 123)
        print('finished mlflow setup')

    # def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     print('on_train_begin')
    #     import pdb; pdb.set_trace()
    #     mlflow = setup_mlflow(config={'bananas': 'yes'},
    #                  tracking_uri='~/proj/nlp_qual/nlp-qual-um/mlruns',
    #                  create_experiment_if_not_exists=True,
    #                  experiment_name='bananas')
    #     mlflow.log_param('banana', 123)
    #     print('finished mlflow setup')



def tune_transformer(cfg : RayTrainerConfig):
    log.setLevel(cfg.train_config.log_level)

    ds_train = tr.load_train_dataset(cfg.train_config)
    ds_train = tr.preprocess_dataset(cfg.train_config, ds_train)
    model_str = cfg.train_config.hf_model_family +'/'+ cfg.train_config.hf_model_name

    tokenizer = tr.get_tokenizer(model_str)
    tok_func = tr.make_tok_func(cfg.train_config, tokenizer)
    ds_train_tok = tr.tokenize(ds_train, tok_func)

    model = tr.get_model(model_str)
    get_model_func = lambda _: tr.get_model(model_str)

    training_args = TrainingArguments(**cfg.train_config.trainer_args.model_dump())

    compute_metrics = tr.make_compute_metrics_func(cfg.train_config.metrics)
    ds_tok_train, ds_tok_eval = tr.split_train_eval(ds_train_tok, cfg.train_config.eval_size, 
                                                    cfg.train_config.random_seed)
    if cfg.train_config.smoke_test:
        ds_tok_train = ds_tok_train.shuffle(seed=cfg.train_config.random_seed).select(range(100)) # small number for rapid testing
        ds_tok_eval = ds_tok_eval.shuffle(seed=cfg.train_config.random_seed).select(range(25))

    trainer = Trainer(
        model_init=get_model_func,
        args=training_args,
        train_dataset=ds_tok_train,
        eval_dataset=ds_tok_eval,
        compute_metrics=compute_metrics,
        callbacks=[MyMLFlowLogger]
    )
    trainer.pop_callback(MLflowCallback)

    res = trainer.hyperparameter_search(
        hp_space=lambda _: {
            'num_train_epochs': 1
        },
        compute_objective=lambda metrics: metrics['eval_accuracy'],
        resources_per_trial = {
            'cpu': 1
        },
        backend='ray',
        n_trials=1,
        storage_path='~/proj/nlp_qual/nlp-qual-um/ray_results/',
        # callbacks=[
        #     MLflowLoggerCallback(
        #         tracking_uri='./mlruns',
        #         experiment_name='ray_mlflow_cb_example'
        #     )
        # ]
    )
    print(res)


# def raytrain(cfg : TrainConfig):
#     if type(cfg) == dict:
#         cfg = TrainConfigNoCLI.model_validate(cfg)

#     log.setLevel(cfg.log_level)

#     ds_train = tr.load_train_dataset(cfg)
#     ds_train = tr.preprocess_dataset(cfg, ds_train)
#     model_str = cfg.hf_model_family + '/' + cfg.hf_model_name

#     tokenizer = tr.get_tokenizer(model_str)
#     tok_func = tr.make_tok_func(cfg, tokenizer)
#     ds_train_tok = tr.tokenize(ds_train, tok_func)

#     model = tr.get_model(model_str)

#     training_args = TrainingArguments(**cfg.trainer_args.model_dump())

#     compute_metrics = tr.make_compute_metrics_func(cfg.metrics, use_ray=False)
#     ds_tok_train, ds_tok_eval = tr.split_train_eval(ds_train_tok, cfg.eval_size, cfg.random_seed)
#     if cfg.smoke_test:
#         ds_tok_train = ds_tok_train.shuffle(seed=cfg.random_seed).select(range(100)) # small number for rapid testing
#         ds_tok_eval = ds_tok_eval.shuffle(seed=cfg.random_seed).select(range(25))

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=ds_tok_train,
#         eval_dataset=ds_tok_eval,
#         compute_metrics=compute_metrics
#     )
#     callback = ray.train.huggingface.transformers.RayTrainReportCallback()
#     trainer.add_callback(callback)
#     trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

#     trainer.train()

if __name__ == '__main__':
    cfg = RayTrainerConfig()
    tune_transformer(cfg)
    # print(cfg.train_config.trainer_args.eval_strategy)
    # print(cfg.train_config.trainer_args.save_strategy)
    # cfg.train_config.dataset_path = '~/proj/nlp_qual/nlp-qual-um/data/processed/hf_dataset/'
    # tuner = tune.Tuner(
    #     tr.train,
    #     param_space=cfg.train_config.model_dump(),
    #     tune_config=tune.TuneConfig(
    #         mode='max',
    #         metric='eval_accuracy'
    #     ),
    #     run_config=train.RunConfig(
    #         storage_path='~/proj/nlp_qual/nlp-qual-um/ray_results/',
    #         verbose=1
    #     )
    # )
    # tuner.fit()
    # tr.train(cfg)
    # cfg_dict = cfg.model_dump()
    # cfg_dict['trainer_args']['num_train_epochs'] = tune.randint(1,3)
    # tuner = tune.Tuner(
    #     raytrain,
    #     param_space=cfg.model_dump(),
    #     tune_config=tune.TuneConfig(
    #         mode='max',
    #         metric='eval_accuracy',
    #         num_samples=2
    #     ),
    #     run_config=train.RunConfig(
    #         storage_path='~/proj/nlp_qual/nlp-qual-um/ray_results/',
    #         callbacks=[
    #             MLflowLoggerCallback(
    #                 experiment_name=cfg.mlflow_experiment_name,
    #                 save_artifact=True
    #             )
    #         ],
    #         checkpoint_config=train.CheckpointConfig(
    #             num_to_keep=1
    #         ),
    #         verbose=1
    #     )
    # )
    # results = tuner.fit()
    # best = results.get_best_result('eval_accuracy', mode='max')
