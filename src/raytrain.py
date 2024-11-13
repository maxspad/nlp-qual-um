from transformers.trainer_callback import TrainerControl, TrainerState
import train as tr
from config import RayTrainerConfig
from ray import tune, train
from transformers import TrainerCallback
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
  

if __name__ == '__main__':
    cfg = RayTrainerConfig()
    print(cfg.train_config.smoke_test)
    print(cfg.train_config.trainer_args.num_train_epochs)
    tuner = tune.Tuner(
        tr.train,
        param_space=cfg.train_config.model_dump(),
        tune_config=tune.TuneConfig(
            mode='max',
            metric='eval_accuracy',
            num_samples=1,
        ),
        # run_config=train.RunConfig(
        #     callbacks=[MLflowLoggerCallback(experiment_name='banana')]
        # )
    )
    results = tuner.fit()