from config import TrainConfig
import numpy as np
from scipy.stats import loguniform, uniform
from train import train

if __name__ == '__main__':
    cfg = TrainConfig()
    print('Launching random trial...')

    cfg = cfg.model_dump()
    params = {
        'num_train_epochs': np.random.randint(1, 5),
        'learning_rate': loguniform.rvs(1e-5, 1e-1),
        'weight_decay': uniform.rvs(),
        'warmup_steps': np.random.randint(0, 100)
    }
    print('Chosen parameters:')
    print(params)

    cfg['trainer_args'].update(params)
    cfg = TrainConfig.model_validate(cfg)
    
    train(cfg)



