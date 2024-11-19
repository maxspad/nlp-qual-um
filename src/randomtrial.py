print('Starting imports')
print('config...')
from config import TrainConfig
print('numpy...')
import numpy as np
print('scipy...')
from scipy.stats import loguniform, uniform
print('train...')
from train import train

if __name__ == '__main__':
    cfg = TrainConfig()
    print('Launching random trial...')

    cfg = cfg.model_dump()
    params = {
        'learning_rate': loguniform.rvs(1e-5, 1e-1),
        'per_device_train_batch_size': np.random.choice([8, 16, 32]),
        'num_train_epochs': np.random.randint(1, 5),
        'warmup_ratio': np.random.uniform(0, 0.1),
        'weight_decay': np.random.uniform(0, 0.01),
        'gradient_clipping': np.random.uniform(1.0, 5.0),
    }
    cfg['tokenizer_max_length'] = np.random.choice([128, 256, 512])
    print('Chosen parameters:')
    print(params)

    cfg['trainer_args'].update(params)
    cfg = TrainConfig.model_validate(cfg)
    
    train(cfg)



