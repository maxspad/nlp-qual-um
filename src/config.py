from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from typing import Union, Literal, Optional, Any, List, Callable 

class Config(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True,
                                      nested_model_default_partial_update=True,
                                      env_nested_delimiter='__')

class UMMakeDatasetConfig(Config):
    
    log_level : str = "INFO"
    random_seed : int = 43

    # inputs
    mapping_file : str = 'conf/mapping.xlsx'
    data_dir : str = 'data/interim/um/'

    text_col : str = 'text'
    text_col_split_char : str = "|"
    text_col_join_char : str = " "
    text_col_blank_repl_str : str = ""
    text_all_blank_repl_str : str = "blank"
    
    # output
    output_file : str = 'data/processed/um.csv'

class UMRunModelConfig(Config):
    log_level : str = "INFO"
    random_seed : int = 43
    smoke_test : bool = False

    # inputs
    data_file : str = 'data/processed/um.csv'
    text_col : str = 'text'
    target_col : str = 'qual'

    # model
    hf_model_name : str = 'maxspad/nlp-qual-qual'
    hf_model_max_length : int = 512
    hf_model_device : str = 'mps'
    hf_eval_batch_size : int = 16

    # output
    output_file_prefix : str = f'data/processed/preds_'


class MakeDataSetConfig(Config):

    log_level : str = "INFO"
    random_seed : int = 43

    # inputs
    masterdb_path : str = "data/raw/masterdbFromRobMac.xlsx"
    mac_path : str = "data/raw/mcmaster-database-de-identified-comments.xlsx"
    sas_path : str = "data/raw/sask-database-de-identified-comments.xlsx"

    # outputs
    output_folder : str = "data/processed"
    dataset_folder : str = output_folder + '/hf_dataset'

    save_csvs : bool = True
    masterdb_csv_fn : str = output_folder + "/masterdb.csv"
    train_csv_fn : str = output_folder + "/train.csv"
    test_csv_fn : str = output_folder + "/test.csv"

    test_size : float = 0.2

    text_var : str = 'comment'

    q1_col_name : str = 'Q1'
    q1_level_names : list[str] = ['none','low','medium','high']
    q2_col_name : str = 'Q2'
    q2_level_names : list[str] = ['no suggestion','suggestion']
    q3_col_name : str = 'Q3'
    q3_level_names : list[str] = ['no linkage', 'linkage']
    qual_col_name : str = 'QUAL'

    q1_condense_col_name : str = "Q1c"
    q1_condense_level_names : list[str] = ['low','high']
    q2_invert_col_name : str = "Q2i"
    q3_invert_col_name : str = "Q3i"
    qual_condense_col_name : str = "QUALc"

class MyTrainingArguments(BaseModel):
    output_dir : str = '.'
    
    per_device_train_batch_size : int = 8
    per_device_eval_batch_size : int = 8

    gradient_accumulation_steps : int = 1

    eval_strategy : str = 'epoch'
    eval_steps : Optional[int] = None
    eval_on_start : bool = False

    learning_rate : float = 5e-5
    weight_decay : float = 0.0

    num_train_epochs : int = 1
    max_steps : int = -1

    warmup_ratio : float = 0.0
    warmup_steps : int = 0

    log_level : Optional[str] = 'passive'

    logging_dir : Optional[str] = None
    logging_strategy : str = 'no'
    logging_first_step : bool = False
    logging_steps : float = 500

    save_strategy : str = 'no'
    save_steps : float = 500
    save_total_limit : Optional[int] = None

    no_cuda : bool = False
    use_cpu : bool = False
    use_mps_device : bool = False

    seed : int = 43

class TrainConfig(Config):

    script_log_level : str = "INFO"
    random_seed : int = 43

    dataset_path : str = 'data/processed/hf_dataset'
    
    text_col : str = 'comment'
    target_col : str = 'Q2i'

    smoke_test : bool = False
    smoke_test_train_size : int = 100
    smoke_test_eval_size : int = 25

    hf_model_family : str = 'distilbert'
    hf_model_name : str = 'distilbert-base-uncased'

    tokenizer_max_length : Union[Literal['model_max_length'], int] = 'model_max_length'
   
    mlflow_tracking_uri : str = './mlruns/'
    mlflow_experiment_name : str = 'scratch'
    mlflow_run_name : str = 'train-run'
    
    reload_model_after_training : bool = False
    log_mlflow_model : bool = False
    validate_mlflow_model : bool = False

    # if n_folds is not None, do k-fold cross validation
    n_folds : Optional[int] = None
    # otherwise, if eval_size is not None, use an eval set of size eval_size
    eval_size : Optional[float] = None

    metrics : list[str] = [
        'accuracy', 'f1', 
        'hyperml/balanced_accuracy', 
        'matthews_correlation'
    ]

    trainer_args : MyTrainingArguments = MyTrainingArguments()


class TrainConfigNoCLI(TrainConfig):
    '''This is a hack used to get around the fact that you can't instantiate
    CLI pydantic-settings classes in jupyter notebooks'''
    model_config = SettingsConfigDict(cli_parse_args=False)
