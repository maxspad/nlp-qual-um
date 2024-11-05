from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    log_level : str = "INFO"

class MakeDataSetConfig(Config):
    masterdb_path : str = "data/raw/masterdbFromRobMac.xlsx"
    mac_path : str = "data/raw/mcmaster-database-de-identified-comments.xlsx"
    sas_path : str = "data/raw/sask-database-de-identified-comments.xlsx"
    output_path : str = "data/interim/masterdbforNLP.csv"

    q1_condense : bool = True
    q1_condense_col_name : str = "Q1c"
    q2_invert : bool = True
    q2_invert_col_name : str = "Q2i"
    q3_invert : bool = True
    q3_invert_col_name : str = "Q3i"
    qual_condense : bool = True
    qual_condense_col_name : str = "QUALc"
    
    text_var: str = "comment"

class SplitTrainTestConfig(Config):
    dataset_path : str = 'data/interim/masterdbforNLP.csv'
    output_dir : str = 'data/processed'
    train_path : str = output_dir + '/train.csv'
    test_path : str = output_dir + '/test.csv'
    test_size : float = 0.2
    random_state : int = 43

class TrainConfig(Config):
    train_path : str = 'data/processed/train.csv' 
    text_col : str = 'comment'
    target_col : str = 'Q2i'