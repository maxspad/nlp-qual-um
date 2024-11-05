from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    log_level : str = "INFO"

class MakeDataSetConfig(Config):
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
    random_seed : int = 43

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