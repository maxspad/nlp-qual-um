from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    log_level : str = "INFO"

class MakeDataSetConfig(Config):
    model_config = SettingsConfigDict(cli_parse_args=True)

    masterdb_path : str = "data/raw/masterdbFromRobMac.xlsx"
    mac_path : str = "data/raw/mcmaster-database-de-identified-comments.xlsx"
    sas_path : str = "data/raw/sask-database-de-identified-comments.xlsx"
    output_path : str = "data/interim/maxterdbforNLP.xlsx"

    q1_condense : bool = True
    q1_condense_col_name : str = "Q1c"
    q2_invert : bool = True
    q2_invert_col_name : str = "Q2i"
    q3_invert : bool = True
    q3_invert_col_name : str = "Q3i"
    qual_condense : bool = True
    qual_condense_col_name : str = "QUALc"
    
    text_var: str = "comment"