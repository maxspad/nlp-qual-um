from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from typing import Union, Literal, Optional, Any, List

class Config(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    log_level : str = "INFO"
    random_seed : int = 43

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

class TrainerArgs(BaseModel):

    output_dir : str = 'hf_training_outputs'
    overwrite_output_dir : bool = False
    do_train : bool = True
    do_predict : bool = True
    eval_strategy : Literal['no','epoch','steps'] = 'epoch'
    per_device_train_batch_size : int = 8
    per_device_eval_batch_size : int = 8 
    gradient_accumulation_steps : int = 1
    eval_accumulation_steps : int = 1
    eval_steps : Union[int, float] = 500
    
    learning_rate : float = 5e-5
    weight_decay : float = 0
    adam_beta1 : float = 0.9
    adam_beta2 : float = 0.999
    adam_epsilon : float = 1e-8
    max_grad_norm : float = 1.0
    
    num_train_epochs : float = 3.0
    max_steps : int = -1

    lr_scheduler_type : str = 'linear'
    lr_scheduler_kwargs : dict[str, Any] = {}
    warmup_ratio : float = 0.0
    warmup_steps : int = 0

    logging_dir : Optional[str] = None
    logging_strategy : Literal['no','epoch','steps'] = 'epoch'
    logging_first_step : bool = False 
    logging_steps : Union[int, float] = 500
    logging_nan_inf_filter : bool = True

    save_strategy : Literal['no','epoch','steps'] = 'epoch'
    save_steps : Union[int, float] = 500
    save_total_limit : Optional[int] = None
    save_safetensors : bool = True
    save_on_each_node : bool = False
    save_only_model : bool = False

    use_cpu : bool = False
    seed : int = 43
    data_seed : int = seed

    jit_mode_eval : bool = False
    use_ipex : bool = False
    bf16 : bool = False
    fp16 : bool = False
    fp16_opt_level : Literal['O0','O1','O2','O3'] = 'O1'
    half_precision_backend : Literal['auto','apex','cpu_amp'] = 'auto'
    bf16_full_eval : bool = False

    tf32 : Optional[bool] = None

    local_rank : int = -1
    ddp_backend : Optional[Literal['nccl','mpi','ccl','gloo','hccl']] = None
    tpu_num_cores : Optional[int] = None

    dataloader_drop_last : bool = False
    dataloader_num_workers : int = 0
    past_index : int = -1

    run_name : str = output_dir
    disable_tqdm : Optional[bool] = None
    remove_unused_columns : bool = False

    label_names : Optional[List[str]] = None

    load_best_model_at_end : bool = False
    metric_for_best_model : Optional[str] = None
    greater_is_better : Optional[bool] = None

    ignore_data_skip : bool = False
    
    fsdp : Optional[Union[Literal[
        'full_shard',
        'shard_grad_op',
        'hybrid_shard',
        'hybrid_shard_zero2',
        'offload',
        'auto_wrap'
    ], bool, str, list]] = ''
    fsdp_config : Optional[Union[str, dict[str,Any]]] = None

    deepspeed : Optional[Union[str,dict[str,Any]]] = None

    accelerator_config : Optional[Union[str, dict[str, Any]]] = None

    label_smoothing_factor : float = 0.0

    debug : Union[str, list] = ""

    optim : str = 'adamw_torch'

    push_to_hub : bool = False
    hub_model_id : Optional[str] = None
    hub_strategy : str = 'every_save'
    hub_token : Optional[str] = None
    hub_private_repo : bool = False
    hub_always_push : bool = False

    gradient_checkpointing : bool = False
    gradient_checkpointing_kwargs : Optional[dict] = None

    include_for_metrics : Optional[List[str]] = []
    eval_do_concat_batches : bool = True
    auto_find_batch_size : bool = False
    full_determinism : bool = False

    torchdynamo : Optional[str] = None
    torch_compile : bool = False
    torch_compile_backend : Optional[str] = None
    torch_compile_mode : Optional[str] = None
    
    ray_scope : str = 'last'
    ddp_timeout : int = 1800
    split_batches : Optional[bool] = None
    include_tokens_per_second : Optional[bool] = None
    include_num_input_tokens_seen : Optional[bool] = None

    neftune_noise_alpha : Optional[float] = None
    optim_target_modules : Optional[Union[str, List[str]]] = None
    
    batch_eval_metrics : Optional[bool] = False
    eval_on_start : bool = False
    eval_use_gather_object : bool = False
    use_liger_kernel : bool = False


class TrainConfig(Config):
    dataset_path : str = 'data/processed/hf_dataset'
    
    text_col : str = 'comment'
    target_col : str = 'Q2i'

    smoke_test : bool = False

    hf_model_family : str = 'distilbert'
    hf_model_name : str = 'distilbert-base-uncased'

    max_length : Union[Literal['model_max_length'], int] = 'model_max_length'

    trainer_args : TrainerArgs = TrainerArgs()

