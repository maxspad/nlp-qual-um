# Transformers
from transformers import (
    Trainer, TrainingArguments, 
    TrainerCallback, TrainerState,
    TrainerControl
)
from transformers.trainer import get_last_checkpoint

# Ray
from ray import train 
from ray.train import Checkpoint

# mlflow
import mlflow

# File manipulation
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

# Logging
import logging
logging.basicConfig()
log = logging.getLogger(__name__)

class RayMLFlowCallback(TrainerCallback):
    CHECKPOINT_NAME = "checkpoint"
    CLASS_NAME = 'RayMLFlowCallback'

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        log.debug(f'{self.CLASS_NAME}: on_evaluate')
        with TemporaryDirectory() as tmpdir:
            metrics = {}
            for entry in state.log_history:
                metrics.update(entry)

            log.debug(f'{self.CLASS_NAME}: checkpoint tmpdir: {tmpdir}')
            # Copy ckpt files and construct a Ray Train Checkpoint
            source_ckpt_path = get_last_checkpoint(args.output_dir)
            if source_ckpt_path is not None:
                target_ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
                shutil.copytree(source_ckpt_path, target_ckpt_path)
                checkpoint = Checkpoint.from_directory(tmpdir)
            else:
                checkpoint = None
            
            # Log to both mlflow and ray train
            mlflow.log_metrics(metrics=metrics, step=state.global_step)
            train.report(metrics=metrics, checkpoint=checkpoint)


