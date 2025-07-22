from pathlib import Path
from typing import Dict, Optional

from clearml import OutputModel, Task
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from src.configs import ProjectConfig
from src.constants import PROJECT_NAME
from src.utils.logger import LOGGER


class ClearMLTrack(Callback):

    def __init__(self, config: ProjectConfig, label_enumeration: Optional[Dict[str, int]] = None):
        super().__init__()
        self.config = config
        self.label_enumeration = label_enumeration

        self.task: Optional[Task] = None
        self.output_model: Optional[OutputModel] = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        self._setup_task()

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        final_checkpoint = self._select_checkpoint_for_export(trainer)
        extension = final_checkpoint.split('.')[-1]

        LOGGER.info(f'Uploading checkpoint {final_checkpoint} to ClearML')

        self.output_model.update_weights(
            weights_filename=final_checkpoint,
            target_filename=f'best.{extension}',
            auto_delete_file=True,
        )

    def _setup_task(self):
        Task.force_requirements_env_freeze()

        name_experiment = list(self.config.models.keys())[self.config.active_model_index]
        name_model = list(self.config.models.values())[self.config.active_model_index].name

        self.task = Task.init(
            project_name=PROJECT_NAME,
            task_name=f'{name_experiment}-{name_model}',
            output_uri=True,
            reuse_last_task_id=False,
            auto_connect_frameworks={'pytorch': False},
        )

        self.task.connect_configuration(self.config.model_dump())

        self.output_model = OutputModel(
            task=self.task,
            label_enumeration=self.label_enumeration
        )

    def _select_checkpoint_for_export(self, trainer: Trainer) -> str:
        checkpoint: Optional[ModelCheckpoint] = None

        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint = callback
                break

        if checkpoint is not None:
            path_checkpoint = checkpoint.best_model_path

            if Path(path_checkpoint).is_file():
                LOGGER.info(f'Selected best checkpoint at {path_checkpoint}')
                return path_checkpoint
            else:
                LOGGER.warning('Found no best checkpoint')

        path_checkpoint = Path(trainer.log_dir) / 'trainer-checkpoint.pth'

        trainer.save_checkpoint(path_checkpoint)
        LOGGER.info(f'Saved checkpoint to {path_checkpoint}')

        return path_checkpoint
