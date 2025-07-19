import logging

import lightning
import torch
from clearml import Task
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.callbacks import BatchVisualize, ClearMLTrack, ModelSummary, ONNXExport
from src.configs import ProjectConfig
from src.constants import PROJECT_NAME
from src.data.data_module import ClassificationDataModule
from src.model import ClassificationLightningModule

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)


def train(config: ProjectConfig, path_dataset: str) -> str:
    lightning.seed_everything(0)

    Task.init(
        project_name=PROJECT_NAME,
        task_name='Training task',
        task_type=Task.TaskTypes.training
    )

    data_module = ClassificationDataModule(config)
    data_module.path_dataset = path_dataset

    callbacks = [
        ModelSummary(),
        BatchVisualize(config.dataset, every_n_epochs=5),
        ClearMLTrack(config, label_enumeration=data_module.class_to_idx),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor='valid_f2',
            mode='max',
            patience=3,
            verbose=True
        ),
        ModelCheckpoint(
            monitor='valid_f2',
            mode='max',
            save_top_k=3,
            filename='checkpoint-{epoch}-{valid_f2:.4f}'
        ),
        ONNXExport(config)
    ]

    model = ClassificationLightningModule(config, data_module.class_to_idx)

    logger.info(f'Current device: {config.trainer.accelerator}')

    trainer = lightning.Trainer(**dict(config.trainer), callbacks=callbacks)

    logger.info('Training started...')
    trainer.fit(model, datamodule=data_module)
    logger.info('Finished training')

    return str(trainer.checkpoint_callback.best_model_path)
