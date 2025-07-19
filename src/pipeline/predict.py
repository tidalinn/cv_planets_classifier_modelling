import logging

import torch
from clearml import Task
from lightning import Trainer

from src.configs import ProjectConfig
from src.constants import PROJECT_NAME
from src.data.data_module import ClassificationDataModule
from src.model import ClassificationLightningModule

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)


def predict(config: ProjectConfig, path_model: str):
    Task.init(
        project_name=PROJECT_NAME,
        task_name='Prediction task',
        task_type=Task.TaskTypes.inference
    )

    data_module = ClassificationDataModule(config)
    model = ClassificationLightningModule.load_from_checkpoint(
        checkpoint_path=path_model,
        config=config,
        class_to_idx=data_module.class_to_idx
    )
    trainer = Trainer(**dict(config.trainer))

    logger.info('Testing started...')
    trainer.test(model, datamodule=data_module)
    logger.info('Testing finished')
