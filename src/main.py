import logging

import lightning
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.callbacks import BatchVisualize, ClearMLTrack, ModelSummary, ONNXExport
from src.configs import ProjectConfig
from src.configs.initializer import init_config
from src.constants import PATH_CONFIGS
from src.data.data_module import ClassificationDataModule
from src.model import ClassificationLightningModule
from src.utils.logger import init_logger

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)


def train(config: ProjectConfig):
    lightning.seed_everything(0)

    data_module = ClassificationDataModule(config)

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
    trainer = lightning.Trainer(**dict(config.trainer), callbacks=callbacks)

    logger.info(f'Current device: {config.trainer.accelerator}')

    logger.info('Training started...')
    trainer.fit(model, datamodule=data_module)
    logger.info('Finished training')

    logger.info('Testing started...')
    trainer.test(model, datamodule=data_module)
    logger.info('Finished testing')


if __name__ == '__main__':
    init_logger()
    config = init_config(PATH_CONFIGS)
    train(config)
