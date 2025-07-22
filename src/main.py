import os

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
from src.constants import DEVICE, PATH_CHECKPOINTS, PATH_CONFIGS
from src.data.data_module import ClassificationDataModule
from src.model import ClassificationLightningModule
from src.utils.logger import LOGGER

torch.set_float32_matmul_precision('high')


def train(config: ProjectConfig):
    lightning.seed_everything(config.seed, workers=True)

    path_checkpoints = PATH_CHECKPOINTS / list(
        config.models.keys()
    )[config.active_model_index]

    os.makedirs(path_checkpoints, exist_ok=True)

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
            filename='checkpoint-{epoch}-{valid_f2:.4f}',
            dirpath=path_checkpoints
        ),
        ONNXExport(config, path_checkpoints)
    ]

    model = ClassificationLightningModule(config, data_module.class_to_idx)

    LOGGER.info(f'Current device: {DEVICE}')

    trainer = lightning.Trainer(
        accelerator=DEVICE,
        **dict(config.trainer),
        callbacks=callbacks
    )

    LOGGER.info('Training started...')
    trainer.fit(model, datamodule=data_module)
    LOGGER.info('Finished training')

    LOGGER.info('Testing started...')
    trainer.test(model, datamodule=data_module)
    LOGGER.info('Finished testing')


if __name__ == '__main__':
    config = init_config(PATH_CONFIGS)
    train(config)
