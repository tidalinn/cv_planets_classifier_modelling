import json
import os
import pickle

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
from src.constants import DEVICE, PATH_CHECKPOINTS, PROJECT_NAME
from src.data.data_module import ClassificationDataModule
from src.model import ClassificationLightningModule
from src.utils.logger import LOGGER

torch.set_float32_matmul_precision('high')


def train(config: ProjectConfig, id_task_preprocess: str) -> str:  # noqa: WPS210
    lightning.seed_everything(config.seed, workers=True)

    path_checkpoints = PATH_CHECKPOINTS / list(
        config.models.keys()
    )[config.active_model_index]

    os.makedirs(path_checkpoints, exist_ok=True)

    task = Task.init(
        project_name=PROJECT_NAME,
        task_name='Training task',
        task_type=Task.TaskTypes.training,
        auto_connect_arg_parser=False
    )

    task_preprocess = Task.get_task(task_id=id_task_preprocess)
    path_x_split = task_preprocess.artifacts['x_split'].get_local_copy()
    path_y_labels = task_preprocess.artifacts['y_labels'].get_local_copy()
    path_class_to_idx = task_preprocess.artifacts['class_to_idx'].get_local_copy()
    # path_class_weights = task_preprocess.artifacts['class_weights'].get_local_copy()

    with open(path_x_split, 'rb') as file_x:
        x_split = pickle.load(file_x)

    with open(path_y_labels, 'rb') as file_y:
        y_labels = pickle.load(file_y)

    with open(path_class_to_idx, 'r') as file_class:
        class_to_idx = json.load(file_class)

    # class_weights = torch.load(path_class_weights)

    data_module = ClassificationDataModule(config, is_preprocessed=True)
    data_module.x_split = x_split
    data_module.y_labels = y_labels
    data_module._class_to_index = class_to_idx
    # data_module._pos_weight = class_weights

    callbacks = [
        ModelSummary(),
        BatchVisualize(config.dataset, every_n_epochs=5),
        ClearMLTrack(config, label_enumeration=class_to_idx),
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

    task.upload_artifact('path_model', artifact_object=trainer.checkpoint_callback.best_model_path)

    return task.id
