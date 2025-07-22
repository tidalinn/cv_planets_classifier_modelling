import json
import pickle

import lightning
import torch
from clearml import Task

from src.configs import ProjectConfig
from src.constants import PROJECT_NAME
from src.data.data_module import ClassificationDataModule
from src.model import ClassificationLightningModule
from src.utils.logger import LOGGER

torch.set_float32_matmul_precision('high')


def predict(config: ProjectConfig, id_task_preprocess: str, id_task_train: str):  # noqa: WPS210
    lightning.seed_everything(config.seed, workers=True)

    Task.init(
        project_name=PROJECT_NAME,
        task_name='Prediction task',
        task_type=Task.TaskTypes.inference
    )

    task_preprocess = Task.get_task(task_id=id_task_preprocess)
    task_train = Task.get_task(task_id=id_task_train)

    path_x_split = task_preprocess.artifacts['x_split'].get_local_copy()
    path_y_labels = task_preprocess.artifacts['y_labels'].get_local_copy()
    path_class_to_idx = task_preprocess.artifacts['class_to_idx'].get_local_copy()
    # path_class_weights = task_train.artifacts['class_weights'].get_local_copy()
    path_model = task_train.artifacts['path_model'].get_local_copy()

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

    model = ClassificationLightningModule.load_from_checkpoint(
        checkpoint_path=path_model,
        config=config,
        class_to_idx=data_module.class_to_idx
    )
    trainer = lightning.Trainer(**dict(config.trainer))

    LOGGER.info('Testing started...')
    trainer.test(model, datamodule=data_module)
    LOGGER.info('Testing finished')
