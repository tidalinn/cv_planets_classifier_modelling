import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from clearml import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.configs import ProjectConfig
from src.constants import (
    DATASET_NAME,
    PATH_DATASET,
    PATH_IMAGES_TEST,
    PATH_IMAGES_TRAIN,
    PROJECT_NAME,
)
from src.data.dataset import ClassificationDataset
from src.data.preprocessing import (
    encode_labels,
    get_class_to_index,
    get_pos_weight,
    split_data,
)
from src.data.transforms import get_train_transforms, get_valid_transforms
from src.utils.logger import LOGGER


class ClassificationDataModule(LightningDataModule):  # noqa: WPS230

    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.config = config

        self._transform_train = get_train_transforms(config.dataset)
        self._transform_valid = get_valid_transforms(config.dataset)

        self.save_hyperparameters(logger=False)

        self.path_dataset = PATH_DATASET
        self.initialized = False

        self.data_train: Optional[ClassificationDataset] = None
        self.data_val: Optional[ClassificationDataset] = None
        self.data_test: Optional[ClassificationDataset] = None

    @property
    def class_to_idx(self) -> Dict[str, int]:
        if not self.initialized:
            self.prepare_data()
            self.process_data()
            self.setup('test')

        return self._class_to_index

    @property
    def class_weights(self) -> torch.Tensor:
        return self._pos_weight

    def prepare_data(self):
        if 'planet' in os.listdir(PATH_DATASET) and PATH_DATASET.exists():
            LOGGER.info(f'Taking dataset from {PATH_DATASET}')
            return

        LOGGER.info(f'Downloading dataset {DATASET_NAME} from ClearML (project: {PROJECT_NAME})')

        self.path_dataset = Path(
            Dataset.get(
                dataset_project=PROJECT_NAME,
                dataset_name=DATASET_NAME
            ).get_local_copy()
        )
        LOGGER.info('Downloaded dataset')

    def process_data(self):
        LOGGER.info('Preprocessing data...')

        self.x_split = split_data(self.config, self.path_dataset)
        self.y_encoded = encode_labels(self.x_split['x_train'], self.x_split['x_valid'])

        self._class_to_index = get_class_to_index(self.path_dataset)
        self._pos_weight = get_pos_weight(self.path_dataset, y_train=self.y_encoded['y_train'])

        LOGGER.info('Finished preprocessing')

    def setup(self, stage: str):
        if stage == 'fit':
            self.data_train = ClassificationDataset(
                dataframe=self.x_split['x_train'],
                labels=self.y_encoded['y_train'],
                path=self.path_dataset / PATH_IMAGES_TRAIN,
                transform=self._transform_train
            )

            self.data_valid = ClassificationDataset(
                dataframe=self.x_split['x_valid'],
                labels=self.y_encoded['y_valid'],
                path=self.path_dataset / PATH_IMAGES_TRAIN,
                transform=self._transform_valid
            )

        elif stage == 'test':
            self.data_test = ClassificationDataset(
                dataframe=self.x_split['x_test'],
                labels=np.zeros((
                    self.x_split['x_test'].shape[0],
                    self.config.num_classes
                )),
                path=self.path_dataset / PATH_IMAGES_TEST,
                transform=self._transform_valid
            )

        self.initialized = True

    def train_dataloader(self) -> 'DataLoader':
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.dataset.persistent_workers,
            shuffle=True
        )

    def val_dataloader(self) -> 'DataLoader':
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.dataset.persistent_workers,
            shuffle=False
        )

    def test_dataloader(self) -> 'DataLoader':
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.dataset.persistent_workers,
            shuffle=False
        )
