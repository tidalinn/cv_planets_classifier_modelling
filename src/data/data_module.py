import logging
import os
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from clearml import Dataset
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader

from src.configs import ProjectConfig
from src.constants import (
    DATASET_NAME,
    PATH_DATASET,
    PATH_ENCODER,
    PATH_FILE_TEST,
    PATH_FILE_TRAIN,
    PATH_IMAGES_TEST,
    PATH_IMAGES_TRAIN,
    PROJECT_NAME,
)
from src.data.dataset import ClassificationDataset
from src.data.transforms import get_train_transforms, get_valid_transforms

logger = logging.getLogger(__name__)


class ClassificationDataModule(LightningDataModule):

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

        os.makedirs(PATH_ENCODER, exist_ok=True)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        if not self.initialized:
            self.prepare_data()
            self.process_data()
            self.setup('test')

        return self._class_to_idx

    @property
    def class_weights(self) -> torch.Tensor:
        return self._pos_weight

    def prepare_data(self):
        if 'planet' in os.listdir(self.path_dataset) and os.path.exists(self.path_dataset):
            logger.info(f'Taking dataset from {self.path_dataset}')
            return

        logger.info(f'Downloading dataset {DATASET_NAME} from ClearML (project: {PROJECT_NAME})')

        self.path_dataset = Dataset.get(dataset_project=PROJECT_NAME, dataset_name=DATASET_NAME).get_local_copy()
        logger.info('Downloaded dataset')

    def process_data(self):
        logger.info('Preprocessing data...')

        self.data_train = pd.read_csv(os.path.join(self.path_dataset, PATH_FILE_TRAIN))
        self.data_train['tags'] = self.data_train.tags.str.split()

        tags = self.data_train.tags.explode().unique()
        self._class_to_idx = {tag: index for index, tag in enumerate(sorted(tags))}

        X_split = train_test_split(
            self.data_train,
            train_size=self.config.dataset.data_split[0],
            test_size=self.config.dataset.data_split[1]
        )

        self.X_train = X_split[0]
        self.X_valid = X_split[1]

        self.X_train = self.X_train.reset_index(drop=True)
        self.X_valid = self.X_valid.reset_index(drop=True)

        mlb = MultiLabelBinarizer()
        self.y_train = mlb.fit_transform(self.X_train.tags.values)
        joblib.dump(mlb, os.path.join(PATH_ENCODER, 'mlb.pkl'))
        self.y_valid = mlb.transform(self.X_valid.tags.values)

        total_tags = self.y_train.sum(axis=0)
        self._pos_weight = torch.tensor(
            data=(len(self.y_train) - total_tags) / (total_tags + 1e-5),
            dtype=torch.float32
        )

        data_test_full = pd.read_csv(os.path.join(self.path_dataset, PATH_FILE_TEST))
        self.X_test = data_test_full[data_test_full.image_name.str.contains('test')].reset_index(drop=True)
        self.X_test_additional = data_test_full[data_test_full.image_name.str.contains('file')].reset_index(drop=True)

        logger.info('Finished preprocessing')

    def setup(self, stage: str):
        if stage == 'fit':
            self.data_train = ClassificationDataset(
                dataframe=self.X_train,
                labels=self.y_train,
                path=os.path.join(self.path_dataset, PATH_IMAGES_TRAIN),
                transform=self._transform_train
            )

            self.data_valid = ClassificationDataset(
                dataframe=self.X_valid,
                labels=self.y_valid,
                path=os.path.join(self.path_dataset, PATH_IMAGES_TRAIN),
                transform=self._transform_valid
            )

        elif stage == 'test':
            self.data_test = ClassificationDataset(
                dataframe=self.X_test,
                labels=np.zeros((self.X_test.shape[0], self.config.num_classes)),
                path=os.path.join(self.path_dataset, PATH_IMAGES_TEST),
                transform=self._transform_valid
            )

            self.initialized = True

    def train_dataloader(self) -> 'DataLoader':
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            persistent_workers=self.config.dataset.persistent_workers,
            shuffle=True
        )

    def val_dataloader(self) -> 'DataLoader':
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            persistent_workers=self.config.dataset.persistent_workers,
            shuffle=False
        )

    def test_dataloader(self) -> 'DataLoader':
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            persistent_workers=self.config.dataset.persistent_workers,
            shuffle=False
        )
