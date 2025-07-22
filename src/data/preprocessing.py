from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from src.configs import ProjectConfig
from src.constants import PATH_FILE_TEST, PATH_FILE_TRAIN


def split_data(config: ProjectConfig, path: Path) -> Dict[str, pd.DataFrame]:
    data_train = pd.read_csv(path / PATH_FILE_TRAIN)
    data_train['tags'] = data_train.tags.str.split()

    (x_train, x_valid) = train_test_split(
        data_train,
        train_size=config.dataset.data_split[0],
        test_size=config.dataset.data_split[1],
        random_state=config.seed,
        shuffle=True
    )

    x_train = x_train.reset_index(drop=True)
    x_valid = x_valid.reset_index(drop=True)

    data_test_full = pd.read_csv(path / PATH_FILE_TEST)
    x_test = data_test_full[data_test_full.image_name.str.contains('test')].reset_index(drop=True)

    return {
        'x_train': x_train,
        'x_valid': x_valid,
        'x_test': x_test
    }


def encode_labels(x_train: pd.DataFrame, x_valid: pd.DataFrame) -> Dict[str, np.ndarray]:
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(x_train.tags.values)
    y_valid = mlb.transform(x_valid.tags.values)

    return {
        'y_train': y_train,
        'y_valid': y_valid
    }


def get_tags(path: Path) -> np.ndarray:
    data_train = pd.read_csv(path / PATH_FILE_TRAIN)
    data_train['tags'] = data_train.tags.str.split()

    return data_train.tags.explode().unique()


def get_class_to_index(path: Path) -> Dict[str, int]:
    tags = get_tags(path)
    return {tag: index for index, tag in enumerate(sorted(tags))}


def get_pos_weight(path: Path, y_train: np.ndarray) -> torch.Tensor:
    tags = get_tags(path)

    return torch.tensor(
        data=(len(y_train) - len(tags)) / (len(tags) + 1e-5),  # noqa: WPS432
        dtype=torch.float32
    )
