import argparse
import os
from pathlib import Path

from clearml import Dataset
from kaggle.api.kaggle_api_extended import KaggleApi

from src.constants import DATASET_NAME, PATH_DATASET, PROJECT_NAME
from src.utils.logger import LOGGER


def download_dataset(dataset: str, path: str):
    LOGGER.info('Downloading dataset...')

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset=dataset, path=path, unzip=True)

    LOGGER.info('Finished downloading')


def upload_to_clearml(path: str, project_name: str, dataset_name: str):
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=project_name,
    )

    LOGGER.info(f'Uploading dataset {dataset_name} from {path} to ClearML (project: {project_name})...')

    dataset.add_files(path)
    dataset.upload()
    dataset.finalize()

    LOGGER.info('Finished uploading')


def upload_dataset_to_clearml(
    kaggle_dataset: str,
    path: str,
    project_name: str,
    dataset_name: str
):
    try:  # noqa: WPS229
        Dataset.get(dataset_project=project_name, dataset_name=dataset_name)
        dataset_exists_in_clearml = True
    except Exception:
        dataset_exists_in_clearml = False

    if dataset_exists_in_clearml:
        LOGGER.info(f'Dataset {dataset_name} exists in ClearML (project: {project_name})')

    else:
        if path and Path(path).exists():
            upload_to_clearml(path, project_name, dataset_name)

        elif kaggle_dataset:
            download_dataset(kaggle_dataset, PATH_DATASET)
            os.remove(PATH_DATASET / f'{dataset_name}.zip')
            upload_to_clearml(PATH_DATASET, project_name, dataset_name)

        else:
            LOGGER.error('No kaggle_dataset or local path to dataset passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle_dataset', default=None)
    parser.add_argument('--path', default=None)
    parser.add_argument('--project_name', default=PROJECT_NAME)
    parser.add_argument('--dataset_name', default=DATASET_NAME)
    args = parser.parse_args()

    upload_dataset_to_clearml(args.kaggle_dataset, args.path, args.project_name, args.dataset_name)
