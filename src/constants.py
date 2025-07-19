import os
from pathlib import Path

PROJECT_NAME = 'planets-classifier'
DATASET_NAME = 'planets-dataset'

PATH_PROJECT = Path(__file__).resolve().parents[1]
PATH_PROJECT_ROOT = Path(os.getenv('FP_MODELLING_ROOT', PATH_PROJECT))

PATH_CONFIGS = PATH_PROJECT_ROOT / 'configs'

PATH_LOGS = 'logs'
PATH_DATASET = 'dataset'

PATH_OUTPUT = 'output'
PATH_CHECKPOINTS = f'{PATH_OUTPUT}/checkpoints'
PATH_ENCODER = f'{PATH_OUTPUT}/encoder'
PATH_BEST = f'{PATH_OUTPUT}/best'

PATH_FILE_TRAIN = 'planet/planet/train_classes.csv'
PATH_FILE_TEST = 'planet/planet/sample_submission.csv'
PATH_IMAGES_TRAIN = 'planet/planet/train-jpg'
PATH_IMAGES_TEST = 'planet/planet/test-jpg'
