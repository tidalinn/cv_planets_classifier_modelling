import os
from pathlib import Path

import torch

PROJECT_NAME = 'planets-classifier'
DATASET_NAME = 'planets-dataset'

DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'

PATH_PROJECT = Path(__file__).resolve().parents[1]
PATH_PROJECT_ROOT = Path(os.getenv('FP_MODELLING_ROOT', PATH_PROJECT))

PATH_CONFIGS = PATH_PROJECT_ROOT / 'configs'

PATH_LOGS = Path('logs')
PATH_DATASET = Path('dataset')

PATH_OUTPUT = Path('output')
PATH_CHECKPOINTS = PATH_OUTPUT / 'checkpoints'
PATH_BEST = PATH_OUTPUT / 'best'

PATH_PLANET = Path('planet') / 'planet'
PATH_FILE_TRAIN = PATH_PLANET / 'train_classes.csv'
PATH_FILE_TEST = PATH_PLANET / 'sample_submission.csv'
PATH_IMAGES_TRAIN = PATH_PLANET / 'train-jpg'
PATH_IMAGES_TEST = PATH_PLANET / 'test-jpg'
