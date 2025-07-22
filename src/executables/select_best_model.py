import os
import shutil
from glob import glob
from pathlib import Path

from src.constants import PATH_BEST, PATH_CHECKPOINTS
from src.utils.logger import LOGGER


def select_best_model(path: Path, suffix: str = '.onnx', depth: str = '*') -> str:  # noqa: WPS210
    models = glob(str(path / f'{depth}{suffix}'), recursive=True)

    if not models:
        raise FileNotFoundError(f'No models with suffix "{suffix}" found in {path}')

    best_score = 0
    best_model = None

    for model in models:
        score_str = (
            model.replace(suffix, '')
                 .split('valid_f2=')[-1]
                 .split('-')[0]
        )
        score = float(score_str)

        if score > best_score:
            best_score = score
            best_model = model

    return best_model


if __name__ == '__main__':
    if PATH_BEST.exists():
        shutil.rmtree(PATH_BEST)

    os.makedirs(PATH_BEST, exist_ok=True)

    path_file = PATH_BEST / 'classificator.onnx'
    best_model = select_best_model(PATH_CHECKPOINTS, depth='**/*')

    LOGGER.info(f'Selected best model at {best_model}')
    shutil.copy(best_model, path_file)
    LOGGER.info(f'Copied best model to {path_file}')
