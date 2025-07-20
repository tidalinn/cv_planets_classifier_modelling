import glob
import logging
import os
import shutil

from src.constants import PATH_BEST, PATH_CHECKPOINTS
from src.utils.logger import init_logger

logger = logging.getLogger(__name__)


def select_best_model():
    if os.path.exists(PATH_BEST):
        shutil.rmtree(PATH_BEST)

    os.makedirs(PATH_BEST, exist_ok=True)

    path_file = os.path.join(PATH_BEST, 'classificator.onnx')
    files = glob.glob(os.path.join(PATH_CHECKPOINTS, '**/*.onnx'), recursive=True)

    if len(files) == 0:
        logger.error('No models found')
        return

    best_f2 = 0
    best_file = None

    for file_onnx in files:
        file_edited = file_onnx.replace('.onnx', '')
        file_edited = file_edited.split('valid_f2=')

        valid_f2 = float(file_edited[-1])

        if valid_f2 > best_f2:
            best_f2 = valid_f2
            best_file = file_onnx

    logger.info(f'Selected best model at {best_file}')
    shutil.copy(best_file, path_file)
    logger.info(f'Copied best model to {path_file}')


if __name__ == '__main__':
    init_logger()
    select_best_model()
