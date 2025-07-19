import logging

from clearml import PipelineDecorator

from src.configs.initializer import init_config
from src.constants import PATH_CONFIGS, PROJECT_NAME
from src.utils.logger import init_logger

logger = logging.getLogger(__name__)


@PipelineDecorator.component(cache=False)
def preprocess_component() -> str:
    from src.pipeline.preprocess import preprocess
    config = init_config(PATH_CONFIGS)
    path_output = preprocess(config)
    return path_output


@PipelineDecorator.component(cache=False)
def train_component(path_dataset: str) -> str:
    from src.pipeline.train import train
    config = init_config(PATH_CONFIGS)
    path_model = train(config, path_dataset)
    return path_model


@PipelineDecorator.component(cache=False)
def predict_component(path_model: str):
    from src.pipeline.predict import predict
    config = init_config(PATH_CONFIGS)
    predict(config, path_model)


@PipelineDecorator.pipeline(name='MultiStage ClearML Pipeline', project=PROJECT_NAME, version='1.0')
def run_pipeline():
    path_output = preprocess_component()
    path_model = train_component(path_output)
    predict_component(path_model)


if __name__ == '__main__':
    init_logger()
    PipelineDecorator.run_locally()
    run_pipeline()
