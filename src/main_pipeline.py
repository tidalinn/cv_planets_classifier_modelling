from clearml import PipelineDecorator

from src.configs.initializer import init_config
from src.constants import PATH_CONFIGS, PROJECT_NAME


@PipelineDecorator.component(cache=False)
def preprocess_component() -> str:
    from src.pipeline.preprocess import preprocess
    config = init_config(PATH_CONFIGS)
    return preprocess(config)


@PipelineDecorator.component(cache=False)
def train_component(id_task_preprocess: str) -> str:
    from src.pipeline.train import train
    config = init_config(PATH_CONFIGS)
    return train(config, id_task_preprocess)


@PipelineDecorator.component(cache=False)
def predict_component(id_task_preprocess: str, id_task_train: str):
    from src.pipeline.predict import predict
    config = init_config(PATH_CONFIGS)
    return predict(config, id_task_preprocess, id_task_train)


@PipelineDecorator.pipeline(name='MultiStage ClearML Pipeline', project=PROJECT_NAME, version='1.0')
def run_pipeline():
    id_task_preprocess = preprocess_component()
    id_task_train = train_component(id_task_preprocess)
    predict_component(id_task_preprocess, id_task_train)


if __name__ == '__main__':
    PipelineDecorator.run_locally()
    run_pipeline()
