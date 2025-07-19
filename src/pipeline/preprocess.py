import logging

from clearml import Task

from src.configs import ProjectConfig
from src.constants import PROJECT_NAME
from src.data.data_module import ClassificationDataModule

logger = logging.getLogger(__name__)


def preprocess(config: ProjectConfig) -> str:
    Task.init(
        project_name=PROJECT_NAME,
        task_name='Preprocessing',
        task_type=Task.TaskTypes.data_processing
    )

    data_module = ClassificationDataModule(config)
    data_module.prepare_data()
    data_module.process_data()
    data_module.setup(stage='test')

    return str(data_module.path_dataset)
