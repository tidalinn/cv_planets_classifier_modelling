import lightning
from clearml import Task

from src.configs import ProjectConfig
from src.constants import PROJECT_NAME
from src.data.data_module import ClassificationDataModule


def preprocess(config: ProjectConfig) -> str:
    lightning.seed_everything(config.seed, workers=True)

    task = Task.init(
        project_name=PROJECT_NAME,
        task_name='Preprocessing',
        task_type=Task.TaskTypes.data_processing,
        auto_connect_arg_parser=False
    )

    data_module = ClassificationDataModule(config)
    data_module.prepare_data()
    data_module.process_data()
    data_module.setup(stage='test')

    task.upload_artifact('x_split', artifact_object=data_module.x_split)
    task.upload_artifact('y_labels', artifact_object=data_module.y_labels)
    task.upload_artifact('class_to_idx', artifact_object=data_module.class_to_idx)
    # task.upload_artifact('class_weights', artifact_object=data_module.class_weights)

    return task.id
