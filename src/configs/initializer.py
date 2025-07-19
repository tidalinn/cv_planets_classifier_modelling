from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.configs.project_config import ProjectConfig


def init_config(path: str) -> 'ProjectConfig':
    initialize_config_dir(config_dir=str(path), version_base=None)
    config = compose(config_name='project')
    config_dict = OmegaConf.to_container(config, resolve=True)
    return ProjectConfig(**config_dict)
