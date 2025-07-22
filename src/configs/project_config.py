from typing import Dict, List, Tuple

from pydantic import BaseModel, ConfigDict


class BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=()
    )


class ModelConfig(BaseValidatedConfig):
    name: str
    pretrained: bool
    freeze_backbone: bool
    dropout: float
    hidden_dim: int
    learning_rate: float
    memory_efficient: bool


class DatasetConfig(BaseValidatedConfig):
    img_size: Tuple[int, int]
    mean: List[float]
    std: List[float]
    batch_size: int
    data_split: List[float]
    num_workers: int
    persistent_workers: bool


class TrainerConfig(BaseValidatedConfig):
    devices: int
    min_epochs: int
    max_epochs: int
    log_every_n_steps: int
    deterministic: bool
    overfit_batches: float
    logger: bool
    enable_progress_bar: bool


class OnnxConfig(BaseValidatedConfig):
    providers: List[str]
    logit_tolerance: float
    dummy_batch_size: int


class ProjectConfig(BaseValidatedConfig):
    task: str
    num_classes: int
    active_model_index: int
    seed: int

    dataset: DatasetConfig
    models: Dict[str, ModelConfig]
    trainer: TrainerConfig
    onnx: OnnxConfig
