from typing import Any, Dict, List, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field


class BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=()
    )


class ModelConfig(BaseValidatedConfig):
    name: str
    pretrained: bool = Field(default=False)
    freeze_backbone: bool = Field(default=True)
    dropout: float = Field(default=0)
    hidden_dim: int
    kwargs: Dict[str, Any] = Field(default={})
    learning_rate: float = Field(default=1e-4)


class DatasetConfig(BaseValidatedConfig):
    img_size: Tuple[int, int]
    mean: List[float]
    std: List[float]
    batch_size: int = Field(default=32)
    data_split: List[float] = Field(default=[0.8, 0.2])
    num_workers: int = Field(default=0)
    pin_memory: bool = Field(default=False)
    persistent_workers: bool = Field(default=False)


class TrainerConfig(BaseValidatedConfig):
    accelerator: str = Field(default='gpu' if torch.cuda.is_available() else 'cpu')
    devices: int = Field(default=1)
    min_epochs: int = Field(default=1)
    max_epochs: int = Field(default=5)
    log_every_n_steps: int = Field(default=10)
    deterministic: bool = Field(default=False)
    overfit_batches: float = Field(default=1.0)
    logger: bool = Field(default=True)
    enable_progress_bar: bool = Field(default=True)


class OnnxConfig(BaseValidatedConfig):
    providers: List[str]
    logit_tolerance: float = Field(default=1e-4),
    dummy_batch_size: int = Field(default=10)


class MetricsConfig(BaseValidatedConfig):
    common: Dict[str, Any]
    f2: Dict[str, Any]


class ProjectConfig(BaseValidatedConfig):
    task: str
    num_classes: int
    active_model_index: int = Field(default=0)

    dataset: DatasetConfig
    models: Dict[str, ModelConfig]
    trainer: TrainerConfig
    onnx: OnnxConfig
    metrics: MetricsConfig
