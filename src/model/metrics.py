from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelFBetaScore

from src.configs.project_config import MetricsConfig


def get_metrics(config: MetricsConfig) -> MetricCollection:
    return MetricCollection(
        {
            'f2': MultilabelFBetaScore(**config.common, **config.f2)
        }
    )
