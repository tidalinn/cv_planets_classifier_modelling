from torchmetrics import MetricCollection

from src.model.score_fbeta import FBetaScore


def get_metrics(num_labels: int) -> MetricCollection:
    return MetricCollection(
        {
            'f2': FBetaScore(num_labels)
        }
    )
