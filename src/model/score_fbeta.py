from typing import Optional, Union

import numpy as np
import torch
from sklearn.metrics import fbeta_score
from torchmetrics import Metric


class FBetaScore(Metric):
    full_state_update = True

    def __init__(
        self,
        num_classes: int,
        beta: float = 2.0,
        average: float = 'samples',
        threshold: Optional[Union[np.ndarray, float]] = None
    ):
        super().__init__()

        self.num_classes = num_classes
        self.beta = beta
        self.average = average
        self.threshold = threshold

        self.add_state('probs', default=[], dist_reduce_fx='cat')
        self.add_state('targets', default=[], dist_reduce_fx='cat')

    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        self.probs.append(probs.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> torch.Tensor:  # noqa: WPS210
        probs = torch.cat(self.probs, dim=0)
        targets = torch.cat(self.targets, dim=0)

        if self.threshold is None:
            best_thresholds = []

            for index in range(self.num_classes):
                best_threshold = 0.5
                best_score = 0.0  # noqa: WPS358

                for threshold in np.arange(0.1, 0.9, 0.1):  # noqa: WPS432
                    preds = (
                        probs[:, index] > threshold  # noqa: WPS478
                    ).int()

                    score = fbeta_score(
                        y_true=targets[:, index],  # noqa: WPS478
                        y_pred=preds,
                        beta=self.beta,
                        zero_division=0.0  # noqa: WPS358
                    )

                    if score > best_score:
                        best_score = score  # noqa: WPS220
                        best_threshold = threshold  # noqa: WPS220

                best_thresholds.append(best_threshold)

            best_thresholds = torch.tensor(best_thresholds)

        else:
            best_thresholds = self.threshold

        return torch.tensor(
            fbeta_score(
                y_true=targets.numpy(),
                y_pred=(probs > best_thresholds).int().numpy(),
                beta=self.beta,
                average=self.average,
                zero_division=0.0  # noqa: WPS358
            )
        )
