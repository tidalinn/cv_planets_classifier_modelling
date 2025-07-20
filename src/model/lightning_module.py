
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn  # noqa: WPS301
from lightning import LightningModule
from timm import create_model
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric
from torchmetrics.classification import MultilabelFBetaScore

from src.configs import ProjectConfig
from src.model.metrics import get_metrics

logger = logging.getLogger(__name__)


class ClassificationLightningModule(LightningModule):

    def __init__(self, config: ProjectConfig, class_to_idx: Dict[str, int], class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config

        if class_weights is None:
            self._loss = nn.BCEWithLogitsLoss()
        else:
            self._loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)

        self._loss_train = MeanMetric()
        self._loss_valid = MeanMetric()

        self.config_model = list(self.config.models.values())[self.config.active_model_index]
        self.model = self._build_model(num_classes=len(class_to_idx))

        self.config.metrics.common.update({'num_labels': len(class_to_idx)})
        metrics = get_metrics(self.config.metrics)

        self._metrics_train = metrics.clone(prefix='train_')
        self._metrics_valid = metrics.clone(prefix='valid_')

        self._logits_valid = []
        self._targets_valid = []

        self.save_hyperparameters()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        images, targets = batch
        logits = self(images)
        loss = self._loss(logits, targets)

        self._loss_train(loss)
        self._metrics_train(torch.sigmoid(logits), targets)

        self.log('loss_train_step', loss, on_step=True, prog_bar=True)

        return {'loss': loss}

    def on_train_epoch_end(self):
        self.log('loss_train_epoch', self._loss_train.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self._metrics_train.compute(), on_epoch=True, prog_bar=True, logger=True)
        self._metrics_train.reset()

    def validation_step(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        images, targets = batch
        logits = self(images)
        loss = self._loss(logits, targets)

        self.log('loss_valid_step', loss, on_step=True, prog_bar=True)

        self._loss_valid(loss)
        self._metrics_valid(torch.sigmoid(logits), targets)

        self._logits_valid.append(logits.detach().cpu())
        self._targets_valid.append(targets.detach().cpu())

        return {'loss': loss}

    def on_validation_epoch_end(self):
        self.log('loss_valid_epoch', self._loss_valid.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self._metrics_valid.compute(), on_epoch=True, prog_bar=True, logger=True)
        self._metrics_valid.reset()

        self._search_threshold()

    def test_step(self, batch: List[torch.Tensor]) -> torch.Tensor:
        images, _ = batch
        logits = self(images)

        return logits

    def configure_optimizers(self) -> Dict[str, Any]:
        # params = filter(lambda x: x.requires_grad, self.model.parameters())
        model_parameters = self.model.parameters()

        optimizer = Adam(model_parameters, lr=self.config_model.learning_rate)
        # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=3, factor=0.5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_f2',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _build_model(self, num_classes: int) -> nn.Module:
        model = create_model(
            model_name=self.config_model.name,
            pretrained=self.config_model.pretrained,
            num_classes=self.config.num_classes,
            **self.config_model.kwargs
        )

        if self.config_model.freeze_backbone:
            for model_parameters in model.parameters():
                model_parameters.requires_grad = False

        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        model.fc = nn.Sequential(
            nn.Dropout(self.config_model.dropout),
            nn.Linear(model.get_classifier().in_features, self.config_model.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config_model.hidden_dim, num_classes)
        )

        for fc_parameters in model.fc.parameters():
            fc_parameters.requires_grad = True

        return model

    def _search_threshold(self):
        logits = torch.cat(self._logits_valid, dim=0)
        probs = torch.sigmoid(logits)
        targets = torch.cat(self._targets_valid, dim=0)

        prob_min = probs.min().item()
        prob_mean = probs.mean().item()
        prob_max = probs.max().item()

        logger.info(f'Probabilities | min {prob_min:.4f} | mean {prob_mean:.4f} | max {prob_max:.4f}')

        thresholds = np.arange(0.1, 0.95, 0.1)
        best_threshold = 0.5
        best_score = 0.0  # noqa: WPS358

        for threshold in thresholds:
            f2 = MultilabelFBetaScore(**self.config.metrics.common, **self.config.metrics.f2)
            f2.threshold = threshold

            score = f2(probs, targets).item()

            if score > best_score:
                best_score = score
                best_threshold = threshold

        logger.info(f'Best threshold: {best_threshold}')
