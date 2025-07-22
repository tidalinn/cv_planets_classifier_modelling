
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn  # noqa: WPS301
from lightning import LightningModule
from timm import create_model
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric

from src.configs import ProjectConfig
from src.model.metrics import get_metrics


class ClassificationLightningModule(LightningModule):

    def __init__(
        self,
        config: ProjectConfig,
        class_to_idx: Dict[str, int],
        class_weights: Optional[torch.Tensor] = None
    ):
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

        metrics = get_metrics(len(class_to_idx))
        self._metrics_train = metrics.clone(prefix='train_')
        self._metrics_valid = metrics.clone(prefix='valid_')

        self.save_hyperparameters()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        images, targets = batch
        logits = self(images)
        loss = self._loss(logits, targets)

        self._loss_train(loss)
        self.log('loss_train_step', loss, on_step=True, prog_bar=True)

        self._metrics_train(torch.sigmoid(logits), targets)

        return {'loss': loss}

    def on_train_epoch_end(self):
        self.log(
            'loss_train_epoch',
            self._loss_train.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        self.log_dict(self._metrics_train.compute(), on_epoch=True, prog_bar=True, logger=True)

        self._metrics_train.reset()

    def validation_step(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        images, targets = batch
        logits = self(images)
        loss = self._loss(logits, targets)

        self._loss_valid(loss)
        self.log('loss_valid_step', loss, on_step=True, prog_bar=True)

        self._metrics_valid(torch.sigmoid(logits), targets)

        return {'loss': loss}

    def on_validation_epoch_end(self):
        self.log(
            'loss_valid_epoch',
            self._loss_valid.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        self.log_dict(self._metrics_valid.compute(), on_epoch=True, prog_bar=True, logger=True)

        self._metrics_valid.reset()

    def test_step(self, batch: List[torch.Tensor]) -> torch.Tensor:
        images, _ = batch
        logits = self(images)

        return logits

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.config_model.freeze_backbone:
            model_parameters = filter(lambda module: module.requires_grad, self.model.parameters())
        else:
            model_parameters = self.model.parameters()

        optimizer = Adam(model_parameters, lr=self.config_model.learning_rate)
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
            num_classes=self.config.num_classes
        )

        if self.config_model.freeze_backbone:
            for model_parameters in model.parameters():
                model_parameters.requires_grad = False

        if hasattr(model.get_classifier(), 'in_features'):
            in_features = model.get_classifier().in_features
        else:
            in_features = model.get_classifier()[0].in_features

        model.fc = nn.Sequential(
            nn.Dropout(self.config_model.dropout),
            nn.Linear(in_features, self.config_model.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config_model.hidden_dim, num_classes)
        )

        for fc_parameters in model.fc.parameters():
            fc_parameters.requires_grad = True

        return model
