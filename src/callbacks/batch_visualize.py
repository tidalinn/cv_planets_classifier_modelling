import numpy as np
import torch
from lightning import Callback, Trainer
from torchvision.utils import make_grid

from src.configs.project_config import DatasetConfig
from src.model.lightning_module import ClassificationLightningModule


class BatchVisualize(Callback):

    def __init__(self, config: DatasetConfig, every_n_epochs: int):
        super().__init__()
        self.config = config
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer: Trainer, lightning_module: ClassificationLightningModule):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        images = next(iter(trainer.train_dataloader))[0]
        visualizations = []

        for image in images:
            visualizations.append(
                self._denormalize(image)
            )

        grid = make_grid(visualizations, normalize=False)

        trainer.logger.experiment.add_image(
            'Batch preview',
            img_tensor=grid,
            global_step=trainer.global_step,
        )

    def _denormalize(self, image: torch.Tensor) -> np.ndarray:
        mean = torch.tensor(self.config.mean).view(-1, 1, 1)
        std = torch.tensor(self.config.std).view(-1, 1, 1)
        return image * std + mean
