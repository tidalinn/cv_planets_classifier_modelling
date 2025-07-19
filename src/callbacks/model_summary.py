from lightning import Callback, LightningModule, Trainer
from torchinfo import summary


class ModelSummary(Callback):
    def on_train_start(self, trainer: Trainer, lightning_module: LightningModule):
        images = next(iter(trainer.train_dataloader))[0]
        images = images.to(lightning_module.device)

        summary(lightning_module.model, input_data=images)
