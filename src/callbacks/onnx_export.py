import os
from contextlib import contextmanager
from pathlib import Path

import onnx
import onnxruntime
import onnxsim
import torch
from lightning import Callback, Trainer
from pytorch_lightning.loops import _EvaluationLoop
from pytorch_lightning.trainer.states import RunningStage

from src.configs import ProjectConfig
from src.executables.select_best_model import select_best_model
from src.model.lightning_module import ClassificationLightningModule
from src.model.metrics import get_metrics
from src.utils.logger import LOGGER


class ONNXExport(Callback):

    def __init__(self, config: ProjectConfig, path_checkpoints: Path):
        super().__init__()
        self.config = config
        self.path_checkpoints = path_checkpoints

    def on_train_end(self, trainer: Trainer, lightning_module: ClassificationLightningModule):
        path_checkpoint = select_best_model(self.path_checkpoints, suffix='.ckpt')
        LOGGER.info(f'Selected checkpoint {path_checkpoint}')

        model = ClassificationLightningModule.load_from_checkpoint(path_checkpoint, map_location='cpu').model
        model_file = os.path.basename(path_checkpoint).replace('.ckpt', '.onnx')
        path_onnx = self.path_checkpoints / model_file

        with self._set_eval_mode(model):
            self._convert_to_onnx(
                model=model,
                path=path_onnx,
                input_h=self.config.dataset.img_size[1],
                input_w=self.config.dataset.img_size[0],
            )

        self._validate_onnx(trainer, lightning_module, path_onnx)

    def _convert_to_onnx(self, model: torch.nn.Module, path: str, input_h: int, input_w: int):
        dummy_input = torch.randn(
            self.config.onnx.dummy_batch_size, 3, input_h, input_w,
            dtype=torch.float32,
            device='cpu'
        )

        with torch.no_grad():
            model(dummy_input)

        LOGGER.info('Exporting model to ONNX...')

        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=path,
            verbose=False,
            export_params=True,
            opset_version=12,  # noqa: WPS432
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': [0],
                'output': [0]
            },
        )

        model = onnx.load(path)
        onnx.checker.check_model(model)

        model, _ = onnxsim.simplify(
            model=model,
            dynamic_input_shape=False,
            overwrite_input_shapes={
                'input': [-1, 3, input_h, input_w]
            },
        )

        onnx.save(model, path)
        LOGGER.info(f'Saved model to {path}')

    @contextmanager
    def _set_eval_mode(self, net: torch.nn.Module):
        try:  # noqa: WPS229
            net.eval()
            yield net
        finally:
            if net.training:
                net.train()

    def _validate_onnx(  # noqa: WPS210
        self,
        trainer: Trainer,
        lightning_module: ClassificationLightningModule,
        path: str
    ):
        dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(dataloader))
        _, labels = batch

        metrics = get_metrics(labels.shape[1])
        metrics.reset()

        ort_session = onnxruntime.InferenceSession(
            path_or_bytes=path,
            providers=self.config.onnx.providers,
        )

        for images, targets in dataloader:
            ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)

            logits = torch.tensor(ort_outputs[0])
            probs = torch.sigmoid(logits)
            metrics(probs, targets)

        _EvaluationLoop._print_results(
            results=[metrics.compute()],
            stage=RunningStage.TESTING
        )
