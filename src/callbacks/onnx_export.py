import logging
import os
from contextlib import contextmanager
from glob import glob

import onnx
import onnxruntime
import onnxsim
import torch
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loops import _EvaluationLoop
from pytorch_lightning.trainer.states import RunningStage

from src.configs import ProjectConfig
from src.constants import PATH_CHECKPOINTS
from src.model.lightning_module import ClassificationLightningModule
from src.model.metrics import get_metrics

logger = logging.getLogger(__name__)


class ONNXExport(Callback):

    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.config = config

        self.path_checkpoints = os.path.join(
            PATH_CHECKPOINTS,
            list(self.config.models.keys())[self.config.active_model_index]
        )
        os.makedirs(self.path_checkpoints, exist_ok=True)

    def on_train_end(self, trainer: Trainer, lightning_module: ClassificationLightningModule):
        path_checkpoint = self._select_checkpoint_for_export(trainer)
        logger.info(f'Selected checkpoint {path_checkpoint}')

        model = ClassificationLightningModule.load_from_checkpoint(path_checkpoint, map_location='cpu').model
        model_file = os.path.basename(path_checkpoint).replace('.ckpt', '.onnx')
        path_onnx = os.path.join(self.path_checkpoints, model_file)

        with self._set_eval_mode(model):
            self._convert_to_onnx(
                model=model,
                path=path_onnx,
                input_h=self.config.dataset.img_size[1],
                input_w=self.config.dataset.img_size[0],
            )

        self._validate_onnx(trainer, lightning_module)

    def _select_checkpoint_for_export(self, trainer: Trainer) -> str:
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if os.path.isfile(callback.best_model_path):
                    return callback.best_model_path

        path_checkpoint = os.path.join(trainer.log_dir, 'best_checkpoint.ckpt')
        trainer.save_checkpoint(path_checkpoint)
        return path_checkpoint

    def _convert_to_onnx(self, model: torch.nn.Module, path: str, input_h: int, input_w: int):
        dummy_input = torch.randn(
            self.config.onnx.dummy_batch_size, 3, input_h, input_w,
            dtype=torch.float32,
            device='cpu'
        )

        with torch.no_grad():
            model(dummy_input)

        logger.info('Exporting model to ONNX...')

        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=path,
            verbose=False,
            export_params=True,
            opset_version=12,
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
        logger.info(f'Saved model to {path}')

    @contextmanager
    def _set_eval_mode(self, net: torch.nn.Module):
        try:  # noqa: WPS229
            net.eval()
            yield net
        finally:
            if net.training:
                net.train()

    def _validate_onnx(self, trainer: Trainer, lightning_module: ClassificationLightningModule):
        path = self._get_last_model()
        logger.info(f'Evaluating ONNX model at {path}')

        dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(dataloader))
        _, labels = batch

        self.config.metrics.common.update({'num_labels': labels.shape[1]})

        metrics = get_metrics(self.config.metrics)
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

        eval_results = metrics.compute()
        _EvaluationLoop._print_results([eval_results], RunningStage.TESTING)

    def _get_last_model(self) -> str:
        models = glob(os.path.join(self.path_checkpoints, '*.onnx'))

        if not models:
            text = f'No ONNX models found in {self.path_checkpoints}'
            logger.error(text)
            raise FileNotFoundError(text)

        models.sort(key=os.path.getmtime, reverse=True)
        return models[0]
