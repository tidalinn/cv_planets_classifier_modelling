# Эксперименты с моделями

№ | model_index | Модель | F2 | Optimizer | Scheduler | Количество эпох | Слои | Model
-|-|-|-|-|-|-|-|-
1 | 0 | ResNet50 | 0.438 | Adam | StepLR | 5 | AdaptiveAvgPool2d<br>Linear<br>ReLU<br>Dropout<br>Linear | [experiment.onnx](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/e8afb7c40e2b434a877f2e7fda6d633e/output/execution)
2 | 1 | ResNet50 | 0.581 | Adam | StepLR | 5 | AdaptiveAvgPool2d<br>Linear<br>ReLU<br>Dropout<br>Linear | [experiment.onnx](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/7c7b8e7d28c3483a9b810aed9c2ccc57/output/execution)
3 | 2 | ResNet50 | 0.698 | Adam | CosineAnnealingWarmRestarts | 15 | AdaptiveAvgPool2d<br>BatchNorm<br>ReLU<br>Dropout<br>Linear<br>ReLU<br>Linear | [experiment.onnx](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/bbab27f5e9554fffa2788c32a58280b3/output/execution)
4 | 3 | ResNet18 | 0.541 | Adam | CosineAnnealingWarmRestarts | 5 | AdaptiveAvgPool2d<br>BatchNorm<br>ReLU<br>Dropout<br>Linear<br>ReLU<br>Linear | [experiment.onnx](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/7ab055d637294235b4093eb9d8925aad/output/execution)
5 | 4 | ResNet50 | 0.702 | Adam | ReduceLROnPlateau | 15 | Dropout<br>Linear<br>ReLU<br>Linear | [experiment.onnx](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/7c5d539620c14f9c8b82f29d336d40c4/output/execution)

## Замечания

* Балансирование классов плохо влияет на метрику
