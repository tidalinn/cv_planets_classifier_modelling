# Структура обучения

Компонент | Реализация | Описание
-|-|-
Тип задачи | | Multilabel-классификация
Модель | `torchvision.models.resnet50` | - Предобученная нейросеть<br>- С замороженными параметрами<br>- С переопределённым классификатором
Метрика | $ F_{\beta} = \frac{(1 + \beta^2)  \cdot precision \cdot recall}{(\beta^2 \cdot precision) + recall} $<br><br>`sklearn.metrics.fbeta_score` | Гармоническое среднее (с акцентом на Recall) между:<br>- Precision (точность) - доля правильно предсказанных меток $ \frac{TP}{TP + FP} $<br>- Recall (полнота) - доля предсказанных меток, которые действительно правильные $ \frac{TP}{TP + FN} $
Функция потерь | `torch.nn.BCEWithLogitsLoss` | Объединяет:<br>- `torch.nn.Sigmoid` - преобразовывает выходные значения в диапазон `[0, 1]`<br>- `torch.nn.BCELoss` - измеряет разницу между истинными вероятностями и предсказанными

# Эксперименты с моделями

active_model_index | Модель | F2 | pos_weight | Optimizer | Scheduler | Количество эпох | Слои | ClearML task
-|-|-|-|-|-|-|-|-
0 | ResNet50 | 0.901 | + | `torch.optim.Adam` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | 5 | Dropout<br>Linear<br>ReLU<br>Linear | [experiment](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/faa86310b0da4608947e66da9a085254/output/execution)
0 | ResNet50 | 0.924 | - | `torch.optim.Adam` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | 5 | Dropout<br>Linear<br>ReLU<br>Linear | [experiment](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/42724d4697c6482bbf39edc7aba2ad0b/output/execution)
1 | ResNet50 | 0.922 | - | `torch.optim.Adam` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | 5 | Dropout<br>Linear<br>ReLU<br>Linear | [experiment](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/ec10e73c7cdd4576a3a57b64bbdd9b51/output/execution)
2 | EfficientNet B0 | 0.921 | - | `torch.optim.Adam` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | 5 | Dropout<br>Linear<br>ReLU<br>Linear | [experiment](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/bbfcac8e672f489f863f733dc43b4945/output/execution)
3 | EfficientNet B1 | 0.917 | - | `torch.optim.Adam` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | 5 | Dropout<br>Linear<br>ReLU<br>Linear | [experiment](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/03fd6cda6d024b24862e9ee0eaf1d863/output/execution)
4 | ResNet18 | 0.920 | - | `torch.optim.Adam` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | 5 | Dropout<br>Linear<br>ReLU<br>Linear | [experiment](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/94f9b4c4e2a24687a3d80fc9a4ff77a5/output/execution)
5 | ResNet50 | 0.878 | - | `torch.optim.Adam` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | 5 | Dropout<br>Linear<br>ReLU<br>Linear | [experiment](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/0593417505fa42ffbaa28e35faa96fb7/output/execution)
0 | ResNet50 | 0.924 | - | `torch.optim.Adam` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | 15 | Dropout<br>Linear<br>ReLU<br>Linear | [experiment](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/87f8c31fcb5b4102b3192908657f2db9/output/execution)
