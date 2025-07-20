# final_project [modelling]

> [Финальный проект (GitLab)](https://gitlab.deepschool.ru/dl-deploy2/lectures/-/tree/main/big-hw)

Свойство | Значение
-|-
Источник данных | [Kaggle (planets-dataset)](https://www.kaggle.com/datasets/nikitarom/planets-dataset/data)
Характер данных | Спутниковые снимки Амазонки
Задача | Мульти-классовая классификация
Инструменты | Python, Pandas, Lightning, ClearML, PyTorch, DVC, CI/CD, Linters

<br>

## Файлы

Расположение | Предназначение | Примечание
-|-|-
`.env` | Настройка перемнных в `docker-compose.yml` |
`configs/models/models.yaml` | Настройка моделей, которые используются в обучении | Для выбора модели из списка необходимо задать её индекс (`int`) в файле `project.yaml` в переменной `active_model_index: int` $ \in [0, \infin) $
`src/constants.py` | Настройка именований в ClearML и путей к файлам
`src/main.py` | Стандартный запуск обучения и тестирования |
`src/main_pipeline.py` | Запуск обучения и тестирования через пайплайн ClearML |
`output/best` | Лучшая модель, которая была определена командой `make select_best_model`

## Запуск

>* Первый запуск может занять ~1час (дольше всего ставится образ и скачивается-закачивается датасет в ClearML)
>---
>* На хосте должен быть установлен NVIDIA Docker2 и NVIDIA Container Toolkit
>* Docker-образ весит ~21Gb
>* Первая сборка docker-образа занимает ~20мин
>* Если датасет уже скачан локально, можно поместить его в папку `dataset`, а для загрузки в ClearML использовать команду `make upload.dataset_from_local_to_clearml` (изменить в `command` в `docker-compose.yml`)
>* Если датасет не скачан локально, то будет произведено его 1) скачивание, 2) распаковка, 3) загрузка в ClearML через команду `make upload.dataset_from_url_to_clearml` (по дефолту в `command` в `docker-compose.yml`), что занимает ~30мин
>* Предусмотрена единственная загрузка датасета - в ходе перезапуска производится проверка существования датасета в ClearML и если он есть, то никакие скачивающе-распаковывающе-загружающие действия не производятся

### Обучение

Запуск обучения:
```
make run.gpu
```
Последовательность выполнения:
1. Сборка образа
2. Запуск контейнера
3. Контейнер выполняет команду `make run.modelling` (`docker-compose.yml` -> `command`):
    - Загрузка датасета в ClearML. Команда задаётся вручную в зависимости от места расположения проекта: с kaggle или с хоста. По дефолту - с kaggle
        >Обе команды проверяют наличие датасета в ClearML, и если он существует, то пропускают дальнейшие шаги
        - `make upload.dataset_from_url_to_clearml` - скачивание/распаковка/загрузка в ClearML набора данных
        - `make upload.dataset_from_local_to_clearml` - загрузка существующего на хосте набора данных
    - Запуск обучения `make train` (стандартный запуск) или `make train.pipeline` (ClearML Pipeline)
    - Выбор лучшей модели `select_best_model` (выбирается по максимальному значению `valid_f2`)

Загрузка файлов `.env`, энкодера классов, модели в dvc:
```
make dvc.add.files
```

### Сервисы

Сервис |Ссылка
-|-
ClearML Dataset | [planets-classifier/planets-dataset](https://app.clear.ml/projects/c4785f8f04004d8b94d2217c10e51ebc/experiments/1054f86e63d24375ab4bf533cb7aac22/output/execution)
ClearML Model | [artifact](https://app.clear.ml/projects/*/models/d853949561ce4e7f92caf2359af30937)
ClearML Pipeline | [pipeline](https://app.clear.ml/projects/923d14e57e8a425e8d490cbab124ed39/experiments/d6fa7e37477e4bb2a2535e9a97ef2df1/output/execution)
Эксперименты | [experiments.md](experiments.md)
CI/CD Pipeline | [pipeline](https://gitlab.deepschool.ru/dl-deploy2/p.kukhtenkova/final_project_modelling/-/pipelines)

### CICD

Пайплайн CI/CD отрабатывает при каждом коммите в ветку `dev`.

## Структура проекта

```
├── .dvc/                                      # конфигурация DVC
├── configs/                                   # конфигурационные файлы
│    ├── models/
│    │    └── models.yaml                      # модели для экспериментов
│    └── project.yaml                          # весь проект
├── dataset/                                   # данные
├── logs/                                      # логи
├── notebooks/                                 # Jupyter-ноутбуки
├── output/                                    # модели в ONNX
│    └── best/                                 # лучшая модель по метрике
├── requirements/                              # зависимости
├── src/                                       # исходный код
│    ├── callbacks/                            # моудль коллбеков
│    │    ├── batch_visualize.py               # визуализация батчей
│    │    ├── clearml_track.py                 # сохранение чекпоитов
│    │    ├── model_summary.py                 # логировние слоёв модели
│    │    └── onnx_export.py                   # экспорт и валидация ONNX-модели
│    ├── configs/                              # модуль конфигураций
│    │    └── project_config.py                # конфигурация
│    ├── data/                                 # модуль подготовки данных
│    │    ├── data_module.py                   # подготовка датасета
│    │    ├── dataset.py                       # кастомный датасет
│    │    └── transforms.py                    # трансформации изображений
│    ├── executables/                          # модуль исполняемых скриптов
│    │    ├── select_best_model.py             # выбор лучшей модели
│    │    └── upload_dataset_to_clearml.py     # загрузка датасета в ClearML
│    ├── model/                                # модуль подготовки модели
│    │    ├── lightning_module.py              # модель
│    │    └── metrics.py                       # метрики
│    ├── pipeline/                             # модуль папйплайна ClearML
│    │    ├── predict.py                       # инференс
│    │    ├── preprocess.py                    # препроцессинг
│    │    └── train.py                         # обучение
│    ├── utils/                                # модуль полезных функций
│    │    └── logger.py                        # кастомный логгер
│    ├── constants.py                          # константы
│    ├── main_pipeine.py                       # запуск пайплайна ClearML
│    └── main.py                               # запуск обучения
├── .gitlab-ci.yml                             # настройка CICD
├── Makefile                                   # управляющие команды
└── ...
```
