# final_project [modelling]

> [Финальный проект (GitLab)](https://gitlab.deepschool.ru/dl-deploy2/lectures/-/tree/main/big-hw)

Свойство | Значение
-|-
Источник данных | [Kaggle (planets-dataset)](https://www.kaggle.com/datasets/nikitarom/planets-dataset/data)
Характер данных | Спутниковые снимки Амазонки
Задача | Multilabel-классификация
Инструменты | Python, Pandas, Lightning, ClearML, PyTorch, DVC, Linters

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

>* Первый запуск может занять ~1час (тяжёлый образ + скачивание-закачивание датасета в ClearML)
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

Загрузка файлов модели в dvc:
```
make -f Makefile.dvc dvc.add_files
```

### Сервисы

Сервис |Ссылка
-|-
ClearML Dataset | [planets-classifier/planets-dataset](https://app.clear.ml/projects/c4785f8f04004d8b94d2217c10e51ebc/experiments/1054f86e63d24375ab4bf533cb7aac22/output/execution)
ClearML Model | [artifact](https://app.clear.ml/projects/ee3520d1dd974ce3819097322568b32e/experiments/87f8c31fcb5b4102b3192908657f2db9/output/execution)
ClearML Pipeline | [pipeline](https://app.clear.ml/projects/923d14e57e8a425e8d490cbab124ed39/experiments/1aed9e9a0f424939b0c6e5c7c28996d1/output/execution)
Эксперименты | [experiments.md](experiments.md)
