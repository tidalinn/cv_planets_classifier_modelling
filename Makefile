.PHONY = *


STAGE  := dev
PYTHON := python3 -m

KAGGLE_DATASET := nikitarom/planets-dataset

VENV     := ~/.$(TRAIN_IMAGE)
VENV_DVC := $(VENV)/bin/dvc
VENV_PIP := $(VENV)/bin/pip


# -- EXECUTABLES ---

upload.dataset_from_kaggle_to_clearml:
	$(PYTHON) src.executables.upload_dataset_to_clearml --kaggle_dataset $(KAGGLE_DATASET)

upload.dataset_from_local_to_clearml:
	$(PYTHON) src.executables.upload_dataset_to_clearml --path ./dataset

train:
	$(PYTHON) src.main

train.pipeline:
	$(PYTHON) src.main_pipeline

select_best_model:
	$(PYTHON) src.executables.select_best_model

run.modelling: upload.dataset_from_kaggle_to_clearml train select_best_model


# --- VENV ---

venv.create:
ifeq ($(wildcard $(VENV)/),)
	$(PYTHON) venv $(VENV)
	$(VENV_PIP) install --no-cache-dir -r requirements/requirements-dvc.txt
	$(VENV_PIP) install --upgrade pip
ifeq ($(STAGE),dev)
	$(VENV_PIP) install --no-cache-dir -r requirements/requirements-dev.txt
endif
endif

venv.remove:
	rm -rf $(VENV)


# -- DOCKER ---

down:
	docker compose down --volumes --remove-orphans

run.gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

run.cpu:
	docker compose up --build
