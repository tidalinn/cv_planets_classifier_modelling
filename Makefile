.PHONY = run.modelling dvc.connect dvc.add.files down run.gpu run.cpu

include .env
export


STAGE  := dev
PYTHON := python3 -m

KAGGLE_DATASET := nikitarom/planets-dataset

VENV        := ~/.venv
VENV_PYTHON := $(VENV)/bin/$(PYTHON)
VENV_PIP    := $(VENV)/bin/pip


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
	$(VENV_PIP) install --no-cache-dir -r requirements/requirements-local.txt
	$(VENV_PIP) install --upgrade pip
ifeq ($(STAGE),dev)
	$(VENV_PIP) install --no-cache-dir -r requirements/requirements-dev.txt
endif
endif


# --- DVC ---

dvc.connect:
	ssh $(STAGING_USERNAME)@$(STAGING_HOST)

dvc.init: venv.create
ifeq ($(wildcard .dvc/),)
	$(VENV_PYTHON) dvc init
	$(VENV_PYTHON) dvc remote add --default $(STAGING_HOST) ssh://$(STAGING_HOST)/home/$(STAGING_USERNAME)/fp_modelling
	$(VENV_PYTHON) dvc remote modify $(STAGING_HOST) user $(STAGING_USERNAME)
	$(VENV_PYTHON) dvc remote list
	$(VENV_PYTHON) dvc config core.autostage true
endif

dvc.add.files: dvc.init
	$(VENV_PYTHON) dvc add .env
	$(VENV_PYTHON) dvc add output/encoder/mlb.pkl
	$(VENV_PYTHON) dvc add output/best/classificator.onnx
	$(VENV_PYTHON) dvc push;


# -- DOCKER ---

down:
	docker compose down --volumes --remove-orphans

run.gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

run.cpu:
	docker compose up --build
