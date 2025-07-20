.PHONY = dvc.connect run.modelling down build run.gpu run.cpu

include .env
export


PYTHON := python3 -m

KAGGLE_DATASET := nikitarom/planets-dataset


# --- DVC ---

dvc.connect:
	ssh $(STAGING_USERNAME)@$(STAGING_HOST)

dvc.init:
ifeq ($(wildcard .dvc/),)
	dvc init --no-scm
	dvc remote add --default $(STAGING_HOST) ssh://$(STAGING_HOST)/home/$(STAGING_USERNAME)/fp_modelling
	dvc remote modify $(STAGING_HOST) user $(STAGING_USERNAME)
	dvc remote list
	dvc config core.autostage true
endif

dvc.add.files: dvc.init
	dvc add .env
	dvc add output/encoder/mlb.pkl
	dvc add output/best/classificator.onnx
	dvc push


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

run.modelling: upload.dataset_from_kaggle_to_clearml train select_best_model dvc.add.files


# -- DOCKER ---

down:
	docker compose down --volumes --remove-orphans

build:
	@SSH_PRIVATE_KEY="$$(cat ~/.ssh/id_rsa)" docker compose build

run.cpu:
	docker compose up

run.gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
