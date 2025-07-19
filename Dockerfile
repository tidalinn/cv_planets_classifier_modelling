FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y make libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/requirements-heavy.txt /app/requirements/
RUN pip install --no-cache-dir --prefer-binary -r requirements/requirements-heavy.txt

COPY requirements/requirements-base.txt /app/requirements/
RUN pip install --no-cache-dir -r requirements/requirements-base.txt

ENV PYTHONPATH=.

COPY . .
