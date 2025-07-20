FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y make libgl1 libglib2.0-0 openssh-client && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/requirements-heavy.txt /app/requirements/
RUN pip install --no-cache-dir --prefer-binary -r requirements/requirements-heavy.txt

COPY requirements/requirements-base.txt /app/requirements/
RUN pip install --no-cache-dir -r requirements/requirements-base.txt

COPY requirements/requirements-dvc.txt .
RUN pip install --no-cache-dir -r requirements-dvc.txt

ARG STAGING_HOST
ARG SSH_PRIVATE_KEY

RUN mkdir -p /root/.ssh && \
    echo "$SSH_PRIVATE_KEY" > /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa && \
    ssh-keyscan $STAGING_HOST >> /root/.ssh/known_hosts

ENV PYTHONPATH=.

COPY . .
