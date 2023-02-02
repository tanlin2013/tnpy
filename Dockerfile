FROM python:3.10.8 as python
LABEL maintainer="TaoLin tanlin2013@gmail.com"

ARG WORKDIR=/home
ENV PYTHONPATH="${PYTHONPATH}:$WORKDIR" \
    PATH="/root/.local/bin:$PATH" \
    PYTHONUNBUFFERED=true
WORKDIR $WORKDIR


FROM python as runtime
COPY . $WORKDIR

RUN apt update && \
    apt-get install -y --no-install-recommends  \
    gfortran libblas-dev liblapack-dev graphviz

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.3.2 &&  \
    poetry config virtualenvs.create false --local &&  \
    poetry install -vvv --without dev --all-extras

RUN apt-get -y clean &&  \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT /bin/bash
