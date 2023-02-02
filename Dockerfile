FROM python:3.10.8 as python
LABEL maintainer="TaoLin tanlin2013@gmail.com"
ENV PYTHONUNBUFFERED=true
WORKDIR /app


FROM python as runtime
ENV POETRY_HOME=/opt/poetry
ENV PATH="/app/.venv/bin:$PATH"
COPY . ./

RUN apt update && \
    apt-get install -y --no-install-recommends  \
     gfortran libblas-dev liblapack-dev graphviz \

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.3.2 &&  \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi -vvv --without dev

RUN apt-get -y clean &&  \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT /bin/bash
