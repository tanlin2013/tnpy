FROM python:3.10
LABEL maintainer="TaoLin tanlin2013@gmail.com"

ARG WORKDIR=/home/tnpy
ENV PYTHONPATH="${PYTHONPATH}:$WORKDIR" \
    PATH="/root/.local/bin:$PATH"
WORKDIR $WORKDIR
COPY . $WORKDIR

# Install fortran, blas, lapack
RUN apt update && \
    apt-get install -y --no-install-recommends \
      gfortran libblas-dev liblapack-dev graphviz
RUN apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

# Install required python packages and tnpy
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false --local && \
    poetry install --no-dev

ENTRYPOINT /bin/bash
