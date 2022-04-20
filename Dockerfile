FROM python:3.10
MAINTAINER "TaoLin" <tanlin2013@gmail.com>

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
#RUN pip install --upgrade pip && \
#    pip install -r requirements.txt && \
#    python setup.py install && \
#    rm requirements.txt requirements_dev.txt setup.py && \
#    rm -rf build dist tnpy tnpy.egg-info
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.2.0b1 python3 - && \
    poetry config virtualenvs.create false --local && \
    poetry install --without test,lint,docs -vvv

ENTRYPOINT /bin/bash