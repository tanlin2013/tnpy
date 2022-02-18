FROM python:3.9
MAINTAINER "TaoLin" <tanlin2013@gmail.com>

ARG WORKDIR=/home/tnpy
ENV PYTHONPATH "${PYTHONPATH}:$WORKDIR"
WORKDIR $WORKDIR

# Install fortran, blas, lapack
RUN apt update && \
    apt-get install -y --no-install-recommends \
      gfortran libblas-dev liblapack-dev graphviz

# Install required python packages and tnpy
COPY . $WORKDIR
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python setup.py install && \
    rm requirements.txt setup.py && \
    rm -rf tnpy

ENTRYPOINT /bin/bash