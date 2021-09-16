FROM python:3.9
MAINTAINER "TaoLin" <tanlin2013@gmail.com>

ARG WORKDIR=/home
ENV PYTHONPATH "${PYTHONPATH}:$WORKDIR"
WORKDIR $WORKDIR

# Install fortran, blas, lapack
RUN apt update
RUN apt-get install -y gfortran libblas-dev liblapack-dev

# Install required python packages
COPY . $WORKDIR
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install tnpy
RUN python setup.py install

ENTRYPOINT /bin/bash