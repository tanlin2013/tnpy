FROM python:3.6.5
MAINTAINER "TaoLin" <tanlin2013@gmail.com>

ARG WORKDIR=/home/project
ENV PYTHONPATH "${PYTHONPATH}:$WORKDIR"

# Install required python packages
COPY . $WORKDIR
WORKDIR $WORKDIR
RUN pip install -r $WORKDIR/requirements.txt

# Install Blas Lapack
RUN apt update
RUN apt-get install -y --no-install-recommends --allow-unauthenticated gfortran
RUN apt-get install -y --no-install-recommends --allow-unauthenticated libblas-dev liblapack-dev

# Install Primme release-2.2
RUN git clone https://github.com/primme/primme.git
RUN cd primme && \
git checkout release-2.2 &&\
make python_install &&\
cd ..

# Install TNpy
RUN python setup.py install

ENTRYPOINT /bin/bash/
#ENTRYPOINT python -m unittest discover -s $WORKDIR/test -p '*_test.py'
