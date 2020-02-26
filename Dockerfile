FROM
MAINTAINER "TaoLin" <tanlin2013@gmail.com>

# Install Primme
RUN git clone https://github.com/primme/primme && \
cd primme && \
make python_install

# Install required python packages
RUN pip install -r requirements.txt
