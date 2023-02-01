FROM python:3.11-slim as python
ENV PYTHONUNBUFFERED=true
WORKDIR /app


FROM python as lapack
RUN apt update && \
    apt-get install -y --no-install-recommends  \
    gfortran libblas-dev liblapack-dev graphviz


FROM python as poetry
ENV POETRY_HOME=/opt/poetry
ENV PATH="$POETRY_HOME/bin:$PATH"
COPY . ./
RUN apt update && \
    apt-get install -y curl
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.3.2 &&  \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi -vvv --without dev


FROM python as runtime
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=poetry /app /app
RUN apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*
ENTRYPOINT /bin/bash
