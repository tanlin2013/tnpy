#!/bin/bash
set -euxo pipefail

poetry run cruft check
poetry run safety check -i 39462 -i 40291
poetry run bandit -c pyproject.toml -r tnpy/
poetry run isort --check --diff tnpy/ tests/
poetry run black --check tnpy/ tests/
poetry run flake8 tnpy/ tests/
poetry run mypy \
           --install-types \
           --non-interactive \
           tnpy/
#  https://mypy.readthedocs.io/en/stable/running_mypy.html#library-stubs-not-installed
