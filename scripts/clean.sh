#!/bin/bash
set -euxo pipefail

poetry run isort tnpy/ tests/
poetry run black tnpy/ tests/
