import logging
from importlib import metadata
from pathlib import Path

import toml  # tomllib is coming out on python 3.11 as standard lib


# -- Version ----------------------
__version__ = metadata.version(__name__)

# -- Define logger and the associated formatter and handler -------------
formatter = logging.Formatter(
    "%(asctime)s [%(filename)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

logger = logging.getLogger("tnpy")
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# -- Load configurations from toml file ----------------------
class ConfigReader:
    def __init__(self, file):
        self.__dict__.update(**toml.load(file))


with open(Path(__file__).absolute().parent / "config.toml", "r", encoding="utf-8") as f:
    config = ConfigReader(f)
