import logging
from pathlib import Path
import yaml


# -- Load configurations from yaml file ----------------------
class ConfigReader:
    def __init__(self, file):
        self.__dict__.update(**yaml.safe_load(file))


with open(Path(__file__).absolute().parent / 'config.yaml', 'r') as f:
    config = ConfigReader(f)

# -- Version ----------------------
__version__ = config.version

# -- Define logger and the associated formatter and handler ----------------------
formatter = logging.Formatter('%(asctime)s [%(filename)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

logger = logging.getLogger('tnpy')
logger.setLevel(logging.INFO)
logger.addHandler(handler)
