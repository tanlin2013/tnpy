import logging
from importlib import metadata

# -- Version -----------------------------------------------------------------

__version__ = metadata.version(__name__)

# -- Define logger and the associated formatter and handler ------------------

formatter = logging.Formatter(
    "%(asctime)s [%(filename)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

logger = logging.getLogger("tnpy")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
