import logging

__version__ = '0.0.9'


formatter = logging.Formatter('%(asctime)s [%(filename)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

logger = logging.getLogger('tnpy')
logger.setLevel(logging.INFO)
logger.addHandler(handler)
