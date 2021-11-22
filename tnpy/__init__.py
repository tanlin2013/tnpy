import logging

__version__ = '0.0.4'

logging.basicConfig(format='%(asctime)s [%(filename)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.root.setLevel(level=logging.INFO)
