import logging

from .formatter import CustomFormatter

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(CustomFormatter())

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler])

def getLogger(name):
    return logging.getLogger(name)
