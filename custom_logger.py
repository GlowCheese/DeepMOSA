import os
import logging

from functools import lru_cache
from colorama import Style, Fore, Back


_lf = f'{Back.LIGHTBLACK_EX}'
_rg = f'{Back.RESET}{Fore.RESET}'
lvlname = {
    'INFO':    f'     {_lf} INFO {_rg}',
    'DEBUG':   f'    {_lf} DEBUG {_rg}',
    'WARNING': f'  {_lf} WARNING {_rg}',
    'ERROR':   f'    {_lf} ERROR {_rg}',
}

class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_format = (
            f'{lvlname[record.levelname]}  '
            f'{Fore.CYAN}'
            '{name:<13} \033[1;37m| '
            f'{Style.RESET_ALL}'
        )

        if hasattr(record, 'title'):
            log_format += '{title}\n' + ' ' * 27 + '| '
        log_format += '{message}'
        
        formatter = logging.Formatter(log_format, style='{')
        return formatter.format(record)

def log(f, *, title: str, message: str, exc_info=None):
    '''
    A custom log function tailored to use with title.
    >>> log(logger.info, 'Title', 'Message')
    '''
    f(message, exc_info=exc_info, extra={'title': title})

@lru_cache(maxsize=1)
def _getHandler():
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    return handler

def getLogger(name):
    logger = logging.getLogger(name)
    verbose = min(int(os.getenv("VERBOSE", "0")), 1)
    log_level = [logging.INFO, logging.DEBUG][verbose]
    logger.setLevel(log_level)
    logger.addHandler(_getHandler())
    return logger
