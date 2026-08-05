# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.log as module_0
import termcolor.termcolor as module_1

def test_case_0():
    var_0 = module_0.get_worker_id()
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}

def test_case_1():
    var_0 = module_0.get_logging_levels()
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_1 = None
    var_2 = 'z?spWL'
    var_3 = True
    with pytest.raises(ValueError):
        module_0.log(var_2, var_1, var_1, include_proc_id=var_3)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.log(var_0, include_proc_id=var_0)

def test_case_3():
    var_0 = None
    var_1 = False
    with pytest.raises(ValueError):
        module_0.set_logging_level(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.MultiprocessingFileHandler(var_0)

def test_case_5():
    var_0 = module_0.get_logging_levels()
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}

def test_case_6():
    var_0 = 'O\t :7+\tWQIevU+'
    var_1 = module_0.set_log_file(var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert f'{type(module_0.LOGGER.handlers).__module__}.{type(module_0.LOGGER.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.LOGGER.handlers) == 1
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_2 = module_0.set_log_file(var_0)
    var_3 = module_1.colored(var_1)
    assert var_3 == 'None'
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.ATTRIBUTES == {'bold': 1, 'dark': 2, 'italic': 3, 'underline': 4, 'blink': 5, 'reverse': 7, 'concealed': 8, 'strike': 9}
    assert module_1.HIGHLIGHTS == {'on_black': 40, 'on_grey': 40, 'on_red': 41, 'on_green': 42, 'on_yellow': 43, 'on_blue': 44, 'on_magenta': 45, 'on_cyan': 46, 'on_light_grey': 47, 'on_dark_grey': 100, 'on_light_red': 101, 'on_light_green': 102, 'on_light_yellow': 103, 'on_light_blue': 104, 'on_light_magenta': 105, 'on_light_cyan': 106, 'on_white': 107}
    assert module_1.COLORS == {'black': 30, 'grey': 30, 'red': 31, 'green': 32, 'yellow': 33, 'blue': 34, 'magenta': 35, 'cyan': 36, 'light_grey': 37, 'dark_grey': 90, 'light_red': 91, 'light_green': 92, 'light_yellow': 93, 'light_blue': 94, 'light_magenta': 95, 'light_cyan': 96, 'white': 97}
    assert module_1.RESET == '\x1b[0m'

def test_case_7():
    var_0 = None
    var_1 = 'O\t :7+\tWQIevU+'
    var_2 = module_0.set_console_logging_function(var_1)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert f'{type(module_0.LOGGER.handlers).__module__}.{type(module_0.LOGGER.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.LOGGER.handlers) == 1
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_3 = module_0.get_worker_id()
    with pytest.raises(ValueError):
        module_0.set_logging_level(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'k:bV'
    module_0.log(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = None
    module_0.log(var_1)
    var_2 = module_1.colored(var_1, var_1, no_color=var_1, force_color=var_1)
    var_3 = var_2.format(var_0)
    var_4 = var_3.acquire()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.get_logging_levels()
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert f'{type(module_0.LOGGER.handlers).__module__}.{type(module_0.LOGGER.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.LOGGER.handlers) == 1
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_1 = None
    var_2 = module_0.get_worker_id()
    var_3 = False
    module_0.log(var_2, force_console=var_1, timestamp=var_3)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0.get_logging_levels()
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert f'{type(module_0.LOGGER.handlers).__module__}.{type(module_0.LOGGER.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.LOGGER.handlers) == 1
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_1 = '^ET'
    var_2 = True
    module_0.log(var_1, force_console=var_2)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'O\t 7+\tWQIevU+'
    var_1 = module_0.MultiprocessingFileHandler(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.log.MultiprocessingFileHandler'
    assert var_1.filters == []
    assert var_1.level == 0
    assert var_1.formatter is None
    assert f'{type(var_1.lock).__module__}.{type(var_1.lock).__qualname__}' == '_thread.RLock'
    assert f'{type(var_1.queue).__module__}.{type(var_1.queue).__qualname__}' == 'multiprocessing.queues.Queue'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert f'{type(module_0.LOGGER.handlers).__module__}.{type(module_0.LOGGER.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.LOGGER.handlers) == 1
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_2 = var_1.emit(var_0)
    var_2.close()

def test_case_13():
    var_0 = 'info'
    var_1 = True
    var_2 = module_0.set_logging_level(var_0, var_1, var_1)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert f'{type(module_0.LOGGER.handlers).__module__}.{type(module_0.LOGGER.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.LOGGER.handlers) == 1
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_3 = False
    with pytest.raises(ValueError):
        module_0.set_logging_level(var_2, var_3, var_3)

def test_case_14():
    var_0 = 'info'
    var_1 = False
    var_2 = module_0.set_logging_level(var_0, var_1, var_1)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert f'{type(module_0.LOGGER.handlers).__module__}.{type(module_0.LOGGER.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.LOGGER.handlers) == 1
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_3 = True
    var_4 = False
    with pytest.raises(ValueError):
        module_0.set_logging_level(var_2, var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'O\t 7+\tWQIevU+'
    var_1 = module_0.get_worker_id()
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert f'{type(module_0.LOGGER.handlers).__module__}.{type(module_0.LOGGER.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.LOGGER.handlers) == 1
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_2 = module_0.set_log_file(var_0)
    var_3 = module_0.set_console_logging_function(var_0)
    var_4 = module_0.MultiprocessingFileHandler(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.log.MultiprocessingFileHandler'
    assert var_4.filters == []
    assert var_4.level == 0
    assert var_4.formatter is None
    assert f'{type(var_4.lock).__module__}.{type(var_4.lock).__qualname__}' == '_thread.RLock'
    assert f'{type(var_4.queue).__module__}.{type(var_4.queue).__qualname__}' == 'multiprocessing.queues.Queue'
    var_5 = var_4.handle(var_3)
    assert var_5 is True
    var_6 = var_4.send(var_2)
    var_7 = var_4.setFormatter(var_5)
    assert var_4.formatter is True
    module_0.log(var_2)