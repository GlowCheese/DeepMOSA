# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.log as module_0

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

def test_case_2():
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
    var_1 = module_0.log(var_0)

def test_case_3():
    var_0 = -1762
    with pytest.raises(ValueError):
        module_0.set_logging_level(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.set_log_file(var_0)

def test_case_5():
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
    var_1 = None
    var_2 = module_0.set_console_logging_function(var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.log(var_0, force_console=var_0, include_proc_id=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.log(var_0, timestamp=var_0)

def test_case_8():
    var_0 = 'infPo'
    with pytest.raises(ValueError):
        module_0.log(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'lk)k&(;'
    module_0.log(var_0, force_console=var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.set_log_file(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0._get_console_logging_function()
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
    var_1 = '\n'
    var_2 = lambda msg: console_output.write(msg + var_1)
    var_3 = module_0.set_console_logging_function(var_2)
    var_4 = 'est info mesage'
    var_5 = 'info'
    module_0.log(var_4, var_5)

def test_case_12():
    var_0 = 'info'
    var_1 = module_0.set_logging_level(var_0)
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

def test_case_13():
    var_0 = 'info'
    var_1 = False
    var_2 = module_0.set_logging_level(var_0, var_1)
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
    module_0.log(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'info'
    var_1 = None
    var_2 = module_0.get_worker_id()
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
    var_3 = module_0.set_logging_level(var_0, file=var_1)
    module_0.log(var_1)

def test_case_15():
    var_0 = 'test_logger'
    var_1 = '/dev/null'
    var_2 = var_1 if var_0 else var_0
    var_3 = module_0.MultiprocessingFileHandler(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.log.MultiprocessingFileHandler'
    assert var_3.filters == []
    assert var_3.level == 0
    assert var_3.formatter is None
    assert f'{type(var_3.lock).__module__}.{type(var_3.lock).__qualname__}' == '_thread.RLock'
    assert f'{type(var_3.queue).__module__}.{type(var_3.queue).__qualname__}' == 'multiprocessing.queues.Queue'
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
    var_4 = var_3.emit(var_0)

def test_case_16():
    var_0 = 'test_logger'
    var_1 = 'test_logger'
    var_2 = 20
    var_3 = 'Error occurred'
    var_4 = '/dev/null'
    var_5 = 'nul'
    var_6 = var_4 if var_0 else var_5
    var_7 = module_0.MultiprocessingFileHandler(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.log.MultiprocessingFileHandler'
    assert var_7.filters == []
    assert var_7.level == 0
    assert var_7.formatter is None
    assert f'{type(var_7.lock).__module__}.{type(var_7.lock).__qualname__}' == '_thread.RLock'
    assert f'{type(var_7.queue).__module__}.{type(var_7.queue).__qualname__}' == 'multiprocessing.queues.Queue'
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
    var_8 = var_7.emit(var_1)
    var_9 = var_7.close()
    var_10 = var_2 != var_3
    var_11 = var_4 if var_10 else var_5
    var_12 = module_0.MultiprocessingFileHandler(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'flutes.log.MultiprocessingFileHandler'
    assert var_12.filters == []
    assert var_12.level == 0
    assert var_12.formatter is None
    assert f'{type(var_12.lock).__module__}.{type(var_12.lock).__qualname__}' == '_thread.RLock'
    assert f'{type(var_12.queue).__module__}.{type(var_12.queue).__qualname__}' == 'multiprocessing.queues.Queue'
    var_13 = var_12.emit(var_1)
    var_14 = var_12.close()
    var_15 = var_12.close()
    var_16 = var_7.send(var_4)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'test_logger'
    var_1 = '/dev/null'
    var_2 = 'nul'
    var_3 = var_1 if var_0 else var_2
    var_4 = module_0.get_worker_id()
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
    var_5 = module_0.MultiprocessingFileHandler(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.log.MultiprocessingFileHandler'
    assert var_5.filters == []
    assert var_5.level == 0
    assert var_5.formatter is None
    assert f'{type(var_5.lock).__module__}.{type(var_5.lock).__qualname__}' == '_thread.RLock'
    assert f'{type(var_5.queue).__module__}.{type(var_5.queue).__qualname__}' == 'multiprocessing.queues.Queue'
    var_6 = module_0.set_log_file(var_3)
    assert f'{type(module_0.LOGGER.handlers).__module__}.{type(module_0.LOGGER.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.LOGGER.handlers) == 1
    var_6.createLock()

def test_case_18():
    var_0 = 'Test messae %s'
    var_1 = 0.1
    var_2 = ValueError(var_0)
    var_3 = '/dev/null'
    var_4 = var_3 if var_0 else var_1
    var_5 = module_0.MultiprocessingFileHandler(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.log.MultiprocessingFileHandler'
    assert var_5.filters == []
    assert var_5.level == 0
    assert var_5.formatter is None
    assert f'{type(var_5.lock).__module__}.{type(var_5.lock).__qualname__}' == '_thread.RLock'
    assert f'{type(var_5.queue).__module__}.{type(var_5.queue).__qualname__}' == 'multiprocessing.queues.Queue'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert f'{type(module_0.LOGGER.handlers).__module__}.{type(module_0.LOGGER.handlers).__qualname__}' == 'builtins.list'
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_6 = var_5.emit(var_3)
    var_7 = var_5.emit(var_2)
    module_0.log(var_7)