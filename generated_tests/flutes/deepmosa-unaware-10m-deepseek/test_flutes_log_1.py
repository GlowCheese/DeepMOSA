# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.log as module_0

def test_case_0():
    var_0 = module_0.get_worker_id()
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert module_0.LOGGER.handlers == []
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '%(levelname)s - %(message)s'
    var_1 = module_0.log(var_0)
    module_0.log(var_1, var_1, timestamp=var_1, include_proc_id=var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.get_worker_id()
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert module_0.LOGGER.handlers == []
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_1 = module_0.log(var_0, include_proc_id=var_0)

def test_case_3():
    var_0 = b'\xe2\x1f\x821\x89\xf9-\xfe\xde\xe8\x8eO`\x1f\xa2'
    with pytest.raises(ValueError):
        module_0.set_logging_level(var_0)

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
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert module_0.LOGGER.handlers == []
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.set_log_file(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'tMAV~*DyI@!I4]'
    var_1 = module_0.log(var_0)
    var_2 = 146
    var_3 = module_0.set_console_logging_function(var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.log(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.log(var_0, timestamp=var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = True
    module_0.log(var_0, force_console=var_0)

def test_case_11():
    var_0 = 'info'
    var_1 = True
    var_2 = False
    var_3 = module_0.set_logging_level(var_0, var_1, var_2)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert module_0.LOGGER.handlers == []
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_0.log(var_0)

def test_case_13():
    var_0 = 'info'
    var_1 = True
    var_2 = False
    var_3 = module_0.set_logging_level(var_0, var_1, var_2)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert module_0.LOGGER.handlers == []
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_4 = 'warning'
    var_5 = module_0.set_logging_level(var_4, var_2, var_1)
    assert module_0.LOGGER.level == 30
    var_6 = 'error'
    var_7 = module_0.set_logging_level(var_6, var_1, var_1)
    assert module_0.LOGGER.level == 40

def test_case_14():
    var_0 = True
    var_1 = 'info'
    var_2 = True
    var_3 = module_0.set_logging_level(var_1, var_2, var_0)
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 20
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert module_0.LOGGER.handlers == []
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_4 = 'warning'
    var_5 = module_0.set_logging_level(var_4, var_0, var_2)
    assert module_0.LOGGER.level == 30
    var_6 = 'error'
    var_7 = module_0.set_logging_level(var_6, var_2, var_2)
    assert module_0.LOGGER.level == 40

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 42
    var_1 = 'Test message %s'
    var_2 = 'arg1'
    var_3 = (var_2,)
    var_4 = 'Test exception'
    var_5 = ValueError(var_4)
    var_6 = 'nt'
    var_7 = '/dev/null'
    var_8 = 'NUL'
    var_9 = var_7 if var_1 else var_8
    var_10 = module_0.MultiprocessingFileHandler(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'flutes.log.MultiprocessingFileHandler'
    assert var_10.filters == []
    assert var_10.level == 0
    assert var_10.formatter is None
    assert f'{type(var_10.lock).__module__}.{type(var_10.lock).__qualname__}' == '_thread.RLock'
    assert f'{type(var_10.queue).__module__}.{type(var_10.queue).__qualname__}' == 'multiprocessing.queues.Queue'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.LOGGER).__module__}.{type(module_0.LOGGER).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.LOGGER.filters == []
    assert module_0.LOGGER.name == 'flutes.log'
    assert module_0.LOGGER.level == 40
    assert f'{type(module_0.LOGGER.parent).__module__}.{type(module_0.LOGGER.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.LOGGER.propagate is True
    assert module_0.LOGGER.handlers == []
    assert module_0.LOGGER.disabled is False
    assert f'{type(module_0.LOGGER.manager).__module__}.{type(module_0.LOGGER.manager).__qualname__}' == 'logging.Manager'
    assert module_0.COLOR_MAP == {'success': 'green', 'warning': 'yellow', 'error': 'red', 'info': 'white'}
    assert f'{type(module_0.LOGGING_MAP).__module__}.{type(module_0.LOGGING_MAP).__qualname__}' == 'builtins.dict'
    assert len(module_0.LOGGING_MAP) == 4
    assert module_0.LEVEL_MAP == {'success': 20, 'warning': 30, 'error': 40, 'info': 20, 'quiet': 999}
    var_11 = var_10.close()
    var_12 = var_0 != var_6
    var_13 = var_7 if var_12 else var_8
    var_14 = module_0.MultiprocessingFileHandler(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'flutes.log.MultiprocessingFileHandler'
    assert var_14.filters == []
    assert var_14.level == 0
    assert var_14.formatter is None
    assert f'{type(var_14.lock).__module__}.{type(var_14.lock).__qualname__}' == '_thread.RLock'
    assert f'{type(var_14.queue).__module__}.{type(var_14.queue).__qualname__}' == 'multiprocessing.queues.Queue'
    var_15 = var_14.close()
    var_16 = var_3 != var_6
    var_17 = var_7 if var_16 else var_8
    var_18 = module_0.MultiprocessingFileHandler(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'flutes.log.MultiprocessingFileHandler'
    assert var_18.filters == []
    assert var_18.level == 0
    assert var_18.formatter is None
    assert f'{type(var_18.lock).__module__}.{type(var_18.lock).__qualname__}' == '_thread.RLock'
    assert f'{type(var_18.queue).__module__}.{type(var_18.queue).__qualname__}' == 'multiprocessing.queues.Queue'
    var_5.close()