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

def test_case_1():
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
def test_case_2():
    var_0 = 'k'
    var_1 = module_0.log(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
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

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.log(var_0, timestamp=var_0)

def test_case_5():
    var_0 = -1762
    with pytest.raises(ValueError):
        module_0.set_logging_level(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.set_log_file(var_0)

def test_case_7():
    var_0 = False
    var_1 = module_0.get_logging_levels()
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
    var_2 = module_0.set_console_logging_function(var_0)
    var_3 = None
    with pytest.raises(ValueError):
        module_0.set_logging_level(var_3, file=var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = True
    module_0.log(var_0, force_console=var_0, include_proc_id=var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.log(var_0)
    var_1 = ':@st\nuSu<6K%%*D'
    var_3 = module_0.log(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'info'
    var_1 = module_0.set_logging_level(var_0)
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
    module_0.MultiprocessingFileHandler(var_1)

def test_case_11():
    var_0 = 'info'
    var_1 = module_0.set_logging_level(var_0)
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
    var_2 = 'warning'
    var_3 = True
    var_4 = False
    var_5 = module_0.set_logging_level(var_2, var_3, var_4)
    var_6 = 'error'
    var_7 = module_0.set_logging_level(var_6, var_4, var_3)
    assert module_0.LOGGER.level == 40
    var_8 = 'invalid_level'
    with pytest.raises(ValueError):
        module_0.set_logging_level(var_8)

def test_case_12():
    var_0 = 'k'
    var_1 = module_0.log(var_0)
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