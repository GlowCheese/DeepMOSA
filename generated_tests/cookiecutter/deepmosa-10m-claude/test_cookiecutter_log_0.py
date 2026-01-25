# Check out: https://github.com/GlowCheese/deepmosa
import builtins as module_0

import cookiecutter.log as module_1
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = {}
    var_1 = module_0.object(**var_0)
    var_2 = [var_1, var_1]
    module_1.configure_logger(debug_file=var_2)

def test_case_1():
    var_0 = module_1.configure_logger()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert var_0.filters == []
    assert var_0.name == 'cookiecutter'
    assert var_0.level == 10
    assert f'{type(var_0.parent).__module__}.{type(var_0.parent).__qualname__}' == 'logging.RootLogger'
    assert var_0.propagate is True
    assert f'{type(var_0.handlers).__module__}.{type(var_0.handlers).__qualname__}' == 'builtins.list'
    assert len(var_0.handlers) == 1
    assert var_0.disabled is False
    assert f'{type(var_0.manager).__module__}.{type(var_0.manager).__qualname__}' == 'logging.Manager'
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.LOG_LEVELS == {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}
    assert module_1.LOG_FORMATS == {'DEBUG': '%(levelname)s %(name)s: %(message)s', 'INFO': '%(levelname)s: %(message)s'}