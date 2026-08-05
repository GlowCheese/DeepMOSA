# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.log as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '\rwG)@ZL>GMBPg#'
    module_0.configure_logger(var_0, var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0.configure_logger(debug_file=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert var_1.filters == []
    assert var_1.name == 'cookiecutter'
    assert var_1.level == 10
    assert f'{type(var_1.parent).__module__}.{type(var_1.parent).__qualname__}' == 'logging.RootLogger'
    assert var_1.propagate is True
    assert f'{type(var_1.handlers).__module__}.{type(var_1.handlers).__qualname__}' == 'builtins.list'
    assert len(var_1.handlers) == 1
    assert var_1.disabled is False
    assert f'{type(var_1.manager).__module__}.{type(var_1.manager).__qualname__}' == 'logging.Manager'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.LOG_LEVELS == {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}
    assert module_0.LOG_FORMATS == {'DEBUG': '%(levelname)s %(name)s: %(message)s', 'INFO': '%(levelname)s: %(message)s'}