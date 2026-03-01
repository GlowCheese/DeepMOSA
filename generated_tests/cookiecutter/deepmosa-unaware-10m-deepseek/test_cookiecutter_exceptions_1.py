# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.exceptions as module_0

def test_case_0():
    var_0 = module_0.ConfigDoesNotExistException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.exceptions.ConfigDoesNotExistException'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False

def test_case_1():
    var_0 = []
    var_1 = module_0.CookiecutterException(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.exceptions.CookiecutterException'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    var_2 = module_0.ContextDecodingException()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'cookiecutter.exceptions.ContextDecodingException'
    var_3 = None
    var_4 = module_0.UndefinedVariableInTemplate(var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'cookiecutter.exceptions.UndefinedVariableInTemplate'
    assert var_4.message is None
    assert var_4.error is None
    assert var_4.context is None

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.UndefinedVariableInTemplate(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.exceptions.UndefinedVariableInTemplate'
    assert var_1.message is None
    assert var_1.error is None
    assert var_1.context is None
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    var_1.__str__()