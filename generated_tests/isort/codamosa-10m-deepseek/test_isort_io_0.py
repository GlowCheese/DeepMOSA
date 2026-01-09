# Check out: https://github.com/GlowCheese/deepmosa
import inspect as module_1

import isort.exceptions as module_2
import isort.io as module_0
import pytest


def test_case_0():
    pass

def test_case_1():
    var_0 = module_0._EmptyIO()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    var_1 = module_0._EmptyIO()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.io._EmptyIO'
    var_2 = None
    var_3 = [var_2, var_2]
    var_4 = var_0.write(*var_3)
    var_5 = None
    with pytest.raises(TypeError):
        module_1.getgeneratorlocals(var_5)

def test_case_2():
    var_0 = None
    var_1 = module_0.File(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.io.File'
    assert var_1.stream is None
    assert var_1.path is None
    assert var_1.encoding is None
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.File.extension).__module__}.{type(module_0.File.extension).__qualname__}' == 'builtins.property'
    with pytest.raises(module_2.UnsupportedEncoding):
        var_1.detect_encoding(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = '<^u"-itX>wWw'
    var_2 = module_0.File(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.io.File'
    assert var_2.stream is None
    assert var_2.path is None
    assert var_2.encoding == '<^u"-itX>wWw'
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.File.extension).__module__}.{type(module_0.File.extension).__qualname__}' == 'builtins.property'
    var_2.from_contents(var_0, var_0)