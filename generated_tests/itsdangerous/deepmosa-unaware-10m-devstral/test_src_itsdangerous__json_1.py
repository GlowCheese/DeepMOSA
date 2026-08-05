# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import src.itsdangerous._json as module_0
import json.scanner as module_1

def test_case_0():
    var_0 = module_0._CompactJSON()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous._json._CompactJSON'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0._CompactJSON()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous._json._CompactJSON'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = None
    var_0.loads(var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0._CompactJSON()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous._json._CompactJSON'
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    var_1 = None
    var_2 = var_0.dumps(var_1)
    assert var_2 == 'null'
    var_3 = None
    module_1.py_make_scanner(var_3)