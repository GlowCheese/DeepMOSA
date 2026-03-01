# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokenize_yaml as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = "!la+3\rsB4zs'Al"
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '".+BP4\n'
    module_0.validate_yaml(var_0, var_0)

def test_case_3():
    var_0 = b'Vix\xa2_g\x96bZ\xc5A.'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = ''
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = b'\xaa8'
    module_0.validate_yaml(var_0, var_0)

def test_case_6():
    var_0 = 'a:\n  b: c'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '&r'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'int: 1\nfloat: 1.0\nbool: true\nnull: null'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'int': 1, 'float': 1.0, 'bool': True, 'null': None})
    var_1.lookup(var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '[/\r]'
    module_0.validate_yaml(var_0, var_0)