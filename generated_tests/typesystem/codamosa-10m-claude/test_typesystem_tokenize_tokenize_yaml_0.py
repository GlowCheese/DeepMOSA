# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '",?J'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = "!la+3\rsB4zs'Al"
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = b'\xe5\x1b\xa1\xb0[|\x15\xe2\xa1p\xbc0'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = ''
    module_0.validate_yaml(var_0, var_0)

def test_case_5():
    var_0 = b'\xc2$T\xe0\xf8'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '-'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '&i'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '6'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '8:'
    module_0.validate_yaml(var_0, var_0)

def test_case_10():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = '42'
    var_3 = module_0.tokenize_yaml(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = '3.14'
    var_5 = module_0.tokenize_yaml(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = module_0.tokenize_yaml(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = 'false'
    var_8 = module_0.tokenize_yaml(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = 'null'
    var_10 = module_0.tokenize_yaml(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_11 = '[1, 2, 3]'
    var_12 = module_0.tokenize_yaml(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = 'key: value'
    var_16 = module_0.tokenize_yaml(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_17 = b'hello'
    var_18 = module_0.tokenize_yaml(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_19 = 'name: John\nage: 30'
    var_20 = module_0.tokenize_yaml(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_21 = ''
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_21)