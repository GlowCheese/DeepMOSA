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
    var_0 = None
    module_0.validate_yaml(var_0, var_0)

def test_case_2():
    var_0 = b'"\xc9\x81V\xb78'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = b'G\xd0\x1e'
    module_0.tokenize_yaml(var_0)

def test_case_4():
    var_0 = '~eviG+B)os?;'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_5():
    var_0 = b''
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_6():
    var_0 = '8 #w\\K\\@It(NhG'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = b'"\xc9V\xb78'
    var_1 = b'\xed\xb78.'
    module_0.validate_yaml(var_1, var_0)

def test_case_8():
    var_0 = b'\xc3\x85*\xc8:\xf1\xaa\xb9\xdb'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

def test_case_9():
    var_0 = '-'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'HaSPTM^I '
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_3.__repr__()
    assert var_4 == 'False'
    module_0.validate_yaml(var_4, var_4)