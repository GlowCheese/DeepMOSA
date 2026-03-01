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
    var_0 = "!la+3\rsB4zs'Al"
    module_0.validate_yaml(var_0, var_0)

def test_case_2():
    var_0 = b'\xed]'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'qws'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = b'\xa3yv\xcaf\x97$\xb2\xb0\xde\x10\x94\xdd'
    module_0.tokenize_yaml(var_0)

def test_case_5():
    var_0 = b''
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_6():
    var_0 = '7'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_7():
    var_0 = 'O:6):'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = b'\xdd-'
    module_0.validate_yaml(var_0, var_0)

def test_case_9():
    var_0 = b'8.'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_10():
    var_0 = b'o\xc9n'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'