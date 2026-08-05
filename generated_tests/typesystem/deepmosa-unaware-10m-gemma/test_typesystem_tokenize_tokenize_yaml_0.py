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
    var_0 = b'\xee\x99'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'T:'
    module_0.validate_yaml(var_0, var_0)

def test_case_7():
    var_0 = '1'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = b'\x80.7'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = 'Lv&'
    var_3 = module_0.tokenize_yaml(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = [var_3]
    var_3.lookup(var_4)

def test_case_9():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = '123'
    var_3 = module_0.tokenize_yaml(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = 'true'
    var_5 = module_0.tokenize_yaml(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = 'null'
    var_7 = module_0.tokenize_yaml(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 2
    with pytest.raises(IndexError):
        var_12 = var_9.value[var_11]