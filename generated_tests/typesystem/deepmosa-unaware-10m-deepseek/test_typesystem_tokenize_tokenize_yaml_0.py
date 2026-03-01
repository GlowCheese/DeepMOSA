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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '@geOj\n}'
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
    var_0 = '9'
    module_0.validate_yaml(var_0, var_0)

def test_case_6():
    var_0 = '-'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'OP;v6:'
    module_0.validate_yaml(var_0, var_0)

def test_case_8():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2 = '\n    outer:\n      inner: nested\n      list:\n        - item1\n        - item2\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_4 = 'outer'
    var_5 = var_3.value[var_4]
    var_6 = 'list'
    var_7 = var_3.value[var_4][var_6]
    var_8 = '\n    string: hello\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_9 = module_0.tokenize_yaml(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_10 = '- first\n- second\n- third'
    var_11 = module_0.tokenize_yaml(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_12 = ''
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_12)