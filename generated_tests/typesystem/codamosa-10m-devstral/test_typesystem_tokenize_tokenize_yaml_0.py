# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.tokenize_yaml(var_0)

def test_case_1():
    var_0 = b"'\xe3J\x8a\x8c"
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = "!la+3\rsB4zs'Al"
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = b'\xae\xae[\x10o\x9e\xb0\x1e>\x93\xd9\xfb\xea\xee'
    module_0.validate_yaml(var_0, var_0)

def test_case_4():
    var_0 = ''
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_5():
    var_0 = b'\xac\xbf)K\xef\x9d\xdc\xe6\xa4\x8b\xcb\xc9'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'items:\n  - 123\n  - banana'
    var_1 = None
    module_0.validate_yaml(var_0, var_1)

def test_case_7():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2 = '- item1\n- item2'
    var_3 = module_0.tokenize_yaml(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4 = 'scalar_value'
    var_5 = module_0.tokenize_yaml(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = 'int_val: 42\nfloat_val: 3.14\nbool_val: true\nnull_val: null'
    var_7 = module_0.tokenize_yaml(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_8 = 'nested:\n  key: value\n  list:\n    - item1\n    - item2'
    var_9 = module_0.tokenize_yaml(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_10 = b'key: value'
    var_11 = module_0.tokenize_yaml(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'