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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = b'G\xd0\x1e'
    module_0.tokenize_yaml(var_0)

def test_case_3():
    var_0 = '7YAn'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_4():
    var_0 = b'\x1e'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '\n    items:\n      - 1\n      - 2\n      - 3\n    '
    var_1 = None
    module_0.validate_yaml(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '7:'
    var_1 = None
    module_0.validate_yaml(var_0, var_1)

def test_case_7():
    var_0 = '}ZzD\\E/Hl\\\t/u1VPfnW'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = b'\xbc2\x88\x90.3'
    module_0.validate_yaml(var_1, var_0)

def test_case_9():
    var_0 = 1
    var_1 = 0
    var_2 = module_1.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 1
    assert var_2.char_index == 0
    var_3 = 'key: value'
    var_4 = module_0.tokenize_yaml(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_5 = len(var_3)
    var_6 = var_5 - var_0
    var_7 = b'key: value'
    var_8 = module_0.tokenize_yaml(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_9 = '- item1\n- item2'
    var_10 = module_0.tokenize_yaml(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_11 = 'scalar'
    var_12 = module_0.tokenize_yaml(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_13 = '42'
    var_14 = '3.14'
    var_15 = module_0.tokenize_yaml(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_16 = 'true'
    var_17 = module_0.tokenize_yaml(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_18 = module_0.tokenize_yaml(var_13)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_19 = 'key: value: extra'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_19)