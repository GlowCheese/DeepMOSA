# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_1
import typesystem.tokenize.tokenize_yaml as module_0


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
    var_0 = b'\x94'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_6():
    var_0 = '\n4'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_7():
    var_0 = '~'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '5#/t>)s^9`j,:'
    module_0.validate_yaml(var_0, var_0)

def test_case_9():
    var_0 = '-'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

def test_case_10():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 0
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_2)
    var_8 = 4
    var_9 = 3
    var_10 = module_1.Position(var_5, var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 0
    assert var_10.column_no == 4
    assert var_10.char_index == 3
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)

def test_case_11():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.string
    assert var_2 == 'false'
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 1
    assert var_5.char_index == 0
    var_6 = var_1.start
    var_7 = 4
    var_8 = module_1.Position(var_3, var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == 1
    assert f'{type(var_8.column_no).__module__}.{type(var_8.column_no).__qualname__}' == 'typesystem.base.Position'
    assert var_8.char_index == 4
    var_9 = bool(var_1.end == var_8)