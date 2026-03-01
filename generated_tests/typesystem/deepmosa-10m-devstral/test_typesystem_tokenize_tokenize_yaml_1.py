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
    var_0 = b'\x9f\xd0\x1e'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_6():
    var_0 = 'uyKyE:'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

def test_case_7():
    var_0 = '-'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

def test_case_8():
    var_0 = b'3\xf0\xa51\xdf\xcc'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_9():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 1
    assert var_7.column_no == 1
    assert var_7.char_index == 0
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 5
    var_11 = module_1.Position(var_5, var_10, var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no == 1
    assert var_11.column_no == 5
    assert var_11.char_index == pytest.approx(3.14, abs=0.01, rel=0.01)
    var_12 = var_1.end

def test_case_10():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.value
    assert var_2 is True
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 1
    assert var_5.char_index == 0
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 5
    var_9 = 4
    var_10 = module_1.Position(var_3, var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 1
    assert var_10.column_no == 5
    assert var_10.char_index == 4
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)