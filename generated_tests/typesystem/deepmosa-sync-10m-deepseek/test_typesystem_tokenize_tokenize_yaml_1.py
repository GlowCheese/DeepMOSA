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
    var_0 = '   \n  \t  '
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = b'6\x9e\xe7\xd7'
    module_0.validate_yaml(var_0, var_0)

def test_case_7():
    var_0 = b'invalid: \x81\x82'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '-'
    module_0.validate_yaml(var_0, var_0)

def test_case_9():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == 1
    assert var_6.column_no == 1
    assert var_6.char_index == 0
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = module_1.Position(var_4, var_9, var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 1
    assert var_10.column_no == 4
    assert var_10.char_index == 1
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)

def test_case_10():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = 0
    var_3 = module_1.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_3.line_no).__module__}.{type(var_3.line_no).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    assert f'{type(var_3.column_no).__module__}.{type(var_3.column_no).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    assert var_3.char_index == 0
    var_4 = var_1.start
    var_5 = bool(var_1.start == var_3)
    var_6 = 840
    var_7 = module_1.Position(var_6, var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 840
    assert var_7.column_no is False
    assert var_7.char_index == 840
    var_8 = var_1.end
    var_9 = bool(var_1.end == var_7)