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
    var_0 = 'key: ['
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_3():
    var_0 = b'Vix\xa2_g\x96bZ\xc5A.'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_4():
    var_0 = ''
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_5():
    var_0 = '5\r'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'key: !!invalid_tag value'
    module_0.tokenize_yaml(var_0)

def test_case_7():
    var_0 = 'key:\n  - 1\n  - two'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2 = var_1.value
    var_3 = var_1.string
    assert var_3 == 'key:\n  - 1\n  - two'
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
    var_9 = 6
    var_10 = 17
    var_11 = module_1.Position(var_4, var_9, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no == 1
    assert var_11.column_no == 6
    assert var_11.char_index == 17
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.value
    var_3 = var_1.string
    assert var_3 == '3.14'
    module_0.validate_yaml(var_0, var_3)

def test_case_9():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = var_1.start
    var_6 = bool(var_1.start == var_2)
    var_7 = 5
    var_8 = 4
    var_9 = module_1.Position(var_4, var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no == 1
    assert var_9.column_no == 5
    assert var_9.char_index == 4
    var_10 = bool(var_1.end == var_9)
    assert var_10 is True

def test_case_10():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
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
    var_10 = module_1.Position(var_4, var_9, var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 1
    assert var_10.column_no == 4
    assert var_10.char_index is None
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)