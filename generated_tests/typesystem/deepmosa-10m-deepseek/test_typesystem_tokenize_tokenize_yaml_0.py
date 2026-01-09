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
    var_0 = "!la+3\rsB4zs'Al"
    module_0.validate_yaml(var_0, var_0)

def test_case_2():
    var_0 = ':'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_3():
    var_0 = 'XC]%<O'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = b'\x7fF\xcc\x16\xa6\xb4\x80\x8b\xee\x99{'
    module_0.validate_yaml(var_0, var_0)

def test_case_5():
    var_0 = ''
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = b'\xa78\xac'
    module_0.validate_yaml(var_0, var_0)

def test_case_7():
    var_0 = 'q:'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

def test_case_8():
    var_0 = '-'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '.7'
    module_0.validate_yaml(var_0, var_0)

def test_case_10():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
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
    var_9 = 5
    var_10 = module_1.Position(var_4, var_9, var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 1
    assert var_10.column_no == 5
    assert var_10.char_index == 'false'
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)