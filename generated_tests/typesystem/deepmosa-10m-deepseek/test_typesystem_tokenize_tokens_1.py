# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = {}
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2.lookup_key(var_1)

def test_case_1():
    var_0 = {}
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = var_2.lookup(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True

def test_case_2():
    var_0 = {}
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = None
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.DictToken()

def test_case_4():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = True
    var_1 = [var_0]
    var_2 = None
    var_3 = -1426
    var_4 = module_0.Token(var_2, var_3, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4.lookup(var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.ScalarToken(var_0, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3.lookup_key(var_1)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2.__repr__()

def test_case_8():
    var_0 = None
    var_1 = None
    var_2 = module_0.ScalarToken(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8688422110360
    var_4 = var_3.__eq__(var_0)
    var_5 = ' hh'
    var_6 = -748
    var_7 = module_0.Token(var_5, var_6, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = False
    var_2 = False
    var_3 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = module_0.ListToken(var_0, var_0, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5 = var_3.__eq__(var_2)
    assert var_5 is False
    var_6 = [var_4, var_4]
    var_4.lookup(var_6)

def test_case_10():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    with pytest.raises(NotImplementedError):
        var_3 = var_2.value

def test_case_11():
    var_0 = 100
    var_1 = 'abU'
    var_2 = module_0.ListToken(var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    with pytest.raises(AttributeError):
        var_3 = bool(not var_2 == var_2)
    assert var_3 is True

def test_case_12():
    var_0 = None
    var_1 = []
    var_2 = -4379
    var_3 = module_0.Token(var_0, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = ''
    var_5 = module_0.Token(var_0, var_2, var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_6 = var_5.lookup(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_7 = 449
    var_8 = module_0.ScalarToken(var_0, var_0, var_7, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = var_8.__hash__()
    assert var_9 == 8688422110360
    var_10 = module_0.ScalarToken(var_9, var_9, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_11 = var_8.__hash__()
    assert var_11 == 8688422110360
    var_12 = -2417
    var_13 = module_1.Position(var_11, var_12, var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no == 8688422110360
    assert var_13.column_no == -2417
    assert var_13.char_index == 8688422110360
    var_14 = var_8.__eq__(var_10)
    assert var_14 is False

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 5
    var_1 = 29
    var_2 = '\\5m$nh1aq1p4\t'
    var_3 = module_0.ScalarToken(var_1, var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = {var_3: var_0}
    var_5 = [var_4, var_0, var_1]
    var_6 = False
    var_7 = False
    var_8 = module_0.ScalarToken(var_4, var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = {}
    var_10 = module_0.DictToken(*var_5, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_10.__eq__(var_8)

def test_case_14():
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 5
    var_1 = {}
    var_2 = [var_1, var_0, var_0]
    var_3 = {}
    var_4 = module_0.DictToken(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_5 = var_4._value
    var_6 = 2015.843432
    var_7 = [var_6]
    var_4.lookup_key(var_7)

def test_case_16():
    var_0 = {}
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 5
    var_1 = 29
    var_2 = '\\5m$nh1aq1p4\t'
    var_3 = module_0.ScalarToken(var_1, var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = {}
    var_5 = [var_4, var_0, var_1]
    var_6 = False
    var_7 = False
    var_8 = module_0.ScalarToken(var_4, var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = {}
    var_10 = module_0.DictToken(*var_5, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_11 = var_10.__eq__(var_8)
    assert var_11 is False
    var_12 = bool(var_10._value == var_4)
    assert var_12 is True
    var_13 = var_10.__eq__(var_10)
    assert var_13 is True
    var_14 = var_10._start_index
    var_15 = var_10._end_index
    var_15.lookup_key(var_7)

def test_case_18():
    var_0 = None
    var_1 = 10
    var_2 = 'line1\nline2\nline3'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_3.start
    var_5 = var_4.line_no
    assert var_5 == 2
    var_6 = var_4.column_no
    assert var_6 == 1
    with pytest.raises(AttributeError):
        var_7 = var_4.index
    assert var_7 == 5

def test_case_19():
    var_0 = None
    var_1 = 0
    var_2 = 12
    var_3 = 'line1\nline2\nline3'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5 = var_4.end
    var_6 = var_5.line_no
    assert var_6 == 3
    var_7 = var_5.column_no
    assert var_7 == 1
    with pytest.raises(AttributeError):
        var_8 = var_5.index
    assert var_8 == 12

def test_case_20():
    var_0 = None
    var_1 = -26
    var_2 = 41
    var_3 = 'line1\nline2\nline3'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5 = var_4.start
    var_6 = var_5.line_no
    assert var_6 == 2
    var_7 = var_5.column_no
    assert var_7 == 1
    with pytest.raises(AttributeError):
        var_8 = var_5.index
    assert var_8 == 5

def test_case_21():
    var_0 = -4427
    var_1 = ''
    var_2 = module_0.ListToken(var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True