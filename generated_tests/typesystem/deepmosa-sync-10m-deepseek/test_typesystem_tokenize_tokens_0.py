# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0
import builtins as module_1
import typesystem.base as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = -272
    var_2 = module_0.Token(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.Token(var_0, var_0, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_5 = [var_1, var_4]
    var_4.lookup_key(var_5)

def test_case_1():
    var_0 = None
    var_1 = []
    var_2 = 2007
    var_3 = module_0.ScalarToken(var_0, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.lookup(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = var_5.__hash__()
    assert var_6 == 7901703723672
    var_7 = var_6.__hash__()
    assert var_7 == 7901703723672
    var_8 = var_7.__hash__()
    assert var_8 == 7901703723672
    var_9 = var_8.__hash__()
    assert var_9 == 7901703723672
    var_10 = var_9.__hash__()
    assert var_10 == 7901703723672
    var_11 = var_10.__hash__()
    assert var_11 == 7901703723672

def test_case_2():
    var_0 = 4
    var_1 = None
    var_2 = module_0.ListToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.DictToken()

def test_case_4():
    var_0 = None
    var_1 = module_0.ListToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.ListToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_1.__repr__()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5 = var_4.end
    var_4.lookup_key(var_5)

def test_case_7():
    var_0 = True
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__hash__()
    assert var_2 == 1
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = -4187.0
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.ScalarToken(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.__hash__()
    assert var_4 == 7901703723672
    var_5 = var_4.__repr__()
    assert var_5 == '7901703723672'
    var_6 = module_0.ListToken(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6.lookup(var_1)

def test_case_9():
    var_0 = True
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_10():
    var_0 = None
    var_1 = -317
    var_2 = module_0.Token(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    with pytest.raises(NotImplementedError):
        var_3 = var_2.value

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = -265
    var_2 = module_0.Token(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_3 = False
    var_4 = var_2.__eq__(var_0)
    assert var_4 is False
    var_5 = module_0.Token(var_0, var_0, var_3, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_6 = [var_1]
    var_5.lookup_key(var_6)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_1.object()
    var_1 = None
    var_2 = True
    var_3 = module_0.Token(var_1, var_2, var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = module_0.ScalarToken(var_2, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = 1016
    var_6 = module_0.ListToken(var_1, var_5, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4.__eq__(var_6)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_1.__hash__()
    assert var_3 == 1
    var_4 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = module_0.Token(var_0, var_0, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_6 = var_4.__hash__()
    assert var_6 == 7901703723672
    var_7 = var_3.__hash__()
    assert var_7 == 1
    var_8 = {var_7: var_3, var_0: var_0, var_0: var_0}
    var_9 = var_8.__eq__(var_3)
    var_10 = [var_8, var_3, var_2]
    var_11 = var_3.__hash__()
    assert var_11 == 1
    var_12 = var_6.__hash__()
    assert var_12 == 7901703723672
    var_13 = var_12.__hash__()
    assert var_13 == 7901703723672
    var_14 = var_3.__repr__()
    assert var_14 == '1'
    module_0.DictToken(*var_10)

def test_case_14():
    var_0 = {}
    var_1 = 5
    var_2 = '  {}'
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = {var_4: var_0, var_5: var_1, var_6: var_2, var_5: var_2}
    var_8 = module_0.DictToken(*var_3, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_9 = var_8._start_index

def test_case_15():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5 = var_4.end
    with pytest.raises(AttributeError):
        var_6 = var_5.line
    assert var_6 == 1

def test_case_16():
    var_0 = None
    var_1 = 5
    var_2 = 10
    var_3 = 'line1\nline2\nline3'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5 = var_4.start
    with pytest.raises(AttributeError):
        var_6 = var_5.column
    assert var_6 == 1

def test_case_17():
    var_0 = 5
    var_1 = module_0.ListToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_2 = var_1.start
    with pytest.raises(AttributeError):
        var_3 = var_2.line
    assert var_3 == 2

def test_case_18():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = 20
    var_4 = '  {}'
    var_5 = []
    var_6 = 'value'
    var_7 = 'start_index'
    var_8 = None
    var_9 = module_0.Token(var_8, var_8, var_3, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_10 = var_9.__eq__(var_6)
    assert var_10 is False
    var_11 = module_0.ScalarToken(var_8, var_10, var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_12 = 'end_index'
    var_13 = 'content'
    var_14 = {var_6: var_2, var_7: var_3, var_12: var_4, var_13: var_4}
    var_15 = module_0.ScalarToken(var_8, var_3, var_10)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_16 = module_0.DictToken(*var_5, **var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_17 = -943
    var_18 = module_0.Token(var_8, var_8, var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_19 = var_16._start_index
    assert var_19 == 5
    with pytest.raises(AttributeError):
        var_20 = var_7._end_index
    assert var_20 == 6

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = var_2.__hash__()
    assert var_5 == 7901703723672
    var_6 = var_5.__hash__()
    assert var_6 == 7901703723672
    var_7 = module_0.ScalarToken(var_0, var_5, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = module_0.Token(var_5, var_0, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_9 = var_7.__hash__()
    assert var_9 == 7901703723672
    var_10 = var_9.__hash__()
    assert var_10 == 7901703723672
    var_11 = {var_7: var_6, var_2: var_5, var_5: var_5}
    var_12 = var_11.__eq__(var_6)
    var_13 = var_6.__hash__()
    assert var_13 == 7901703723672
    var_14 = var_9.__hash__()
    assert var_14 == 7901703723672
    var_15 = var_5.__hash__()
    assert var_15 == 7901703723672
    var_16 = var_15.__repr__()
    assert var_16 == '7901703723672'
    var_17 = var_7.__repr__()
    assert var_17 == "ScalarToken('')"
    module_0.DictToken(**var_9)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_2.__repr__()
    assert var_4 == "ScalarToken('')"
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True
    var_6 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = module_0.Token(var_0, var_0, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_8 = var_6.__hash__()
    assert var_8 == 7901703723672
    var_9 = var_1.__hash__()
    assert var_9 == 0
    var_10 = {var_9: var_2, var_0: var_0, var_0: var_0}
    var_11 = var_8.__eq__(var_4)
    var_12 = [var_10, var_4, var_2]
    var_13 = module_0.ScalarToken(var_9, var_11, var_8, var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_14 = module_2.Position(var_1, var_13, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no is False
    assert f'{type(var_14.column_no).__module__}.{type(var_14.column_no).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    assert f'{type(var_14.char_index).__module__}.{type(var_14.char_index).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_15 = var_6.__eq__(var_13)
    assert var_15 is False
    module_0.DictToken(*var_12)