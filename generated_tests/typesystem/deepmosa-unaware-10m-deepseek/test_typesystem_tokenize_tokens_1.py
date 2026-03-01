# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
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

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = 2148
    var_2 = 236
    var_3 = module_0.ScalarToken(var_0, var_2, var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.__hash__()
    assert var_4 == 8323099425944
    var_5 = var_4.__hash__()
    assert var_5 == 8323099425944
    var_6 = var_5.__hash__()
    assert var_6 == 8323099425944
    var_7 = var_6.__hash__()
    assert var_7 == 8323099425944
    var_8 = var_7.__hash__()
    assert var_8 == 8323099425944
    var_9 = var_8.__hash__()
    assert var_9 == 8323099425944
    var_10 = var_9.__hash__()
    assert var_10 == 8323099425944
    var_11 = module_0.Token(var_0, var_0, var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_12 = module_0.Token(var_0, var_0, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_13 = None
    var_14 = module_1.Position(var_1, var_1, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no == 2148
    assert var_14.column_no == 2148
    assert var_14.char_index is None
    var_15 = [var_14]
    var_16 = module_0.ListToken(var_0, var_0, var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_17 = {}
    var_18 = [var_17, var_17, var_17]
    var_19 = -1617
    var_20 = '8<$3 Hf'
    var_21 = module_0.Token(var_13, var_19, var_0, var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_22 = var_21.__eq__(var_0)
    assert var_22 is False
    var_23 = module_0.DictToken(*var_18)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_23.lookup_key(var_15)

def test_case_2():
    var_0 = None
    var_1 = None
    var_2 = False
    var_3 = True
    var_4 = module_0.Token(var_1, var_2, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.DictToken()

def test_case_4():
    var_0 = {}
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

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
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = -1044
    var_3 = module_0.ScalarToken(var_0, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_1.__repr__()

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_1.lookup_key(var_0)

def test_case_9():
    var_0 = None
    var_1 = None
    var_2 = module_0.ScalarToken(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8323099425944
    var_4 = var_3.__eq__(var_0)
    var_5 = ' hh'
    var_6 = -748
    var_7 = module_0.Token(var_5, var_6, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'

def test_case_10():
    var_0 = None
    var_1 = var_0.__repr__()
    assert var_1 == 'None'
    var_2 = -1044
    var_3 = module_0.ScalarToken(var_0, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = ()
    var_5 = module_0.ListToken(var_4, var_3, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6 = var_3.__eq__(var_5)
    assert var_6 is False
    var_7 = var_6.__repr__()
    assert var_7 == 'False'

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__hash__()
    assert var_2 == 8323099425944
    var_3 = module_0.Token(var_0, var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_2.__hash__()
    assert var_4 == 8323099425944
    var_3.__eq__(var_3)

def test_case_12():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_0.__hash__()
    assert var_2 == 8323099425944
    var_3 = var_2.__repr__()
    assert var_3 == '8323099425944'
    var_4 = module_0.ScalarToken(var_3, var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = var_2.__hash__()
    assert var_5 == 8323099425944
    var_6 = False
    var_7 = module_0.ScalarToken(var_0, var_6, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = var_1.__eq__(var_7)
    assert var_8 is False
    var_9 = var_5.__repr__()
    assert var_9 == '8323099425944'

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__hash__()
    assert var_2 == 8323099425944
    var_3 = var_1.__hash__()
    assert var_3 == 8323099425944
    var_4 = module_0.ListToken(var_2, var_0, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5 = var_3.__hash__()
    assert var_5 == 8323099425944
    var_6 = var_3.__repr__()
    assert var_6 == '8323099425944'
    var_7 = var_1.__eq__(var_3)
    assert var_7 is False
    var_8 = var_5.__repr__()
    assert var_8 == '8323099425944'
    var_9 = None
    var_10 = module_0.ScalarToken(var_9, var_9, var_9, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_11 = -1044
    var_12 = module_0.ScalarToken(var_9, var_11, var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_13 = module_0.ListToken(var_8, var_3, var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_12.__eq__(var_13)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 2148
    var_1 = None
    var_2 = module_1.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 2148
    assert var_2.column_no == 2148
    assert var_2.char_index is None
    var_3 = [var_2, var_2]
    var_4 = module_0.ListToken(var_1, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5 = {}
    var_6 = [var_5, var_5, var_5]
    var_7 = -1617
    var_8 = '8<$3 Hf'
    var_9 = module_0.Token(var_1, var_7, var_1, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_10 = module_0.DictToken(*var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_10.lookup_key(var_3)

def test_case_15():
    var_0 = None
    var_1 = 2148
    var_2 = module_0.Token(var_0, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_3 = False
    var_4 = module_0.ScalarToken(var_1, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = var_4.__hash__()
    assert var_5 == 2148
    var_6 = None
    var_7 = module_1.Position(var_1, var_1, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 2148
    assert var_7.column_no == 2148
    assert var_7.char_index is None
    var_8 = module_0.ListToken(var_0, var_0, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_9 = {var_4: var_3}
    var_10 = [var_9, var_9, var_9]
    var_11 = var_4.__eq__(var_0)
    assert var_11 is False
    var_12 = module_0.DictToken(*var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_13 = module_0.ListToken(var_0, var_5, var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

def test_case_16():
    var_0 = 'test content'
    var_1 = 3
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_3 = var_2.start
    var_4 = 'different'
    var_5 = module_0.Token(var_4, var_1, var_1, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    with pytest.raises(AttributeError):
        var_6 = var_4.value

def test_case_17():
    var_0 = 'test content'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_2, var_1, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_3.start
    var_5 = 'different'
    var_6 = module_0.Token(var_5, var_1, var_2, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_7 = repr(var_3)
    assert var_7 == "Token('test')"
    with pytest.raises(NotImplementedError):
        var_8 = var_3.value

def test_case_18():
    var_0 = 'test content'
    var_1 = 3
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_3 = var_2.start
    var_4 = var_2.end
    var_5 = module_0.Token(var_0, var_3, var_1, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_6 = repr(var_2)
    with pytest.raises(NotImplementedError):
        var_7 = var_2.value

def test_case_19():
    var_0 = 'test content'
    var_1 = 'test'
    var_2 = -3613
    var_3 = 3
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5 = var_4.start
    with pytest.raises(AttributeError):
        var_6 = var_3.end