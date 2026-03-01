# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0

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
    var_1 = [var_0]
    var_2 = 4553
    var_3 = module_0.ScalarToken(var_0, var_0, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.__hash__()
    assert var_4 == 7930287635096
    var_5 = var_3.__eq__(var_3)
    assert var_5 is True
    var_3.lookup_key(var_1)

def test_case_2():
    var_0 = 860
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = bool(var_1 == var_1)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.DictToken()

def test_case_4():
    var_0 = 860
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = bool(var_1 == var_1)
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

def test_case_6():
    var_0 = None
    var_1 = None
    var_2 = 837
    var_3 = module_0.ListToken(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4 = var_3.__repr__()
    assert var_4 == "ListToken('')"
    var_5 = var_3.__eq__(var_0)
    assert var_5 is False

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = -20
    var_1 = 4
    var_2 = module_0.ScalarToken(var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = bool(var_2 == var_2)
    assert var_3 is True
    var_2.lookup_key(var_3)

def test_case_8():
    var_0 = None
    var_1 = 4553
    var_2 = module_0.ScalarToken(var_0, var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 7930287635096
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = module_0.ListToken(var_3, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

def test_case_9():
    var_0 = None
    var_1 = 0
    var_2 = 'line1\nline2'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_3.end
    with pytest.raises(AttributeError):
        var_5 = var_4.index
    assert var_5 == 8

def test_case_10():
    var_0 = 6
    var_1 = 0
    var_2 = 'xyz'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    with pytest.raises(NotImplementedError):
        var_4 = bool(var_3 == var_3)
    assert var_4 is True

def test_case_11():
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

def test_case_12():
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

def test_case_13():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_tokens == {})
    assert var_11 is True

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_keys == {})
    assert var_11 is True
    var_12 = var_9._child_tokens
    var_13 = bool(var_9._child_tokens == {})
    assert var_13 is True
    var_14 = var_9._value
    var_15 = [var_13, var_4]
    var_9.lookup_key(var_15)

def test_case_15():
    var_0 = []
    var_1 = 1
    var_2 = 0
    var_3 = ''
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5 = module_0.ListToken(var_0, var_2, var_2, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6 = bool(not var_4 == var_5)
    assert var_6 is True

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {var_0: var_2, var_1: var_2, var_0: var_0}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    module_0.DictToken(*var_4, **var_5)

def test_case_17():
    var_0 = None
    var_1 = 10
    var_2 = False
    var_3 = module_0.Token(var_0, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = module_0.ListToken(var_0, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5 = module_0.Token(var_0, var_0, var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = var_4.start
    with pytest.raises(AttributeError):
        var_8 = var_7.column
    assert var_8 == 1

def test_case_18():
    var_0 = None
    var_1 = None
    var_2 = 3032
    var_3 = -3736
    var_4 = module_0.ScalarToken(var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = {var_4: var_4}
    var_7 = 17
    var_8 = None
    var_9 = '{}'
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = False
    var_16 = module_0.ListToken(var_8, var_15, var_8)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_17 = {var_11: var_6, var_12: var_7, var_13: var_7, var_14: var_9}
    var_18 = module_0.DictToken(*var_10, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_19 = var_4.__hash__()
    assert var_19 == 7930287635096
    with pytest.raises(AttributeError):
        var_20 = bool(var_9._child_keys == {})
    assert var_20 is True

def test_case_19():
    var_0 = 1
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = [var_3]
    var_5 = module_0.ListToken(var_4, var_1, var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6 = []
    var_7 = module_0.ListToken(var_6, var_1, var_1, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    with pytest.raises(NotImplementedError):
        var_8 = bool(not var_5 == var_7)
    assert var_8 is True