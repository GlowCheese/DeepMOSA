# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0
import builtins as module_1

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
    assert var_6 == 8262497395352
    var_7 = var_6.__hash__()
    assert var_7 == 8262497395352
    var_8 = var_7.__hash__()
    assert var_8 == 8262497395352
    var_9 = var_8.__hash__()
    assert var_9 == 8262497395352
    var_10 = var_9.__hash__()
    assert var_10 == 8262497395352
    var_11 = var_10.__hash__()
    assert var_11 == 8262497395352

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = {var_1: var_1, var_2: var_1, var_2: var_1, var_0: var_0, var_0: var_0}
    var_4 = [var_3, var_0, var_2]
    module_0.DictToken(*var_4)

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
    var_1 = True
    var_2 = 951
    var_3 = 'f5X6l,T}O"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5 = var_4.__repr__()
    assert var_5 == 'Token(\'5X6l,T}O"\')'
    var_6 = None
    var_7 = True
    var_8 = module_0.ScalarToken(var_6, var_7, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = module_0.Token(var_6, var_6, var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_10 = var_9.__eq__(var_6)
    assert var_10 is False
    var_9.lookup_key(var_6)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = -4187.0
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.ScalarToken(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.__hash__()
    assert var_4 == 8262497395352
    var_5 = var_4.__repr__()
    assert var_5 == '8262497395352'
    var_6 = module_0.ListToken(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6.lookup(var_1)

def test_case_8():
    var_0 = True
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = module_0.Token(var_0, var_0, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_7 = var_2.__eq__(var_2)
    assert var_7 is True
    var_8 = var_5.__hash__()
    assert var_8 == 8262497395352
    var_9 = {var_5: var_6}
    var_10 = {}
    var_11 = module_0.ScalarToken(var_4, var_4, var_0, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_12 = var_8.__eq__(var_3)
    var_13 = [var_9, var_3, var_2]
    var_14 = var_3.__hash__()
    assert var_14 == -8519004450457363868
    var_15 = module_1.object()
    var_16 = module_0.DictToken(*var_13)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_16.__eq__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
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
def test_case_11():
    var_0 = None
    var_1 = True
    var_2 = var_0.__repr__()
    assert var_2 == 'None'
    var_3 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = module_0.ListToken(var_2, var_3, var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5 = {var_3: var_4}
    var_6 = [var_5, var_2, var_2]
    var_7 = module_0.DictToken(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_7.__eq__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = -982
    var_1 = None
    var_2 = 'L++<;M3=_C'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4 = None
    var_5 = True
    var_6 = module_0.ScalarToken(var_4, var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = var_6.__repr__()
    assert var_7 == "ScalarToken('')"
    var_8 = var_6.__eq__(var_6)
    assert var_8 is True
    var_9 = var_6.__hash__()
    assert var_9 == 8262497395352
    var_10 = var_9.__hash__()
    assert var_10 == 8262497395352
    var_11 = module_0.ScalarToken(var_4, var_9, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_12 = module_0.Token(var_9, var_4, var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_13 = var_11.__hash__()
    assert var_13 == 8262497395352
    var_14 = var_13.__hash__()
    assert var_14 == 8262497395352
    var_15 = var_10.__hash__()
    assert var_15 == 8262497395352
    var_16 = {var_15: var_10, var_9: var_9, var_9: var_9}
    var_17 = var_16.__eq__(var_10)
    var_18 = var_6.__hash__()
    assert var_18 == 8262497395352
    var_19 = var_15.__hash__()
    assert var_19 == 8262497395352
    var_20 = var_6.__eq__(var_11)
    assert var_20 is False
    var_21 = var_18.__hash__()
    assert var_21 == 8262497395352
    var_22 = var_21.__hash__()
    assert var_22 == 8262497395352
    module_0.DictToken()

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = var_4.__hash__()
    assert var_5 == 1
    var_6 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = var_2.__eq__(var_5)
    assert var_7 is False
    var_8 = var_6.__hash__()
    assert var_8 == 8262497395352
    var_9 = var_5.__repr__()
    assert var_9 == '1'
    var_10 = {var_6: var_6}
    var_11 = {}
    var_12 = module_0.ScalarToken(var_9, var_9, var_0, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_13 = var_8.__eq__(var_3)
    var_14 = [var_10, var_3, var_2]
    var_15 = var_9.__hash__()
    assert var_15 == -4984512977167346622
    var_16 = module_0.DictToken(*var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_17 = var_16.__eq__(var_2)
    assert var_17 is False
    module_0.DictToken()

def test_case_14():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = {}
    var_4 = var_2.__hash__()
    assert var_4 == 8262497395352
    var_5 = [var_3, var_4, var_4]
    var_6 = var_4.__hash__()
    assert var_6 == 8262497395352
    var_7 = var_6.__hash__()
    assert var_7 == 8262497395352
    var_8 = module_0.DictToken(*var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = var_2.__eq__(var_4)
    assert var_5 is False
    var_6 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = module_0.ListToken(var_3, var_5, var_0, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_8 = var_2.__eq__(var_5)
    assert var_8 is False
    var_9 = var_6.__hash__()
    assert var_9 == 8262497395352
    var_10 = var_5.__repr__()
    assert var_10 == 'False'
    var_11 = {var_6: var_7}
    var_12 = module_0.ScalarToken(var_10, var_10, var_0, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_13 = var_9.__eq__(var_3)
    var_14 = var_9.__eq__(var_9)
    assert var_14 is True
    var_15 = [var_11, var_3, var_2]
    var_16 = var_10.__hash__()
    assert var_16 == -5630973282200761205
    var_17 = module_1.object()
    var_18 = module_0.DictToken(*var_15)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_19 = var_9.__eq__(var_16)
    assert var_19 is False
    var_18.lookup(var_15)

def test_case_16():
    var_0 = None
    var_1 = 5
    var_2 = 'test'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    with pytest.raises(AttributeError):
        var_4 = var_3.end._index
    assert var_4 == 5

def test_case_17():
    var_0 = None
    var_1 = 5
    var_2 = 'test'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    with pytest.raises(AttributeError):
        var_4 = var_3.start._index
    assert var_4 == 2

def test_case_18():
    var_0 = None
    var_1 = -20
    var_2 = 2
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    with pytest.raises(AttributeError):
        var_5 = var_4.start._index
    assert var_5 == 2