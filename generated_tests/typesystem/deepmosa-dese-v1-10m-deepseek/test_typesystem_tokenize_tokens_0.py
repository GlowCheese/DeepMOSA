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
    assert var_6 == 8443204990104
    var_7 = var_6.__hash__()
    assert var_7 == 8443204990104
    var_8 = var_7.__hash__()
    assert var_8 == 8443204990104
    var_9 = var_8.__hash__()
    assert var_9 == 8443204990104
    var_10 = var_9.__hash__()
    assert var_10 == 8443204990104
    var_11 = var_10.__hash__()
    assert var_11 == 8443204990104

def test_case_2():
    var_0 = None
    var_1 = None
    var_2 = 'd,#rj[ZsdI*'
    var_3 = module_0.Token(var_1, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = None
    var_6 = -607
    var_7 = module_0.ListToken(var_5, var_6, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_8 = var_7.__eq__(var_5)
    assert var_8 is False

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

def test_case_7():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8443204990104
    var_4 = var_2.__eq__(var_1)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = -4187.0
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.ScalarToken(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.__hash__()
    assert var_4 == 8443204990104
    var_5 = var_4.__repr__()
    assert var_5 == '8443204990104'
    var_6 = module_0.ListToken(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6.lookup(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '>)N='
    var_1 = -44
    var_2 = True
    var_3 = module_0.ListToken(var_0, var_2, var_1, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_3.__eq__(var_3)

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
    var_5.__eq__(var_5)

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
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = None
    var_6 = [var_5, var_5]
    module_0.DictToken(*var_6)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_3.__hash__()
    assert var_4 == 1
    var_5 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = var_5.__repr__()
    assert var_6 == "ScalarToken('')"
    var_7 = var_5.__hash__()
    assert var_7 == 8443204990104
    var_8 = var_4.__hash__()
    assert var_8 == 1
    var_9 = {var_8: var_4, var_0: var_0, var_0: var_0}
    var_10 = var_9.__eq__(var_4)
    var_11 = [var_9, var_7, var_2]
    var_12 = var_4.__hash__()
    assert var_12 == 1
    var_13 = var_7.__hash__()
    assert var_13 == 8443204990104
    var_14 = var_13.__hash__()
    assert var_14 == 8443204990104
    var_15 = var_4.__repr__()
    assert var_15 == '1'
    module_0.DictToken(*var_11)

@pytest.mark.xfail(strict=True)
def test_case_15():
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
    assert var_9 == 8443204990104
    var_10 = var_9.__hash__()
    assert var_10 == 8443204990104
    var_11 = module_0.ScalarToken(var_4, var_9, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_12 = module_0.Token(var_9, var_4, var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_13 = var_11.__hash__()
    assert var_13 == 8443204990104
    var_14 = var_13.__hash__()
    assert var_14 == 8443204990104
    var_15 = var_10.__hash__()
    assert var_15 == 8443204990104
    var_16 = {var_15: var_10, var_9: var_9, var_9: var_9}
    var_17 = var_16.__eq__(var_10)
    var_18 = var_6.__hash__()
    assert var_18 == 8443204990104
    var_19 = var_15.__hash__()
    assert var_19 == 8443204990104
    var_20 = var_6.__eq__(var_11)
    assert var_20 is False
    var_21 = var_18.__hash__()
    assert var_21 == 8443204990104
    var_22 = var_21.__hash__()
    assert var_22 == 8443204990104
    module_0.DictToken()

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = var_4.__hash__()
    assert var_5 == 1
    var_6 = module_0.Token(var_0, var_0, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_7 = var_5.__hash__()
    assert var_7 == 1
    var_8 = {}
    var_9 = var_8.__eq__(var_5)
    var_10 = [var_8, var_3, var_2]
    var_11 = var_5.__hash__()
    assert var_11 == 1
    var_12 = var_11.__hash__()
    assert var_12 == 1
    var_13 = var_5.__repr__()
    assert var_13 == '1'
    var_14 = None
    var_15 = module_0.ListToken(var_1, var_1, var_12)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_16 = var_5.__repr__()
    assert var_16 == '1'
    var_17 = module_0.DictToken(*var_10)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_14.lookup_key(var_0)

@pytest.mark.xfail(strict=True)
def test_case_17():
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
    var_7 = var_6.__repr__()
    assert var_7 == "ScalarToken('')"
    var_8 = module_0.Token(var_0, var_0, var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_9 = var_6.__hash__()
    assert var_9 == 8443204990104
    var_10 = var_5.__hash__()
    assert var_10 == 1
    var_11 = {var_2: var_4}
    var_12 = var_11.__eq__(var_5)
    var_13 = [var_11, var_3, var_2]
    var_14 = None
    var_15 = var_5.__hash__()
    assert var_15 == 1
    var_16 = var_9.__hash__()
    assert var_16 == 8443204990104
    var_17 = var_16.__hash__()
    assert var_17 == 8443204990104
    var_18 = var_13.__repr__()
    assert var_18 == '[{ScalarToken(\'\'): True}, "ScalarToken(\'\')", ScalarToken(\'\')]'
    var_19 = module_0.ListToken(var_1, var_1, var_16)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_20 = var_5.__repr__()
    assert var_20 == '1'
    var_21 = module_0.DictToken(*var_13)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2.lookup_key(var_14)

@pytest.mark.xfail(strict=True)
def test_case_18():
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
    var_6 = var_5.__hash__()
    assert var_6 == 1
    var_7 = var_2.__repr__()
    assert var_7 == "ScalarToken('')"
    var_8 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = module_0.Token(var_0, var_0, var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_10 = module_0.ScalarToken(var_2, var_6, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_11 = var_6.__hash__()
    assert var_11 == 1
    var_12 = {}
    var_13 = var_6.__eq__(var_10)
    var_14 = [var_12, var_5, var_6]
    var_15 = var_2.__eq__(var_10)
    assert var_15 is False
    var_16 = var_10.__hash__()
    assert var_16 == 8443204990104
    var_17 = var_6.__hash__()
    assert var_17 == 1
    var_18 = var_10.__hash__()
    assert var_18 == 8443204990104
    var_19 = var_18.__hash__()
    assert var_19 == 8443204990104
    var_20 = var_14.__repr__()
    assert var_20 == '[{}, True, 1]'
    var_21 = module_0.ListToken(var_1, var_1, var_18)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_22 = var_6.__repr__()
    assert var_22 == '1'
    module_0.DictToken(*var_11)

@pytest.mark.xfail(strict=True)
def test_case_19():
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
    var_7 = module_0.Token(var_0, var_0, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_8 = var_6.__hash__()
    assert var_8 == 8443204990104
    var_9 = var_5.__hash__()
    assert var_9 == 1
    var_10 = {}
    var_11 = var_2.__hash__()
    assert var_11 == 8443204990104
    var_12 = [var_10, var_3, var_2]
    var_13 = module_0.ScalarToken(var_8, var_0, var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_14 = var_5.__hash__()
    assert var_14 == 1
    var_15 = var_8.__hash__()
    assert var_15 == 8443204990104
    var_16 = var_15.__hash__()
    assert var_16 == 8443204990104
    var_17 = var_12.__repr__()
    assert var_17 == '[{}, "ScalarToken(\'\')", ScalarToken(\'\')]'
    var_18 = var_16.__repr__()
    assert var_18 == '8443204990104'
    var_19 = module_0.ListToken(var_1, var_1, var_15)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_20 = var_5.__repr__()
    assert var_20 == '1'
    var_21 = module_0.DictToken(*var_12)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_22 = [var_14, var_19, var_10]
    var_21.lookup_key(var_22)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = True
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
    var_7 = module_0.Token(var_0, var_0, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_8 = var_6.__hash__()
    assert var_8 == 8443204990104
    var_9 = var_5.__hash__()
    assert var_9 == 1
    var_10 = {}
    var_11 = var_10.__eq__(var_5)
    var_12 = [var_10, var_3, var_2]
    var_13 = None
    var_14 = None
    var_15 = var_5.__hash__()
    assert var_15 == 1
    var_16 = 2334
    var_17 = module_0.ScalarToken(var_8, var_16, var_15)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_18 = var_17.__hash__()
    assert var_18 == 8443204990104
    var_19 = var_5.__hash__()
    assert var_19 == 1
    var_20 = var_19.__repr__()
    assert var_20 == '1'
    var_21 = -1337
    var_22 = module_0.ListToken(var_14, var_5, var_21, var_9)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_23 = var_2.__repr__()
    assert var_23 == "ScalarToken('')"
    var_24 = module_0.DictToken(*var_12)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_25 = [var_13]
    var_24.lookup_key(var_25)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = ''
    var_1 = False
    var_2 = None
    var_3 = module_0.Token(var_2, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = module_0.ListToken(var_0, var_1, var_3, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5.__eq__(var_5)