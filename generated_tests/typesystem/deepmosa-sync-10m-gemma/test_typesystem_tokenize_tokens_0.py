# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0

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
    assert var_6 == 8670034069656
    var_7 = var_6.__hash__()
    assert var_7 == 8670034069656
    var_8 = var_7.__hash__()
    assert var_8 == 8670034069656
    var_9 = var_8.__hash__()
    assert var_9 == 8670034069656
    var_10 = var_9.__hash__()
    assert var_10 == 8670034069656
    var_11 = var_10.__hash__()
    assert var_11 == 8670034069656

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
    assert var_3 == 8670034069656
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
    assert var_4 == 8670034069656
    var_5 = var_4.__repr__()
    assert var_5 == '8670034069656'
    var_6 = module_0.ListToken(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6.lookup(var_1)

def test_case_9():
    var_0 = 1
    var_1 = 0
    var_2 = '1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = [var_3]
    var_5 = module_0.ListToken(var_4, var_1, var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    with pytest.raises(NotImplementedError):
        var_6 = bool(var_5 == var_5)
    assert var_6 is True

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
    var_0 = True
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__repr__()
    assert var_2 == "ScalarToken('')"
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_4 = var_3.__hash__()
    assert var_4 == 1
    var_5 = var_1.__hash__()
    assert var_5 == 1
    var_6 = var_4.__repr__()
    assert var_6 == '1'
    module_0.DictToken()

def test_case_12():
    var_0 = -2100
    var_1 = '4l826T(>P'
    var_2 = module_0.Token(var_0, var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_3 = []
    var_4 = module_0.ListToken(var_3, var_0, var_0, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5 = bool(var_4 == var_4)
    assert var_5 is True

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
    var_6 = var_5.__hash__()
    assert var_6 == 1
    var_7 = module_0.Token(var_0, var_0, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_8 = var_6.__hash__()
    assert var_8 == 1
    var_9 = var_5.__hash__()
    assert var_9 == 1
    var_10 = {var_9: var_5, var_0: var_0, var_0: var_0}
    var_11 = var_10.__eq__(var_5)
    var_12 = [var_10, var_3, var_2]
    var_13 = var_5.__hash__()
    assert var_13 == 1
    var_14 = var_6.__eq__(var_5)
    assert var_14 is True
    module_0.DictToken(*var_12)

def test_case_14():
    var_0 = None
    var_1 = True
    var_2 = var_1.__repr__()
    assert var_2 == 'True'
    var_3 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = module_0.Token(var_0, var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5 = var_3.__hash__()
    assert var_5 == 8670034069656
    var_6 = {}
    var_7 = var_6.__eq__(var_0)
    var_8 = [var_6, var_2, var_5]
    var_9 = module_0.DictToken(*var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = {var_0, var_2}
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False
    var_5 = var_2.__repr__()
    assert var_5 == "ScalarToken('')"
    var_6 = var_2.__eq__(var_2)
    assert var_6 is True
    var_7 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = module_0.Token(var_0, var_0, var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_9 = var_7.__hash__()
    assert var_9 == 8670034069656
    var_10 = module_0.ScalarToken(var_9, var_9, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_11 = {}
    var_12 = var_11.__eq__(var_0)
    var_13 = var_6.__hash__()
    assert var_13 == 1
    var_14 = var_2.__eq__(var_7)
    assert var_14 is False
    var_15 = var_7.__hash__()
    assert var_15 == 8670034069656
    var_8.__repr__()

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8670034069656
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = module_0.Token(var_0, var_0, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_7 = var_5.__hash__()
    assert var_7 == 8670034069656
    var_8 = module_0.ScalarToken(var_7, var_7, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = {}
    var_10 = var_9.__eq__(var_0)
    var_11 = var_6.__eq__(var_0)
    assert var_11 is False
    var_12 = None
    var_13 = var_7.__eq__(var_12)
    var_14 = var_4.__hash__()
    assert var_14 == 1
    var_15 = var_7.__hash__()
    assert var_15 == 8670034069656
    var_16 = var_15.__hash__()
    assert var_16 == 8670034069656
    var_17 = var_8.__hash__()
    assert var_17 == 8670034069656
    var_18 = False
    var_19 = var_2.__eq__(var_8)
    assert var_19 is False
    var_20 = module_0.ScalarToken(var_15, var_14, var_15)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_21 = var_18.__repr__()
    assert var_21 == 'False'
    module_0.DictToken(*var_15)

def test_case_17():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = module_0.Token(var_0, var_0, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_7 = var_5.__hash__()
    assert var_7 == 8670034069656
    var_8 = var_5.__hash__()
    assert var_8 == 8670034069656
    var_9 = {var_5: var_7}
    var_10 = var_9.__eq__(var_0)
    var_11 = [var_9, var_3, var_2]
    var_12 = None
    var_13 = None
    var_14 = var_7.__eq__(var_12)
    var_15 = var_4.__hash__()
    assert var_15 == 1
    var_16 = var_7.__hash__()
    assert var_16 == 8670034069656
    var_17 = var_16.__hash__()
    assert var_17 == 8670034069656
    var_18 = True
    var_19 = var_2.__eq__(var_13)
    assert var_19 is False
    var_20 = var_2.__hash__()
    assert var_20 == 8670034069656
    var_21 = var_18.__repr__()
    assert var_21 == 'True'
    var_22 = module_0.DictToken(*var_11)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'