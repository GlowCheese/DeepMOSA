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
    assert var_6 == 8498232589720
    var_7 = var_6.__hash__()
    assert var_7 == 8498232589720
    var_8 = var_7.__hash__()
    assert var_8 == 8498232589720
    var_9 = var_8.__hash__()
    assert var_9 == 8498232589720
    var_10 = var_9.__hash__()
    assert var_10 == 8498232589720
    var_11 = var_10.__hash__()
    assert var_11 == 8498232589720

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
    assert var_4 == 8498232589720
    var_5 = var_4.__repr__()
    assert var_5 == '8498232589720'
    var_6 = module_0.ListToken(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6.lookup(var_1)

def test_case_9():
    var_0 = True
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

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
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = {var_1: var_1, var_0: var_0, var_0: var_0}
    var_6 = var_0.__eq__(var_3)
    var_7 = [var_5, var_3, var_2]
    var_8 = var_6.__hash__()
    assert var_8 == 8498232589721
    var_9 = var_0.__hash__()
    assert var_9 == 8498232589720
    var_10 = var_2.__eq__(var_9)
    assert var_10 is False
    var_11 = var_10.__hash__()
    assert var_11 == 0
    module_0.DictToken(*var_7)

def test_case_14():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8498232589720
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = {}
    var_6 = [var_5, var_3, var_2]
    var_7 = var_4.__hash__()
    assert var_7 == 1
    var_8 = var_3.__hash__()
    assert var_8 == 8498232589720
    var_9 = var_3.__hash__()
    assert var_9 == 8498232589720
    var_10 = var_3.__hash__()
    assert var_10 == 8498232589720
    var_11 = module_1.object()
    var_12 = module_0.DictToken(*var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

def test_case_15():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8498232589720
    var_4 = {}
    var_5 = [var_4, var_1, var_2]
    var_6 = var_1.__hash__()
    assert var_6 == 0
    var_7 = var_2.__hash__()
    assert var_7 == 8498232589720
    var_8 = var_2.__hash__()
    assert var_8 == 8498232589720
    var_9 = var_8.__hash__()
    assert var_9 == 8498232589720
    var_10 = module_0.DictToken(*var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_11 = var_10.__eq__(var_10)
    assert var_11 is True
    var_12 = var_7.__hash__()
    assert var_12 == 8498232589720

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8498232589720
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = var_2.__repr__()
    assert var_5 == "ScalarToken('')"
    var_6 = {var_2: var_0}
    var_7 = [var_6, var_4, var_2]
    var_8 = var_2.__repr__()
    assert var_8 == "ScalarToken('')"
    var_9 = var_2.__hash__()
    assert var_9 == 8498232589720
    var_10 = var_2.__hash__()
    assert var_10 == 8498232589720
    var_11 = var_10.__hash__()
    assert var_11 == 8498232589720
    var_12 = module_0.DictToken(*var_7)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_13 = var_11.__hash__()
    assert var_13 == 8498232589720
    var_12.__eq__(var_12)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_2.__repr__()
    assert var_4 == "ScalarToken('')"
    var_5 = {}
    var_6 = [var_5, var_3, var_2]
    var_7 = var_2.__repr__()
    assert var_7 == "ScalarToken('')"
    var_8 = var_2.__hash__()
    assert var_8 == 8498232589720
    var_9 = var_7.__hash__()
    assert var_9 == 9048023310776595122
    var_10 = module_0.DictToken(*var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_11 = var_10.__eq__(var_10)
    assert var_11 is True
    var_12 = var_7.__repr__()
    assert var_12 == '"ScalarToken(\'\')"'
    var_10.lookup(var_6)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == "ScalarToken('')"
    var_5 = {}
    var_6 = [var_5, var_3, var_2]
    var_7 = var_3.__hash__()
    assert var_7 == 0
    var_8 = var_2.__hash__()
    assert var_8 == 8498232589720
    var_9 = var_2.__hash__()
    assert var_9 == 8498232589720
    var_10 = var_9.__hash__()
    assert var_10 == 8498232589720
    var_11 = module_0.DictToken(*var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_12 = var_11.__eq__(var_11)
    assert var_12 is True
    var_13 = var_9.__eq__(var_9)
    assert var_13 is True
    var_14 = var_11.__eq__(var_2)
    assert var_14 is False
    var_15 = var_10.__hash__()
    assert var_15 == 8498232589720
    module_0.DictToken(*var_7, **var_5)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = module_0.Token(var_0, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_2.__hash__()
    assert var_4 == 8498232589720
    var_5 = var_2.__repr__()
    assert var_5 == "ScalarToken('')"
    var_6 = {}
    var_7 = [var_6, var_4, var_2]
    var_8 = var_4.__hash__()
    assert var_8 == 8498232589720
    var_9 = var_2.__hash__()
    assert var_9 == 8498232589720
    var_10 = var_9.__hash__()
    assert var_10 == 8498232589720
    var_11 = module_0.DictToken(*var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_12 = var_11.__eq__(var_11)
    assert var_12 is True
    var_13 = 1819
    var_14 = module_0.ScalarToken(var_9, var_4, var_13, var_10)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_15 = var_10.__hash__()
    assert var_15 == 8498232589720
    var_16 = module_0.Token(var_0, var_8, var_1)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_17 = var_2.__hash__()
    assert var_17 == 8498232589720
    var_18 = -3945
    var_19 = (var_18,)
    var_20 = [var_19]
    var_11.lookup_key(var_20)