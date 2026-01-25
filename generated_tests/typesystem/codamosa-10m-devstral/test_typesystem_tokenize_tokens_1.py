# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
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
def test_case_1():
    var_0 = False
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = var_1.__hash__()
    assert var_5 == 0
    var_6 = [var_2]
    var_1.lookup_key(var_6)

def test_case_2():
    var_0 = False
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = var_1.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_1.__eq__(var_0)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.DictToken()

def test_case_4():
    var_0 = -3088
    var_1 = module_0.ListToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = -3153
    var_2 = '&O@?'
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = None
    var_5 = 665
    var_6 = module_0.Token(var_4, var_4, var_5, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_6.__repr__()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_1.lookup_key(var_0)

def test_case_7():
    var_0 = None
    var_1 = None
    var_2 = module_0.ScalarToken(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 7706063008664
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
def test_case_8():
    var_0 = {}
    var_1 = [var_0, var_0, var_0]
    var_2 = None
    var_3 = module_0.DictToken(*var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_4 = -522
    var_5 = module_0.ListToken(var_2, var_4, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6 = var_5.__eq__(var_2)
    assert var_6 is False
    var_7 = var_5.__eq__(var_2)
    assert var_7 is False
    var_5.lookup(var_1)

def test_case_9():
    var_0 = {}
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

def test_case_10():
    var_0 = False
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_0.ListToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_2 = False
    var_3 = module_0.ScalarToken(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3.__eq__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = False
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__hash__()
    assert var_2 == 0
    var_3 = None
    var_4 = module_0.ScalarToken(var_3, var_2, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = var_1.__eq__(var_4)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == "ScalarToken('')"
    var_7 = var_2.__eq__(var_0)
    assert var_7 is True
    var_8 = [var_7]
    var_4.lookup_key(var_8)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = -522
    var_2 = module_0.ListToken(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_3 = False
    var_4 = module_0.ScalarToken(var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = None
    var_6 = var_4.__eq__(var_5)
    assert var_6 is False
    var_7 = var_4.__repr__()
    assert var_7 == "ScalarToken('')"
    var_8 = module_0.ScalarToken(var_5, var_3, var_5, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = module_0.ScalarToken(var_0, var_0, var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_10 = var_8.__hash__()
    assert var_10 == 7706063008664
    var_11 = True
    var_12 = module_0.ScalarToken(var_4, var_11, var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_13 = var_8.__eq__(var_9)
    assert var_13 is False
    var_9.lookup_key(var_10)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = False
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = None
    var_3 = False
    var_4 = False
    var_5 = module_0.Token(var_2, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5.__eq__(var_5)

def test_case_15():
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = -522
    var_2 = module_0.ListToken(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_3 = False
    var_4 = module_0.ScalarToken(var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = None
    var_6 = var_4.__hash__()
    assert var_6 == 0
    var_7 = 2291
    var_8 = [var_2, var_2, var_7]
    var_9 = -2979
    var_10 = module_0.ListToken(var_0, var_3, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_11 = (var_8, var_7, var_10)
    var_12 = {var_3: var_1, var_4: var_11, var_9: var_2, var_5: var_0}
    var_13 = [var_12, var_1, var_5]
    module_0.DictToken(*var_13)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = module_0.DictToken(*var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_4 = None
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True
    var_6 = 66
    var_7 = module_0.ScalarToken(var_4, var_6, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3.lookup_key(var_1)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = {}
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = -522
    var_4 = "<'sX2+|w]Ut9"
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True
    var_6 = 'X:/\nCq0REN {N'
    var_7 = False
    var_8 = module_0.ScalarToken(var_4, var_3, var_7, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = var_8.__repr__()
    assert var_9 == "ScalarToken('X')"
    var_10 = [var_1]
    var_11 = module_0.DictToken(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_11.lookup_key(var_10)