# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

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

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = -124
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8222983486616
    var_4 = var_3.__hash__()
    assert var_4 == 8222983486616
    var_5 = module_0.ScalarToken(var_0, var_3, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = {var_3: var_0, var_5: var_0, var_2: var_5, var_3: var_1}
    var_7 = [var_6]
    var_5.lookup_key(var_7)

def test_case_2():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8222983486616
    var_4 = var_2.__eq__(var_1)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.DictToken()

def test_case_4():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_5():
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
def test_case_6():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = module_0.Token(var_0, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_3.lookup_key(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = -4187.0
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.ScalarToken(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.__hash__()
    assert var_4 == 8222983486616
    var_5 = var_4.__repr__()
    assert var_5 == '8222983486616'
    var_6 = module_0.ListToken(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6.lookup(var_1)

def test_case_8():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    with pytest.raises(NotImplementedError):
        var_4 = var_3.value

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = -124
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_1.__hash__()
    assert var_3 == -124
    var_4 = var_3.__hash__()
    assert var_4 == -124
    var_5 = True
    var_6 = module_0.ScalarToken(var_0, var_1, var_5, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = {var_4: var_0, var_6: var_0, var_2: var_6, var_4: var_1}
    var_8 = [var_7]
    var_9 = module_0.Token(var_0, var_0, var_0, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_10 = module_0.Token(var_8, var_5, var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_9.lookup_key(var_8)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = -139
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8222983486616
    var_4 = var_3.__hash__()
    assert var_4 == 8222983486616
    var_5 = var_4.__hash__()
    assert var_5 == 8222983486616
    var_6 = True
    var_7 = module_0.ScalarToken(var_4, var_3, var_6, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = {var_5: var_0, var_7: var_0, var_2: var_7, var_5: var_1}
    var_9 = [var_8]
    var_10 = module_0.Token(var_0, var_0, var_0, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_10.lookup_key(var_9)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8222983486616
    var_4 = {var_1: var_0, var_0: var_1}
    var_5 = var_2.__eq__(var_4)
    assert var_5 is False
    var_6 = ' UTvgv(c\n\t,U1hX]g'
    var_7 = [var_4, var_4, var_6, var_1]
    module_0.DictToken(*var_7)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = -124
    var_2 = False
    var_3 = 'mV8'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5 = var_4.__repr__()
    assert var_5 == "ListToken('m')"
    var_6 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4.__eq__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = 1545
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = -502
    var_5 = module_0.ListToken(var_2, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6 = {}
    var_7 = var_2.__hash__()
    assert var_7 == 8222983486616
    var_8 = ' UTvgv(c\n\t,U1hX]g'
    var_9 = [var_6, var_6, var_8, var_7]
    var_10 = module_0.DictToken(*var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_11 = 'HT+\t:g:puWc'
    var_12 = var_7.__hash__()
    assert var_12 == 8222983486616
    var_13 = var_12.__hash__()
    assert var_13 == 8222983486616
    var_14 = var_11.__hash__()
    assert var_14 == -5081378271157335289
    var_15 = [var_14]
    var_16 = module_0.Token(var_8, var_15, var_6)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_13.lookup_key(var_9)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = 1545
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8222983486616
    var_4 = var_3.__hash__()
    assert var_4 == 8222983486616
    var_5 = var_2.__repr__()
    assert var_5 == "ScalarToken('')"
    var_6 = -502
    var_7 = {}
    var_8 = var_2.__eq__(var_0)
    assert var_8 is False
    var_9 = ' UTvgv(c\n\t,U1hX]g'
    var_10 = [var_7, var_7, var_9, var_8]
    var_11 = module_0.DictToken(*var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_12 = 'HT+\t:g:puWc'
    var_13 = module_0.Token(var_3, var_0, var_4, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_14 = module_0.ScalarToken(var_0, var_4, var_4, var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_15 = var_13.__eq__(var_0)
    assert var_15 is False
    var_16 = var_14.__hash__()
    assert var_16 == 8222983486616
    var_17 = var_16.__hash__()
    assert var_17 == 8222983486616
    var_18 = var_17.__hash__()
    assert var_18 == 8222983486616
    var_19 = True
    var_20 = module_0.ScalarToken(var_0, var_16, var_19, var_17)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_21 = var_16.__hash__()
    assert var_21 == 8222983486616
    var_22 = module_0.Token(var_16, var_16, var_6, var_18)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_11.lookup_key(var_10)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = var_0.__hash__()
    assert var_1 == 8222983486616
    var_2 = var_1.__repr__()
    assert var_2 == '8222983486616'
    var_3 = module_0.ListToken(var_1, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4 = {}
    var_5 = [var_4, var_4, var_0, var_1]
    var_6 = module_0.DictToken(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_7 = module_0.Token(var_1, var_0, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_8 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = var_8.__eq__(var_6)
    assert var_9 is False
    var_10 = var_9.__hash__()
    assert var_10 == 0
    var_11 = module_0.ScalarToken(var_0, var_0, var_10, var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_12 = var_9.__hash__()
    assert var_12 == 0
    var_1.lookup_key(var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = 1545
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8222983486616
    var_4 = var_3.__hash__()
    assert var_4 == 8222983486616
    var_5 = var_2.__repr__()
    assert var_5 == "ScalarToken('')"
    var_6 = module_0.ListToken(var_3, var_2, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_7 = {}
    var_8 = ' UTvgv(c\n\t,U1hX]g'
    var_9 = [var_7, var_7, var_8, var_4]
    var_10 = module_0.DictToken(*var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_11 = module_0.Token(var_4, var_0, var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_12 = module_0.ScalarToken(var_0, var_4, var_4, var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_13 = var_12.__eq__(var_10)
    assert var_13 is False
    var_14 = var_13.__hash__()
    assert var_14 == 0
    var_15 = True
    var_16 = module_0.ScalarToken(var_0, var_0, var_15, var_13)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_17 = {var_14: var_0, var_16: var_0, var_12: var_16, var_14: var_15}
    var_18 = var_13.__hash__()
    assert var_18 == 0
    var_19 = [var_17]
    var_20 = {}
    var_21 = 2415
    var_22 = module_0.Token(var_20, var_13, var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_23 = 736
    var_24 = module_0.Token(var_18, var_23, var_0)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_10.lookup_key(var_19)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = 1576
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8222983486616
    var_4 = var_3.__hash__()
    assert var_4 == 8222983486616
    var_5 = var_2.__repr__()
    assert var_5 == "ScalarToken('')"
    var_6 = module_0.ListToken(var_3, var_2, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_7 = {var_2: var_3}
    var_8 = ' UTvgv(c\n\t,U1hX]g'
    var_9 = [var_7, var_7, var_8, var_4]
    var_10 = module_0.DictToken(*var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_11 = module_0.Token(var_4, var_0, var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_12 = module_0.ScalarToken(var_0, var_4, var_4, var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_12.__eq__(var_10)

def test_case_18():
    var_0 = 'test'
    var_1 = 3
    var_2 = module_0.ListToken(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    with pytest.raises(AttributeError):
        var_3 = var_2.value

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = ''
    var_1 = 25
    var_2 = 1065
    var_3 = module_0.ListToken(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4 = var_3.value
    var_5 = 1
    var_6 = 4
    var_7 = module_1.Position(var_5, var_6, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 1
    assert var_7.column_no == 4
    assert var_7.char_index == 25
    var_8 = module_1.Position(var_5, var_6, var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == 1
    assert var_8.column_no == 4
    assert var_8.char_index == 1065
    var_9 = 0
    var_10 = [var_9]
    var_3.lookup(var_10)