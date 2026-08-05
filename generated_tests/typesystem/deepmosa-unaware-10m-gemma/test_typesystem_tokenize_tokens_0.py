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
    assert var_6 == 7920157196696
    var_7 = var_6.__hash__()
    assert var_7 == 7920157196696
    var_8 = var_7.__hash__()
    assert var_8 == 7920157196696
    var_9 = var_8.__hash__()
    assert var_9 == 7920157196696
    var_10 = var_9.__hash__()
    assert var_10 == 7920157196696
    var_11 = var_10.__hash__()
    assert var_11 == 7920157196696

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
    var_3 = var_1.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = -4187.0
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.ScalarToken(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.__hash__()
    assert var_4 == 7920157196696
    var_5 = var_4.__repr__()
    assert var_5 == '7920157196696'
    var_6 = module_0.ListToken(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6.lookup(var_1)

def test_case_9():
    var_0 = False
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_10():
    var_0 = None
    var_1 = 0
    var_2 = 'xd&](+\\z7&ZqM]f'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    with pytest.raises(NotImplementedError):
        var_4 = var_3.value

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
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_2.__hash__()
    assert var_4 == 7920157196696
    var_5 = module_0.ListToken(var_3, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5.__eq__(var_2)

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
    assert var_8 == 7920157196697
    var_9 = var_0.__hash__()
    assert var_9 == 7920157196696
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
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = {}
    var_5 = [var_4, var_1, var_2]
    var_6 = var_2.__hash__()
    assert var_6 == 7920157196696
    var_7 = var_2.__hash__()
    assert var_7 == 7920157196696
    var_8 = var_7.__hash__()
    assert var_8 == 7920157196696
    var_9 = module_0.DictToken(*var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = {}
    var_5 = [var_4, var_1, var_2]
    var_6 = var_1.__hash__()
    assert var_6 == 0
    var_7 = var_2.__hash__()
    assert var_7 == 7920157196696
    var_8 = var_7.__hash__()
    assert var_8 == 7920157196696
    var_9 = module_0.DictToken(*var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_10 = var_9.__eq__(var_9)
    assert var_10 is True
    module_0.DictToken()

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = {}
    var_5 = [var_4, var_1, var_2]
    var_6 = var_1.__hash__()
    assert var_6 == 0
    var_7 = var_2.__hash__()
    assert var_7 == 7920157196696
    var_8 = var_2.__hash__()
    assert var_8 == 7920157196696
    var_9 = var_8.__hash__()
    assert var_9 == 7920157196696
    var_10 = module_0.DictToken(*var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_11 = module_0.ListToken(var_7, var_6, var_7, var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_12 = var_10.__eq__(var_10)
    assert var_12 is True
    var_13 = [var_8]
    var_10.lookup_key(var_13)

def test_case_17():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 7920157196696
    var_4 = var_3.__hash__()
    assert var_4 == 7920157196696
    var_5 = module_0.ListToken(var_4, var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = var_6.__eq__(var_6)
    assert var_7 is True
    var_8 = var_6.__repr__()
    assert var_8 == "ScalarToken('')"
    var_9 = {}
    var_10 = [var_9, var_7, var_6]
    var_11 = var_7.__hash__()
    assert var_11 == 1
    var_12 = var_6.__hash__()
    assert var_12 == 7920157196696
    var_13 = var_6.__hash__()
    assert var_13 == 7920157196696
    var_14 = var_13.__hash__()
    assert var_14 == 7920157196696
    var_15 = module_0.DictToken(*var_10)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_16 = var_15.__eq__(var_15)
    assert var_16 is True
    var_17 = var_2.__eq__(var_6)
    assert var_17 is False
    var_18 = var_14.__hash__()
    assert var_18 == 7920157196696

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_2.__repr__()
    assert var_4 == "ScalarToken('')"
    var_5 = {var_2: var_2}
    var_6 = [var_5, var_3, var_2]
    var_7 = var_2.__hash__()
    assert var_7 == 7920157196696
    var_8 = var_4.__hash__()
    assert var_8 == -393108282781882309
    var_9 = module_0.DictToken(*var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_10 = var_9.__eq__(var_9)
    assert var_10 is True
    var_11 = var_6.__repr__()
    assert var_11 == "[{ScalarToken(''): ScalarToken('')}, True, ScalarToken('')]"
    var_9.lookup(var_6)

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
    var_4 = var_2.__repr__()
    assert var_4 == "ScalarToken('')"
    var_5 = var_2.__repr__()
    assert var_5 == "ScalarToken('')"
    var_6 = {}
    var_7 = [var_6, var_4, var_2]
    var_8 = var_4.__hash__()
    assert var_8 == -393108282781882309
    var_9 = var_2.__hash__()
    assert var_9 == 7920157196696
    var_10 = var_9.__hash__()
    assert var_10 == 7920157196696
    var_11 = module_0.DictToken(*var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_12 = var_11.__eq__(var_11)
    assert var_12 is True
    var_13 = var_11.__eq__(var_2)
    assert var_13 is False
    var_14 = var_9.__repr__()
    assert var_14 == '7920157196696'
    var_3.lookup(var_0)

def test_case_20():
    var_0 = 'line1\nline2\nline3'
    var_1 = None
    var_2 = 7
    var_3 = module_0.ScalarToken(var_1, var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.start

def test_case_21():
    var_0 = 'line1\nline2\nline3'
    var_1 = None
    var_2 = -28
    var_3 = module_0.ScalarToken(var_1, var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.start

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_2.__repr__()
    assert var_4 == "ScalarToken('')"
    var_5 = var_2.__repr__()
    assert var_5 == "ScalarToken('')"
    var_6 = var_2.__hash__()
    assert var_6 == 7920157196696
    var_7 = var_2.__repr__()
    assert var_7 == "ScalarToken('')"
    var_8 = var_2.__hash__()
    assert var_8 == 7920157196696
    var_9 = var_2.__hash__()
    assert var_9 == 7920157196696
    var_10 = ''
    var_11 = 1209
    var_12 = module_0.ListToken(var_10, var_11, var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_13 = var_12.__eq__(var_2)
    assert var_13 is False
    var_14 = var_9.__repr__()
    assert var_14 == '7920157196696'
    var_15 = var_2.__repr__()
    assert var_15 == "ScalarToken('')"
    var_16 = 1402
    var_17 = module_0.Token(var_12, var_6, var_16, var_8)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_17.lookup(var_13)