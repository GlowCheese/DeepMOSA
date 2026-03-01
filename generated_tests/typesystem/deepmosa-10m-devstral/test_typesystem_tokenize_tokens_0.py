# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = [var_0, var_0]
    var_2.lookup_key(var_4)

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
    assert var_6 == 7708171716760
    var_7 = var_6.__hash__()
    assert var_7 == 7708171716760
    var_8 = var_7.__hash__()
    assert var_8 == 7708171716760
    var_9 = var_8.__hash__()
    assert var_9 == 7708171716760
    var_10 = var_9.__hash__()
    assert var_10 == 7708171716760
    var_11 = var_10.__hash__()
    assert var_11 == 7708171716760

def test_case_2():
    var_0 = 1464
    var_1 = None
    var_2 = module_0.ScalarToken(var_1, var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = bool(not var_2 == var_2)

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

def test_case_6():
    var_0 = 1464
    var_1 = None
    var_2 = module_0.ScalarToken(var_1, var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = var_2.__hash__()
    assert var_4 == 7708171716760
    var_5 = bool(not var_2 == var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = -4187.0
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.ScalarToken(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.__hash__()
    assert var_4 == 7708171716760
    var_5 = var_4.__repr__()
    assert var_5 == '7708171716760'
    var_6 = module_0.ListToken(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6.lookup(var_1)

def test_case_8():
    var_0 = 'm2:'
    var_1 = module_0.ListToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    with pytest.raises(AttributeError):
        var_2 = bool(not var_1 == var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_9():
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

def test_case_11():
    var_0 = 1464
    var_1 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = bool(not var_1 == var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'a'
    var_1 = 23
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = -26
    var_4 = [var_2, var_3, var_1, var_0]
    var_5 = {}
    module_0.DictToken(*var_4, **var_5)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = True
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = var_2.__hash__()
    assert var_5 == 7708171716760
    var_6 = var_5.__hash__()
    assert var_6 == 7708171716760
    var_7 = module_0.ScalarToken(var_0, var_5, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = module_0.Token(var_5, var_0, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_9 = var_7.__hash__()
    assert var_9 == 7708171716760
    var_10 = var_9.__hash__()
    assert var_10 == 7708171716760
    var_11 = var_6.__hash__()
    assert var_11 == 7708171716760
    var_12 = {var_11: var_6, var_5: var_5, var_5: var_5}
    var_13 = var_12.__eq__(var_6)
    var_14 = var_2.__hash__()
    assert var_14 == 7708171716760
    var_15 = var_11.__hash__()
    assert var_15 == 7708171716760
    var_16 = var_2.__eq__(var_7)
    assert var_16 is False
    var_17 = var_14.__hash__()
    assert var_17 == 7708171716760
    var_18 = var_17.__hash__()
    assert var_18 == 7708171716760
    module_0.DictToken()

def test_case_14():
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

def test_case_15():
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

def test_case_16():
    var_0 = ''
    var_1 = -27
    var_2 = module_0.ListToken(var_0, var_1, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_3 = bool(not var_2 == var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = {}
    var_1 = 0
    var_2 = ']N'
    var_3 = []
    var_4 = 'value'
    var_5 = None
    var_6 = module_0.ScalarToken(var_5, var_5, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = 'start_index'
    var_8 = 'end_index'
    var_9 = 'content'
    var_10 = {var_4: var_0, var_7: var_1, var_8: var_1, var_9: var_2}
    var_11 = module_0.DictToken(*var_3, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_12 = var_11.__eq__(var_11)
    assert var_12 is True
    var_13 = var_11._child_keys
    var_11.lookup_key(var_4)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = {}
    var_1 = [var_0]
    var_2 = None
    var_3 = 'start_index'
    var_4 = 'end_index'
    var_5 = 'content'
    var_6 = {var_4: var_0, var_3: var_5, var_3: var_2, var_4: var_1, var_4: var_1, var_4: var_2, var_5: var_5, var_5: var_2}
    var_7 = module_0.DictToken(*var_1, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = bool(var_7._child_keys == {})
    assert var_9 is True
    var_10 = module_0.ScalarToken(var_9, var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7.lookup_key(var_1)

def test_case_19():
    var_0 = {}
    var_1 = []
    var_2 = 'value'
    var_3 = 'start_index'
    var_4 = 'end_index'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_0, var_4: var_4}
    var_6 = module_0.DictToken(*var_1, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_7 = None
    var_8 = True
    var_9 = module_0.ScalarToken(var_7, var_8, var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_10 = var_6.__eq__(var_9)
    assert var_10 is False

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = None
    var_2 = module_0.ScalarToken(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = {var_2: var_2, var_2: var_1}
    var_4 = [var_3, var_2, var_3]
    var_5 = module_0.DictToken(*var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_5.lookup_key(var_0)