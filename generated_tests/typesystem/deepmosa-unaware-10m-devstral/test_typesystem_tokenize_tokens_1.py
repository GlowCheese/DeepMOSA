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
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_2.lookup(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3.lookup_key(var_1)

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
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

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
    assert var_3 == 8403752210840
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
    assert var_2 == 8403752210840
    var_3 = module_0.Token(var_0, var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_2.__hash__()
    assert var_4 == 8403752210840
    var_3.__eq__(var_3)

def test_case_12():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_0.__hash__()
    assert var_2 == 8403752210840
    var_3 = var_2.__repr__()
    assert var_3 == '8403752210840'
    var_4 = module_0.ScalarToken(var_3, var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = var_2.__hash__()
    assert var_5 == 8403752210840
    var_6 = False
    var_7 = module_0.ScalarToken(var_0, var_6, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = var_1.__eq__(var_7)
    assert var_8 is False
    var_9 = var_5.__repr__()
    assert var_9 == '8403752210840'

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__hash__()
    assert var_2 == 8403752210840
    var_3 = var_1.__hash__()
    assert var_3 == 8403752210840
    var_4 = module_0.ListToken(var_2, var_0, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5 = var_3.__hash__()
    assert var_5 == 8403752210840
    var_6 = var_3.__repr__()
    assert var_6 == '8403752210840'
    var_7 = var_1.__eq__(var_3)
    assert var_7 is False
    var_8 = var_5.__repr__()
    assert var_8 == '8403752210840'
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
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2.lookup_key(var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = -703
    var_2 = module_0.ScalarToken(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8403752210840
    var_4 = {var_2: var_3}
    var_5 = [var_4, var_4, var_4, var_4]
    var_6 = module_0.DictToken(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_7 = var_3.__hash__()
    assert var_7 == 8403752210840
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_7.lookup_key(var_5)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = [var_0]
    var_2.lookup_key(var_5)