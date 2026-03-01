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

def test_case_1():
    var_0 = []
    var_1 = -299
    var_2 = True
    var_3 = '\x0ctg`A'
    var_4 = module_0.ListToken(var_0, var_2, var_1, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_5 = var_4.lookup(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6 = var_5.__eq__(var_4)
    assert var_6 is True

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
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__eq__(var_1)
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
def test_case_8():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_1.lookup_key(var_0)

def test_case_9():
    var_0 = None
    var_1 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__hash__()
    assert var_2 == 7769023273624
    var_3 = var_2.__hash__()
    assert var_3 == 7769023273624
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = []
    var_2 = -4379
    var_3 = module_0.Token(var_0, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_3.lookup(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    var_5 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = var_3.__repr__()
    assert var_6 == "Token('')"
    var_7 = module_0.ScalarToken(var_0, var_0, var_0, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = var_7.__hash__()
    assert var_8 == 7769023273624
    var_9 = var_7.__eq__(var_7)
    assert var_9 is True
    var_3.__eq__(var_3)

def test_case_11():
    var_0 = '$7qu<U3k5\rd'
    var_1 = {}
    var_2 = -18
    var_3 = [var_1, var_2, var_2, var_0]
    var_4 = module_0.ScalarToken(var_0, var_3, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_5 = module_0.DictToken(*var_3, **var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_6 = var_5.__eq__(var_4)
    assert var_6 is False

def test_case_12():
    var_0 = None
    var_1 = []
    var_2 = '\x0ctg`A'
    var_3 = module_0.ListToken(var_1, var_2, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'a'
    var_1 = -5
    var_2 = {var_0: var_1, var_1: var_1}
    var_3 = -22
    var_4 = 5
    var_5 = [var_2, var_3, var_4, var_0]
    var_6 = {}
    module_0.DictToken(*var_5, **var_6)

def test_case_14():
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = ''
    var_1 = {}
    var_2 = 2
    var_3 = [var_1, var_2, var_2, var_0]
    var_4 = module_0.DictToken(*var_3, **var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_4.lookup_key(var_3)

def test_case_16():
    var_0 = {}
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.DictToken(*var_1, **var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = ''
    var_1 = {}
    var_2 = 3
    var_3 = [var_1, var_2, var_2, var_0]
    var_4 = module_0.DictToken(*var_3, **var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_5 = [var_0]
    var_4.lookup_key(var_5)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = [var_0]
    var_2 = '\x0ctg`A'
    var_3 = module_0.ListToken(var_1, var_2, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_3.__eq__(var_3)

def test_case_19():
    var_0 = {}
    var_1 = 3
    var_2 = [var_0, var_1, var_1, var_0]
    var_3 = module_0.ScalarToken(var_0, var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = None
    var_5 = module_0.ListToken(var_4, var_2, var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6 = module_0.DictToken(*var_2, **var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_7 = None
    var_8 = module_0.ScalarToken(var_7, var_1, var_1, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_9 = var_6.__eq__(var_3)
    assert var_9 is False
    var_10 = var_6.__eq__(var_6)
    assert var_10 is True
    var_11 = True
    var_12 = module_0.Token(var_6, var_6, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_13 = bool(var_6._start_index == var_3)
    with pytest.raises(AttributeError):
        var_14 = var_9._end_index