# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__repr__()
    assert var_2 == "ScalarToken('')"
    var_3 = module_0.ListToken(var_2, var_1, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4 = [var_0]
    var_5 = var_3.lookup(var_4)
    assert var_5 == 'S'
    with pytest.raises(AttributeError):
        var_6 = bool(not var_3 == var_1)
    assert var_6 is True

def test_case_1():
    var_0 = False
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = []
    var_3 = var_1.lookup(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_0.__repr__()
    assert var_4 == 'False'
    var_5 = module_0.ListToken(var_4, var_1, var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    with pytest.raises(AttributeError):
        var_6 = bool(not var_5 == var_1)
    assert var_6 is True

def test_case_2():
    var_0 = 'f'
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = bool(not var_1 == var_1)
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.DictToken()

def test_case_4():
    var_0 = 'te^st conCent'
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = bool(not var_1 == var_1)

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
    var_0 = 'te^st conCent'
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = bool(not var_1 == var_1)
    var_1.__repr__()

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'te^st conCent'
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = bool(not var_1 == var_1)
    var_1.lookup_key(var_2)

def test_case_9():
    var_0 = 'te^st conCent'
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__hash__()
    assert var_2 == -7462338998448175674
    var_3 = bool(not var_1 == var_1)

def test_case_10():
    var_0 = 'key'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_1, var_1, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = {var_2: var_2}
    var_4 = 0
    var_5 = 10
    var_6 = 'some content'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = {}
    var_9 = module_0.DictToken(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_keys == {'key': var_2})
    var_12 = var_9._child_tokens
    var_13 = bool(var_9._child_tokens == {'key': var_1})

def test_case_11():
    var_0 = 0
    var_1 = 'test content'
    var_2 = module_0.Token(var_1, var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    with pytest.raises(NotImplementedError):
        var_3 = bool(not var_2 == var_2)
    assert var_3 is True

def test_case_12():
    var_0 = True
    var_1 = var_0.__repr__()
    assert var_1 == 'True'
    var_2 = module_0.ListToken(var_1, var_1, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    with pytest.raises(AttributeError):
        var_3 = bool(not var_2 == var_2)
    assert var_3 is True

def test_case_13():
    var_0 = {}
    var_1 = 'value'
    var_2 = 'start_index'
    var_3 = 'end_index'
    var_4 = {var_1: var_0, var_2: var_2, var_3: var_3, var_3: var_1}
    var_5 = module_0.DictToken(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = {}
    var_1 = 'value'
    var_2 = 'start_index'
    var_3 = 'end_index'
    var_4 = {var_1: var_0, var_2: var_2, var_3: var_3, var_3: var_0}
    var_5 = module_0.DictToken(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_6 = [var_0, var_3]
    var_5.lookup_key(var_6)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_1]
    var_3 = 'U/(S#F]"@WP'
    var_4 = 'az:U/,TMmsE.d\x0b\n'
    var_5 = {var_3: var_1, var_4: var_1}
    var_6 = module_0.ScalarToken(var_1, var_0, var_3, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = module_0.ScalarToken(var_1, var_4, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = var_7.__eq__(var_6)
    assert var_8 is False
    module_0.DictToken(*var_2, **var_5)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 'U/(S#F]"@WP'
    var_3 = 'az:U/,TMmsE.d\x0b\n'
    var_4 = {var_2: var_0, var_3: var_0}
    var_5 = module_0.ScalarToken(var_2, var_0, var_2, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = module_0.ScalarToken(var_0, var_3, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = var_6.__eq__(var_5)
    assert var_7 is False
    module_0.DictToken(*var_1, **var_4)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = {}
    var_1 = 'value'
    var_2 = 'start_index'
    var_3 = 'end_index'
    var_4 = {var_1: var_0, var_2: var_1, var_3: var_3, var_3: var_2}
    var_5 = module_0.DictToken(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_6 = [var_1]
    var_5.lookup_key(var_6)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = []
    var_1 = None
    var_2 = module_0.ListToken(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_3 = [var_2, var_2, var_1, var_2]
    var_4 = False
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True
    var_6 = (var_3, var_4)
    var_7 = -1982
    var_8 = True
    var_9 = module_0.ListToken(var_1, var_8, var_1, var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_10 = False
    var_11 = True
    var_12 = module_0.ScalarToken(var_1, var_10, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_13 = ']'
    var_14 = module_0.ScalarToken(var_6, var_7, var_7, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_14.__hash__()