# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokens as module_0
import builtins as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'value'
    var_3 = 'start_index'
    var_4 = 'end_index'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_0, var_4: var_2}
    var_6 = module_0.DictToken(*var_1, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_7 = [var_0, var_4]
    var_6.lookup(var_7)

def test_case_1():
    var_0 = {}
    var_1 = []
    var_2 = 'value'
    var_3 = 'start_index'
    var_4 = 'end_index'
    var_5 = {var_2: var_0, var_3: var_0, var_4: var_0, var_4: var_2}
    var_6 = module_0.DictToken(*var_1, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_7 = var_6.__eq__(var_6)
    assert var_7 is True
    var_8 = var_6.lookup(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

def test_case_2():
    var_0 = {}
    var_1 = 'value'
    var_2 = 'start_index'
    var_3 = 'end_index'
    var_4 = {var_1: var_2, var_1: var_0, var_2: var_0, var_3: var_0, var_3: var_1}
    var_5 = module_0.DictToken(*var_0, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_6 = var_5.__eq__(var_4)
    assert var_6 is False

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.DictToken()

def test_case_4():
    var_0 = None
    var_1 = module_0.ListToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = -4187.0
    var_1 = [var_0]
    var_2 = 'MI(McZ,@\r^skV94'
    var_3 = None
    var_4 = -1275
    var_5 = module_0.ScalarToken(var_3, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = module_0.ScalarToken(var_3, var_4, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7.lookup(var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.ListToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_1.__repr__()

@pytest.mark.xfail(strict=True)
def test_case_7():
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

def test_case_8():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8650633180568
    var_4 = var_2.__eq__(var_1)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = -4187.0
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.ScalarToken(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.__hash__()
    assert var_4 == 8650633180568
    var_5 = var_4.__repr__()
    assert var_5 == '8650633180568'
    var_6 = module_0.ListToken(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_6.lookup(var_1)

def test_case_10():
    var_0 = {}
    var_1 = 'value'
    var_2 = 'start_index'
    var_3 = 'end_index'
    var_4 = {var_1: var_2, var_1: var_0, var_2: var_0, var_3: var_0, var_3: var_1}
    var_5 = module_0.DictToken(*var_0, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True

def test_case_11():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    with pytest.raises(NotImplementedError):
        var_3 = var_2.value

@pytest.mark.xfail(strict=True)
def test_case_12():
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
def test_case_13():
    var_0 = False
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.__repr__()
    assert var_2 == "ScalarToken('')"
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    module_0.DictToken()

@pytest.mark.xfail(strict=True)
def test_case_14():
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
def test_case_15():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__repr__()
    assert var_3 == "ScalarToken('')"
    var_4 = var_1.__hash__()
    assert var_4 == 0
    var_5 = var_4.__hash__()
    assert var_5 == 0
    var_6 = var_4.__eq__(var_5)
    assert var_6 is True
    var_7 = {var_4: var_4, var_0: var_0, var_0: var_0}
    var_8 = [var_7, var_3, var_2]
    var_9 = var_5.__eq__(var_4)
    assert var_9 is True
    module_0.DictToken(*var_8)

@pytest.mark.xfail(strict=True)
def test_case_16():
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
    assert var_9 == 8650633180568
    var_10 = var_9.__hash__()
    assert var_10 == 8650633180568
    var_11 = module_0.ScalarToken(var_4, var_9, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_12 = module_0.Token(var_9, var_4, var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_13 = var_11.__hash__()
    assert var_13 == 8650633180568
    var_14 = var_13.__hash__()
    assert var_14 == 8650633180568
    var_15 = var_10.__hash__()
    assert var_15 == 8650633180568
    var_16 = {var_15: var_10, var_9: var_9, var_9: var_9}
    var_17 = var_16.__eq__(var_10)
    var_18 = var_6.__hash__()
    assert var_18 == 8650633180568
    var_19 = var_15.__hash__()
    assert var_19 == 8650633180568
    var_20 = var_6.__eq__(var_11)
    assert var_20 is False
    var_21 = var_18.__hash__()
    assert var_21 == 8650633180568
    var_22 = var_21.__hash__()
    assert var_22 == 8650633180568
    module_0.DictToken()

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = module_0.ListToken(var_0, var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_3.__repr__()

@pytest.mark.xfail(strict=True)
def test_case_18():
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
    var_7 = var_6.__hash__()
    assert var_7 == 8650633180568
    var_8 = None
    var_9 = module_0.ScalarToken(var_3, var_8, var_0, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_10 = var_9.__hash__()
    assert var_10 == -5123205663399990105
    var_11 = var_9.__eq__(var_6)
    assert var_11 is False
    var_12 = var_7.__eq__(var_0)
    var_13 = var_2.__hash__()
    assert var_13 == 8650633180568
    var_14 = [var_10, var_10, var_10]
    module_0.DictToken(*var_14, **var_10)

def test_case_19():
    var_0 = {}
    var_1 = 'value'
    var_2 = 'start_index'
    var_3 = 'end_index'
    var_4 = {var_1: var_2, var_1: var_0, var_2: var_0, var_3: var_0, var_3: var_1}
    var_5 = module_0.DictToken(*var_0, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = True
    var_1 = [var_0, var_0]
    var_2 = True
    var_3 = None
    var_4 = module_0.ListToken(var_1, var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4.__eq__(var_4)

def test_case_21():
    var_0 = None
    var_1 = 5
    var_2 = 'line1\nline2\nline3'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_4 = var_3.start
    var_5 = var_4.column_no

def test_case_22():
    var_0 = None
    var_1 = 0
    var_2 = 12
    var_3 = 'line1\nline2\nline3'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_5 = var_4.end
    var_6 = var_5.line_no
    assert var_6 == 3
    var_7 = var_5.column_no
    assert var_7 == 1
    with pytest.raises(AttributeError):
        var_8 = var_5.index
    assert var_8 == 12

def test_case_23():
    var_0 = None
    var_1 = 1875
    var_2 = True
    var_3 = module_0.ScalarToken(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = None
    var_5 = -54
    var_6 = 1003
    var_7 = 'line1\nline2\nline3'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_9 = var_8.start
    var_10 = var_9.line_no
    assert var_10 == 2
    var_11 = var_9.column_no
    assert var_11 == 1
    with pytest.raises(AttributeError):
        var_12 = var_9.index
    assert var_12 == 5

def test_case_24():
    var_0 = None
    var_1 = False
    var_2 = module_0.ScalarToken(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = var_2.__hash__()
    assert var_3 == 8650633180568
    var_4 = var_3.__repr__()
    assert var_4 == '8650633180568'
    var_5 = var_4.__hash__()
    assert var_5 == -8405956310129722429
    var_6 = {var_2: var_2}
    var_7 = []
    var_8 = 'value'
    var_9 = 'start_index'
    var_10 = 'end_index'
    var_11 = {var_8: var_6, var_9: var_6, var_10: var_6, var_10: var_8}
    var_12 = []
    var_13 = var_5.__eq__(var_12)
    var_14 = module_0.DictToken(*var_7, **var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_15 = var_14.__eq__(var_14)
    assert var_15 is True

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = -1732
    var_1 = var_0.__hash__()
    assert var_1 == -1732
    var_2 = var_1.__hash__()
    assert var_2 == -1732
    var_3 = var_2.__hash__()
    assert var_3 == -1732
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_3.__hash__()
    assert var_5 == -1732
    var_6 = var_5.__hash__()
    assert var_6 == -1732
    var_7 = var_6.__hash__()
    assert var_7 == -1732
    var_8 = var_7.__hash__()
    assert var_8 == -1732
    var_9 = var_8.__hash__()
    assert var_9 == -1732
    var_10 = {}
    var_11 = []
    var_12 = 'value'
    var_13 = 'start_index'
    var_14 = 'end_index'
    var_15 = {var_12: var_10, var_13: var_10, var_14: var_10, var_14: var_12}
    var_16 = module_0.DictToken(*var_11, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_17 = var_16.__eq__(var_16)
    assert var_17 is True
    var_18 = [var_6]
    var_16.lookup_key(var_18)