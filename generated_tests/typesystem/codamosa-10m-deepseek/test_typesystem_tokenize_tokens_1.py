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
    var_1 = 3215
    var_2 = module_0.ListToken(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

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
    assert var_3 == 8093889401496
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
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = False
    var_3 = False
    var_4 = module_0.ListToken(var_0, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_4.lookup(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = -3153
    var_2 = '&O@?'
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = None
    var_6 = 665
    var_7 = module_0.Token(var_5, var_5, var_6, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.Token'
    assert f'{type(module_0.Token.string).__module__}.{type(module_0.Token.string).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.value).__module__}.{type(module_0.Token.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.start).__module__}.{type(module_0.Token.start).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Token.end).__module__}.{type(module_0.Token.end).__qualname__}' == 'builtins.property'
    var_7.__repr__()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = False
    var_1 = module_0.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = None
    var_3 = False
    var_4 = '0N_\\e"v02< hcll'
    var_5 = module_0.ScalarToken(var_2, var_2, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = None
    var_7 = var_5.__eq__(var_6)
    assert var_7 is False
    var_8 = var_1.__eq__(var_5)
    assert var_8 is False
    var_9 = var_5.__repr__()
    assert var_9 == "ScalarToken('0')"
    var_10 = [var_9, var_3, var_7, var_4]
    var_1.lookup_key(var_10)

def test_case_11():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = 'value'
    var_5 = 4
    var_6 = 9
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = 'key1'
    var_9 = 3
    var_10 = module_0.ScalarToken(var_8, var_1, var_9, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_11 = 'value1'
    var_12 = 5
    var_13 = 11
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_15 = 'key2'
    var_16 = 13
    var_17 = 16
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_19 = 'value2'
    var_20 = 18
    var_21 = 24
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_23 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_24 = 'nested_key'
    var_25 = module_0.ScalarToken(var_24, var_5, var_16, var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_26 = 'nested_value'
    var_27 = 15
    var_28 = 26
    var_29 = module_0.ScalarToken(var_26, var_27, var_28, var_26)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_30 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_31 = 'item1'
    var_32 = module_0.ScalarToken(var_31, var_5, var_17, var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_33 = 'item2'
    var_34 = 10
    var_35 = 14
    var_36 = module_0.ScalarToken(var_33, var_34, var_35, var_33)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_37 = [var_32, var_36]
    var_38 = 'item1, item2'
    var_39 = module_0.ListToken(var_37, var_5, var_35, var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_40 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_41 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_42 = module_0.ScalarToken(var_8, var_1, var_9, var_8)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_43 = module_0.ScalarToken(var_11, var_12, var_13, var_11)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_44 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_45 = 22
    var_46 = module_0.ScalarToken(var_31, var_20, var_45, var_31)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_47 = 28
    var_48 = module_0.ScalarToken(var_33, var_21, var_47, var_33)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_49 = [var_46, var_48]
    var_50 = module_0.ListToken(var_49, var_20, var_47, var_38)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_51 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_52 = module_0.ScalarToken(var_11, var_5, var_34, var_11)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_53 = 12
    var_54 = module_0.ScalarToken(var_0, var_53, var_35, var_0)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_55 = module_0.ScalarToken(var_19, var_17, var_45, var_19)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_56 = 123
    var_57 = '123'
    var_58 = module_0.ScalarToken(var_56, var_1, var_2, var_57)
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_59 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_60 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_61 = None
    var_62 = 'null'
    var_63 = module_0.ScalarToken(var_61, var_5, var_20, var_62)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_64 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    assert f'{type(var_64).__module__}.{type(var_64).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'