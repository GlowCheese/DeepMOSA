# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.accounts as module_0
import dataclasses as module_1

def test_case_0():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'

def test_case_1():
    pass

def test_case_2():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.__iter__()
    var_2 = var_0.print()

def test_case_3():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.print()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = None
    var_2 = module_1.dataclass(unsafe_hash=var_0, frozen=var_1, kw_only=var_0)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_3 = var_2.__repr__()
    var_4 = var_3.__repr__()
    var_5 = var_4.__repr__()
    var_6 = var_5.__eq__(var_2)
    var_7 = var_4.__eq__(var_3)
    assert var_7 is False
    var_8 = var_2.__repr__()
    var_9 = var_8.__eq__(var_8)
    assert var_9 is True
    var_10 = var_5.__eq__(var_2)
    module_0.COA(rootspec=var_3)

def test_case_5():
    var_0 = None
    var_1 = None
    var_2 = module_0.COA()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_3 = module_1.dataclass(var_1, init=var_0, repr=var_0, eq=var_0, frozen=var_1)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_3.__repr__()
    var_6 = '4PE\nO~umn]P!:9\re9!'
    with pytest.raises(ValueError):
        var_2.add(var_0, var_1, var_6)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.COA()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_2 = var_1.find(var_0)
    var_3 = var_1.__post_init__(var_0)
    var_4 = None
    var_5 = 'sGhO'
    var_6 = module_0.SubAccount(var_4, var_5, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_6.code is None
    assert var_6.name == 'sGhO'
    assert var_6.parent is None
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_7 = module_1.field(default_factory=var_0, hash=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'dataclasses.Field'
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_1.Field.compare).__module__}.{type(module_1.Field.compare).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.default).__module__}.{type(module_1.Field.default).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.default_factory).__module__}.{type(module_1.Field.default_factory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.hash).__module__}.{type(module_1.Field.hash).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.init).__module__}.{type(module_1.Field.init).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.kw_only).__module__}.{type(module_1.Field.kw_only).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.metadata).__module__}.{type(module_1.Field.metadata).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.name).__module__}.{type(module_1.Field.name).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.repr).__module__}.{type(module_1.Field.repr).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.type).__module__}.{type(module_1.Field.type).__qualname__}' == 'builtins.member_descriptor'
    var_7.__delattr__(var_2)

def test_case_7():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = None
    with pytest.raises(ValueError):
        var_0.add(var_0, var_1, var_0)

def test_case_8():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5