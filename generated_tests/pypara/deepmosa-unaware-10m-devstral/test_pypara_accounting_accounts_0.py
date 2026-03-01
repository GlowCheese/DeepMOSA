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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_1.field(default_factory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'dataclasses.Field'
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
    var_2 = var_1.__hash__()
    var_3 = var_1.__eq__(var_2)
    module_0.COA(_subaccounts=var_3, rootspec=var_3)

def test_case_3():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.print()

def test_case_4():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.__iter__()
    var_2 = var_0.print()

def test_case_5():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.__iter__()
    var_2 = var_0.find(var_1)

def test_case_6():
    var_0 = 'cr$PH'
    var_1 = module_0.COA()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_2 = None
    with pytest.raises(ValueError):
        var_1.add(var_0, var_2, var_0)

def test_case_7():
    var_0 = None
    var_1 = 'N`!f'
    var_2 = None
    var_3 = module_0.COA(_subaccounts=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_3.add(var_0, var_0, var_1)

def test_case_8():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = [(code, acct.name) for (code, acct) in var_0]

def test_case_9():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_1.__hash__()
    assert var_2 == 1
    var_3 = var_2.__repr__()
    assert var_3 == '1'
    var_4 = var_0.add(var_3, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code == 1
    assert var_4.name == 1
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_5 = var_0.add(var_3, var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_5.code == 1
    assert var_5.name == 1
    assert f'{type(var_5.parent).__module__}.{type(var_5.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'

def test_case_10():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_1.__hash__()
    assert var_2 == 1
    var_3 = var_2.__hash__()
    assert var_3 == 1
    var_4 = var_3.__repr__()
    assert var_4 == '1'
    var_5 = var_0.add(var_4, var_2, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_5.code == 1
    assert var_5.name == 1
    assert f'{type(var_5.parent).__module__}.{type(var_5.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_6 = var_0.add(var_4, var_3, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_6.code == 1
    assert var_6.name == 1
    assert f'{type(var_6.parent).__module__}.{type(var_6.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_7 = var_0.print()

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.print()
    var_2 = var_0.__eq__(var_0)
    assert var_2 is True
    var_3 = var_2.__hash__()
    assert var_3 == 1
    var_4 = var_0.print()
    var_5 = None
    var_6 = var_3.__repr__()
    assert var_6 == '1'
    var_7 = var_2.__hash__()
    assert var_7 == 1
    var_8 = var_6.__hash__()
    assert var_8 == -7988072494311411251
    var_9 = var_3.__repr__()
    assert var_9 == '1'
    var_10 = var_0.add(var_9, var_2, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_10.code is True
    assert var_10.name == -7988072494311411251
    assert f'{type(var_10.parent).__module__}.{type(var_10.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_11 = var_8.__eq__(var_9)
    var_12 = 'mR[k5W'
    var_13 = var_0.add(var_9, var_5, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_13.code is None
    assert var_13.name == 'mR[k5W'
    assert f'{type(var_13.parent).__module__}.{type(var_13.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_8.print()

def test_case_12():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.print()
    var_2 = var_0.__eq__(var_0)
    assert var_2 is True
    var_3 = var_2.__hash__()
    assert var_3 == 1
    var_4 = var_3.__repr__()
    assert var_4 == '1'
    var_5 = var_2.__hash__()
    assert var_5 == 1
    var_6 = var_3.__repr__()
    assert var_6 == '1'
    var_7 = var_0.add(var_6, var_2, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_7.code is True
    assert var_7.name == 1
    assert f'{type(var_7.parent).__module__}.{type(var_7.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_8 = module_1.InitVar(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'dataclasses.InitVar'
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_1.InitVar.type).__module__}.{type(module_1.InitVar.type).__qualname__}' == 'builtins.member_descriptor'
    var_9 = 'mR[k5W'
    with pytest.raises(ValueError):
        var_0.add(var_6, var_5, var_9)

def test_case_13():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.print()
    var_2 = var_0.__eq__(var_0)
    assert var_2 is True
    var_3 = var_2.__hash__()
    assert var_3 == 1
    var_4 = var_0.print()
    var_5 = var_3.__repr__()
    assert var_5 == '1'
    var_6 = var_2.__hash__()
    assert var_6 == 1
    var_7 = var_5.__hash__()
    assert var_7 == -7988072494311411251
    var_8 = var_3.__repr__()
    assert var_8 == '1'
    var_9 = var_0.add(var_8, var_2, var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_9.code is True
    assert var_9.name == -7988072494311411251
    assert f'{type(var_9.parent).__module__}.{type(var_9.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_10 = var_0.add(var_5, var_4, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_10.code is None
    assert var_10.name == '1'
    assert f'{type(var_10.parent).__module__}.{type(var_10.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_11 = '\\E.='
    with pytest.raises(ValueError):
        var_0.add(var_1, var_2, var_11)