# Check out: https://github.com/GlowCheese/deepmosa
import dataclasses as module_1

import pypara.accounting.accounts as module_0
import pytest


def test_case_0():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.__hash__()
    assert var_1 == 5740354900026072187
    var_2 = module_1.InitVar(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'dataclasses.InitVar'
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_1.InitVar.type).__module__}.{type(module_1.InitVar.type).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_0.print()
    var_4 = var_1.__hash__()
    assert var_4 == 1128668881598684285
    module_0.COA(_subaccounts=var_4, rootspec=var_1)

def test_case_2():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.print()

def test_case_3():
    var_0 = None
    var_1 = 'j!jg\\nl2g<%cUf"VmX`'
    var_2 = module_0.COA()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_3 = module_0.COA()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.accounts.COA'
    var_4 = var_2.__iter__()
    with pytest.raises(ValueError):
        var_2.add(var_3, var_0, var_1)

def test_case_4():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_2 = var_1.find(var_0)
    var_3 = '"8!2y#MN{&-=~`w|t8y'
    var_4 = module_0.SubAccount(var_0, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code is None
    assert var_4.name == '"8!2y#MN{&-=~`w|t8y'
    assert var_4.parent is None
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_5 = var_1.print()
    with pytest.raises(ValueError):
        var_1.add(var_5, var_0, var_5)

def test_case_5():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_1.add(var_0, var_0, var_0)

def test_case_6():
    var_0 = None
    var_1 = 'j!jg\\nl2g<%cUf"VmX`'
    var_2 = module_0.COA()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_2.add(var_1, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = False
    var_2 = module_0.COA(rootspec=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_3 = var_2.__eq__(var_0)
    module_0.ReadChartOfAccounts(*var_2)

def test_case_8():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = 'Invalid Parent'
    var_3 = var_0.add(var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_3.code == 'Invalid Parent'
    assert var_3.name == 'Invalid Parent'
    assert f'{type(var_3.parent).__module__}.{type(var_3.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_4 = var_0.add(var_1, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code == 'Invalid Parent'
    assert var_4.name == 'Invalid Parent'
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'

def test_case_9():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = 'Invalid Parent'
    var_3 = var_0.add(var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_3.code == 'Invalid Parent'
    assert var_3.name == 'Invalid Parent'
    assert f'{type(var_3.parent).__module__}.{type(var_3.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_4 = 'Self Parent'
    var_5 = var_0.add(var_1, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_5.code == 'Self Parent'
    assert var_5.name == 'Self Parent'
    assert f'{type(var_5.parent).__module__}.{type(var_5.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    with pytest.raises(ValueError):
        var_0.add(var_1, var_4, var_2)

def test_case_10():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = '9999'
    var_3 = 'Invalid Parent'
    var_4 = var_0.add(var_1, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code == 'Invalid Parent'
    assert var_4.name == 'Invalid Parent'
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_5 = var_0.add(var_1, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_5.code == 'Invalid Parent'
    assert var_5.name == 'Invalid Parent'
    assert f'{type(var_5.parent).__module__}.{type(var_5.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_6 = 'Liquidity'
    var_7 = var_0.add(var_1, var_2, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_7.code == '9999'
    assert var_7.name == 'Liquidity'
    assert f'{type(var_7.parent).__module__}.{type(var_7.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'

def test_case_11():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = '000'
    var_3 = '9000'
    var_4 = 'Invalid Parent'
    var_5 = var_0.add(var_1, var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_5.code == '000'
    assert var_5.name == 'Invalid Parent'
    assert f'{type(var_5.parent).__module__}.{type(var_5.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_6 = var_0.print()
    var_7 = 'Self Parent'
    var_8 = var_0.add(var_1, var_3, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_8.code == '9000'
    assert var_8.name == 'Self Parent'
    assert f'{type(var_8.parent).__module__}.{type(var_8.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_9 = var_0.print()
    with pytest.raises(ValueError):
        var_0.add(var_0, var_6, var_6)

def test_case_12():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = '000'
    var_3 = '2'
    var_4 = '9000'
    var_5 = 'Invalid Parent'
    var_6 = var_0.add(var_1, var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_6.code == '000'
    assert var_6.name == 'Invalid Parent'
    assert f'{type(var_6.parent).__module__}.{type(var_6.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_7 = var_0.print()
    var_8 = 'Self Parent'
    var_9 = var_0.add(var_1, var_4, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_9.code == '9000'
    assert var_9.name == 'Self Parent'
    assert f'{type(var_9.parent).__module__}.{type(var_9.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    with pytest.raises(ValueError):
        var_0.add(var_4, var_1, var_3)