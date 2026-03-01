# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.accounts as module_0
import dataclasses as module_1
import builtins as module_2

def test_case_0():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.AccountType.EQUITIES
    var_1 = None
    module_0.COA(var_1, rootspec=var_0)

def test_case_2():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = [(code, acct.name) for (code, acct) in var_0]

def test_case_3():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = None
    var_3 = var_0.add(var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_3.code is None
    assert var_3.name is None
    assert f'{type(var_3.parent).__module__}.{type(var_3.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_4 = None
    with pytest.raises(ValueError):
        var_0.add(var_4, var_4, var_2)

def test_case_4():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.print()

def test_case_5():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = None
    with pytest.raises(ValueError):
        var_0.add(var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_1.dataclass(eq=var_0, frozen=var_0, slots=var_0)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_2 = module_0.COA()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_3 = var_2.find(var_1)
    var_3.add(var_0, var_1, var_0)

def test_case_7():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = var_0.add(var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert f'{type(var_2.code).__module__}.{type(var_2.code).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert f'{type(var_2.name).__module__}.{type(var_2.name).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert f'{type(var_2.parent).__module__}.{type(var_2.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_3 = var_0.add(var_1, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert f'{type(var_3.code).__module__}.{type(var_3.code).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert f'{type(var_3.name).__module__}.{type(var_3.name).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert f'{type(var_3.parent).__module__}.{type(var_3.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'

def test_case_8():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = None
    var_3 = var_0.add(var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_3.code is None
    assert var_3.name is None
    assert f'{type(var_3.parent).__module__}.{type(var_3.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_4 = None
    var_5 = var_0.nodify(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.accounts.COA.Node'
    assert f'{type(var_5.account).__module__}.{type(var_5.account).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_5.children == []
    var_6 = 'f'
    with pytest.raises(ValueError):
        var_0.add(var_4, var_1, var_6)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = var_0.print()
    var_3 = var_0.add(var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_3.code is None
    assert var_3.name is None
    assert f'{type(var_3.parent).__module__}.{type(var_3.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_4 = var_0.add(var_1, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code is None
    assert var_4.name is None
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_5 = var_0.print()
    var_6 = var_0.__hash__()
    assert var_6 == 5740354900026072187
    module_2.object(*var_6)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.print()
    var_2 = None
    var_3 = var_0.print()
    var_4 = module_0.COA(rootspec=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.COA'
    var_5 = '1'
    var_6 = None
    var_7 = var_0.add(var_5, var_6, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_7.code is None
    assert var_7.name is None
    assert f'{type(var_7.parent).__module__}.{type(var_7.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_8 = var_4.__repr__()
    assert var_8 == "COA(_accounts=OrderedDict([('1', RootAccount(code='1', name='Assets', type=<AccountType.ASSETS: 'ASSETS'>, coa=...)), ('2', RootAccount(code='2', name='Liabilities', type=<AccountType.LIABILITIES: 'LIABILITIES'>, coa=...)), ('3', RootAccount(code='3', name='Equities', type=<AccountType.EQUITIES: 'EQUITIES'>, coa=...)), ('4', RootAccount(code='4', name='Revenues', type=<AccountType.REVENUES: 'REVENUES'>, coa=...)), ('5', RootAccount(code='5', name='Expenses', type=<AccountType.EXPENSES: 'EXPENSES'>, coa=...))]), _subaccounts=OrderedDict())"
    var_9 = var_8.__eq__(var_2)
    var_10 = var_0.nodify(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.accounting.accounts.COA.Node'
    assert f'{type(var_10.account).__module__}.{type(var_10.account).__qualname__}' == 'builtins.NotImplementedType'
    assert var_10.children == []
    var_11 = var_0.add(var_5, var_9, var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert f'{type(var_11.code).__module__}.{type(var_11.code).__qualname__}' == 'builtins.NotImplementedType'
    assert var_11.name is None
    assert f'{type(var_11.parent).__module__}.{type(var_11.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_12 = var_8.__hash__()
    assert var_12 == -1450695749073668557
    var_13 = var_9.__repr__()
    assert var_13 == 'NotImplemented'
    module_2.object(**var_12)

def test_case_11():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.__repr__()
    assert var_1 == "COA(_accounts=OrderedDict([('1', RootAccount(code='1', name='Assets', type=<AccountType.ASSETS: 'ASSETS'>, coa=...)), ('2', RootAccount(code='2', name='Liabilities', type=<AccountType.LIABILITIES: 'LIABILITIES'>, coa=...)), ('3', RootAccount(code='3', name='Equities', type=<AccountType.EQUITIES: 'EQUITIES'>, coa=...)), ('4', RootAccount(code='4', name='Revenues', type=<AccountType.REVENUES: 'REVENUES'>, coa=...)), ('5', RootAccount(code='5', name='Expenses', type=<AccountType.EXPENSES: 'EXPENSES'>, coa=...))]), _subaccounts=OrderedDict())"
    var_2 = module_0.COA()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.accounts.COA'
    var_3 = '1'
    var_4 = var_2.print()
    var_5 = var_0.__eq__(var_4)
    var_6 = var_0.add(var_3, var_4, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_6.code is None
    assert var_6.name is None
    assert f'{type(var_6.parent).__module__}.{type(var_6.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_0.add(var_3, var_4, var_0)