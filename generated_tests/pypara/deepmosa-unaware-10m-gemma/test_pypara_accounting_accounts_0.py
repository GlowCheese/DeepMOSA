# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.accounts as module_0
import collections as module_1

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
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.print()
    var_2 = '1'
    var_3 = '01'
    var_4 = 'Ghost count'
    var_5 = var_0.add(var_2, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_5.code == '01'
    assert var_5.name == 'Ghost count'
    assert f'{type(var_5.parent).__module__}.{type(var_5.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_6 = var_0.__repr__()
    assert var_6 == "COA(_accounts=OrderedDict([('1', RootAccount(code='1', name='Assets', type=<AccountType.ASSETS: 'ASSETS'>, coa=...)), ('2', RootAccount(code='2', name='Liabilities', type=<AccountType.LIABILITIES: 'LIABILITIES'>, coa=...)), ('3', RootAccount(code='3', name='Equities', type=<AccountType.EQUITIES: 'EQUITIES'>, coa=...)), ('4', RootAccount(code='4', name='Revenues', type=<AccountType.REVENUES: 'REVENUES'>, coa=...)), ('5', RootAccount(code='5', name='Expenses', type=<AccountType.EXPENSES: 'EXPENSES'>, coa=...)), ('01', SubAccount(code='01', name='Ghost count', parent=RootAccount(code='1', name='Assets', type=<AccountType.ASSETS: 'ASSETS'>, coa=...)))]), _subaccounts=OrderedDict([(RootAccount(code='1', name='Assets', type=<AccountType.ASSETS: 'ASSETS'>, coa=...), [SubAccount(code='01', name='Ghost count', parent=RootAccount(code='1', name='Assets', type=<AccountType.ASSETS: 'ASSETS'>, coa=...))])]))"
    var_7 = var_6.__repr__()
    assert var_7 == '"COA(_accounts=OrderedDict([(\'1\', RootAccount(code=\'1\', name=\'Assets\', type=<AccountType.ASSETS: \'ASSETS\'>, coa=...)), (\'2\', RootAccount(code=\'2\', name=\'Liabilities\', type=<AccountType.LIABILITIES: \'LIABILITIES\'>, coa=...)), (\'3\', RootAccount(code=\'3\', name=\'Equities\', type=<AccountType.EQUITIES: \'EQUITIES\'>, coa=...)), (\'4\', RootAccount(code=\'4\', name=\'Revenues\', type=<AccountType.REVENUES: \'REVENUES\'>, coa=...)), (\'5\', RootAccount(code=\'5\', name=\'Expenses\', type=<AccountType.EXPENSES: \'EXPENSES\'>, coa=...)), (\'01\', SubAccount(code=\'01\', name=\'Ghost count\', parent=RootAccount(code=\'1\', name=\'Assets\', type=<AccountType.ASSETS: \'ASSETS\'>, coa=...)))]), _subaccounts=OrderedDict([(RootAccount(code=\'1\', name=\'Assets\', type=<AccountType.ASSETS: \'ASSETS\'>, coa=...), [SubAccount(code=\'01\', name=\'Ghost count\', parent=RootAccount(code=\'1\', name=\'Assets\', type=<AccountType.ASSETS: \'ASSETS\'>, coa=...))])]))"'
    module_0.COA(rootspec=var_7)

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
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = 'V'
    var_3 = var_0.add(var_1, var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_3.code == 'V'
    assert var_3.name == '1'
    assert f'{type(var_3.parent).__module__}.{type(var_3.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'

def test_case_7():
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

def test_case_8():
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

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = module_0.COA()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_2 = var_1.__eq__(var_0)
    var_3 = var_1.__iter__()
    module_1.OrderedDict(*var_3)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = var_0.__hash__()
    assert var_2 == 5740354900026072187
    var_3 = ''
    var_4 = '9A!'
    var_5 = var_0.add(var_1, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_5.code == ''
    assert var_5.name == '9A!'
    assert f'{type(var_5.parent).__module__}.{type(var_5.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_6 = var_0.print()
    module_0.ReadChartOfAccounts()

def test_case_11():
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

def test_case_12():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = '2'
    var_3 = '9999'
    with pytest.raises(ValueError):
        var_0.add(var_1, var_2, var_3)

def test_case_13():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = '1'
    var_2 = '9999'
    var_3 = 'G9WQb'
    var_4 = var_0.add(var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code == '9999'
    assert var_4.name == 'G9WQb'
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_5 = None
    var_6 = var_0.add(var_1, var_5, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_6.code is None
    assert f'{type(var_6.name).__module__}.{type(var_6.name).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert f'{type(var_6.parent).__module__}.{type(var_6.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'

def test_case_14():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.print()
    var_2 = '1'
    var_3 = 'G9W|b'
    var_4 = var_0.add(var_2, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code is None
    assert var_4.name is None
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_5 = None
    with pytest.raises(ValueError):
        var_0.add(var_2, var_5, var_3)