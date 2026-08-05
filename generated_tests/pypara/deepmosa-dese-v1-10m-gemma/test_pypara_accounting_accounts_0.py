# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.accounts as module_0

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
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_1.__hash__()
    assert var_2 == 1
    var_3 = var_0.subaccounts(var_2)
    var_4 = var_2.__repr__()
    assert var_4 == '1'
    var_5 = var_1.__hash__()
    assert var_5 == 1
    var_6 = var_0.add(var_4, var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_6.code is True
    assert var_6.name is True
    assert f'{type(var_6.parent).__module__}.{type(var_6.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    module_0.COA(_subaccounts=var_1, rootspec=var_4)

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
    var_0 = None
    var_1 = module_0.COA()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_2 = var_1.__iter__()
    with pytest.raises(ValueError):
        var_1.add(var_1, var_2, var_0)

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
    var_1 = var_0.__hash__()
    assert var_1 == 5740354900026072187
    var_2 = list(var_0)
    var_3 = len(var_2)

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
    var_4 = var_0.add(var_3, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code is True
    assert var_4.name is True
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_5 = var_1.__repr__()
    assert var_5 == 'True'
    var_6 = var_0.add(var_3, var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_6.code is True
    assert var_6.name is True
    assert f'{type(var_6.parent).__module__}.{type(var_6.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_7 = var_5.__hash__()
    assert var_7 == 2221613377188479492

@pytest.mark.xfail(strict=True)
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
    var_3 = var_2.__repr__()
    assert var_3 == '1'
    var_4 = var_0.add(var_3, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code is True
    assert var_4.name is True
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_5 = var_0.print()
    var_6 = var_1.__repr__()
    assert var_6 == 'True'
    var_7 = var_0.add(var_3, var_1, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_7.code is True
    assert var_7.name is True
    assert f'{type(var_7.parent).__module__}.{type(var_7.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_8 = var_2.__hash__()
    assert var_8 == 1
    var_9 = var_6.__hash__()
    assert var_9 == 2221613377188479492
    var_10 = var_2.__eq__(var_3)
    var_1.print()

@pytest.mark.xfail(strict=True)
def test_case_11():
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
    var_4 = var_0.add(var_3, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code is True
    assert var_4.name is True
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_5 = var_3.__repr__()
    assert var_5 == "'1'"
    var_6 = var_0.print()
    var_7 = var_5.__eq__(var_3)
    assert var_7 is False
    var_8 = '"Gklx1ZmTp=\\B\tk)9'
    var_9 = var_0.add(var_3, var_5, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_9.code == "'1'"
    assert var_9.name == '"Gklx1ZmTp=\\B\tk)9'
    assert f'{type(var_9.parent).__module__}.{type(var_9.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_10 = var_7.__repr__()
    assert var_10 == 'False'
    var_11 = var_2.__eq__(var_6)
    var_2.print()

def test_case_12():
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
    var_4 = var_0.add(var_3, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code is True
    assert var_4.name is True
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_5 = var_0.print()
    var_6 = var_4.__eq__(var_3)
    var_7 = '"Gklx1ZmTp=\\B\tk)9'
    with pytest.raises(ValueError):
        var_0.add(var_3, var_1, var_7)

def test_case_13():
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
    var_4 = var_0.add(var_3, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_4.code is True
    assert var_4.name is True
    assert f'{type(var_4.parent).__module__}.{type(var_4.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_0.add(var_1, var_3, var_1)