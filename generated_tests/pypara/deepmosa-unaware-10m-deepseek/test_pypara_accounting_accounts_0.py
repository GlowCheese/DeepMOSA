# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.accounts as module_0
import dataclasses as module_1
import collections as module_2
import typing as module_3

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
    var_2 = var_0.__iter__()
    var_3 = var_1.__hash__()
    assert var_3 == 1
    var_4 = None
    var_5 = var_3.__repr__()
    assert var_5 == '1'
    var_6 = module_1.dataclass(var_4, repr=var_5, eq=var_1, order=var_4, slots=var_4)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_7 = var_3.__repr__()
    assert var_7 == '1'
    var_8 = var_0.print()
    var_9 = var_0.nodify(var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.accounting.accounts.COA.Node'
    assert var_9.account == 1
    assert var_9.children == []
    var_10 = var_0.add(var_5, var_3, var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_10.code == 1
    assert var_10.name == '1'
    assert f'{type(var_10.parent).__module__}.{type(var_10.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_11 = var_1.__hash__()
    assert var_11 == 1
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

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.COA()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_2 = var_1.__eq__(var_0)
    var_3 = var_1.__iter__()
    module_2.OrderedDict(*var_3)

@pytest.mark.xfail(strict=True)
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
    var_3 = None
    var_4 = var_2.__repr__()
    assert var_4 == '1'
    var_5 = var_0.print()
    var_6 = var_0.add(var_4, var_2, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_6.code == 1
    assert var_6.name is True
    assert f'{type(var_6.parent).__module__}.{type(var_6.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_7 = var_0.print()
    var_8 = var_1.__repr__()
    assert var_8 == 'True'
    var_1.__delattr__(var_3)

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
    var_4 = module_3.Generic()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typing.Generic'
    assert module_3.EXCLUDED_ATTRIBUTES == ['__parameters__', '__orig_bases__', '__orig_class__', '_is_protocol', '_is_runtime_protocol', '__abstractmethods__', '__annotations__', '__dict__', '__doc__', '__init__', '__module__', '__new__', '__slots__', '__subclasshook__', '__weakref__', '__class_getitem__', '_MutableMapping__marker']
    assert f'{type(module_3.T).__module__}.{type(module_3.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT).__module__}.{type(module_3.VT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.V_co).__module__}.{type(module_3.V_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.T_contra).__module__}.{type(module_3.T_contra).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.CT_co).__module__}.{type(module_3.CT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.AnyStr).__module__}.{type(module_3.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_3.TYPE_CHECKING is False
    var_5 = var_0.add(var_3, var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_5.code == 1
    assert f'{type(var_5.name).__module__}.{type(var_5.name).__qualname__}' == 'typing.Generic'
    assert f'{type(var_5.parent).__module__}.{type(var_5.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_6 = var_0.add(var_3, var_2, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_6.code == 1
    assert f'{type(var_6.name).__module__}.{type(var_6.name).__qualname__}' == 'typing.Generic'
    assert f'{type(var_6.parent).__module__}.{type(var_6.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_4.__delattr__(var_0)

def test_case_11():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_0.__iter__()
    var_3 = var_1.__hash__()
    assert var_3 == 1
    var_4 = var_3.__repr__()
    assert var_4 == '1'
    var_5 = var_3.__repr__()
    assert var_5 == '1'
    var_6 = var_0.print()
    var_7 = var_0.add(var_4, var_3, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_7.code == 1
    assert var_7.name == '1'
    assert f'{type(var_7.parent).__module__}.{type(var_7.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_0.add(var_5, var_3, var_1)

def test_case_12():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_0.__iter__()
    var_3 = var_1.__hash__()
    assert var_3 == 1
    var_4 = var_3.__repr__()
    assert var_4 == '1'
    var_5 = var_3.__repr__()
    assert var_5 == '1'
    var_6 = var_0.print()
    var_7 = var_0.add(var_4, var_3, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_7.code == 1
    assert var_7.name == '1'
    assert f'{type(var_7.parent).__module__}.{type(var_7.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    var_8 = 'c[5YZ((,hatd'
    var_9 = module_0.SubAccount(var_3, var_8, var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_9.code == 1
    assert var_9.name == 'c[5YZ((,hatd'
    assert f'{type(var_9.parent).__module__}.{type(var_9.parent).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    var_10 = "JF'_U,#/t%Fm'Xh2"
    var_11 = var_0.add(var_4, var_9, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert f'{type(var_11.code).__module__}.{type(var_11.code).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_11.name == "JF'_U,#/t%Fm'Xh2"
    assert f'{type(var_11.parent).__module__}.{type(var_11.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    var_12 = var_1.__repr__()
    assert var_12 == 'True'
    with pytest.raises(ValueError):
        var_0.add(var_12, var_4, var_5)

def test_case_13():
    var_0 = module_0.COA()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.accounting.accounts.COA'
    assert module_0.COA.rootspec is None
    assert f'{type(module_0.COA.accounts).__module__}.{type(module_0.COA.accounts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.toplevel).__module__}.{type(module_0.COA.toplevel).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.COA.structure).__module__}.{type(module_0.COA.structure).__qualname__}' == 'builtins.property'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_0.__iter__()
    var_3 = var_1.__hash__()
    assert var_3 == 1
    var_4 = None
    var_5 = var_3.__repr__()
    assert var_5 == '1'
    var_6 = var_3.__repr__()
    assert var_6 == '1'
    var_7 = var_0.print()
    var_8 = var_0.add(var_5, var_3, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.accounting.accounts.SubAccount'
    assert var_8.code == 1
    assert var_8.name == '1'
    assert f'{type(var_8.parent).__module__}.{type(var_8.parent).__qualname__}' == 'pypara.accounting.accounts.RootAccount'
    assert f'{type(module_0.SubAccount.type).__module__}.{type(module_0.SubAccount.type).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SubAccount.coa).__module__}.{type(module_0.SubAccount.coa).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_0.add(var_3, var_6, var_4)