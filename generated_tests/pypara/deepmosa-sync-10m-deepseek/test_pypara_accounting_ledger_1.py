# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.ledger as module_0
import dataclasses as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.GeneralLedgerProgram()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}
    var_2 = None
    var_3 = module_0.compile_general_ledger_program(var_0, var_0)
    var_4 = module_0.Ledger(var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_4.account is None
    assert var_4.initial is None
    assert var_4.entries == []
    var_5 = b'\xee\xb7\xe4o#sm\x89;'
    var_6 = var_2.__eq__(var_2)
    assert var_6 is True
    var_4.add(var_5)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_1.dataclass(var_0, order=var_0)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_2 = module_0.compile_general_ledger_program(var_1, var_1)
    var_2.__call__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.Ledger(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_1.account is None
    assert var_1.initial is None
    assert var_1.entries == []
    var_1.add(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}
    var_2 = var_1.__repr__()
    assert var_2 == 'GeneralLedger(period={}, ledgers={})'
    var_3 = {var_2: var_2, var_2: var_2, var_2: var_2, var_2: var_2}
    module_0.build_general_ledger(var_2, var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}
    var_2 = var_1.__repr__()
    assert var_2 == 'GeneralLedger(period={}, ledgers={})'
    module_0.build_general_ledger(var_2, var_2, var_0)