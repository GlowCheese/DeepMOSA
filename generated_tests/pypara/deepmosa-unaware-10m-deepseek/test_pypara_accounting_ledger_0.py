# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.ledger as module_0
import dataclasses as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0.ReadInitialBalances()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.compile_general_ledger_program(var_0, var_0)
    var_2 = module_0.Ledger(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_2.account is None
    assert var_2.initial is None
    assert var_2.entries == []
    var_2.add(var_2)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_1.dataclass(init=var_0, repr=var_0, order=var_0, unsafe_hash=var_0, match_args=var_0)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_2 = var_1.__hash__()
    var_3 = var_2.__hash__()
    var_4 = var_3.__hash__()
    var_5 = var_4.__repr__()
    var_6 = var_5.__repr__()
    var_7 = module_0.compile_general_ledger_program(var_0, var_0)
    var_8 = var_7.__repr__()
    var_9 = var_6.__eq__(var_1)
    var_10 = var_7.__eq__(var_8)
    var_7.__call__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.Ledger(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_1.account is None
    assert var_1.initial is None
    assert var_1.entries == []
    var_1.add(var_1)

def test_case_4():
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}
    var_2 = var_0.__repr__()
    assert var_2 == '{}'
    var_3 = {var_2: var_2, var_2: var_2, var_2: var_2}
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