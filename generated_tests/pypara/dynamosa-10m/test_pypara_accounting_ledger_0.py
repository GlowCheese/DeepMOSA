# Check out: https://github.com/GlowCheese/deepmosa
import pypara.accounting.ledger as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0.ReadInitialBalances()

def test_case_1():
    var_0 = None
    var_1 = module_0.compile_general_ledger_program(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.compile_general_ledger_program(var_0, var_0)
    var_1.__call__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.Ledger(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_1.account is None
    assert var_1.initial is None
    assert var_1.entries == []
    var_1.add(var_0)

def test_case_4():
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = {}
    var_1 = var_0.__repr__()
    assert var_1 == '{}'
    module_0.build_general_ledger(var_1, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    module_0.build_general_ledger(var_1, var_1, var_1)