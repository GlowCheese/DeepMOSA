# Check out: https://github.com/GlowCheese/deepmosa
import dataclasses as module_1

import pypara.accounting.ledger as module_0
import pytest


def test_case_0():
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.ReadInitialBalances()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}
    var_2 = module_0.compile_general_ledger_program(var_1, var_1)
    var_3 = var_1.__repr__()
    assert var_3 == 'GeneralLedger(period={}, ledgers={})'
    module_0.build_general_ledger(var_3, var_3, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.Ledger(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_1.account is None
    assert var_1.initial is None
    assert var_1.entries == []
    var_1.add(var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_1.dataclass(var_0, frozen=var_0, kw_only=var_0)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_2 = var_1.__hash__()
    var_3 = {var_2: var_1, var_1: var_2, var_2: var_2, var_2: var_2}
    module_0.build_general_ledger(var_0, var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.LedgerEntry(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.LedgerEntry'
    assert var_1.ledger is None
    assert var_1.posting is None
    assert var_1.balance is None
    assert f'{type(module_0.LedgerEntry.date).__module__}.{type(module_0.LedgerEntry.date).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.description).__module__}.{type(module_0.LedgerEntry.description).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.amount).__module__}.{type(module_0.LedgerEntry.amount).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.cntraccts).__module__}.{type(module_0.LedgerEntry.cntraccts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.is_debit).__module__}.{type(module_0.LedgerEntry.is_debit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.is_credit).__module__}.{type(module_0.LedgerEntry.is_credit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.debit).__module__}.{type(module_0.LedgerEntry.debit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.credit).__module__}.{type(module_0.LedgerEntry.credit).__qualname__}' == 'builtins.property'
    var_2 = var_1.__eq__(var_0)
    var_3 = module_0.compile_general_ledger_program(var_0, var_0)
    var_4 = module_1.dataclass(eq=var_0, slots=var_0)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_5 = var_4.__eq__(var_0)
    var_3.__call__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}
    var_2 = var_0.__repr__()
    assert var_2 == '{}'
    module_0.build_general_ledger(var_2, var_2, var_0)