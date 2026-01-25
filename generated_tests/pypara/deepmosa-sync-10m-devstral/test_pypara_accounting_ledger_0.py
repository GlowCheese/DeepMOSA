# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.ledger as module_0
import dataclasses as module_1
import pypara.accounting.generic as module_2

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
    var_2.add(var_0)

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
    var_1.add(var_0)

def test_case_4():
    var_0 = set()
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = {var_1: var_1, var_1: var_1, var_1: var_1, var_1: var_1, var_1: var_1}
    var_3 = module_0.build_general_ledger(var_1, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_3.period is True
    assert f'{type(var_3.ledgers).__module__}.{type(var_3.ledgers).__qualname__}' == 'builtins.dict'
    assert len(var_3.ledgers) == 1

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = '{'
    var_2 = {var_1: var_0, var_0: var_0, var_1: var_0, var_1: var_0}
    module_0.build_general_ledger(var_0, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = '{'
    var_2 = set()
    var_3 = var_1.__eq__(var_0)
    var_4 = {var_3: var_0, var_3: var_0, var_0: var_0, var_3: var_0, var_3: var_3, var_3: var_0}
    var_5 = module_0.build_general_ledger(var_3, var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert f'{type(var_5.period).__module__}.{type(var_5.period).__qualname__}' == 'builtins.NotImplementedType'
    assert f'{type(var_5.ledgers).__module__}.{type(var_5.ledgers).__qualname__}' == 'builtins.dict'
    assert len(var_5.ledgers) == 2
    var_6 = var_3.__repr__()
    assert var_6 == 'NotImplemented'
    var_7 = module_0.LedgerEntry(var_6, var_6, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.accounting.ledger.LedgerEntry'
    assert var_7.ledger == 'NotImplemented'
    assert var_7.posting == 'NotImplemented'
    assert var_7.balance is None
    assert f'{type(module_0.LedgerEntry.date).__module__}.{type(module_0.LedgerEntry.date).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.description).__module__}.{type(module_0.LedgerEntry.description).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.amount).__module__}.{type(module_0.LedgerEntry.amount).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.cntraccts).__module__}.{type(module_0.LedgerEntry.cntraccts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.is_debit).__module__}.{type(module_0.LedgerEntry.is_debit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.is_credit).__module__}.{type(module_0.LedgerEntry.is_credit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.debit).__module__}.{type(module_0.LedgerEntry.debit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.credit).__module__}.{type(module_0.LedgerEntry.credit).__qualname__}' == 'builtins.property'
    var_8 = module_2.Balance(var_3, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert f'{type(var_8.date).__module__}.{type(var_8.date).__qualname__}' == 'builtins.NotImplementedType'
    assert var_8.value is None
    var_9 = module_0.Ledger(var_2, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_9.account == {*()}
    assert f'{type(var_9.initial).__module__}.{type(var_9.initial).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_9.entries == []
    var_10 = module_0.compile_general_ledger_program(var_0, var_3)
    var_9.add(var_7)