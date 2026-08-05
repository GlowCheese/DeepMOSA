# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.ledger as module_0
import dataclasses as module_1
import pypara.accounting.journaling as module_2

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
    var_2.add(var_1)

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
    var_0 = {}
    var_1 = module_0.build_general_ledger(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_1.period == {}
    assert var_1.ledgers == {}

def test_case_5():
    var_0 = 1
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_2.JournalEntry(var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.journaling.JournalEntry'
    assert var_2.date == {1: 1}
    assert var_2.description == 1
    assert var_2.source == 1
    assert var_2.postings == []
    assert f'{type(module_2.JournalEntry.increments).__module__}.{type(module_2.JournalEntry.increments).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.JournalEntry.decrements).__module__}.{type(module_2.JournalEntry.decrements).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.JournalEntry.debits).__module__}.{type(module_2.JournalEntry.debits).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.JournalEntry.credits).__module__}.{type(module_2.JournalEntry.credits).__qualname__}' == 'builtins.property'
    var_3 = [var_2, var_2, var_2, var_2, var_2, var_2]
    var_4 = module_0.build_general_ledger(var_0, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_4.period == 1
    assert f'{type(var_4.ledgers).__module__}.{type(var_4.ledgers).__qualname__}' == 'builtins.dict'
    assert len(var_4.ledgers) == 1

def test_case_6():
    var_0 = 1
    var_1 = {}
    var_2 = module_2.JournalEntry(var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.journaling.JournalEntry'
    assert var_2.date == {}
    assert var_2.description == 1
    assert var_2.source == 1
    assert var_2.postings == []
    assert f'{type(module_2.JournalEntry.increments).__module__}.{type(module_2.JournalEntry.increments).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.JournalEntry.decrements).__module__}.{type(module_2.JournalEntry.decrements).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.JournalEntry.debits).__module__}.{type(module_2.JournalEntry.debits).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.JournalEntry.credits).__module__}.{type(module_2.JournalEntry.credits).__qualname__}' == 'builtins.property'
    var_3 = [var_2, var_2, var_2, var_2, var_2, var_2]
    var_4 = module_0.build_general_ledger(var_0, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_4.period == 1
    assert var_4.ledgers == {}