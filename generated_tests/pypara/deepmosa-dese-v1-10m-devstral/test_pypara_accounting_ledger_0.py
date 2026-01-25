# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.ledger as module_0
import pypara.accounting.generic as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0.ReadInitialBalances()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.compile_general_ledger_program(var_0, var_0)
    var_2 = module_0.Ledger(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_2.entries == []
    var_2.add(var_2)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = ''
    var_2 = None
    var_3 = module_1.Balance(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_3.date is None
    assert var_3.value is None
    var_4 = {var_2: var_3}
    var_5 = module_0.build_general_ledger(var_1, var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_5.period == ''
    assert f'{type(var_5.ledgers).__module__}.{type(var_5.ledgers).__qualname__}' == 'builtins.dict'
    assert len(var_5.ledgers) == 1
    var_6 = module_0.Ledger(var_5, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert f'{type(var_6.account).__module__}.{type(var_6.account).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert f'{type(var_6.initial).__module__}.{type(var_6.initial).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_6.entries == []
    var_7 = module_0.compile_general_ledger_program(var_0, var_2)
    var_8 = var_1.__eq__(var_0)
    var_9 = var_6.__repr__()
    assert var_9 == "Ledger(account=GeneralLedger(period='', ledgers={None: Ledger(account=None, initial=Balance(date=None, value=None), entries=[])}), initial=Balance(date=None, value=None), entries=[])"
    var_10 = module_0.GeneralLedger(var_0, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_10.period is None
    assert var_10.ledgers is None
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
    var_0 = ''
    var_1 = {var_0: var_0}
    var_2 = module_0.build_general_ledger(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_2.period == ''
    assert f'{type(var_2.ledgers).__module__}.{type(var_2.ledgers).__qualname__}' == 'builtins.dict'
    assert len(var_2.ledgers) == 1

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = ''
    var_1 = None
    var_2 = {var_1: var_1}
    var_3 = module_0.build_general_ledger(var_0, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_3.period == ''
    assert f'{type(var_3.ledgers).__module__}.{type(var_3.ledgers).__qualname__}' == 'builtins.dict'
    assert len(var_3.ledgers) == 1
    module_0.build_general_ledger(var_1, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = ''
    var_1 = None
    var_2 = module_1.Balance(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_2.date is None
    assert var_2.value is None
    var_3 = {var_1: var_2}
    var_4 = module_0.build_general_ledger(var_0, var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_4.period == ''
    assert f'{type(var_4.ledgers).__module__}.{type(var_4.ledgers).__qualname__}' == 'builtins.dict'
    assert len(var_4.ledgers) == 1
    var_5 = module_0.Ledger(var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_5.account is None
    assert f'{type(var_5.initial).__module__}.{type(var_5.initial).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_5.entries == []
    var_6 = module_0.LedgerEntry(var_5, var_4, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.accounting.ledger.LedgerEntry'
    assert f'{type(var_6.ledger).__module__}.{type(var_6.ledger).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert f'{type(var_6.posting).__module__}.{type(var_6.posting).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert f'{type(var_6.balance).__module__}.{type(var_6.balance).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert f'{type(module_0.LedgerEntry.date).__module__}.{type(module_0.LedgerEntry.date).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.description).__module__}.{type(module_0.LedgerEntry.description).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.amount).__module__}.{type(module_0.LedgerEntry.amount).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.cntraccts).__module__}.{type(module_0.LedgerEntry.cntraccts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.is_debit).__module__}.{type(module_0.LedgerEntry.is_debit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.is_credit).__module__}.{type(module_0.LedgerEntry.is_credit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.debit).__module__}.{type(module_0.LedgerEntry.debit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.credit).__module__}.{type(module_0.LedgerEntry.credit).__qualname__}' == 'builtins.property'
    var_5.add(var_6)