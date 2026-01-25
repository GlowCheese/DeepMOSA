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
    var_2 = module_0.Ledger(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.ledger.Ledger'
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
    var_0 = None
    var_1 = module_2.Balance(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_1.date is None
    assert var_1.value is None
    var_2 = {var_0: var_1}
    module_0.build_general_ledger(var_2, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '}'
    var_1 = {}
    var_2 = module_0.build_general_ledger(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_2.period == {}
    assert var_2.ledgers == {}
    module_0.build_general_ledger(var_0, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_2.Balance(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_1.date is None
    assert var_1.value is None
    var_2 = {}
    var_3 = module_0.build_general_ledger(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_3.period == {}
    assert var_3.ledgers == {}
    var_4 = module_0.Ledger(var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_4.account is None
    assert var_4.initial is None
    assert var_4.entries == []
    var_5 = module_0.LedgerEntry(var_3, var_4, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.ledger.LedgerEntry'
    assert f'{type(var_5.ledger).__module__}.{type(var_5.ledger).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert f'{type(var_5.posting).__module__}.{type(var_5.posting).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert f'{type(var_5.balance).__module__}.{type(var_5.balance).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert f'{type(module_0.LedgerEntry.date).__module__}.{type(module_0.LedgerEntry.date).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.description).__module__}.{type(module_0.LedgerEntry.description).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.amount).__module__}.{type(module_0.LedgerEntry.amount).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.cntraccts).__module__}.{type(module_0.LedgerEntry.cntraccts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.is_debit).__module__}.{type(module_0.LedgerEntry.is_debit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.is_credit).__module__}.{type(module_0.LedgerEntry.is_credit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.debit).__module__}.{type(module_0.LedgerEntry.debit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.credit).__module__}.{type(module_0.LedgerEntry.credit).__qualname__}' == 'builtins.property'
    var_6 = var_5.__repr__()
    assert var_6 == 'LedgerEntry(ledger=GeneralLedger(period={}, ledgers={}), posting=Ledger(account=None, initial=None, entries=[]), balance=GeneralLedger(period={}, ledgers={}))'
    var_7 = var_3.__eq__(var_0)
    var_8 = module_0.Ledger(var_7, var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert f'{type(var_8.account).__module__}.{type(var_8.account).__qualname__}' == 'builtins.NotImplementedType'
    assert f'{type(var_8.initial).__module__}.{type(var_8.initial).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_8.entries == []
    var_9 = module_1.field(init=var_7, repr=var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'dataclasses.Field'
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_1.Field.compare).__module__}.{type(module_1.Field.compare).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.default).__module__}.{type(module_1.Field.default).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.default_factory).__module__}.{type(module_1.Field.default_factory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.hash).__module__}.{type(module_1.Field.hash).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.init).__module__}.{type(module_1.Field.init).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.kw_only).__module__}.{type(module_1.Field.kw_only).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.metadata).__module__}.{type(module_1.Field.metadata).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.name).__module__}.{type(module_1.Field.name).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.repr).__module__}.{type(module_1.Field.repr).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.type).__module__}.{type(module_1.Field.type).__qualname__}' == 'builtins.member_descriptor'
    var_10 = var_1.__repr__()
    assert var_10 == 'Balance(date=None, value=None)'
    var_8.add(var_5)