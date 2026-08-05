# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.accounting.ledger as module_0
import dataclasses as module_1
import pypara.accounting.generic as module_2
import pypara.accounting.journaling as module_3

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
    var_1 = set()
    var_2 = {}
    var_3 = module_0.build_general_ledger(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_3.period == {}
    assert var_3.ledgers == {}
    var_4 = var_3.__eq__(var_0)
    var_5 = module_0.Ledger(var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_5.account == {*()}
    assert f'{type(var_5.initial).__module__}.{type(var_5.initial).__qualname__}' == 'builtins.NotImplementedType'
    assert var_5.entries == []
    var_6 = module_0.compile_general_ledger_program(var_0, var_0)
    var_7 = module_1.field(init=var_0, repr=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'dataclasses.Field'
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
    var_8 = var_7.__eq__(var_0)
    var_9 = var_8.__eq__(var_1)
    var_6.__call__(var_7)

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
    var_0 = None
    var_1 = module_1.field(init=var_0, metadata=var_0, kw_only=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'dataclasses.Field'
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
    var_2 = var_1.__eq__(var_0)
    var_3 = module_2.Balance(var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_3.date is None
    assert var_3.value is None
    var_4 = {var_0: var_3}
    module_0.build_general_ledger(var_4, var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = '{'
    var_2 = set()
    var_3 = module_2.Balance(var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_3.date is None
    assert var_3.value is None
    var_4 = {}
    var_5 = module_0.build_general_ledger(var_4, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.accounting.ledger.GeneralLedger'
    assert var_5.period == {}
    assert var_5.ledgers == {}
    var_6 = var_5.__eq__(var_0)
    var_7 = module_0.Ledger(var_2, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_7.account == {*()}
    assert f'{type(var_7.initial).__module__}.{type(var_7.initial).__qualname__}' == 'pypara.accounting.generic.Balance'
    assert var_7.entries == []
    var_8 = module_0.compile_general_ledger_program(var_0, var_0)
    var_9 = module_1.field(init=var_0, repr=var_0)
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
    var_10 = var_9.__eq__(var_0)
    var_11 = module_3.JournalEntry(var_9, var_1, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.accounting.journaling.JournalEntry'
    assert f'{type(var_11.date).__module__}.{type(var_11.date).__qualname__}' == 'dataclasses.Field'
    assert var_11.description == '{'
    assert f'{type(var_11.source).__module__}.{type(var_11.source).__qualname__}' == 'builtins.NotImplementedType'
    assert var_11.postings == []
    assert f'{type(module_3.JournalEntry.increments).__module__}.{type(module_3.JournalEntry.increments).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.JournalEntry.decrements).__module__}.{type(module_3.JournalEntry.decrements).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.JournalEntry.debits).__module__}.{type(module_3.JournalEntry.debits).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.JournalEntry.credits).__module__}.{type(module_3.JournalEntry.credits).__qualname__}' == 'builtins.property'
    var_12 = var_10.__repr__()
    assert var_12 == 'NotImplemented'
    var_13 = module_0.Ledger(var_7, var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert f'{type(var_13.account).__module__}.{type(var_13.account).__qualname__}' == 'pypara.accounting.ledger.Ledger'
    assert var_13.initial is None
    assert var_13.entries == []
    var_14 = module_0.LedgerEntry(var_11, var_10, var_6)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.accounting.ledger.LedgerEntry'
    assert f'{type(var_14.ledger).__module__}.{type(var_14.ledger).__qualname__}' == 'pypara.accounting.journaling.JournalEntry'
    assert f'{type(var_14.posting).__module__}.{type(var_14.posting).__qualname__}' == 'builtins.NotImplementedType'
    assert f'{type(var_14.balance).__module__}.{type(var_14.balance).__qualname__}' == 'builtins.NotImplementedType'
    assert f'{type(module_0.LedgerEntry.date).__module__}.{type(module_0.LedgerEntry.date).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.description).__module__}.{type(module_0.LedgerEntry.description).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.amount).__module__}.{type(module_0.LedgerEntry.amount).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.cntraccts).__module__}.{type(module_0.LedgerEntry.cntraccts).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.is_debit).__module__}.{type(module_0.LedgerEntry.is_debit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.is_credit).__module__}.{type(module_0.LedgerEntry.is_credit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.debit).__module__}.{type(module_0.LedgerEntry.debit).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.LedgerEntry.credit).__module__}.{type(module_0.LedgerEntry.credit).__qualname__}' == 'builtins.property'
    var_7.add(var_14)