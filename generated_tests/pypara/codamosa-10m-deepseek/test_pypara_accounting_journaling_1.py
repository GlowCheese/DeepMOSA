# Check out: https://github.com/GlowCheese/deepmosa
import pypara.accounting.journaling as module_0


def test_case_0():
    var_0 = None
    var_1 = module_0.JournalEntry(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.accounting.journaling.JournalEntry'
    assert var_1.date is None
    assert var_1.description is None
    assert var_1.source is None
    assert var_1.postings == []
    assert f'{type(module_0.JournalEntry.increments).__module__}.{type(module_0.JournalEntry.increments).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.JournalEntry.decrements).__module__}.{type(module_0.JournalEntry.decrements).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.JournalEntry.debits).__module__}.{type(module_0.JournalEntry.debits).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.JournalEntry.credits).__module__}.{type(module_0.JournalEntry.credits).__qualname__}' == 'builtins.property'
    var_2 = var_1.validate()

def test_case_1():
    pass