# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._precord as module_0
import pytest


def test_case_0():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_0.set(var_0, var_0)

def test_case_2():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.update()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_1) == 0
    var_2 = module_0._PRecordEvolver(var_1, var_0, _ignore_extra=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._precord._PRecordEvolver'
    assert len(var_2) == 0
    with pytest.raises(AttributeError):
        var_2.set(var_1, var_1)

def test_case_3():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.update()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_1) == 0
    var_2 = var_1.evolver()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._precord._PRecordEvolver'
    assert len(var_2) == 0
    var_3 = var_1.set()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_3) == 0
    var_4 = module_0._PRecordEvolver(var_1, var_0, _ignore_extra=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._precord._PRecordEvolver'
    assert len(var_4) == 0
    var_5 = module_0.PRecord(**var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_5) == 0
    var_6 = var_0.__repr__()
    assert var_6 == 'PRecord()'
    with pytest.raises(AttributeError):
        var_4.set(var_1, var_1)

def test_case_4():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()
    var_2 = var_0.update()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_2) == 0
    var_3 = var_2.__reduce__()
    var_4 = module_0._PRecordEvolver(var_1, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._precord._PRecordEvolver'
    assert len(var_4) == 0
    var_5 = var_2.evolver()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._precord._PRecordEvolver'
    assert len(var_5) == 0
    var_6 = var_0.set()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_6) == 0
    var_7 = module_0._PRecordEvolver(var_6, var_6, _ignore_extra=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._precord._PRecordEvolver'
    assert len(var_7) == 0
    var_8 = var_0.__repr__()
    assert var_8 == 'PRecord()'
    with pytest.raises(AttributeError):
        var_2.set(var_5, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = '2W},y5\n_!(^6WZ6'
    var_2 = {var_1: var_1, var_1: var_0}
    module_0.PRecord(**var_2)

def test_case_6():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize(var_0)