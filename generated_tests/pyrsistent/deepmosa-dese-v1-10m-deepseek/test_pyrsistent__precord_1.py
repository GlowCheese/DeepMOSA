# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_0
import pytest


def test_case_0():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'

def test_case_1():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'PRecord()'

def test_case_2():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_1) == 0

def test_case_3():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()
    var_2 = None
    var_3 = module_1.pmap(pre_size=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__str__()
    assert var_4 == 'pmap({})'

def test_case_4():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_1) == 0
    var_2 = module_1.pmap()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__repr__()
    assert var_3 == 'pmap({})'
    var_4 = var_0.__str__()
    assert var_4 == 'PRecord()'
    module_0._PRecordMeta()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = ':7%]\tj`\rU'
    var_2 = None
    var_3 = {var_1: var_2}
    var_0.set(**var_3)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_0.set(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = ''
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    module_0.PRecord(**var_1)