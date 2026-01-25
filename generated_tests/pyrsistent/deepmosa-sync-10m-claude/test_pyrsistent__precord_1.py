# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_2
import pyrsistent._precord as module_1
import pytest


def test_case_0():
    var_0 = module_0.field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'

def test_case_1():
    var_0 = module_1.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_1.PFIELD_NO_INITIAL).__module__}.{type(module_1.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.PRecord.create).__module__}.{type(module_1.PRecord.create).__qualname__}' == 'builtins.method'

def test_case_2():
    var_0 = module_1.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_1.PFIELD_NO_INITIAL).__module__}.{type(module_1.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.PRecord.create).__module__}.{type(module_1.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__str__()
    assert var_1 == 'PRecord()'

def test_case_3():
    var_0 = module_1.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_1.PFIELD_NO_INITIAL).__module__}.{type(module_1.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.PRecord.create).__module__}.{type(module_1.PRecord.create).__qualname__}' == 'builtins.method'
    with pytest.raises(AttributeError):
        var_0.set(var_0, var_0)

def test_case_4():
    var_0 = module_1.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_1.PFIELD_NO_INITIAL).__module__}.{type(module_1.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.PRecord.create).__module__}.{type(module_1.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()
    var_2 = None
    var_3 = var_2.__str__()
    assert var_3 == 'None'

def test_case_5():
    var_0 = module_1.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_1.PFIELD_NO_INITIAL).__module__}.{type(module_1.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.PRecord.create).__module__}.{type(module_1.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_1.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_1.PFIELD_NO_INITIAL).__module__}.{type(module_1.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.PRecord.create).__module__}.{type(module_1.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_1) == 0
    var_2 = module_2.pmap()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__repr__()
    assert var_3 == 'pmap({})'
    var_4 = var_0.__str__()
    assert var_4 == 'PRecord()'
    module_1._PRecordMeta()

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_1.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_1.PFIELD_NO_INITIAL).__module__}.{type(module_1.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.PRecord.create).__module__}.{type(module_1.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = ':7%]\tj`\rU'
    var_2 = None
    var_3 = {var_1: var_2}
    var_0.set(**var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = ''
    var_1 = None
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1, var_0: var_1}
    module_1.PRecord(**var_2)