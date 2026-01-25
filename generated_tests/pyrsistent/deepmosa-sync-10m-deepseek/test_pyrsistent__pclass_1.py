# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._field_common as module_1
import pyrsistent._pclass as module_0
import pytest


def test_case_0():
    pass

def test_case_1():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.PClass()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.__ne__(var_0)
    assert var_2 is True
    var_3 = module_1.field()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_1.PFIELD_NO_TYPE == ()
    assert f'{type(module_1.PFIELD_NO_INITIAL).__module__}.{type(module_1.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_1.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_1.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_1._PField.factory).__module__}.{type(module_1._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_1._PField.initial).__module__}.{type(module_1._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1._PField.invariant).__module__}.{type(module_1._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1._PField.mandatory).__module__}.{type(module_1._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1._PField.serializer).__module__}.{type(module_1._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1._PField.type).__module__}.{type(module_1._PField.type).__qualname__}' == 'builtins.member_descriptor'
    var_3.persistent()

def test_case_3():
    var_0 = None
    var_1 = module_0._PClassEvolver(var_0, var_0)
    var_2 = var_1.persistent()
    var_3 = var_1.persistent()
    var_3.transform()

def test_case_4():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()

def test_case_5():
    var_0 = module_0.PClass()
    var_1 = var_0.__repr__()
    var_2 = var_0.evolver()
    module_1.check_type(var_1, var_2, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'PClass()'
    var_0.set(*var_1)

def test_case_7():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    module_1.field(var_0, mandatory=var_0, factory=var_0, serializer=var_0)

def test_case_9():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__ne__(var_0)
    assert var_1 is False

def test_case_10():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__hash__()
    assert var_1 == 5740354900026072187

def test_case_11():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'

def test_case_12():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()
    var_1.__delitem__(var_1)

def test_case_13():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__hash__()
    assert var_1 == 5740354900026072187
    var_2 = var_0.transform()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pclass.PClass'
    with pytest.raises(AttributeError):
        var_0.__setattr__(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'
    var_1.set(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.set()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pclass.PClass'
    var_0.remove(var_1)

def test_case_16():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()

def test_case_17():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()
    var_2 = var_0.__repr__()
    assert var_2 == 'PClass()'
    var_3 = var_0.transform()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pclass.PClass'

def test_case_18():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()
    var_2 = var_1.set(var_1, var_1)

def test_case_19():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()
    var_2 = var_1.__setattr__(var_0, var_1)
    var_1.__delattr__(var_1)

def test_case_20():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()
    var_2 = var_1.__setitem__(var_1, var_1)

def test_case_21():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    with pytest.raises(AttributeError):
        var_0.__delattr__(var_0)

def test_case_22():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()
    var_2 = var_1.set(var_1, var_1)
    var_3 = var_0.__reduce__()
    var_2.persistent()

def test_case_23():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'