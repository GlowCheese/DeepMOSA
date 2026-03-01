# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._checked_types as module_0
import typing as module_1
import abc as module_2
import pyrsistent._pvector as module_3

def test_case_0():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.get_type(var_0)

def test_case_2():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'

def test_case_3():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'

def test_case_4():
    var_0 = module_0.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_5():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 1
    var_2 = module_0.wrap_invariant(var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.CheckedTypeError(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_1.source_class is None
    assert var_1.expected_types is None
    assert var_1.actual_type is None
    assert var_1.actual_value is None
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    module_0.get_type(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = module_0.CheckedPMap()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    module_0.get_types(var_1)

def test_case_8():
    var_0 = module_0.optional()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_9():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.append(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 1

def test_case_10():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    with pytest.raises(TypeError):
        module_0.maybe_parse_user_type(var_1)

def test_case_11():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 1
    var_3 = var_2.__str__()
    assert var_3 == 'CheckedPSet([None])'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'CheckedPSet()'
    var_2 = var_0.serialize()
    var_3 = var_0.__repr__()
    assert var_3 == 'CheckedPSet()'
    var_2.__hash__()

def test_case_13():
    var_0 = module_0.optional()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.maybe_parse_user_type(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = None
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = var_0.add(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4) == 1
    var_4.serialize()

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.append(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 1
    var_2 = var_1.__reduce__()
    var_3 = module_0.InvariantException()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_3.invariant_errors == ()
    assert var_3.missing_fields == ()
    var_4 = var_3.__str__()
    assert var_4 == ', invariant_errors=[], missing_fields=[]'
    module_0.get_type(var_2)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = module_0.InvariantException(missing_fields=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_1.invariant_errors == ()
    assert var_1.missing_fields is None
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.optional()
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = var_2.__str__()
    assert var_4 == "(<class 'NoneType'>,)"
    var_5 = module_1.Generic()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typing.Generic'
    assert module_1.EXCLUDED_ATTRIBUTES == ['__parameters__', '__orig_bases__', '__orig_class__', '_is_protocol', '_is_runtime_protocol', '__abstractmethods__', '__annotations__', '__dict__', '__doc__', '__init__', '__module__', '__new__', '__slots__', '__subclasshook__', '__weakref__', '__class_getitem__', '_MutableMapping__marker']
    assert f'{type(module_1.T).__module__}.{type(module_1.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT).__module__}.{type(module_1.VT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.V_co).__module__}.{type(module_1.V_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.T_contra).__module__}.{type(module_1.T_contra).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CT_co).__module__}.{type(module_1.CT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.AnyStr).__module__}.{type(module_1.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_1.TYPE_CHECKING is False
    var_6 = module_0.CheckedPVector()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_6) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_7 = var_6.__repr__()
    assert var_7 == 'CheckedPVector([])'
    module_0.get_type(var_3)

def test_case_17():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.extend(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 0
    var_3 = var_0.__iter__()
    var_4 = module_0.CheckedPMap()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_5 = module_0.InvariantException(var_2, *var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_5.invariant_errors == ()
    assert var_5.missing_fields == ()
    var_6 = var_0.__hash__()
    assert var_6 == 133146708735736
    var_7 = module_0.maybe_parse_user_type(var_2)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = module_0.InvariantException()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_2.invariant_errors == ()
    assert var_2.missing_fields == ()
    var_3 = var_2.__str__()
    assert var_3 == ', invariant_errors=[], missing_fields=[]'
    var_4 = module_0.wrap_invariant(var_1)
    var_5 = var_3.__repr__()
    assert var_5 == "', invariant_errors=[], missing_fields=[]'"
    var_4.serialize(var_3)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = module_0.optional()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.get_types(var_0)
    var_2 = module_0.maybe_parse_user_type(var_0)
    var_3 = var_0.__str__()
    assert var_3 == "(<class 'NoneType'>,)"
    var_4 = module_1.Generic()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typing.Generic'
    assert module_1.EXCLUDED_ATTRIBUTES == ['__parameters__', '__orig_bases__', '__orig_class__', '_is_protocol', '_is_runtime_protocol', '__abstractmethods__', '__annotations__', '__dict__', '__doc__', '__init__', '__module__', '__new__', '__slots__', '__subclasshook__', '__weakref__', '__class_getitem__', '_MutableMapping__marker']
    assert f'{type(module_1.T).__module__}.{type(module_1.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT).__module__}.{type(module_1.VT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.V_co).__module__}.{type(module_1.V_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.T_contra).__module__}.{type(module_1.T_contra).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CT_co).__module__}.{type(module_1.CT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.AnyStr).__module__}.{type(module_1.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_1.TYPE_CHECKING is False
    module_0.get_type(var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = None
    var_3 = module_0.CheckedPVector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_4 = module_0.InvariantException(missing_fields=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_4.invariant_errors == ()
    assert var_4.missing_fields is None
    var_5 = module_0.CheckedPSet()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_5) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_6 = var_0.evolver()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap.Evolver'
    assert len(var_6) == 0
    var_7 = var_5.__reduce__()
    var_8 = module_0.optional()
    var_9 = var_0.set(var_2, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_9) == 1
    var_10 = module_0.maybe_parse_user_type(var_8)
    var_11 = var_3.extend(var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_11) == 1
    module_2.ABCMeta(**var_2)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_0.set(var_0, var_0)

def test_case_22():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 1
    var_2 = var_1.__reduce__()
    with pytest.raises(TypeError):
        var_1.set(var_2, var_1)

def test_case_23():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 1

def test_case_24():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = None
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = var_0.__iter__()
    var_5 = module_0.CheckedPMap()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_5) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_6 = module_0.InvariantException(var_3, *var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_6.invariant_errors == (None,)
    assert var_6.missing_fields == ()
    var_7 = var_0.__hash__()
    assert var_7 == 133146708735736
    with pytest.raises(TypeError):
        module_0.maybe_parse_user_type(var_3)

def test_case_25():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 1
    with pytest.raises(KeyError):
        var_1.remove(var_1)

def test_case_26():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'CheckedPVector([])'
    var_2 = var_0.append(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 1
    var_3 = module_0.CheckedPMap()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_4 = var_3.items()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_4) == 0
    with pytest.raises(TypeError):
        module_0.store_invariants(var_2, var_4, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = None
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = module_0.CheckedPMap()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_5 = var_0.add(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_5) == 1
    var_6 = var_4.serialize(var_5)
    var_7 = var_0.__repr__()
    assert var_7 == 'CheckedPSet()'
    var_8 = module_0.InvariantException(var_3, *var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_8.invariant_errors == (None,)
    assert var_8.missing_fields == ()
    var_9 = var_5.update(var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_9) == 1
    var_10 = var_5.discard(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_10) == 1
    var_11 = var_1.__str__()
    assert var_11 == 'CheckedPVector([])'
    var_12 = var_11.__str__()
    assert var_12 == 'CheckedPVector([])'
    var_13 = module_0.CheckedPSet()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_13) == 0
    var_14 = var_4.set(var_2, var_7)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_14) == 1
    module_1.Generic(*var_8)

def test_case_28():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = module_0.CheckedPMap()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_3 = var_0.__repr__()
    assert var_3 == 'CheckedPSet()'
    var_4 = var_2.serialize()
    var_5 = var_3.__str__()
    assert var_5 == 'CheckedPSet()'

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = None
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = module_0.CheckedPMap()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_5 = var_0.__repr__()
    assert var_5 == 'CheckedPSet()'
    var_6 = None
    var_7 = module_0.InvariantException(var_3, *var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_7.invariant_errors == (None,)
    assert var_7.missing_fields == ()
    var_8 = var_6.__str__()
    assert var_8 == 'None'
    var_9 = var_7.__str__()
    assert var_9 == ', invariant_errors=[None], missing_fields=[]'
    module_0.get_type(var_6)

def test_case_30():
    var_0 = module_0.optional()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = var_1.__str__()
    assert var_2 == "(<class 'NoneType'>,)"
    var_3 = module_0.maybe_parse_user_type(var_2)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.__reduce__()
    var_3 = var_0.__repr__()
    assert var_3 == 'CheckedPSet()'
    module_0.InvariantException(var_2, *var_1)

def test_case_32():
    var_0 = module_0.CheckedType()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    with pytest.raises(NotImplementedError):
        var_0.serialize()

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = None
    var_3 = module_3.python_pvector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 0
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_3.BRANCH_FACTOR == 32
    assert module_3.BIT_MASK == 31
    assert module_3.SHIFT == 5
    var_4 = var_1.append(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 1
    var_5 = module_0.CheckedPMap()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_5) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_6 = var_0.add(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_6) == 1
    var_6.__new__(var_6, var_4)

def test_case_34():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__add__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = None
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = module_0.CheckedPMap()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_5 = var_0.add(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_5) == 1
    var_6 = var_4.serialize(var_5)
    var_7 = var_0.__repr__()
    assert var_7 == 'CheckedPSet()'
    var_8 = var_0.copy()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_8) == 0
    var_9 = module_0.InvariantException(var_3, *var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_9.invariant_errors == (None,)
    assert var_9.missing_fields == ()
    var_10 = module_0.CheckedKeyTypeError(var_3, var_8, var_4, var_3, **var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedKeyTypeError'
    assert f'{type(var_10.source_class).__module__}.{type(var_10.source_class).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_10.source_class) == 1
    assert f'{type(var_10.expected_types).__module__}.{type(var_10.expected_types).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_10.expected_types) == 0
    assert f'{type(var_10.actual_type).__module__}.{type(var_10.actual_type).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_10.actual_type) == 0
    assert f'{type(var_10.actual_value).__module__}.{type(var_10.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_10.actual_value) == 1
    var_11 = var_5.discard(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_11) == 1
    var_12 = var_1.__str__()
    assert var_12 == 'CheckedPVector([])'
    var_13 = var_12.__str__()
    assert var_13 == 'CheckedPVector([])'
    var_14 = var_4.set(var_2, var_7)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_14) == 1
    var_15 = var_14.discard(var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_15) == 1
    var_16 = var_15.serialize(var_6)
    module_0.get_type(var_3)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = None
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = module_0.CheckedPMap()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_5 = var_0.add(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_5) == 1
    var_6 = var_4.serialize(var_5)
    var_7 = module_0.InvariantException(var_3, *var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_7.invariant_errors == (None,)
    assert var_7.missing_fields == ()
    var_8 = var_7.__str__()
    assert var_8 == ', invariant_errors=[None], missing_fields=[]'
    var_9 = var_5.discard(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_9) == 1
    var_10 = var_1.__str__()
    assert var_10 == 'CheckedPVector([])'
    var_11 = var_10.__str__()
    assert var_11 == 'CheckedPVector([])'
    var_12 = module_0.CheckedPSet()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_12) == 0
    var_13 = var_4.set(var_2, var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_13) == 1
    var_14 = var_13.discard(var_3)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_14) == 1
    var_15 = module_0.CheckedTypeError(var_2, var_2, var_13, var_5)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_15.source_class is None
    assert var_15.expected_types is None
    assert f'{type(var_15.actual_type).__module__}.{type(var_15.actual_type).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_15.actual_type) == 1
    assert f'{type(var_15.actual_value).__module__}.{type(var_15.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_15.actual_value) == 1
    var_16 = var_14.items()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_16) == 1
    var_17 = var_14.serialize(var_6)
    module_0.get_type(var_3)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = module_0.CheckedPMap()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_3 = var_0.add(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3) == 1
    var_4 = var_2.serialize(var_3)
    var_5 = var_0.__repr__()
    assert var_5 == 'CheckedPSet()'
    var_6 = module_0.InvariantException(var_3, *var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert f'{type(var_6.invariant_errors).__module__}.{type(var_6.invariant_errors).__qualname__}' == 'builtins.tuple'
    assert len(var_6.invariant_errors) == 1
    assert var_6.missing_fields == ()
    var_7 = var_6.__str__()
    assert var_7 == ', invariant_errors=[CheckedPVector([])], missing_fields=[]'
    var_8 = var_3.discard(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_8) == 1
    var_9 = var_1.__str__()
    assert var_9 == 'CheckedPVector([])'
    var_10 = var_9.__str__()
    assert var_10 == 'CheckedPVector([])'
    var_11 = var_2.set(var_8, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_11) == 1
    var_12 = var_11.discard(var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_12) == 1
    var_13 = module_0.CheckedTypeError(var_10, var_10, var_11, var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_13.source_class == 'CheckedPVector([])'
    assert var_13.expected_types == 'CheckedPVector([])'
    assert f'{type(var_13.actual_type).__module__}.{type(var_13.actual_type).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_13.actual_type) == 1
    assert f'{type(var_13.actual_value).__module__}.{type(var_13.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_13.actual_value) == 1
    var_14 = var_12.items()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_14) == 1
    var_12.serialize(var_4)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = module_0.InvariantException(missing_fields=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_2.invariant_errors == ()
    assert var_2.missing_fields is None
    var_3 = module_0.optional()
    var_4 = var_0.set(var_1, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 1
    var_4.__new__(var_1, var_4)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = 'test'
    var_1 = False
    var_2 = 'Error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    module_0._invariant_errors(var_0, var_9)

def test_case_40():
    var_0 = '__invariants__'
    var_1 = {var_0: var_0}
    var_2 = ()
    var_3 = '__invariants__'
    var_4 = '__invariants__'
    with pytest.raises(TypeError):
        module_0.store_invariants(var_1, var_2, var_3, var_4)

def test_case_41():
    var_0 = '__invariants__'
    var_1 = True
    var_2 = lambda : var_1
    var_3 = {var_0: var_2}
    var_4 = ()
    var_5 = '__invariants__'
    var_6 = '__invariants__'
    var_7 = module_0.store_invariants(var_3, var_4, var_5, var_6)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'