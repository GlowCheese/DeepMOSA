# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._checked_types as module_0
import builtins as module_1
import pyrsistent._pvector as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0._CheckedTypeMeta()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_1.dict
    var_2 = module_0.get_type(var_1)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2.__subclasscheck__(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.get_type(var_0)

def test_case_3():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'

def test_case_4():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'

def test_case_5():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'

def test_case_6():
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

def test_case_7():
    var_0 = None
    var_1 = module_0.wrap_invariant(var_0)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.optional()
    var_3 = module_0.CheckedPSet()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'

def test_case_8():
    var_0 = None
    var_1 = module_0.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_2 = module_0.CheckedPSet()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_3 = module_0.CheckedTypeError(var_0, var_0, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_3.source_class is None
    assert var_3.expected_types is None
    assert var_3.actual_type is None
    assert var_3.actual_value is None

def test_case_9():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.get_types(var_0)

def test_case_10():
    var_0 = module_0.optional()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_11():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = var_0.__reduce__()
    var_3 = module_0.CheckedPSet()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_4 = var_3.serialize()
    var_5 = var_2.__str__()
    assert var_5 == "(<function _restore_pickle at 0x726be3d1d6c0>, (<class 'pyrsistent._checked_types.CheckedPMap'>, {}))"
    var_6 = module_0.optional()
    var_7 = module_0.optional()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_0.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    module_0.maybe_parse_many_user_types(var_0)

def test_case_13():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()
    var_2 = module_0.CheckedPMap()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.__repr__()
    assert var_2 == 'CheckedPVector([])'
    var_0.set(var_2, var_1)

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

def test_case_16():
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
    var_2 = var_0.serialize()

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_0.set(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'CheckedPSet()'
    var_1.__new__(var_1, var_1, var_1, var_1, var_1)

def test_case_19():
    var_0 = module_0.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_20():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__str__()
    assert var_1 == 'CheckedPMap({})'

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_0.extend(var_1)

def test_case_22():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 1
    var_3 = var_2.serialize()

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_0.set(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 1
    var_4 = module_0.CheckedPSet()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_5 = var_4.serialize(var_2)
    var_6 = module_0.optional()
    var_7 = var_1.serialize(var_5)
    var_8 = module_0.maybe_parse_many_user_types(var_6)
    var_9 = module_1.BaseException(*var_8)
    var_10 = module_0.CheckedType()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    var_11 = module_0.CheckedPVector()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_11) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_12 = var_4.add(var_2)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_12) == 1
    var_13 = module_0.InvariantException()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_13.invariant_errors == ()
    assert var_13.missing_fields == ()
    var_14 = var_13.__str__()
    assert var_14 == ', invariant_errors=[], missing_fields=[]'
    var_15 = var_11.__repr__()
    assert var_15 == 'CheckedPVector([])'
    var_16 = [var_3, var_2, var_15, var_5]
    var_8.update_with(var_14, *var_16)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.set(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1
    var_2.serialize()

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPSet()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_2 = module_0.optional()
    var_3 = var_0.update()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    var_4 = module_0.maybe_parse_many_user_types(var_0)
    var_5 = module_0.maybe_parse_user_type(var_2)
    var_6 = module_0.CheckedPVector()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_6) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_7 = var_1.__repr__()
    assert var_7 == 'CheckedPSet()'
    var_8 = var_1.add(var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_8) == 1
    var_9 = module_0.InvariantException()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_9.invariant_errors == ()
    assert var_9.missing_fields == ()
    var_10 = var_9.__str__()
    assert var_10 == ', invariant_errors=[], missing_fields=[]'
    var_11 = var_4.__repr__()
    assert var_11 == '()'
    var_12 = var_0.itervalues()
    module_0._CheckedTypeMeta()

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.iteritems()
    var_2 = var_0.set(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1
    var_3 = module_0.CheckedPVector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_4 = var_2.serialize()
    var_5 = var_0.transform()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_5) == 0
    var_6 = var_0.serialize()
    var_7 = var_3.__ne__(var_4)
    assert var_7 is True
    var_8 = module_2.python_pvector(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_8) == 1
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_2.BRANCH_FACTOR == 32
    assert module_2.BIT_MASK == 31
    assert module_2.SHIFT == 5
    var_9 = var_2.itervalues()
    module_0._CheckedTypeMeta()

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.evolver()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap.Evolver'
    assert len(var_2) == 0
    var_3 = var_0.set(var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 1
    var_4 = module_0.CheckedPVector()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_3.__new__(var_3, var_3)

def test_case_28():
    var_0 = module_0.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__str__()
    assert var_1 == ', invariant_errors=[], missing_fields=[]'

def test_case_29():
    var_0 = module_0.CheckedType()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    with pytest.raises(NotImplementedError):
        var_0.serialize()

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.iteritems()
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1
    var_3 = module_0.CheckedPVector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2.serialize()

def test_case_31():
    var_0 = None
    var_1 = var_0.__repr__()
    assert var_1 == 'None'
    var_2 = module_0.InvariantException(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_2.invariant_errors == ('N',)
    assert var_2.missing_fields == 'o'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_32():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.mset()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_0.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = var_3.serialize()
    var_5 = var_3.__str__()
    assert var_5 == 'CheckedPVector([None])'
    var_6 = var_3.serialize()
    var_7 = module_0.InvariantException(var_6, *var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_7.invariant_errors == (None,)
    assert var_7.missing_fields == ()
    var_8 = module_0.InvariantException(missing_fields=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_8.invariant_errors == ()
    assert var_8.missing_fields == 'CheckedPVector([None])'
    var_9 = module_0.optional()

def test_case_33():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPSet()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_2 = module_0.optional()
    var_3 = module_0.maybe_parse_many_user_types(var_2)
    var_4 = module_0.CheckedPVector()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_5 = var_1.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_5) == 1
    var_6 = module_0.InvariantException()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_6.invariant_errors == ()
    assert var_6.missing_fields == ()
    var_7 = var_6.__str__()
    assert var_7 == ', invariant_errors=[], missing_fields=[]'
    with pytest.raises(TypeError):
        module_0.store_invariants(var_7, var_2, var_3, var_7)

def test_case_34():
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
    var_2 = var_1.serialize()
    var_3 = var_1.serialize()
    var_4 = module_0.InvariantException(var_3, *var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_4.invariant_errors == ([],)
    assert var_4.missing_fields == ()

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()
    var_1.__new__(var_1, var_1, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_0.set(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 1
    var_4 = module_0.CheckedPSet()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_5 = var_4.serialize(var_2)
    var_6 = module_0.optional()
    var_7 = module_0.maybe_parse_many_user_types(var_6)
    var_8 = module_1.BaseException(*var_7)
    var_9 = module_0.CheckedType()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    var_10 = module_0.CheckedPVector()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_10) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_11 = var_4.add(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_11) == 1
    module_0.get_types(var_3)

def test_case_37():
    var_0 = module_0.optional()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.maybe_parse_user_type(var_0)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()
    var_1.__new__(var_0, var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.update(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_1) == 0
    var_2 = var_1.__gt__(var_1)
    assert var_2 is False
    var_1.__new__(var_1, var_1, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'CheckedPMap({})'
    var_2 = None
    var_3 = var_0.set(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 1
    var_4 = module_0.CheckedPSet()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_5 = var_4.serialize(var_2)
    var_6 = module_0.optional()
    var_7 = module_0.maybe_parse_many_user_types(var_6)
    var_8 = var_5.__repr__()
    assert var_8 == 'set()'
    var_9 = module_0.CheckedType()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    var_10 = module_0.CheckedPVector()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_10) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_11 = var_4.add(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_11) == 1
    var_12 = var_10.__repr__()
    assert var_12 == 'CheckedPVector([])'
    var_13 = [var_3, var_2, var_12, var_5]
    var_7.update_with(var_3, *var_13)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPSet()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_2 = module_0.optional()
    var_3 = module_0.wrap_invariant(var_1)
    var_4 = module_0.maybe_parse_user_type(var_2)
    var_5 = module_0.CheckedPVector()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_5) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_6 = var_1.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == "CheckedPSet([(<class 'NoneType'>,)])"
    var_8 = var_0.evolver()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap.Evolver'
    assert len(var_8) == 0
    var_9 = var_3.__repr__()
    module_0._CheckedTypeMeta()

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.__iter__()
    var_3 = var_2.__str__()
    var_4 = module_0.CheckedPVector()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_5 = var_4.append(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_5) == 1
    var_6 = module_0.maybe_parse_user_type(var_3)
    var_7 = var_6.__str__()
    var_8 = var_3.__repr__()
    var_9 = var_6.__str__()
    module_2.python_pvector(var_1)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = module_0.CheckedPSet()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_3 = var_0.set(var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 1
    var_2.__new__(var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = None
    var_1 = module_0.CheckedPSet()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 1
    var_3 = var_1.__repr__()
    assert var_3 == 'CheckedPSet()'
    var_4 = module_0.wrap_invariant(var_0)
    var_2.serialize(var_2)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.serialize()
    var_3 = module_0.CheckedPSet()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_4 = var_3.serialize(var_1)
    var_5 = var_4.__str__()
    assert var_5 == 'set()'
    var_6 = var_0.set(var_0, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_6) == 1
    var_7 = module_0.maybe_parse_many_user_types(var_6)
    var_8 = module_0.CheckedType()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    var_9 = module_0.CheckedPVector()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_9) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_10 = var_3.add(var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_10) == 1
    var_11 = module_0.InvariantException()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_11.invariant_errors == ()
    assert var_11.missing_fields == ()
    var_12 = var_5.__str__()
    assert var_12 == 'set()'
    var_13 = var_11.__str__()
    assert var_13 == ', invariant_errors=[], missing_fields=[]'
    var_14 = module_0.InvariantException(var_6)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert f'{type(var_14.invariant_errors).__module__}.{type(var_14.invariant_errors).__qualname__}' == 'builtins.tuple'
    assert len(var_14.invariant_errors) == 1
    assert var_14.missing_fields == ()
    var_15 = var_7.__repr__()
    assert var_15 == '()'
    var_16 = var_14.__str__()
    assert var_16 == ', invariant_errors=[CheckedPMap({})], missing_fields=[]'
    module_2.python_pvector(var_1)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = module_0.CheckedPSet()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_3 = var_2.serialize(var_1)
    var_4 = var_3.__str__()
    assert var_4 == 'set()'
    var_5 = var_3.__reduce__()
    var_6 = var_5.__contains__(var_5)
    assert var_6 is False
    var_7 = module_0.CheckedType()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    var_8 = module_0.CheckedPVector()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_8) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_9 = var_0.serialize()
    var_10 = var_2.add(var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_10) == 1
    var_11 = module_0.InvariantException()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_11.invariant_errors == ()
    assert var_11.missing_fields == ()
    var_12 = var_4.__str__()
    assert var_12 == 'set()'
    var_13 = var_11.__str__()
    assert var_13 == ', invariant_errors=[], missing_fields=[]'
    var_14 = module_0.InvariantException(var_5)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_14.invariant_errors == ({*()}, ([],), None)
    assert var_14.missing_fields == ()
    var_15 = var_6.__repr__()
    assert var_15 == 'False'
    var_6.set(var_5, var_3)

def test_case_47():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.set(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1
    var_3 = module_0.CheckedPSet()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_4 = var_3.serialize(var_1)
    var_5 = module_0.optional()
    var_6 = module_0.maybe_parse_many_user_types(var_4)
    var_7 = module_0.maybe_parse_many_user_types(var_5)
    var_8 = var_4.__repr__()
    assert var_8 == 'set()'
    var_9 = module_0.CheckedType()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    var_10 = module_0.CheckedPVector()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_10) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_11 = var_3.add(var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_11) == 1
    with pytest.raises(TypeError):
        module_0.store_invariants(var_2, var_7, var_11, var_1)