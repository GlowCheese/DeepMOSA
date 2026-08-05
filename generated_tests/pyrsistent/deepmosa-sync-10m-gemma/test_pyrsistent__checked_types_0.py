# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import builtins as module_0
import pyrsistent._checked_types as module_1
import pyrsistent._pvector as module_2
import numbers as module_3

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.dict
    var_2 = module_1.get_type(var_1)
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2.__subclasscheck__(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_1.get_type(var_0)

def test_case_3():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'

def test_case_4():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'

def test_case_5():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'

def test_case_6():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.set(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1

def test_case_7():
    var_0 = None
    var_1 = module_1.wrap_invariant(var_0)
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_1.optional()
    var_3 = module_1.CheckedPSet()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3) == 0
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'

def test_case_8():
    var_0 = None
    var_1 = module_1.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_2 = module_1.CheckedPSet()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 0
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_3 = module_1.CheckedTypeError(var_0, var_0, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_3.source_class is None
    assert var_3.expected_types is None
    assert var_3.actual_type is None
    assert var_3.actual_value is None

def test_case_9():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = module_1.get_types(var_0)

def test_case_10():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_11():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = module_1.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = var_0.__reduce__()
    var_3 = module_1.CheckedPSet()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3) == 0
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_4 = var_3.serialize()
    var_5 = var_2.__str__()
    assert var_5 == "(<function _restore_pickle at 0x79ce6750cf70>, (<class 'pyrsistent._checked_types.CheckedPMap'>, {}))"
    var_6 = module_1.optional()
    var_7 = module_1.optional()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_1.maybe_parse_many_user_types(var_0)

def test_case_13():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()
    var_2 = module_1.CheckedPMap()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'

def test_case_14():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__ne__(var_0)
    assert var_1 is False
    var_2 = None
    var_3 = var_0.__repr__()
    assert var_3 == 'CheckedPVector([])'
    var_4 = module_1.CheckedKeyTypeError(var_2, var_3, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedKeyTypeError'
    assert var_4.source_class is None
    assert var_4.expected_types == 'CheckedPVector([])'
    assert var_4.actual_type is None
    assert var_4.actual_value is None

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_0.set(var_1, var_1)

def test_case_16():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__ne__(var_0)
    assert var_1 is False
    var_2 = var_0.__repr__()
    assert var_2 == 'CheckedPVector([])'
    var_3 = var_0.serialize()
    var_4 = var_0.serialize()

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 1
    var_0.__new__(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'CheckedPSet()'
    var_1.__new__(var_1, var_1, var_1, var_1, var_1)

def test_case_19():
    var_0 = module_1.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_20():
    var_0 = None
    var_1 = module_1.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_2 = module_1.CheckedPSet()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 0
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_3 = var_1.__str__()
    assert var_3 == 'CheckedPMap({})'
    var_4 = module_1.CheckedTypeError(var_0, var_0, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_4.source_class is None
    assert var_4.expected_types is None
    assert var_4.actual_type is None
    assert var_4.actual_value is None

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_0.extend(var_1)

def test_case_22():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 1
    var_3 = var_2.__ne__(var_2)
    assert var_3 is False
    var_4 = var_2.serialize()
    var_5 = module_1.CheckedTypeError(var_3, var_3, var_3, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_5.source_class is False
    assert var_5.expected_types is False
    assert var_5.actual_type is False
    assert var_5.actual_value is None

def test_case_23():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = [var_0, var_0]
    var_2 = var_0.serialize()
    var_3 = module_1.maybe_parse_many_user_types(var_1)
    var_4 = 'float'
    var_5 = module_1.CheckedPVector()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_5) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_6 = module_1.CheckedPMap()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_6) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_7 = None
    var_8 = module_1.CheckedKeyTypeError(var_7, var_4, var_7, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedKeyTypeError'
    assert var_8.source_class is None
    assert var_8.expected_types == 'float'
    assert var_8.actual_type is None
    assert f'{type(var_8.actual_value).__module__}.{type(var_8.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_8.actual_value) == 0
    var_9 = 'jN3]vrR'
    var_10 = var_0.evolver()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet.Evolver'
    assert len(var_10) == 0
    var_11 = var_6.set(var_9, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_11) == 1
    var_12 = var_6.serialize(var_7)
    var_13 = module_1.maybe_parse_many_user_types(var_4)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.set(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1
    var_2.serialize()

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.update_with(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    var_3 = None
    var_4 = var_0.set(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 1
    var_5 = module_1.CheckedPVector()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_5) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_6 = var_4.copy()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_6) == 1
    var_7 = var_6.__reduce__()
    var_8 = var_4.itervalues()
    module_1.store_invariants(var_6, var_6, var_8, var_4)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.iteritems()
    var_2 = var_0.set(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1
    var_3 = module_1.CheckedPVector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_4 = var_2.serialize()
    var_5 = var_0.transform()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_5) == 0
    var_6 = var_0.serialize()
    var_7 = var_2.itervalues()
    module_1._CheckedTypeMeta()

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.evolver()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap.Evolver'
    assert len(var_2) == 0
    var_3 = var_0.set(var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 1
    var_4 = module_1.CheckedPVector()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_3.__new__(var_3, var_3)

def test_case_28():
    var_0 = module_1.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__str__()
    assert var_1 == ', invariant_errors=[], missing_fields=[]'

def test_case_29():
    var_0 = module_1.CheckedType()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedType.create).__module__}.{type(module_1.CheckedType.create).__qualname__}' == 'builtins.method'
    with pytest.raises(NotImplementedError):
        var_0.serialize()

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.iteritems()
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1
    var_3 = module_1.CheckedPVector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2.serialize()

def test_case_31():
    var_0 = 'f9;t'
    var_1 = module_1.InvariantException(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_1.invariant_errors == ('f',)
    assert var_1.missing_fields == '9'
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = module_1.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.__add__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 0
    var_3 = var_1.__ne__(var_0)
    assert var_3 is True
    module_2.python_pvector(var_0)

def test_case_33():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.iteritems()
    with pytest.raises(TypeError):
        module_1.store_invariants(var_1, var_2, var_1, var_2)

def test_case_34():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.__eq__(var_1)
    assert var_2 is False
    var_3 = var_0.append(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = var_3.serialize()
    var_5 = var_3.serialize()
    var_6 = module_1.CheckedTypeError(var_4, var_3, var_5, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_6.source_class == [[]]
    assert f'{type(var_6.expected_types).__module__}.{type(var_6.expected_types).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_6.expected_types) == 1
    assert var_6.actual_type == [[]]
    assert var_6.actual_value is False

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()
    var_1.__new__(var_1, var_1, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.set(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1
    var_3 = module_1.CheckedPVector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_4 = var_3.append(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 1
    var_5 = module_1.wrap_invariant(var_2)
    var_6 = var_4.__reduce__()
    module_1.get_types(var_4)

def test_case_37():
    var_0 = 'float'
    var_1 = (var_0, var_0)
    var_2 = module_1.maybe_parse_user_type(var_1)
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = bool(var_2 == ('float', 'bool'))

def test_case_38():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()

def test_case_39():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'CheckedPSet()'
    var_2 = var_0.update(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 11
    var_3 = 'float'
    var_4 = module_1.CheckedPVector()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_5 = None
    var_6 = module_1.CheckedKeyTypeError(var_5, var_3, var_5, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedKeyTypeError'
    assert var_6.source_class is None
    assert var_6.expected_types == 'float'
    assert var_6.actual_type is None
    assert f'{type(var_6.actual_value).__module__}.{type(var_6.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_6.actual_value) == 0
    var_7 = var_1.__repr__()
    assert var_7 == "'CheckedPSet()'"
    var_8 = bool(var_1 == ('float', 'bool'))

def test_case_40():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__str__()
    assert var_1 == 'CheckedPSet()'
    var_2 = module_1.CheckedPVector()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_3 = module_1.CheckedPMap()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = module_1.CheckedPSet()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 0
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_3 = var_0.set(var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 1
    var_2.__new__(var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = -5
    var_1 = 0
    var_2 = 'must be positive'
    var_3 = lambda x: (x > var_1, var_2)
    var_4 = 'must be negative'
    var_5 = lambda x: (x < var_1, var_4)
    var_6 = [var_3, var_5]
    module_1._invariant_errors(var_0, var_6)

def test_case_43():
    var_0 = 'float'
    var_1 = module_1.InvariantException(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_1.invariant_errors == ('f',)
    assert var_1.missing_fields == 'l'
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__str__()
    assert var_2 == "('o', 'a', 't'), invariant_errors=[f], missing_fields=[l]"
    var_3 = bool(var_2 == ('float', 'bool'))

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = 'dynamic_err'
    var_1 = lambda : var_0
    var_2 = 'static_err'
    var_3 = (var_1, var_2)
    var_4 = ()
    var_5 = {}
    module_1.InvariantException(var_3, var_4, **var_5)

def test_case_45():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.update(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_1) == 0
    var_2 = 'float'
    var_3 = module_1.CheckedPVector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_4 = var_1.__str__()
    assert var_4 == 'CheckedPSet()'
    var_5 = module_1.CheckedPMap()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_5) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_6 = None
    var_7 = module_1.CheckedKeyTypeError(var_6, var_2, var_6, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedKeyTypeError'
    assert var_7.source_class is None
    assert var_7.expected_types == 'float'
    assert var_7.actual_type is None
    assert f'{type(var_7.actual_value).__module__}.{type(var_7.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_7.actual_value) == 0
    var_8 = module_2.python_pvector()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_8) == 0
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_2.BRANCH_FACTOR == 32
    assert module_2.BIT_MASK == 31
    assert module_2.SHIFT == 5
    var_9 = var_8.evolver()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pvector.PythonPVector.Evolver'
    assert len(var_9) == 0
    var_10 = bool(var_8 == ('float', 'bool'))

def test_case_46():
    var_0 = ''
    var_1 = 'dst'
    var_2 = lambda x: x
    var_3 = {var_0: var_0, var_2: var_1, var_2: var_2}
    var_4 = []
    with pytest.raises(TypeError):
        module_1.store_invariants(var_3, var_4, var_1, var_0)

def test_case_47():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = lambda x: x
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = module_1.store_invariants(var_3, var_4, var_1, var_0)
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = bool(var_1 in var_3)
    assert var_6 is True
    var_7 = 0
    var_8 = var_3[var_1][var_7]
    var_9 = callable(var_8)
    var_10 = bool(var_9)
    assert var_10 is True

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = None
    var_3 = var_0.add(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3) == 1
    var_4 = var_3.serialize()
    var_5 = var_4.__contains__(var_1)
    assert var_5 is True
    var_5.evolver()

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = module_3.Number
    var_1 = module_1.maybe_parse_many_user_types(var_0)
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__repr__()
    assert var_2 == "[<class 'numbers.Number'>]"
    var_3 = var_2.__hash__()
    assert var_3 == -8529965509606449155
    var_4 = var_3.__repr__()
    assert var_4 == '-8529965509606449155'
    var_5 = var_4.__len__()
    assert var_5 == 20
    var_6 = var_5.__repr__()
    assert var_6 == '20'
    var_6.items()