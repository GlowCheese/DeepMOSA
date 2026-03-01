# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._checked_types as module_0
import pyrsistent._pvector as module_1
import typing as module_2

def test_case_0():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'

def test_case_1():
    var_0 = None
    with pytest.raises(TypeError):
        module_0.maybe_parse_user_type(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.get_type(var_0)

def test_case_3():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'

def test_case_4():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.InvariantException(var_0, var_0, *var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_1.invariant_errors == ()
    assert f'{type(var_1.missing_fields).__module__}.{type(var_1.missing_fields).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1.missing_fields) == 0
    var_2 = None
    var_3 = var_0.__iter__()
    var_3.__new__(var_2, var_3, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.wrap_invariant(var_0)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1.serialize()

@pytest.mark.xfail(strict=True)
def test_case_7():
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
def test_case_8():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()
    module_0.get_types(var_1)

def test_case_9():
    var_0 = module_0.optional()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_10():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.evolver()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector.Evolver'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.get_types(var_0)
    var_2 = module_0.CheckedPVector()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_3 = var_0.__iter__()
    var_4 = module_0.InvariantException(var_0, var_0, *var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_4.invariant_errors == ()
    assert f'{type(var_4.missing_fields).__module__}.{type(var_4.missing_fields).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4.missing_fields) == 0
    var_5 = var_3.__repr__()
    var_6 = var_0.serialize()
    var_7 = var_2.append(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_7) == 1
    var_8 = var_7.serialize()
    module_0._CheckedTypeMeta(**var_6)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.extend(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_0.evolver()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector.Evolver'
    assert len(var_3) == 0
    var_4 = module_0.CheckedKeyTypeError(var_1, var_2, var_2, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedKeyTypeError'
    assert f'{type(var_4.source_class).__module__}.{type(var_4.source_class).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4.source_class) == 0
    assert var_4.expected_types is None
    assert var_4.actual_type is None
    assert f'{type(var_4.actual_value).__module__}.{type(var_4.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4.actual_value) == 0
    var_5 = None
    module_0.get_type(var_5)

def test_case_13():
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

def test_case_14():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'CheckedPVector([])'

@pytest.mark.xfail(strict=True)
def test_case_15():
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
    var_2 = var_0.__iter__()
    var_3 = module_0.CheckedPMap(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_4 = module_0.InvariantException(var_0, var_0, *var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_4.invariant_errors == ()
    assert f'{type(var_4.missing_fields).__module__}.{type(var_4.missing_fields).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4.missing_fields) == 0
    var_5 = var_4.__str__()
    assert var_5 == ', invariant_errors=[], missing_fields=[]'
    var_6 = var_1.append(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_6) == 1
    var_7 = var_3.set(var_5, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_7) == 1
    var_8 = var_7.serialize()
    var_9 = var_6.__str__()
    assert var_9 == 'CheckedPVector([CheckedPSet()])'
    var_4.serialize()

@pytest.mark.xfail(strict=True)
def test_case_16():
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
    var_2 = var_0.__iter__()
    var_3 = module_0.CheckedPMap(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_4 = var_0.__str__()
    assert var_4 == 'CheckedPSet()'
    var_5 = var_0.serialize()
    var_6 = var_1.append(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_6) == 1
    var_7 = None
    var_8 = var_3.set(var_4, var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_8) == 1
    var_9 = var_5.__ne__(var_7)
    var_10 = var_9.__str__()
    assert var_10 == 'NotImplemented'
    module_0.get_type(var_2)

def test_case_17():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()

def test_case_18():
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

def test_case_19():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = 'custom_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = module_0.maybe_parse_user_type(var_1)
    var_5 = None
    with pytest.raises(TypeError):
        module_0.maybe_parse_user_type(var_5)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_0.set(var_1, var_1)

def test_case_21():
    var_0 = {}
    var_1 = module_0.CheckedPSet(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.evolver()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet.Evolver'
    assert len(var_2) == 0

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_2 = var_0.evolver()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector.Evolver'
    assert len(var_2) == 0
    var_3 = var_1.update(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    var_4 = var_3.set(var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 1
    var_5 = module_0.maybe_parse_many_user_types(var_3)
    var_5.append(var_3)

def test_case_23():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize(var_0)
    var_2 = module_0.CheckedType()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    var_3 = var_0.evolver()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap.Evolver'
    assert len(var_3) == 0

@pytest.mark.xfail(strict=True)
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
    var_2 = module_0.CheckedPMap(*var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_3 = var_2.set(var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 1
    var_3.serialize()

@pytest.mark.xfail(strict=True)
def test_case_25():
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
    var_2 = var_0.__iter__()
    var_3 = module_0.CheckedPMap(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_4 = module_0.InvariantException(var_0, var_0, *var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_4.invariant_errors == ()
    assert f'{type(var_4.missing_fields).__module__}.{type(var_4.missing_fields).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4.missing_fields) == 0
    var_5 = var_4.__str__()
    assert var_5 == ', invariant_errors=[], missing_fields=[]'
    var_6 = var_1.append(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_6) == 1
    var_7 = var_3.set(var_5, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_7) == 1
    var_8 = var_7.serialize()
    var_9 = var_7.__str__()
    assert var_9 == "CheckedPMap({', invariant_errors=[], missing_fields=[]': ', invariant_errors=[], missing_fields=[]'})"
    module_0.get_type(var_9)

@pytest.mark.xfail(strict=True)
def test_case_26():
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
    var_2 = var_0.__repr__()
    assert var_2 == 'CheckedPSet()'
    var_3 = module_0.InvariantException(var_2, var_2, *var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_3.invariant_errors == ('C', 'h', 'e', 'c', 'k', 'e', 'd', 'P', 'S', 'e', 't', '(', ')')
    assert var_3.missing_fields == 'CheckedPSet()'
    var_4 = None
    var_5 = var_2.__lt__(var_4)
    module_0._CheckedTypeMeta(*var_0)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.__reduce__()
    var_2.__hash__()

def test_case_28():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()
    var_2 = var_1.__str__()
    assert var_2 == 'set()'

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
    var_2 = var_0.__reduce__()
    module_0.InvariantException(var_2, var_2, *var_2)

@pytest.mark.xfail(strict=True)
def test_case_30():
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
    var_2 = module_0.CheckedPMap(*var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_3 = var_1.append(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = var_2.set(var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 1
    var_4.serialize()

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = module_0.wrap_invariant(var_0)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__repr__()
    var_3 = module_0.CheckedPSet()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_3.__new__(var_3, var_2)

@pytest.mark.xfail(strict=True)
def test_case_32():
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
    var_2 = var_0.__iter__()
    var_3 = module_0.CheckedPMap(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_4 = module_0.InvariantException(var_0, var_0, *var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_4.invariant_errors == ()
    assert f'{type(var_4.missing_fields).__module__}.{type(var_4.missing_fields).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4.missing_fields) == 0
    var_5 = var_4.__str__()
    assert var_5 == ', invariant_errors=[], missing_fields=[]'
    var_6 = var_0.add(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_6) == 1
    var_6.serialize(var_5)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = module_0.wrap_invariant(var_0)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__repr__()
    var_3 = module_1.python_pvector(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 54
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_1.BRANCH_FACTOR == 32
    assert module_1.BIT_MASK == 31
    assert module_1.SHIFT == 5
    var_4 = module_0.CheckedPVector()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_5 = var_3.__iter__()
    var_6 = module_0.InvariantException(var_3, var_3, *var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert f'{type(var_6.missing_fields).__module__}.{type(var_6.missing_fields).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_6.missing_fields) == 54
    var_7 = var_6.__str__()
    var_3.serialize()

def test_case_34():
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
    var_2 = var_1.__len__()
    assert var_2 == 0
    var_3 = module_0.InvariantException(var_0, var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_3.invariant_errors == ()
    assert f'{type(var_3.missing_fields).__module__}.{type(var_3.missing_fields).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3.missing_fields) == 0
    var_4 = var_3.__str__()
    assert var_4 == ', invariant_errors=[], missing_fields=[]'
    var_5 = var_0.serialize()
    var_6 = var_1.append(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_6) == 1
    with pytest.raises(TypeError):
        module_0.store_invariants(var_0, var_5, var_6, var_6)

@pytest.mark.xfail(strict=True)
def test_case_35():
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
    var_2 = var_0.__iter__()
    var_3 = module_0.InvariantException(var_0, var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_3.invariant_errors == ()
    assert f'{type(var_3.missing_fields).__module__}.{type(var_3.missing_fields).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3.missing_fields) == 0
    var_4 = var_3.__str__()
    assert var_4 == ', invariant_errors=[], missing_fields=[]'
    var_5 = var_0.serialize()
    var_1.append(var_0)

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
    var_2 = var_0.__iter__()
    var_3 = module_0.InvariantException(var_0, var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_3.invariant_errors == ()
    assert f'{type(var_3.missing_fields).__module__}.{type(var_3.missing_fields).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_3.missing_fields) == 0
    var_4 = var_3.__str__()
    assert var_4 == ', invariant_errors=[], missing_fields=[]'
    var_5 = module_0.CheckedType()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    var_5.serialize()

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
    var_2 = module_0.CheckedPMap(*var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_3 = var_1.__repr__()
    assert var_3 == 'CheckedPVector([])'
    var_4 = var_2.set(var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 1
    var_5 = var_4.serialize()
    var_6 = var_2.serialize(var_4)

@pytest.mark.xfail(strict=True)
def test_case_38():
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
    var_2 = var_0.__iter__()
    var_3 = var_0.evolver()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet.Evolver'
    assert len(var_3) == 0
    var_4 = module_0.CheckedPMap(*var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_5 = module_0.InvariantException(var_0, var_0, *var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_5.invariant_errors == ()
    assert f'{type(var_5.missing_fields).__module__}.{type(var_5.missing_fields).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_5.missing_fields) == 0
    var_6 = var_5.__str__()
    assert var_6 == ', invariant_errors=[], missing_fields=[]'
    var_7 = None
    var_8 = var_1.append(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_8) == 1
    var_9 = var_8.serialize()
    var_10 = var_0.__repr__()
    assert var_10 == 'CheckedPSet()'
    var_2.set(var_8, var_7)

def test_case_39():
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
    var_2 = var_1.serialize()
    var_3 = var_0.__iter__()
    var_4 = module_0.CheckedPMap(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_5 = module_0.InvariantException(var_0, var_0, *var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_5.invariant_errors == ()
    assert f'{type(var_5.missing_fields).__module__}.{type(var_5.missing_fields).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_5.missing_fields) == 0
    var_6 = None
    var_7 = var_1.__eq__(var_6)
    assert var_7 is False
    var_8 = var_5.__str__()
    assert var_8 == ', invariant_errors=[], missing_fields=[]'
    var_9 = var_5.__str__()
    assert var_9 == ', invariant_errors=[], missing_fields=[]'
    var_10 = var_0.serialize()
    var_11 = var_1.append(var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_11) == 1
    var_12 = var_3.__repr__()
    var_13 = var_4.set(var_9, var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_13) == 1
    var_14 = var_0.serialize()
    var_15 = var_0.serialize()
    var_16 = module_2._Final
    var_17 = module_0.get_type(var_16)
    assert module_2.EXCLUDED_ATTRIBUTES == ['__parameters__', '__orig_bases__', '__orig_class__', '_is_protocol', '_is_runtime_protocol', '__abstractmethods__', '__annotations__', '__dict__', '__doc__', '__init__', '__module__', '__new__', '__slots__', '__subclasshook__', '__weakref__', '__class_getitem__', '_MutableMapping__marker']
    assert f'{type(module_2.T).__module__}.{type(module_2.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT).__module__}.{type(module_2.VT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.V_co).__module__}.{type(module_2.V_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.T_contra).__module__}.{type(module_2.T_contra).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.CT_co).__module__}.{type(module_2.CT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.AnyStr).__module__}.{type(module_2.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_2.TYPE_CHECKING is False

def test_case_40():
    var_0 = set()
    var_1 = module_0.optional(*var_0)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.maybe_parse_user_type(var_1)
    var_3 = None
    with pytest.raises(TypeError):
        module_0.maybe_parse_user_type(var_3)

def test_case_41():
    var_0 = 0
    var_1 = 'not a callable'
    var_2 = {var_1: var_1}
    var_3 = ()
    var_4 = '_invariants'
    var_5 = '_invariant'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_7 = True
    var_8 = 'dct_inv'
    var_9 = (var_7, var_8)
    var_10 = lambda self: var_9
    var_11 = {var_5: var_10}
    var_12 = module_0.store_invariants(var_11, var_3, var_4, var_5)
    var_13 = var_11[var_4]
    var_14 = len(var_13)
    var_15 = ()
    var_16 = module_0.store_invariants(var_11, var_15, var_4, var_5)
    var_17 = var_11[var_4][var_0]
    var_18 = ()
    var_19 = module_0.store_invariants(var_11, var_18, var_4, var_5)
    var_20 = var_11[var_4][var_0]
    var_21 = {}
    var_22 = ()
    var_23 = module_0.store_invariants(var_21, var_22, var_4, var_5)