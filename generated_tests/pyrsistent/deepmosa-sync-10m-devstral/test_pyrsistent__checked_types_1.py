# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._checked_types as module_0
import pyrsistent._pvector as module_1

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

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.wrap_invariant(var_0)
    var_1.itervalues()

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
    var_0 = 'te?t'
    module_0.get_types(var_0)

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
    with pytest.raises(TypeError):
        var_0.set(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = 'te?t'
    var_2 = module_0.optional()
    var_3 = module_0.maybe_parse_many_user_types(var_2)
    var_4 = bool(var_1 == [])
    module_0.maybe_parse_many_user_types(var_4)

def test_case_11():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == 'CheckedPMap({CheckedPSet(): CheckedPSet()})'

def test_case_12():
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
def test_case_13():
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

def test_case_15():
    var_0 = 'f'
    var_1 = module_0.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.__add__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 1

def test_case_16():
    var_0 = module_0.optional()
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.maybe_parse_many_user_types(var_0)

@pytest.mark.xfail(strict=True)
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
    var_2 = None
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = var_0.add(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4) == 1
    var_4.serialize()

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPSet()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_2 = var_0.serialize()
    module_0.get_type(var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.optional()
    var_2 = module_0.maybe_parse_many_user_types(var_1)
    var_3 = var_0.__reduce__()
    var_1.serialize()

def test_case_20():
    var_0 = module_0.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPVector(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.__repr__()
    assert var_2 == 'CheckedPVector([])'
    var_3 = module_0.optional()
    var_4 = module_0.maybe_parse_many_user_types(var_3)
    var_5 = bool(var_3 == [])

@pytest.mark.xfail(strict=True)
def test_case_21():
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
    var_3 = var_1.mset()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    var_4 = var_1.append(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 1
    var_5 = module_0.CheckedPMap()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_5) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_6 = var_0.__repr__()
    assert var_6 == 'CheckedPSet()'
    var_7 = None
    var_8 = module_0.InvariantException(var_4, *var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_8.invariant_errors == (None,)
    assert var_8.missing_fields == ()
    var_9 = var_7.__str__()
    assert var_9 == 'None'
    var_10 = var_8.__str__()
    assert var_10 == ', invariant_errors=[None], missing_fields=[]'
    module_0.get_type(var_7)

@pytest.mark.xfail(strict=True)
def test_case_22():
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
def test_case_23():
    var_0 = module_0.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPSet()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_2 = module_0.CheckedPVector()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 0
    var_3 = module_0.optional()
    var_4 = module_0.maybe_parse_many_user_types(var_3)
    var_5 = module_0.get_types(var_3)
    var_6 = var_2.extend(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_6) == 1
    module_0.get_type(var_4)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_0.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedPSet.create).__module__}.{type(module_0.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_2 = var_0.__reduce__()
    var_1.set(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = []
    var_1 = module_0.CheckedPMap(*var_0)
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
    var_3 = module_0.CheckedPVector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_4 = module_0.optional()
    var_5 = module_0.maybe_parse_many_user_types(var_4)
    var_6 = var_5.__repr__()
    assert var_6 == "(<class 'NoneType'>,)"
    var_7 = module_0.CheckedKeyTypeError(var_4, var_6, var_5, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedKeyTypeError'
    assert f'{type(var_7.source_class).__module__}.{type(var_7.source_class).__qualname__}' == 'builtins.tuple'
    assert len(var_7.source_class) == 1
    assert var_7.expected_types == "(<class 'NoneType'>,)"
    assert f'{type(var_7.actual_type).__module__}.{type(var_7.actual_type).__qualname__}' == 'builtins.tuple'
    assert len(var_7.actual_type) == 1
    assert f'{type(var_7.actual_value).__module__}.{type(var_7.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_7.actual_value) == 0
    var_8 = var_3.extend(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_8) == 1
    var_9 = var_1.__reduce__()
    module_0.get_type(var_5)

def test_case_26():
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

def test_case_27():
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
def test_case_28():
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
    var_2 = var_1.evolver()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet.Evolver'
    assert len(var_2) == 0
    var_3 = var_0.serialize(var_2)
    var_4 = module_0.wrap_invariant(var_2)
    var_5 = module_0.InvariantException(*var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_5.invariant_errors == ()
    assert var_5.missing_fields == ()
    var_6 = var_0.__hash__()
    assert var_6 == 133146708735736
    var_7 = var_2.__repr__()
    var_8 = var_5.__str__()
    assert var_8 == ', invariant_errors=[], missing_fields=[]'
    var_1.set(var_2, var_2)

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

def test_case_31():
    var_0 = 'test'
    var_1 = module_0.maybe_parse_many_user_types(var_0)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = []
    var_3 = module_0._invariant_errors(var_0, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True
    var_5 = var_1.remove(var_0)

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
    var_2 = var_1.__reduce__()
    var_3 = var_0.__repr__()
    assert var_3 == 'CheckedPSet()'
    module_0.InvariantException(var_2, *var_1)

def test_case_33():
    var_0 = module_0.CheckedType()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    with pytest.raises(NotImplementedError):
        var_0.serialize()

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = []
    var_1 = module_0.CheckedPMap(*var_0)
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
    var_3 = module_0.CheckedPVector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPVector.create).__module__}.{type(module_0.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_4 = module_0.optional()
    var_5 = module_0.maybe_parse_many_user_types(var_4)
    var_6 = var_3.__reduce__()
    var_7 = module_0.get_types(var_4)
    var_8 = var_5.__str__()
    assert var_8 == "(<class 'NoneType'>,)"
    var_9 = var_3.extend(var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_9) == 1
    var_2.__new__(var_6, var_6)

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
    var_3 = module_1.python_pvector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_1.BRANCH_FACTOR == 32
    assert module_1.BIT_MASK == 31
    assert module_1.SHIFT == 5
    var_4 = var_1.append(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 1
    var_5 = module_0.CheckedPMap()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_5) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_6 = var_5.update()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_6) == 0
    var_7 = module_0.InvariantException(var_4, *var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_7.invariant_errors == (None,)
    assert var_7.missing_fields == ()
    var_8 = var_5.set(var_2, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_8) == 1
    var_9 = var_8.discard(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_9) == 1
    var_10 = module_0.CheckedTypeError(var_2, var_2, var_8, var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_10.source_class is None
    assert var_10.expected_types is None
    assert f'{type(var_10.actual_type).__module__}.{type(var_10.actual_type).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_10.actual_type) == 1
    assert f'{type(var_10.actual_value).__module__}.{type(var_10.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_10.actual_value) == 0
    module_0.get_type(var_9)

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
    var_3 = module_0.CheckedPMap()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_4 = var_0.add(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4) == 1
    var_5 = var_3.serialize(var_4)
    var_6 = var_0.__repr__()
    assert var_6 == 'CheckedPSet()'
    var_7 = module_0.InvariantException(var_3, *var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_7.invariant_errors == ()
    assert var_7.missing_fields == ()
    var_8 = var_3.__str__()
    assert var_8 == 'CheckedPMap({})'
    var_9 = var_3.set(var_2, var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_9) == 1
    var_10 = var_9.discard(var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_10) == 1
    var_11 = module_0.CheckedTypeError(var_2, var_2, var_9, var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_11.source_class is None
    assert var_11.expected_types is None
    assert f'{type(var_11.actual_type).__module__}.{type(var_11.actual_type).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_11.actual_type) == 1
    assert f'{type(var_11.actual_value).__module__}.{type(var_11.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_11.actual_value) == 1
    var_12 = var_10.serialize(var_5)
    module_0.get_type(var_6)

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
    var_2 = None
    var_3 = module_0.CheckedPMap()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_4 = var_0.add(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4) == 1
    var_5 = var_3.serialize(var_4)
    var_6 = module_0.InvariantException(var_3, *var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_6.invariant_errors == ()
    assert var_6.missing_fields == ()
    var_7 = var_6.__str__()
    assert var_7 == ', invariant_errors=[], missing_fields=[]'
    var_8 = var_3.set(var_2, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_8) == 1
    var_9 = var_8.discard(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_9) == 1
    var_10 = module_0.CheckedTypeError(var_2, var_2, var_8, var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_10.source_class is None
    assert var_10.expected_types is None
    assert f'{type(var_10.actual_type).__module__}.{type(var_10.actual_type).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_10.actual_type) == 1
    assert f'{type(var_10.actual_value).__module__}.{type(var_10.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_10.actual_value) == 1
    var_11 = var_9.serialize(var_5)
    module_0.get_type(var_0)

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
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 1
    var_3 = module_0.CheckedPMap()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    assert f'{type(module_0.CheckedPMap.create).__module__}.{type(module_0.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_4 = var_0.add(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_4) == 1
    var_5 = var_3.serialize(var_4)
    var_6 = module_0.InvariantException(var_2, *var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert f'{type(var_6.invariant_errors).__module__}.{type(var_6.invariant_errors).__qualname__}' == 'builtins.tuple'
    assert len(var_6.invariant_errors) == 1
    assert var_6.missing_fields == ()
    var_7 = var_6.__str__()
    assert var_7 == ', invariant_errors=[CheckedPVector([])], missing_fields=[]'
    var_8 = var_3.__str__()
    assert var_8 == 'CheckedPMap({})'
    var_9 = var_3.set(var_4, var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_9) == 1
    var_10 = var_9.discard(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_10) == 1
    var_11 = module_0.CheckedTypeError(var_8, var_8, var_9, var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedTypeError'
    assert var_11.source_class == 'CheckedPMap({})'
    assert var_11.expected_types == 'CheckedPMap({})'
    assert f'{type(var_11.actual_type).__module__}.{type(var_11.actual_type).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_11.actual_type) == 1
    assert f'{type(var_11.actual_value).__module__}.{type(var_11.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_11.actual_value) == 1
    var_10.serialize(var_5)

def test_case_39():
    var_0 = 'invariant'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = 'invariant'
    with pytest.raises(TypeError):
        module_0.store_invariants(var_2, var_3, var_4, var_5)

def test_case_40():
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
    var_2 = var_1.__lt__(var_1)
    assert var_2 is False
    var_3 = module_0.optional()
    var_4 = module_0.maybe_parse_many_user_types(var_3)
    var_5 = var_1.extend(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_5) == 1
    var_6 = var_4.__str__()
    assert var_6 == "(<class 'NoneType'>,)"
    var_7 = module_0.CheckedType()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_0.CheckedType.create).__module__}.{type(module_0.CheckedType.create).__qualname__}' == 'builtins.method'
    with pytest.raises(TypeError):
        module_0.store_invariants(var_4, var_5, var_6, var_2)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 5
    var_1 = False
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'error2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    module_0._invariant_errors(var_0, var_8)

def test_case_42():
    var_0 = 'source_name'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = 'source_name'
    var_6 = module_0._store_types(var_2, var_3, var_4, var_5)
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_7 = bool(var_2[var_4])
    assert var_7 is True