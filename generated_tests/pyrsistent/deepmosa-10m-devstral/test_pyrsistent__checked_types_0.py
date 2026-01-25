# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import builtins as module_0
import pyrsistent._checked_types as module_1

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
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
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
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'

def test_case_5():
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

def test_case_6():
    var_0 = module_1.CheckedType()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedType'
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedType.create).__module__}.{type(module_1.CheckedType.create).__qualname__}' == 'builtins.method'
    with pytest.raises(NotImplementedError):
        var_0.serialize()

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_1.wrap_invariant(var_0)
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    module_1.maybe_parse_many_user_types(var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = "}C.-[!jV'"
    var_2 = 'Fo`u,Nt\x0cQ5E\tY{'
    var_3 = {var_1: var_0, var_2: var_0}
    module_1.CheckedKeyTypeError(var_0, var_0, var_0, var_0, **var_3)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_1.get_types(var_0)

def test_case_10():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_1.maybe_parse_user_type(var_0)

def test_case_11():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()
    var_2 = None
    var_3 = var_0.set(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 1

def test_case_12():
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

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_0.set(var_1, var_1)

def test_case_14():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.mset()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    var_2 = module_1.CheckedPSet(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 0
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_3 = var_0.__repr__()
    assert var_3 == 'CheckedPVector([])'
    var_4 = module_1.CheckedPMap()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_5 = module_1.optional()
    var_6 = module_1.maybe_parse_user_type(var_5)

def test_case_15():
    var_0 = module_1.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__str__()
    assert var_1 == ', invariant_errors=[], missing_fields=[]'

def test_case_16():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'

def test_case_17():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__str__()
    assert var_1 == 'CheckedPSet()'

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__str__()
    assert var_1 == 'CheckedPVector([])'
    var_2 = var_1.__str__()
    assert var_2 == 'CheckedPVector([])'
    var_3 = module_1.CheckedPMap()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_4 = var_0.extend(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 18
    var_5 = var_3.__repr__()
    assert var_5 == 'CheckedPMap({})'
    var_6 = var_0.__str__()
    assert var_6 == 'CheckedPVector([])'
    var_6.__reduce__()

def test_case_19():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()

def test_case_20():
    var_0 = 'source_name'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = 'source_name'
    var_6 = module_1._store_types(var_2, var_3, var_4, var_5)
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_7 = bool(var_4 in var_2)
    assert var_7 is True

def test_case_21():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.mset()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    var_2 = var_1.__reduce__()
    var_3 = var_0.__hash__()
    assert var_3 == 5740354900026072187

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = []
    var_1 = module_1.get_types(var_0)
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1.set(var_1, var_1)

def test_case_23():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()

def test_case_24():
    var_0 = module_1.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_1.CheckedPMap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_1) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.__reduce__()
    var_3 = None
    var_4 = var_1.__reduce__()
    var_5 = var_1.set(var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_5) == 1
    var_6 = var_5.serialize()
    var_7 = module_1.CheckedPVector()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_7) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    with pytest.raises(TypeError):
        module_1.maybe_parse_user_type(var_2)

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
    var_2 = var_0.__repr__()
    assert var_2 == 'CheckedPMap({})'
    var_3 = var_0.serialize()
    var_4 = var_0.set(var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_4) == 1
    var_0.__new__(var_4, var_4)

def test_case_26():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.__str__()
    assert var_2 == 'CheckedPSet()'
    var_3 = module_1.wrap_invariant(var_1)
    var_4 = var_0.serialize()
    var_5 = var_0.add(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_5) == 1
    var_6 = var_5.__reduce__()
    var_7 = module_1.CheckedPMap()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_7) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_8 = None
    var_9 = var_7.__reduce__()
    var_10 = var_9.__repr__()
    assert var_10 == "(<function _restore_pickle at 0x7e0bddfc9e10>, (<class 'pyrsistent._checked_types.CheckedPMap'>, {}))"
    var_11 = var_7.set(var_8, var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_11) == 1
    var_12 = var_9.__str__()
    assert var_12 == "(<function _restore_pickle at 0x7e0bddfc9e10>, (<class 'pyrsistent._checked_types.CheckedPMap'>, {}))"

def test_case_27():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_2) == 1

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()
    var_2 = module_1.CheckedPMap()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    module_1.get_types(var_1)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_1.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_1.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = var_0.__str__()
    assert var_2 == ', invariant_errors=[], missing_fields=[]'
    var_3 = module_1.InvariantException(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_3.invariant_errors == (',',)
    assert var_3.missing_fields == ' '
    var_4 = var_1.extend(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 40
    var_1.set(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = module_1.maybe_parse_user_type(var_0)
    var_2 = module_1.CheckedPMap()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_2) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_3 = None
    var_4 = None
    var_5 = module_1.CheckedValueTypeError(var_3, var_4, var_1, var_2, *var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedValueTypeError'
    assert var_5.source_class is None
    assert var_5.expected_types is None
    assert var_5.actual_type == ()
    assert f'{type(var_5.actual_value).__module__}.{type(var_5.actual_value).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_5.actual_value) == 0
    var_6 = var_2.set(var_4, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_6) == 1
    var_7 = var_6.__hash__()
    assert var_7 == -8614310427604454130
    var_8 = var_6.update()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_8) == 1
    var_9 = var_0.serialize(var_6)
    module_1.CheckedTypeError(var_4, var_9, var_9, var_3, *var_8, **var_9)

def test_case_31():
    var_0 = module_1.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_1.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = var_0.__str__()
    assert var_2 == ', invariant_errors=[], missing_fields=[]'
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = module_1.wrap_invariant(var_3)
    var_5 = var_1.mset()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_5) == 0
    var_6 = module_1.CheckedPSet(*var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_6) == 20
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_7 = module_1.CheckedPMap()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_7) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_8 = module_1.optional()
    var_9 = var_8.__ge__(var_1)
    var_10 = var_7.set(var_8, var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_10) == 1
    var_11 = var_3.serialize(var_2)
    var_12 = var_5.__eq__(var_8)
    assert var_12 is False
    var_13 = var_4.__repr__()
    var_14 = var_5.tolist()
    var_15 = module_1.maybe_parse_user_type(var_10)

def test_case_32():
    var_0 = module_1.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_1.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = var_0.__str__()
    assert var_2 == ', invariant_errors=[], missing_fields=[]'
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = module_1.wrap_invariant(var_3)
    var_5 = var_1.mset()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_5) == 0
    var_6 = module_1.CheckedPSet(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_6) == 0
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_7 = module_1.CheckedPMap()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_7) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_8 = module_1.optional()
    var_9 = var_8.__ge__(var_1)
    var_10 = var_7.set(var_8, var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_10) == 1
    var_11 = var_3.serialize(var_2)
    var_12 = var_5.__eq__(var_8)
    assert var_12 is False
    var_13 = var_4.__repr__()
    var_14 = module_1.maybe_parse_user_type(var_10)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_1.CheckedPSet()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.__str__()
    assert var_2 == 'CheckedPSet()'
    var_3 = module_1.wrap_invariant(var_1)
    var_4 = var_0.serialize()
    var_5 = var_0.add(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_5) == 1
    var_6 = var_5.__reduce__()
    var_7 = var_0.__reduce__()
    var_8 = module_1.CheckedPMap()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_8) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_9 = var_8.__reduce__()
    var_10 = var_9.__repr__()
    assert var_10 == "(<function _restore_pickle at 0x7e0bddfc9e10>, (<class 'pyrsistent._checked_types.CheckedPMap'>, {}))"
    var_11 = var_8.set(var_5, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_11) == 1
    var_11.serialize()

def test_case_34():
    var_0 = module_1.CheckedPMap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.keys()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_0.set(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_3) == 1
    var_4 = module_1.optional()
    var_5 = var_0.serialize()
    with pytest.raises(TypeError):
        module_1.store_invariants(var_4, var_4, var_2, var_1)

def test_case_35():
    var_0 = module_1.InvariantException()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_0.invariant_errors == ()
    assert var_0.missing_fields == ()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_1.CheckedPVector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_1) == 0
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_2 = module_1.InvariantException(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_2.invariant_errors == ()
    assert var_2.missing_fields == ()
    var_3 = var_1.append(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = module_1.wrap_invariant(var_3)
    var_5 = var_1.mset()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_5) == 0
    var_6 = var_1.serialize()
    var_7 = module_1.CheckedPSet(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_7) == 0
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_8 = module_1.CheckedPMap()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_8) == 0
    assert f'{type(module_1.CheckedPMap.create).__module__}.{type(module_1.CheckedPMap.create).__qualname__}' == 'builtins.method'
    var_9 = var_5.__repr__()
    assert var_9 == 'CheckedPVector([])'
    var_10 = var_9.__ge__(var_6)
    var_11 = var_8.set(var_9, var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._checked_types.CheckedPMap'
    assert len(var_11) == 1
    var_12 = var_3.serialize(var_3)
    var_13 = var_12.__eq__(var_3)
    var_14 = var_4.__repr__()
    var_15 = var_7.__repr__()
    assert var_15 == 'CheckedPSet()'
    var_16 = module_1.maybe_parse_user_type(var_9)

@pytest.mark.xfail(strict=True)
def test_case_36():
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
    var_2 = var_1.__str__()
    assert var_2 == 'CheckedPSet()'
    var_3 = var_2.__len__()
    assert var_3 == 13
    var_4 = var_3.__str__()
    assert var_4 == '13'
    var_4.__reduce__()

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__str__()
    assert var_1 == 'CheckedPVector([])'
    var_2 = module_1.InvariantException(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.InvariantException'
    assert var_2.invariant_errors == ('C',)
    assert var_2.missing_fields == 'h'
    var_3 = var_0.append(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 1
    var_4 = module_1.wrap_invariant(var_3)
    var_5 = var_0.serialize()
    var_6 = module_1.CheckedPSet(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._checked_types.CheckedPSet'
    assert len(var_6) == 0
    assert f'{type(module_1.CheckedPSet.create).__module__}.{type(module_1.CheckedPSet.create).__qualname__}' == 'builtins.method'
    var_7 = var_2.__str__()
    assert var_7 == "('e', 'c', 'k', 'e', 'd', 'P', 'V', 'e', 'c', 't', 'o', 'r', '(', '[', ']', ')'), invariant_errors=[C], missing_fields=[h]"
    var_8 = var_4.__repr__()
    var_1.evolver()

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = 'error1'
    var_1 = 'error2'
    var_2 = lambda : var_1
    var_3 = (var_0, var_2)
    var_4 = {}
    module_1.InvariantException(var_3, **var_4)

def test_case_39():
    var_0 = 'invariant'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'invariants'
    var_5 = 'invariant'
    with pytest.raises(TypeError):
        module_1.store_invariants(var_2, var_3, var_4, var_5)

def test_case_40():
    var_0 = 'inv1'
    var_1 = 'inv2'
    var_2 = True
    var_3 = lambda : var_2
    var_4 = False
    var_5 = lambda : var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = []
    var_8 = 'invariants'
    var_9 = 'inv1'
    var_10 = module_1.store_invariants(var_6, var_7, var_8, var_9)
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_11 = var_6[var_8]

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 'test'
    var_1 = True
    var_2 = 'data1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'data2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    module_1._invariant_errors(var_0, var_8)