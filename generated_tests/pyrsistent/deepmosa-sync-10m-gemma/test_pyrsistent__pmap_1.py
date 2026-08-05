# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._pmap as module_0
import typing as module_1
import pyrsistent._pvector as module_2
import builtins as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.values()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_2) == 0
    var_3 = var_2.__iter__()
    module_0.PMapItems(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'd4i-\x0c2l-_hB2Q'
    module_0.PMapValues(var_0)

def test_case_2():
    var_0 = 1
    var_1 = '^"b'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__add__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = 'Q-\r!_x_%;AY:'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
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
    var_5 = var_3.set(var_3, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    var_6 = var_3.update()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    module_0.PMapValues(var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__str__()
    assert var_2 == 'pmap({})'
    var_3 = var_2.__eq__(var_0)
    var_4 = var_1.__contains__(var_2)
    assert var_4 is False
    var_5 = var_1.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 0
    var_6 = var_4.__repr__()
    assert var_6 == 'False'
    with pytest.raises(TypeError):
        module_0.PMapView(var_0)

def test_case_5():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = module_0.PMapItems(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = module_0.PMapItems(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 2
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

def test_case_6():
    var_0 = 23
    var_1 = '?+'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.discard(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = bool(var_3 != var_2)

def test_case_7():
    var_0 = 23
    var_1 = 'b'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = bool(var_3 != var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_2.iterkeys()
    var_4 = var_1.__iter__()
    var_3.values()

def test_case_9():
    pass

def test_case_10():
    var_0 = 1
    var_1 = {var_0: var_0}
    var_2 = module_0.PMapItems(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_2) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = bool(not 'not_a_tuple' in var_2)
    assert var_3 is True

def test_case_11():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap(pre_size=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    var_3.__getitem__(var_0)

def test_case_13():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
    assert var_6 is True
    with pytest.raises(TypeError):
        var_5.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_2.iterkeys()
    var_2.__lt__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'Q-\r!_x_%;AY:'
    var_1 = var_0.__repr__()
    assert var_1 == "'Q-\\r!_x_%;AY:'"
    var_2 = module_0.m()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2.transform(*var_1)

def test_case_16():
    var_0 = 2
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = bool(var_1 != var_3)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    module_0.pmap(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = -15
    var_1 = '?vX\x0c'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.discard(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_4.__getitem__(var_3)

def test_case_19():
    var_0 = 23
    var_1 = '?+'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.discard(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.__add__(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.__reduce__()
    var_7 = bool(var_3 != var_2)

def test_case_20():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = {}
    var_3 = module_0.PMapItems(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_3) == 0
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

def test_case_21():
    var_0 = 23
    var_1 = '?+'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.discard(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 1
    var_6 = module_2.python_pvector(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_6) == 1
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_2.BRANCH_FACTOR == 32
    assert module_2.BIT_MASK == 31
    assert module_2.SHIFT == 5
    var_7 = bool(var_3 != var_2)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 23
    var_1 = '}NEb'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.iterkeys()
    var_5 = var_3.discard(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.keys()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 1
    var_7 = var_3.items()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 1
    var_8 = var_3.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_8) == 1
    var_9 = var_8.__contains__(var_5)
    assert var_9 is False
    var_10 = var_3.__repr__()
    assert var_10 == "pmap({'}NEb': 23})"
    var_11 = var_8.__str__()
    assert var_11 == "pmap_items([('}NEb', 23)])"
    var_12 = var_5.copy()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = var_12.discard(var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 0
    var_14 = var_3.__add__(var_3)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 1
    var_15 = bool(var_3 != var_2)
    var_12.__add__(var_7)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = 'Q-\r!_x_%;AY:'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.__eq__(var_0)
    var_6 = var_4.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 2
    var_7 = var_6.__eq__(var_0)
    assert var_7 is False
    var_5.keys()

def test_case_24():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__eq__(var_0)
    var_5 = var_3.__contains__(var_4)
    assert var_5 is False
    var_6 = var_3.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 1
    var_7 = var_5.__repr__()
    assert var_7 == 'False'
    with pytest.raises(AttributeError):
        var_3.__getattr__(var_3)

def test_case_25():
    var_0 = 23
    var_1 = '?+'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.discard(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 1
    var_6 = var_4.copy()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_3.__add__(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = bool(var_3 != var_2)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.evolver()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_3) == 1
    var_4 = var_3.__len__()
    assert var_4 == 1
    var_5 = module_0.m()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_5.__repr__()
    assert var_6 == 'pmap({})'
    var_6.items()

def test_case_27():
    var_0 = {}
    var_1 = 4
    var_2 = module_0._turbo_mapping(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = bool(var_2 == {})
    assert var_4 is True

def test_case_28():
    var_0 = '#*'
    var_1 = 'b'
    var_2 = -29
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapValues(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_5) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

def test_case_29():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = 'Q-\r!_x_%;AY:'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.__str__()
    assert var_5 == "pmap({'Q-\\r!_x_%;AY:': None, 'd4i-\\x0c2l-_hB2Q': None})"
    var_6 = var_4.__eq__(var_0)
    var_7 = var_4.items()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 2
    var_8 = var_7.__repr__()
    assert var_8 == "pmap_items([('Q-\\r!_x_%;AY:', None), ('d4i-\\x0c2l-_hB2Q', None)])"
    var_9 = module_0.pmap()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    with pytest.raises(TypeError):
        module_0.PMapView(var_5)

def test_case_30():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = 'Q-\r!_x_%;AY:'
    var_3 = {var_1: var_0, var_2: var_0, var_1: var_0}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.__str__()
    assert var_5 == "pmap({'Q-\\r!_x_%;AY:': None, 'd4i-\\x0c2l-_hB2Q': None})"
    var_6 = var_4.__eq__(var_0)
    var_7 = var_4.__contains__(var_5)
    assert var_7 is False
    var_8 = var_4.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_8) == 2
    var_9 = var_7.__repr__()
    assert var_9 == 'False'
    with pytest.raises(TypeError):
        var_8.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = 'Q-\r!_x_%;AY:'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__eq__(var_0)
    var_5 = var_3.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 1
    var_6 = var_4.__repr__()
    assert var_6 == 'NotImplemented'
    var_3.__getattr__(var_5)

def test_case_32():
    var_0 = 23
    var_1 = '?+'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_4) == 1
    var_5 = var_3.__hash__()
    assert var_5 == -8042623584985161951
    var_6 = var_4.__eq__(var_3)
    assert var_6 is False

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = 'Q-\r!_x_%;AY:'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.PMapValues(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__contains__(var_0)
    assert var_4 is True
    var_5 = var_3.__eq__(var_0)
    assert var_5 is False
    var_6 = var_0.__repr__()
    assert var_6 == 'None'
    var_1.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = 'Q-\r!_x_%;AY:'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.__eq__(var_0)
    var_6 = var_4.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 2
    var_7 = var_5.__repr__()
    assert var_7 == 'NotImplemented'
    var_8 = module_0.m()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_8.transform(*var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 6
    var_8.update(*var_5)

def test_case_35():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = 'Q-\r!_x_%;AY:'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.__eq__(var_0)
    var_6 = var_4.__contains__(var_5)
    assert var_6 is False
    var_7 = var_4.items()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 2
    with pytest.raises(TypeError):
        var_7.__setattr__(var_5, var_5)

def test_case_36():
    var_0 = 1
    var_1 = 'b'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_3}
    var_7 = module_0.m(**var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 2
    var_8 = bool(var_3 == var_7)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = None
    var_1 = 'd4x-\x0cgl-_hm2Q'
    var_2 = 'Q-\r!_x_%;AY:'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.__contains__(var_1)
    assert var_5 is True
    var_6 = var_4.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 2
    var_7 = var_6.__contains__(var_6)
    assert var_7 is False
    var_2.items()

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = 'Q-\r!_x_%;AY:'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 2
    var_6 = var_3.__repr__()
    assert var_6 == "{'d4i-\\x0c2l-_hB2Q': None, 'Q-\\r!_x_%;AY:': None}"
    var_7 = module_0.m()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_7.transform(*var_6)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = None
    var_1 = 'd4x-\x0cgl-_hm2Q'
    var_2 = 'Q-\r!_x_%;AY:'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.__contains__(var_1)
    assert var_5 is True
    var_6 = var_4.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 2
    var_7 = var_4.__repr__()
    assert var_7 == "pmap({'Q-\\r!_x_%;AY:': None, 'd4x-\\x0cgl-_hm2Q': None})"
    var_8 = module_0.m()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_8.transform(*var_7)

def test_case_40():
    var_0 = 43
    var_1 = '}NEb'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.iterkeys()
    var_5 = var_3.discard(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.keys()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 1
    var_7 = var_3.items()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 1
    var_8 = var_5.copy()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_8.discard(var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_3.__add__(var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = bool(var_3 != var_2)

def test_case_41():
    var_0 = 23
    var_1 = '?+'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_4) == 1
    var_5 = var_4.__repr__()
    assert var_5 == 'pmap_values([23])'
    var_6 = var_4.__eq__(var_3)
    assert var_6 is False
    var_7 = bool(var_3 != var_2)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = None
    var_1 = 'Q-\r!_x_%;AY:'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_3.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 1
    var_3.discard(var_5)

def test_case_43():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = 2
    var_5 = 'b'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = hash(var_3)
    var_9 = hash(var_7)
    var_10 = bool(var_3 != var_7)
    assert var_10 is True

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.update()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.m(**var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_3.__contains__(var_0)
    assert var_4 is False
    var_5 = var_3.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 0
    var_6 = var_5.__repr__()
    assert var_6 == 'pmap_items([])'
    var_7 = var_3.__eq__(var_1)
    assert var_7 is True
    var_8 = var_0.__repr__()
    assert var_8 == 'None'
    var_9 = module_0.m()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_9.transform(*var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 2
    var_11 = var_3.__len__()
    assert var_11 == 0
    var_12 = [var_11, var_5, var_0]
    var_9.transform(*var_12)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.update()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.m(**var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_3.__eq__(var_0)
    var_5 = var_3.__contains__(var_2)
    assert var_5 is False
    var_6 = var_5.__repr__()
    assert var_6 == 'False'
    var_7 = var_3.__eq__(var_1)
    assert var_7 is True
    var_8 = var_4.__repr__()
    assert var_8 == 'NotImplemented'
    var_9 = module_3.object()
    var_9.transform(*var_8)

def test_case_46():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = 2
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = bool(var_3 != var_7)
    assert var_8 is True
    var_9 = bool(var_7 != var_3)
    assert var_9 is True

def test_case_47():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.values()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_1) == 0
    var_2 = var_1.__str__()
    assert var_2 == 'pmap_values([])'
    var_2.__reduce__()

def test_case_49():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = module_0.PMapItems(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = bool(('a', 1) in var_4)
    assert var_5 is True

def test_case_50():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapValues(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_5) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

def test_case_51():
    var_0 = 23
    var_1 = '}NEb'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.iterkeys()
    var_5 = var_3.set(var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    var_6 = var_3.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 1
    var_7 = var_6.__contains__(var_5)
    assert var_7 is False
    var_8 = var_3.__repr__()
    assert var_8 == "pmap({'}NEb': 23})"
    var_9 = var_5.copy()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 2
    var_10 = var_9.discard(var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = var_8.__eq__(var_5)
    var_12 = var_3.__add__(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = bool(var_3 != var_2)

def test_case_52():
    var_0 = 23
    var_1 = '}NEb'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.iterkeys()
    var_5 = var_3.keys()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_5) == 1
    var_6 = var_3.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 1
    var_7 = var_3.items()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 1
    var_8 = var_7.__contains__(var_5)
    assert var_8 is False
    var_9 = var_3.__repr__()
    assert var_9 == "pmap({'}NEb': 23})"
    var_10 = var_5.copy()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_10) == 1
    var_11 = var_10.discard(var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_11) == 0
    var_12 = var_3.__add__(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = var_10.__eq__(var_11)
    assert var_13 is False
    var_14 = bool(var_3 != var_2)
    var_15 = var_12.__add__(var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 1