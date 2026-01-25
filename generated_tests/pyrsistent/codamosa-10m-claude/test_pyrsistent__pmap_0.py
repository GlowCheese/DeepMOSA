# Check out: https://github.com/GlowCheese/deepmosa
import typing as module_1

import pyrsistent._pmap as module_0
import pyrsistent._pvector as module_3
import pyrsistent._transformations as module_2
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.PMapItems(var_0)

def test_case_1():
    var_0 = 'sGUhHC8\\>ZI,~\n!'
    var_1 = None
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_4) == 1
    var_5 = var_4.__repr__()
    assert var_5 == 'pmap_values([None])'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.update(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_2) == 0
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_0.remove(var_0)

def test_case_3():
    var_0 = False
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(AttributeError):
        var_1.__getattr__(var_0)

def test_case_4():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__contains__(var_0)
    assert var_1 is False
    var_2 = var_0.set(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    with pytest.raises(TypeError):
        module_0.PMapView(var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__repr__()
    assert var_2 == 'pmap({})'
    var_2.items()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.transform()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_2.set(var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    var_5 = var_3.__str__()
    assert var_5 == 'pmap({pmap({}): pmap({})})'
    var_5.remove(var_5)

def test_case_7():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.transform()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.__contains__(var_2)
    assert var_3 is False
    var_4 = var_1.set(var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    var_5 = var_2.__str__()
    assert var_5 == 'pmap({pmap({}): pmap({})})'
    var_5.remove(var_5)

def test_case_9():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    with pytest.raises(TypeError):
        var_2.__reversed__()

def test_case_10():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__add__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_1.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_2) == 0
    var_3 = True
    var_4 = var_2.__contains__(var_3)
    assert var_4 is False
    var_5 = module_0.pmap(pre_size=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.__add__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_3) == 1
    module_0.pmap(pre_size=var_2)

@pytest.mark.xfail(strict=True)
def test_case_12():
    module_0.PMap()

def test_case_13():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    module_0.pmap(var_0, var_0)

def test_case_15():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.__iter__()

def test_case_16():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.__str__()
    assert var_2 == 'pmap({})'
    var_3 = var_1.remove(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__reduce__()
    var_1.evolver()

def test_case_18():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_19():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_1.copy()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0

def test_case_20():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_0.set(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_0.__lt__(var_0)

def test_case_22():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = module_0.m()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_1.__add__(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_2.__contains__(var_6)
    assert var_7 is False
    var_8 = var_6.__repr__()
    assert var_8 == 'pmap({})'
    var_9 = module_1.Generic()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typing.Generic'
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
    var_10 = var_0.evolver()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_10) == 0
    var_11 = var_10.__eq__(var_0)
    var_12 = var_10.__len__()
    assert var_12 == 0
    var_13 = module_1.Generic(*var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typing.Generic'
    var_14 = module_0.PMapItems(var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_14) == 0
    var_15 = var_14.__contains__(var_0)
    assert var_15 is False
    var_16 = var_6.set(var_7, var_6)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 1
    with pytest.raises(TypeError):
        var_1.__reversed__()

def test_case_23():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.transform()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1

def test_case_24():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = module_0.m()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.__contains__(var_1)
    assert var_3 is False
    var_4 = module_0.pmap(pre_size=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_4.__add__(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_4.__contains__(var_4)
    assert var_6 is False
    var_7 = var_4.__repr__()
    assert var_7 == 'pmap({})'
    var_8 = var_0.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_8) == 0
    var_9 = var_5.set(var_3, var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 1
    var_10 = var_4.__eq__(var_0)
    assert var_10 is True
    var_11 = var_9.__len__()
    assert var_11 == 1
    var_12 = var_9.__contains__(var_3)
    assert var_12 is True
    var_13 = module_0.PMapItems(var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_13) == 0
    var_14 = var_11.__hash__()
    assert var_14 == 1
    var_15 = var_10.__eq__(var_6)
    assert var_15 is False
    var_16 = var_13.__eq__(var_11)
    assert var_16 is False

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.m()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.update_with(var_0, *var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.transform()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_5.__contains__(var_5)
    assert var_6 is False
    var_7 = var_1.set(var_0, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = var_1.transform(*var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_0.__add__(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_7.__contains__(var_4)
    assert var_10 is True
    var_11 = module_0.PMapItems(var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_11) == 1
    var_12 = module_1.Generic(**var_2)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typing.Generic'
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
    var_13 = var_10.__repr__()
    assert var_13 == 'True'
    var_14 = var_1.__str__()
    assert var_14 == 'pmap({})'
    var_15 = var_8.set(var_13, var_9)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 1
    var_16 = {}
    var_17 = var_3.__eq__(var_16)
    assert var_17 is True
    var_18 = var_11.__eq__(var_14)
    assert var_18 is False
    var_19 = var_11.__len__()
    assert var_19 == 1
    var_19.__contains__(var_5)

def test_case_26():
    var_0 = None
    var_1 = None
    var_2 = module_0.m()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.set(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.transform()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_4.__contains__(var_3)
    assert var_5 is False
    var_6 = var_4.set(var_2, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_3.__contains__(var_6)
    assert var_7 is False
    var_8 = var_4.__repr__()
    assert var_8 == 'pmap({pmap({}): pmap({})})'
    var_9 = var_3.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_9) == 1
    var_10 = var_3.set(var_0, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 2
    var_11 = var_3.__str__()
    assert var_11 == 'pmap({pmap({}): pmap({})})'
    var_12 = var_11.__str__()
    assert var_12 == 'pmap({pmap({}): pmap({})})'
    var_13 = var_3.values()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_13) == 1
    var_14 = var_3.values()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_14) == 1
    with pytest.raises(TypeError):
        var_14.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = None
    var_2 = module_0.m()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.set(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.transform()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_4.__contains__(var_3)
    assert var_5 is False
    var_6 = var_4.set(var_2, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_4.__repr__()
    assert var_7 == 'pmap({pmap({}): pmap({})})'
    var_8 = var_4.__contains__(var_6)
    assert var_8 is False
    var_9 = var_3.__str__()
    assert var_9 == 'pmap({pmap({}): pmap({})})'
    var_10 = var_6.copy()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_6.remove(var_0)

def test_case_28():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.transform()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.__contains__(var_3)
    assert var_4 is False
    var_5 = var_2.set(var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    var_6 = var_3.__str__()
    assert var_6 == 'pmap({pmap({}): pmap({})})'
    var_7 = var_5.copy()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 2
    var_8 = var_7.remove(var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1

def test_case_29():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.__eq__(var_1)
    assert var_2 is False
    var_3 = var_1.transform()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.__contains__(var_1)
    assert var_4 is False
    var_5 = var_1.__contains__(var_4)
    assert var_5 is False
    var_6 = var_1.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 1
    var_7 = var_3.__str__()
    assert var_7 == 'pmap({pmap({}): pmap({})})'
    var_8 = var_6.__contains__(var_5)
    assert var_8 is False

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = {}
    var_2 = module_0.PMapValues(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_2) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2.set(var_0, var_2)

def test_case_31():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.__eq__(var_1)
    assert var_2 is False
    var_3 = var_1.transform()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_1.__contains__(var_3)
    assert var_4 is False
    var_5 = var_1.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 1
    var_6 = var_5.__contains__(var_4)
    assert var_6 is False

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.transform()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.__contains__(var_2)
    assert var_4 is False
    var_5 = var_3.set(var_1, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_3.__contains__(var_5)
    assert var_6 is False
    var_7 = var_2.__str__()
    assert var_7 == 'pmap({pmap({}): pmap({})})'
    var_8 = var_5.copy()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_7.iterkeys()

def test_case_33():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = var_0.__contains__(var_1)
    assert var_5 is False
    var_6 = var_2.set(var_0, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_2.transform(*var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_2.__contains__(var_0)
    assert var_8 is False
    var_9 = var_3.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_9) == 0
    var_10 = module_1.Generic()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typing.Generic'
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
    var_11 = var_2.__contains__(var_7)
    assert var_11 is False
    var_12 = var_11.__hash__()
    assert var_12 == 0
    var_13 = var_9.__str__()
    assert var_13 == 'pmap_items([])'
    with pytest.raises(AttributeError):
        var_4.__getattr__(var_13)

def test_case_34():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = var_0.__contains__(var_1)
    assert var_5 is False
    var_6 = var_2.set(var_0, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_2.transform(*var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_6.__add__(var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_2.__contains__(var_6)
    assert var_9 is False
    var_10 = module_1.Generic()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typing.Generic'
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
    var_11 = module_1.Generic()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typing.Generic'
    var_12 = var_7.evolver()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_12) == 0
    var_13 = var_1.__hash__()
    assert var_13 == 133146708735736
    var_14 = var_12.__contains__(var_13)
    assert var_14 is False
    var_15 = module_0.PMapItems(var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_15) == 0
    var_16 = var_15.__contains__(var_0)
    assert var_16 is False
    var_17 = var_13.__hash__()
    assert var_17 == 133146708735736
    var_18 = var_8.set(var_9, var_8)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 2
    with pytest.raises(TypeError):
        var_15.__setattr__(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.update_with(var_0, *var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = var_2.set(var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_2.transform(*var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_4.__add__(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_2.__len__()
    assert var_7 == 0
    var_8 = var_2.__contains__(var_4)
    assert var_8 is False
    var_9 = var_1.__eq__(var_4)
    assert var_9 is False
    var_10 = var_2.__add__(var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = var_1.__hash__()
    assert var_11 == 133146708735736
    var_12 = var_11.__hash__()
    assert var_12 == 133146708735736
    var_13 = var_6.set(var_12, var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 2
    var_14 = var_2.__contains__(var_9)
    assert var_14 is False
    var_15 = var_0.__hash__()
    assert var_15 == 133146708735736
    var_16 = var_13.set(var_15, var_2)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 2
    var_17 = [var_4, var_14]
    module_1.Generic(*var_17)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.m(**var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.values()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 0
    var_4 = var_1.transform()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_1.__eq__(var_4)
    assert var_5 is True
    var_6 = var_3.__str__()
    assert var_6 == 'pmap_values([])'
    var_6.discard(var_0)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = None
    var_1 = None
    var_2 = 'i'
    var_3 = "/m\x0cl+Itt'Nb+w"
    var_4 = ':]2<tD'
    var_5 = {var_2: var_0, var_2: var_1, var_3: var_0, var_4: var_0}
    var_6 = module_0.m(**var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 3
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_7 = var_6.discard(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 3
    var_8 = var_6.__reduce__()
    var_8.__new__(var_0, var_6, var_0)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = module_0.m()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_0.__contains__(var_1)
    assert var_6 is False
    var_7 = var_3.transform()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_2.set(var_0, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_2.transform(*var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_8.__add__(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = var_2.__contains__(var_8)
    assert var_11 is False
    var_12 = module_0.PMapItems(var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_12) == 0
    var_13 = module_1.Generic()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typing.Generic'
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
    var_14 = var_8.__repr__()
    assert var_14 == 'pmap({pmap({}): pmap({})})'
    var_15 = module_1.Generic()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typing.Generic'
    var_16 = var_9.evolver()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_16) == 0
    var_17 = var_8.__eq__(var_0)
    assert var_17 is False
    var_18 = var_3.__contains__(var_9)
    assert var_18 is False
    var_19 = var_6.__lt__(var_7)
    var_20 = module_0.PMapItems(var_5)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_20) == 0
    var_21 = var_16.__contains__(var_3)
    assert var_21 is False
    var_22 = var_14.__hash__()
    assert var_22 == -5402913308243726285
    var_23 = var_7.items()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_23) == 0
    var_24 = var_23.__repr__()
    assert var_24 == 'pmap_items([])'
    var_25 = var_24.__eq__(var_11)
    var_21.__len__()

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_3.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = var_1.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 0
    var_6 = var_0.__contains__(var_1)
    assert var_6 is False
    var_7 = var_2.transform(*var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_1.__add__(var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_2.__contains__(var_3)
    assert var_9 is False
    var_10 = module_1.Generic()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typing.Generic'
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
    var_11 = module_1.Generic()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typing.Generic'
    var_12 = var_7.evolver()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_12) == 0
    var_13 = var_12.__len__()
    assert var_13 == 0
    var_14 = module_0.pmap(pre_size=var_3)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 0
    var_15 = var_14.__eq__(var_3)
    assert var_15 is True
    var_13.__len__()

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_1.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_2.update_with(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = module_0.m()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_2.transform(*var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_6.set(var_3, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = module_0.pmap()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_5.__add__(var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_7.__contains__(var_5)
    assert var_10 is True
    var_11 = module_0.PMapItems(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_11) == 0
    var_12 = module_1.Generic()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typing.Generic'
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
    var_13 = var_9.__repr__()
    assert var_13 == 'pmap({})'
    var_14 = var_13.__str__()
    assert var_14 == 'pmap({})'
    var_15 = var_12.__eq__(var_2)
    var_16 = module_2.transform(var_1, var_13)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 4
    var_17 = var_16.__len__()
    assert var_17 == 4
    var_18 = var_9.__contains__(var_7)
    assert var_18 is False
    var_18.__contains__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = module_0.m()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_3.update(*var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_3.transform()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_2.set(var_0, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_2.transform(*var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_8.__add__(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = var_2.__contains__(var_8)
    assert var_11 is False
    var_12 = module_0.PMapItems(var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_12) == 0
    var_13 = module_1.Generic()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typing.Generic'
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
    var_14 = module_0.PMapValues(var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_14) == 0
    var_15 = var_14.__str__()
    assert var_15 == 'pmap_values([])'
    var_16 = var_10.set(var_4, var_1)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 2
    var_17 = var_8.__eq__(var_0)
    assert var_17 is False
    var_18 = var_14.__contains__(var_1)
    assert var_18 is False
    var_19 = var_18.__lt__(var_5)
    module_0.PMapItems(var_11)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.m()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.__iter__()
    var_5 = var_2.set(var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_0.transform(*var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_5.__add__(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = var_5.__contains__(var_5)
    assert var_8 is False
    var_9 = var_2.__repr__()
    assert var_9 == 'pmap({})'
    var_10 = var_1.values()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_10) == 0
    var_11 = var_7.update()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 1
    var_12 = var_2.items()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_12) == 0
    var_13 = var_7.set(var_0, var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1
    var_14 = var_5.__eq__(var_0)
    assert var_14 is False
    var_15 = var_13.__len__()
    assert var_15 == 1
    var_16 = var_14.__lt__(var_5)
    var_17 = module_0.PMapItems(var_1)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_17) == 0
    var_18 = var_17.__contains__(var_0)
    assert var_18 is False
    var_19 = var_15.__hash__()
    assert var_19 == 1
    var_20 = var_1.itervalues()
    var_21 = var_10.__eq__(var_8)
    assert var_21 is False
    module_0.pmap(var_19)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.m()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.update_with(var_0, *var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.transform()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_5.__contains__(var_5)
    assert var_6 is False
    var_7 = var_1.set(var_0, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = var_1.transform(*var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_0.__add__(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_7.__contains__(var_4)
    assert var_10 is True
    var_11 = module_1.Generic(**var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typing.Generic'
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
    var_12 = var_10.__repr__()
    assert var_12 == 'True'
    var_13 = var_1.__str__()
    assert var_13 == 'pmap({})'
    var_14 = var_8.set(var_12, var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 1
    var_15 = {}
    var_16 = var_3.__eq__(var_15)
    assert var_16 is True
    var_17 = var_0.__eq__(var_13)
    var_13.__contains__(var_5)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_1.keys()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_2) == 0
    var_3 = module_0.m()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_3.set(var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_4.__add__(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_1.__contains__(var_4)
    assert var_6 is False
    var_7 = module_0.PMapItems(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 0
    var_8 = var_5.set(var_2, var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 2
    var_9 = var_3.copy()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_8.__len__()
    assert var_10 == 2
    var_11 = module_0.PMapItems(var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_11) == 1
    var_12 = var_7.__contains__(var_8)
    assert var_12 is False
    var_13 = var_5.__hash__()
    assert var_13 == -4783114350154387147
    var_14 = var_5.__eq__(var_1)
    assert var_14 is False
    var_15 = module_0.pmap(pre_size=var_2)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_16 = module_3.python_pvector(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_16) == 0
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_3.BRANCH_FACTOR == 32
    assert module_3.BIT_MASK == 31
    assert module_3.SHIFT == 5
    var_16.__contains__(var_12)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = module_0.m()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_3.update(*var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_5.__iter__()
    var_8 = var_3.transform()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_2.transform(*var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_8.__add__(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = var_2.__contains__(var_2)
    assert var_11 is False
    var_12 = module_0.PMapItems(var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_12) == 0
    var_13 = module_1.Generic()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typing.Generic'
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
    var_14 = var_11.__repr__()
    assert var_14 == 'False'
    var_15 = var_14.__str__()
    assert var_15 == 'False'
    var_16 = var_10.set(var_4, var_1)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 1
    var_17 = var_8.__eq__(var_0)
    assert var_17 is True
    var_18 = var_16.__len__()
    assert var_18 == 1
    var_19 = var_16.__contains__(var_18)
    assert var_19 is False
    var_20 = var_17.__lt__(var_0)
    var_21 = module_0.PMapItems(var_1)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_21) == 0
    var_22 = var_21.__contains__(var_0)
    assert var_22 is False
    var_23 = var_18.__hash__()
    assert var_23 == 1
    var_24 = var_21.__eq__(var_12)
    assert var_24 is True
    var_25 = var_1.__eq__(var_0)
    assert var_25 is True
    module_0.pmap(var_20)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_1.transform()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.copy()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.transform()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_3.keys()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 0
    var_7 = module_0.m()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_5.update(*var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_5.transform()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_3.set(var_0, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = var_3.transform(*var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 0
    var_12 = var_10.__add__(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = var_3.__contains__(var_10)
    assert var_13 is False
    var_14 = module_0.PMapItems(var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_14) == 0
    var_15 = module_1.Generic()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typing.Generic'
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
    var_16 = var_10.__repr__()
    assert var_16 == 'pmap({pmap({}): pmap({})})'
    var_17 = var_16.__str__()
    assert var_17 == 'pmap({pmap({}): pmap({})})'
    var_18 = var_12.set(var_6, var_1)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 2
    var_19 = var_10.__eq__(var_0)
    assert var_19 is False
    var_20 = var_18.__len__()
    assert var_20 == 2
    var_21 = var_18.__contains__(var_20)
    assert var_21 is False
    var_22 = var_19.__lt__(var_10)
    var_23 = module_0.PMapItems(var_1)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_23) == 0
    var_24 = var_23.__contains__(var_0)
    assert var_24 is False
    var_25 = var_20.__hash__()
    assert var_25 == 2
    var_26 = var_23.__eq__(var_10)
    assert var_26 is False
    var_27 = var_1.__eq__(var_0)
    assert var_27 is True
    var_0.__getattr__(var_23)

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = module_0.m()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_3.update(*var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_3.transform()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_2.set(var_0, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_2.transform(*var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_8.__add__(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = var_2.__contains__(var_8)
    assert var_11 is False
    var_12 = module_0.PMapItems(var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_12) == 0
    var_13 = module_1.Generic()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typing.Generic'
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
    var_14 = var_8.__repr__()
    assert var_14 == 'pmap({pmap({}): pmap({})})'
    var_15 = var_3.__iter__()
    var_16 = var_7.set(var_4, var_3)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 1
    var_17 = var_16.update()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 1
    var_18 = var_17.__eq__(var_10)
    assert var_18 is False
    var_19 = var_8.__len__()
    assert var_19 == 1
    var_20 = var_0.__contains__(var_3)
    assert var_20 is False
    var_21 = var_11.__lt__(var_0)
    module_0.PMapItems(var_20)

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = module_0.m()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_0.__contains__(var_1)
    assert var_5 is False
    var_6 = var_2.set(var_0, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_0.transform(*var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_6.__add__(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_6.__contains__(var_6)
    assert var_9 is False
    var_10 = var_2.__repr__()
    assert var_10 == 'pmap({})'
    var_11 = var_1.values()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_11) == 0
    var_12 = var_8.update()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = var_6.__repr__()
    assert var_13 == 'pmap({pmap({}): pmap({})})'
    var_14 = var_2.items()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_14) == 0
    var_15 = var_8.set(var_3, var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 2
    var_16 = var_4.itervalues()
    var_17 = var_9.__eq__(var_7)
    var_4.discard(var_11)

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = module_0.m()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_0.__contains__(var_1)
    assert var_5 is False
    var_6 = var_2.set(var_0, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_0.transform(*var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_6.__add__(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_2.__repr__()
    assert var_9 == 'pmap({})'
    var_10 = var_1.values()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_10) == 0
    var_11 = var_8.update()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 1
    var_12 = var_6.__repr__()
    assert var_12 == 'pmap({pmap({}): pmap({})})'
    var_13 = var_8.set(var_3, var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 2
    var_14 = var_6.__eq__(var_0)
    assert var_14 is False
    var_15 = var_13.__len__()
    assert var_15 == 2
    var_16 = var_13.__contains__(var_15)
    assert var_16 is False
    var_17 = var_14.__lt__(var_6)
    var_18 = module_0.PMapItems(var_1)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_18) == 0
    var_19 = var_18.__contains__(var_0)
    assert var_19 is False
    var_20 = var_15.__hash__()
    assert var_20 == 2
    var_21 = var_1.itervalues()
    var_22 = var_10.__eq__(var_10)
    assert var_22 is True
    var_23 = var_18.__eq__(var_15)
    assert var_23 is False
    module_0.pmap(var_20)

def test_case_50():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = module_0.m()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_3.__iter__()
    var_6 = var_0.__contains__(var_1)
    assert var_6 is False
    var_7 = module_0.pmap(pre_size=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_0.transform(*var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_7.__add__(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_2.__repr__()
    assert var_10 == 'pmap({})'
    var_11 = var_1.values()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_11) == 0
    var_12 = var_9.update()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 0
    var_13 = var_7.__repr__()
    assert var_13 == 'pmap({})'
    var_14 = var_2.items()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_14) == 0
    var_15 = var_7.__eq__(var_0)
    assert var_15 is True
    var_16 = var_13.__len__()
    assert var_16 == 8
    var_17 = var_9.__contains__(var_6)
    assert var_17 is False
    var_18 = var_15.__lt__(var_7)
    var_19 = module_0.PMapItems(var_1)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_19) == 0
    var_20 = var_19.__contains__(var_0)
    assert var_20 is False
    var_21 = var_16.__hash__()
    assert var_21 == 8
    var_22 = var_1.itervalues()
    var_23 = var_11.__eq__(var_3)
    assert var_23 is False
    var_24 = var_4.__eq__(var_9)
    assert var_24 is True
    var_25 = module_0.pmap(var_7)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_25) == 0

@pytest.mark.xfail(strict=True)
def test_case_51():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.update_with(var_0, *var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = module_0.m()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_3.evolver()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pset.PSet._Evolver'
    assert len(var_5) == 0
    var_5.remove(var_2)