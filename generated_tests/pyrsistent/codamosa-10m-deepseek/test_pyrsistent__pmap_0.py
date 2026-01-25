# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._pmap as module_0
import pyrsistent._pvector as module_1
import pyrsistent._transformations as module_2
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.PMapItems(var_0)

def test_case_1():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__add__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

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
    var_0 = None
    var_1 = module_1.python_pvector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_1.BRANCH_FACTOR == 32
    assert module_1.BIT_MASK == 31
    assert module_1.SHIFT == 5
    var_2 = var_1.__len__()
    assert var_2 == 0
    var_3 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(AttributeError):
        var_3.__getattr__(var_2)

def test_case_4():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    var_3 = var_1.set(var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = None
    with pytest.raises(TypeError):
        module_0.PMapView(var_4)

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
    var_1.remove(var_1)

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
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.__str__()
    assert var_3 == 'pmap({})'
    var_2.remove(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__repr__()
    assert var_3 == 'pmap({pmap({}): pmap({})})'
    var_4 = var_0.__eq__(var_2)
    var_5 = var_1.set(var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    module_0.pmap(pre_size=var_5)

def test_case_10():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_11():
    var_0 = None
    var_1 = '2jt_g\x0c`'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.set(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    var_5 = module_0.PMapView(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_5) == 1

def test_case_12():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__iter__()

def test_case_13():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    var_3 = var_1.set(var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    with pytest.raises(TypeError):
        var_1.__reversed__()

def test_case_14():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__add__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__len__()
    assert var_2 == 0

def test_case_15():
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
def test_case_16():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__reduce__()
    var_1.evolver()

def test_case_17():
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

def test_case_18():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    module_0.pmap(var_0, var_0)

def test_case_20():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.values()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_2) == 0
    var_3 = var_0.__str__()
    assert var_3 == 'pmap({})'
    var_4 = var_1.remove(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0

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

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.evolver()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_1) == 0
    var_2 = var_1.__len__()
    assert var_2 == 0
    var_3 = var_0.__str__()
    assert var_3 == 'pmap({})'
    var_4 = var_0.__str__()
    assert var_4 == 'pmap({})'
    var_2.__reduce__()

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

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_2) == 0
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_3.__iter__()

def test_case_25():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = module_0.PMapView(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_6) == 2
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.PMapView(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_9) == 2
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 3
    var_12 = [var_0, var_7, var_11]
    with pytest.raises(TypeError):
        module_0.PMapView(var_12)

def test_case_26():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_1.keys()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_2) == 0
    var_3 = var_0.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = var_0.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_4) == 0
    with pytest.raises(TypeError):
        var_4.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.__contains__(var_1)
    assert var_2 is False
    var_3 = var_0.__str__()
    assert var_3 == 'pmap({})'
    var_4 = var_1.set(var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    var_5 = var_4.__add__(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    var_6 = var_1.set(var_4, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = None
    var_3.set(var_7, var_4)

@pytest.mark.xfail(strict=True)
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
    var_2.remove(var_0)

def test_case_29():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = {var_0: var_2, var_1: var_0}
    var_4 = module_0.pmap(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = module_0.PMapValues(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_5) == 2
    var_6 = repr(var_5)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = module_0.PMapValues(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_4) == 0
    var_5 = var_3.__eq__(var_0)
    var_6 = var_2.set(var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_2.transform(*var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_4.__contains__(var_6)
    assert var_8 is False
    var_7.remove(var_2)

@pytest.mark.xfail(strict=True)
def test_case_31():
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
    var_4 = var_3.__contains__(var_1)
    assert var_4 is False
    var_5 = var_2.__len__()
    assert var_5 == 0
    var_6 = var_2.__str__()
    assert var_6 == 'pmap({})'
    var_2.remove(var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = var_1.copy()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.evolver()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_4) == 0
    var_5 = var_1.__contains__(var_0)
    assert var_5 is False
    var_6 = var_3.set(var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_1.discard(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = module_0.PMapItems(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_8) == 0
    var_9 = var_1.__eq__(var_6)
    assert var_9 is False
    var_10 = var_6.__len__()
    assert var_10 == 1
    var_11 = var_10.__eq__(var_0)
    var_12 = var_8.__eq__(var_0)
    assert var_12 is False
    var_13 = var_3.iteritems()
    var_14 = var_11.__str__()
    assert var_14 == 'NotImplemented'
    var_1.remove(var_11)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = var_3.__iter__()
    var_5 = var_1.__contains__(var_0)
    assert var_5 is False
    var_6 = var_2.set(var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = [var_0, var_0, var_0, var_0]
    var_8 = var_2.transform(*var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_2.__contains__(var_6)
    assert var_9 is False
    var_10 = var_1.discard(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = module_0.PMapItems(var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_11) == 1
    var_12 = var_1.__eq__(var_6)
    assert var_12 is False
    var_13 = var_5.__repr__()
    assert var_13 == 'False'
    var_14 = var_13.__eq__(var_0)
    var_15 = var_11.__str__()
    assert var_15 == 'pmap_items([(pmap({}), pmap({}))])'
    var_16 = var_10.set(var_9, var_7)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 1
    var_13.remove(var_5)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_2.update(*var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.evolver()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_4) == 0
    var_5 = var_2.set(var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_3.__len__()
    assert var_6 == 0
    var_7 = var_2.transform(*var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_2.__contains__(var_5)
    assert var_8 is False
    var_9 = var_1.discard(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_2.__contains__(var_5)
    assert var_10 is False
    var_11 = module_0.PMapItems(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_11) == 0
    var_12 = var_5.__len__()
    assert var_12 == 1
    var_13 = var_5.remove(var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 0
    var_14 = var_12.__eq__(var_2)
    var_15 = var_2.__eq__(var_13)
    assert var_15 is True
    var_16 = module_1.python_pvector()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_16) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_1.BRANCH_FACTOR == 32
    assert module_1.BIT_MASK == 31
    assert module_1.SHIFT == 5
    var_12.set(var_0, var_14)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__iter__()
    var_3 = var_0.__contains__(var_1)
    assert var_3 is False
    var_4 = var_0.__eq__(var_2)
    var_5 = var_1.set(var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_1.__contains__(var_5)
    assert var_6 is False
    var_7 = var_0.__eq__(var_0)
    assert var_7 is True
    var_8 = var_0.discard(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = module_0.PMapItems(var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_9) == 0
    var_10 = var_0.__eq__(var_5)
    assert var_10 is False
    var_11 = var_9.__eq__(var_6)
    assert var_11 is False
    var_12 = module_0.pmap(var_5, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = var_12.set(var_3, var_10)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 2
    var_14 = var_9.__contains__(var_13)
    assert var_14 is False
    var_15 = var_8.set(var_12, var_12)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 1
    var_16 = var_15.set(var_5, var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 1
    var_17 = var_12.__add__(var_5)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 1
    var_18 = var_17.set(var_5, var_4)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 2
    var_13.remove(var_7)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = module_0.PMapValues(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_4) == 0
    var_5 = var_3.__eq__(var_0)
    var_6 = var_2.set(var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_2.transform(*var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_4.__eq__(var_0)
    assert var_8 is False
    var_9 = var_8.__repr__()
    assert var_9 == 'False'
    var_10 = var_9.__str__()
    assert var_10 == 'False'
    var_8.set(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.update_with(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_2.__iter__()
    var_4 = var_1.__reduce__()
    var_5 = module_0.pmap(pre_size=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_1.transform(*var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_1.__contains__(var_5)
    assert var_7 is False
    var_8 = var_0.discard(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = module_0.PMapItems(var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_9) == 0
    var_10 = var_0.__eq__(var_5)
    assert var_10 is True
    var_11 = var_9.__repr__()
    assert var_11 == 'pmap_items([])'
    var_4.update_with(var_8)

def test_case_38():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.set(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1

def test_case_39():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = module_0.PMapValues(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_2) == 0
    var_3 = var_1.set(var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    with pytest.raises(TypeError):
        var_2.__setattr__(var_3, var_0)

def test_case_40():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 3
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_8 = module_0.PMapValues(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_8) == 3
    var_9 = list(var_8)
    var_10 = {}
    var_11 = module_0.pmap(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 0
    var_12 = module_0.PMapValues(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_12) == 0
    var_13 = list(var_12)
    var_14 = {var_0: var_3, var_1: var_3, var_2: var_4}
    var_15 = module_0.pmap(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 3
    var_16 = module_0.PMapValues(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_16) == 3
    var_17 = list(var_16)
    var_18 = {var_0: var_3, var_1: var_4}
    var_19 = module_0.pmap(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_19) == 2
    var_20 = module_0.PMapValues(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_20) == 2
    var_21 = iter(var_20)
    var_22 = next(var_21)
    var_23 = list(var_21)
    var_24 = 1000
    var_25 = range(var_24)
    with pytest.raises(NameError):
        var_26 = {i: i * var_4 for i in var_25}

def test_case_41():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = module_0.PMapValues(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_6) == 2
    var_7 = str(var_6)
    assert var_7 == 'pmap_values([2, 4])'

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_1.update(*var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.evolver()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_3) == 0
    var_4 = var_1.set(var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_0.discard(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = module_0.PMapItems(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 0
    var_7 = var_4.__len__()
    assert var_7 == 1
    var_8 = (var_0, var_5, var_7, var_5)
    var_9 = var_2.__eq__(var_8)
    var_10 = module_0.m()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = var_6.__eq__(var_7)
    assert var_11 is False
    var_12 = var_4.discard(var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 0
    var_13 = var_0.__eq__(var_2)
    assert var_13 is True
    var_14 = var_0.__iter__()
    var_15 = var_3.set(var_8, var_8)
    assert len(var_3) == 1
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_15) == 1
    var_16 = var_6.__contains__(var_7)
    assert var_16 is False
    var_17 = var_5.set(var_8, var_8)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 1
    var_13.set(var_2, var_15)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_2.update(*var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_3.__iter__()
    var_5 = var_1.evolver()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_5) == 0
    var_6 = var_1.__contains__(var_0)
    assert var_6 is False
    var_7 = var_2.set(var_1, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = [var_0, var_0, var_0, var_0]
    var_9 = var_2.transform(*var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_2.__contains__(var_7)
    assert var_10 is False
    var_2.discard(var_8)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__iter__()
    var_3 = var_0.__contains__(var_1)
    assert var_3 is False
    var_4 = None
    var_5 = var_0.__eq__(var_4)
    var_6 = var_1.set(var_0, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_1.transform(*var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_1.__contains__(var_6)
    assert var_8 is False
    var_9 = var_8.__repr__()
    assert var_9 == 'False'
    var_10 = var_0.discard(var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = module_0.PMapItems(var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_11) == 0
    var_12 = var_0.__eq__(var_6)
    assert var_12 is False
    var_13 = var_6.__len__()
    assert var_13 == 1
    var_14 = var_11.__eq__(var_8)
    assert var_14 is False
    var_15 = var_6.discard(var_10)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_16 = module_0.pmap(var_6, var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 1
    var_17 = var_15.__eq__(var_7)
    assert var_17 is True
    var_18 = var_10.__hash__()
    assert var_18 == 133146708735736
    var_19 = var_7.__repr__()
    assert var_19 == 'pmap({})'
    var_20 = var_0.__eq__(var_15)
    assert var_20 is True
    var_21 = var_16.set(var_7, var_12)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_21) == 1
    var_22 = var_11.__contains__(var_13)
    assert var_22 is False
    var_23 = var_10.set(var_16, var_16)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_23) == 1
    var_24 = var_20.__eq__(var_20)
    assert var_24 is True
    var_25 = var_23.set(var_6, var_15)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_25) == 1
    var_26 = var_16.__add__(var_6)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_26) == 1
    var_27 = var_16.set(var_8, var_1)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_27) == 2
    var_22.set(var_20, var_25)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.keys()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_0.set(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.__contains__(var_3)
    assert var_4 is False
    var_5 = var_3.__eq__(var_2)
    assert var_5 is False
    var_4.discard(var_4)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__iter__()
    var_3 = var_0.__contains__(var_1)
    assert var_3 is False
    var_4 = var_0.__eq__(var_2)
    var_5 = var_1.set(var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_1.__contains__(var_5)
    assert var_6 is False
    var_7 = var_0.__eq__(var_0)
    assert var_7 is True
    var_8 = var_0.discard(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = module_0.PMapItems(var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_9) == 0
    var_10 = var_5.__len__()
    assert var_10 == 1
    var_11 = var_9.__eq__(var_6)
    assert var_11 is False
    var_12 = var_5.discard(var_8)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 0
    var_13 = module_0.pmap(var_5, var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1
    var_14 = var_8.__hash__()
    assert var_14 == 133146708735736
    var_15 = var_11.__repr__()
    assert var_15 == 'False'
    var_16 = module_0.PMapValues(var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_16) == 0
    var_17 = var_13.set(var_3, var_14)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 2
    var_18 = var_9.__contains__(var_10)
    assert var_18 is False
    var_19 = var_8.set(var_13, var_13)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_19) == 1
    var_20 = var_16.__eq__(var_16)
    assert var_20 is True
    var_21 = var_19.set(var_5, var_12)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_21) == 1
    var_22 = var_13.__add__(var_5)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_22) == 1
    var_23 = None
    var_10.remove(var_23)

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__iter__()
    var_3 = var_0.__contains__(var_1)
    assert var_3 is False
    var_4 = var_0.__eq__(var_2)
    var_5 = var_1.set(var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_1.__contains__(var_5)
    assert var_6 is False
    var_7 = var_0.__eq__(var_0)
    assert var_7 is True
    var_8 = var_0.discard(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = module_0.PMapItems(var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_9) == 0
    var_10 = var_5.__len__()
    assert var_10 == 1
    var_11 = var_9.__eq__(var_6)
    assert var_11 is False
    var_12 = var_5.discard(var_8)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 0
    var_13 = module_0.pmap(var_5, var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1
    var_14 = var_8.__hash__()
    assert var_14 == 133146708735736
    var_15 = var_11.__repr__()
    assert var_15 == 'False'
    var_16 = module_0.PMapValues(var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_16) == 0
    var_17 = var_13.set(var_3, var_14)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 2
    var_18 = var_9.__contains__(var_10)
    assert var_18 is False
    var_19 = var_17.evolver()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_19) == 2
    var_20 = var_19.__contains__(var_4)
    assert var_20 is False
    var_21 = var_19.set(var_1, var_3)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_21) == 2
    var_22 = var_16.__eq__(var_13)
    assert var_22 is False
    var_8.set(var_16, var_15)

def test_case_48():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__iter__()
    var_3 = var_0.__contains__(var_1)
    assert var_3 is False
    var_4 = var_0.__eq__(var_2)
    var_5 = var_1.set(var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_1.__contains__(var_5)
    assert var_6 is False
    var_7 = var_0.__eq__(var_0)
    assert var_7 is True
    var_8 = var_0.discard(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = module_0.PMapItems(var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_9) == 0
    var_10 = var_0.update()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = var_0.__eq__(var_5)
    assert var_11 is False
    var_12 = var_5.__len__()
    assert var_12 == 1
    var_13 = var_9.__eq__(var_6)
    assert var_13 is False
    var_14 = var_5.discard(var_8)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 0
    var_15 = module_0.pmap(var_5, var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 1
    var_16 = var_8.__hash__()
    assert var_16 == 133146708735736
    var_17 = var_11.__repr__()
    assert var_17 == 'False'
    var_18 = module_0.PMapValues(var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_18) == 0
    var_19 = var_5.__reduce__()
    var_20 = var_15.set(var_3, var_11)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_20) == 2
    var_21 = var_9.__contains__(var_12)
    assert var_21 is False
    var_22 = var_9.__contains__(var_20)
    assert var_22 is False
    var_23 = var_8.set(var_15, var_15)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_23) == 1
    var_24 = var_18.__eq__(var_18)
    assert var_24 is True
    var_25 = None
    var_26 = var_11.__add__(var_25)
    var_27 = var_20.set(var_3, var_22)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_27) == 2
    var_28 = module_0.PMap(*var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_28) == 0
    var_29 = var_27.remove(var_11)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_29) == 1

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__iter__()
    var_3 = var_0.__contains__(var_1)
    assert var_3 is False
    var_4 = var_0.__eq__(var_2)
    var_5 = var_1.set(var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_0.__eq__(var_0)
    assert var_6 is True
    var_7 = var_0.discard(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = module_0.PMapItems(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_8) == 0
    var_9 = var_0.update()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_0.__eq__(var_5)
    assert var_10 is False
    var_11 = var_5.__len__()
    assert var_11 == 1
    var_12 = var_8.__eq__(var_8)
    assert var_12 is True
    var_13 = var_5.discard(var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 0
    var_14 = module_0.pmap(var_5, var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 1
    var_15 = var_7.__hash__()
    assert var_15 == 133146708735736
    var_16 = var_10.__repr__()
    assert var_16 == 'False'
    var_17 = module_0.PMapValues(var_0)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_17) == 0
    var_18 = var_5.__reduce__()
    var_19 = var_14.set(var_3, var_10)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_19) == 2
    var_20 = var_8.__contains__(var_11)
    assert var_20 is False
    var_21 = var_8.__contains__(var_19)
    assert var_21 is False
    var_22 = var_10.__str__()
    assert var_22 == 'False'
    var_23 = var_22.__eq__(var_16)
    assert var_23 is True
    var_24 = var_19.set(var_15, var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_24) == 3
    var_25 = var_6.__add__(var_12)
    assert var_25 == 2
    var_26 = var_24.set(var_22, var_12)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_26) == 4
    var_14.remove(var_5)

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__iter__()
    var_3 = var_0.__contains__(var_1)
    assert var_3 is False
    var_4 = module_0.pmap(var_2, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_1.transform(*var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = module_0.PMapItems(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 0
    var_7 = var_0.discard(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = module_0.PMapItems(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_8) == 0
    var_9 = var_8.__eq__(var_6)
    assert var_9 is True
    var_10 = var_4.discard(var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = module_0.pmap(var_4, var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 0
    var_12 = var_7.__hash__()
    assert var_12 == 133146708735736
    var_13 = var_0.__eq__(var_10)
    assert var_13 is True
    var_14 = var_11.set(var_5, var_8)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 1
    var_15 = module_0.pmap(var_5)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_16 = var_5.__hash__()
    assert var_16 == 133146708735736
    module_0.PMapValues(var_13)

@pytest.mark.xfail(strict=True)
def test_case_51():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__iter__()
    var_3 = var_0.__contains__(var_1)
    assert var_3 is False
    var_4 = var_0.__eq__(var_2)
    var_5 = var_1.set(var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_0.__eq__(var_0)
    assert var_6 is True
    var_7 = var_0.discard(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = module_0.PMapItems(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_8) == 0
    var_9 = var_0.update()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_0.__eq__(var_5)
    assert var_10 is False
    var_11 = var_5.keys()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_11) == 1
    var_12 = var_11.discard(var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_12) == 0
    module_0.PMapItems(var_6)

@pytest.mark.xfail(strict=True)
def test_case_52():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__iter__()
    var_3 = var_0.__contains__(var_1)
    assert var_3 is False
    var_4 = var_0.__eq__(var_2)
    var_5 = var_1.set(var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_0.__eq__(var_0)
    assert var_6 is True
    var_7 = var_0.discard(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = module_0.PMapItems(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_8) == 0
    var_9 = var_0.update()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = var_0.__eq__(var_5)
    assert var_10 is False
    var_11 = var_5.__len__()
    assert var_11 == 1
    var_12 = var_8.__eq__(var_8)
    assert var_12 is True
    var_13 = var_5.discard(var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 0
    var_14 = var_1.update()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 0
    var_15 = module_0.pmap(var_5, var_12)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 1
    var_16 = var_7.__hash__()
    assert var_16 == 133146708735736
    var_17 = var_10.__repr__()
    assert var_17 == 'False'
    var_18 = module_0.PMapValues(var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_18) == 0
    var_19 = var_5.__reduce__()
    var_20 = var_15.set(var_3, var_10)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_20) == 2
    var_21 = var_1.__contains__(var_1)
    assert var_21 is False
    var_22 = var_14.set(var_11, var_12)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_22) == 1
    var_23 = None
    var_24 = var_7.__eq__(var_23)
    module_2.transform(var_7, var_17)

@pytest.mark.xfail(strict=True)
def test_case_53():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.copy()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__contains__(var_1)
    assert var_2 is False
    var_3 = var_0.__eq__(var_2)
    var_4 = var_1.set(var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_0.__eq__(var_0)
    assert var_5 is True
    var_6 = var_0.discard(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = module_0.PMapItems(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 0
    var_8 = var_0.update()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_0.__eq__(var_4)
    assert var_9 is False
    var_10 = var_4.__len__()
    assert var_10 == 1
    var_11 = var_7.__eq__(var_7)
    assert var_11 is True
    var_12 = var_4.discard(var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 0
    var_13 = module_0.pmap(var_4, var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1
    var_14 = var_6.__hash__()
    assert var_14 == 133146708735736
    var_15 = var_9.__repr__()
    assert var_15 == 'False'
    var_16 = module_0.PMapValues(var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_16) == 0
    var_17 = var_4.__reduce__()
    var_18 = var_13.set(var_2, var_9)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 2
    var_19 = var_0.__repr__()
    assert var_19 == 'pmap({})'
    var_20 = var_7.__contains__(var_10)
    assert var_20 is False
    var_21 = var_7.__contains__(var_18)
    assert var_21 is False
    var_22 = var_6.set(var_13, var_13)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_22) == 1
    var_23 = var_16.__eq__(var_16)
    assert var_23 is True
    var_24 = var_22.set(var_4, var_12)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_24) == 1
    var_25 = var_13.__add__(var_4)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_25) == 1
    var_0.__getattr__(var_17)