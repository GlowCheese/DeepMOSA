# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._pmap as module_0
import pyrsistent._transformations as module_1
import pyrsistent._pvector as module_2

def test_case_0():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = module_0.pmap(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.__str__()
    assert var_3 == 'pmap({})'

def test_case_1():
    var_0 = None
    with pytest.raises(TypeError):
        module_0.PMapView(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.update_with(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_2) == 0
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_2.evolver()

def test_case_3():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.discard(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1

def test_case_4():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__contains__(var_0)
    assert var_1 is False

def test_case_5():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__repr__()
    assert var_1 == 'pmap({})'

def test_case_6():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_0.remove(var_0)

def test_case_8():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.update()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

def test_case_9():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.iterkeys()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.discard(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_2.__contains__(var_2)
    assert var_4 is False
    var_5 = var_1.keys()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_5) == 1
    module_0.pmap(pre_size=var_5)

def test_case_11():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 2
    module_0.pmap(var_0)

def test_case_13():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_0.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.__reduce__()
    var_4 = var_1.iterkeys()
    var_4.__len__()

def test_case_15():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_16():
    var_0 = None
    var_1 = 'S'
    var_2 = 'x-">SU_!~Rki'
    var_3 = {var_1: var_0, var_2: var_0, var_1: var_0}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.__contains__(var_0)
    assert var_5 is False
    with pytest.raises(TypeError):
        module_0.PMapView(var_0)

def test_case_17():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1 in var_1

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.evolver()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_2) == 0
    var_3 = var_1.__repr__()
    assert var_3 == 'pmap({})'
    var_4 = var_3.__hash__()
    assert var_4 == 4327297822388814452
    var_5 = module_1.transform(var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = module_1.transform(var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_6) == 0
    var_7 = var_5.discard(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_4.__iter__()

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
    var_3 = var_2.values()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 1
    var_4 = module_0.pmap()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_1.__str__()
    assert var_5 == 'pmap({})'

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_0.__contains__(var_0)
    assert var_3 is False
    var_4 = var_2.discard(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_2.__hash__()
    assert var_5 == -4783114350154387147
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = var_0.__add__(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_4.__add__(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_8.iterkeys()
    var_10 = var_4.__contains__(var_5)
    assert var_10 is False
    var_11 = var_2.keys()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_11) == 1
    var_12 = var_2.__len__()
    assert var_12 == 1
    var_13 = var_8.__eq__(var_1)
    var_14 = var_7.transform()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 0
    var_15 = module_0.pmap(pre_size=var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_13.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = 'x"\x0bL;y^A@uA#|Gh'
    var_2 = 'NXR*YA3xj2ZRXJa'
    var_3 = {var_1: var_0, var_1: var_0, var_2: var_0, var_2: var_0}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.__iter__()
    var_5.keys()

def test_case_22():
    var_0 = {}
    var_1 = module_0.PMapValues(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_2.python_pvector(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 0
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_2.BRANCH_FACTOR == 32
    assert module_2.BIT_MASK == 31
    assert module_2.SHIFT == 5

def test_case_23():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__hash__()
    assert var_2 == 133146708735736
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_3) == 0
    with pytest.raises(TypeError):
        var_3.__setattr__(var_2, var_0)

def test_case_24():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = {}
    var_2 = var_0.__eq__(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_0.__contains__(var_0)
    assert var_3 is False
    var_4 = var_2.discard(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_2.__hash__()
    assert var_5 == -4783114350154387147
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = var_0.__add__(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_5.set(var_6, var_5)

def test_case_26():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__eq__(var_0)
    var_3 = module_0.PMapValues(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 0
    var_4 = var_1.__contains__(var_0)
    assert var_4 is False
    with pytest.raises(TypeError):
        var_3.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.__hash__()
    assert var_5 == -4783114350154387147
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = var_0.__add__(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_3.__add__(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_0.__eq__(var_4)
    assert var_9 is True
    var_10 = var_2.__str__()
    assert var_10 == 'pmap({pmap({}): pmap({})})'
    var_11 = var_4.__eq__(var_7)
    assert var_11 is True
    var_12 = var_7.__str__()
    assert var_12 == 'pmap({})'
    var_13 = var_1.__eq__(var_3)
    assert var_13 is False
    var_14 = var_2.__eq__(var_6)
    var_15 = var_7.discard(var_10)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_16 = var_5.__add__(var_7)
    var_17 = var_7.__len__()
    assert var_17 == 0
    var_0.__lt__(var_7)

def test_case_28():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_0.update_with(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.__contains__(var_2)
    assert var_4 is True
    var_5 = var_2.copy()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_2.__contains__(var_5)
    assert var_6 is False

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.__hash__()
    assert var_5 == -4783114350154387147
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = var_0.__add__(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_3.__add__(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_2.__reduce__()
    var_10 = var_0.__eq__(var_9)
    var_11 = var_4.__str__()
    assert var_11 == 'pmap({})'
    var_12 = var_11.__eq__(var_7)
    var_13 = var_1.__eq__(var_11)
    assert var_13 is False
    var_14 = var_7.discard(var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 0
    var_15 = var_7.__eq__(var_4)
    assert var_15 is True
    var_16 = var_3.__contains__(var_6)
    assert var_16 is False
    var_17 = var_14.__reduce__()
    var_18 = var_5.__str__()
    assert var_18 == '-4783114350154387147'
    var_19 = module_1.transform(var_0, var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_19) == 7
    module_0.pmap(pre_size=var_18)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.__iter__()
    var_6 = var_0.update_with(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_4.iteritems()
    var_8 = var_4.update()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_2.__contains__(var_4)
    assert var_9 is True
    var_10 = var_4.copy()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = var_1.__contains__(var_10)
    assert var_11 is False
    var_12 = var_3.__add__(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = {}
    var_14 = var_0.__eq__(var_13)
    assert var_14 is True
    var_15 = module_0.PMapValues(var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_15) == 1
    var_16 = var_15.__eq__(var_5)
    assert var_16 is False
    var_17 = var_1.__eq__(var_15)
    assert var_17 is False
    var_18 = var_3.discard(var_16)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 1
    var_19 = var_14.__add__(var_5)
    var_20 = var_4.transform()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_20) == 0
    var_21 = var_5.__str__()
    var_22 = module_1.transform(var_0, var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'pyrsistent._pmap.PMap'
    module_0.pmap(var_19)

def test_case_31():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.__contains__(var_0)
    assert var_2 is False
    var_3 = var_1.__hash__()
    assert var_3 == -4783114350154387147
    var_4 = var_1.__eq__(var_0)
    assert var_4 is False
    var_5 = var_0.__add__(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_0.__len__()
    assert var_6 == 0
    var_7 = var_1.keys()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_7) == 1
    var_8 = var_1.__len__()
    assert var_8 == 1
    var_9 = var_7.remove(var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_9) == 0
    var_10 = var_0.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__eq__(var_0)
    var_3 = module_0.PMapValues(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 0
    var_4 = var_1.__contains__(var_0)
    assert var_4 is False
    var_5 = var_3.__str__()
    assert var_5 == 'pmap_values([])'
    module_0.PMapItems(var_2)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.__iter__()
    var_6 = var_0.update_with(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_0.__add__(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_2.__contains__(var_4)
    assert var_8 is True
    var_9 = var_3.iterkeys()
    var_10 = var_3.__reduce__()
    var_11 = var_1.__contains__(var_10)
    assert var_11 is False
    var_12 = var_3.__add__(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = {}
    var_14 = var_0.__eq__(var_13)
    assert var_14 is True
    var_15 = module_0.m()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_16 = module_0.PMapValues(var_3)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_16) == 1
    var_17 = var_16.__contains__(var_10)
    assert var_17 is False
    var_18 = var_16.__eq__(var_7)
    assert var_18 is False
    var_19 = var_1.__eq__(var_16)
    assert var_19 is False
    var_20 = var_7.discard(var_18)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_20) == 0
    var_21 = var_14.__add__(var_5)
    var_22 = var_1.__contains__(var_17)
    assert var_22 is False
    var_23 = var_4.transform()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_23) == 0
    var_24 = None
    var_25 = var_12.set(var_24, var_19)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_25) == 2
    var_26 = var_20.__reduce__()
    var_27 = var_5.__str__()
    var_28 = module_1.transform(var_0, var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'pyrsistent._pmap.PMap'
    module_0.pmap(var_21)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__repr__()
    assert var_2 == 'pmap({})'
    var_3 = var_1.__iter__()
    var_4 = var_1.copy()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = module_0.PMapValues(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_5) == 0
    var_6 = var_5.__contains__(var_4)
    assert var_6 is False
    var_7 = var_5.__repr__()
    assert var_7 == 'pmap_values([])'
    var_8 = var_4.update()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = None
    var_10 = None
    var_11 = module_0.pmap(pre_size=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 0
    var_12 = var_11.discard(var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 0
    var_13 = var_12.__contains__(var_11)
    assert var_13 is False
    var_14 = var_12.__len__()
    assert var_14 == 0
    var_15 = var_2.__str__()
    assert var_15 == 'pmap({})'
    var_16 = var_11.transform(*var_12)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 0
    var_17 = var_12.__add__(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 0
    var_18 = var_14.__eq__(var_0)
    var_19 = var_1.transform()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_19) == 0
    var_20 = module_0.PMapValues(var_17)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_20) == 0
    var_6.update()

def test_case_35():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_0.__add__(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_4.__len__()
    assert var_5 == 0
    var_6 = var_3.iterkeys()
    var_7 = var_3.__reduce__()
    var_8 = var_1.__contains__(var_7)
    assert var_8 is False
    var_9 = var_0.__eq__(var_2)
    assert var_9 is False
    var_10 = module_0.PMapValues(var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_10) == 1
    var_11 = var_8.__eq__(var_3)
    var_12 = var_10.__eq__(var_4)
    assert var_12 is False
    var_13 = var_4.discard(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 0
    var_14 = var_3.__str__()
    assert var_14 == 'pmap({pmap({}): pmap({})})'
    var_15 = module_1.transform(var_0, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 5
    var_16 = module_0.pmap(var_13)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 0
    var_17 = var_9.__lt__(var_3)

def test_case_36():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.__iter__()
    var_6 = var_0.__add__(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_4.iteritems()
    var_8 = var_3.iterkeys()
    var_9 = var_3.__reduce__()
    var_10 = var_1.__contains__(var_9)
    assert var_10 is False
    var_11 = var_0.__eq__(var_2)
    assert var_11 is False
    var_12 = module_0.PMapValues(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_12) == 1
    var_13 = var_10.__eq__(var_3)
    var_14 = var_12.__eq__(var_6)
    assert var_14 is False
    var_15 = var_6.discard(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_16 = var_5.__str__()
    var_17 = module_1.transform(var_0, var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 17
    var_18 = module_0.pmap(var_4)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 0

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_0.__add__(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_1.__str__()
    assert var_5 == 'pmap({pmap({}): pmap({})})'
    var_6 = var_4.__eq__(var_3)
    var_7 = var_2.__eq__(var_4)
    assert var_7 is True
    var_8 = var_6.__eq__(var_5)
    var_9 = var_2.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_9) == 0
    var_8.__add__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_2.__iter__()
    var_5 = var_0.update_with(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_3.iteritems()
    var_7 = var_2.__contains__(var_3)
    assert var_7 is False
    var_8 = module_0.pmap()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_0.update_with(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = {}
    var_11 = var_0.__eq__(var_10)
    assert var_11 is True
    var_12 = var_1.__eq__(var_2)
    assert var_12 is False
    var_13 = var_8.discard(var_8)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 0
    var_14 = var_11.__add__(var_4)
    var_15 = var_13.transform()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_16 = var_13.__reduce__()
    var_17 = var_4.__str__()
    module_0.pmap(var_14)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.__len__()
    assert var_2 == 0
    var_3 = var_2.__hash__()
    assert var_3 == 0
    var_4 = var_2.__eq__(var_0)
    var_5 = var_0.__add__(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = {}
    var_7 = var_0.__eq__(var_6)
    assert var_7 is True
    var_8 = var_1.__str__()
    assert var_8 == 'pmap_items([])'
    var_9 = var_8.__eq__(var_5)
    var_10 = var_7.__add__(var_3)
    assert var_10 == 1
    module_1.transform(var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_0.__add__(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = module_0.m(**var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_0.__eq__(var_2)
    assert var_6 is True
    var_7 = var_1.__str__()
    assert var_7 == 'pmap({pmap({}): pmap({})})'
    var_8 = var_4.__eq__(var_3)
    var_9 = var_2.__eq__(var_4)
    assert var_9 is True
    var_10 = var_8.__eq__(var_7)
    var_11 = var_2.items()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_11) == 0
    var_5.__add__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_1.__len__()
    assert var_4 == 0
    var_5 = var_3.iterkeys()
    var_6 = var_3.__reduce__()
    var_7 = var_1.__contains__(var_6)
    assert var_7 is False
    var_8 = var_0.__eq__(var_2)
    assert var_8 is False
    var_9 = module_0.PMapValues(var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_9) == 1
    var_10 = var_7.__eq__(var_3)
    var_11 = var_9.__eq__(var_7)
    assert var_11 is False
    var_10.discard(var_11)

def test_case_42():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_0 in var_1

def test_case_43():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_1.__repr__()
    assert var_4 == 'pmap_items([])'
    var_5 = var_3.discard(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_2.__hash__()
    assert var_6 == -4783114350154387147
    var_7 = var_2.__eq__(var_0)
    assert var_7 is False
    var_8 = var_0.__add__(var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_3.__add__(var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 1
    var_10 = {}
    var_11 = var_0.__eq__(var_10)
    assert var_11 is True
    var_12 = var_5.__str__()
    assert var_12 == 'pmap({})'
    var_13 = var_11.__str__()
    assert var_13 == 'True'
    var_14 = var_2.__eq__(var_8)
    assert var_14 is False
    var_15 = var_3.__iter__()
    var_16 = var_11.__eq__(var_9)
    with pytest.raises(TypeError):
        var_5.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.__hash__()
    assert var_5 == -4783114350154387147
    var_6 = var_3.__eq__(var_5)
    var_7 = var_0.__add__(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_3.__add__(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = {}
    var_10 = var_0.__eq__(var_9)
    assert var_10 is True
    var_11 = var_4.__str__()
    assert var_11 == 'pmap({})'
    var_12 = module_0.PMapValues(var_7)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_12) == 0
    var_13 = var_1.__eq__(var_11)
    assert var_13 is False
    var_7.discard(var_12)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_0.update_with(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_0.__add__(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_4.iteritems()
    var_8 = var_2.__contains__(var_4)
    assert var_8 is True
    var_9 = var_3.iterkeys()
    var_10 = var_3.__reduce__()
    var_11 = var_1.__contains__(var_10)
    assert var_11 is False
    var_12 = {}
    var_13 = var_0.__eq__(var_12)
    assert var_13 is True
    var_14 = module_0.PMapValues(var_3)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_14) == 1
    var_15 = var_5.__eq__(var_3)
    assert var_15 is False
    var_16 = var_1.__eq__(var_14)
    assert var_16 is False
    var_17 = var_6.discard(var_9)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 0
    var_18 = var_13.__add__(var_4)
    var_19 = var_17.__reduce__()
    var_20 = var_19.__str__()
    assert var_20 == '(<function pmap at 0x7892bb3b5870>, ({},))'
    var_21 = module_1.transform(var_0, var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_21) == 17
    module_0.pmap(var_18)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.__iter__()
    var_6 = var_0.update_with(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_0.__add__(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_4.iteritems()
    var_9 = var_2.__contains__(var_4)
    assert var_9 is True
    var_10 = var_4.copy()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = var_1.__contains__(var_10)
    assert var_11 is False
    var_12 = var_3.__add__(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = {}
    var_14 = var_0.__eq__(var_13)
    assert var_14 is True
    var_15 = module_0.PMapValues(var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_15) == 1
    var_16 = var_3.set(var_3, var_8)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 2
    var_17 = var_6.__eq__(var_3)
    assert var_17 is False
    var_18 = var_15.__eq__(var_7)
    assert var_18 is False
    var_19 = var_1.__eq__(var_15)
    assert var_19 is False
    var_20 = var_7.discard(var_18)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_20) == 0
    var_21 = var_14.__add__(var_5)
    var_22 = var_4.transform()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_22) == 0
    var_23 = var_20.__reduce__()
    var_24 = var_20.update()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_24) == 0
    var_4.__getattr__(var_23)

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.__iter__()
    var_6 = var_4.iteritems()
    var_7 = var_3.iterkeys()
    var_8 = var_3.__reduce__()
    var_9 = var_1.__contains__(var_8)
    assert var_9 is False
    var_10 = var_3.__add__(var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = {}
    var_12 = var_0.__eq__(var_11)
    assert var_12 is True
    var_13 = module_0.PMapValues(var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_13) == 1
    var_14 = var_9.__eq__(var_3)
    var_15 = var_13.__eq__(var_13)
    assert var_15 is True
    var_8.discard(var_15)

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.__iter__()
    var_6 = var_0.update_with(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_0.__add__(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_4.iteritems()
    var_9 = var_2.__contains__(var_4)
    assert var_9 is True
    var_10 = var_4.copy()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = var_1.__contains__(var_10)
    assert var_11 is False
    var_12 = var_3.__add__(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = {}
    var_14 = var_0.__eq__(var_13)
    assert var_14 is True
    var_15 = var_1.__iter__()
    var_16 = var_7.items()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_16) == 0
    var_17 = var_16.__eq__(var_7)
    assert var_17 is False
    var_18 = var_1.__eq__(var_16)
    assert var_18 is True
    var_19 = var_7.discard(var_17)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_19) == 0
    var_20 = var_14.__add__(var_5)
    var_21 = var_4.transform()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_21) == 0
    var_22 = var_19.__reduce__()
    var_23 = var_5.__str__()
    var_24 = module_1.transform(var_0, var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'pyrsistent._pmap.PMap'
    module_0.pmap(var_20)

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_2.__iter__()
    var_6 = var_0.update_with(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_0.__add__(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_4.iteritems()
    var_9 = var_2.__contains__(var_4)
    assert var_9 is True
    var_10 = var_4.copy()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = var_1.__contains__(var_10)
    assert var_11 is False
    var_12 = var_3.__add__(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = {}
    var_14 = var_0.__eq__(var_13)
    assert var_14 is True
    var_15 = module_0.PMapValues(var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_15) == 1
    var_16 = var_6.__eq__(var_3)
    assert var_16 is False
    var_17 = var_10.set(var_7, var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 1
    var_18 = var_17.__eq__(var_3)
    assert var_18 is False
    var_19 = var_8.__eq__(var_0)
    var_19.discard(var_14)

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_3.transform()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_2.__iter__()
    var_7 = var_0.update_with(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_0.__add__(var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_2.__contains__(var_4)
    assert var_9 is True
    var_10 = var_0.__eq__(var_7)
    assert var_10 is True
    var_11 = var_3.__reduce__()
    var_12 = var_1.__contains__(var_11)
    assert var_12 is False
    var_13 = var_3.__add__(var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1
    var_14 = {}
    var_15 = var_0.__eq__(var_14)
    assert var_15 is True
    var_16 = var_3.copy()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 1
    var_17 = module_0.m()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 0
    var_18 = False
    var_19 = var_16.set(var_18, var_15)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_19) == 2
    var_20 = var_19.__eq__(var_6)
    var_21 = var_1.__eq__(var_0)
    assert var_21 is False
    var_22 = var_8.__eq__(var_18)
    var_23 = var_19.discard(var_8)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_23) == 1
    var_24 = var_12.__str__()
    assert var_24 == 'False'
    var_24.iterkeys()