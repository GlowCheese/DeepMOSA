# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._pmap as module_0
import pyrsistent._transformations as module_1
import pytest


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

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.PMapItems(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_0.transform(*var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_0.__getattr__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__contains__(var_1)
    var_3 = var_0.__str__()

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

def test_case_7():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.m()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_2.discard(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    with pytest.raises(AttributeError):
        var_3.__getattr__(var_1)

def test_case_8():
    var_0 = '>2T0^9H)8y'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__add__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_4.discard(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.update()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__contains__(var_1)
    var_3 = var_0.__str__()

def test_case_10():
    var_0 = -16
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1

def test_case_11():
    var_0 = 1
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__reduce__()
    var_3 = var_1 == var_2
    assert var_3 is False
    var_4 = var_1.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0

def test_case_12():
    var_0 = 1
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1 == var_0
    assert var_2 is False

def test_case_13():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '>2T0^9H)8y'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__add__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_3.__lt__(var_1)

def test_case_15():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_0.__reversed__()

def test_case_16():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__str__()
    assert var_1 == 'pmap({})'

def test_case_17():
    var_0 = -2661
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.discard(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0

def test_case_18():
    var_0 = 2
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__reduce__()
    var_3 = {}
    var_4 = var_1 == var_3

def test_case_19():
    var_0 = 1
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__repr__()
    assert var_2 == 'pmap({})'
    var_3 = var_1.set(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_1 == var_1
    var_5 = var_3.__add__(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1

def test_case_20():
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

def test_case_21():
    var_0 = -16
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 1
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = 'O^&\x0cIdxV5\\-h'
    var_3 = var_1.set(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.discard(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = None
    module_0.pmap(var_5, var_5)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    var_2 = var_0.transform(*var_0)
    var_3 = var_0.set(var_0, var_2)
    var_4 = var_3.discard(var_2)
    var_5 = var_3.__add__(var_4)
    var_6 = var_0.__reduce__()
    var_7 = module_0.PMapItems(var_2)
    var_8 = var_0.__eq__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = 1
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.update()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = {}
    var_4 = None
    var_5 = var_4.__eq__(var_1)
    var_6 = var_1 == var_3
    var_7 = var_2.discard(var_1)

def test_case_25():
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
        var_3.__setattr__(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    var_2 = module_0.PMapValues(var_0)
    var_3 = var_0.transform(*var_2)
    var_4 = var_1.discard(var_3)
    var_5 = var_4.__eq__(var_3)
    var_6 = var_4.__len__()
    var_6.__len__()

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = module_0.PMapValues(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_2) == 0
    var_3 = var_0.__contains__(var_0)
    var_4 = var_2.__repr__()
    module_0.PMapItems(var_3)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_0.__repr__()
    assert var_2 == 'pmap({})'
    var_3 = module_0.PMapValues(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 0
    var_4 = var_0.__contains__(var_0)
    var_5 = var_0.discard(var_1)
    var_6 = module_0.PMapItems(var_5)
    var_7 = module_0.m(**var_5)
    var_8 = var_6.__repr__()
    var_0.__getattr__(var_5)

@pytest.mark.xfail(strict=True)
def test_case_29():
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
    var_4 = var_1.transform(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_1.set(var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.discard(var_4)
    var_7 = var_5.__add__(var_6)
    var_8 = var_5.discard(var_5)
    var_9 = var_3.__contains__(var_0)
    var_10 = var_5.__contains__(var_1)
    var_11 = var_1.__reduce__()
    var_12 = module_1.transform(var_11, var_3)
    var_13 = module_0.PMapItems(var_7)
    var_14 = var_13.__eq__(var_11)
    var_9.__len__()

def test_case_30():
    var_0 = None
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.copy()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    module_0.pmap(var_0, var_0)

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
    var_5 = var_1.set(var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    with pytest.raises(TypeError):
        var_3.__reversed__()

def test_case_33():
    var_0 = 1
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.values()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_2) == 0
    var_3 = var_1.update()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.__iter__()
    var_5 = module_0.pmap(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = {}
    var_7 = None
    var_8 = var_7.__eq__(var_1)
    var_9 = var_1 == var_6
    var_10 = var_3.discard(var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0

def test_case_34():
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
    var_6 = var_5.__repr__()
    assert var_6 == "pmap({'a': 1, 'b': 2})"
    var_7 = var_5.transform(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 8

def test_case_35():
    var_0 = 1
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.__len__()
    assert var_4 == 1
    var_5 = module_0.PMapItems(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 1
    var_6 = module_0.PMapItems(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 1
    var_7 = var_2.items()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 1
    var_8 = '\\tZ,\x0bv0:f5MaUDJJEi$'
    var_9 = (var_3, var_8)
    var_10 = var_6.__contains__(var_9)
    assert var_10 is False
    var_11 = var_2.discard(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 1
    var_12 = module_0.PMapItems(var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_12) == 0

def test_case_36():
    var_0 = None
    var_1 = module_0.pmap(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.values()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_2) == 0
    var_3 = module_0.m()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_3.update()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_3.iteritems()
    var_6 = var_4.__add__(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_2.__str__()
    assert var_7 == 'pmap_values([])'
    var_8 = var_3.set(var_7, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = {}
    var_10 = var_8.__eq__(var_3)
    assert var_10 is False
    var_11 = var_3 == var_9
    var_12 = var_8.discard(var_7)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 0

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.transform(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_1)
    var_3 = var_2.discard(var_1)
    var_4 = var_2.discard(var_2)
    var_5 = var_2.__contains__(var_0)
    var_6 = var_0.__reduce__()
    var_7 = var_0.__eq__(var_2)
    var_8 = var_2.set(var_3, var_3)
    var_4.set(var_6, var_6)

def test_case_38():
    var_0 = 24
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = {}
    var_3 = var_1 == var_2

@pytest.mark.xfail(strict=True)
def test_case_39():
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
    var_4 = var_1.transform(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_1.set(var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.discard(var_4)
    var_7 = var_5.__add__(var_6)
    var_8 = var_5.discard(var_5)
    var_9 = var_3.__contains__(var_0)
    var_10 = var_5.__contains__(var_1)
    var_11 = module_0.PMapItems(var_5)
    var_12 = module_0.PMapItems(var_4)
    var_13 = var_6.iteritems()
    var_14 = module_0.PMapItems(var_7)
    var_15 = var_14.__eq__(var_11)
    var_9.__len__()

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = 1
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__repr__()
    assert var_2 == 'pmap({})'
    var_3 = var_1.set(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_3.__add__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = {}
    var_6 = module_0.pmap(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_1 == var_5
    var_8 = module_0.PMapItems(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_8) == 1
    var_9 = var_6.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_9) == 0
    var_10 = var_8.__contains__(var_1)
    assert var_10 is False
    var_11 = var_3.discard(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 0
    var_12 = module_0.m()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 0
    var_13 = None
    var_14 = var_9.__str__()
    assert var_14 == 'pmap_items([])'
    module_0.pmap(var_13, var_13)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.PMapValues(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 0
    var_4 = var_1.transform(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_1.set(var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.discard(var_4)
    var_7 = var_2.__str__()
    var_8 = var_5.__add__(var_6)
    var_9 = var_5.discard(var_5)
    var_10 = var_5.__contains__(var_1)
    var_11 = module_0.pmap(pre_size=var_3)
    var_12 = var_1.__reduce__()
    var_13 = var_9.__eq__(var_7)
    var_14 = module_0.PMapItems(var_8)
    var_15 = module_0.PMapItems(var_9)
    var_16 = var_1.__eq__(var_5)
    var_17 = var_12.__len__()
    var_18 = var_5.__contains__(var_17)
    var_16.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__eq__(var_0)
    var_3 = var_1.evolver()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_3) == 0
    var_1.transform(*var_3)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    var_2 = module_0.PMapValues(var_0)
    var_3 = var_0.transform(*var_2)
    var_4 = var_0.set(var_0, var_3)
    var_5 = var_4.__add__(var_4)
    var_6 = var_0.transform(*var_0)
    var_7 = var_4.discard(var_4)
    var_8 = var_4.__contains__(var_0)
    var_9 = var_0.__reduce__()
    var_10 = module_0.PMapItems(var_7)
    var_11 = var_0.__eq__(var_4)
    var_12 = var_9.__len__()
    var_13 = var_4.set(var_6, var_6)
    var_14 = var_5.__eq__(var_4)
    var_15 = var_14.__str__()
    var_16 = module_0.pmap(var_13)
    var_17 = var_13.__eq__(var_16)
    var_18 = None
    module_0.PMapItems(var_18)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.keys()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    var_2 = module_0.PMapValues(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_2) == 0
    var_3 = var_0.transform(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_0.set(var_0, var_3)
    var_5 = var_4.discard(var_3)
    var_6 = var_4.__add__(var_5)
    var_7 = var_5.transform(*var_5)
    var_8 = var_4.discard(var_4)
    var_9 = var_4.__contains__(var_0)
    var_10 = module_0.PMapItems(var_8)
    var_11 = var_0.__eq__(var_4)
    var_12 = var_8.__len__()
    var_13 = var_4.set(var_5, var_5)
    var_14 = var_9.__eq__(var_2)
    var_15 = var_14.__str__()
    var_16 = module_0.pmap(var_13)
    var_17 = var_13.__eq__(var_16)
    var_18 = None
    module_0.PMapItems(var_18)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    var_2 = module_0.PMapValues(var_0)
    var_3 = var_0.transform(*var_2)
    var_4 = var_0.set(var_0, var_3)
    var_5 = var_4.discard(var_3)
    var_6 = var_4.__add__(var_5)
    var_7 = var_5.__eq__(var_3)
    var_8 = var_5.__len__()
    var_9 = var_0.__reduce__()
    module_0.PMapItems(var_7)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = None
    var_1 = module_0.pmap(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.values()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_2) == 0
    var_3 = 1
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = module_0.m()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_5.update()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_5.__repr__()
    assert var_7 == 'pmap({})'
    var_8 = var_5.set(var_7, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = {}
    var_10 = var_8.__eq__(var_5)
    assert var_10 is False
    var_11 = var_5 == var_9
    var_12 = var_5.items()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_12) == 0
    var_13 = var_8.discard(var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 0
    var_8.__add__(var_10)

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    var_2 = var_1.transform()
    var_3 = module_0.PMapValues(var_0)
    var_4 = var_0.transform(*var_3)
    var_5 = var_0.set(var_0, var_4)
    var_6 = var_5.discard(var_4)
    var_7 = var_4.__str__()
    var_8 = var_5.discard(var_5)
    var_9 = var_5.__contains__(var_0)
    var_10 = module_0.pmap()
    var_11 = var_0.__reduce__()
    var_12 = module_0.PMapItems(var_8)
    var_13 = var_0.__eq__(var_5)
    var_14 = var_11.__len__()
    var_15 = var_5.set(var_6, var_6)
    var_16 = var_7.__eq__(var_5)
    var_17 = var_16.__str__()
    var_18 = var_7.__str__()
    var_19 = module_0.pmap(pre_size=var_6)
    var_20 = var_5.__eq__(var_19)
    var_21 = module_0.PMapItems(var_15)
    var_22 = module_0.pmap()
    var_23 = var_12.__contains__(var_2)
    var_24 = var_10.__contains__(var_13)
    var_25 = var_4.__eq__(var_8)
    var_26 = var_1.__eq__(var_18)
    var_27 = var_1.set(var_13, var_24)
    var_28 = var_8.set(var_19, var_19)
    var_29 = module_0.PMapValues(var_28)
    var_8.__getattr__(var_11)

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    var_2 = module_0.PMapValues(var_0)
    var_3 = var_0.transform(*var_2)
    var_4 = var_0.set(var_0, var_3)
    var_5 = var_4.discard(var_3)
    var_6 = var_4.__add__(var_5)
    var_7 = var_3.__str__()
    var_8 = var_4.discard(var_4)
    var_9 = var_4.__contains__(var_0)
    var_10 = module_0.pmap()
    var_11 = var_0.__reduce__()
    var_12 = module_0.PMapItems(var_8)
    var_13 = var_0.__eq__(var_4)
    var_14 = var_4.set(var_5, var_5)
    var_15 = var_6.__eq__(var_4)
    var_16 = var_15.__str__()
    var_17 = module_0.pmap(pre_size=var_5)
    var_18 = var_4.__eq__(var_6)
    var_19 = None
    var_20 = module_0.PMapItems(var_14)
    var_21 = module_0.pmap()
    var_22 = var_12.__contains__(var_16)
    var_23 = var_10.__contains__(var_13)
    var_24 = var_5.__eq__(var_19)
    var_25 = var_6.set(var_9, var_10)
    var_26 = {var_17: var_0, var_9: var_1}
    var_27 = module_0.PMapValues(var_26)
    var_28 = var_0.__str__()
    var_29 = var_10.__eq__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__iter__()
    var_2 = var_0.discard(var_0)
    var_3 = var_2.transform()
    var_4 = var_1.__str__()
    var_5 = var_0.transform(*var_4)
    var_6 = var_0.set(var_0, var_5)
    var_7 = var_6.__add__(var_6)
    var_8 = var_2.transform(*var_2)
    var_9 = var_6.discard(var_6)
    var_10 = var_6.__contains__(var_0)
    var_11 = var_0.__reduce__()
    var_12 = var_5.__contains__(var_2)
    var_13 = var_0.__eq__(var_6)
    var_14 = var_11.__len__()
    var_15 = var_6.set(var_7, var_7)
    var_16 = var_7.__eq__(var_6)
    var_17 = var_16.__str__()
    var_18 = module_0.pmap(var_15)
    var_19 = var_15.__eq__(var_18)
    var_1.update(*var_3)

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    var_2 = module_0.PMapValues(var_0)
    var_3 = var_0.transform(*var_2)
    var_4 = var_0.set(var_0, var_3)
    var_5 = var_4.discard(var_3)
    var_6 = var_4.__add__(var_4)
    var_7 = var_5.transform(*var_5)
    var_8 = var_3.set(var_7, var_6)
    var_9 = var_7.__contains__(var_7)
    var_10 = var_6.__reduce__()
    var_11 = module_0.PMapItems(var_1)
    var_12 = var_0.__eq__(var_5)
    var_13 = var_7.__len__()
    var_14 = var_5.set(var_0, var_7)
    var_15 = var_1.set(var_14, var_1)
    var_16 = var_3.iteritems()
    var_17 = module_0.pmap(var_6)
    var_18 = var_12.__eq__(var_7)
    var_19 = module_0.pmap(var_17, var_13)
    var_20 = None
    var_21 = var_8.__eq__(var_17)
    var_22 = module_0.PMapItems(var_5)
    var_23 = module_0.pmap(pre_size=var_5)
    var_24 = var_19.__contains__(var_14)
    var_25 = var_22.__contains__(var_20)
    var_15.__getitem__(var_15)

@pytest.mark.xfail(strict=True)
def test_case_51():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.discard(var_0)
    var_2 = module_0.PMapValues(var_0)
    var_3 = var_0.transform(*var_2)
    var_4 = var_0.set(var_0, var_3)
    var_5 = var_4.discard(var_3)
    var_6 = var_4.__add__(var_4)
    var_7 = var_5.transform(*var_5)
    var_8 = var_4.discard(var_4)
    var_9 = var_4.__contains__(var_0)
    var_10 = var_0.__reduce__()
    var_11 = module_0.PMapItems(var_8)
    var_12 = var_0.__eq__(var_4)
    var_13 = var_10.__len__()
    var_14 = var_4.set(var_5, var_5)
    var_15 = var_5.set(var_7, var_7)
    var_16 = var_15.iteritems()
    var_17 = var_1.set(var_4, var_12)
    var_18 = var_7.__eq__(var_5)
    var_19 = module_0.pmap(var_16)
    var_20 = var_2.__eq__(var_18)
    var_21 = var_15.values()
    var_22 = module_0.pmap()
    var_23 = var_15.__contains__(var_13)
    var_24 = var_21.__contains__(var_3)
    var_1.__setattr__(var_9, var_4)

def test_case_52():
    var_0 = 1
    var_1 = module_0.m()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = var_1.update()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_3.__add__(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_1.__repr__()
    assert var_5 == 'pmap({})'
    var_6 = var_1.set(var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = {}
    var_8 = var_6.copy()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_6.__eq__(var_1)
    assert var_9 is False
    var_10 = var_1 == var_7
    var_11 = var_6.keys()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_11) == 1
    var_12 = var_11.discard(var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_12) == 0
    var_13 = var_8.__add__(var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1