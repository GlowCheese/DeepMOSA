# Check out: https://github.com/GlowCheese/deepmosa
import builtins as module_1

import pyrsistent._pmap as module_0
import pyrsistent._transformations as module_2
import pytest


def test_case_0():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.items()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False

def test_case_1():
    var_0 = None
    with pytest.raises(TypeError):
        module_0.PMapView(var_0)

def test_case_2():
    var_0 = 'x'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.pmap(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1

def test_case_3():
    var_0 = -1109
    var_1 = 'S'
    var_2 = {var_1: var_0, var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.set(var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2

def test_case_4():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__str__()
    assert var_4 == "pmap({'d4i-\\x0c2l-_hB2Q': None})"
    var_5 = var_3.__eq__(var_0)
    var_6 = var_3.update()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_3.__contains__(var_0)
    assert var_7 is False

def test_case_5():
    var_0 = -15
    var_1 = 'b'
    var_2 = {var_1: var_0, var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.set(var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2

def test_case_6():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_4.set(var_3, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2

def test_case_7():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = bool(var_1 == var_0)
    assert var_2 is True

def test_case_8():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.update()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = module_0.PMapItems(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 1
    var_6 = var_4.discard(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_6.__repr__()
    assert var_7 == "pmap({'d4i-\\x0c2l-_hB2Q': None})"

def test_case_9():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_10():
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

def test_case_11():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.update()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = module_0.PMapItems(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 1
    var_6 = module_0.PMapView(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_6) == 1
    with pytest.raises(TypeError):
        var_4.__reversed__()

def test_case_12():
    var_0 = -15
    var_1 = 'b'
    var_2 = {var_1: var_0, var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__str__()
    assert var_4 == "pmap({'b': -15})"
    var_5 = bool(not var_3 == var_2)
    var_6 = var_3.update_with(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1

@pytest.mark.xfail(strict=True)
def test_case_13():
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

def test_case_14():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_4.set(var_3, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    var_6 = var_5.discard(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1

def test_case_15():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.transform()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = bool(var_1 == var_0)
    assert var_3 is True
    var_4 = module_0.pmap(pre_size=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0

def test_case_16():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    module_0.pmap(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__reduce__()
    var_3 = var_2.__contains__(var_0)
    assert var_3 is False
    var_4 = var_2.__lt__(var_0)
    var_4.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.update()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.PMapItems(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_3) == 0
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False
    var_3.set(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.PMapValues(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__str__()
    assert var_4 == 'pmap_values([None])'
    var_5 = var_3.__len__()
    assert var_5 == 1
    var_3.update()

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.discard(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == "pmap_items([('d4i-\\x0c2l-_hB2Q', None)])"
    var_8 = var_6.__eq__(var_0)
    assert var_8 is False
    var_3.update(*var_8)

def test_case_22():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__str__()
    assert var_4 == "pmap({'d4i-\\x0c2l-_hB2Q': None})"
    var_5 = var_3.__eq__(var_0)
    var_6 = var_3.update()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_6.set(var_6, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 2
    var_8 = var_7.__contains__(var_5)
    assert var_8 is False
    with pytest.raises(TypeError):
        module_0.PMapView(var_0)

def test_case_23():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = var_5.__add__(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.copy()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.keys()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_5) == 1
    var_4.remove(var_5)

@pytest.mark.xfail(strict=True)
def test_case_25():
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

def test_case_26():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = bool(var_1 == var_0)
    assert var_2 is True
    var_3 = module_0.pmap(pre_size=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0

def test_case_27():
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
    var_7 = var_4.__contains__(var_5)
    assert var_7 is False
    var_8 = var_4.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_8) == 2
    var_9 = var_7.__repr__()
    assert var_9 == 'False'
    with pytest.raises(TypeError):
        var_8.__reversed__()

def test_case_28():
    var_0 = None
    var_1 = 'k> Z7Ck\'%S5(/Js"'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.discard(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_2.__repr__()
    assert var_6 == '{\'k> Z7Ck\\\'%S5(/Js"\': None}'
    var_7 = var_3.items()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 1
    var_8 = var_7.__contains__(var_0)
    assert var_8 is False

def test_case_29():
    var_0 = None
    var_1 = 'd4i-\x0c2U]l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_4.update()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.set(var_3, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = var_6.discard(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1

def test_case_30():
    var_0 = {}
    var_1 = module_0.PMapView(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.m(**var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

def test_case_31():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.discard(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_3.set(var_0, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = var_6.set(var_3, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 3
    var_8 = var_7.discard(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 3
    var_9 = var_7.__add__(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 3
    var_10 = var_9.set(var_7, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 4

def test_case_32():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.PMapValues(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__str__()
    assert var_4 == 'pmap_values([None])'
    var_5 = var_3.__eq__(var_0)
    assert var_5 is False

def test_case_33():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_4.update()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.set(var_3, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = var_6.discard(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1

def test_case_34():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.__contains__(var_0)
    assert var_5 is False
    var_6 = var_4.__eq__(var_3)
    assert var_6 is True
    var_7 = var_4.update()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = var_7.set(var_7, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 2
    var_9 = var_8.__contains__(var_5)
    assert var_9 is False
    var_10 = module_1.object()

def test_case_35():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__str__()
    assert var_4 == "pmap({'d4i-\\x0c2l-_hB2Q': None})"
    var_5 = var_3.update()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.discard(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_6.discard(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = var_6.values()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_8) == 1
    var_9 = var_8.__repr__()
    assert var_9 == 'pmap_values([None])'
    with pytest.raises(TypeError):
        module_0.PMapView(var_0)

def test_case_36():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.discard(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_4.__str__()
    assert var_6 == "pmap({'d4i-\\x0c2l-_hB2Q': None})"
    var_7 = var_4.__eq__(var_3)
    assert var_7 is True
    var_8 = var_4.update()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_8.set(var_8, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 2
    var_10 = var_8.__iter__()
    var_11 = var_3.set(var_7, var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 2
    var_12 = module_0.PMapView(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_12) == 1

def test_case_37():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_4.update()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.set(var_3, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = var_6.discard(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = var_5.__add__(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_7.set(var_1, var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 1

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3.__getattr__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3.discard(var_2)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = None
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__str__()
    assert var_2 == 'pmap({})'
    var_3 = var_1.update()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = module_0.PMapItems(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_4) == 0
    var_5 = var_4.__repr__()
    assert var_5 == 'pmap_items([])'
    var_6 = var_3.discard(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_4.__eq__(var_6)
    assert var_7 is False
    var_2.set(var_0, var_0)

def test_case_41():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.discard(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_4.update()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_6.set(var_3, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 2
    var_8 = var_7.discard(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = var_6.discard(var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 1
    var_10 = var_3.__add__(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = var_10.set(var_7, var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 2
    var_12 = var_10.values()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_12) == 1
    var_13 = var_6.__contains__(var_4)
    assert var_13 is False
    var_14 = var_12.__contains__(var_11)
    assert var_14 is False
    var_15 = module_1.object()

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = None
    var_1 = '\n4-\x0c-_hB'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.PMapItems(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_4) == 1
    var_5 = module_0.m(**var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.discard(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_5.update()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = var_5.iteritems()
    var_9 = module_2.transform(var_5, var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 4
    var_10 = var_9.set(var_6, var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 5
    var_11 = var_9.discard(var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 4
    var_12 = module_0.PMapValues(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_12) == 1
    var_8.__add__(var_6)

def test_case_43():
    var_0 = None
    var_1 = 'dHT-52l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.discard(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_3.set(var_0, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = var_6.set(var_3, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 3
    var_8 = var_7.discard(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 3
    var_9 = module_0.PMapValues(var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_9) == 1
    var_10 = var_7.__eq__(var_6)
    assert var_10 is False
    var_11 = var_7.__add__(var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 3
    var_12 = var_8.__add__(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 3
    var_13 = var_11.set(var_7, var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 4

def test_case_44():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.discard(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_4.update()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_6.set(var_3, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 2
    var_8 = var_7.discard(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = module_0.PMapValues(var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_9) == 1
    var_10 = var_6.discard(var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = var_3.__add__(var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 1
    var_12 = var_6.set(var_5, var_8)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 2
    var_13 = module_0.m()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 0
    var_14 = var_12.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_14) == 2
    var_15 = var_14.remove(var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_15) == 1
    var_16 = var_7.__contains__(var_12)
    assert var_16 is False
    var_17 = module_0.PMapView(var_8)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_17) == 1

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.items()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_4) == 1
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_4.update()

def test_case_46():
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
    var_6 = 'a'
    var_7 = 'c'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 2
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.set(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    var_5 = var_4.__contains__(var_4)
    assert var_5 is False
    var_6 = module_0.m(**var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_4.discard(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = var_4.set(var_4, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 3
    var_9 = var_7.set(var_8, var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 2
    var_10 = var_8.discard(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 3
    module_0.PMapValues(var_5)

def test_case_48():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.discard(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_3.set(var_0, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = module_0.PMapValues(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_7) == 1
    with pytest.raises(TypeError):
        var_7.__setattr__(var_3, var_3)

def test_case_49():
    var_0 = None
    var_1 = 'd4i-\x0c2l-_hB2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__contains__(var_0)
    assert var_4 is False
    var_5 = var_3.discard(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_3.set(var_0, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = var_6.set(var_3, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 3
    var_8 = var_7.discard(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 3
    var_9 = module_0.PMapValues(var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_9) == 0
    var_10 = var_7.__add__(var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 3
    var_11 = var_7.items()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_11) == 3
    var_12 = var_11.__contains__(var_6)
    assert var_12 is False

def test_case_50():
    var_0 = None
    var_1 = 'k> Z7Ck\'%S5(/Js"'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.discard(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.set(var_3, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = var_6.__eq__(var_2)
    assert var_7 is False
    var_8 = var_6.discard(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 2
    var_9 = var_5.__add__(var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 1
    var_10 = var_7.__repr__()
    assert var_10 == 'False'
    var_11 = var_3.items()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_11) == 1
    var_12 = var_11.__contains__(var_8)
    assert var_12 is False

def test_case_51():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__add__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = module_0.PMapValues(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_2) == 0
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = -2
    var_5 = module_0._turbo_mapping(var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    with pytest.raises(TypeError):
        var_5.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_52():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.items()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_2.items()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_4) == 1
    var_5 = var_4.__contains__(var_0)
    assert var_5 is False
    var_6 = var_0.keys()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 0
    var_7 = var_0.__eq__(var_2)
    assert var_7 is False
    var_8 = var_2.keys()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_8) == 1
    var_9 = var_4.__str__()
    assert var_9 == 'pmap_items([(pmap({}), pmap({}))])'
    var_10 = var_5.__hash__()
    assert var_10 == 0
    var_11 = var_4.__eq__(var_1)
    assert var_11 is False
    var_10.__iter__()

def test_case_53():
    var_0 = None
    var_1 = 'dHT-52l-h2Q'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_3.discard(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_3.set(var_0, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = var_6.set(var_3, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 3
    var_8 = var_7.discard(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 3
    var_9 = module_0.PMapValues(var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_9) == 1
    var_10 = var_7.__eq__(var_6)
    assert var_10 is False
    var_11 = var_7.__add__(var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 3
    var_12 = var_5.__contains__(var_3)
    assert var_12 is False
    var_13 = var_11.set(var_7, var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 4