# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._pmap as module_0
import pyrsistent._pvector as module_2
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
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(AttributeError):
        var_0.__getattr__(var_0)

def test_case_4():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__contains__(var_0)
    assert var_1 is False
    var_2 = var_0.__hash__()
    assert var_2 == 133146708735736
    var_3 = None
    with pytest.raises(TypeError):
        module_0.PMapView(var_3)

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
    var_5.remove(var_5)

def test_case_8():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.discard(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_3.__repr__()
    assert var_4 == 'pmap({})'
    var_5 = module_0.pmap()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_3.__eq__(var_1)
    assert var_6 is False

def test_case_9():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.update()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

def test_case_10():
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

def test_case_11():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 20
    var_6 = module_0._turbo_mapping(var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = dict(var_6)
    var_9 = bool(var_8 == {'a': 1, 'b': 2})
    assert var_9 is True
    var_10 = var_8.__repr__()
    assert var_10 == "{'b': 2, 'a': 1}"

def test_case_12():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_13():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.itervalues()
    var_3 = module_0.pmap(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0

def test_case_14():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_0.__reversed__()

def test_case_15():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_0.__str__()
    assert var_2 == 'pmap({})'

@pytest.mark.xfail(strict=True)
def test_case_16():
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

def test_case_17():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_18():
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

def test_case_19():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__iter__()
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_4 = var_1.items()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_4) == 0
    var_5 = var_4.__iter__()
    var_6 = module_0.pmap()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0

@pytest.mark.xfail(strict=True)
def test_case_20():
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
    assert var_4 == -4180171864684324220
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

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.m()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2.update_with(var_0, *var_1)

def test_case_22():
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

def test_case_23():
    var_0 = 2
    var_1 = '2>M7X^iHO9\\X9jr'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.pmap()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.update()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = module_0.m(**var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_5.set(var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = var_5.__add__(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1

def test_case_24():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__reduce__()
    var_2 = var_1.__repr__()
    assert var_2 == '(<function pmap at 0x7e3fd6ea6b00>, ({},))'
    var_3 = var_0.__repr__()
    assert var_3 == 'pmap({})'
    var_4 = var_1.__repr__()
    assert var_4 == '(<function pmap at 0x7e3fd6ea6b00>, ({},))'
    var_5 = var_2.__iter__()
    var_6 = var_0.transform(*var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 17

def test_case_25():
    var_0 = 2
    var_1 = 'a'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = None
    var_5 = var_3.set(var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = 'b'
    var_7 = {var_6: var_3, var_6: var_0}
    var_8 = var_3 == var_7

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.discard(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.__iter__()
    var_5 = var_3.__repr__()
    assert var_5 == 'pmap({})'
    var_6 = var_3.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_6) == 0
    var_7 = module_0.pmap(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_3.__lt__(var_2)

def test_case_27():
    var_0 = {}
    var_1 = module_0.PMapValues(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1 == var_1
    assert var_2 is True

def test_case_28():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.pmap()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_3) == 0
    var_4 = var_3.__contains__(var_0)
    assert var_4 is False
    var_5 = module_0.pmap()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = var_3.__len__()
    assert var_6 == 0

def test_case_29():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = '@(6F^'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = var_5 == var_8

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = None
    var_1.__new__(var_1, var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    module_0.pmap(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = -5656
    var_1 = '@'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.PMapValues(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__reduce__()
    var_5 = module_0.m(**var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_4.__repr__()
    assert var_6 == "(<function _reconstructor at 0x7e4008736950>, (<class 'pyrsistent._pmap.PMapValues'>, <class 'object'>, None), {'_map': pmap({'@': -5656})})"
    var_7 = var_5.discard(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_3.__repr__()
    assert var_8 == 'pmap_values([-5656])'
    var_9 = var_4.__repr__()
    assert var_9 == "(<function _reconstructor at 0x7e4008736950>, (<class 'pyrsistent._pmap.PMapValues'>, <class 'object'>, None), {'_map': pmap({'@': -5656})})"
    var_10 = 'hjS}B+R='
    var_11 = {var_10: var_0}
    var_12 = var_5 == var_11
    var_13 = var_6.__iter__()
    var_14 = var_5.discard(var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 1
    var_15 = var_14.set(var_12, var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 2
    var_16 = var_5.__add__(var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 1
    var_3.transform(*var_6)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_0.m()
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
    var_3 = var_0.__iter__()
    var_4 = var_2.__str__()
    assert var_4 == 'pmap({})'
    var_5 = var_1.values()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_5) == 1
    var_6 = var_5.__str__()
    assert var_6 == 'pmap_values([pmap({})])'
    var_6.iteritems()

def test_case_34():
    var_0 = module_0.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = bool(var_0 == {})
    assert var_1 is True

def test_case_35():
    var_0 = 1
    var_1 = 2
    var_2 = 'b'
    var_3 = {var_2: var_0, var_2: var_1}
    var_4 = module_0.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = None
    var_6 = var_4.set(var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    var_7 = var_4.__eq__(var_0)
    var_8 = var_4 == var_3
    assert var_8 is True

def test_case_36():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_1.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_2) == 1
    var_3 = var_1.discard(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_0.items()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_4) == 0
    var_5 = var_0.__iter__()
    var_6 = var_1.__repr__()
    assert var_6 == 'pmap({pmap({}): pmap({})})'
    with pytest.raises(TypeError):
        var_4.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_0.m()
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
    var_3 = var_2.update()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.__str__()
    assert var_4 == 'pmap({pmap({}): pmap({})})'
    var_5 = var_1.values()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_5) == 1
    var_6 = var_1.__contains__(var_0)
    assert var_6 is True
    var_7 = var_5.__contains__(var_2)
    assert var_7 is True
    var_8 = var_3.iteritems()
    var_5.__hash__()

def test_case_38():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__iter__()
    var_3 = module_0.pmap(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = bool(var_1 is not var_3)
    assert var_4 is True
    var_5 = bool(var_1 == var_3)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = module_0.m()
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
    var_3 = var_0.__iter__()
    var_4 = var_1.__str__()
    assert var_4 == 'pmap({pmap({}): pmap({})})'
    var_5 = var_1.__contains__(var_2)
    assert var_5 is True
    var_6 = module_2.python_pvector()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_6) == 0
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_2.BRANCH_FACTOR == 32
    assert module_2.BIT_MASK == 31
    assert module_2.SHIFT == 5
    var_3.items()

def test_case_40():
    var_0 = 4
    var_1 = '1>P.OS6daY=429Af'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.pmap()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = module_0.m(**var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = None
    var_7 = var_5.set(var_1, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = var_5 == var_2
    assert var_8 is True
    var_9 = var_5.__contains__(var_3)
    assert var_9 is False
    var_10 = var_5.discard(var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = var_10.__add__(var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 1

def test_case_41():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = '1>P.OS6daY=429Af'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.pmap()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = var_5.update()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = module_0.m(**var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 2
    var_8 = None
    var_9 = var_7.set(var_2, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 2
    var_10 = 'a'
    var_11 = var_9.__eq__(var_7)
    assert var_11 is False
    var_12 = '!kR'
    var_13 = {var_10: var_0, var_12: var_1}
    var_14 = var_7 == var_13
    var_15 = var_7.__contains__(var_5)
    assert var_15 is False
    var_16 = var_9.discard(var_8)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 2
    var_17 = var_6.__repr__()
    assert var_17 == 'pmap({})'
    var_18 = var_16.__add__(var_7)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 2

def test_case_42():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = module_0.PMapView(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_2) == 0
    var_3 = var_0.discard(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    with pytest.raises(TypeError):
        var_2.__setattr__(var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 3
    var_1 = ''
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.pmap()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.m(**var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = var_4.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 1
    var_6 = var_4.discard(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_4 == var_5
    var_4.discard(var_5)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.update(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.PMapValues(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_3) == 0
    var_4 = var_2.update()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = None
    var_6 = var_3.__eq__(var_5)
    assert var_6 is False
    var_5.set(var_4, var_0)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = module_0.PMapView(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_2) == 1
    var_3 = var_1.discard(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = var_0.discard(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_3.iterkeys()
    var_6 = var_3.__repr__()
    assert var_6 == 'pmap({pmap({}): pmap({})})'
    var_7 = var_6.__str__()
    assert var_7 == 'pmap({pmap({}): pmap({})})'
    var_8 = module_0.PMapValues(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_8) == 0
    var_9 = var_0.update()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = module_0.PMapView(var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_10) == 1
    var_11 = None
    var_12 = var_3.__eq__(var_11)
    var_13 = var_9.set(var_3, var_4)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1
    var_14 = var_13.__contains__(var_13)
    assert var_14 is False
    var_15 = module_0.pmap()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_16 = var_15.__contains__(var_4)
    assert var_16 is False
    var_16.__reversed__()

def test_case_46():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.discard(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.__iter__()
    var_5 = var_3.__repr__()
    assert var_5 == 'pmap({})'
    var_6 = module_2.python_pvector()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_6) == 0
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_2.BRANCH_FACTOR == 32
    assert module_2.BIT_MASK == 31
    assert module_2.SHIFT == 5
    var_7 = var_3.items()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 0
    var_8 = var_3.update()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = var_3.__add__(var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = None
    var_11 = var_7.__eq__(var_10)
    assert var_11 is False
    var_12 = var_9.__contains__(var_8)
    assert var_12 is False
    var_13 = var_1.set(var_8, var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1
    var_14 = var_2.copy()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 0
    var_15 = var_13.__contains__(var_6)
    assert var_15 is False
    var_16 = var_1.set(var_15, var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 2
    var_17 = var_16.set(var_3, var_3)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 2
    var_18 = module_0.pmap()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 0
    var_19 = var_0.__contains__(var_5)
    assert var_19 is False
    var_20 = var_13.__contains__(var_11)
    assert var_20 is False
    var_21 = var_18.items()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_21) == 0
    var_22 = var_21.__contains__(var_10)
    assert var_22 is False
    var_23 = module_0.pmap()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_23) == 0
    var_24 = var_13.__add__(var_1)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_24) == 1
    var_25 = var_3.__len__()
    assert var_25 == 0
    var_26 = var_21.__repr__()
    assert var_26 == 'pmap_items([])'

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = None
    var_1 = module_0.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_2) == 0
    var_3 = var_2.__contains__(var_0)
    assert var_3 is False
    module_0.PMapItems(var_0)

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.PMapItems(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_1) == 0
    var_2 = module_0.PMapView(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMapView'
    assert len(var_2) == 0
    var_3 = var_1.__str__()
    assert var_3 == 'pmap_items([])'
    var_4 = var_0.discard(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_1.iterkeys()

def test_case_49():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.values()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMapValues'
    assert len(var_1) == 0
    var_2 = var_0.set(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = var_0.discard(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.discard(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = var_4.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 0
    var_6 = var_4.update()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = var_4.__add__(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = None
    var_9 = var_3.__repr__()
    assert var_9 == 'pmap({})'
    var_10 = var_7.__contains__(var_6)
    assert var_10 is False
    var_11 = var_2.set(var_6, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 1
    var_12 = var_3.copy()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 0
    var_13 = var_11.__contains__(var_8)
    assert var_13 is False
    var_14 = var_2.set(var_13, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 2
    var_15 = var_14.set(var_4, var_4)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 2
    var_16 = module_0.pmap(var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 0
    var_17 = var_5.__contains__(var_14)
    assert var_17 is False
    var_18 = module_0.pmap(var_6)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 0
    var_19 = var_16.__contains__(var_10)
    assert var_19 is False
    var_20 = var_10.__eq__(var_14)

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = module_0.m()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.set(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 1
    var_2 = var_0.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = var_1.discard(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_1.__iter__()
    var_5 = var_3.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_5) == 0
    var_6 = var_3.update()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = module_0.PMapItems(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMapItems'
    assert len(var_7) == 0
    var_8 = var_3.__add__(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_0.__getattr__(var_5)

def test_case_51():
    var_0 = 3
    var_1 = '@'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.pmap()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__reduce__()
    var_5 = module_0.m(**var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = var_4.__repr__()
    assert var_6 == '(<function pmap at 0x7e3fd6ea6b00>, ({},))'
    var_7 = var_5.discard(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = var_5.__str__()
    assert var_8 == "pmap({'@': 3})"
    var_9 = 'hjS}B+R='
    var_10 = {var_9: var_3, var_9: var_0}
    var_11 = var_5 == var_10
    var_12 = var_6.__iter__()
    var_13 = var_5.discard(var_8)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1
    var_14 = var_13.set(var_11, var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 2
    var_15 = var_5.__add__(var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 1
    var_16 = var_15.transform(*var_6)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 18