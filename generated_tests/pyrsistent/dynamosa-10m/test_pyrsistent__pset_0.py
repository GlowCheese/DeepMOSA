# Check out: https://github.com/GlowCheese/deepmosa
import builtins as module_2

import pyrsistent._pmap as module_1
import pyrsistent._pset as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = module_0.pset(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_2) == 0
    var_3 = var_2.__str__()
    assert var_3 == 'pset()'
    var_4 = var_1.__str__()
    assert var_4 == 'pset()'
    var_3.copy()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.s(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.remove(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = var_3.__iter__()
    var_5 = var_3.__reduce__()
    var_6 = module_0.s()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 0
    var_4.__contains__(var_0)

def test_case_2():
    var_0 = module_0.s()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.update(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    var_2 = var_0.__str__()
    assert var_2 == 'pset()'
    var_3 = None
    var_4 = module_0.pset(pre_size=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = var_4.update(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_5) == 0
    var_6 = var_4.discard(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 0
    var_7 = var_4.__iter__()
    var_8 = var_7.__repr__()
    var_9 = var_5.__hash__()
    assert var_9 == 133146708735736
    var_10 = var_4.__repr__()
    assert var_10 == 'pset()'

def test_case_3():
    var_0 = None
    var_1 = None
    var_2 = module_0.pset(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_2) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(KeyError):
        var_2.remove(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.pset(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__hash__()
    assert var_2 == 133146708735736
    var_3 = None
    var_4 = module_0.s()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = var_4.__iter__()
    var_6 = var_5.__hash__()
    var_7 = None
    var_8 = module_0.s()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_8) == 0
    var_9 = var_8.__contains__(var_7)
    assert var_9 is False
    var_10 = var_8.add(var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_10) == 1
    var_9.remove(var_3)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.pset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.evolver()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet._Evolver'
    assert len(var_1) == 0
    var_2 = var_1.__repr__()
    var_2.evolver()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.s()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__contains__(var_0)
    assert var_1 is False
    var_2 = var_0.__iter__()
    var_2.__len__()

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_1.pmap()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.pset(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_2) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    var_3.update(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.s()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    module_0.PSet()

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_3 = module_0.s(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.remove(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = var_4.__iter__()
    var_6 = var_4.__reduce__()
    var_7 = module_0.s()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_7) == 0
    var_8 = var_3.__str__()
    assert var_8 == 'pset([None])'
    var_9 = module_0.pset()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_9) == 0
    var_10 = var_9.__len__()
    assert var_10 == 0
    var_10.__len__()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.s()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.update(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    var_2 = var_1.evolver()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet._Evolver'
    assert len(var_2) == 0
    var_3 = module_0.pset(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = var_1.__reduce__()
    var_5 = module_2.object()
    var_6 = var_1.__contains__(var_3)
    assert var_6 is False
    var_7 = module_0.pset(pre_size=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_7) == 0
    var_8 = var_1.evolver()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pset.PSet._Evolver'
    assert len(var_8) == 0
    var_5.__iter__()

def test_case_11():
    var_0 = module_0.pset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.add(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 1
    var_2 = var_0.copy()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_2) == 0
    var_3 = None
    var_4 = var_0.discard(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = var_4.evolver()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pset.PSet._Evolver'
    assert len(var_5) == 0
    var_6 = var_2.update(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 1
    var_7 = module_0.PSet(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_7) == 0
    with pytest.raises(KeyError):
        var_7.remove(var_3)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_0.pset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.add(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 1
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_2) == 0
    var_3 = var_2.__iter__()
    var_3.evolver()