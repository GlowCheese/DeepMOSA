# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0.thaw(var_0)

def test_case_2():
    pass

def test_case_3():
    var_0 = None
    var_1 = module_0.mutant(var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)

def test_case_5():
    var_0 = None
    var_1 = module_0.thaw(var_0, var_0)

def test_case_6():
    var_0 = ()
    var_1 = module_0.mutant(var_0)
    var_2 = None
    var_3 = module_0.freeze(var_1, var_1)
    var_4 = module_0.thaw(var_2, var_2)
    var_5 = module_0.thaw(var_2)
    var_6 = module_0.mutant(var_2)
    var_7 = module_0.thaw(var_0)
    var_8 = module_0.freeze(var_4)

def test_case_7():
    var_0 = None
    var_1 = module_0.thaw(var_0)
    var_2 = None
    var_3 = module_0.thaw(var_1, var_2)
    var_4 = module_0.mutant(var_2)
    var_5 = [var_2, var_4]
    var_6 = module_0.freeze(var_5, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_6) == 2
    var_7 = module_0.mutant(var_0)
    var_8 = module_0.freeze(var_7, var_2)
    var_9 = module_0.thaw(var_6, var_8)

def test_case_8():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = []
    var_1 = module_0.freeze(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0
    var_2 = module_0.freeze(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 0
    var_3 = module_0.thaw(var_2)
    var_4 = module_0.mutant(var_3)
    var_2.__missing__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = []
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = None
    var_4 = var_1.__lt__(var_3)
    var_5 = module_0.freeze(var_2, var_1)
    var_4.keys()

def test_case_11():
    var_0 = ()
    var_1 = module_0.freeze(var_0)

def test_case_12():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1

def test_case_13():
    var_0 = module_1.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0)

def test_case_14():
    var_0 = module_1.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.freeze(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.mutant(var_0)
    var_3 = (var_1,)
    var_4 = module_0.thaw(var_3)
    var_5 = module_0.freeze(var_0)
    var_6 = module_0.mutant(var_5)
    var_7 = module_0.mutant(var_1)
    var_5.__contains__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.freeze(var_0)
    var_3 = module_0.mutant(var_1)
    var_4 = var_1.__lt__(var_1)
    var_5 = module_1.pmap()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = module_0.thaw(var_5)
    var_7 = module_0.freeze(var_3, var_1)
    var_8 = module_0.mutant(var_7)
    var_9 = module_0.thaw(var_6)
    var_10 = module_0.freeze(var_0, var_9)
    var_11 = module_0.mutant(var_4)
    var_3.__reversed__()

def test_case_17():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1, var_0)
    var_3 = module_0.freeze(var_2)
    var_4 = bool(var_3 == (1, 2, 3))

def test_case_18():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    var_2 = module_0.thaw(var_1)
    var_3 = var_1.evolver()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet._Evolver'
    assert len(var_3) == 0
    var_4 = module_2.pset()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'

def test_case_19():
    var_0 = None
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.freeze(var_0)
    var_3 = module_0.mutant(var_1)
    var_4 = module_1.pmap()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = module_0.thaw(var_4, var_3)
    var_6 = module_0.freeze(var_2, var_0)
    var_7 = var_6.__ge__(var_4)
    var_8 = {var_7: var_5, var_2: var_2}
    var_9 = module_0.freeze(var_8, var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 2
    var_10 = None
    var_11 = module_0.freeze(var_10)
    var_12 = module_0.thaw(var_6)
    var_13 = module_0.thaw(var_12, var_2)
    var_14 = module_0.thaw(var_9)

def test_case_20():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0