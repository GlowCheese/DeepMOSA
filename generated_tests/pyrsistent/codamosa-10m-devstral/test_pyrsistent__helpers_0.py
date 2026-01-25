# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)

def test_case_2():
    var_0 = None
    var_1 = module_0.thaw(var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.thaw(var_0, var_0)

def test_case_4():
    pass

def test_case_5():
    var_0 = None
    var_1 = module_0.mutant(var_0)

def test_case_6():
    var_0 = ()
    var_1 = module_0.mutant(var_0)
    var_2 = None
    var_3 = module_0.thaw(var_2, var_2)
    var_4 = module_0.thaw(var_2)
    var_5 = module_0.mutant(var_2)
    var_6 = module_0.thaw(var_0)
    var_7 = module_0.freeze(var_3)

def test_case_7():
    var_0 = ()
    var_1 = module_0.mutant(var_0)
    var_2 = None
    var_3 = module_0.freeze(var_1, var_1)
    var_4 = module_0.thaw(var_2, var_2)
    var_5 = module_0.thaw(var_2)
    var_6 = module_0.mutant(var_2)
    var_7 = module_0.thaw(var_0)
    var_8 = module_0.freeze(var_7)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.thaw(var_0)
    var_2 = None
    var_3 = module_0.thaw(var_2)
    var_4 = None
    var_5 = module_0.thaw(var_4, var_0)
    var_6 = [var_2, var_4, var_5, var_1]
    var_7 = [var_6]
    var_8 = module_0.mutant(var_7)
    var_9 = module_0.thaw(var_1, var_1)
    var_10 = module_0.mutant(var_0)
    var_11 = module_0.freeze(var_6, var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_11) == 4
    var_12 = module_0.freeze(var_0)
    var_13 = module_0.mutant(var_0)
    var_5.update(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = module_0.mutant(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = module_0.freeze(var_0, var_0)
    var_4 = [var_2, var_0]
    var_5 = module_0.thaw(var_4, var_2)
    var_1.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0
    var_2 = module_0.freeze(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 0
    var_3 = module_0.thaw(var_2)
    var_4 = module_0.mutant(var_3)
    var_5 = module_0.thaw(var_3, var_2)
    var_2.popitem()

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = {}
    var_1 = module_0.thaw(var_0)
    var_2 = None
    var_3 = module_0.mutant(var_2)
    var_4 = module_0.mutant(var_2)
    var_5 = module_0.mutant(var_4)
    var_6 = module_0.thaw(var_2, var_2)
    var_6.copy()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = []
    var_1 = None
    var_2 = module_0.freeze(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 0
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 0
    var_4 = module_0.thaw(var_2)
    var_5 = module_0.mutant(var_1)
    var_2.__ge__(var_5)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.thaw(var_0)
    var_2 = None
    var_3 = module_0.thaw(var_2)
    var_4 = None
    var_5 = module_0.thaw(var_4, var_0)
    var_6 = None
    var_7 = module_0.thaw(var_5)
    var_8 = module_0.thaw(var_6)
    var_9 = None
    var_10 = module_0.mutant(var_9)
    var_11 = True
    var_12 = (var_11,)
    var_13 = module_0.thaw(var_12, var_0)
    var_8.endswith(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)
    var_2 = module_0.thaw(var_0, var_1)
    var_3 = module_0.mutant(var_2)
    var_4 = module_0.thaw(var_3)
    var_5 = module_0.freeze(var_3)
    var_6 = module_0.thaw(var_5)
    var_7 = module_0.mutant(var_5)
    var_8 = module_0.thaw(var_6)
    var_9 = {var_6: var_0, var_5: var_5}
    var_10 = module_0.freeze(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_0.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)
    var_2 = module_0.freeze(var_0, var_0)
    var_3 = module_0.thaw(var_0, var_2)
    var_4 = module_0.mutant(var_1)
    var_5 = None
    var_6 = module_1.pset()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_7 = module_0.thaw(var_6)
    var_8 = module_0.freeze(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_8) == 0
    var_9 = module_0.thaw(var_5)
    var_10 = module_0.mutant(var_5)
    var_11 = module_0.thaw(var_5)
    var_12 = {var_9: var_1, var_8: var_8}
    var_13 = module_0.freeze(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 2
    var_14 = module_0.mutant(var_8)
    var_15 = var_6.__repr__()
    assert var_15 == 'pset()'
    var_16 = module_0.thaw(var_13, var_1)
    var_17 = module_0.mutant(var_0)
    var_3.__iadd__(var_13)

def test_case_16():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_5) == 2
    var_6 = {var_1: var_2}
    var_7 = module_2.pmap(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_8 = (var_0, var_5)
    var_9 = module_0.freeze(var_8)
    var_10 = 2
    var_11 = [var_0, var_10]
    var_12 = set(var_11)
    var_13 = module_0.freeze(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_13) == 2
    var_14 = [var_0, var_10]
    var_15 = module_1.pset(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_15) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_16 = 'b'
    var_17 = [var_10, var_2]
    var_18 = {var_1: var_0, var_16: var_17}
    var_19 = module_0.freeze(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_19) == 2
    var_20 = {var_1: var_2}
    var_21 = [var_0, var_20]
    var_22 = False
    var_23 = module_0.freeze(var_21, var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_23) == 2
    var_24 = {var_1: var_0}
    var_25 = module_2.pmap(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_25) == 1
    var_26 = module_0.freeze(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_26) == 1
    var_27 = [var_0, var_10]
    var_28 = module_1.pset(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_28) == 2
    var_29 = module_0.freeze(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_29) == 2
    var_30 = 42
    var_31 = module_0.freeze(var_30)
    assert var_31 == 42
    var_32 = 'hello'
    var_33 = module_0.freeze(var_32)
    assert var_33 == 'hello'