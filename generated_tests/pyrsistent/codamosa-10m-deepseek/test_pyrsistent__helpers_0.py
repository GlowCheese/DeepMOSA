# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_2
import pyrsistent._pset as module_1
import pytest


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
    var_0 = None
    var_1 = module_0.mutant(var_0)

def test_case_5():
    var_0 = ()
    var_1 = None
    var_2 = module_0.freeze(var_0, var_0)
    var_3 = module_0.thaw(var_1, var_1)
    var_4 = module_0.mutant(var_1)
    var_5 = module_0.thaw(var_0)
    var_6 = module_0.freeze(var_5)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = module_1.pset(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 1
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_8) == 2
    module_2.pmap(var_3)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.mutant(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = module_0.freeze(var_0, var_0)
    var_4 = [var_2, var_0]
    var_5 = module_0.thaw(var_4, var_2)
    var_1.__getitem__(var_1)

def test_case_8():
    var_0 = 1
    var_1 = None
    var_2 = module_0.thaw(var_1)
    var_3 = 2
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_5) == 2
    var_6 = [var_0, var_3]
    var_7 = module_1.pset(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_7) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_8 = 'a'
    var_9 = [var_0, var_5]
    var_10 = module_0.freeze(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_10) == 2
    var_11 = {var_8: var_10}
    var_12 = module_2.pmap(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_13 = module_0.thaw(var_10, var_5)
    var_14 = module_0.freeze(var_10)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_14) == 2

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = {}
    var_1 = module_0.thaw(var_0)
    var_2 = None
    var_3 = module_0.mutant(var_2)
    var_4 = module_0.mutant(var_2)
    var_5 = module_0.mutant(var_4)
    var_6 = module_0.thaw(var_2, var_2)
    var_6.copy()

def test_case_10():
    var_0 = 1
    var_1 = None
    var_2 = module_0.thaw(var_1)
    var_3 = 2
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_5) == 2
    var_6 = [var_0, var_3]
    var_7 = module_1.pset(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_7) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_8 = 'a'
    var_9 = 3
    var_10 = [var_0, var_5]
    var_11 = module_0.freeze(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_11) == 2
    var_12 = {var_8: var_9}
    var_13 = module_2.pmap(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_14 = module_0.freeze(var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_14) == 2

@pytest.mark.xfail(strict=True)
def test_case_11():
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

def test_case_12():
    var_0 = 'key'
    var_1 = {var_0: var_0}
    var_2 = module_2.pmap(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = 'All tests passed for mutant'
    var_6 = print(var_5)

def test_case_13():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 2
    var_5 = [var_0, var_1]
    var_6 = module_1.pset(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_7 = 'a'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = [var_0, var_9]
    var_11 = module_0.freeze(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_11) == 2
    var_12 = {var_7: var_8}
    var_13 = module_2.pmap(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 1
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_14 = []
    var_15 = module_0.thaw(var_13)
    var_16 = (var_0, var_14)
    var_17 = module_0.freeze(var_16)

def test_case_14():
    var_0 = 1
    var_1 = None
    var_2 = module_0.thaw(var_1)
    var_3 = 2
    var_4 = [var_0, var_3]
    var_5 = set(var_4)
    var_6 = module_0.freeze(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 2
    var_7 = [var_0, var_3]
    var_8 = module_1.pset(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_8) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_9 = 'a'
    var_10 = 3
    var_11 = [var_0, var_6]
    var_12 = module_0.freeze(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_12) == 2
    var_13 = {var_9: var_10}
    var_14 = module_2.pmap(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 1
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_15 = module_0.freeze(var_12)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_15) == 2

def test_case_15():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_8) == 2
    var_9 = {var_4: var_5}
    var_10 = module_2.pmap(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_11 = []
    var_12 = (var_0, var_11)
    var_13 = module_0.freeze(var_12)

def test_case_16():
    var_0 = 2
    var_1 = [var_0, var_0]
    var_2 = set(var_1)
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 1
    var_4 = [var_0]
    var_5 = module_1.pset(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_5) == 1
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_6 = module_0.thaw(var_5, var_3)
    var_7 = 3
    var_8 = module_0.freeze(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_8) == 2
    var_9 = {var_8: var_7}
    var_10 = module_2.pmap(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_11 = var_5.__eq__(var_10)