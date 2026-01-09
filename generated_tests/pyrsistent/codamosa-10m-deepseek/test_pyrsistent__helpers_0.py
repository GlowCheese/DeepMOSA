# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2
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
    var_1 = module_0.mutant(var_0)
    var_2 = None
    var_3 = module_0.freeze(var_1, var_1)
    var_4 = module_0.thaw(var_2, var_2)
    var_5 = module_0.thaw(var_2)
    var_6 = module_0.mutant(var_2)
    var_7 = module_0.thaw(var_0)
    var_8 = module_0.freeze(var_4)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = ()
    var_1 = module_0.mutant(var_0)
    var_2 = None
    var_3 = module_0.freeze(var_1, var_1)
    var_4 = module_0.thaw(var_2, var_2)
    var_5 = module_0.freeze(var_0)
    var_4.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
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
def test_case_8():
    var_0 = None
    var_1 = module_0.mutant(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = module_0.freeze(var_0, var_0)
    var_4 = [var_2, var_0]
    var_5 = module_0.thaw(var_4, var_2)
    var_1.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
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
def test_case_10():
    var_0 = {}
    var_1 = module_0.thaw(var_0)
    var_2 = None
    var_3 = module_0.mutant(var_2)
    var_4 = module_0.mutant(var_2)
    var_5 = module_0.mutant(var_4)
    var_6 = module_0.thaw(var_2, var_2)
    var_4.format(**var_6)

@pytest.mark.xfail(strict=True)
def test_case_11():
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
def test_case_12():
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
def test_case_13():
    var_0 = None
    var_1 = module_0.thaw(var_0, var_0)
    var_2 = module_0.thaw(var_0)
    var_3 = module_0.thaw(var_0)
    var_4 = module_0.mutant(var_1)
    var_5 = module_0.thaw(var_0)
    var_6 = {var_3: var_0, var_1: var_1}
    var_7 = module_0.freeze(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = module_0.mutant(var_0)
    var_2.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)
    var_2 = module_0.freeze(var_0, var_0)
    var_3 = module_0.thaw(var_0, var_2)
    var_4 = None
    var_5 = module_0.thaw(var_0)
    var_6 = module_0.freeze(var_3)
    var_7 = module_0.thaw(var_4)
    var_8 = module_0.mutant(var_4)
    var_9 = module_0.thaw(var_4)
    var_10 = {var_7: var_1, var_6: var_6}
    var_11 = module_0.freeze(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 1
    var_12 = module_0.mutant(var_2)
    var_13 = module_0.thaw(var_11, var_3)
    var_14 = module_0.freeze(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 1
    var_5.endswith(var_8, var_3, var_6)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = module_0.thaw(var_0)
    var_2 = set()
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = None
    var_5 = module_0.thaw(var_3)
    var_6 = module_0.thaw(var_4, var_4)
    var_7 = module_0.thaw(var_4)
    var_6.__iadd__(var_4)

def test_case_16():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_4) == 3
    var_5 = [var_0, var_1, var_2]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_1.pmap(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 2
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_10 = {var_0, var_1, var_2}
    var_11 = module_0.freeze(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_11) == 3
    var_12 = {var_0, var_1, var_2}
    var_13 = module_2.pset(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_13) == 3
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    var_14 = (var_0, var_9)
    var_15 = module_0.freeze(var_14)
    var_16 = {var_6: var_0}
    var_17 = [var_16, var_4]
    var_18 = module_0.freeze(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_18) == 2
    var_19 = {var_6: var_0}
    var_20 = module_1.pmap(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_20) == 1
    var_21 = {var_1, var_2}
    var_22 = module_2.pset(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_22) == 2
    var_23 = module_0.thaw(var_5, var_22)
    var_24 = [var_0, var_1, var_2]
    var_25 = False
    var_26 = module_0.freeze(var_24, var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_26) == 3
    var_27 = [var_0, var_1, var_2]
    var_28 = True
    var_29 = module_0.freeze(var_27, var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_29) == 3
    var_30 = print(var_9)

def test_case_17():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = None
    var_4 = module_0.thaw(var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = '\\^?x4x)(>;\\9} 9hK"t'
    var_7 = {var_4: var_0, var_6: var_1}
    var_8 = module_0.freeze(var_3, var_5)
    var_9 = module_0.freeze(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 2
    var_10 = {var_4: var_0, var_6: var_1}
    var_11 = module_1.pmap(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 2
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_12 = {var_0, var_1, var_2}
    var_13 = module_0.freeze(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_13) == 3
    var_14 = {var_0, var_1, var_2}
    var_15 = module_2.pset(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_15) == 3
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    var_16 = [var_1, var_2]
    var_17 = (var_0, var_16)
    var_18 = module_0.freeze(var_17)