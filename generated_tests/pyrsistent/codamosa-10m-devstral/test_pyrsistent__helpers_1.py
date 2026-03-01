# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2
import pyrsistent._pvector as module_3

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)

def test_case_2():
    var_0 = 'S5i9Bt'
    var_1 = module_0.thaw(var_0, var_0)
    assert var_1 == 'S5i9Bt'

def test_case_3():
    var_0 = None
    var_1 = module_0.thaw(var_0, var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.mutant(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_1.pset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.mutant(var_0)
    var_3 = None
    var_4 = module_0.freeze(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = module_0.mutant(var_3)
    var_1.upper()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 1
    var_1 = 3
    var_2 = 'ea'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_2.pmap(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = module_0.thaw(var_5)
    var_7 = [var_0, var_1, var_1]
    var_8 = module_1.pset(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_8) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_9 = module_0.thaw(var_8)
    var_10 = (var_0, var_1, var_0)
    var_11 = module_0.thaw(var_10)
    var_12 = False
    var_13 = [var_0, var_1, var_9]
    var_14 = module_0.thaw(var_13, var_12)
    var_15 = 4
    var_16 = [var_15, var_15]
    var_17 = module_1.pset(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_17) == 1
    var_18 = module_0.freeze(var_6)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 2
    var_19 = var_9.__reduce__()
    var_19.partition(var_18)

def test_case_7():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_2.pmap(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = module_0.thaw(var_5)
    var_7 = [var_0, var_1, var_1]
    var_8 = module_1.pset(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_8) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_9 = module_0.thaw(var_8)
    var_10 = (var_0, var_1, var_0)
    var_11 = module_0.thaw(var_10)
    var_12 = [var_1, var_1]
    var_13 = module_0.freeze(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_13) == 2
    var_14 = module_0.freeze(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_14) == 2
    var_15 = False
    var_16 = module_0.thaw(var_7, var_15)
    var_17 = 4
    var_18 = [var_17, var_17]
    var_19 = module_1.pset(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_19) == 1
    var_20 = module_0.thaw(var_0)
    assert var_20 == 1
    var_21 = 'hello'
    var_22 = module_0.thaw(var_21)
    assert var_22 == 'hello'
    var_23 = module_0.freeze(var_9)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_23) == 2

def test_case_8():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_2.pmap(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = module_0.thaw(var_5)
    var_7 = [var_0, var_1, var_1]
    var_8 = module_1.pset(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_8) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_9 = module_0.thaw(var_8)
    var_10 = (var_5, var_1, var_0)
    var_11 = module_0.thaw(var_8)
    var_12 = module_0.thaw(var_10)
    var_13 = module_0.freeze(var_0)
    assert var_13 == 1
    var_14 = False
    var_15 = None
    var_16 = module_0.mutant(var_15)
    var_17 = [var_0, var_1, var_9]
    var_18 = module_0.thaw(var_17, var_14)
    var_19 = module_0.freeze(var_12)
    var_20 = module_0.freeze(var_6)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_20) == 2
    var_21 = module_0.freeze(var_20, var_5)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_21) == 2
    var_22 = module_0.freeze(var_12)
    var_23 = var_20.set(var_5, var_5)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_23) == 3

def test_case_9():
    var_0 = 1
    var_1 = 2
    var_2 = None
    var_3 = module_0.freeze(var_2)
    var_4 = module_3.python_pvector()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_4) == 0
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_3.BRANCH_FACTOR == 32
    assert module_3.BIT_MASK == 31
    assert module_3.SHIFT == 5
    var_5 = module_0.thaw(var_4)
    var_6 = (var_0, var_1, var_3)
    var_7 = module_0.thaw(var_6)
    var_8 = False
    var_9 = [var_0, var_1, var_4]
    var_10 = module_0.thaw(var_9, var_8)
    var_11 = module_0.thaw(var_0)
    assert var_11 == 1
    var_12 = module_0.thaw(var_11, var_2)
    assert var_12 == 1
    var_13 = module_0.thaw(var_9, var_12)
    var_14 = module_0.freeze(var_7)

def test_case_10():
    var_0 = 8
    var_1 = 2
    var_2 = 'j#'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_2.pmap(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 2
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = module_0.thaw(var_5)
    var_7 = [var_0, var_1, var_1]
    var_8 = module_1.pset(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_8) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_9 = module_0.thaw(var_8)
    var_10 = (var_0, var_1, var_0)
    var_11 = module_0.thaw(var_10)
    var_12 = False
    var_13 = [var_0, var_1, var_9]
    var_14 = module_0.thaw(var_13, var_12)
    var_15 = 4
    var_16 = [var_15, var_15]
    var_17 = module_1.pset(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_17) == 1
    var_18 = 'hello'
    var_19 = module_0.thaw(var_18)
    assert var_19 == 'hello'
    var_20 = module_0.freeze(var_9)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_20) == 2

def test_case_11():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}
    var_7 = module_2.pmap(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 2
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_8 = module_0.thaw(var_5)
    assert var_8 == 'b'
    var_9 = module_0.thaw(var_7)
    var_10 = 'c'
    var_11 = {var_10: var_2}
    var_12 = module_2.pmap(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = {var_4: var_0, var_5: var_12}
    var_14 = module_2.pmap(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 2
    var_15 = module_0.thaw(var_14)
    var_16 = {var_0, var_1, var_2}
    var_17 = module_1.pset(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_17) == 3
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_18 = module_0.thaw(var_17)
    var_19 = {var_2, var_3}
    var_20 = module_1.pset(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_20) == 2
    var_21 = {var_4: var_0}
    var_22 = module_2.pmap(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_22) == 1
    var_23 = module_0.thaw(var_22, var_12)
    var_24 = module_0.thaw(var_6, var_12)
    var_25 = None
    var_26 = module_0.thaw(var_22, var_25)

def test_case_12():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_1.pset(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 3
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_5 = module_0.thaw(var_4)
    var_6 = (var_0, var_1, var_2)
    var_7 = module_0.thaw(var_6)
    var_8 = False
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.thaw(var_9, var_8)
    var_11 = module_0.thaw(var_0)
    assert var_11 == 1
    var_12 = 'hello'
    var_13 = module_0.thaw(var_12)
    assert var_13 == 'hello'
    var_14 = None
    var_15 = module_0.thaw(var_14)
    assert var_15 is None

def test_case_13():
    var_0 = 1
    var_1 = 2
    var_2 = None
    var_3 = module_0.freeze(var_2)
    var_4 = 1247
    var_5 = module_3.python_pvector()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_5) == 0
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_3.BRANCH_FACTOR == 32
    assert module_3.BIT_MASK == 31
    assert module_3.SHIFT == 5
    var_6 = module_0.thaw(var_5)
    var_7 = False
    var_8 = [var_0, var_1, var_4]
    var_9 = module_0.thaw(var_8, var_7)
    var_10 = module_0.thaw(var_0)
    assert var_10 == 1
    var_11 = module_0.thaw(var_3)
    var_12 = module_0.mutant(var_10)
    var_13 = None
    var_14 = module_0.thaw(var_13)
    assert var_14 is None