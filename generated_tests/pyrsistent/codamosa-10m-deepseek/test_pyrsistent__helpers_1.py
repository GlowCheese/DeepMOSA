# Check out: https://github.com/GlowCheese/deepmosa
import collections as module_4

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_2
import pyrsistent._pset as module_1
import pyrsistent._pvector as module_3
import pytest


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

def test_case_5():
    var_0 = module_1.pset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.mutant(var_0)
    var_3 = module_0.freeze(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_3) == 0
    var_4 = module_0.freeze(var_3, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 0
    var_5 = module_0.thaw(var_4)

def test_case_6():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0

def test_case_7():
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
    var_8 = []
    var_9 = (var_0, var_8)
    var_10 = module_0.freeze(var_9)
    var_11 = 2
    var_12 = [var_0, var_11]
    var_13 = set(var_12)
    var_14 = module_0.freeze(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_14) == 2
    var_15 = [var_0, var_11]
    var_16 = module_1.pset(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_16) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_17 = 'b'
    var_18 = 4
    var_19 = 5
    var_20 = [var_18, var_19]
    var_21 = {var_17: var_20}
    var_22 = module_0.freeze(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_22) == 1
    var_23 = module_0.freeze(var_0)
    assert var_23 == 1
    var_24 = 'test'
    var_25 = module_0.freeze(var_24)
    assert var_25 == 'test'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_1.pset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    var_2 = var_0.__reduce__()
    var_3 = module_0.thaw(var_1)
    var_4 = module_0.thaw(var_0)
    var_5 = module_0.thaw(var_2, var_3)
    var_6 = module_0.freeze(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 0
    var_7 = module_0.freeze(var_2)
    var_2.__new__(var_2, var_0, var_2)

def test_case_9():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_4) == 3
    var_5 = var_4.__iter__()
    var_6 = [var_0, var_5]
    var_7 = module_0.freeze(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_7) == 2
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_8: var_0, var_9: var_1}
    var_11 = module_0.freeze(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 2
    var_12 = {var_8: var_0, var_9: var_1}
    var_13 = module_2.pmap(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 2
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_14 = module_0.freeze(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_14) == 2
    var_15 = {var_9: var_1, var_13: var_11}
    var_16 = module_2.pmap(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_16) == 2
    var_17 = {var_8: var_16}
    var_18 = module_2.pmap(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_18) == 1
    var_19 = {var_0, var_1, var_2}
    var_20 = module_0.freeze(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_20) == 3
    var_21 = {var_0, var_1, var_2}
    var_22 = module_1.pset(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_22) == 3
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_23 = module_0.freeze(var_4)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_23) == 3
    var_24 = 42
    var_25 = module_0.freeze(var_24)
    assert var_25 == 42

def test_case_10():
    var_0 = 1
    var_1 = 2
    var_2 = module_2.m()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = module_2.m(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = module_0.thaw(var_3)
    var_5 = 'hello'
    var_6 = module_0.thaw(var_5)
    assert var_6 == 'hello'

def test_case_11():
    var_0 = 1
    var_1 = 3
    var_2 = module_2.m()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = module_3.v()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 0
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_3.BRANCH_FACTOR == 32
    assert module_3.BIT_MASK == 31
    assert module_3.SHIFT == 5
    var_4 = (var_0, var_3)
    var_5 = module_0.thaw(var_4)
    var_6 = module_2.m()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = 42
    var_8 = module_0.thaw(var_7)
    assert var_8 == 42
    var_9 = 'hello'
    var_10 = module_0.thaw(var_9)
    assert var_10 == 'hello'
    var_11 = module_3.v()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_11) == 0
    var_12 = module_0.thaw(var_11)
    var_13 = module_2.m()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_13) == 0
    var_14 = module_0.thaw(var_13)
    var_15 = module_1.s()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_15) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_16 = module_0.thaw(var_15)
    var_17 = set()

def test_case_12():
    var_0 = 1
    var_1 = 2
    var_2 = -3133
    var_3 = 'a'
    var_4 = module_3.python_pvector()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_4) == 0
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_3.BRANCH_FACTOR == 32
    assert module_3.BIT_MASK == 31
    assert module_3.SHIFT == 5
    var_5 = module_0.thaw(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_1.pset(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_7) == 3
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_8 = module_0.thaw(var_7)
    var_9 = {var_3: var_0}
    var_10 = module_2.pmap(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_11 = [var_1, var_2]
    var_12 = module_1.pset(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_12) == 2

def test_case_13():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_2.pmap(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 2
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_7 = module_0.thaw(var_6)
    var_8 = module_0.thaw(var_1)
    assert var_8 == 2
    var_9 = {var_3: var_0}
    var_10 = module_2.pmap(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = [var_1, var_2]
    var_12 = module_1.pset(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_12) == 2
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'

def test_case_14():
    var_0 = 1
    var_1 = None
    var_2 = module_0.freeze(var_1, var_0)
    var_3 = 3
    var_4 = module_2.m()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = module_3.v()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_5) == 0
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_3.BRANCH_FACTOR == 32
    assert module_3.BIT_MASK == 31
    assert module_3.SHIFT == 5
    var_6 = module_0.thaw(var_4, var_4)
    var_7 = module_0.thaw(var_2)
    var_8 = module_0.thaw(var_6)
    var_9 = module_2.m()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = 42
    var_11 = module_0.thaw(var_10)
    assert var_11 == 42
    var_12 = 'hello'
    var_13 = module_0.thaw(var_12)
    assert var_13 == 'hello'
    var_14 = module_0.thaw(var_4)
    var_15 = module_2.m()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_16 = module_0.thaw(var_15)
    var_17 = module_1.s()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_17) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_18 = module_0.thaw(var_17)
    var_19 = set()

def test_case_15():
    var_0 = 1
    var_1 = 2
    var_2 = None
    var_3 = module_0.freeze(var_2, var_0)
    var_4 = module_2.m()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = module_3.v()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_5) == 0
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_3.BRANCH_FACTOR == 32
    assert module_3.BIT_MASK == 31
    assert module_3.SHIFT == 5
    var_6 = module_0.thaw(var_4, var_4)
    var_7 = (var_0, var_5)
    var_8 = module_0.thaw(var_7)
    var_9 = [var_1, var_4]
    var_10 = module_2.m()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 0
    var_11 = module_0.freeze(var_10, var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 0
    var_12 = 42
    var_13 = module_0.thaw(var_12)
    assert var_13 == 42
    var_14 = module_0.thaw(var_9)
    var_15 = module_2.m()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_15) == 0
    var_16 = module_0.thaw(var_15)
    var_17 = module_1.s()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_17) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_18 = module_0.thaw(var_17)
    var_19 = set()

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_4.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_4.defaultdict.default_factory).__module__}.{type(module_4.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_1.evolver()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap._Evolver'
    assert len(var_3) == 0
    var_4 = module_0.freeze(var_2)
    var_3.isspace()

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_1.pset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_4.defaultdict(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_4.defaultdict.default_factory).__module__}.{type(module_4.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_2 = module_0.thaw(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    var_3 = var_2.setdefault(var_0)
    assert len(var_1) == 1
    assert len(var_2) == 1
    var_4 = module_0.thaw(var_0)
    var_5 = var_0.__contains__(var_3)
    assert var_5 is False
    var_6 = module_0.freeze(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = var_2.values()
    var_7.index(var_7)