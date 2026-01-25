# Check out: https://github.com/GlowCheese/deepmosa
import collections as module_2

import pyrsistent._helpers as module_0
import pyrsistent._pvector as module_1
import pytest


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
    var_0 = {}
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = module_0.thaw(var_1)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.mutant(var_0)
    var_2 = module_0.mutant(var_0)
    var_3 = (var_1, var_1)
    var_4 = module_0.freeze(var_3)
    var_5 = module_0.thaw(var_1)
    var_6 = module_0.freeze(var_5, var_5)
    var_7 = module_0.mutant(var_2)
    var_1.__len__()

def test_case_8():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = module_0.mutant(var_0)
    var_2 = module_0.mutant(var_0)
    var_3 = (var_1, var_1)
    var_4 = module_0.freeze(var_3)
    var_5 = module_0.thaw(var_1)
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = module_0.thaw(var_6)
    var_7.__reduce__()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 2053
    var_1 = [var_0]
    var_2 = module_0.thaw(var_1)
    var_3 = module_0.freeze(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 1
    var_4 = module_0.mutant(var_3)
    var_5 = module_0.freeze(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_5) == 1
    var_6 = module_0.mutant(var_2)
    var_7 = None
    var_8 = module_0.mutant(var_0)
    var_9 = var_2.pop()
    assert var_9 == 2053
    var_5.endswith(var_7)

def test_case_11():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 2063
    var_1 = module_1.python_pvector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_1.BRANCH_FACTOR == 32
    assert module_1.BIT_MASK == 31
    assert module_1.SHIFT == 5
    var_2 = module_0.thaw(var_1)
    var_3 = module_0.thaw(var_2)
    var_4 = module_0.freeze(var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_4) == 0
    var_5 = module_0.mutant(var_4)
    var_6 = module_0.mutant(var_3)
    var_7 = module_0.mutant(var_0)
    var_8 = var_1.evolver()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pvector.PythonPVector.Evolver'
    assert len(var_8) == 0
    var_9 = var_4.mset(*var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_9) == 0
    var_10 = module_0.thaw(var_6)
    var_6.__reduce__()

def test_case_13():
    var_0 = {}
    var_1 = None
    var_2 = module_0.thaw(var_0, var_1)
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = module_0.mutant(var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = {}
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.freeze(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.thaw(var_2)
    var_2.split(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    var_2 = {var_1: var_0, var_1: var_1}
    var_3 = module_0.thaw(var_2)
    var_4 = var_3.__repr__()
    assert var_4 == '{None: None}'
    var_5 = module_0.mutant(var_4)
    var_6 = module_0.freeze(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = module_0.freeze(var_4, var_3)
    assert var_7 == '{None: None}'
    var_8 = None
    var_9 = module_0.thaw(var_8, var_4)
    var_9.setdefault(var_8)

def test_case_16():
    var_0 = module_2.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

def test_case_17():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = module_0.thaw(var_1)

def test_case_18():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0

def test_case_19():
    var_0 = module_2.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = None
    var_2 = var_0.__setitem__(var_1, var_1)
    assert len(var_0) == 1
    var_3 = module_0.thaw(var_1)
    var_4 = module_0.freeze(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1
    var_5 = module_0.freeze(var_3)