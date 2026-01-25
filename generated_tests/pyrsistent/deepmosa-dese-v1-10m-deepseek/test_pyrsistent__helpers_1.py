# Check out: https://github.com/GlowCheese/deepmosa
import collections as module_2

import pyrsistent._helpers as module_0
import pyrsistent._pset as module_3
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

@pytest.mark.xfail(strict=True)
def test_case_11():
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

def test_case_12():
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
def test_case_13():
    var_0 = {}
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.freeze(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.thaw(var_2)
    var_2.split(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
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

def test_case_15():
    var_0 = module_2.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

def test_case_16():
    var_0 = module_3.pset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_0) == 0
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0, var_0)
    var_2 = None
    var_3 = module_0.mutant(var_2)
    var_4 = module_0.mutant(var_2)
    var_5 = module_0.thaw(var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_0.thaw(var_2)
    var_4 = module_0.freeze(var_0, var_3)
    var_5 = module_2.defaultdict(*var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.defaultdict'
    assert len(var_5) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_6 = module_0.mutant(var_1)
    var_7 = module_0.freeze(var_3, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = module_0.freeze(var_0, var_7)
    var_9 = module_0.freeze(var_0, var_0)
    var_10 = module_0.freeze(var_6)
    var_11 = None
    var_12 = module_0.thaw(var_0)
    var_13 = module_0.mutant(var_1)
    var_14 = var_6.__str__()
    var_15 = {var_6, var_14}
    var_16 = module_0.freeze(var_15, var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_16) == 2
    var_17 = module_0.freeze(var_0, var_11)
    var_17.strip(var_17)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    var_2 = {}
    var_3 = module_0.thaw(var_2)
    var_4 = module_2.defaultdict(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.defaultdict'
    assert len(var_4) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_4.__setitem__(var_0, var_0)
    assert len(var_4) == 1
    var_6 = var_1.__repr__()
    assert var_6 == 'None'
    var_7 = module_0.freeze(var_4, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = module_0.freeze(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 0
    var_9 = module_0.freeze(var_4, var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 1
    var_10 = module_0.freeze(var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 1
    var_11 = var_8.__eq__(var_1)
    var_12 = var_9.__repr__()
    assert var_12 == 'pmap({None: None})'
    var_11.__complex__()