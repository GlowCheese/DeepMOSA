# Check out: https://github.com/GlowCheese/deepmosa
import collections as module_3

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_2
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
    var_0 = 'x'
    var_1 = 'z'
    var_2 = 1
    var_3 = 2
    var_4 = 'y'
    var_5 = module_1.python_pvector()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_5) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_1.BRANCH_FACTOR == 32
    assert module_1.BIT_MASK == 31
    assert module_1.SHIFT == 5
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = 5
    var_9 = -454
    var_10 = {var_8, var_9}
    var_11 = {var_0: var_7, var_1: var_10}
    var_12 = module_0.freeze(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 2
    var_13 = bool(var_12 == {'x': [1, 2, {'y': (3, 4)}], 'z': {5, 6}})

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = []
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.mutant(var_1)
    var_3 = None
    var_4 = var_1.__lt__(var_3)
    var_5 = module_0.freeze(var_2, var_1)
    var_4.keys()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = ()
    var_1 = module_0.mutant(var_0)
    var_2 = None
    var_3 = module_0.mutant(var_2)
    var_4 = module_0.mutant(var_2)
    var_5 = module_0.freeze(var_0)
    var_6 = None
    var_7 = module_0.freeze(var_5)
    var_8 = module_0.mutant(var_2)
    var_3.update(var_6)

def test_case_11():
    var_0 = module_2.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0)

def test_case_12():
    var_0 = 'x'
    var_1 = 3
    var_2 = 4
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = bool(var_5 == {'x': [1, 2, {'y': (3, 4)}], 'z': {5, 6}})

def test_case_13():
    var_0 = module_2.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.freeze(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0

@pytest.mark.xfail(strict=True)
def test_case_14():
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
def test_case_15():
    var_0 = None
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.freeze(var_0)
    var_3 = module_0.mutant(var_1)
    var_4 = var_1.__lt__(var_1)
    var_5 = module_2.pmap()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_6 = module_0.thaw(var_5)
    var_7 = module_0.freeze(var_3, var_1)
    var_8 = module_0.mutant(var_7)
    var_9 = module_0.thaw(var_6)
    var_10 = module_0.freeze(var_0, var_9)
    var_11 = module_0.mutant(var_4)
    var_3.__reversed__()

def test_case_16():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 3
    var_5 = module_0.thaw(var_4)
    var_6 = bool(var_4 == {1, 2, 3})
    assert var_6 is True

def test_case_17():
    var_0 = None
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.freeze(var_0)
    var_3 = module_0.mutant(var_1)
    var_4 = module_2.pmap()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
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

def test_case_18():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0.mutant(var_0)
    var_3 = ''
    var_4 = {var_3: var_0}
    var_5 = module_3.defaultdict(*var_1, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.defaultdict'
    assert len(var_5) == 1
    assert f'{type(module_3.defaultdict.default_factory).__module__}.{type(module_3.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_6 = module_0.freeze(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 1
    var_7 = ''
    var_8 = module_0.thaw(var_2)
    var_9 = module_0.freeze(var_7)
    assert var_9 == ''
    var_10 = module_0.mutant(var_9)
    var_5.replace(var_7, var_9)