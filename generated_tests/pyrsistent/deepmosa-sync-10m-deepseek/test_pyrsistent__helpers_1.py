# Check out: https://github.com/GlowCheese/deepmosa
import collections as module_2

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_3
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

def test_case_7():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)

def test_case_8():
    var_0 = ()
    var_1 = module_0.freeze(var_0)

def test_case_9():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = module_0.mutant(var_1)

def test_case_10():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = (var_0, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = module_0.thaw(var_5, var_3)
    var_7 = module_0.thaw(var_5, var_5)

def test_case_11():
    var_0 = ()
    var_1 = None
    var_2 = module_0.mutant(var_1)
    var_3 = module_0.freeze(var_0)
    var_4 = ()
    var_5 = module_0.thaw(var_3)
    var_6 = bool(var_3 == var_4)
    assert var_6 is True

@pytest.mark.xfail(strict=True)
def test_case_12():
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

def test_case_13():
    var_0 = 1
    var_1 = [var_0]
    var_2 = lambda x: x + var_1
    var_3 = module_0.mutant(var_2)
    var_4 = [var_3]
    with pytest.raises(NameError):
        var_5 = var_3(var_4)

def test_case_14():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0

def test_case_15():
    var_0 = lambda x, y: x + y
    var_1 = module_0.mutant(var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = 2
    var_5 = [var_4]
    var_6 = var_1(var_3, y=var_5)

def test_case_16():
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
def test_case_17():
    var_0 = {}
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.freeze(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.thaw(var_2)
    var_2.split(var_2, var_1)

def test_case_18():
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.mutant(var_2)
    with pytest.raises(TypeError):
        var_5 = var_4(var_3)

def test_case_19():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_1.pmap(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = True
    var_5 = module_0.freeze(var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 1
    var_6 = module_0.thaw(var_3, var_5)
    var_7 = {var_0: var_4}
    var_8 = module_1.pmap(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 1
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

def test_case_20():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_2.defaultdict(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.defaultdict'
    assert len(var_5) == 1
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_6 = False
    var_7 = module_0.freeze(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 1
    var_8 = bool(var_7 is var_5)

def test_case_21():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0

def test_case_22():
    var_0 = 1
    var_1 = [var_0]
    var_2 = lambda x=[]: x + var_1
    var_3 = module_0.mutant(var_2)
    var_4 = 0
    var_5 = [var_4]
    with pytest.raises(NameError):
        var_6 = var_3(x=var_5)

def test_case_23():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 3
    var_5 = {var_0, var_1, var_2}
    var_6 = module_0.mutant(var_4)
    var_7 = module_0.mutant(var_4)
    var_8 = module_3.pset(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_8) == 3
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    var_9 = bool(var_4 == var_8)
    assert var_9 is True
    var_10 = module_0.thaw(var_8, var_8)