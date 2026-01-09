# Check out: https://github.com/GlowCheese/deepmosa
import collections as module_1

import pyrsistent._helpers as module_0
import pyrsistent._pvector as module_2
import pytest


def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0.thaw(var_0)

def test_case_2():
    var_0 = None
    var_1 = module_0.mutant(var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.thaw(var_0, var_0)

def test_case_5():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

def test_case_6():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = module_0.freeze(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0

def test_case_7():
    var_0 = {}
    var_1 = module_0.thaw(var_0)

def test_case_8():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = module_0.thaw(var_1)

def test_case_9():
    var_0 = 2139.21468
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.freeze(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1

def test_case_10():
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.freeze(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 1
    var_3 = module_0.thaw(var_2)

def test_case_11():
    var_0 = []
    var_1 = module_0.thaw(var_0)

def test_case_12():
    var_0 = ()
    var_1 = {var_0, var_0, var_0, var_0}
    var_2 = module_0.freeze(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_2) == 1

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.freeze(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 4
    var_2.__setitem__(var_2, var_2)

def test_case_14():
    var_0 = []
    var_1 = module_0.freeze(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0

def test_case_15():
    var_0 = -2263.0
    var_1 = (var_0,)
    var_2 = module_0.freeze(var_1, var_0)

def test_case_16():
    var_0 = ()
    var_1 = module_0.freeze(var_0)

def test_case_17():
    var_0 = module_1.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_2.python_pvector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_2.BRANCH_FACTOR == 32
    assert module_2.BIT_MASK == 31
    assert module_2.SHIFT == 5
    var_2 = module_0.freeze(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 0

def test_case_18():
    var_0 = []
    var_1 = module_0.freeze(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0
    var_2 = module_0.thaw(var_1)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.freeze(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 4
    var_3 = module_0.thaw(var_2, var_0)
    var_0.__setitem__(var_0, var_2)

def test_case_20():
    var_0 = -2263.0
    var_1 = (var_0,)
    var_2 = module_0.thaw(var_1)

def test_case_21():
    var_0 = ()
    var_1 = module_0.thaw(var_0)

def test_case_22():
    var_0 = ()
    var_1 = {var_0, var_0, var_0, var_0}
    var_2 = module_0.freeze(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_2) == 1
    var_3 = module_0.thaw(var_2)

def test_case_23():
    var_0 = module_1.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_1.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = module_0.thaw(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.__setitem__(var_3, var_2)
    assert len(var_0) == 1
    assert len(var_2) == 1
    module_0.freeze(var_2, var_3)