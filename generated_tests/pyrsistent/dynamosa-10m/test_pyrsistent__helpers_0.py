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
    var_0 = None
    var_1 = module_0.mutant(var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.thaw(var_0, var_0)

def test_case_5():
    var_0 = ()
    var_1 = module_0.thaw(var_0, var_0)

def test_case_6():
    var_0 = []
    var_1 = module_0.thaw(var_0)

def test_case_7():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0

def test_case_8():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0
    var_2 = module_0.thaw(var_1, var_1)

def test_case_9():
    var_0 = ()
    var_1 = module_0.freeze(var_0, var_0)

def test_case_10():
    var_0 = module_1.python_pvector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_1.BRANCH_FACTOR == 32
    assert module_1.BIT_MASK == 31
    assert module_1.SHIFT == 5
    var_1 = module_0.thaw(var_0, var_0)

def test_case_11():
    var_0 = module_2.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

def test_case_12():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0

def test_case_13():
    var_0 = False
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.thaw(var_1)
    var_3 = module_0.freeze(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 4

def test_case_14():
    var_0 = module_3.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_3.defaultdict.default_factory).__module__}.{type(module_3.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_1.python_pvector()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert module_1.BRANCH_FACTOR == 32
    assert module_1.BIT_MASK == 31
    assert module_1.SHIFT == 5
    var_2 = module_0.thaw(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    var_3 = module_0.freeze(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_3) == 0
    var_4 = module_0.thaw(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.defaultdict'
    assert len(var_4) == 0
    var_5 = module_0.mutant(var_1)
    var_6 = module_0.freeze(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0

def test_case_15():
    var_0 = module_2.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0, var_0)

def test_case_16():
    var_0 = module_2.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0, var_0)
    var_2 = module_0.freeze(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0

def test_case_17():
    var_0 = module_2.pmap()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_0) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.thaw(var_0)
    var_2 = module_0.thaw(var_1)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_2.pmap(pre_size=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = (var_1, var_1)
    var_3 = module_0.freeze(var_2)
    var_0.isalnum()

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = (var_0,)
    var_2 = module_0.thaw(var_1)
    var_1.format_map(var_2)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = None
    var_2 = module_0.mutant(var_1)
    var_3 = {var_0: var_2, var_2: var_1, var_0: var_0, var_0: var_0, var_1: var_1}
    var_4 = module_0.thaw(var_3)
    var_4.__delitem__(var_4)

def test_case_21():
    var_0 = module_3.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_3.defaultdict.default_factory).__module__}.{type(module_3.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

def test_case_22():
    var_0 = None
    var_1 = b'P\xe4&\r\xd6f\xacJ\xb2\x820\xd1S\xa3\x1a\xb5'
    var_2 = {var_1: var_0, var_1: var_1}
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 1
    var_4 = module_0.freeze(var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 1