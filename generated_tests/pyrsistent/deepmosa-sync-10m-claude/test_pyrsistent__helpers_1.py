# Check out: https://github.com/GlowCheese/deepmosa
import collections as module_2

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
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
    var_1 = 3
    var_2 = (var_0, var_0, var_1)
    var_3 = None
    var_4 = module_0.thaw(var_3)
    var_5 = module_0.freeze(var_2)

def test_case_8():
    var_0 = ()
    var_1 = module_0.freeze(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True
    var_3 = '@-X[?aV7i'
    var_4 = module_0.thaw(var_3)
    assert var_4 == '@-X[?aV7i'

def test_case_9():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = module_0.mutant(var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = module_0.mutant(var_0)
    var_2 = module_0.mutant(var_0)
    var_3 = (var_1, var_1)
    var_4 = module_0.freeze(var_3)
    var_5 = module_0.thaw(var_1)
    var_6 = module_0.freeze(var_4, var_5)
    var_7 = module_0.thaw(var_6)
    var_7.__reduce__()

def test_case_11():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = [var_2, var_2]
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = module_0.freeze(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 2
    var_9 = module_0.thaw(var_8)
    var_10 = module_0.thaw(var_9)
    var_11 = {var_4: var_5}
    var_12 = module_1.pmap(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'

def test_case_12():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0
    var_2 = list(var_1)
    var_3 = set()
    var_4 = set()
    var_5 = bool(var_4 == var_4)
    assert var_5 is True
    var_6 = module_0.freeze(var_3, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_6) == 0
    var_7 = ()
    var_8 = module_0.freeze(var_7)
    var_9 = module_0.thaw(var_2)
    var_10 = bool(var_8 == ())
    assert var_10 is True

def test_case_13():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.thaw(var_0)
    assert var_2 == 1
    var_3 = 27
    var_4 = [var_0, var_1, var_3]
    var_5 = module_0.freeze(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_5) == 3
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2, 3])

def test_case_14():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_15():
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
    var_0 = None
    var_1 = '+= tgR9EoH@.1'
    var_2 = "W(/A/\r'Q+\x0c@+"
    var_3 = 'K37Rsu !]P2zwOM\x0b'
    var_4 = 'Q@<#aP\\_,'
    var_5 = {var_1: var_0, var_2: var_0, var_3: var_0, var_4: var_0}
    var_6 = module_2.defaultdict(**var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.defaultdict'
    assert len(var_6) == 4
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_7 = module_0.thaw(var_3)
    assert var_7 == 'K37Rsu !]P2zwOM\x0b'
    var_8 = module_0.freeze(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_8) == 4
    var_9 = module_0.freeze(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 4
    var_10 = module_0.freeze(var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_10) == 4

def test_case_19():
    var_0 = 2
    var_1 = 'a'
    var_2 = 'b'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_1.m(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 2
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_5 = module_0.thaw(var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})

def test_case_20():
    var_0 = module_2.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

def test_case_21():
    var_0 = set()
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_1) == 0

def test_case_22():
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