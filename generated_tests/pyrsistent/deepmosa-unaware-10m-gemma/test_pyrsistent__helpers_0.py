# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import collections as module_2

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0.thaw(var_0, var_0)

def test_case_2():
    var_0 = None
    var_1 = module_0.mutant(var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.thaw(var_0)

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
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)
    var_2 = module_0.freeze(var_0, var_0)
    var_3 = module_0.thaw(var_0, var_0)
    var_4 = None
    var_5 = module_0.mutant(var_4)
    var_6 = module_0.mutant(var_4)
    var_7 = module_0.thaw(var_4)
    var_8 = module_0.mutant(var_5)
    var_9 = [var_6]
    var_10 = module_0.freeze(var_9, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_10) == 1
    var_11 = module_0.thaw(var_10)
    var_8.__add__(var_5)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)
    var_2 = module_0.freeze(var_0, var_0)
    var_3 = module_0.thaw(var_0, var_0)
    var_4 = None
    var_5 = module_0.mutant(var_4)
    var_6 = module_0.mutant(var_4)
    var_7 = module_0.thaw(var_4)
    var_8 = module_0.mutant(var_5)
    var_9 = None
    var_10 = [var_6]
    var_11 = module_0.freeze(var_10, var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_11) == 1
    var_12 = module_0.mutant(var_9)
    var_8.update(var_5)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = []
    var_1 = module_0.freeze(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0
    var_2 = module_0.freeze(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 0
    var_3 = module_0.thaw(var_2)
    var_4 = module_0.mutant(var_3)
    var_2.__missing__(var_1)

def test_case_9():
    var_0 = []
    var_1 = module_0.thaw(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = ()
    var_1 = module_0.mutant(var_0)
    var_2 = None
    var_3 = module_0.freeze(var_0, var_1)
    var_4 = module_0.mutant(var_1)
    var_5 = module_0.thaw(var_4)
    var_6 = module_0.mutant(var_2)
    var_7 = module_0.thaw(var_2)
    var_5.__iadd__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_0.mutant(var_0)
    var_2 = None
    var_3 = {var_2, var_1, var_0}
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 2
    var_5 = module_0.thaw(var_4)
    var_4.total()

def test_case_12():
    var_0 = {}
    var_1 = module_0.thaw(var_0)

def test_case_13():
    var_0 = None
    var_1 = module_0.mutant(var_0)
    var_2 = module_1.pmap(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.thaw(var_2)
    var_4 = module_0.mutant(var_2)
    var_5 = module_0.thaw(var_1)
    var_6 = module_0.freeze(var_2, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = module_0.mutant(var_0)
    var_2 = module_0.thaw(var_0, var_0)
    var_3 = module_0.freeze(var_2, var_0)
    var_4 = module_0.thaw(var_3)
    var_5 = None
    var_6 = module_0.freeze(var_5)
    var_7 = (var_6, var_6)
    var_8 = module_0.freeze(var_7)
    var_9 = module_0.freeze(var_1, var_3)
    var_10 = module_0.mutant(var_4)
    var_11 = module_0.thaw(var_8)
    var_12 = module_0.mutant(var_3)
    var_13 = module_0.thaw(var_11)
    var_14 = module_0.thaw(var_12)
    var_15 = module_0.mutant(var_8)
    var_9.remove(var_14)

def test_case_15():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_2.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0
    var_2 = module_0.thaw(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    var_3 = module_0.freeze(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = module_0.thaw(var_3)
    var_3.isalnum()

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = []
    var_1 = None
    var_2 = module_0.thaw(var_1)
    var_3 = None
    var_4 = var_2.__lt__(var_3)
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True
    var_6 = module_0.thaw(var_5)
    assert var_6 is True
    var_7 = module_0.freeze(var_1)
    var_8 = module_0.thaw(var_0)
    var_9 = module_0.mutant(var_3)
    var_10 = module_0.thaw(var_9)
    var_11 = {var_3: var_0}
    var_12 = module_0.freeze(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = module_0.mutant(var_9)
    var_14 = module_0.thaw(var_12)
    var_15 = module_0.mutant(var_1)
    var_15.__add__(var_5)

def test_case_18():
    var_0 = module_2.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_2.defaultdict.default_factory).__module__}.{type(module_2.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0