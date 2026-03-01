# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._helpers as module_0
import collections as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.freeze(var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0.thaw(var_0, var_0)

def test_case_2():
    var_0 = bool(not (not False and True))

def test_case_3():
    var_0 = None
    var_1 = module_0.mutant(var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.freeze(var_0, var_0)

def test_case_5():
    var_0 = None
    var_1 = module_0.thaw(var_0)

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
    var_9 = [var_6]
    var_10 = module_0.freeze(var_9, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_10) == 1
    var_11 = module_0.thaw(var_10)
    var_8.__add__(var_5)

def test_case_8():
    var_0 = []
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_1) == 0

def test_case_9():
    var_0 = []
    var_1 = None
    var_2 = module_0.freeze(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_2) == 0
    var_3 = None
    var_4 = module_0.freeze(var_3, var_3)
    var_5 = module_0.freeze(var_4)
    var_6 = module_0.thaw(var_3)
    var_7 = module_0.freeze(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_7) == 0
    var_8 = None
    var_9 = module_0.thaw(var_8)
    var_10 = module_0.freeze(var_8)
    var_11 = lambda x: x
    var_12 = module_0.mutant(var_11)
    var_13 = var_12(var_12)

def test_case_10():
    var_0 = []
    var_1 = None
    var_2 = module_1.defaultdict(*var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = module_0.thaw(var_0)
    var_5 = module_0.freeze(var_1)
    var_6 = module_0.mutant(var_4)

@pytest.mark.xfail(strict=True)
def test_case_11():
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
def test_case_12():
    var_0 = None
    var_1 = module_0.mutant(var_0)
    var_2 = None
    var_3 = {var_2, var_1, var_0}
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pset.PSet'
    assert len(var_4) == 2
    var_5 = module_0.thaw(var_4)
    var_4.total()

def test_case_13():
    var_0 = -1964.3672
    var_1 = None
    var_2 = module_0.freeze(var_1, var_0)
    var_3 = [var_0, var_0, var_0, var_0]
    var_4 = module_0.freeze(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pvector.PythonPVector'
    assert len(var_4) == 4
    var_5 = None
    var_6 = module_0.thaw(var_5)
    var_7 = None
    var_8 = module_0.thaw(var_7)
    var_9 = module_0.freeze(var_7)
    var_10 = lambda x: x
    var_11 = module_0.mutant(var_10)
    var_12 = var_11(var_11)

def test_case_14():
    var_0 = []
    var_1 = module_1.defaultdict(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_2 = module_1.defaultdict(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    var_3 = module_0.thaw(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.defaultdict'
    assert len(var_3) == 0
    var_4 = module_0.freeze(var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_4) == 0
    var_5 = module_0.thaw(var_4)

@pytest.mark.xfail(strict=True)
def test_case_15():
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

def test_case_16():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    var_2 = None
    var_3 = (var_1, var_1)
    var_4 = module_0.thaw(var_3, var_1)
    var_5 = module_0.thaw(var_0)
    var_6 = module_0.thaw(var_2)
    var_7 = module_1.defaultdict()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.defaultdict'
    assert len(var_7) == 0
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_8 = module_0.mutant(var_7)
    with pytest.raises(TypeError):
        var_9 = var_8(var_7)
    assert var_9 == 1

def test_case_17():
    var_0 = {}
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

def test_case_18():
    var_0 = []
    var_1 = module_1.defaultdict(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.defaultdict'
    assert len(var_1) == 0
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_2 = module_0.freeze(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_2) == 0
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = []
    var_1 = None
    var_2 = module_1.defaultdict(*var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.defaultdict'
    assert len(var_2) == 0
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_3 = module_0.freeze(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_3) == 0
    var_4 = var_2.__lt__(var_1)
    var_5 = module_0.thaw(var_0)
    var_6 = module_0.freeze(var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_6) == 0
    var_7 = module_0.freeze(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_7) == 0
    var_8 = module_0.thaw(var_6)
    var_9 = module_0.mutant(var_5)
    var_10 = module_0.thaw(var_9)
    var_11 = {var_4: var_10}
    var_12 = module_0.freeze(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_12) == 1
    var_13 = module_0.mutant(var_2)
    var_14 = module_0.thaw(var_2, var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.defaultdict'
    assert len(var_14) == 0
    var_12.find(var_3)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.freeze(var_0)
    var_2 = []
    var_3 = None
    var_4 = module_1.defaultdict(*var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.defaultdict'
    assert len(var_4) == 0
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_5 = module_0.freeze(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_5) == 0
    var_6 = None
    var_7 = var_4.__lt__(var_6)
    var_8 = var_4.__eq__(var_4)
    assert var_8 is True
    var_9 = module_0.freeze(var_4, var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_9) == 0
    var_10 = module_0.freeze(var_7, var_7)
    var_11 = module_0.freeze(var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_11) == 0
    var_12 = module_0.thaw(var_9)
    var_13 = module_0.mutant(var_6)
    var_14 = module_0.thaw(var_13)
    var_15 = {var_7: var_14}
    var_16 = module_0.mutant(var_9)
    var_17 = module_0.freeze(var_15)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 1
    var_18 = module_0.mutant(var_4)
    var_19 = module_0.thaw(var_4, var_3)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.defaultdict'
    assert len(var_19) == 0
    var_20 = module_0.mutant(var_13)
    var_21 = module_0.thaw(var_17)
    var_4.count(var_17)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = []
    var_1 = None
    var_2 = var_1.__lt__(var_1)
    var_3 = var_0.__eq__(var_0)
    assert var_3 is True
    var_4 = module_0.thaw(var_3)
    assert var_4 is True
    var_5 = module_0.freeze(var_2, var_2)
    var_6 = module_0.freeze(var_5)
    var_7 = module_0.thaw(var_5)
    var_8 = module_0.mutant(var_7)
    var_9 = module_0.thaw(var_8)
    var_10 = {}
    var_11 = module_0.thaw(var_6)
    var_12 = module_0.mutant(var_8)
    var_13 = module_0.thaw(var_10)
    var_14 = module_0.mutant(var_1)
    var_6.count(var_9)

def test_case_22():
    var_0 = module_1.defaultdict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'collections.defaultdict'
    assert len(var_0) == 0
    assert f'{type(module_1.defaultdict.default_factory).__module__}.{type(module_1.defaultdict.default_factory).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.freeze(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_1) == 0

def test_case_23():
    var_0 = lambda x: x
    var_1 = module_0.mutant(var_0)
    var_2 = var_1(var_0)