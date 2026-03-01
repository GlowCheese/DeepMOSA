# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._pbag as module_0
import pyrsistent._pmap as module_1
import builtins as module_2

def test_case_0():
    var_0 = '4yv:fO'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.b(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__repr__()
    assert var_3 == "pbag(['4yv:fO'])"
    var_4 = None
    var_5 = var_2.__sub__(var_4)
    var_6 = var_2.update(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 2
    with pytest.raises(TypeError):
        var_6.__eq__(var_1)

def test_case_1():
    var_0 = None
    var_1 = module_0.b()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.count(var_1)
    assert var_2 == 0
    var_3 = var_1.update(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_4 = var_1.__or__(var_2)
    var_5 = var_1.add(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_5.__hash__()
    assert var_6 == -6274228174228753898

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '4yv:fO'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.b(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.pbag(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 1
    var_4 = var_2.__repr__()
    assert var_4 == "pbag(['4yv:fO'])"
    var_5 = None
    var_6 = var_2.__and__(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = var_2.__sub__(var_5)
    var_4.__add__(var_7)

def test_case_3():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__repr__()
    assert var_2 == 'pbag([])'
    var_3 = var_1.__add__(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_4 = var_0.__hash__()
    assert var_4 == 8282064600984
    var_5 = module_0.PBag(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    var_6 = var_3.add(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1

def test_case_4():
    var_0 = None
    var_1 = None
    var_2 = module_0.b()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.add(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 1
    with pytest.raises(TypeError):
        var_3.__eq__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = ()
    var_2 = module_0.pbag(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__add__(var_0)
    var_4 = var_3.__hash__()
    assert var_4 == 8282064600985
    var_4.__iter__()

def test_case_6():
    var_0 = None
    var_1 = ()
    var_2 = module_0.pbag(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__sub__(var_0)
    var_4 = var_2.__add__(var_3)
    with pytest.raises(KeyError):
        var_2.remove(var_4)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.b()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    var_3 = var_1.__or__(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_3.__lt__(var_2)

def test_case_8():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = 'g{t.F_i01'
    var_3 = None
    var_4 = module_0.PBag(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    var_5 = var_4.__and__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.b()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = True
    module_0.pbag(var_1)

def test_case_10():
    var_0 = None
    var_1 = None
    var_2 = module_0.pbag(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__contains__(var_0)
    assert var_3 is False

def test_case_11():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.pbag(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    var_3 = var_1.__add__(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_4 = var_2.__hash__()
    assert var_4 == 133146708735736
    var_5 = var_3.__iter__()

def test_case_12():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.pbag(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    var_3 = var_1.__len__()
    assert var_3 == 0
    var_4 = var_2.__sub__(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = var_4.__repr__()
    assert var_5 == 'pbag([])'
    var_6 = module_0.pbag(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = module_0.PBag(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    var_8 = var_1.add(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_8) == 1
    var_9 = var_1.__and__(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0

def test_case_13():
    var_0 = '4yv:fO'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.b(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__or__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 1
    var_4 = None
    var_5 = var_2.__sub__(var_4)
    var_6 = var_2.update(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 2
    with pytest.raises(TypeError):
        var_6.__eq__(var_1)

def test_case_14():
    var_0 = '4yv:fO'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.b(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 2
    var_4 = var_2.__repr__()
    assert var_4 == "pbag(['4yv:fO'])"
    var_5 = var_2.__and__(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_2.__sub__(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = var_5.__hash__()
    assert var_7 == 8647032160749056065
    var_8 = var_6.__sub__(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_8) == 0
    var_9 = None
    var_10 = var_2.__and__(var_9)
    var_11 = module_0.b(*var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_11) == 0
    with pytest.raises(KeyError):
        var_8.remove(var_10)

def test_case_15():
    var_0 = "w'EAa\x0cRq6VW4@f"
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.b(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__sub__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_4 = var_2.update(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 2
    with pytest.raises(TypeError):
        var_4.__eq__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.pbag(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    var_3 = var_2.__and__(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_4 = var_2.__repr__()
    assert var_4 == 'pbag([])'
    var_5 = var_1.__and__(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0
    var_6 = var_2.__or__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = var_2.__eq__(var_5)
    assert var_7 is True
    var_8 = var_3.__repr__()
    assert var_8 == 'pbag([])'
    var_6.__lt__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = '4yv:fO'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.b(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 2
    var_4 = var_2.__repr__()
    assert var_4 == "pbag(['4yv:fO'])"
    var_5 = None
    var_6 = var_3.__sub__(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = var_3.update(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 2
    var_8 = var_6.__eq__(var_3)
    assert var_8 is False
    var_9 = None
    var_10 = var_3.__repr__()
    assert var_10 == "pbag(['4yv:fO', '4yv:fO'])"
    var_11 = var_4.__lt__(var_10)
    assert var_11 is False
    var_12 = module_0.pbag(var_8)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_12) == 0
    var_13 = module_0.PBag(var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pbag.PBag'
    var_4.add(var_9)

def test_case_18():
    var_0 = 'j'
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.b(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = var_2.update(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 2
    var_4 = var_1.__or__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 2
    var_5 = var_3.__add__(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 3
    var_6 = var_2.__and__(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = var_2.__sub__(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 0
    var_8 = var_4.add(var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_8) == 3
    var_9 = var_7.__and__(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = var_9.__or__(var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 1
    var_11 = var_1.__eq__(var_5)
    assert var_11 is False
    var_12 = var_6.__hash__()
    assert var_12 == -791900924879735308
    var_13 = var_10.__and__(var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_13) == 1
    var_14 = module_0.b()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_14) == 0
    var_15 = var_3.add(var_6)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_15) == 3
    var_16 = var_15.remove(var_2)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_16) == 2
    var_17 = module_1.pmap()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pmap.PMap'
    assert len(var_17) == 0
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_18 = var_5.__and__(var_14)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_18) == 0
    var_19 = var_15.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'j'
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.b(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = var_2.update(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 2
    var_4 = var_3.__add__(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 3
    var_5 = var_3.__iter__()
    var_6 = var_2.__and__(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = var_2.__and__(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 1
    var_8 = var_7.__or__(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_8) == 1
    var_9 = var_1.__eq__(var_4)
    assert var_9 is False
    var_10 = var_6.__hash__()
    assert var_10 == -791900924879735308
    var_11 = module_0.PBag(var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pbag.PBag'
    var_12 = var_8.__and__(var_10)
    var_13 = module_0.b()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_13) == 0
    var_11.remove(var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'x'
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.b(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = var_2.update(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 2
    var_4 = var_3.__add__(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 3
    var_5 = module_2.object()
    var_6 = var_2.__and__(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = None
    var_8 = var_6.add(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_8) == 2
    var_9 = var_2.__sub__(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = var_9.__and__(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 0
    var_11 = var_10.__or__(var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_11) == 1
    var_12 = var_1.__hash__()
    assert var_12 == 8089643079168336415
    var_13 = var_1.__eq__(var_4)
    assert var_13 is False
    var_14 = var_13.__hash__()
    assert var_14 == 0
    var_15 = var_4.update(var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_15) == 3
    var_16 = var_11.__and__(var_14)
    var_17 = module_0.b()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_17) == 0
    var_18 = var_15.remove(var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_18) == 2
    module_1.pmap(var_11)