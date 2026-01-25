# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._pbag as module_0
import pytest


def test_case_0():
    var_0 = b'\xe26\xa9TT\\K'
    var_1 = True
    var_2 = False
    var_3 = (var_0, var_0, var_1, var_2)
    var_4 = None
    var_5 = module_0.pbag(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_6 = var_5.update(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = var_6.__or__(var_3)
    var_8 = module_0.b()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_8) == 0
    var_9 = var_8.__and__(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = var_5.add(var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 1

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = [var_2, var_2, var_2, var_2]
    var_4 = module_0.b(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 4
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.remove(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 3
    var_5.__lt__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = None
    var_4 = var_2.__repr__()
    assert var_4 == 'pbag([pbag([])])'
    var_5 = module_0.b(*var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_2.add(var_3)

def test_case_3():
    var_0 = None
    var_1 = module_0.b()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.add(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = module_0.b(*var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_4 = var_3.add(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 1
    var_5 = var_2.__sub__(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0

def test_case_4():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.b()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = var_2.update(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_1.__add__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = var_1.__or__(var_3)
    var_8 = var_1.__len__()
    assert var_8 == 0
    var_9 = None
    var_10 = module_0.b()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 0
    with pytest.raises(TypeError):
        var_5.__eq__(var_9)

def test_case_5():
    var_0 = module_0.b()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__and__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    var_2 = var_1.add(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = module_0.b()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_4 = module_0.pbag(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = var_4.add(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_0.__iter__()
    var_7 = var_0.__and__(var_6)
    var_8 = var_2.__eq__(var_1)
    assert var_8 is False
    var_9 = var_5.__add__(var_8)
    var_10 = var_2.__sub__(var_8)
    var_11 = module_0.b()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_11) == 0
    var_12 = var_7.__lt__(var_7)
    var_13 = var_3.add(var_4)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_13) == 1
    var_14 = module_0.b()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_14) == 0
    var_15 = var_11.add(var_10)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_15) == 1
    var_16 = var_0.__iter__()
    var_17 = module_0.b()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_17) == 0
    var_18 = var_2.__and__(var_11)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_18) == 0
    var_19 = var_15.add(var_15)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_19) == 2
    var_20 = var_18.__sub__(var_18)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_20) == 0

def test_case_6():
    var_0 = module_0.b()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.add(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 1
    var_2 = module_0.b()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    var_3 = var_0.__sub__(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0

def test_case_7():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = module_0.pbag(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_5.add(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 2
    var_7 = var_6.__sub__(var_0)

def test_case_8():
    var_0 = None
    var_1 = module_0.b()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__or__(var_0)

def test_case_9():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_1]
    var_3 = module_0.b(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 2
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__and__(var_0)
    var_5 = var_4.__hash__()
    assert var_5 == 8066175221401
    var_6 = None
    var_7 = module_0.PBag(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'

def test_case_10():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.b()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0.b()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__repr__()
    assert var_1 == 'pbag([])'
    var_2 = None
    var_3 = module_0.PBag(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    var_4 = var_0.__iter__()
    var_4.__len__()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = None
    var_2 = module_0.pbag(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2.__lt__(var_0)

def test_case_13():
    var_0 = b'\xe26\xa9TT\\K'
    var_1 = True
    var_2 = False
    var_3 = (var_0, var_0, var_1, var_2)
    var_4 = None
    var_5 = None
    var_6 = module_0.pbag(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_7 = var_6.update(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 0
    var_8 = var_7.__or__(var_3)
    var_9 = module_0.b()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = var_9.__len__()
    assert var_10 == 0
    with pytest.raises(KeyError):
        var_7.remove(var_2)

def test_case_14():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = module_0.pbag(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.add(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 1
    var_5 = var_3.__iter__()
    var_6 = var_4.add(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 2
    var_7 = var_6.__sub__(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 1

def test_case_15():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.b()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    var_3 = var_1.add(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 1
    var_4 = module_0.b()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = var_3.__and__(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0
    var_6 = var_5.add(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = var_5.__sub__(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 0

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = None
    var_4 = var_2.update(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 2
    var_5 = var_1.__add__(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0
    var_6 = var_1.__sub__(var_0)
    var_7 = module_0.pbag(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 1
    var_8 = var_7.__contains__(var_2)
    assert var_8 is False
    var_9 = var_8.__eq__(var_4)
    var_10 = module_0.pbag(var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 0
    var_11 = var_7.add(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_11) == 2
    var_12 = module_0.b()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_12) == 0
    var_13 = var_12.add(var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_13) == 1
    var_14 = var_4.__iter__()
    var_15 = var_12.__and__(var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_15) == 0
    var_16 = var_5.__eq__(var_13)
    assert var_16 is False
    var_17 = var_13.add(var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_17) == 2
    var_18 = var_10.__iter__()
    var_14.__and__(var_4)

def test_case_17():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.PBag(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    var_3 = var_1.add(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 1
    var_4 = None
    var_5 = var_1.__add__(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_3.__or__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = var_3.__len__()
    assert var_7 == 1
    var_8 = module_0.b()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_8) == 0
    var_9 = module_0.b()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = module_0.PBag(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    with pytest.raises(TypeError):
        var_10.__eq__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.update(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    var_3 = var_2.__or__(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_4 = module_0.b()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = var_3.__len__()
    assert var_5 == 0
    var_5.count(var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.b()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    var_3 = var_1.add(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 1
    var_4 = var_2.__sub__(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = var_3.__and__(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_2.__add__(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = var_4.__contains__(var_5)
    assert var_7 is False
    var_7.__contains__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = var_1.__or__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 1
    var_4 = None
    var_5 = var_2.__repr__()
    assert var_5 == 'pbag([pbag([])])'
    var_6 = var_2.update(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = var_1.__add__(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 0
    var_8 = var_1.__len__()
    assert var_8 == 0
    var_9 = var_6.__hash__()
    assert var_9 == -6274228174228753898
    var_8.remove(var_0)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = module_0.b()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__and__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    var_3 = var_2.add(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 1
    var_4 = module_0.b()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = module_0.pbag(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0
    var_6 = var_5.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = var_1.__iter__()
    var_8 = var_1.__and__(var_7)
    var_9 = var_3.remove(var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = var_3.__eq__(var_2)
    assert var_10 is False
    var_11 = var_6.__add__(var_10)
    var_12 = var_3.__sub__(var_10)
    var_13 = var_6.__iter__()
    var_14 = module_0.b()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_14) == 0
    var_15 = var_8.__lt__(var_8)
    var_16 = var_4.add(var_5)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_16) == 1
    var_17 = module_0.b()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_17) == 0
    var_18 = var_14.add(var_12)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_18) == 1
    var_19 = var_1.__iter__()
    var_20 = var_4.__sub__(var_16)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_20) == 0
    var_21 = var_20.__and__(var_10)
    var_21.__add__(var_0)