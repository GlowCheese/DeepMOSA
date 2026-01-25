# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._pbag as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = module_0.pbag(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__sub__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = var_4.update(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0
    with pytest.raises(KeyError):
        var_3.remove(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = [var_0, var_0, var_0, var_0]
    var_5 = module_0.b(*var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 4
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_6 = var_5.remove(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 3
    var_7 = var_6.remove(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 2
    var_8 = var_7.__or__(var_3)
    var_9 = var_6.__sub__(var_8)
    var_10 = var_5.__and__(var_1)
    var_7.__lt__(var_7)

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
    var_5 = var_2.update(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_1.__add__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = var_1.__or__(var_3)
    var_8 = var_1.__len__()
    assert var_8 == 0
    var_9 = module_0.b()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = module_0.b()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 0
    var_11 = var_1.__contains__(var_7)
    assert var_11 is False
    var_7.add(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
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
    var_11 = var_1.__contains__(var_7)
    assert var_11 is False
    with pytest.raises(TypeError):
        var_5.__eq__(var_9)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = None
    var_4 = module_0.pbag(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = var_1.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_5.__or__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = var_1.__add__(var_0)
    var_8 = var_6.__sub__(var_3)
    var_9 = module_0.b()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_4.update(var_7)

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
    assert var_5 == 8410885330585
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
    var_0 = None
    var_1 = module_0.b()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = module_0.b()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_4 = var_3.__and__(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = module_0.pbag(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0
    var_5.__lt__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = True
    var_1 = None
    var_2 = module_0.pbag(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.update(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_3) == 0
    var_4 = var_3.__or__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 0
    var_5 = var_2.count(var_3)
    assert var_5 == 0
    var_6 = module_0.PBag(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    var_5.count(var_5)

def test_case_13():
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

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == 'pbag([pbag([])])'
    var_4 = var_2.update(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 1
    var_5 = var_1.__add__(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0
    var_6 = var_1.__or__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = module_0.pbag(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 1
    var_8 = var_1.__len__()
    assert var_8 == 0
    var_9 = module_0.b()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = module_0.b()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 0
    var_11 = var_1.__contains__(var_6)
    assert var_11 is False
    var_12 = module_0.pbag(var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_12) == 0
    var_13 = var_3.__len__()
    assert var_13 == 16
    var_14 = var_4.update(var_2)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_14) == 2
    var_15 = var_2.__and__(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_15) == 1
    var_14.__contains__(var_13)

def test_case_16():
    var_0 = module_0.b()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True

def test_case_17():
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
    var_5 = var_2.update(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_1.__add__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = var_1.__or__(var_3)
    var_8 = var_1.__len__()
    assert var_8 == 0
    var_9 = module_0.b()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = module_0.b()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 0
    var_11 = var_1.__contains__(var_7)
    assert var_11 is False
    var_12 = module_0.pbag(var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_12) == 0
    var_13 = var_9.add(var_10)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_13) == 1
    var_14 = var_2.__and__(var_5)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_14) == 1
    var_15 = var_6.add(var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_15) == 1

@pytest.mark.xfail(strict=True)
def test_case_18():
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
    var_7 = module_0.b(*var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 1
    var_8 = module_0.b()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_8) == 0
    var_9 = var_1.__contains__(var_0)
    assert var_9 is False
    var_10 = var_7.__eq__(var_5)
    assert var_10 is True
    var_11 = var_2.__sub__(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_11) == 0
    var_12 = var_8.update(var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_12) == 0
    var_13 = var_12.__sub__(var_0)
    var_2.__contains__(var_0)

def test_case_19():
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
    var_5 = var_2.update(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_1.__add__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = var_5.__add__(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 2
    var_8 = var_5.__or__(var_4)
    var_9 = var_5.__len__()
    assert var_9 == 1
    var_10 = module_0.b()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 0
    var_11 = module_0.b(*var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_11) == 2
    var_12 = var_2.__contains__(var_7)
    assert var_12 is False
    with pytest.raises(TypeError):
        var_7.__eq__(var_0)

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
    var_3 = None
    var_4 = var_2.__repr__()
    assert var_4 == 'pbag([pbag([])])'
    var_5 = var_2.update(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 1
    var_6 = var_1.__add__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = var_1.__or__(var_3)
    var_8 = var_1.__len__()
    assert var_8 == 0
    var_9 = module_0.b()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = module_0.b()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 0
    var_11 = var_1.__contains__(var_7)
    assert var_11 is False
    var_12 = module_0.pbag(var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_12) == 0
    var_13 = var_9.__and__(var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_13) == 0
    var_14 = var_5.update(var_2)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_14) == 2
    var_15 = var_9.add(var_10)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_15) == 1
    var_16 = var_5.update(var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_16) == 3
    var_17 = var_13.__and__(var_3)
    var_18 = module_0.pbag(var_4)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_18) == 16
    var_18.__or__(var_5)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == 'pbag([pbag([])])'
    var_4 = var_2.update(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_4) == 1
    var_5 = var_1.__add__(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0
    var_6 = var_1.__or__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 0
    var_7 = module_0.pbag(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 1
    var_8 = var_1.__len__()
    assert var_8 == 0
    var_9 = module_0.b()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_9) == 0
    var_10 = module_0.b()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_10) == 0
    var_11 = var_1.__contains__(var_6)
    assert var_11 is False
    var_12 = module_0.pbag(var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_12) == 0
    var_13 = var_3.__len__()
    assert var_13 == 16
    var_14 = module_0.b()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_14) == 0
    var_15 = var_2.__and__(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_15) == 0
    var_16 = var_14.__contains__(var_13)
    assert var_16 is False
    var_11.add(var_9)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = module_0.pbag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.add(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_2) == 1
    var_3 = None
    var_4 = var_1.__repr__()
    assert var_4 == 'pbag([])'
    var_5 = var_2.remove(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_5) == 0
    var_6 = var_5.add(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_6) == 1
    var_7 = var_6.update(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_7) == 1
    var_8 = var_5.__add__(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pbag.PBag'
    assert len(var_8) == 0
    var_9 = var_5.__iter__()
    var_9.__or__(var_3)