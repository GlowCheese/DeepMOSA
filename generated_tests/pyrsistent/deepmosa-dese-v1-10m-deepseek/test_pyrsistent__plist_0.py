# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._plist as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0._PListBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = -690
    var_2 = var_1.__float__()
    assert var_2 == pytest.approx(-690.0, abs=0.01, rel=0.01)
    var_3 = var_0.append_plist(var_2)
    assert var_3 == pytest.approx(-690.0, abs=0.01, rel=0.01)
    var_4 = var_2.__repr__()
    assert var_4 == '-690.0'
    var_4.mcons(var_3)

def test_case_1():
    var_0 = module_0.l()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = module_0.plist(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    var_2 = var_0.__hash__()
    assert var_2 == 5740354900026072187
    var_3 = var_0.__len__()
    assert var_3 == 0

def test_case_2():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__len__()
    assert var_3 == 3
    var_4 = var_3.__lt__(var_0)
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True

def test_case_3():
    var_0 = module_0.l()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'

def test_case_4():
    var_0 = module_0._PListBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 4
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__bool__()
    assert var_3 is True
    var_4 = var_2.split(var_3)
    var_5 = var_2.remove(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 3
    var_3.build()

def test_case_6():
    var_0 = module_0._PListBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.build()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 1
    var_1 = 2
    var_2 = -9
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_6) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_7 = var_6[var_1::var_0]
    var_8 = var_6.__reduce__()
    var_8.split(var_6)

def test_case_8():
    var_0 = module_0.l()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_0.remove(var_0)

def test_case_9():
    var_0 = 4
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2[var_0:]

def test_case_10():
    var_0 = 4
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2[var_0:]
    var_4 = module_0.plist(var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 5

def test_case_11():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__len__()
    assert var_3 == 3
    var_4 = var_2.split(var_3)
    var_5 = module_0._PListBuilder()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    var_6 = module_0.l()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_6) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_7 = var_2.__eq__(var_2)
    assert var_7 is True
    var_8 = var_2.__lt__(var_3)
    var_9 = var_1.__getitem__(var_7)

def test_case_12():
    var_0 = module_0.l()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    with pytest.raises(TypeError):
        var_0.__getitem__(var_0)

def test_case_13():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = -2907
    with pytest.raises(IndexError):
        var_4 = var_2[var_3]

def test_case_14():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__bool__()
    assert var_3 is True
    var_4 = var_2.split(var_3)
    var_5 = var_2.remove(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 2
    var_6 = module_0._PListBuilder()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    var_7 = var_2.__eq__(var_4)
    var_8 = var_3.__and__(var_0)
    var_9 = var_6.build()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_9) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_10 = module_0.l()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_10) == 0
    with pytest.raises(TypeError):
        var_9.__getitem__(var_6)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_0._PListBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.l()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = var_1.reverse()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_2) == 0
    var_2.__int__()

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.reverse()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 3
    var_4 = var_3.__lt__(var_0)
    var_5 = var_0.__bool__()
    assert var_5 is False
    var_6 = var_2.__bool__()
    assert var_6 is True
    var_7 = var_2.cons(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_7) == 4
    var_8 = var_2.__hash__()
    assert var_8 == 8464103606981680943
    module_0.PList()

def test_case_17():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__bool__()
    assert var_3 is True
    var_4 = var_2.remove(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 2
    var_5 = var_3.__bool__()
    assert var_5 is True
    var_6 = var_3.__and__(var_0)
    var_7 = module_0.l()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_7) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    with pytest.raises(IndexError):
        var_7.__getitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0._PListBase()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._PListBase'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_2 = [var_0, var_0, var_0]
    var_3 = module_0.l(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 3
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = var_1.mcons(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    var_4.__ror__(var_0)

def test_case_19():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = var_1.__len__()
    assert var_2 == 3
    var_3 = var_2.__ror__(var_0)
    var_4 = module_0._PListBuilder()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_5 = module_0.l()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_5) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = var_6.__lt__(var_2)
    assert var_7 is True
    var_8 = var_1.__getitem__(var_6)

def test_case_20():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__bool__()
    assert var_3 is True
    var_4 = var_2.remove(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 2
    var_5 = module_0._PListBuilder()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    var_6 = var_2.__eq__(var_2)
    assert var_6 is True
    var_7 = var_3.__and__(var_0)
    var_8 = var_5.build()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_8) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_9 = var_8.__bool__()
    assert var_9 is False
    var_10 = var_2.cons(var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_10) == 4
    var_11 = var_10.__getitem__(var_3)
    var_12 = var_10.__lt__(var_10)
    assert var_12 is False

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_0._PListBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.plist(reverse=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = None
    var_3 = var_1.split(var_2)
    var_4 = None
    module_0.l(*var_4)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 16
    var_1 = 29
    var_2 = 5
    var_3 = [var_0, var_1, var_0, var_2]
    var_4 = module_0.plist(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 4
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_4[var_2:]
    var_6 = var_4.__eq__(var_5)
    assert var_6 is False
    module_0.plist(var_6)

def test_case_23():
    var_0 = 3
    var_1 = -7
    var_2 = 5
    var_3 = [var_2, var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 4
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_4[var_0:]
    var_6 = var_4.__eq__(var_5)
    assert var_6 is False

def test_case_24():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.cons(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 4
    with pytest.raises(ValueError):
        var_2.remove(var_2)

def test_case_25():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = var_0.__bool__()
    assert var_2 is False
    var_3 = var_1.remove(var_0)
    var_4 = var_2.__floordiv__(var_1)
    var_5 = var_2.__and__(var_0)
    var_6 = module_0.l()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_6) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    with pytest.raises(IndexError):
        var_6.__getitem__(var_2)

def test_case_26():
    var_0 = 2
    var_1 = 3
    var_2 = -7
    var_3 = 5
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 4
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = var_5[var_0:]
    var_7 = var_5.__eq__(var_6)
    assert var_7 is False
    var_8 = module_0.plist(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_8) == 4

def test_case_27():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__lt__(var_2)
    assert var_3 is False
    var_4 = var_2.split(var_3)
    var_5 = module_0._PListBuilder()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    var_6 = module_0._PListBuilder()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    var_7 = var_2.__eq__(var_2)
    assert var_7 is True
    var_8 = var_7.__floordiv__(var_0)
    var_9 = var_3.__and__(var_0)
    var_10 = var_2.__lt__(var_3)
    var_11 = var_2.cons(var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_11) == 4
    var_12 = var_7.__or__(var_9)
    var_13 = module_0.l()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_13) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_14 = var_7.__xor__(var_9)

def test_case_28():
    var_0 = -16
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    with pytest.raises(IndexError):
        var_3 = var_2[var_0:]

def test_case_29():
    var_0 = 1
    var_1 = 2
    var_2 = 4
    var_3 = 5
    var_4 = [var_0, var_1, var_2, var_2, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = var_5[::var_1]
    var_7 = [var_0, var_6, var_3]
    var_8 = module_0.plist(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_8) == 3

def test_case_30():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = 5
    var_4 = [var_1, var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = var_5[var_3:var_2]
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.plist(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_8) == 3

def test_case_31():
    var_0 = 1
    var_1 = 2
    var_2 = -1
    var_3 = [var_0, var_1, var_2, var_0, var_1]
    var_4 = module_0.plist(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_4[var_1::var_0]

def test_case_32():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = 5
    var_4 = [var_1, var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = var_5[var_0::var_0]
    var_7 = [var_0, var_2]
    var_8 = module_0.plist(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_8) == 2