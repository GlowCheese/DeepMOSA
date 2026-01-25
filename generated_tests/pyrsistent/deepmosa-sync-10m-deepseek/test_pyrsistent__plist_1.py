# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._plist as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0._PListBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.build()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = module_0._PListBuilder()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    var_3 = None
    var_4 = var_2.append_plist(var_3)
    var_5 = module_0.plist()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_5) == 0
    var_6 = module_0.l()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_6) == 0
    var_5.__abs__()

def test_case_1():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    with pytest.raises(TypeError):
        var_0.__getitem__(var_0)

def test_case_2():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'

def test_case_3():
    var_0 = module_0._EmptyPList()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'

def test_case_4():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.split(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0._PListBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.build()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_2) == 0
    var_2.__abs__()

def test_case_6():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = None
    var_3 = var_0.__reduce__()
    with pytest.raises(TypeError):
        var_0.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.__repr__()
    assert var_1 == 'plist([])'
    var_1.reverse()

def test_case_8():
    var_0 = module_0.l()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'

def test_case_9():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 5
    var_4 = [var_0, var_1, var_2, var_1, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = None
    with pytest.raises(ValueError):
        var_5.remove(var_6)

def test_case_10():
    var_0 = module_0.l()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.__lt__(var_0)
    assert var_1 is False
    var_2 = None
    var_3 = module_0.l()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_3) == 0
    var_4 = False
    var_5 = var_4.__lt__(var_2)
    var_6 = var_3.__eq__(var_3)
    assert var_6 is True
    with pytest.raises(ValueError):
        var_3.remove(var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_2.split(var_2)

def test_case_12():
    var_0 = 4
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = bool(var_2[1:3] == var_1)
    var_4 = bool(var_2[:3] == var_3)
    var_5 = var_2[3:]
    var_6 = var_2[::2]

def test_case_13():
    var_0 = module_0._PListBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.build()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = module_0._PListBuilder()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    with pytest.raises(ValueError):
        var_1.remove(var_1)

def test_case_14():
    var_0 = 4
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__hash__()
    assert var_3 == 6412379544719930125
    var_4 = bool(var_2[1:3] == var_3)
    var_5 = var_3.__pos__()
    assert var_5 == 6412379544719930125
    var_6 = bool(var_2[:3] == var_5)
    var_7 = var_2[3:]
    var_8 = var_2[::2]

def test_case_15():
    var_0 = 4
    var_1 = 5
    var_2 = [var_0, var_1, var_0, var_0, var_1]
    var_3 = module_0.plist(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = var_3[0]
    var_5 = var_3[2]
    var_6 = var_3.__eq__(var_3)
    assert var_6 is True
    var_7 = var_3[4]
    assert var_7 == 5

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.__bool__()
    assert var_1 is False
    var_2 = var_0.__eq__(var_0)
    assert var_2 is True
    var_1.__divmod__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.reverse()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    var_0.__rdivmod__(var_0)

def test_case_18():
    var_0 = 9
    var_1 = 2
    var_2 = [var_0, var_1, var_1, var_1, var_1]
    var_3 = module_0.plist(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = var_3[0]
    var_5 = var_3[-1]

def test_case_19():
    var_0 = []
    var_1 = module_0.plist(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = var_1.__len__()
    assert var_2 == 0
    var_3 = var_1.mcons(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_3) == 0
    var_4 = 0
    with pytest.raises(IndexError):
        var_3.__getitem__(var_4)

def test_case_20():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.mcons(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 2
    var_4 = 0
    var_5 = var_3.__getitem__(var_4)
    assert var_5 == 1

def test_case_21():
    var_0 = 3
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 4
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = 10
    with pytest.raises(IndexError):
        var_4 = var_2[var_3]

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = module_0._PListBase()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._PListBase'
    var_2 = var_0.__eq__(var_0)
    assert var_2 is True
    var_3 = var_1.__lt__(var_2)
    var_4 = var_2.__float__()
    assert var_4 == pytest.approx(1.0, abs=0.01, rel=0.01)
    var_5 = None
    var_4.__getitem__(var_5)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3, var_1]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = [var_1, var_2]
    var_7 = module_0.plist(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_7) == 2
    var_8 = var_5[1:3]
    var_9 = bool(var_5[1:3] == var_0)
    var_10 = var_5[1:]
    var_11 = bool(var_5[1:] == var_7)
    var_12 = [var_0, var_1, var_2]
    var_13 = module_0.plist(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_13) == 3
    var_14 = var_5[:3]
    var_15 = bool(var_5[:3] == var_13)
    assert var_15 is True
    module_0.plist(var_0)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0.l(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = module_0.plist()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_3) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_4 = var_3.reverse()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_4) == 0
    var_5 = var_2.__eq__(var_4)
    assert var_5 is False
    var_6 = var_5.__invert__()
    assert var_6 == -1
    var_7 = var_6.__rdivmod__(var_0)
    var_4.__divmod__(var_4)

def test_case_25():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = module_0.plist(reverse=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_5) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_6 = module_0.plist(reverse=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_6) == 0

def test_case_26():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 5
    var_4 = [var_0, var_1, var_2, var_1, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = None
    var_7 = var_5[var_0:var_6:var_0]
    var_8 = [var_0, var_3]
    var_9 = module_0.plist(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_9) == 2
    var_10 = bool(var_7 == var_9)

def test_case_27():
    var_0 = 1
    var_1 = 2
    var_2 = 4
    var_3 = -832
    var_4 = [var_0, var_1, var_2, var_2, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = module_0.l()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_6) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_7 = var_5.__bool__()
    assert var_7 is True
    var_8 = [var_1, var_3]
    var_9 = module_0.plist(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_9) == 2
    var_10 = var_7.__bool__()
    assert var_10 is True
    var_11 = [var_1, var_0, var_2, var_3]
    var_12 = module_0.plist(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_12) == 4
    var_13 = var_12[1:]
    var_14 = var_5.reverse()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_14) == 5
    var_15 = bool(var_5[1:] == var_13)
    assert var_15 is True
    var_16 = var_5[:3]
    var_17 = bool(var_5[:3] == var_3)
    assert var_17 is True
    var_18 = [var_3, var_2, var_8, var_1, var_0]
    var_19 = module_0.plist(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_19) == 5
    var_20 = var_12.__getitem__(var_15)
    assert var_20 == 2
    var_21 = var_15.__rshift__(var_17)
    assert var_21 == 0
    var_22 = var_5[::-1]
    with pytest.raises(TypeError):
        var_23 = bool(var_15[::-1] == var_19)
    assert var_23 is True

def test_case_28():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_6) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_7 = var_6[var_0::var_1]
    var_8 = [var_1, var_3]
    var_9 = module_0.plist(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_9) == 2
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

def test_case_29():
    var_0 = 4
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__len__()
    assert var_3 == 5
    var_4 = bool(var_2[1:] == var_3)
    var_5 = [var_3, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_6) == 4
    var_7 = var_5.__len__()
    assert var_7 == 4
    var_8 = var_2[:3]
    var_9 = var_2.split(var_3)
    var_10 = module_0._PListBuilder()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    var_11 = bool(var_2[:3] == var_6)
    var_12 = [var_11, var_0, var_0, var_6, var_3]
    var_13 = module_0.plist(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_13) == 5
    var_14 = var_2[::-1]
    var_15 = bool(var_2[::-1] == var_13)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 2
    var_1 = 4
    var_2 = [var_1, var_0, var_1, var_1, var_0]
    var_3 = module_0.plist(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = var_0.__bool__()
    assert var_4 is True
    var_5 = [var_0, var_3, var_0, var_0]
    var_6 = var_5.__len__()
    assert var_6 == 4
    var_7 = var_3[:3]
    var_8 = var_3.split(var_1)
    var_9 = module_0._PListBuilder()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    var_7.__ror__(var_9)

def test_case_31():
    var_0 = -724
    var_1 = 4
    var_2 = 5
    var_3 = [var_1, var_0, var_1, var_1, var_2]
    var_4 = module_0.plist(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = [var_0, var_1]
    var_6 = module_0.plist(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_6) == 2
    var_7 = bool(var_4[1:] == var_6)
    var_8 = module_0.l()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_8) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_9 = [var_2, var_4, var_0, var_2]
    var_10 = var_9.__len__()
    assert var_10 == 4
    var_11 = var_4[:3]
    var_12 = var_4.split(var_7)
    var_13 = module_0._PListBuilder()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._plist._PListBuilder'
    var_14 = bool(var_4[:3] == var_3)
    var_15 = [var_2, var_1, var_1, var_0, var_7]
    var_16 = module_0.plist(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_16) == 5
    var_17 = var_4[::-1]
    var_18 = bool(var_4[::-1] == var_16)

def test_case_32():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.plist(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = var_3[var_1]
    var_5 = bool(False)
    var_6 = -4
    with pytest.raises(IndexError):
        var_7 = var_3[var_6]

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_0._PListBase()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._PListBase'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    var_1 = 2
    var_2 = -1377
    var_3 = [var_1, var_1, var_2, var_2, var_1]
    var_4 = module_0.plist(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 5
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = [var_4]
    var_6 = module_0.plist(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_6) == 1
    var_7 = var_4[1:3]
    var_8 = var_7[1:]
    var_9 = var_6.remove(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_9) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_4.__float__()