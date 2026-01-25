# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._plist as module_0
import pytest


def test_case_0():
    var_0 = 1
    var_1 = module_0.plist()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = [var_0, var_0, var_0, var_0, var_0]
    var_3 = module_0.plist(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 5
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = var_3.__lt__(var_1)
    assert var_4 is False
    var_5 = var_3[var_0:]
    with pytest.raises(ValueError):
        var_5.remove(var_3)

def test_case_1():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = -1
    with pytest.raises(IndexError):
        var_2 = var_0[var_1]

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
    var_0 = module_0.l()
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

@pytest.mark.xfail(strict=True)
def test_case_6():
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

def test_case_7():
    var_0 = 1
    var_1 = module_0.plist()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = var_1.split(var_1)
    var_3 = module_0.plist(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 2
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = var_3.__reduce__()
    var_5 = var_3[var_0:]
    var_6 = var_1.__eq__(var_0)
    var_7 = module_0._EmptyPList()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_7) == 0
    with pytest.raises(TypeError):
        var_3.__getitem__(var_2)

def test_case_8():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0[:]

def test_case_9():
    var_0 = module_0.l()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'

def test_case_10():
    var_0 = module_0.l()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.split(var_0)
    var_2 = module_0.plist()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_2) == 0
    with pytest.raises(TypeError):
        var_3 = var_2[var_1]

def test_case_11():
    var_0 = 1
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__lt__(var_2)
    assert var_3 is False
    var_4 = var_2[var_0:]
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True

def test_case_12():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.cons(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_1) == 1
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_2 = var_1.reverse()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 1
    var_3 = module_0.l()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_3) == 0
    var_4 = var_0.__iter__()

def test_case_13():
    var_0 = 1
    var_1 = module_0.plist()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_2) == 0
    var_3 = var_2[var_0:]
    var_4 = var_1.__eq__(var_0)
    var_5 = module_0._EmptyPList()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_5) == 0
    var_6 = var_3.__eq__(var_1)
    assert var_6 is True

def test_case_14():
    var_0 = module_0.l()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.split(var_0)
    var_2 = var_0.reverse()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_2) == 0
    var_3 = module_0.plist()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_3) == 0
    with pytest.raises(ValueError):
        var_3.remove(var_2)

def test_case_15():
    var_0 = 2
    var_1 = -10
    var_2 = 1183
    var_3 = [var_1, var_0, var_1, var_2, var_2]
    var_4 = module_0.plist(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_4[var_0::var_0]
    var_6 = var_5.__hash__()
    assert var_6 == -3671716052555869749

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 4
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2[var_0:]
    var_2.__floordiv__(var_2)

def test_case_17():
    var_0 = module_0.l()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.split(var_0)
    var_2 = var_0.reverse()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_2) == 0
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = module_0.plist()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_4) == 0
    with pytest.raises(TypeError):
        var_5 = var_4[var_1]

def test_case_18():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.reverse()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    with pytest.raises(TypeError):
        var_2 = var_0[var_1:]

def test_case_19():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.__bool__()
    assert var_1 is False
    var_2 = var_0[var_1:]

def test_case_20():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.__repr__()
    assert var_1 == 'plist([])'
    var_2 = -1
    with pytest.raises(IndexError):
        var_3 = var_0[var_2]

def test_case_21():
    var_0 = 99
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    with pytest.raises(IndexError):
        var_3 = var_2[var_0]

def test_case_22():
    var_0 = 1
    var_1 = 2
    var_2 = 4
    var_3 = 5
    var_4 = [var_0, var_1, var_0, var_2, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = var_5[var_0::var_1]
    var_7 = var_5.__eq__(var_6)
    assert var_7 is False
    var_8 = var_5.mcons(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_8) == 10

def test_case_23():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = 5
    var_2 = module_0.plist(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_2) == 0
    var_3 = var_0.reverse()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_3) == 0
    var_4 = var_2[var_1:]

def test_case_24():
    var_0 = 1
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__bool__()
    assert var_3 is True
    var_4 = var_2.__lt__(var_1)
    var_5 = var_2[var_0:]
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    with pytest.raises(ValueError):
        var_5.remove(var_2)

def test_case_25():
    var_0 = 99
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = -1
    var_4 = var_2[var_3]

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_1 = var_0.split(var_0)
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 2
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    module_0.l(*var_3)

def test_case_27():
    var_0 = 1
    var_1 = 4
    var_2 = module_0.plist()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_2) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_3 = [var_0, var_1, var_1, var_1, var_1]
    var_4 = module_0.plist(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 5
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_4[var_1:]
    var_6 = var_4.__eq__(var_2)
    assert var_6 is False

def test_case_28():
    var_0 = 14
    var_1 = 3
    var_2 = [var_0, var_1, var_1, var_1, var_1]
    var_3 = module_0.plist(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = var_3[var_1:]
    var_5 = module_0.plist(reverse=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_5) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'

def test_case_29():
    var_0 = module_0.plist()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    with pytest.raises(TypeError):
        var_1 = var_0[var_0:]

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 1
    var_1 = 4
    var_2 = [var_0, var_0, var_0, var_1, var_0]
    var_3 = module_0.plist(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = var_3.reverse()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 5
    var_5 = var_3[var_1:]
    module_0.l(*var_1)

def test_case_31():
    var_0 = 1
    var_1 = 1443
    var_2 = 3
    var_3 = 5
    var_4 = [var_0, var_1, var_2, var_2, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = var_5[var_0::var_1]
    var_7 = var_5.__eq__(var_6)
    assert var_7 is False

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = 1
    var_1 = module_0.plist()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = [var_0, var_0, var_0, var_0, var_0]
    var_3 = var_1.split(var_1)
    var_4 = module_0.plist(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 5
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_4.split(var_0)
    var_6 = None
    var_7 = var_4.__lt__(var_6)
    var_5.remove(var_6)

def test_case_33():
    var_0 = 1
    var_1 = module_0.plist()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = [var_0, var_0, var_0, var_0, var_0]
    var_3 = var_1.split(var_1)
    var_4 = module_0.plist(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 5
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_4.__lt__(var_1)
    assert var_5 is False
    var_6 = var_4[var_0:]
    var_7 = var_4.split(var_5)
    with pytest.raises(ValueError):
        var_6.remove(var_4)

def test_case_34():
    var_0 = 1
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.plist(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_2) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__lt__(var_0)
    var_4 = var_2[var_0:]
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = var_2.remove(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_6) == 4

def test_case_35():
    var_0 = 1
    var_1 = -13
    var_2 = 3
    var_3 = 5
    var_4 = [var_0, var_1, var_2, var_0, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = var_5[var_0::var_1]
    var_7 = [var_1, var_0]
    var_8 = module_0.plist(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_8) == 2
    var_9 = var_8.__eq__(var_6)
    assert var_9 is False

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = 25
    var_1 = module_0.plist()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_2 = [var_0, var_0, var_0, var_0, var_0]
    var_3 = module_0.plist(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 5
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = var_3.__lt__(var_1)
    assert var_4 is False
    var_5 = var_3[var_4:]
    var_6 = var_3.split(var_4)
    var_7 = var_5.split(var_0)
    var_7.remove(var_7)

def test_case_37():
    var_0 = 1
    var_1 = 19
    var_2 = 5
    var_3 = [var_0, var_2, var_2, var_1, var_2]
    var_4 = module_0.plist(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_4) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_4[var_0:var_2]
    var_6 = [var_5, var_5]
    var_7 = module_0.plist(reverse=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_7) == 0
    assert f'{type(module_0._EmptyPList.first).__module__}.{type(module_0._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._EmptyPList.rest).__module__}.{type(module_0._EmptyPList.rest).__qualname__}' == 'builtins.property'

def test_case_38():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1, var_0, var_0, var_0]
    var_3 = module_0.plist(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_3) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_4 = -1
    var_5 = var_3[var_4::var_4]

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = 1
    var_1 = 2
    var_2 = -1375
    var_3 = 5
    var_4 = [var_0, var_1, var_0, var_2, var_3]
    var_5 = module_0.plist(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._plist.PList'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PList.first).__module__}.{type(module_0.PList.first).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.PList.rest).__module__}.{type(module_0.PList.rest).__qualname__}' == 'builtins.member_descriptor'
    var_6 = var_5[var_0::var_0]
    var_5.__float__()