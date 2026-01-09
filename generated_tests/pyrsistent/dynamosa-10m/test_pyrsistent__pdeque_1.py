# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._pdeque as module_0
import pyrsistent._plist as module_1
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = False
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    module_0.PDeque(*var_1)

def test_case_1():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    module_0.pdeque(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.PDeque(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2.extend(var_0)

def test_case_4():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.appendleft(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 1
    var_2 = var_0.extendleft(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 1

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = False
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.extend(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    module_0.PDeque(*var_1)

def test_case_6():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_0.remove(var_0)

def test_case_7():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__hash__()
    assert var_1 == 5740354900026072187
    var_2 = var_0.reverse()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_2.__hash__()
    assert var_3 == 5740354900026072187
    var_4 = var_1.__floor__()
    assert var_4 == 5740354900026072187

def test_case_8():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__reduce__()

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = module_0.dq()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__lt__(var_0)
    var_2.count(var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0.dq(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 2
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_3 = var_2.count(var_0)
    assert var_3 == 2
    var_2.__rmod__(var_2)

def test_case_11():
    var_0 = False
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.extendleft(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_2.count(var_2)
    assert var_3 == 0
    var_4 = var_1.pop()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 0
    var_5 = var_1.rotate(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    var_6 = var_0.__xor__(var_4)
    var_7 = var_1.__eq__(var_2)
    assert var_7 is True
    var_8 = var_0.__hash__()
    assert var_8 == 0
    var_9 = module_1.plist(reverse=var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._plist._EmptyPList'
    assert len(var_9) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1._EmptyPList.first).__module__}.{type(module_1._EmptyPList.first).__qualname__}' == 'builtins.property'
    assert f'{type(module_1._EmptyPList.rest).__module__}.{type(module_1._EmptyPList.rest).__qualname__}' == 'builtins.property'
    var_10 = var_4.__repr__()
    assert var_10 == 'pdeque([], maxlen=False)'

def test_case_12():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 1
    var_3 = [var_2, var_2]
    var_4 = var_1.__eq__(var_3)
    var_5 = None
    with pytest.raises(TypeError):
        var_3.__new__(var_4, var_5, var_5, var_4, var_1)

def test_case_13():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.append(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 1

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = False
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_2.__float__()

def test_case_15():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__lt__(var_0)
    assert var_1 is False

def test_case_16():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.pop()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0

def test_case_17():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    with pytest.raises(TypeError):
        var_0.__getitem__(var_0)

def test_case_18():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__hash__()
    assert var_1 == 5740354900026072187
    var_2 = var_0.__hash__()
    assert var_2 == 5740354900026072187
    var_3 = var_1.__floor__()
    assert var_3 == 5740354900026072187

def test_case_19():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__hash__()
    assert var_1 == 5740354900026072187
    var_2 = var_0.rotate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_0.reverse()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    var_4 = var_3.__hash__()
    assert var_4 == 5740354900026072187

def test_case_20():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.appendleft(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 1
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'pdeque([])'
    var_5 = {var_4: var_4}
    module_0.PDeque(**var_5)

def test_case_22():
    var_0 = None
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.popleft()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0

def test_case_23():
    var_0 = None
    var_1 = module_0.dq()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__eq__(var_0)

def test_case_24():
    var_0 = None
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.appendleft(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 1
    var_3 = var_2.popleft()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    var_4 = var_1.__repr__()
    assert var_4 == 'pdeque([])'

def test_case_25():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.appendleft(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 1
    var_2 = var_0.extendleft(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_2.popleft()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    var_4 = var_3.reverse()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 0
    var_5 = var_1.pop()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    with pytest.raises(TypeError):
        var_0.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.appendleft(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 1
    var_2 = var_1.extendleft(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 2
    var_3 = var_2.popleft()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_0.__repr__()
    assert var_4 == 'pdeque([])'
    var_5 = var_2.reverse()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 2
    var_5.__truediv__(var_5)

def test_case_27():
    var_0 = True
    var_1 = None
    var_2 = module_0.pdeque(maxlen=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_3 = var_2.appendleft(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False
    with pytest.raises(IndexError):
        var_3.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = False
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = None
    var_3 = module_0.pdeque(maxlen=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_4 = var_3.__eq__(var_2)
    var_5 = var_3.appendleft(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 1
    var_6 = var_5.__getitem__(var_0)
    var_7 = var_3.__repr__()
    assert var_7 == 'pdeque([])'
    var_1.reverse()

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = False
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.extendleft(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_2.popleft()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    var_4 = [var_0, var_0, var_1, var_1]
    var_5 = module_0.dq(*var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 4
    var_6 = var_1.__eq__(var_1)
    assert var_6 is True
    var_7 = module_0.dq()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 0
    var_8 = var_7.rotate(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_8) == 0
    module_0.PDeque(*var_5)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = '\r-\n2Kx%+K\rvu)'
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_3 = module_0.PDeque(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_3.remove(var_0)

def test_case_31():
    var_0 = False
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = module_0.pdeque()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_2.__repr__()
    assert var_3 == 'pdeque([])'
    var_4 = var_2.appendleft(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 1
    var_5 = var_1.extendleft(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    var_6 = var_1.__eq__(var_4)
    assert var_6 is False
    with pytest.raises(TypeError):
        var_1.__getitem__(var_1)

def test_case_32():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = module_0.pdeque()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_2.__repr__()
    assert var_3 == 'pdeque([])'
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = var_1.extendleft(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    var_6 = [var_0, var_2, var_1, var_1]
    var_7 = var_1.extendleft(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 1
    var_8 = var_1.__eq__(var_5)
    assert var_8 is True
    var_9 = var_7.append(var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_9) == 1
    var_10 = var_9.rotate(var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_10) == 1
    with pytest.raises(TypeError):
        var_1.__getitem__(var_1)