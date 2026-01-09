# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._pdeque as module_0
import pytest


def test_case_0():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'

def test_case_1():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'

def test_case_2():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = None
    var_2 = var_0.__eq__(var_1)

def test_case_3():
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

def test_case_4():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.append(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 1

def test_case_5():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True

def test_case_6():
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

def test_case_7():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.extendleft(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0

def test_case_8():
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

def test_case_9():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = None
    var_2 = var_0.__lt__(var_1)

def test_case_10():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_0.remove(var_0)

def test_case_11():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.count(var_0)
    assert var_1 == 0

def test_case_12():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.reverse()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0

def test_case_13():
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
    var_2 = var_1.popleft()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0

def test_case_14():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__hash__()
    assert var_1 == 5740354900026072187

def test_case_15():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.extend(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0

def test_case_16():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.append(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 1
    var_2 = var_1.extend(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 2

def test_case_17():
    var_0 = True
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

def test_case_18():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__reduce__()

@pytest.mark.xfail(strict=True)
def test_case_19():
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
    module_0.PDeque()

def test_case_20():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__repr__()
    assert var_2 == 'pdeque([], maxlen=True)'

def test_case_21():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.popleft()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0

def test_case_22():
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
    var_3 = var_2.popleft()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0

def test_case_23():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.count(var_1)
    assert var_2 == 0
    var_3 = var_1.append(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_3.append(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 1
    with pytest.raises(TypeError):
        var_3.__getitem__(var_3)

def test_case_24():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__lt__(var_0)
    assert var_1 is False

def test_case_25():
    var_0 = False
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__lt__(var_1)
    assert var_2 is False
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    var_4 = var_3.__reduce__()
    var_5 = var_3.pop()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    var_6 = var_1.extendleft(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 0
    var_7 = var_5.append(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 0
    with pytest.raises(TypeError):
        var_3.__getitem__(var_6)

def test_case_26():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    with pytest.raises(TypeError):
        var_0.__getitem__(var_0)

def test_case_27():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = None
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_1.__eq__(var_3)
    assert var_4 is False
    with pytest.raises(IndexError):
        var_1.__getitem__(var_0)

def test_case_28():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__repr__()
    assert var_1 == 'pdeque([])'

def test_case_29():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_0.__invert__()
    assert var_2 == -2
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_4 = var_1.append(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 1
    var_5 = var_4.rotate(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 1

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.dq(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 4
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_3 = var_2.pop()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 3
    var_4 = None
    module_0.pdeque(var_4)

def test_case_31():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    with pytest.raises(IndexError):
        var_1.__getitem__(var_0)

def test_case_32():
    var_0 = False
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    with pytest.raises(IndexError):
        var_1.__getitem__(var_0)

def test_case_33():
    var_0 = False
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__lt__(var_1)
    assert var_2 is False
    var_3 = var_2.__invert__()
    assert var_3 == -1
    with pytest.raises(IndexError):
        var_1.__getitem__(var_3)

def test_case_34():
    var_0 = False
    var_1 = None
    var_2 = [var_1]
    var_3 = module_0.dq(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_4 = var_3.__getitem__(var_0)
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = module_0.dq()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 0
    with pytest.raises(ValueError):
        var_3.remove(var_5)

def test_case_35():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = var_1.append(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_3.rotate(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 1

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = False
    var_1 = None
    var_2 = [var_1, var_1]
    var_3 = module_0.dq(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 2
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_4 = var_3.__getitem__(var_0)
    var_5 = module_0.pdeque()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    var_4.pop()

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0.dq(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 2
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_3 = var_2.pop()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_3.__lt__(var_3)
    assert var_4 is False
    var_5 = var_4.__invert__()
    assert var_5 == -1
    var_6 = None
    var_7 = var_3.append(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 2
    var_8 = var_7.pop()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_8) == 1
    var_9 = var_3.__getitem__(var_5)
    var_4.popleft()

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = (2738.66943+3220j)
    var_1 = {var_0, var_0, var_0}
    var_2 = [var_1, var_1, var_1, var_1]
    module_0.PDeque(*var_2)