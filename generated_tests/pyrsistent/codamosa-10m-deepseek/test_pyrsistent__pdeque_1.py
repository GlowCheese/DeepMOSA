# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._pdeque as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__hash__()
    assert var_2 == 5740354900026072187
    var_1.__rmod__(var_1)

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
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.PDeque(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2.extend(var_0)

def test_case_3():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_0.remove(var_0)

def test_case_4():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__reduce__()

def test_case_5():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__lt__(var_0)
    assert var_1 is False
    var_2 = var_0.append(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 1
    var_3 = var_2.__getitem__(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    var_4 = var_2.__getitem__(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 0
    var_5 = var_1.__rxor__(var_3)

def test_case_6():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.reverse()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    var_2 = var_0.pop()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_0.extendleft(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0

def test_case_7():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.reverse()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    var_2 = var_0.pop()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    with pytest.raises(TypeError):
        var_0.__getitem__(var_0)

def test_case_8():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__iter__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'itertools.chain'
    var_2 = var_0.__lt__(var_0)
    assert var_2 is False
    var_3 = var_0.append(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.count(var_0)
    assert var_5 == 1
    var_6 = var_3.__getitem__(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 0
    var_7 = var_3.pop()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 0
    var_8 = var_2.__and__(var_3)
    var_9 = var_6.rotate(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_9) == 0
    var_10 = var_0.extend(var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_10) == 0
    var_11 = module_0.dq()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_11) == 0
    var_12 = var_3.remove(var_7)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_12) == 0
    with pytest.raises(ValueError):
        var_7.remove(var_1)

def test_case_9():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_0.reverse()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_0.pop()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    var_4 = var_0.__hash__()
    assert var_4 == 5740354900026072187
    var_5 = var_0.extendleft(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    with pytest.raises(TypeError):
        var_0.__getitem__(var_0)

def test_case_10():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__lt__(var_0)
    assert var_1 is False
    var_2 = module_0.dq(*var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_0.pop()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    var_4 = var_0.extendleft(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 0

def test_case_11():
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
    var_2 = module_0.dq(*var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_2.extendleft(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    with pytest.raises(TypeError):
        var_0.__getitem__(var_0)

def test_case_12():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__iter__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'itertools.chain'
    var_2 = var_0.__lt__(var_0)
    assert var_2 is False
    var_3 = var_0.append(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__getitem__(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    var_6 = var_3.pop()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 0
    var_7 = var_2.__and__(var_3)
    var_8 = var_0.extend(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_8) == 0
    var_9 = var_4.__lt__(var_0)
    with pytest.raises(IndexError):
        var_0.__getitem__(var_4)

def test_case_13():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_0.pop()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    with pytest.raises(TypeError):
        var_0.__getitem__(var_0)

def test_case_14():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = None
    var_2 = var_0.__lt__(var_1)
    var_3 = module_0.dq(*var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    var_4 = var_0.pop()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 0
    var_5 = var_0.extendleft(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    with pytest.raises(TypeError):
        var_3.__getitem__(var_0)

def test_case_15():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__iter__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'itertools.chain'
    var_2 = var_0.__repr__()
    assert var_2 == 'pdeque([])'
    var_3 = var_0.__lt__(var_0)
    assert var_3 is False
    var_4 = var_0.append(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 1
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = var_4.__getitem__(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 0
    var_7 = module_0.dq()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 0
    var_8 = var_4.pop()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_8) == 0
    var_9 = var_6.rotate(var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_9) == 0
    with pytest.raises(ValueError):
        var_4.remove(var_5)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__lt__(var_0)
    assert var_1 is False
    var_2 = var_0.append(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__getitem__(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 0
    var_5 = var_2.pop()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    var_1.extend(var_1)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__lt__(var_0)
    assert var_1 is False
    var_2 = var_0.append(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 1
    var_3 = var_2.__getitem__(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 0
    var_4 = var_3.extend(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 1
    var_5 = var_0.__lt__(var_3)
    assert var_5 is False
    var_6 = var_2.__getitem__(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 0
    var_7 = var_1.__rxor__(var_5)
    assert var_7 is False
    var_5.remove(var_0)

def test_case_18():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__lt__(var_0)
    assert var_1 is False
    var_2 = var_0.append(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__getitem__(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 0
    var_5 = var_2.pop()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    var_6 = var_1.__and__(var_2)
    var_7 = var_4.rotate(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 0
    var_8 = var_0.extend(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_8) == 0
    var_9 = module_0.dq()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_9) == 0
    var_10 = var_2.remove(var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_10) == 0
    with pytest.raises(ValueError):
        var_5.remove(var_10)

def test_case_19():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.reverse()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    var_2 = var_0.__lt__(var_1)
    assert var_2 is False
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    with pytest.raises(IndexError):
        var_3.__getitem__(var_4)

def test_case_20():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__iter__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'itertools.chain'
    var_2 = var_0.__hash__()
    assert var_2 == 5740354900026072187
    var_3 = var_0.append(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_3.count(var_0)
    assert var_4 == 1
    with pytest.raises(IndexError):
        var_3.__getitem__(var_2)

def test_case_21():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 5
    var_4 = [var_0, var_1, var_2, var_1, var_3]
    var_5 = module_0.pdeque(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_6 = [var_1, var_2, var_3]
    var_7 = module_0.pdeque(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 3
    var_8 = [var_0, var_1, var_2]
    var_9 = module_0.pdeque(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_9) == 3
    var_10 = [var_2, var_2, var_3]
    var_11 = module_0.pdeque(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_11) == 3
    var_12 = [var_0, var_2, var_3]
    var_13 = module_0.pdeque(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_13) == 3
    var_14 = [var_1, var_2, var_2]
    var_15 = module_0.pdeque(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_15) == 3
    var_16 = [var_0, var_1, var_2]
    var_17 = module_0.pdeque(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_17) == 3
    var_18 = [var_1, var_13]
    var_19 = module_0.pdeque(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_19) == 2
    var_20 = [var_0, var_0]
    var_21 = module_0.pdeque(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_21) == 2
    var_22 = [var_2, var_3]
    var_23 = module_0.pdeque(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_23) == 2
    var_24 = 10
    with pytest.raises(IndexError):
        var_25 = var_5[var_24]

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__iter__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'itertools.chain'
    var_2 = var_0.__lt__(var_0)
    assert var_2 is False
    var_3 = var_0.append(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__getitem__(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    var_6 = var_5.__hash__()
    assert var_6 == 5740354900026072187
    var_7 = var_3.pop()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 0
    var_8 = module_0.pdeque(var_1, var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_8) == 0
    var_9 = var_0.extend(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_9) == 0
    var_10 = module_0.dq()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_10) == 0
    var_11 = var_8.extend(var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_11) == 0
    var_12 = module_0.dq()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_12) == 0
    var_1.__or__(var_8)

def test_case_23():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__reduce__()
    with pytest.raises(ValueError):
        var_1.remove(var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = module_0.pdeque()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 3
    var_7 = [var_2, var_3, var_4]
    var_8 = module_0.pdeque(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_8) == 3
    var_9 = [var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_10) == 3
    var_11 = 4
    var_12 = -5
    var_13 = 6
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.pdeque(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_15) == 3
    var_16 = [var_2, var_3, var_4]
    var_17 = module_0.pdeque(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_17) == 3
    var_18 = [var_2, var_3, var_4]
    module_0.pdeque(var_18, var_12)

def test_case_25():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = 5
    var_4 = [var_1, var_0, var_1, var_2, var_3]
    var_5 = module_0.pdeque(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 5
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_6 = [var_1, var_0, var_1, var_2, var_3]
    var_7 = module_0.pdeque(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 5
    var_8 = [var_5, var_0, var_1, var_2, var_3]
    var_9 = module_0.pdeque(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_9) == 5
    var_10 = [var_0, var_1, var_2]
    var_11 = module_0.pdeque(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_11) == 3
    var_12 = [var_1, var_0, var_1]
    var_13 = module_0.pdeque(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_13) == 3
    var_14 = [var_1, var_2, var_3]
    var_15 = module_0.pdeque(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_15) == 3
    var_16 = [var_15, var_1, var_3]
    var_17 = module_0.pdeque(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_17) == 3
    var_18 = [var_15, var_0, var_1, var_2, var_3]
    var_19 = module_0.pdeque(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_19) == 5
    var_20 = [var_3, var_2, var_1, var_0, var_11]
    var_21 = module_0.pdeque(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_21) == 5
    var_22 = [var_3, var_2, var_1]
    var_23 = module_0.pdeque(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_23) == 3
    var_24 = [var_1, var_0, var_1]
    var_25 = module_0.pdeque(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_25) == 3
    var_26 = [var_0, var_0, var_1, var_2, var_3]
    var_27 = module_0.pdeque(var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_27) == 5
    var_28 = [var_0, var_2]
    var_29 = module_0.pdeque(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_29) == 2
    var_30 = [var_5, var_2]
    var_31 = module_0.pdeque(var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_31) == 2
    var_32 = [var_1, var_3]
    var_33 = module_0.pdeque(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_33) == 2
    var_34 = [var_0, var_0, var_1, var_2, var_3]
    var_35 = module_0.pdeque(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_35) == 5
    var_36 = [var_1, var_2, var_3]
    var_37 = module_0.pdeque(var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_37) == 3
    var_38 = [var_17, var_0, var_1]
    var_39 = module_0.pdeque(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_39) == 3
    var_40 = []
    var_41 = module_0.pdeque(var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_41) == 0
    var_42 = [var_29, var_0, var_1, var_2, var_3]
    var_43 = module_0.pdeque(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_43) == 5
    var_44 = [var_0, var_1, var_2]
    var_45 = module_0.pdeque(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_45) == 3
    var_46 = [var_2, var_3]
    var_47 = module_0.pdeque(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_47) == 2
    var_48 = [var_15, var_0]
    var_49 = module_0.pdeque(var_48)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_49) == 2
    var_50 = [var_25, var_0, var_1, var_2, var_3]
    var_51 = module_0.pdeque(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_51) == 5
    var_52 = [var_3, var_1]
    var_53 = module_0.pdeque(var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_53) == 2
    var_54 = [var_3, var_1, var_21]
    var_55 = module_0.pdeque(var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_55) == 3
    var_56 = [var_2, var_0]
    var_57 = module_0.pdeque(var_56)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_57) == 2
    var_58 = [var_45, var_0, var_1, var_2, var_3]
    var_59 = module_0.pdeque(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_59) == 5
    var_60 = 0
    with pytest.raises(ValueError):
        var_61 = var_59[::var_60]