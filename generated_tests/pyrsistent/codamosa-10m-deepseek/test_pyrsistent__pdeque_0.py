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
    var_2 = var_1.popleft()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0

def test_case_1():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'

def test_case_2():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 7
    var_3 = var_2.extendleft(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 13
    var_4 = var_1.count(var_2)
    assert var_4 == 0
    var_5 = var_4.__trunc__()
    assert var_5 == 0

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = True
    var_1 = module_0.pdeque(maxlen=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__le__(var_1)
    var_3 = var_1.append(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 1
    var_2.remove(var_1)

def test_case_4():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 7
    var_3 = var_1.extendleft(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 13
    with pytest.raises(TypeError):
        var_1.__getitem__(var_3)

def test_case_5():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.__repr__()
    assert var_1 == 'pdeque([])'

def test_case_6():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.appendleft(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 7
    var_3 = var_2.__hash__()
    assert var_3 == 1889528378764942238
    var_4 = var_1.extendleft(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 13
    var_5 = var_2.__repr__()
    assert var_5 == "pdeque([pdeque(['9', '.', '_', '=', 'v', 'd']), '9', '.', '_', '=', 'v', 'd'])"
    var_6 = var_1.extend(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 12
    var_7 = var_2.rotate(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 7
    var_8 = var_1.__eq__(var_1)
    assert var_8 is True
    var_9 = var_1.__getitem__(var_8)
    assert var_9 == '.'

def test_case_7():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.extendleft(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 12
    var_3 = var_1.extend(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 12

def test_case_8():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'

def test_case_9():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__hash__()
    assert var_2 == -1856064063567273487
    var_3 = var_1.extendleft(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 12
    var_4 = var_1.extend(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 12
    var_5 = var_2.__trunc__()
    assert var_5 == -1856064063567273487

def test_case_10():
    var_0 = None
    var_1 = module_0.dq()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    with pytest.raises(TypeError):
        var_1.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 's._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__hash__()
    assert var_2 == 207016169190830020
    var_3 = var_1.extend(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 12
    var_4 = var_3.rotate(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 12
    var_4.__trunc__()

def test_case_12():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = var_0.popleft()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 0

def test_case_13():
    var_0 = '9)._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 7
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 8
    var_3 = var_1.__lt__(var_1)
    assert var_3 is False
    var_4 = var_1.extendleft(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 15
    var_5 = var_2.__le__(var_2)
    var_6 = var_1.popleft()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 6
    var_7 = var_6.__lt__(var_5)
    var_8 = var_1.__getitem__(var_3)
    assert var_8 == '9'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 7
    var_3 = var_1.__lt__(var_1)
    assert var_3 is False
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    module_0.PDeque(*var_3)

def test_case_15():
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

def test_case_16():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 7
    var_3 = var_2.__hash__()
    assert var_3 == 3415519494622646258
    var_4 = var_1.extendleft(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 13
    var_5 = var_2.__repr__()
    assert var_5 == "pdeque(['9', '.', '_', '=', 'v', 'd', pdeque(['9', '.', '_', '=', 'v', 'd'])])"
    with pytest.raises(IndexError):
        var_1.__getitem__(var_3)

def test_case_17():
    var_0 = '9)._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 7
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 8
    var_3 = var_1.__lt__(var_1)
    assert var_3 is False
    var_4 = var_1.extendleft(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 15
    var_5 = var_2.__le__(var_2)
    var_6 = var_1.__getitem__(var_3)
    assert var_6 == '9'

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = True
    var_2 = module_0.pdeque(maxlen=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_2.__le__(var_2)
    var_4 = None
    var_5 = var_2.append(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 1
    var_6 = False
    var_7 = var_6.conjugate()
    assert var_7 == 0
    var_8 = var_5.append(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_8) == 1
    var_7.remove(var_4)

def test_case_19():
    var_0 = '9)._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 7
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 8
    var_3 = var_1.__lt__(var_1)
    assert var_3 is False
    var_4 = var_1.extendleft(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 15
    var_5 = var_2.__le__(var_2)
    var_6 = var_1.popleft()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 6
    var_7 = var_1.__getitem__(var_3)
    assert var_7 == '9'

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__hash__()
    assert var_2 == -1856064063567273487
    var_3 = var_1.extend(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 12
    var_4 = var_3.rotate(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 12
    var_4.__trunc__()

def test_case_21():
    var_0 = module_0.pdeque()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_1 = False
    var_2 = module_0.pdeque(maxlen=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 0
    var_3 = var_2.__le__(var_2)
    var_4 = None
    var_5 = var_2.append(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 0
    var_6 = var_0.appendleft(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 1
    with pytest.raises(TypeError):
        var_0.__getitem__(var_0)

def test_case_22():
    var_0 = module_0.dq()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_0) == 0
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_0.remove(var_0)

def test_case_23():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 7
    var_3 = var_1.extendleft(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 13
    var_4 = var_3.__repr__()
    assert var_4 == "pdeque([pdeque(['9', '.', '_', '=', 'v', 'd']), 'd', 'v', '=', '_', '.', '9', '9', '.', '_', '=', 'v', 'd'])"
    with pytest.raises(ValueError):
        var_2.remove(var_2)

def test_case_24():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 7
    var_3 = var_2.__reduce__()
    var_4 = var_2.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'itertools.chain'
    var_5 = var_2.__hash__()
    assert var_5 == 3415519494622646258
    var_6 = var_1.extendleft(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 13
    var_7 = var_2.__repr__()
    assert var_7 == "pdeque(['9', '.', '_', '=', 'v', 'd', pdeque(['9', '.', '_', '=', 'v', 'd'])])"
    var_8 = var_1.__eq__(var_1)
    assert var_8 is True
    with pytest.raises(IndexError):
        var_1.__getitem__(var_5)

def test_case_25():
    var_0 = '9._=vd'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 6
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.__hash__()
    assert var_2 == -1856064063567273487
    var_3 = var_1.extendleft(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_3) == 12
    var_4 = var_1.extend(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 12
    with pytest.raises(IndexError):
        var_1.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_5 = repr(var_4)
    assert var_5 == 'pdeque([1, 2, 3])'
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pdeque(var_6, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 3
    var_8 = repr(var_7)
    assert var_8 == 'pdeque([1, 2, 3], maxlen=3)'
    var_9 = []
    var_10 = 0
    var_11 = module_0.pdeque(var_9, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_11) == 0
    var_12 = repr(var_11)
    assert var_12 == 'pdeque([], maxlen=0)'
    var_13 = []
    var_14 = module_0.pdeque(var_13, var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_14) == 0
    var_15 = repr(var_14)
    assert var_15 == 'pdeque([], maxlen=1)'
    var_16 = [var_0]
    var_17 = module_0.pdeque(var_16, var_0)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_17) == 1
    var_18 = repr(var_17)
    assert var_18 == 'pdeque([1], maxlen=1)'
    var_19 = [var_0, var_1]
    var_20 = module_0.pdeque(var_19, var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_20) == 1
    var_21 = repr(var_20)
    assert var_21 == 'pdeque([2], maxlen=1)'
    var_22 = [var_0, var_1]
    var_23 = module_0.pdeque(var_22, var_1)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_23) == 2
    var_24 = repr(var_23)
    assert var_24 == 'pdeque([1, 2], maxlen=2)'
    var_25 = [var_0, var_1, var_2]
    var_26 = module_0.pdeque(var_25, var_1)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_26) == 2
    var_27 = repr(var_26)
    assert var_27 == 'pdeque([2, 3], maxlen=2)'
    var_28 = [var_0, var_1, var_2]
    var_29 = module_0.pdeque(var_28, var_2)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_29) == 3
    var_30 = repr(var_29)
    assert var_30 == 'pdeque([1, 2, 3], maxlen=3)'
    var_31 = [var_0, var_1, var_2]
    var_32 = 4
    var_33 = module_0.pdeque(var_31, var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_33) == 3
    var_34 = repr(var_33)
    assert var_34 == 'pdeque([1, 2, 3], maxlen=4)'
    var_35 = [var_0, var_1, var_2]
    var_36 = 5
    var_37 = module_0.pdeque(var_35, var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_37) == 3
    var_38 = repr(var_37)
    assert var_38 == 'pdeque([1, 2, 3], maxlen=5)'
    var_39 = [var_0, var_1, var_2]
    var_40 = 6
    var_41 = module_0.pdeque(var_39, var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_41) == 3
    var_42 = repr(var_41)
    assert var_42 == 'pdeque([1, 2, 3], maxlen=6)'
    var_43 = [var_0, var_1, var_2]
    var_44 = 7
    var_45 = module_0.pdeque(var_43, var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_45) == 3
    var_46 = repr(var_45)
    assert var_46 == 'pdeque([1, 2, 3], maxlen=7)'
    var_47 = [var_0, var_1, var_2]
    var_48 = 8
    var_49 = module_0.pdeque(var_47, var_48)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_49) == 3
    var_50 = repr(var_49)
    assert var_50 == 'pdeque([1, 2, 3], maxlen=8)'
    var_51 = [var_0, var_1, var_2]
    var_52 = 9
    var_53 = module_0.pdeque(var_51, var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_53) == 3
    var_54 = repr(var_53)
    assert var_54 == 'pdeque([1, 2, 3], maxlen=9)'
    var_55 = [var_0, var_1, var_2]
    var_56 = 10
    var_57 = module_0.pdeque(var_55, var_56)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_57) == 3
    var_58 = repr(var_57)
    assert var_58 == 'pdeque([1, 2, 3], maxlen=10)'
    var_59 = [var_0, var_1, var_2]
    var_60 = 11
    var_61 = module_0.pdeque(var_59, var_60)
    assert f'{type(var_61).__module__}.{type(var_61).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_61) == 3
    var_62 = repr(var_61)
    assert var_62 == 'pdeque([1, 2, 3], maxlen=11)'
    var_63 = [var_0, var_1, var_2]
    var_64 = 12
    var_65 = module_0.pdeque(var_63, var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_65) == 3
    var_66 = repr(var_65)
    assert var_66 == 'pdeque([1, 2, 3], maxlen=12)'
    var_67 = [var_0, var_1, var_2]
    var_68 = 13
    var_69 = module_0.pdeque(var_67, var_68)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_69) == 3
    var_70 = repr(var_69)
    assert var_70 == 'pdeque([1, 2, 3], maxlen=13)'
    var_71 = [var_0, var_1, var_2]
    var_72 = 14
    var_73 = module_0.pdeque(var_71, var_72)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_73) == 3
    var_74 = repr(var_73)
    assert var_74 == 'pdeque([1, 2, 3], maxlen=14)'
    var_75 = [var_0, var_1, var_2]
    var_76 = 15
    var_77 = module_0.pdeque(var_75, var_76)
    assert f'{type(var_77).__module__}.{type(var_77).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_77) == 3
    var_78 = repr(var_77)
    assert var_78 == 'pdeque([1, 2, 3], maxlen=15)'
    var_79 = [var_0, var_1, var_2]
    var_80 = 16
    var_81 = module_0.pdeque(var_79, var_80)
    assert f'{type(var_81).__module__}.{type(var_81).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_81) == 3
    var_82 = repr(var_81)
    assert var_82 == 'pdeque([1, 2, 3], maxlen=16)'
    var_83 = [var_0, var_1, var_2]
    var_84 = 17
    var_85 = module_0.pdeque(var_83, var_84)
    assert f'{type(var_85).__module__}.{type(var_85).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_85) == 3
    var_86 = repr(var_85)
    assert var_86 == 'pdeque([1, 2, 3], maxlen=17)'
    var_87 = [var_0, var_1, var_2]
    var_88 = 18
    var_89 = module_0.pdeque(var_87, var_88)
    assert f'{type(var_89).__module__}.{type(var_89).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_89) == 3
    var_90 = repr(var_89)
    assert var_90 == 'pdeque([1, 2, 3], maxlen=18)'
    var_91 = [var_0, var_1, var_2]
    var_92 = 19
    var_93 = module_0.pdeque(var_91, var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_93) == 3
    var_94 = repr(var_93)
    assert var_94 == 'pdeque([1, 2, 3], maxlen=19)'
    var_95 = [var_0, var_1, var_2]
    var_96 = 20
    var_97 = module_0.pdeque(var_95, var_96)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_97) == 3
    var_98 = repr(var_97)
    assert var_98 == 'pdeque([1, 2, 3], maxlen=20)'
    var_99 = [var_0, var_1, var_2]
    var_100 = 21
    var_101 = module_0.pdeque(var_99, var_100)
    assert f'{type(var_101).__module__}.{type(var_101).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_101) == 3
    var_102 = repr(var_101)
    assert var_102 == 'pdeque([1, 2, 3], maxlen=21)'
    var_103 = [var_0, var_1, var_2]
    var_104 = 22
    var_105 = module_0.pdeque(var_103, var_104)
    assert f'{type(var_105).__module__}.{type(var_105).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_105) == 3
    var_106 = repr(var_105)
    assert var_106 == 'pdeque([1, 2, 3], maxlen=22)'
    var_107 = [var_0, var_1, var_2]
    var_108 = 23
    var_109 = module_0.pdeque(var_107, var_108)
    assert f'{type(var_109).__module__}.{type(var_109).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_109) == 3
    var_110 = repr(var_109)
    assert var_110 == 'pdeque([1, 2, 3], maxlen=23)'
    var_111 = [var_0, var_1, var_2]
    var_112 = 24
    var_113 = module_0.pdeque(var_111, var_112)
    assert f'{type(var_113).__module__}.{type(var_113).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_113) == 3
    var_114 = repr(var_113)
    assert var_114 == 'pdeque([1, 2, 3], maxlen=24)'
    var_115 = [var_0, var_1, var_2]
    var_116 = 25
    var_117 = module_0.pdeque(var_115, var_116)
    assert f'{type(var_117).__module__}.{type(var_117).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_117) == 3
    var_118 = repr(var_117)
    assert var_118 == 'pdeque([1, 2, 3], maxlen=25)'
    var_119 = [var_0, var_1, var_2]
    var_120 = 26
    var_121 = module_0.pdeque(var_119, var_120)
    assert f'{type(var_121).__module__}.{type(var_121).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_121) == 3
    var_122 = repr(var_121)
    assert var_122 == 'pdeque([1, 2, 3], maxlen=26)'
    var_123 = [var_0, var_1, var_2]
    var_124 = 27
    var_125 = module_0.pdeque(var_123, var_124)
    assert f'{type(var_125).__module__}.{type(var_125).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_125) == 3
    var_126 = repr(var_125)
    assert var_126 == 'pdeque([1, 2, 3], maxlen=27)'
    var_127 = [var_0, var_1, var_2]
    var_128 = 28
    var_129 = module_0.pdeque(var_127, var_128)
    assert f'{type(var_129).__module__}.{type(var_129).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_129) == 3
    var_130 = repr(var_129)
    assert var_130 == 'pdeque([1, 2, 3], maxlen=28)'
    var_131 = [var_0, var_1, var_2]
    var_132 = 29
    var_133 = module_0.pdeque(var_131, var_132)
    assert f'{type(var_133).__module__}.{type(var_133).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_133) == 3
    var_134 = repr(var_133)
    assert var_134 == 'pdeque([1, 2, 3], maxlen=29)'
    var_135 = [var_0, var_1, var_2]
    var_136 = 30
    var_137 = module_0.pdeque(var_135, var_136)
    assert f'{type(var_137).__module__}.{type(var_137).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_137) == 3
    var_138 = repr(var_137)
    assert var_138 == 'pdeque([1, 2, 3], maxlen=30)'
    var_139 = [var_0, var_1, var_2]
    var_140 = 31
    var_141 = module_0.pdeque(var_139, var_140)
    assert f'{type(var_141).__module__}.{type(var_141).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_141) == 3
    var_142 = repr(var_141)
    assert var_142 == 'pdeque([1, 2, 3], maxlen=31)'
    var_143 = [var_0, var_1, var_2]
    var_144 = 32
    var_145 = module_0.pdeque(var_143, var_144)
    assert f'{type(var_145).__module__}.{type(var_145).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_145) == 3
    var_146 = repr(var_145)
    assert var_146 == 'pdeque([1, 2, 3], maxlen=32)'
    var_147 = [var_0, var_1, var_2]
    var_148 = 33
    var_149 = module_0.pdeque(var_147, var_148)
    assert f'{type(var_149).__module__}.{type(var_149).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_149) == 3
    var_150 = repr(var_149)
    assert var_150 == 'pdeque([1, 2, 3], maxlen=33)'
    var_151 = [var_0, var_1, var_2]
    var_152 = 34
    var_153 = module_0.pdeque(var_151, var_152)
    assert f'{type(var_153).__module__}.{type(var_153).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_153) == 3
    var_154 = repr(var_153)
    assert var_154 == 'pdeque([1, 2, 3], maxlen=34)'
    var_155 = [var_0, var_1, var_2]
    var_156 = 35
    var_157 = module_0.pdeque(var_155, var_156)
    assert f'{type(var_157).__module__}.{type(var_157).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_157) == 3
    var_158 = repr(var_157)
    assert var_158 == 'pdeque([1, 2, 3], maxlen=35)'
    var_159 = [var_0, var_1, var_2]
    var_160 = 36
    var_161 = module_0.pdeque(var_159, var_160)
    assert f'{type(var_161).__module__}.{type(var_161).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_161) == 3
    var_162 = repr(var_161)
    assert var_162 == 'pdeque([1, 2, 3], maxlen=36)'
    var_163 = [var_0, var_1, var_2]
    var_164 = 37
    var_165 = module_0.pdeque(var_163, var_164)
    assert f'{type(var_165).__module__}.{type(var_165).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_165) == 3
    var_166 = repr(var_165)
    assert var_166 == 'pdeque([1, 2, 3], maxlen=37)'
    var_167 = [var_0, var_1, var_2]
    var_168 = -209
    module_0.pdeque(var_167, var_168)

def test_case_27():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_5 = repr(var_4)
    assert var_5 == 'pdeque([1, 2, 3])'
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pdeque(var_6, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_7) == 3
    var_8 = repr(var_7)
    assert var_8 == 'pdeque([1, 2, 3], maxlen=3)'
    var_9 = [var_6]
    var_10 = 0
    var_11 = module_0.pdeque(var_9, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_11) == 1
    var_12 = repr(var_11)
    var_13 = []
    var_14 = module_0.pdeque(var_13, var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_14) == 0
    var_15 = repr(var_14)
    assert var_15 == 'pdeque([], maxlen=1)'
    var_16 = [var_0]
    var_17 = module_0.pdeque(var_16, var_0)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_17) == 1
    var_18 = repr(var_17)
    assert var_18 == 'pdeque([1], maxlen=1)'
    var_19 = [var_0, var_1]
    var_20 = module_0.pdeque(var_19, var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_20) == 1
    var_21 = repr(var_20)
    assert var_21 == 'pdeque([2], maxlen=1)'
    var_22 = [var_0, var_1]
    var_23 = module_0.pdeque(var_22, var_1)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_23) == 2
    var_24 = repr(var_23)
    assert var_24 == 'pdeque([1, 2], maxlen=2)'
    var_25 = [var_0, var_1, var_2]
    var_26 = module_0.pdeque(var_25, var_1)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_26) == 2
    var_27 = repr(var_26)
    assert var_27 == 'pdeque([2, 3], maxlen=2)'
    var_28 = [var_0, var_1, var_2]
    var_29 = module_0.pdeque(var_28, var_2)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_29) == 3
    var_30 = repr(var_29)
    assert var_30 == 'pdeque([1, 2, 3], maxlen=3)'
    var_31 = [var_0, var_1, var_2]
    var_32 = 4
    var_33 = module_0.pdeque(var_31, var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_33) == 3
    var_34 = repr(var_33)
    assert var_34 == 'pdeque([1, 2, 3], maxlen=4)'
    var_35 = [var_0, var_1, var_2]
    var_36 = 5
    var_37 = module_0.pdeque(var_35, var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_37) == 3
    var_38 = [var_0, var_1, var_2]
    var_39 = 6
    var_40 = module_0.pdeque(var_38, var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_40) == 3
    var_41 = repr(var_40)
    assert var_41 == 'pdeque([1, 2, 3], maxlen=6)'
    var_42 = [var_0, var_1, var_2]
    var_43 = 7
    var_44 = module_0.pdeque(var_42, var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_44) == 3
    var_45 = repr(var_44)
    assert var_45 == 'pdeque([1, 2, 3], maxlen=7)'
    var_46 = [var_0, var_1, var_2]
    var_47 = 8
    var_48 = module_0.pdeque(var_46, var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_48) == 3
    var_49 = repr(var_48)
    assert var_49 == 'pdeque([1, 2, 3], maxlen=8)'
    var_50 = [var_0, var_1, var_2]
    var_51 = 9
    var_52 = module_0.pdeque(var_50, var_51)
    assert f'{type(var_52).__module__}.{type(var_52).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_52) == 3
    var_53 = repr(var_52)
    assert var_53 == 'pdeque([1, 2, 3], maxlen=9)'
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.pdeque(var_54, var_10)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_55) == 3
    var_56 = repr(var_55)
    var_57 = [var_0, var_1, var_2]
    var_58 = 11
    var_59 = module_0.pdeque(var_57, var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_59) == 3
    var_60 = repr(var_59)
    assert var_60 == 'pdeque([1, 2, 3], maxlen=11)'
    var_61 = [var_0, var_1, var_2]
    var_62 = 12
    var_63 = module_0.pdeque(var_61, var_62)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_63) == 3
    var_64 = repr(var_63)
    assert var_64 == 'pdeque([1, 2, 3], maxlen=12)'
    var_65 = [var_0, var_1, var_2]
    var_66 = 13
    var_67 = module_0.pdeque(var_65, var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_67) == 3
    var_68 = repr(var_67)
    assert var_68 == 'pdeque([1, 2, 3], maxlen=13)'
    var_69 = [var_0, var_1, var_2]
    var_70 = 14
    var_71 = module_0.pdeque(var_69, var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_71) == 3
    var_72 = repr(var_71)
    assert var_72 == 'pdeque([1, 2, 3], maxlen=14)'
    var_73 = [var_0, var_1, var_2]
    var_74 = 15
    var_75 = module_0.pdeque(var_73, var_74)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_75) == 3
    var_76 = repr(var_75)
    assert var_76 == 'pdeque([1, 2, 3], maxlen=15)'
    var_77 = [var_0, var_1, var_2]
    var_78 = 16
    var_79 = module_0.pdeque(var_77, var_78)
    assert f'{type(var_79).__module__}.{type(var_79).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_79) == 3
    var_80 = repr(var_79)
    assert var_80 == 'pdeque([1, 2, 3], maxlen=16)'
    var_81 = 17
    var_82 = module_0.pdeque(var_17, var_81)
    assert f'{type(var_82).__module__}.{type(var_82).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_82) == 1
    var_83 = repr(var_82)
    var_84 = [var_0, var_1, var_2]
    var_85 = 18
    var_86 = module_0.pdeque(var_84, var_85)
    assert f'{type(var_86).__module__}.{type(var_86).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_86) == 3
    var_87 = repr(var_86)
    assert var_87 == 'pdeque([1, 2, 3], maxlen=18)'
    var_88 = [var_0, var_1, var_2]
    var_89 = 19
    var_90 = module_0.pdeque(var_88, var_89)
    assert f'{type(var_90).__module__}.{type(var_90).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_90) == 3
    var_91 = repr(var_90)
    assert var_91 == 'pdeque([1, 2, 3], maxlen=19)'
    var_92 = [var_0, var_1, var_2]
    var_93 = 20
    var_94 = module_0.pdeque(var_92, var_93)
    assert f'{type(var_94).__module__}.{type(var_94).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_94) == 3
    var_95 = repr(var_94)
    assert var_95 == 'pdeque([1, 2, 3], maxlen=20)'
    var_96 = [var_0, var_1, var_2]
    var_97 = 4
    var_98 = module_0.pdeque(var_96, var_97)
    assert f'{type(var_98).__module__}.{type(var_98).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_98) == 3
    var_99 = repr(var_98)
    var_100 = [var_0, var_1, var_2]
    var_101 = 22
    var_102 = module_0.pdeque(var_100, var_101)
    assert f'{type(var_102).__module__}.{type(var_102).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_102) == 3
    var_103 = repr(var_102)
    assert var_103 == 'pdeque([1, 2, 3], maxlen=22)'
    var_104 = [var_0, var_1, var_2]
    var_105 = 23
    var_106 = module_0.pdeque(var_104, var_105)
    assert f'{type(var_106).__module__}.{type(var_106).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_106) == 3
    var_107 = repr(var_106)
    assert var_107 == 'pdeque([1, 2, 3], maxlen=23)'
    var_108 = [var_0, var_1, var_2]
    var_109 = 24
    var_110 = module_0.pdeque(var_108, var_109)
    assert f'{type(var_110).__module__}.{type(var_110).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_110) == 3
    var_111 = repr(var_110)
    assert var_111 == 'pdeque([1, 2, 3], maxlen=24)'
    var_112 = [var_0, var_1, var_2]
    var_113 = 25
    var_114 = module_0.pdeque(var_112, var_113)
    assert f'{type(var_114).__module__}.{type(var_114).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_114) == 3
    var_115 = repr(var_114)
    assert var_115 == 'pdeque([1, 2, 3], maxlen=25)'
    var_116 = [var_0, var_1, var_2]
    var_117 = 26
    var_118 = module_0.pdeque(var_116, var_117)
    assert f'{type(var_118).__module__}.{type(var_118).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_118) == 3
    var_119 = repr(var_118)
    assert var_119 == 'pdeque([1, 2, 3], maxlen=26)'
    var_120 = [var_0, var_1, var_2]
    var_121 = 27
    var_122 = module_0.pdeque(var_120, var_121)
    assert f'{type(var_122).__module__}.{type(var_122).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_122) == 3
    var_123 = repr(var_122)
    assert var_123 == 'pdeque([1, 2, 3], maxlen=27)'
    var_124 = [var_0, var_1, var_2]
    var_125 = 28
    var_126 = module_0.pdeque(var_124, var_125)
    assert f'{type(var_126).__module__}.{type(var_126).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_126) == 3
    var_127 = repr(var_126)
    assert var_127 == 'pdeque([1, 2, 3], maxlen=28)'
    var_128 = [var_0, var_1, var_2]
    var_129 = 29
    var_130 = module_0.pdeque(var_128, var_129)
    assert f'{type(var_130).__module__}.{type(var_130).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_130) == 3
    var_131 = repr(var_130)
    assert var_131 == 'pdeque([1, 2, 3], maxlen=29)'
    var_132 = [var_0, var_1, var_2]
    var_133 = 30
    var_134 = module_0.pdeque(var_132, var_133)
    assert f'{type(var_134).__module__}.{type(var_134).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_134) == 3
    var_135 = repr(var_134)
    assert var_135 == 'pdeque([1, 2, 3], maxlen=30)'
    var_136 = [var_0, var_1, var_2]
    var_137 = 31
    var_138 = module_0.pdeque(var_136, var_137)
    assert f'{type(var_138).__module__}.{type(var_138).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_138) == 3
    var_139 = repr(var_138)
    assert var_139 == 'pdeque([1, 2, 3], maxlen=31)'
    var_140 = [var_0, var_1, var_2]
    var_141 = 32
    var_142 = module_0.pdeque(var_140, var_141)
    assert f'{type(var_142).__module__}.{type(var_142).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_142) == 3
    var_143 = repr(var_142)
    assert var_143 == 'pdeque([1, 2, 3], maxlen=32)'
    var_144 = [var_0, var_1, var_2]
    var_145 = 33
    var_146 = module_0.pdeque(var_144, var_145)
    assert f'{type(var_146).__module__}.{type(var_146).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_146) == 3
    var_147 = repr(var_146)
    assert var_147 == 'pdeque([1, 2, 3], maxlen=33)'
    var_148 = [var_0, var_1, var_2]
    var_149 = 34
    var_150 = module_0.pdeque(var_148, var_149)
    assert f'{type(var_150).__module__}.{type(var_150).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_150) == 3
    var_151 = repr(var_150)
    assert var_151 == 'pdeque([1, 2, 3], maxlen=34)'
    var_152 = [var_0, var_1, var_2]
    var_153 = 35
    var_154 = module_0.pdeque(var_152, var_153)
    assert f'{type(var_154).__module__}.{type(var_154).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_154) == 3
    var_155 = repr(var_154)
    assert var_155 == 'pdeque([1, 2, 3], maxlen=35)'
    var_156 = [var_0, var_1, var_2]
    var_157 = 36
    var_158 = module_0.pdeque(var_156, var_157)
    assert f'{type(var_158).__module__}.{type(var_158).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_158) == 3
    var_159 = repr(var_158)
    assert var_159 == 'pdeque([1, 2, 3], maxlen=36)'
    var_160 = [var_0, var_1, var_2]
    var_161 = 37
    var_162 = module_0.pdeque(var_160, var_161)
    assert f'{type(var_162).__module__}.{type(var_162).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_162) == 3
    var_163 = repr(var_162)
    assert var_163 == 'pdeque([1, 2, 3], maxlen=37)'
    var_164 = [var_0, var_1, var_2]
    var_165 = 38
    var_166 = module_0.pdeque(var_164, var_165)
    assert f'{type(var_166).__module__}.{type(var_166).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_166) == 3
    var_167 = repr(var_166)
    assert var_167 == 'pdeque([1, 2, 3], maxlen=38)'
    var_168 = 39
    var_169 = module_0.pdeque(var_115, var_168)
    assert f'{type(var_169).__module__}.{type(var_169).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_169) == 28
    var_170 = repr(var_169)
    var_171 = [var_0, var_1, var_2]
    var_172 = 40
    var_173 = module_0.pdeque(var_171, var_172)
    assert f'{type(var_173).__module__}.{type(var_173).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_173) == 3
    var_174 = repr(var_173)
    assert var_174 == 'pdeque([1, 2, 3], maxlen=40)'
    var_175 = [var_0, var_1, var_2]
    var_176 = 41
    var_177 = module_0.pdeque(var_175, var_176)
    assert f'{type(var_177).__module__}.{type(var_177).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_177) == 3
    var_178 = repr(var_177)
    assert var_178 == 'pdeque([1, 2, 3], maxlen=41)'
    var_179 = [var_0, var_1, var_2]
    var_180 = 42
    var_181 = module_0.pdeque(var_179, var_180)
    assert f'{type(var_181).__module__}.{type(var_181).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_181) == 3
    var_182 = repr(var_181)
    assert var_182 == 'pdeque([1, 2, 3], maxlen=42)'
    var_183 = var_181

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = 'G2;'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 4
    var_3 = var_1.__lt__(var_1)
    assert var_3 is False
    module_0.PDeque(*var_2)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 'G2;'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 3
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 4
    var_3 = var_1.__lt__(var_1)
    assert var_3 is False
    var_4 = var_1.extendleft(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_4) == 7
    var_5 = var_4.__repr__()
    assert var_5 == "pdeque([pdeque(['G', '2', ';']), ';', '2', 'G', 'G', '2', ';'])"
    var_6 = var_2.rotate(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_6) == 4
    var_7 = var_5.__eq__(var_4)
    var_8 = var_6.__eq__(var_4)
    assert var_8 is False
    var_9 = -583.1
    var_10 = var_1.popleft(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_10) == 0
    var_3.__getitem__(var_6)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = '9.'
    var_1 = module_0.dq(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_1) == 2
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.PDeque.right).__module__}.{type(module_0.PDeque.right).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.left).__module__}.{type(module_0.PDeque.left).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.PDeque.maxlen).__module__}.{type(module_0.PDeque.maxlen).__qualname__}' == 'builtins.property'
    var_2 = var_1.append(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_2) == 3
    var_3 = var_2.__iter__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'itertools.chain'
    var_4 = var_3.__hash__()
    var_5 = var_1.extendleft(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._pdeque.PDeque'
    assert len(var_5) == 5
    var_6 = var_2.__repr__()
    assert var_6 == "pdeque(['9', '.', pdeque(['9', '.'])])"
    var_7 = var_1.__eq__(var_1)
    assert var_7 is True
    var_8 = '*/=&WL&J/3-$wrnL:z\r'
    var_9 = var_7.__eq__(var_8)
    var_10 = var_7.__invert__()
    assert var_10 == -2
    var_11 = var_1.__getitem__(var_10)
    assert var_11 == '9'
    var_2.__trunc__()