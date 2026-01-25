# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._pclass as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_0.set(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_0.remove(var_0)

def test_case_3():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'PClass()'

def test_case_4():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    with pytest.raises(AttributeError):
        var_0.__delattr__(var_0)

def test_case_5():
    var_0 = None
    var_1 = module_0.PClass()
    var_2 = var_1.evolver()
    var_3 = var_2.persistent()
    var_4 = var_1.serialize()
    var_5 = var_3.evolver()
    var_6 = var_3.set(**var_4)
    var_7 = var_1.__str__()
    var_8 = var_3.__repr__()
    var_0.__new__(var_5, **var_2)

def test_case_6():
    var_0 = None
    var_1 = None
    var_2 = module_0._PClassEvolver(var_1, var_1)
    var_2.__setitem__(var_0, var_0)

def test_case_7():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize(var_0)

def test_case_8():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.__ne__(var_1)
    assert var_2 is True
    var_3 = var_2.__ne__(var_2)
    assert var_3 is False

def test_case_9():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__ne__(var_0)
    assert var_1 is False

def test_case_10():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = None
    with pytest.raises(AttributeError):
        var_0.__setattr__(var_1, var_0)

def test_case_11():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()

def test_case_12():
    var_0 = module_0.PClass()
    var_1 = None
    var_2 = var_0.transform()
    var_3 = var_2.__ne__(var_2)
    var_4 = var_2.evolver()
    var_4.set(*var_1)

def test_case_13():
    var_0 = module_0.PClass()
    var_1 = None
    var_2 = var_0.evolver()
    var_3 = var_2.set(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = 'NnCGF'
    var_2 = 'N<{`li[v'
    var_3 = {var_1: var_0, var_1: var_0, var_2: var_0}
    module_0.PClass(**var_3)

def test_case_15():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'

def test_case_16():
    var_0 = ()
    var_1 = None
    var_2 = module_0._PClassEvolver(var_0, var_1)
    var_3 = var_2.persistent()
    var_4 = None
    var_5 = None
    var_6 = None
    var_7 = module_0.PClass()
    var_8 = var_7.serialize()
    var_9 = var_7.__repr__()
    module_1.serialize(var_4, var_5, var_6)

def test_case_17():
    var_0 = module_0.PClass()
    var_1 = None
    var_2 = var_0.evolver()
    var_3 = var_2.set(var_2, var_1)
    var_4 = var_3.__setattr__(var_3, var_1)

def test_case_18():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()
    var_2 = var_1.persistent()
    var_3 = None
    var_4 = None
    var_5 = var_2.__reduce__()
    var_6 = module_0.PClass()
    var_7 = var_1.__setitem__(var_3, var_4)
    var_8 = var_0.__eq__(var_2)
    var_1.persistent()

def test_case_19():
    var_0 = None
    var_1 = module_0.PClass()
    var_2 = var_1.evolver()
    var_3 = var_2.set(var_0, var_0)
    var_4 = var_2.__delitem__(var_0)
    var_5 = var_2.persistent()
    var_6 = var_5.__reduce__()
    var_7 = module_0.PClass()
    var_8 = var_5.__eq__(var_6)
    var_8.persistent()