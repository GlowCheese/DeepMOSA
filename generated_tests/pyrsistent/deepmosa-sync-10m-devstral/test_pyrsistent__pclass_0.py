# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._pclass as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'

def test_case_1():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__hash__()
    assert var_1 == 5740354900026072187

def test_case_2():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()

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
    var_0 = module_0.PClass()
    var_1 = var_0.__hash__()
    var_2 = var_0.evolver()
    var_3 = module_1.CheckedType()
    var_4 = var_2.set(var_2, var_2)
    var_5 = var_4.__getitem__(var_4)

def test_case_6():
    var_0 = None
    var_1 = None
    var_2 = module_0._PClassEvolver(var_1, var_1)
    var_2.__setitem__(var_0, var_0)

def test_case_7():
    var_0 = None
    var_1 = module_0.PClass()
    var_2 = var_1.__ne__(var_0)
    var_3 = var_1.set()
    var_4 = var_1.evolver()
    var_5 = var_1.serialize()
    var_6 = module_0.PClass()
    var_7 = var_6.__repr__()
    var_8 = var_1.__reduce__()
    var_4.__getattr__(var_5)

def test_case_8():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()

def test_case_9():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()

def test_case_10():
    var_0 = None
    var_1 = module_0._PClassEvolver(var_0, var_0)
    var_2 = var_1.persistent()
    var_3 = module_0.PClass()

def test_case_11():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'

def test_case_12():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()
    var_2 = var_0.set()
    var_3 = None
    var_4 = var_2.__eq__(var_3)
    var_5 = var_1.persistent()

def test_case_13():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__ne__(var_0)
    assert var_1 is False

def test_case_14():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.transform()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_0.set(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_0.remove(var_0)

def test_case_17():
    var_0 = None
    var_1 = module_0.PClass()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.__ne__(var_0)
    assert var_2 is True
    var_3 = var_1.set()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pclass.PClass'
    var_4 = None
    var_5 = var_1.serialize()
    var_6 = module_0.PClass()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._pclass.PClass'
    var_7 = var_6.__repr__()
    assert var_7 == 'PClass()'
    with pytest.raises(AttributeError):
        var_6.__setattr__(var_4, var_2)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = 'y6K'
    var_2 = 'Er!mL}%1bH\x0bt\\L%A,Q'
    var_3 = {var_1: var_0, var_1: var_0, var_2: var_0, var_1: var_0}
    module_0.PClass(**var_3)

def test_case_19():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()
    var_2 = var_1.set(var_1, var_1)

def test_case_20():
    var_0 = None
    var_1 = module_0.PClass()
    var_2 = var_1.__ne__(var_1)
    var_3 = var_1.set()
    var_4 = var_1.__repr__()
    var_5 = var_1.__reduce__()
    var_6 = var_1.evolver()
    var_7 = var_6.__setattr__(var_1, var_4)
    var_1.remove(var_0)

def test_case_21():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'

def test_case_22():
    var_0 = module_0.PClass()
    var_1 = var_0.set()
    var_2 = var_0.evolver()
    var_3 = None
    var_4 = var_2.__repr__()
    var_5 = var_2.__hash__()
    var_6 = var_0.serialize()
    var_7 = var_2.set(var_2, var_3)
    var_2.persistent()

def test_case_23():
    var_0 = None
    var_1 = module_0.PClass()
    var_2 = var_1.__ne__(var_1)
    var_3 = var_1.__ne__(var_0)
    var_4 = var_1.set()
    var_5 = var_4.__hash__()
    var_6 = var_1.serialize()
    var_7 = var_1.__repr__()
    var_8 = var_1.evolver()
    var_9 = var_8.__setattr__(var_5, var_7)
    var_10 = var_8.remove(var_5)