# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = []
    var_1 = module_0.PClass(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_2 = module_0.PClass()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._pclass.PClass'
    var_3 = var_2.transform()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pclass.PClass'
    var_4 = var_1.__repr__()
    assert var_4 == 'PClass()'
    var_2.set(*var_4)

def test_case_2():
    var_0 = module_0.PClass()
    var_1 = var_0.transform()
    var_2 = var_0.__repr__()
    var_3 = var_0.__eq__(var_2)
    var_4 = var_1.evolver()
    var_5 = var_0.__reduce__()
    var_4.remove(var_5)

def test_case_3():
    var_0 = []
    var_1 = module_0.PClass(*var_0)
    var_2 = var_1.__ne__(var_1)
    var_3 = var_1.transform()
    var_4 = var_3.transform()
    var_5 = var_3.set()
    var_6 = var_5.__eq__(var_4)
    var_7 = var_5.evolver()
    var_8 = var_4.__hash__()
    var_9 = var_5.__reduce__()
    var_10 = var_7.persistent()
    var_11 = module_0.PClass()

def test_case_4():
    var_0 = None
    var_1 = module_0._PClassEvolver(var_0, var_0)

def test_case_5():
    var_0 = []
    var_1 = module_0.PClass(*var_0)
    var_2 = var_1.evolver()
    module_0.PClass(**var_2)

def test_case_6():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()

def test_case_7():
    var_0 = module_0.PClass()
    var_1 = var_0.evolver()

def test_case_8():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__repr__()
    assert var_1 == 'PClass()'

def test_case_9():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__reduce__()

def test_case_10():
    var_0 = module_0.PClass()
    var_1 = var_0.set()
    var_2 = var_0.__eq__(var_1)
    var_3 = var_1.evolver()
    var_4 = None
    var_5 = var_2.__str__()
    var_6 = var_0.serialize()
    var_7 = var_3.__setattr__(var_2, var_4)

def test_case_11():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_0.remove(var_0)

def test_case_13():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__hash__()
    assert var_1 == 5740354900026072187

def test_case_14():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    with pytest.raises(AttributeError):
        var_0.__setattr__(var_0, var_0)

def test_case_15():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    with pytest.raises(AttributeError):
        var_0.__delattr__(var_0)

def test_case_16():
    var_0 = []
    var_1 = module_0.PClass(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.__hash__()
    assert var_2 == 5740354900026072187
    var_3 = var_1.transform()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._pclass.PClass'

def test_case_17():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.__ne__(var_0)
    assert var_1 is False

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0.PClass()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._pclass.PClass'
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PClass.create).__module__}.{type(module_0.PClass.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.transform()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._pclass.PClass'
    var_1.set(var_1, var_1)

def test_case_19():
    var_0 = module_0.PClass()
    var_1 = None
    var_2 = var_0.set()
    var_3 = var_0.__eq__(var_2)
    var_4 = var_2.evolver()
    var_5 = var_0.__reduce__()
    var_6 = var_3.__str__()
    var_7 = var_4.__setitem__(var_1, var_4)
    var_7.__reduce__()

def test_case_20():
    var_0 = []
    var_1 = module_0.PClass(*var_0)
    var_2 = var_1.__ne__(var_1)
    var_3 = var_1.transform()
    var_4 = var_3.transform()
    var_5 = var_3.set()
    var_6 = var_5.__eq__(var_4)
    var_7 = var_5.evolver()
    var_8 = var_4.__hash__()
    var_9 = var_7.set(var_7, var_2)
    var_10 = var_5.__reduce__()
    var_7.persistent()

def test_case_21():
    var_0 = []
    var_1 = module_0.PClass(*var_0)
    var_2 = None
    var_3 = var_1.evolver()
    var_4 = var_3.__setitem__(var_2, var_2)
    var_5 = var_1.transform()
    var_6 = var_1.__reduce__()
    var_7 = var_5.transform()
    var_8 = var_5.set()
    var_9 = var_8.__eq__(var_8)
    var_10 = var_3.__str__()
    var_11 = None
    var_12 = var_7.serialize()
    var_13 = var_3.remove(var_11)
    var_5.set(var_2, var_7)

def test_case_22():
    var_0 = []
    var_1 = module_0.PClass(*var_0)
    var_2 = None
    var_3 = var_1.evolver()
    var_4 = var_3.__setitem__(var_3, var_2)
    var_5 = var_3.set(var_3, var_4)
    var_5.transform()