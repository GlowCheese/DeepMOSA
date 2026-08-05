# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._precord as module_0

def test_case_0():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'

def test_case_1():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    with pytest.raises(AttributeError):
        var_0.__getattr__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = None
    var_0.set(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.PRecord()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_1) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_2 = var_1.discard(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_2) == 0
    var_3 = var_2.__repr__()
    assert var_3 == 'PRecord()'
    var_4 = ':$hF'
    var_5 = var_2.serialize(var_0)
    var_6 = "#$'!{+CY2L"
    var_7 = var_1.__reduce__()
    var_8 = {var_4: var_0, var_4: var_0, var_6: var_0}
    var_9 = var_1.__len__()
    assert var_9 == 0
    module_0.PRecord(**var_8)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = "#$'!{+CY2L"
    var_2 = {var_1: var_0, var_1: var_0, var_1: var_0}
    module_0.PRecord(**var_2)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.set()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_1) == 0
    var_2 = None
    var_0.__add__(var_2)

def test_case_6():
    var_0 = module_0.PRecord()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._precord.PRecord'
    assert len(var_0) == 0
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.PRecord.create).__module__}.{type(module_0.PRecord.create).__qualname__}' == 'builtins.method'
    var_1 = var_0.serialize()