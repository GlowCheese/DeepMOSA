# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._transformations as module_0
import enum as module_1
import builtins as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.discard(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = b'>9\xae\xb4Kx\xd7X\xe0\x894\x03z\xa8\r$'
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.inc(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.dec(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.rex(var_0)

def test_case_6():
    var_0 = None
    var_1 = module_0.ny(var_0)
    assert var_1 is True

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'R'
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = lambda k, v: v > var_0
    var_3 = module_0._get_keys_and_values(var_1, var_2)
    var_4 = module_0._get_keys_and_values(var_3, var_2)
    var_5 = None
    module_0._get_keys_and_values(var_3, var_5)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'R'
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = lambda k, v: v > var_0
    var_3 = module_0._get_keys_and_values(var_1, var_2)
    var_4 = module_0._get_keys_and_values(var_3, var_2)
    module_0._get_keys_and_values(var_2, var_3)

def test_case_9():
    var_0 = 'R'
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = lambda k, v: v > var_0
    var_3 = module_0._get_keys_and_values(var_1, var_2)
    var_4 = module_0._get_keys_and_values(var_3, var_2)
    var_5 = None
    var_6 = module_0.discard(var_1, var_5)

def test_case_10():
    var_0 = b''
    var_1 = module_0.transform(var_0, var_0)
    assert var_1 == b''

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = var_0.__repr__()
    assert var_1 == '{}'
    module_0.transform(var_0, var_1)

def test_case_12():
    var_0 = []
    var_1 = module_0._items(var_0)

def test_case_13():
    var_0 = 'c'
    var_1 = module_0.rex(var_0)
    var_2 = module_0._get_keys_and_values(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'R'
    var_1 = lambda k, v: v > var_0
    module_0._get_keys_and_values(var_0, var_1)

def test_case_15():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda k, v, x: var_3
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_2, var_4)

def test_case_16():
    var_0 = None
    var_1 = module_0._do_to_path(var_0, var_0, var_0)

def test_case_17():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = lambda x: x.clear()
    var_7 = module_0._do_to_path(var_4, var_5, var_6)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'c'
    var_1 = module_0.rex(var_0)
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = [var_1]
    module_0._do_to_path(var_2, var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = lambda k, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0.discard(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = ''
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = lambda k, v: v > var_0
    var_3 = module_0._get_keys_and_values(var_1, var_2)
    var_4 = None
    var_5 = module_1.Flag
    module_0._get_keys_and_values(var_4, var_5)

def test_case_21():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = lambda k, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = module_1.EnumMeta
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = lambda k, v: v > var_0
    var_2 = module_2.dict
    module_0._get_keys_and_values(var_2, var_1)