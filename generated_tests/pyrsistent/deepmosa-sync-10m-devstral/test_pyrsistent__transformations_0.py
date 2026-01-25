# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._transformations as module_0
import enum as module_1

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
    var_0 = None
    module_0.inc(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.dec(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.rex(var_0)

def test_case_5():
    var_0 = None
    var_1 = module_0.ny(var_0)
    assert var_1 is True

def test_case_6():
    var_0 = ''
    var_1 = module_0.transform(var_0, var_0)
    assert var_1 == ''

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'N'
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'WN'
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'fN'
    var_1 = module_0.rex(var_0)
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = bool(var_0 == [])
    module_0._get_keys_and_values(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.discard(var_0, var_1)
    var_3 = '[i0u\n<{c|NCr\rwE~m'
    var_2.__call__(var_2, var_3, module=var_2)

def test_case_11():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = module_0._get_keys_and_values(var_2, var_3)

def test_case_12():
    var_0 = 'N'
    var_1 = module_0.rex(var_0)
    var_2 = module_0._get_keys_and_values(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'c'
    var_1 = 3
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = lambda k, v: v > var_1
    module_0._get_keys_and_values(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = -1
    var_1 = lambda k, v: v > var_0
    module_0._get_keys_and_values(var_1, var_1)

def test_case_15():
    var_0 = 't'
    var_1 = module_0.rex(var_0)
    var_2 = module_1.EnumMeta
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_1, var_2)

def test_case_16():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

def test_case_17():
    var_0 = 'c'
    var_1 = module_0.rex(var_0)
    var_2 = module_1._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = lambda k, v: v > var_1
    var_4 = module_0._get_keys_and_values(var_2, var_3)

def test_case_18():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = lambda x: x.clear() or x
    var_7 = module_0._do_to_path(var_4, var_5, var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True

def test_case_19():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = module_0._do_to_path(var_4, var_5, var_8)
    var_10 = bool(var_9 == {'c': 3})
    assert var_10 is True

def test_case_20():
    var_0 = 'test_\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = var_1(var_2)
    var_4 = 123
    var_5 = var_1(var_4)
    assert var_5 is False