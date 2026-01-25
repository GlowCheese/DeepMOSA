# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_1
import re as module_2

import pyrsistent._transformations as module_0
import pytest


def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.discard(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = {}
    var_1 = 'any_key'
    var_2 = None
    var_3 = module_0.ny(var_2)
    assert var_3 is True
    var_4 = 'default_value'
    var_5 = module_0._get(var_0, var_1, var_4)
    assert var_5 == 'default_value'
    module_0.transform(var_5, var_5)

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
    var_0 = []
    var_1 = lambda i, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0._get_keys_and_values(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = []
    var_1 = lambda i, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = var_1.__repr__()
    assert var_2 == '{}'
    var_3 = module_0.discard(var_1, var_0)
    var_4 = None
    var_5 = module_0.ny(var_4)
    assert var_5 is True
    module_2.escape(var_4)

def test_case_10():
    var_0 = []
    var_1 = lambda i, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = module_0.transform(var_1, var_2)

def test_case_11():
    var_0 = []
    var_1 = 0
    var_2 = 'default_value'
    var_3 = module_0._get(var_0, var_1, var_2)
    assert var_3 == 'default_value'

def test_case_12():
    var_0 = 5
    var_1 = []
    var_2 = 10
    var_3 = module_0._do_to_path(var_0, var_1, var_2)
    assert var_3 == 10

def test_case_13():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = module_2.error
    module_0._get_keys_and_values(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 19
    var_1 = [var_0, var_0]
    var_2 = lambda i: i % var_1 == var_1
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = -2822
    var_1 = [var_0]
    var_2 = lambda i, v: v > var_1
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = []
    var_1 = lambda i, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0.rex(var_1)

def test_case_18():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = ''
    var_3 = var_1(var_2)

def test_case_19():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = var_1(var_1)

def test_case_20():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda k, v, extra: var_5
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_4, var_6)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = []
    var_1 = lambda i, v: v > var_0
    var_2 = None
    module_0._do_to_path(var_1, var_2, var_1)

def test_case_22():
    var_0 = []
    var_1 = lambda i: i % var_0 == var_0
    var_2 = module_0.ny(var_1)
    assert var_2 is True
    var_3 = module_0._get_keys_and_values(var_0, var_1)