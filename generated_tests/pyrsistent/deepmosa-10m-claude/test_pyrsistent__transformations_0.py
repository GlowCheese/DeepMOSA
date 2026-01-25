# Check out: https://github.com/GlowCheese/deepmosa
import builtins as module_2
import enum as module_1

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
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = module_0.transform(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'abc'
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = []
    var_1 = module_0._items(var_0)
    module_0._get_keys_and_values(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = []
    var_1 = lambda i: i > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0._get_keys_and_values(var_1, var_2)

def test_case_10():
    var_0 = -5063
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = None
    var_3 = module_0.discard(var_1, var_2)
    var_4 = []
    var_5 = lambda i, v: v >= var_4
    var_6 = module_0._get_keys_and_values(var_4, var_5)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = []
    var_1 = lambda i: i > var_0
    var_2 = False
    var_3 = module_0._get_keys_and_values(var_0, var_2)
    module_0._get_keys_and_values(var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 0
    var_1 = [var_0, var_0]
    var_2 = None
    var_3 = module_0.transform(var_2, var_1)
    assert var_3 == 0
    module_0.dec(var_2)

def test_case_13():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    var_2 = ''
    var_3 = var_1(var_2)
    var_4 = 'a'
    var_5 = var_1(var_4)

def test_case_14():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    var_2 = var_1(var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 10
    var_1 = module_0.inc(var_0)
    assert var_1 == 11
    var_2 = lambda i: i > var_1
    module_0._get_keys_and_values(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 10
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.inc(var_0)
    assert var_2 == 11
    var_3 = lambda i: i > var_0
    module_0._get_keys_and_values(var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = []
    var_1 = lambda i, v: v >= var_0
    var_2 = var_1.__str__()
    module_0._get_keys_and_values(var_2, var_1)

def test_case_18():
    var_0 = []
    var_1 = lambda i, v: v >= var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = module_0.ny(var_2)
    assert var_3 is True

def test_case_19():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_2, var_4)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = []
    var_1 = lambda i: i > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0.dec(var_1)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = []
    module_0._do_to_path(var_2, var_3, var_1)
    assert var_4 == 6

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = []
    var_1 = lambda i: i > var_0
    var_2 = module_1.IntEnum
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = []
    var_1 = lambda i, v: v >= var_0
    var_2 = module_2.dict
    module_0._get_keys_and_values(var_2, var_1)