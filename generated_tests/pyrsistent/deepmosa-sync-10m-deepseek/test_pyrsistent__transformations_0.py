# Check out: https://github.com/GlowCheese/deepmosa
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
    var_1 = lambda k, v: var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = module_0.transform(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = []
    var_1 = lambda k, v: var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = b'\x9f\xbe\xef\xe5\xca\xce\x01V\xb0]G('
    module_0.transform(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = []
    var_1 = lambda k, v: var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0._get_keys_and_values(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = []
    var_1 = lambda k, v: var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0._get_keys_and_values(var_1, var_2)

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
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_12():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0._get_keys_and_values(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = True
    var_1 = lambda k, v: var_0
    module_0._get_keys_and_values(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = lambda k: k == var_4
    module_0._get_keys_and_values(var_3, var_5)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = []
    var_1 = lambda k: var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0.discard(var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = []
    var_1 = lambda k, v: var_0
    var_2 = b'\x9f\xbe\xef\xe5\xca\xce\x01V\xb0]G('
    module_0._get_keys_and_values(var_2, var_1)

def test_case_17():
    var_0 = lambda x, y: x + y
    var_1 = module_0._get_arity(var_0)
    assert var_1 == 2

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = False
    var_1 = lambda k: var_0
    var_2 = module_0.ny(var_1)
    assert var_2 is True
    module_0._get_keys_and_values(var_2, var_1)

def test_case_19():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x, y, z: var_3
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_2, var_4)

def test_case_20():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = lambda x: sum(x)
    var_6 = module_0._do_to_path(var_3, var_4, var_5)
    assert var_6 == 6

def test_case_21():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0._do_to_path(var_3, var_4, var_8)
    var_10 = bool(var_9 == [4, 5, 6])
    assert var_10 is True

def test_case_22():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123_extra'
    var_3 = var_1(var_2)
    assert var_3 is None

def test_case_23():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = var_1(var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = var_1(var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True