# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_1
import inspect as module_3
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
    var_0 = -13
    var_1 = var_0.__str__()
    assert var_1 == '-13'
    module_0.transform(var_1, var_1)

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
    var_1 = lambda k, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
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

def test_case_9():
    var_0 = []
    var_1 = lambda k, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = module_0.transform(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 9
    var_1 = (var_0, var_0, var_0)
    var_2 = (var_1, var_1)
    module_0.transform(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = lambda x, y: x + y
    module_0._get_keys_and_values(var_0, var_0)

def test_case_12():
    var_0 = 0
    var_1 = (var_0, var_0)
    var_2 = module_0.transform(var_1, var_1)
    assert var_2 == 0

def test_case_13():
    var_0 = lambda x, y: x + y
    var_1 = module_0._get_arity(var_0)
    assert var_1 == 2

def test_case_14():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = lambda x: sum(x)
    var_6 = module_0._do_to_path(var_3, var_4, var_5)
    assert var_6 == 6

def test_case_15():
    var_0 = '^test.*'
    var_1 = module_0.rex(var_0)
    var_2 = module_0._get_keys_and_values(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 21
    var_1 = [var_0, var_0, var_0]
    var_2 = lambda k, v: v > var_1
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    module_0._get_keys_and_values(var_1, var_1)

def test_case_18():
    var_0 = []
    var_1 = '&lK)C'
    var_2 = module_0.ny(var_1)
    assert var_2 is True
    var_3 = module_1.Enum
    var_4 = module_0._get_keys_and_values(var_0, var_3)

def test_case_19():
    var_0 = {}
    var_1 = None
    var_2 = lambda : var_1
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = []
    var_1 = lambda k, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0.rex(var_2)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = []
    var_1 = lambda k, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = var_1.__str__()
    var_4 = module_0.ny(var_2)
    assert var_4 is True
    var_5 = module_3.Parameter
    var_6 = module_0._get_keys_and_values(var_0, var_5)
    assert f'{type(module_3.mod_dict).__module__}.{type(module_3.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_3.mod_dict) == 168
    assert module_3.k == 512
    assert module_3.v == 'ASYNC_GENERATOR'
    assert module_3.CO_OPTIMIZED == 1
    assert module_3.CO_NEWLOCALS == 2
    assert module_3.CO_VARARGS == 4
    assert module_3.CO_VARKEYWORDS == 8
    assert module_3.CO_NESTED == 16
    assert module_3.CO_GENERATOR == 32
    assert module_3.CO_NOFREE == 64
    assert module_3.CO_COROUTINE == 128
    assert module_3.CO_ITERABLE_COROUTINE == 256
    assert module_3.CO_ASYNC_GENERATOR == 512
    assert module_3.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_3.modulesbyfile == {}
    assert module_3.GEN_CREATED == 'GEN_CREATED'
    assert module_3.GEN_RUNNING == 'GEN_RUNNING'
    assert module_3.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_3.GEN_CLOSED == 'GEN_CLOSED'
    assert module_3.CORO_CREATED == 'CORO_CREATED'
    assert module_3.CORO_RUNNING == 'CORO_RUNNING'
    assert module_3.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_3.CORO_CLOSED == 'CORO_CLOSED'
    module_0.inc(var_6)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = []
    var_1 = lambda k, v: v > var_0
    var_2 = var_1.__repr__()
    var_3 = module_0.rex(var_2)
    var_4 = module_0._get_keys_and_values(var_0, var_1)
    var_5 = 578.0
    var_6 = {var_5: var_2, var_2: var_5}
    var_7 = module_0._get_keys_and_values(var_6, var_3)
    module_0._get_keys_and_values(var_1, var_2)