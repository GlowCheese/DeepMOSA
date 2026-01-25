# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._transformations as module_0
import re as module_1
import enum as module_2

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.discard(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0.transform(var_0, var_1)
    var_3 = 'LQ'
    var_4 = module_0.rex(var_3)
    var_5 = [var_4, var_4, var_4, var_3]
    module_0._do_to_path(var_3, var_5, var_4)

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
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = [var_1, var_0]
    module_0._do_to_path(var_0, var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_1.purge()
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    var_1 = var_0.__repr__()
    assert var_1 == 'None'
    module_0.transform(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.discard(var_0, var_1)
    var_3 = lambda k: k in var_0
    var_4 = [var_3]
    module_0._do_to_path(var_0, var_4, var_3)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = var_0.__repr__()
    assert var_1 == '{}'
    module_0.transform(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'C'
    var_1 = module_0.rex(var_0)
    var_2 = [var_1, var_0]
    module_0._do_to_path(var_0, var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'q,j6~\\U),c_'
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = lambda k, v: v % var_0 == var_0
    var_3 = [var_2]
    module_0._do_to_path(var_0, var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = lambda k, v: v % var_1 == var_1
    var_3 = [var_2]
    module_0._do_to_path(var_1, var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = lambda k, v, x: var_5
    var_7 = [var_6]
    var_8 = lambda x: x
    module_0._do_to_path(var_4, var_7, var_8)

def test_case_15():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'c'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = None
    var_5 = module_0.ny(var_4)
    assert var_5 is True
    var_6 = {var_0: var_1, var_0: var_2, var_0: var_3}
    var_7 = [var_0, var_0, var_0]
    var_8 = lambda k: k in var_7
    var_9 = []
    var_10 = lambda x: x * var_8
    module_0._do_to_path(var_6, var_9, var_10)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = ''
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = lambda k, v: v % var_0 == var_0
    var_3 = [var_2]
    module_0._do_to_path(var_0, var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = '^test_\\w+'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = var_1(var_2)
    var_3.replace()