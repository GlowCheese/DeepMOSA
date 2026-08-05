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

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0.transform(var_0, var_1)
    module_0.transform(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = "~>,m[5}~*Q%'JcIM"
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = ''
    var_1 = lambda k: k == var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = "'\x0cTvG"
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = lambda k, v: v > var_0
    var_3 = module_0._get_keys_and_values(var_1, var_2)
    var_4 = module_0.discard(var_1, var_2)
    module_0._get_keys_and_values(var_0, var_2)

def test_case_10():
    var_0 = "'\x0cTvG"
    var_1 = module_0.rex(var_0)
    var_2 = module_1._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = module_0._get_keys_and_values(var_2, var_0)
    var_4 = module_0._get_keys_and_values(var_3, var_1)

def test_case_11():
    var_0 = []
    var_1 = module_0._items(var_0)
    var_2 = list(var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    module_0._get_keys_and_values(var_1, var_1)

def test_case_13():
    var_0 = ']w'
    var_1 = module_0.rex(var_0)
    var_2 = module_0._get_keys_and_values(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '*My?AG.cS)af0\r'
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = lambda k, v: v > var_0
    module_0._get_keys_and_values(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'H'
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = lambda k, v: v > var_1
    module_0._get_keys_and_values(var_2, var_2)

def test_case_16():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_2, var_4)

def test_case_17():
    var_0 = '*My?AG.cS)af0\r'
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = module_1._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = lambda k, v: v > var_2
    var_4 = module_0._get_keys_and_values(var_2, var_3)

def test_case_18():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = module_0._get_keys_and_values(var_0, var_1)

def test_case_19():
    var_0 = "FE,%<]\t2t7.'H3@8"
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = module_1._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = lambda k, v: v > var_1
    var_4 = module_0.ny(var_3)
    assert var_4 is True
    var_5 = var_1.__str__()
    assert var_5 == 'True'
    var_6 = module_2.float
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_1, var_6)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = lambda x: x[var_0] + var_1
    module_0._do_to_path(var_2, var_3, var_4)
    assert var_5 == 2

def test_case_21():
    var_0 = '\x0cH'
    var_1 = module_0.rex(var_0)
    var_2 = lambda k, v: v > var_1
    var_3 = module_0._get_keys_and_values(var_0, var_1)
    var_4 = module_1.EnumMeta
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_3, var_4)