# Check out: https://github.com/GlowCheese/deepmosa
import builtins as module_1

import isort.sorting as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.sort(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'y'
    module_0.module_key(var_0, var_0, ignore_case=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ''
    module_0.module_key(var_0, var_0, section_name=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.naturally(var_0)

def test_case_4():
    var_0 = 'G6RO=LM'
    var_1 = module_0.naturally(var_0)
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'Ny\nD'
    module_0.module_key(var_0, var_0, var_0, straight_import=var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_1.str
    module_0.naturally(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '+2IAz2P8 xi$T6'
    module_0.naturally(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '.'
    module_0.module_key(var_0, var_0, section_name=var_0)