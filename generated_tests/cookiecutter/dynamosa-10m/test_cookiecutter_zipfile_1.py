# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.zipfile as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '^H#L?uapfYkD]'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.unzip(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ''
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '{Zz<"6P;/5Ua\r-aZ'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'Rqt/'
    module_0.unzip(var_0, var_0, no_input=var_0)