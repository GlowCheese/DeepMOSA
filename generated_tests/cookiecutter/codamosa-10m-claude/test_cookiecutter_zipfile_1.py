# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'Y'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0, no_input=var_0, password=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ''
    module_0.unzip(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'Ee _N])x36/:>q &T_)3'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'Ee _N])x36/:>q &T_)3'
    var_1 = ')/'
    module_0.unzip(var_1, var_0, no_input=var_0)