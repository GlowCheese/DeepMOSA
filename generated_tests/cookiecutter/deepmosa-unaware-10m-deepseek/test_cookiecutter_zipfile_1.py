# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'oH)e'
    var_1 = None
    module_0.unzip(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '/l'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '|^/B7-Ik[R^R+g!'
    var_1 = '*:[^55YU!]Ri`\thSL/'
    var_2 = True
    module_0.unzip(var_1, var_1, no_input=var_2, password=var_0)