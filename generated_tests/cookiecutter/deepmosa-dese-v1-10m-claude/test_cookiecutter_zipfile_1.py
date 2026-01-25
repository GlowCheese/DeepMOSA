# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = True
    module_0.unzip(var_0, var_1)

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
    var_0 = 'A\x0bE?6e{/WS\n$]'
    module_0.unzip(var_0, var_0, no_input=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'project_name/'
    var_1 = True
    module_0.unzip(var_0, var_1, no_input=var_1)