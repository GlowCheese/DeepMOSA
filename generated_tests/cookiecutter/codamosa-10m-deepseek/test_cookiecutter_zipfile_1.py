# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.exceptions as module_1
import cookiecutter.zipfile as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '/tmp/cookiecutter'
    var_1 = True
    module_0.unzip(var_0, var_1, var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0, no_input=var_0, password=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = "zw]s\x0cB'>?)PH7#w40"
    var_1 = None
    module_0.unzip(var_0, var_1)

def test_case_3():
    var_0 = 'https://example.com/test.zip'
    var_1 = True
    var_2 = '/tmp/cookiecutter'
    var_3 = None
    with pytest.raises(module_1.InvalidZipRepository):
        module_0.unzip(var_0, var_1, var_2, var_1, var_3)