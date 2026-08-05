# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0
import cookiecutter.exceptions as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = "/J\\lfj'rm*B.WolHA#S"
    var_1 = True
    module_0.unzip(var_0, var_1, password=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0)

def test_case_2():
    var_0 = 'https://example.com/corrupt.zip'
    var_1 = True
    with pytest.raises(module_1.InvalidZipRepository):
        module_0.unzip(var_0, var_1, no_input=var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'M.%'
    var_1 = False
    module_0.unzip(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.unzip(var_0, var_0, var_0, password=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'https://example.com/corrupt.zip'
    var_1 = None
    module_0.unzip(var_0, var_0, password=var_1)