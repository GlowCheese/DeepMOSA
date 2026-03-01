# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0
import cookiecutter.exceptions as module_1

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
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    module_0.unzip(var_0, var_1)

def test_case_4():
    var_0 = 'https://eample.com/relo.zip'
    var_1 = '/tmp'
    with pytest.raises(module_1.InvalidZipRepository):
        module_0.unzip(var_0, var_0, var_1, var_0)