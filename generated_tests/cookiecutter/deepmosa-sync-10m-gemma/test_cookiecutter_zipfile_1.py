# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0
import cookiecutter.exceptions as module_1

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    module_0.unzip(var_0, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.unzip(var_0, var_0, var_0, password=var_0)

def test_case_3():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    with pytest.raises(module_1.InvalidZipRepository):
        module_0.unzip(var_0, var_1, no_input=var_1, password=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '4MJ\n\tk:\x0b'
    var_1 = None
    module_0.unzip(var_0, var_1)