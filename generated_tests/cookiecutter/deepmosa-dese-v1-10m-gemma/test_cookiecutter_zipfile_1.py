# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0
import cookiecutter.exceptions as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'zr$/zo'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.unzip(var_0, var_0, var_0, password=var_0)

def test_case_3():
    var_0 = 'pject/'
    var_1 = 'http://example.com/repo.zip'
    var_2 = True
    with pytest.raises(module_1.InvalidZipRepository):
        module_0.unzip(var_1, var_2, no_input=var_2, password=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '4MJ\n\tk:\x0b'
    var_1 = None
    module_0.unzip(var_0, var_1)