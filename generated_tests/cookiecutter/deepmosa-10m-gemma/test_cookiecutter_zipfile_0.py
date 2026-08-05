# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0
import cookiecutter.exceptions as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '7\tE76Z*&k7$an8/8'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0, no_input=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ''
    module_0.unzip(var_0, var_0)

def test_case_3():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    with pytest.raises(module_1.InvalidZipRepository):
        module_0.unzip(var_0, var_1, var_0, var_1)