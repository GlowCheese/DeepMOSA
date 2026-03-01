# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0
import cookiecutter.exceptions as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0, password=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = 'aQ^K4(e8oQ_x\n\x0c2uz&'
    module_0.unzip(var_1, var_0, password=var_0)

def test_case_3():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp'
    with pytest.raises(module_1.InvalidZipRepository):
        module_0.unzip(var_0, var_1, var_2, var_1)