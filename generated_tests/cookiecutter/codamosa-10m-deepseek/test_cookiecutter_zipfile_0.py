# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.exceptions as module_1
import cookiecutter.zipfile as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'https://githu*.com/audreyr/cookiecutter-pypackage/archive/master.zip'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0, password=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '|Fv\r'
    var_1 = False
    module_0.unzip(var_0, var_1)

def test_case_3():
    var_0 = 'https://exampe.com/valid.zip'
    var_1 = "/tmp/cookiecutt>3e'"
    var_2 = None
    with pytest.raises(module_1.InvalidZipRepository):
        module_0.unzip(var_0, var_0, var_1, var_0, var_2)