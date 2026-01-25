# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.exceptions as module_1
import cookiecutter.zipfile as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    module_0.unzip(var_0, var_0, password=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = "SJ'gPt3c{M9#7o4\\"
    var_1 = None
    module_0.unzip(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '/tmp/tet_di'
    module_0.unzip(var_0, var_0, var_0, var_0, var_0)

def test_case_4():
    var_0 = 'http://xample.com/repo.zip'
    var_1 = '/tmp/tet_di'
    with pytest.raises(module_1.InvalidZipRepository):
        module_0.unzip(var_0, var_1, var_1, var_1, var_1)