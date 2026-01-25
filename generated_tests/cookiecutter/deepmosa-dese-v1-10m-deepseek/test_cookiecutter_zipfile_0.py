# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.zipfile as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'http://eample.com/epo.zp'
    module_0.unzip(var_0, var_0)

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
    var_0 = 'https://example.com/repo.zip'
    var_1 = '/tmp/test_dir'
    var_2 = True
    module_0.unzip(var_0, var_2, var_1, var_2)
    assert var_3 is True