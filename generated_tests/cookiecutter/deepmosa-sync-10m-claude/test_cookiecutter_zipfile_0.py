# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0

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
    var_0 = 'Z\tw4C;6@c2Q[kc/ \t>'
    module_0.unzip(var_0, var_0, password=var_0)

def test_case_4():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/clone'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    assert var_3 == '/tmp/unzip_base/project_name'