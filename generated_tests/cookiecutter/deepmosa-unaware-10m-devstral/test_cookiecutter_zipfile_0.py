# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0
import certifi.core as module_1
import cookiecutter.exceptions as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'w"bu/f\x0bEG- N\t\t:S'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = 'aQ^K4(e8oQ_x\n\x0c2uz&'
    module_0.unzip(var_1, var_0, password=var_0)

def test_case_3():
    var_0 = 'w"bu/\x0bG- N\t\tuS'
    var_1 = module_1.where()
    assert var_1 == '/workspace/.project-deps/cookiecutter/site-packages/certifi/cacert.pem'
    var_2 = False
    with pytest.raises(module_2.InvalidZipRepository):
        module_0.unzip(var_1, var_2, password=var_0)

def test_case_4():
    var_0 = 'https://example.com/test.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    with pytest.raises(module_2.InvalidZipRepository):
        module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    assert var_4 == '/tmp/test/test_dir'