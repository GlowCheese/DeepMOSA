# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0
import certifi.core as module_1
import cookiecutter.exceptions as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '776-Z*&k7$an8/8'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0, no_input=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ''
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ':N;KV/'
    module_0.unzip(var_0, var_0, var_0, var_0)

def test_case_4():
    var_0 = module_1.where()
    assert var_0 == '/workspace/.project-deps/cookiecutter/site-packages/certifi/cacert.pem'
    var_1 = None
    with pytest.raises(module_2.InvalidZipRepository):
        module_0.unzip(var_0, var_1, password=var_1)