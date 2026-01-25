# Check out: https://github.com/GlowCheese/deepmosa
import certifi.core as module_1
import cookiecutter.exceptions as module_2
import cookiecutter.zipfile as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'Z\t\x0b4C;imc1GQ8[kc/\\>'
    module_0.unzip(var_0, var_0, password=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ''
    module_0.unzip(var_0, var_0, password=var_0)

def test_case_3():
    var_0 = module_1.where()
    assert var_0 == '/workspace/.project-deps/cookiecutter/site-packages/certifi/cacert.pem'
    var_1 = None
    with pytest.raises(module_2.InvalidZipRepository):
        module_0.unzip(var_0, var_1, no_input=var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'qz/'
    module_0.unzip(var_0, var_0, no_input=var_0)