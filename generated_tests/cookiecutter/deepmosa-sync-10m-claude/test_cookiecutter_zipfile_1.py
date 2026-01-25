# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.zipfile as module_0
import certifi.core as module_1
import cookiecutter.exceptions as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'oH)e'
    var_1 = None
    module_0.unzip(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'hE]/WS\n$]'
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = True
    var_1 = 'Gs`\\\t=IyY Q/'
    var_2 = True
    var_3 = None
    module_0.unzip(var_1, var_0, no_input=var_2, password=var_3)

def test_case_5():
    var_0 = None
    var_1 = module_1.where()
    assert var_1 == '/workspace/.project-deps/cookiecutter/site-packages/certifi/cacert.pem'
    var_2 = False
    with pytest.raises(module_2.InvalidZipRepository):
        module_0.unzip(var_1, var_2, password=var_0)