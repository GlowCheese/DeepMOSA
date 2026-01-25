# Check out: https://github.com/GlowCheese/deepmosa
import cookiecutter.exceptions as module_1
import cookiecutter.zipfile as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    module_0.unzip(var_0, var_0, no_input=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ''
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'E/P('
    module_0.unzip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'VB2tVmr0:\\i57P)>=?'
    var_1 = 'E/'
    var_2 = True
    module_0.unzip(var_1, var_2, no_input=var_2, password=var_0)

def test_case_5():
    var_0 = 'http://eample.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/clone_dir'
    var_3 = True
    with pytest.raises(module_1.InvalidZipRepository):
        module_0.unzip(var_0, var_1, var_2, var_3, var_0)