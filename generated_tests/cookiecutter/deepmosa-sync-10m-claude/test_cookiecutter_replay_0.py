# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.replay as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.load(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '*N7h\r7X'
    module_0.load(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'nonexistent.json'
    module_0.load(var_0, var_0)

def test_case_3():
    var_0 = ''
    with pytest.raises(ValueError):
        module_0.dump(var_0, var_0, var_0)