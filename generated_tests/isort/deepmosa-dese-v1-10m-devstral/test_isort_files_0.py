# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.settings as module_0
import isort.files as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.Config(**var_0)

def test_case_1():
    var_0 = '('
    var_1 = module_1.find(var_0, var_0, var_0, var_0)
    with pytest.raises(AttributeError):
        var_2 = list(var_1)

def test_case_2():
    var_0 = '('
    var_1 = [var_0, var_0, var_0]
    var_2 = []
    var_3 = module_1.find(var_1, var_2, var_2, var_2)
    var_4 = list(var_3)

def test_case_3():
    var_0 = '/'
    var_1 = module_1.find(var_0, var_0, var_0, var_0)
    with pytest.raises(AttributeError):
        var_2 = list(var_1)