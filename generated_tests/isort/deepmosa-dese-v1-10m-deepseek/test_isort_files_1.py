# Check out: https://github.com/GlowCheese/deepmosa
import isort.files as module_0
import pytest


def test_case_0():
    var_0 = None
    var_1 = module_0.find(var_0, var_0, var_0, var_0)

def test_case_1():
    var_0 = 'y'
    var_1 = module_0.find(var_0, var_0, var_0, var_0)
    with pytest.raises(AttributeError):
        var_2 = list(var_1)

def test_case_2():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = var_1.__repr__()
    assert var_2 == "{'.py'}"
    var_3 = []
    var_4 = [var_0]
    var_5 = module_0.find(var_4, var_2, var_2, var_3)
    var_6 = list(var_5)

def test_case_3():
    var_0 = 'LB9)/\tQF'
    var_1 = [var_0, var_0]
    var_2 = module_0.find(var_0, var_0, var_0, var_1)
    with pytest.raises(AttributeError):
        var_3 = list(var_2)