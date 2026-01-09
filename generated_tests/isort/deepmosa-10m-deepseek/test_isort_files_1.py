# Check out: https://github.com/GlowCheese/deepmosa
import isort.files as module_0
import pytest


def test_case_0():
    var_0 = None
    var_1 = module_0.find(var_0, var_0, var_0, var_0)

def test_case_1():
    var_0 = 'm\rpHY&|dU"&'
    var_1 = module_0.find(var_0, var_0, var_0, var_0)
    with pytest.raises(AttributeError):
        var_2 = list(var_1)

def test_case_2():
    var_0 = '/some/directory'
    var_1 = [var_0]
    var_2 = {}
    var_3 = []
    var_4 = module_0.find(var_1, var_2, var_3, var_3)
    var_5 = list(var_4)

def test_case_3():
    var_0 = '/s'
    var_1 = module_0.find(var_0, var_0, var_0, var_0)
    with pytest.raises(AttributeError):
        var_2 = list(var_1)