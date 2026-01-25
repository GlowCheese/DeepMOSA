# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.files as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.find(var_0, var_0, var_0, var_0)

def test_case_1():
    var_0 = b"\x02s'"
    var_1 = module_0.find(var_0, var_0, var_0, var_0)
    with pytest.raises(AttributeError):
        var_2 = list(var_1)

def test_case_2():
    var_0 = b'Av_\xde'
    var_1 = module_0.find(var_0, var_0, var_0, var_0)
    with pytest.raises(AttributeError):
        var_2 = list(var_1)

def test_case_3():
    var_0 = b'\x00'
    var_1 = module_0.find(var_0, var_0, var_0, var_0)
    var_2 = list(var_1)