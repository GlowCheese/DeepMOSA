# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.get_in(var_0, var_0, var_0)

def test_case_1():
    var_0 = b'\x89\x80\x9c:\xf5\xfd'
    with pytest.raises(IndexError):
        module_0.get_in(var_0, var_0, var_0, var_0)