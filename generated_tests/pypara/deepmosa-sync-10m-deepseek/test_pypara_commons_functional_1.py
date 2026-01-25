# Check out: https://github.com/GlowCheese/deepmosa
import pypara.commons.functional as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.identity(var_0)

def test_case_1():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.chunk(var_1, var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [[1]])
    assert var_4 is True

def test_case_2():
    var_0 = []
    var_1 = 1
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)