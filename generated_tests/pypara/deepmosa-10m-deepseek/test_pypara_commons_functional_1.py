# Check out: https://github.com/GlowCheese/deepmosa
import pypara.commons.functional as module_0


def test_case_0():
    var_0 = None
    var_1 = module_0.identity(var_0)

def test_case_1():
    var_0 = 1
    var_1 = 3
    var_2 = 4
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = module_0.chunk(var_3, var_0)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [[1], [2], [3], [4]])