# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.comments as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'L7$k8D{#Ph\x0c&+3mo'
    var_1 = None
    var_2 = module_0.add_to_line(var_1, removed=var_1)
    assert var_2 == ''
    var_3 = module_0.parse(var_2)
    var_4 = '0'
    var_5 = module_0.parse(var_4)
    var_6 = module_0.parse(var_0)
    module_0.parse(var_1)

def test_case_1():
    var_0 = ''
    var_1 = module_0.parse(var_0)

def test_case_2():
    var_0 = None
    var_1 = '`gvi'
    var_2 = True
    var_3 = module_0.add_to_line(var_0, var_1, var_2, var_1)
    assert var_3 == '`gvi'
    var_4 = "_I{;=Sl;hqq7+I'z/'"
    var_5 = module_0.parse(var_4)

def test_case_3():
    var_0 = None
    var_1 = module_0.add_to_line(var_0, var_0, comment_prefix=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = -1793
    var_1 = None
    module_0.add_to_line(var_0, removed=var_1)

def test_case_5():
    var_0 = None
    var_1 = False
    var_2 = '\r|\x0b-.y\n?'
    var_3 = module_0.add_to_line(var_0, removed=var_1, comment_prefix=var_2)
    assert var_3 == ''
    var_4 = 'So8Y]Lm1]'
    var_5 = [var_4]
    var_6 = module_0.add_to_line(var_0, var_0, var_0)
    var_7 = module_0.add_to_line(var_5)
    assert var_7 == ' So8Y]Lm1]'

def test_case_6():
    var_0 = 'E<d:NSp+'
    var_1 = [var_0, var_0, var_0]
    var_2 = 'O3enFSnJv9r'
    var_3 = module_0.add_to_line(var_1, var_0, comment_prefix=var_2)
    assert var_3 == 'E<d:NSp+O3enFSnJv9r E<d:NSp+'
    var_4 = 'X\t"T$Powm'
    var_5 = module_0.add_to_line(var_1)
    assert var_5 == ' E<d:NSp+'
    var_6 = None
    var_7 = module_0.add_to_line(var_6)
    assert var_7 == ''
    var_8 = module_0.parse(var_0)
    var_9 = module_0.parse(var_4)