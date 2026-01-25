# Check out: https://github.com/GlowCheese/deepmosa
import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = '2]UN<'
    var_1 = module_0.assignments(var_0)

def test_case_1():
    var_0 = ''
    var_1 = module_0.assignments(var_0)

def test_case_2():
    var_0 = 'list'
    var_1 = module_0.assignment(var_0, var_0, var_0, var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.register_type(var_0, var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.ISortPrettyPrinter(var_0)

def test_case_5():
    var_0 = "\r\x0cbg;T>SPSZN04'"
    var_1 = module_0.assignments(var_0)

def test_case_6():
    var_0 = 'lis4'
    var_1 = module_0.assignment(var_0, var_0, var_0, var_0)

def test_case_7():
    var_0 = 'x = 5\ny = 3\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    var_4 = bool('x = 5' in var_3)
    assert var_4 is True
    var_5 = bool('y = 3' in var_3)
    assert var_5 is True

def test_case_8():
    var_0 = "x = {'b': 2, 'a': 1}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = {}
    var_4 = module_1.Config(**var_3)
    var_5 = module_0.assignment(var_0, var_1, var_2, var_4)
    var_6 = bool('x = ' in var_5)
    assert var_6 is True
    var_7 = bool('{' in var_5)
    assert var_7 is True

def test_case_9():
    var_0 = 'x = [3, 1, 2]  \n'
    var_1 = 'list'
    var_2 = {}
    var_3 = module_1.Config(**var_2)
    var_4 = module_0.assignment(var_0, var_1, var_0, var_3)

def test_case_10():
    var_0 = 'z = {3, 1, 2}'
    var_1 = 'set'
    var_2 = module_0.assignment(var_0, var_1, var_0)
    var_3 = bool('z = ' in var_2)
    assert var_3 is True

def test_case_11():
    var_0 = 'x = (3, 1, 2)'
    var_1 = 'tuple'
    var_2 = '.py'
    var_3 = {}
    var_4 = module_1.Config(**var_3)
    var_5 = module_0.assignment(var_0, var_1, var_2, var_4)
    var_6 = bool('x = ' in var_5)
    assert var_6 is True
    var_7 = bool('(' in var_5)
    assert var_7 is True

def test_case_12():
    var_0 = 'x = invalid_syntax_here'
    var_1 = 'list'
    var_2 = module_0.assignment(var_0, var_1, var_1)

def test_case_13():
    var_0 = 'x = [1, 2, 3]'
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = {}
    var_4 = module_1.Config(**var_3)
    var_5 = module_0.assignment(var_0, var_1, var_2, var_4)

def test_case_14():
    var_0 = {}
    var_1 = module_1.Config(**var_0)
    var_2 = module_0.ISortPrettyPrinter(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5, var_4, var_3]
    var_7 = module_0._unique_list(var_6, var_2)
    assert var_7 == '[1, 2, 3]'
    var_8 = []
    var_9 = module_0._unique_list(var_8, var_2)
    assert var_9 == '[]'
    var_10 = [var_4]
    var_11 = module_0._unique_list(var_10, var_2)
    assert var_11 == '[1]'
    var_12 = 5
    var_13 = [var_12, var_12, var_12]
    var_14 = module_0._unique_list(var_13, var_2)
    assert var_14 == '[5]'
    var_15 = 'c'
    var_16 = 'a'
    var_17 = 'b'
    var_18 = [var_15, var_16, var_17, var_16]
    var_19 = module_0._unique_list(var_18, var_2)
    var_20 = bool("'a'" in var_19 and "'b'" in var_19 and ("'c'" in var_19))
    assert var_20 is True
    var_21 = [var_4, var_5, var_3, var_5, var_4]
    var_22 = module_0._unique_list(var_21, var_2)
    assert var_22 == '[1, 2, 3]'

def test_case_15():
    var_0 = {}
    var_1 = module_1.Config(**var_0)
    var_2 = module_0.ISortPrettyPrinter(var_1)
    var_3 = ()
    var_4 = module_0._unique_tuple(var_3, var_2)
    assert var_4 == '()'