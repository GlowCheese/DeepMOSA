# Check out: https://github.com/GlowCheese/deepmosa
import isort.literal as module_0


def test_case_0():
    var_0 = ']\\FZ!R\x0c&A'
    var_1 = module_0.assignments(var_0)

def test_case_1():
    var_0 = '\x0b'
    var_1 = module_0.assignments(var_0)
    var_2 = None
    var_3 = module_0.assignment(var_2, var_1, var_1)

def test_case_2():
    var_0 = 'a = 1b = 2c = 3'
    var_1 = module_0.assignments(var_0)

def test_case_3():
    var_0 = None
    var_1 = ')qUsFyq-=\nOjVt~}TLi'
    var_2 = module_0.assignment(var_0, var_0, var_1, var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.ISortPrettyPrinter(var_0)

def test_case_5():
    var_0 = 'my_u^ple =( 1, )'
    var_1 = 'tuple'
    var_2 = module_0.assignment(var_0, var_1, var_0)

def test_case_6():
    var_0 = 'uU<=E2_'
    var_1 = 'tuple'
    var_2 = module_0.assignment(var_0, var_1, var_0)

def test_case_7():
    var_0 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_1 = 'dict'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    var_4 = 'my_list = [3, 1, 2]'
    var_5 = 'list'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    var_7 = 'my_set = {3, 1, 2}'
    var_8 = 'my_set = {1, 2, 3}'
    var_9 = 'set'
    var_10 = module_0.assignment(var_7, var_9, var_2)
    var_11 = 'tuple'
    var_12 = module_0.assignment(var_8, var_11, var_2)

def test_case_8():
    var_0 = 'a = 1\nb = 2'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    var_4 = 'test_assignment passed'
    var_5 = print(var_4)

def test_case_9():
    var_0 = "my_dict = {'a': 1, 'b': 2, 'P': 3}"
    var_1 = 'dict'
    var_2 = module_0.assignment(var_0, var_1, var_0)
    var_3 = 'my_tuple = (1, 2, 3)'
    var_4 = module_0.assignment(var_2, var_2, var_3)

def test_case_10():
    var_0 = 'dict'
    var_1 = 'my_list = [3, 1, 2]'
    var_2 = 'list'
    var_3 = module_0.assignment(var_1, var_2, var_2)
    var_4 = 'set'
    var_5 = module_0.assignment(var_0, var_4, var_2)

def test_case_11():
    var_0 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_1 = 'dict'
    var_2 = module_0.assignment(var_0, var_1, var_1)
    var_3 = 'my_list = [3, 1, 2]'
    var_4 = 'my_list = [1, 2, 3]'
    var_5 = 'list'
    var_6 = module_0.assignment(var_3, var_5, var_4)
    var_7 = 'my_tuple = (3, 1, 2)'
    var_8 = 'tuple'
    var_9 = module_0.assignment(var_7, var_8, var_3)
    var_10 = 'my_list = [3, 1, 2, 1, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_11)
    var_13 = 'All tests passed!'
    var_14 = print(var_13)

def test_case_12():
    var_0 = 'my_list = [3, 1, 2]'
    var_1 = 'list'
    var_2 = module_0.assignment(var_0, var_1, var_1)
    var_3 = 'my_set = {1, 2, 3}'
    var_4 = 'set'
    var_5 = module_0.assignment(var_3, var_4, var_1)
    var_6 = 'my_tuple = (3, 1, 2)'
    var_7 = 'tuple'
    var_8 = module_0.assignment(var_6, var_7, var_5)
    var_9 = 'a = 1b = 2c = 3'
    var_10 = print(var_9)

def test_case_13():
    var_0 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_1 = 'dict'
    var_2 = module_0.assignment(var_0, var_1, var_1)
    var_3 = 'my_list = [3, 1, 2]'
    var_4 = 'my_list = [1, 2, 3]'
    var_5 = 'list'
    var_6 = module_0.assignment(var_3, var_5, var_4)
    var_7 = 'my_tuple = (3, 1, 2)'
    var_8 = 'tuple'
    var_9 = module_0.assignment(var_7, var_8, var_3)
    var_10 = 'my_list = [3, 1, 2, 1, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_11)
    var_13 = 'my_tuple = (3, 1, 2, 1, 2)'
    var_14 = 'unique-tuple'
    var_15 = module_0.assignment(var_13, var_14, var_9)
    var_16 = 'All tests passed!'
    var_17 = print(var_16)