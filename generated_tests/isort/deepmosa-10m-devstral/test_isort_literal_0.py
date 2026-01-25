# Check out: https://github.com/GlowCheese/deepmosa
import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = '2]UN<'
    var_1 = module_0.assignments(var_0)

def test_case_1():
    var_0 = None
    var_1 = ''
    var_2 = module_0.assignments(var_1)
    var_3 = module_0.ISortPrettyPrinter(var_0)

def test_case_2():
    var_0 = None
    var_1 = ')qUsFyq-=\nOjVt~}TLi'
    var_2 = module_0.assignment(var_0, var_0, var_1, var_0)

def test_case_3():
    pass

def test_case_4():
    var_0 = None
    var_1 = module_0.ISortPrettyPrinter(var_0)

def test_case_5():
    var_0 = "\r\x0cbg;T>SPSZN04'"
    var_1 = module_0.assignments(var_0)

def test_case_6():
    var_0 = {}
    var_1 = module_1.Config(**var_0)
    var_2 = module_0.ISortPrettyPrinter(var_1)
    var_3 = []
    var_4 = module_0._list(var_3, var_2)
    assert var_4 == '[]'

def test_case_7():
    var_0 = {}
    var_1 = module_1.Config(**var_0)
    var_2 = module_0.ISortPrettyPrinter(var_1)
    var_3 = set()
    var_4 = module_0._set(var_3, var_2)
    assert var_4 == '{}'

def test_case_8():
    var_0 = {}
    var_1 = module_1.Config(**var_0)
    var_2 = module_0.ISortPrettyPrinter(var_1)
    var_3 = None
    var_4 = module_0._tuple(var_3, var_2)