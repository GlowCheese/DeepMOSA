####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda self, _, value: str(value)
    var_6 = [var_1, var_2, var_3]
    var_7 = [var_1, var_2]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = {var_1, var_2}
    var_11 = frozenset(var_10)
    var_12 = {var_3, var_8}
    var_13 = frozenset(var_12)
    var_14 = {var_11, var_13}
    var_15 = set()



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0.pset(var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_6, var_7]
    var_9 = 1
    var_10 = 2
    var_11 = [var_9, var_10]
    var_12 = [var_10, var_6]
    var_13 = lambda x: (x >= 0, 'Negative value')
    var_14 = [var_10, var_11, var_2]
    var_15 = -1
    var_16 = 2
    var_17 = [var_15, var_16]
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = {var_18: var_17}
    var_21 = [var_19, var_20]



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 1
    var_5 = 0



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda self, _, x: x * 2
    var_6 = 4
    var_7 = 5
    var_8 = [var_6, var_7]
    var_9 = [var_6, var_7]
    var_10 = lambda self, _, x: str(x)
    var_11 = 6
    var_12 = 7
    var_13 = 8
    var_14 = [var_11, var_12, var_13]
    var_15 = set()



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = module_0.pset(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 'invalid'
    var_9 = [var_6, var_7, var_8]
    var_10 = lambda x: (x >= 0, 'Negative value')
    var_11 = [var_8, var_9, var_2]
    var_12 = 1
    var_13 = -2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = -1
    var_3 = 15
    var_4 = -1



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = [var_5, var_6]
    var_8 = lambda self, _, value: str(value)
    var_9 = [var_1, var_2, var_3]
    var_10 = set()



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1
    var_4 = -1



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = set()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = lambda self, _, value: value * 2
    var_7 = [var_2, var_3, var_4]
    var_8 = [var_2, var_3]
    var_9 = 4
    var_10 = [var_4, var_9]
    var_11 = [var_2, var_3]
    var_12 = set(var_11)
    var_13 = [var_4, var_9]
    var_14 = set(var_13)
    var_15 = [var_12, var_14]



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Value must be positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda self, _, value: str(value)
    var_6 = [var_1, var_2, var_3]



# Parsed testcases at query #12
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'int'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_3[var_0][var_6]

def test_case_0():
    var_0 = True
    var_1 = 'test2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 1
    var_7 = var_3[var_0][var_6]
    var_8 = {}
    var_9 = var_8[var_0]
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'not_callable'
    var_12 = {}
    var_13 = 'invariants'
    var_14 = 'invariant'

def test_case_0():
    var_0 = True
    var_1 = 'test1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'test2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = var_7[var_0]
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_7[var_0][var_6]

def test_case_0():
    var_0 = True
    var_1 = 'test1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'test2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = var_7[var_0]
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_7[var_0][var_6]



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 6
    var_5 = -1



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'ok')
    var_1 = '_invariant'
    var_2 = 0
    var_3 = None
    var_4 = lambda self: (True, 'ok')
    var_5 = lambda self: (False, 'error')
    var_6 = 1
    var_7 = 'not callable'
    var_8 = lambda self: [(True, 'ok1'), (False, 'error1')]
    var_9 = lambda self: (True, 'ok')
    var_10 = lambda self: (True, 'ok2')
    var_11 = lambda self: (False, 'error2')
    var_12 = 2



# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = set()
    var_2 = 1
    var_3 = [var_2]
    var_4 = 2
    var_5 = 3
    var_6 = [var_2, var_4, var_5]
    var_7 = lambda self, _, value: value * 2
    var_8 = [var_2, var_4, var_5]
    var_9 = [var_2, var_4]
    var_10 = 4
    var_11 = [var_5, var_10]
    var_12 = [var_2, var_4]
    var_13 = set(var_12)
    var_14 = [var_5, var_10]
    var_15 = set(var_14)
    var_16 = [var_13, var_15]



# Parsed testcases at query #19
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'int'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 'str'
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = 123
    var_7 = module_0.maybe_parse_user_type(var_6)
    var_8 = None
    var_9 = module_0.maybe_parse_user_type(var_8)
    var_10 = lambda x: x
    var_11 = module_0.maybe_parse_user_type(var_10)



# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------


import pyrsistent._checked_types as module_0
import builtins as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'int'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = module_1.object()
    var_7 = module_0.maybe_parse_user_type(var_6)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1



# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'Test passed'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_3[var_0][var_6]
    var_8 = callable(var_7)

def test_case_0():
    var_0 = True
    var_1 = 'Another test passed'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_3[var_0]
    var_7 = 'not a function'
    var_8 = {}
    var_9 = 'invariants'
    var_10 = 'invariant'

def test_case_0():
    var_0 = True
    var_1 = 'Test 1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'Test 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = var_7[var_0]
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_7[var_0][var_6]
    var_11 = None

def test_case_0():
    var_0 = True
    var_1 = 'Test 1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'Test 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = var_7[var_0]
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_7[var_0][var_6]
    var_11 = None



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 10



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 'Value must be positive'

def test_case_0():
    var_0 = 100
    var_1 = 'Value must be less than 100'

def test_case_0():
    var_0 = 100
    var_1 = 'Value must be less than 100'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 10



# Parsed testcases at query #3
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 'int'
    var_5 = [var_4, var_0]
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = 123
    var_8 = module_0.maybe_parse_user_type(var_7)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = '__main__.CustomClass'
    var_3 = module_0.get_type(var_2)
    var_4 = 'enum.Enum'
    var_5 = module_0.get_type(var_4)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'str_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = []
    var_7 = module_0.maybe_parse_user_type(var_6)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 2
    var_11 = {var_9: var_10}
    var_12 = lambda k, v: (k < v, 'Key must be less than value')
    var_13 = 3
    var_14 = 4
    var_15 = {var_11: var_1, var_13: var_14}
    var_16 = 1
    var_17 = 0
    var_18 = {var_16: var_17}
    var_19 = [var_18, var_1, var_13]
    var_20 = [var_18, var_1, var_13]
    var_21 = [var_18, var_1, var_13]
    var_22 = {var_18: var_21}
    var_23 = [var_18, var_1, var_13]



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 10



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x'
    var_6 = 'y'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #9
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'pytest.CustomClass'
    var_3 = module_0.get_type(var_2)
    var_4 = 'invalid.type'
    var_5 = module_0.get_type(var_4)



# Parsed testcases at query #10
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = '__main__.TestClass'
    var_3 = module_0.get_type(var_2)
    var_4 = 'collections.OrderedDict'
    var_5 = module_0.get_type(var_4)
    var_6 = 'nonexistent.module.Class'
    var_7 = module_0.get_type(var_6)
    var_8 = 'invalid.type.string'
    var_9 = module_0.get_type(var_8)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 15
    var_6 = -3
    var_7 = 10



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 0
    var_6 = 1



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = lambda k, v: (len(v) > 0, 'Empty value')
    var_1 = '_checked_key_types'
    var_2 = '_checked_value_types'
    var_3 = '_checked_invariants'
    var_4 = '__serializer__'
    var_5 = '__slots__'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'collections.abc.Iterable'
    var_3 = module_0.get_type(var_2)
    var_4 = 'nonexistent.module.Class'
    var_5 = module_0.get_type(var_4)
    var_6 = 'builtins.NonexistentClass'
    var_7 = module_0.get_type(var_6)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = -1



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'test')
    var_1 = {}
    var_2 = '__stored_invariants__'
    var_3 = '__invariants__'
    var_4 = var_1[var_2]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_1[var_2][var_6]
    var_8 = callable(var_7)
    var_9 = lambda self: (True, 'test2')
    var_10 = {}
    var_11 = var_10[var_2]
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = lambda self: (True, 'base')
    var_14 = {}
    var_15 = var_14[var_2]
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'not callable'
    var_18 = {}
    var_19 = '__stored_invariants__'
    var_20 = '__invariants__'
    var_21 = {}
    var_22 = var_21[var_19][var_6]
    var_23 = None



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)



# Parsed testcases at query #18
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'test1'
    var_3 = 'test2'
    var_4 = [var_2, var_3]
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 123
    var_9 = module_0.maybe_parse_user_type(var_8)
    var_10 = None
    var_11 = module_0.maybe_parse_user_type(var_10)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = 10
    var_7 = 'invalid'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 1
    var_11 = 123
    var_12 = {var_10: var_11}
    var_13 = 'one'
    var_14 = 'two'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = lambda k, v: (k < v, 'Key must be less than value')
    var_17 = 3
    var_18 = 4
    var_19 = {var_11: var_12, var_17: var_18}
    var_20 = 5
    var_21 = 3
    var_22 = {var_20: var_21}



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = 'not_int'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 123
    var_11 = {var_9: var_10}
    var_12 = lambda k, v: (k == v, 'Key must equal value')
    var_13 = 1
    var_14 = 2
    var_15 = {var_13: var_14}
    var_16 = {var_15: var_15, var_1: var_1}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'Test1'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_3[var_0][var_6]
    var_8 = callable(var_7)

def test_case_0():
    var_0 = True
    var_1 = 'Test2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_3[var_0]
    var_7 = 'not_callable'
    var_8 = {}
    var_9 = 'invariants'
    var_10 = {}
    var_11 = var_10[var_9]
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = {}
    var_14 = var_13[var_9]
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_13[var_9]
    var_17 = {}
    var_18 = var_17[var_9]
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_17[var_9]

def test_case_0():
    var_0 = True
    var_1 = 'Test2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_3[var_0]
    var_7 = 'not_callable'
    var_8 = {}
    var_9 = 'invariants'
    var_10 = {}
    var_11 = var_10[var_9]
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = {}
    var_14 = var_13[var_9]
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_13[var_9]
    var_17 = {}
    var_18 = var_17[var_9]
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_17[var_9]



# Parsed testcases at query #22
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'str'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = [var_0, var_2]
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = (var_0, var_2)
    var_7 = module_0.maybe_parse_user_type(var_6)
    var_8 = 1
    var_9 = 2
    var_10 = 123
    var_11 = module_0.maybe_parse_user_type(var_10)
    var_12 = None
    var_13 = module_0.maybe_parse_user_type(var_12)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -1
    var_5 = False
    var_6 = tuple()
    var_7 = (var_5, var_6)
    var_8 = 10



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = '_checked_key_types'
    var_2 = '_checked_value_types'
    var_3 = '_checked_invariants'
    var_4 = '__serializer__'



# Parsed testcases at query #25
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'str_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 10



# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = 'Base'
    var_4 = ()
    var_5 = {}
    var_6 = var_5[var_0]
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'not_callable'
    var_9 = {var_1: var_8}
    var_10 = 'invariants'
    var_11 = 'invariant'

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = 'Base'
    var_4 = ()
    var_5 = {}
    var_6 = var_5[var_0]
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'not_callable'
    var_9 = {var_1: var_8}
    var_10 = 'invariants'
    var_11 = 'invariant'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'OK')
    var_1 = lambda self: (True, 'OK')
    var_2 = lambda self: (False, 'Error')
    var_3 = 'not callable'
    var_4 = 0
    var_5 = None
    var_6 = lambda self: (True, 'G')
    var_7 = lambda self: (True, 'H')



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 100
    var_7 = -100



