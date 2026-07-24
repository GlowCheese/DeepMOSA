####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'test_data'
    var_1 = 'json'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -2
    var_4 = -1
    var_5 = 10



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 0
    var_7 = 100



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1
    var_4 = -1



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'json'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 0
    var_6 = 1



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_3[var_0][var_6]

def test_case_0():
    var_0 = True
    var_1 = 'Test1'
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = 'Test2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 1
    var_7 = var_3[var_0][var_6]
    var_8 = 'not_callable'
    var_9 = {}
    var_10 = 'invariants'
    var_11 = 'test_invariant'

def test_case_0():
    var_0 = True
    var_1 = 'Test1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'Test2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = var_7[var_0]
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_7[var_0][var_6]

def test_case_0():
    var_0 = True
    var_1 = 'Test1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'Test2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = var_7[var_0]
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_7[var_0][var_6]



# Parsed testcases at query #8
#--------------------------


import pyrsistent._checked_types as module_0
import builtins as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'str_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = module_1.object()
    var_7 = module_0.maybe_parse_user_type(var_6)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = lambda k, v: (len(v) > 0, 'Empty value')
    var_1 = '_checked_key_types'
    var_2 = '_checked_value_types'
    var_3 = '_checked_invariants'
    var_4 = '__serializer__'
    var_5 = '__slots__'
    var_6 = 'builtins.int'
    var_7 = 'builtins.str'
    var_8 = 1
    var_9 = 2



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = lambda k, v: (len(v) > 0, 'Empty value')
    var_1 = '_checked_key_types'
    var_2 = '_checked_value_types'
    var_3 = '_checked_invariants'
    var_4 = '__serializer__'
    var_5 = '__slots__'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = 5
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_8}
    var_10 = 1
    var_11 = 2
    var_12 = {var_10: var_11}
    var_13 = lambda k, v: (k < v, 'Key must be less than value')
    var_14 = 2
    var_15 = 1
    var_16 = {var_14: var_15}
    var_17 = 3
    var_18 = 4
    var_19 = {var_15: var_16, var_17: var_18}



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 0



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda self, _, value: f'custom_{value}'
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_6, var_7]
    var_9 = set()



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda self, _, value: value * 2
    var_6 = [var_1, var_2, var_3]
    var_7 = [var_1, var_2]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = {var_1, var_2}
    var_11 = frozenset(var_10)
    var_12 = {var_3, var_8}
    var_13 = frozenset(var_12)
    var_14 = {var_11, var_13}
    var_15 = []
    var_16 = set()



# Parsed testcases at query #18
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

def test_case_0():
    var_0 = True
    var_1 = 'Test2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = {}
    var_7 = var_6[var_0]
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 'not_callable'
    var_10 = {}
    var_11 = 'invariants'
    var_12 = 'invariant'

def test_case_0():
    var_0 = True
    var_1 = 'Test3a'
    var_2 = (var_0, var_1)
    var_3 = 'Test3b'
    var_4 = (var_0, var_3)
    var_5 = (var_2, var_4)
    var_6 = {}
    var_7 = var_6[var_0]
    var_8 = len(var_7)
    assert var_8 == 1

def test_case_0():
    var_0 = False
    var_1 = 'Test4'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 1

def test_case_0():
    var_0 = False
    var_1 = 'Test4'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5
    var_3 = 15
    var_4 = 20



# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -1
    var_5 = 10



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 10



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant_a'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)
    var_8 = {}
    var_9 = var_8[var_1]
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'not callable'
    var_12 = {}
    var_13 = 'invariants'
    var_14 = 'invariant_e'
    var_15 = {}
    var_16 = 'invariant_f'
    var_17 = var_15[var_13]
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = var_15[var_13][var_5]
    var_20 = None
    var_21 = 1
    var_22 = {}
    var_23 = 'invariant_g'
    var_24 = var_22[var_13][var_5]



# Parsed testcases at query #25
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = lambda self: (True, 'ok')
    var_1 = {}
    var_2 = '__stored_invariants__'
    var_3 = '__invariants__'
    var_4 = var_1[var_2]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_1[var_2][var_6]
    var_8 = callable(var_7)
    var_9 = lambda self: (True, 'ok1')
    var_10 = lambda self: (True, 'ok2')
    var_11 = [var_9, var_10]
    var_12 = {}
    var_13 = var_12[var_2]
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = lambda self: (True, 'parent')
    var_16 = lambda self: (True, 'child')
    var_17 = {}
    var_18 = var_17[var_2]
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'not callable'
    var_21 = {}
    var_22 = '__stored_invariants__'
    var_23 = '__invariants__'
    var_24 = module_0.store_invariants(var_21, var_2, var_22, var_23)
    var_25 = {}
    var_26 = var_25[var_2]
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = var_25[var_2][var_6]
    var_29 = None



# Parsed testcases at query #26
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'some_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 'custom_type'
    var_5 = 123
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = module_0.maybe_parse_user_type(var_9)



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 0



# Parsed testcases at query #28
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = module_0.wrap_invariant(var_3)
    var_5 = 10
    var_6 = 'Test1'
    var_7 = (var_0, var_6)
    var_8 = False
    var_9 = 'Test2'
    var_10 = (var_8, var_9)
    var_11 = 'Test3'
    var_12 = (var_0, var_11)
    var_13 = [var_7, var_10, var_12]
    var_14 = lambda x: var_13
    var_15 = module_0.wrap_invariant(var_14)
    var_16 = 'Error'
    var_17 = (var_8, var_16)
    var_18 = lambda x: var_17
    var_19 = module_0.wrap_invariant(var_18)
    var_20 = (var_0, var_6)
    var_21 = (var_0, var_9)
    var_22 = [var_20, var_21]
    var_23 = lambda x: var_22
    var_24 = module_0.wrap_invariant(var_23)
    var_25 = []
    var_26 = lambda x: var_25
    var_27 = module_0.wrap_invariant(var_26)



# Parsed testcases at query #29
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
    var_8 = lambda self: (True, 'test2')
    var_9 = lambda self: (True, 'test3')
    var_10 = {}
    var_11 = var_10[var_2]
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = var_10[var_2]
    var_14 = [inv() for inv in var_13]
    var_15 = lambda self: (True, 'parent')
    var_16 = {}
    var_17 = var_16[var_2]
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = var_16[var_2][var_6]
    var_20 = 'not callable'
    var_21 = {}
    var_22 = '__stored_invariants__'
    var_23 = '__invariants__'
    var_24 = {}
    var_25 = var_24[var_22][var_6]



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = lambda : (True, 'Test2')

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = lambda : (True, 'Test2')



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda self, _, value: value * 2
    var_6 = [var_1, var_2, var_3]
    var_7 = set()



# Parsed testcases at query #32
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap()
    var_6 = 5
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_8}
    var_10 = 1
    var_11 = 2
    var_12 = {var_10: var_11}
    var_13 = lambda k, v: (k == v, 'Key must equal value')
    var_14 = {var_10: var_10, var_11: var_11}
    var_15 = 1
    var_16 = 2
    var_17 = {var_15: var_16}
    var_18 = 3
    var_19 = [var_15, var_16, var_18]
    var_20 = [var_15, var_16, var_18]
    var_21 = [var_15, var_16, var_18]
    var_22 = {var_15: var_21}
    var_23 = [var_15, var_16, var_18]



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 10



# Parsed testcases at query #34
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 2
    var_10 = {var_8: var_9}
    var_11 = 5
    var_12 = {var_9: var_2}
    var_13 = {var_9: var_2, var_10: var_3}
    var_14 = module_0.pmap(var_13)



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'OK')
    var_1 = {}
    var_2 = '__stored_invariants__'
    var_3 = '__invariant__'
    var_4 = var_1[var_2]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_1[var_2][var_6]
    var_8 = None
    var_9 = lambda self: (False, 'Error')
    var_10 = {}
    var_11 = var_10[var_2]
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = var_10[var_2][var_6]
    var_14 = 1
    var_15 = var_10[var_2][var_14]
    var_16 = {}
    var_17 = var_16[var_2]
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = var_16[var_2][var_6]
    var_20 = var_16[var_2][var_14]
    var_21 = 'not callable'
    var_22 = {}
    var_23 = '__stored_invariants__'
    var_24 = '__invariant__'
    var_25 = lambda self: [(True, 'OK1'), (False, 'Error1'), (True, 'OK2')]
    var_26 = {}
    var_27 = var_26[var_23]
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = var_26[var_23][var_6]



# Parsed testcases at query #36
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
    var_13 = 'value1'
    var_14 = 'value2'
    var_15 = {var_12: var_13, var_1: var_14}
    var_16 = {var_12: var_2}



# Parsed testcases at query #37
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = lambda self: (True, 'OK')
    var_1 = lambda self: (False, 'Error')
    var_2 = {}
    var_3 = 'invariants'
    var_4 = '__invariant__'
    var_5 = var_2[var_3]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_2[var_3]
    var_8 = var_2[var_3]
    var_9 = 0
    var_10 = var_8[var_9]
    var_11 = None
    var_12 = 1
    var_13 = var_8[var_12]
    var_14 = 'not callable'
    var_15 = {}
    var_16 = 'invariants'
    var_17 = '__invariant__'
    var_18 = module_0.store_invariants(var_15, var_4, var_16, var_17)
    var_19 = lambda self: (True, 'E')
    var_20 = lambda self: (True, 'F')
    var_21 = {}
    var_22 = var_21[var_15]
    var_23 = len(var_22)
    assert var_23 == 2



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = {}
    var_3 = var_2[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_2[var_1][var_5]
    var_7 = callable(var_6)
    var_8 = {}
    var_9 = var_8[var_1]
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = var_8[var_1]
    var_12 = 'not a function'
    var_13 = {}
    var_14 = 'invariants'
    var_15 = {}
    var_16 = var_15[var_14]
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = var_15[var_14][var_5]
    var_19 = None



# Parsed testcases at query #39
#--------------------------




# Parsed testcases at query #40
#--------------------------




# Parsed testcases at query #41
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



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 0
    var_6 = 10



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'test')
    var_1 = '_invariants'
    var_2 = 0
    var_3 = lambda self: (True, 'test1')
    var_4 = lambda self: (True, 'test2')
    var_5 = 'not callable'
    var_6 = None



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -2
    var_4 = -1
    var_5 = -1



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant_a'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)
    var_8 = {}
    var_9 = var_8[var_1]
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = {}
    var_12 = var_11[var_1]
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 'not callable'
    var_15 = {}
    var_16 = 'invariants'
    var_17 = 'invariant_e'
    var_18 = {}
    var_19 = 'invariant_f'
    var_20 = var_18[var_16]
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = var_18[var_16][var_5]



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1
    var_4 = -2
    var_5 = -1
    var_6 = -2



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = 0
    var_4 = 1
    var_5 = 'not_callable'



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = len(var_2)
    assert var_4 == 1
    var_5 = 0

def test_case_0():
    var_0 = False
    var_1 = 'Test2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = 1
    var_5 = 'not callable'
    var_6 = {}

def test_case_0():
    var_0 = True
    var_1 = 'Test1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'Test2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}

def test_case_0():
    var_0 = True
    var_1 = 'Test1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'Test2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = {}
    var_7 = 5
    var_8 = 'invalid'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 1
    var_12 = 123
    var_13 = {var_11: var_12}
    var_14 = 'invalid'
    var_15 = 123
    var_16 = {var_14: var_15}
    var_17 = 3
    var_18 = [var_14, var_15, var_17]
    var_19 = 4
    var_20 = [var_19, var_7]
    var_21 = 1
    var_22 = 2
    var_23 = 3
    var_24 = [var_21, var_22, var_23]
    var_25 = {var_21: var_24}



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'OK')
    var_1 = {}
    var_2 = '__stored_invariants__'
    var_3 = '__invariants__'
    var_4 = var_1[var_2]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_1[var_2][var_6]
    var_8 = callable(var_7)
    var_9 = lambda self: (True, 'OK')
    var_10 = lambda self: (True, 'Also OK')
    var_11 = {}
    var_12 = var_11[var_2]
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = {}
    var_15 = var_14[var_2]
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'not callable'
    var_18 = {}
    var_19 = '__stored_invariants__'
    var_20 = '__invariants__'
    var_21 = {}
    var_22 = var_21[var_19]
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = var_21[var_19][var_6]
    var_25 = None



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)
    var_8 = {}
    var_9 = var_8[var_1]
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = {}
    var_12 = var_11[var_1]
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = var_11[var_1][var_5]
    var_15 = 50
    var_16 = 'not a function'
    var_17 = {}
    var_18 = 'invariants'
    var_19 = 'invariant'
    var_20 = {}
    var_21 = var_20[var_18]
    var_22 = len(var_21)
    assert var_22 == 1



# Parsed testcases at query #53
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



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 10



# Parsed testcases at query #55
#--------------------------




# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 2
    var_11 = {var_9: var_10}
    var_12 = 2.5
    var_13 = b'b'
    var_14 = {var_11: var_2, var_12: var_13}
    var_15 = lambda k, v: (k < v, 'Key must be less than value')
    var_16 = 3
    var_17 = 4
    var_18 = {var_11: var_1, var_16: var_17}
    var_19 = lambda k, v: (k < v, 'Key must be less than value')
    var_20 = 1
    var_21 = 3
    var_22 = 2
    var_23 = {var_20: var_22, var_21: var_20}



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = -1
    var_5 = -1



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'test')
    var_1 = {}
    var_2 = '__stored_invariants__'
    var_3 = '__invariant__'
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
    var_13 = {}
    var_14 = var_13[var_2]
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'not callable'
    var_17 = {}
    var_18 = '__stored_invariants__'
    var_19 = '__invariant__'
    var_20 = {}
    var_21 = var_20[var_18][var_6]
    var_22 = None



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -1
    var_5 = 10
    var_6 = 15
    var_7 = 2



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = {}
    var_4 = var_3[var_1]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_3[var_1][var_6]
    var_8 = None
    var_9 = {}
    var_10 = var_9[var_1]
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = var_9[var_1][var_6]
    var_13 = 1
    var_14 = var_9[var_1][var_13]
    var_15 = {}
    var_16 = var_15[var_1]
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = var_15[var_1][var_6]
    var_19 = var_15[var_1][var_13]
    var_20 = 'not callable'
    var_21 = {}
    var_22 = 'invariants'
    var_23 = 'invariant'
    var_24 = {}
    var_25 = var_24[var_22]
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_24[var_22][var_6]



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'OK')
    var_1 = '__stored_invariants__'
    var_2 = '__invariant__'
    var_3 = 0
    var_4 = None
    var_5 = lambda self: (True, 'C_OK')
    var_6 = lambda self: (True, 'D_OK')
    var_7 = lambda self: [(True, 'F1_OK'), (True, 'F2_OK')]
    var_8 = 'not callable'
    var_9 = '__stored_invariants__'
    var_10 = '__invariant__'
    var_11 = lambda self: (False, 'H_FAIL')



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = 10
    var_7 = 'invalid_key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 1
    var_11 = 123
    var_12 = {var_10: var_11}
    var_13 = lambda k, v: (k == v, 'Key must equal value')
    var_14 = 1
    var_15 = 2
    var_16 = {var_14: var_15}
    var_17 = {var_16: var_16, var_1: var_1}



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -2
    var_4 = 10



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 2
    var_11 = {var_9: var_10}
    var_12 = {var_1: var_2}
    var_13 = 1
    var_14 = 2
    var_15 = 'a'
    var_16 = {var_14: var_15}
    var_17 = {var_13: var_16}
    var_18 = lambda k, v: (k < v, 'Key must be less than value')
    var_19 = 3
    var_20 = 4
    var_21 = {var_15: var_16, var_19: var_20}
    var_22 = 1
    var_23 = 0
    var_24 = {var_22: var_23}
    var_25 = 1.5
    var_26 = True
    var_27 = {var_24: var_25, var_17: var_26}



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 2
    var_10 = {var_8: var_9}
    var_11 = 5
    var_12 = {var_9: var_2}
    var_13 = lambda k, v: (k == v, 'Key must equal value')
    var_14 = {var_9: var_9, var_10: var_10}
    var_15 = 1
    var_16 = 2
    var_17 = {var_15: var_16}
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = [var_16, var_17, var_18]
    var_21 = [var_16, var_17, var_18]
    var_22 = {var_16: var_21}
    var_23 = [var_16, var_17, var_18]



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1
    var_4 = -1



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 5
    var_3 = 15
    var_4 = -5
    var_5 = 0
    var_6 = 100
    var_7 = -3
    var_8 = 3



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 5
    var_3 = 15
    var_4 = -1
    var_5 = 4
    var_6 = 3
    var_7 = -2



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_3[var_0][var_6]

def test_case_0():
    var_0 = True
    var_1 = 'Test2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = {}
    var_7 = var_6[var_0]
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'not callable'
    var_10 = {}
    var_11 = 'invariants'
    var_12 = 'invariant'
    var_13 = {}
    var_14 = var_13[var_11]
    var_15 = len(var_14)
    assert var_15 == 1

def test_case_0():
    var_0 = True
    var_1 = 'Test2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = var_3[var_0]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = {}
    var_7 = var_6[var_0]
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'not callable'
    var_10 = {}
    var_11 = 'invariants'
    var_12 = 'invariant'
    var_13 = {}
    var_14 = var_13[var_11]
    var_15 = len(var_14)
    assert var_15 == 1



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 10
    var_6 = 0



# Parsed testcases at query #72
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 2
    var_10 = {var_8: var_9}
    var_11 = 10
    var_12 = {var_10: var_2}
    var_13 = 5
    var_14 = {var_10: var_2, var_1: var_3}
    var_15 = module_0.pmap(var_14)



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 10



# Parsed testcases at query #74
#--------------------------




####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 10



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'int'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = None
    var_7 = module_0.maybe_parse_user_type(var_6)



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 2
    var_11 = {var_9: var_10}
    var_12 = lambda k, v: (k == v, 'Key must equal value')
    var_13 = {var_11: var_11, var_1: var_1}
    var_14 = 1
    var_15 = 2
    var_16 = {var_14: var_15}
    var_17 = {var_16: var_2, var_1: var_3}
    var_18 = module_0.pmap(var_17)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.InvariantException()
    var_1 = 'error1'
    var_2 = 'error2'
    var_3 = (var_1, var_2)
    var_4 = module_0.InvariantException(var_3)
    var_5 = lambda : var_1
    var_6 = lambda : var_2
    var_7 = (var_5, var_6)
    var_8 = module_0.InvariantException(var_7)
    var_9 = 'field1'
    var_10 = 'field2'
    var_11 = (var_9, var_10)
    var_12 = module_0.InvariantException(missing_fields=var_11)
    var_13 = (var_1,)
    var_14 = (var_9,)
    var_15 = module_0.InvariantException(var_13, var_14)
    var_16 = (var_1, var_2)
    var_17 = (var_9,)
    var_18 = module_0.InvariantException(var_16, var_17)
    var_19 = str(var_18)
    var_20 = str(var_18)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'test_data'
    var_1 = 'json'
    var_2 = 42
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'test_data'
    var_1 = 'json'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'string_data'
    var_1 = (var_0, var_0)
    var_2 = 123
    var_3 = (var_2, var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_4: var_5}
    var_8 = (var_6, var_7)
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = [var_9, var_10, var_11]
    var_14 = (var_12, var_13)
    var_15 = None
    var_16 = (var_15, var_15)
    var_17 = [var_1, var_3, var_8, var_14, var_16]
    var_18 = 'json'
    var_19 = 'test'
    var_20 = 'custom'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'test_inv'
    var_3 = {}
    var_4 = var_3[var_1]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_3[var_1][var_6]
    var_8 = None
    var_9 = {}
    var_10 = var_9[var_1]
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = var_9[var_1][var_6]
    var_13 = 1
    var_14 = var_9[var_1][var_13]
    var_15 = {}
    var_16 = var_15[var_1]
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'not callable'
    var_19 = {}
    var_20 = 'invariants'
    var_21 = 'test_inv'
    var_22 = {}
    var_23 = var_22[var_20][var_6]



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 0
    var_7 = 2
    var_8 = 1



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda self, _, value: value.upper()
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_0, var_1]
    var_10 = 4
    var_11 = [var_2, var_10]
    var_12 = None
    var_13 = [var_0, var_12, var_2]
    var_14 = []
    var_15 = set()



# Parsed testcases at query #7
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'builtins.str'
    var_3 = module_0.get_type(var_2)
    var_4 = 'pytest.TestClass'
    var_5 = module_0.get_type(var_4)
    var_6 = 'invalid.module.InvalidClass'
    var_7 = module_0.get_type(var_6)



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = lambda self, _, value: value * 2
    var_1 = set()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = lambda self, _, value: value.upper()
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = [var_7, var_8, var_9]



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 10



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = {}
    var_8 = var_7[var_1]
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = {}
    var_11 = var_10[var_1]
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_10[var_1][var_5]
    var_14 = 'not callable'
    var_15 = {}
    var_16 = 'invariants'
    var_17 = 'invariant'
    var_18 = {}



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda x: (x.value > 0, 'Non-positive')
    var_6 = lambda self, fmt, val: val * 2
    var_7 = [var_1, var_2, var_3]



# Parsed testcases at query #13
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'builtins.str'
    var_3 = module_0.get_type(var_2)
    var_4 = '__main__.CustomClass'
    var_5 = module_0.get_type(var_4)
    var_6 = 'non.existent.Module'
    var_7 = module_0.get_type(var_6)
    var_8 = 'invalid.type.string'
    var_9 = module_0.get_type(var_8)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 10



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda self, _, value: str(value)
    var_5 = [var_0, var_1, var_2]
    var_6 = []
    var_7 = set()



# Parsed testcases at query #16
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test_invariant'
    var_1 = True
    var_2 = 'test'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = {var_0: var_4}
    var_6 = 'stored_invariants'
    var_7 = var_5[var_6]
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = var_5[var_6][var_9]
    var_11 = callable(var_10)
    var_12 = lambda x: (True, 'base1')
    var_13 = lambda x: (True, 'base2')
    var_14 = (var_1, var_2)
    var_15 = lambda x: var_14
    var_16 = {var_0: var_15}
    var_17 = var_16[var_6]
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ()
    var_20 = module_0.store_invariants(var_16, var_19, var_6, var_0)
    var_21 = var_16[var_6][var_9]
    var_22 = 'input'
    var_23 = 'not_callable'
    var_24 = {var_0: var_23}
    var_25 = ()
    var_26 = 'stored_invariants'
    var_27 = 'test_invariant'
    var_28 = module_0.store_invariants(var_24, var_25, var_26, var_27)
    var_29 = lambda x: (True, 'parent')
    var_30 = {}
    var_31 = 'parent_invariant'
    var_32 = module_0.store_invariants(var_30, var_25, var_6, var_31)
    var_33 = var_30[var_6]
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = lambda x: (True, 'base')
    var_36 = {}
    var_37 = 'base_invariant'
    var_38 = module_0.store_invariants(var_36, var_25, var_6, var_37)
    var_39 = var_36[var_6]
    var_40 = len(var_39)
    assert var_40 == 1



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'Test'
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
    var_1 = 'Test1'
    var_2 = (var_0, var_1)

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
    var_10 = 'invariant'

def test_case_0():
    var_0 = True
    var_1 = 'Test1'
    var_2 = (var_0, var_1)
    var_3 = 'Test2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = var_6[var_0]
    var_8 = len(var_7)
    assert var_8 == 1

def test_case_0():
    var_0 = True
    var_1 = 'Test1'
    var_2 = (var_0, var_1)
    var_3 = 'Test2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = var_6[var_0]
    var_8 = len(var_7)
    assert var_8 == 1



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'OK')
    var_1 = {}
    var_2 = '__stored_invariants__'
    var_3 = '__invariant__'
    var_4 = var_1[var_2]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_1[var_2][var_6]
    var_8 = callable(var_7)
    var_9 = lambda self: (True, 'OK')
    var_10 = lambda self: (True, 'OK2')
    var_11 = {}
    var_12 = var_11[var_2]
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = {}
    var_15 = var_14[var_2]
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'not callable'
    var_18 = {}
    var_19 = '__stored_invariants__'
    var_20 = '__invariant__'
    var_21 = {}
    var_22 = var_21[var_19][var_6]
    var_23 = None



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1
    var_4 = -1



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda self, fmt, val: val * 2
    var_6 = [var_1, var_2, var_3]
    var_7 = [var_1, var_2]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = {var_1, var_2}
    var_11 = frozenset(var_10)
    var_12 = {var_3, var_8}
    var_13 = frozenset(var_12)
    var_14 = {var_11, var_13}
    var_15 = []
    var_16 = set()



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = tuple()
    var_2 = (var_0, var_1)
    var_3 = tuple()
    var_4 = (var_0, var_3)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = -1



# Parsed testcases at query #24
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



# Parsed testcases at query #25
#--------------------------




# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda x: (x.value > 0, 'Non-positive')
    var_6 = lambda self, _, value: str(value)
    var_7 = [var_1, var_2, var_3]



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda self, _, value: value * 2
    var_6 = [var_1, var_2, var_3]



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda self, _, value: value * 2
    var_6 = [var_1, var_2, var_3]
    var_7 = 10
    var_8 = 20
    var_9 = []
    var_10 = set()



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = -1



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 0



# Parsed testcases at query #31
#--------------------------




# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'OK')
    var_1 = '_invariants'
    var_2 = 0
    var_3 = None
    var_4 = lambda self: (True, 'B_OK')
    var_5 = lambda self: (True, 'C_OK')
    var_6 = 1
    var_7 = lambda self: [(True, 'D1'), (True, 'D2')]
    var_8 = 'not callable'
    var_9 = lambda self: [(True, 'F1'), (False, 'F2'), (True, 'F3')]



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 10



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda self, _, value: value * 2
    var_6 = [var_1, var_2, var_3]
    var_7 = [var_1, var_2]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = {var_1, var_2}
    var_11 = frozenset(var_10)
    var_12 = {var_3, var_8}
    var_13 = frozenset(var_12)
    var_14 = {var_11, var_13}



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 12
    var_6 = -3
    var_7 = 0
    var_8 = 100



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 3
    var_5 = -3
    var_6 = 10



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 3
    var_5 = -2
    var_6 = 10



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 0
    var_6 = -100
    var_7 = 15
    var_8 = -5



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1



# Parsed testcases at query #40
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
    var_15 = frozenset(var_14)



# Parsed testcases at query #41
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



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'ok')
    var_1 = '__invariant__'
    var_2 = 0
    var_3 = lambda self: (True, 'ok')
    var_4 = lambda self: (False, 'error')
    var_5 = 'not callable'
    var_6 = None
    var_7 = lambda self: (True, 'ok')
    var_8 = lambda self: (True, 'ok2')



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = -2
    var_4 = -3
    var_5 = 10



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1
    var_4 = 0



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 10



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 10



# Parsed testcases at query #48
#--------------------------




# Parsed testcases at query #49
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
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.maybe_parse_user_type(var_12)



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = -1



# Parsed testcases at query #51
#--------------------------




# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 10



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = -1



# Parsed testcases at query #54
#--------------------------




# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = -1
    var_3 = 15
    var_4 = 0
    var_5 = -1



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 10



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 3
    var_5 = -2
    var_6 = -1



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 10



# Parsed testcases at query #59
#--------------------------




# Parsed testcases at query #60
#--------------------------




# Parsed testcases at query #61
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'int'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 'str'
    var_5 = [var_2, var_4]
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = 123
    var_8 = module_0.maybe_parse_user_type(var_7)
    var_9 = 123
    var_10 = 'int'
    var_11 = [var_9, var_10]
    var_12 = module_0.maybe_parse_user_type(var_11)



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 10



# Parsed testcases at query #63
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = None
    var_7 = module_0.maybe_parse_user_type(var_6)



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = False
    var_7 = tuple()
    var_8 = (var_6, var_7)
    var_9 = 10



# Parsed testcases at query #65
#--------------------------




# Parsed testcases at query #66
#--------------------------




# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = -1
    var_3 = 15
    var_4 = 4
    var_5 = -2



# Parsed testcases at query #68
#--------------------------




# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1
    var_4 = -1



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = -1



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'A')
    var_1 = {}
    var_2 = '__invariants__'
    var_3 = '__invariant__'
    var_4 = var_1[var_2]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_1[var_2][var_6]
    var_8 = callable(var_7)
    var_9 = lambda self: (True, 'B')
    var_10 = {}
    var_11 = var_10[var_2]
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = lambda self: (True, 'C1')
    var_14 = lambda self: (True, 'C2')
    var_15 = [var_13, var_14]
    var_16 = {}
    var_17 = var_16[var_2]
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = 'not callable'
    var_20 = {}
    var_21 = '__invariants__'
    var_22 = '__invariant__'
    var_23 = lambda self: [(True, 'E1'), (True, 'E2')]
    var_24 = {}
    var_25 = var_24[var_21]
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_24[var_21][var_6]
    var_28 = None
    var_29 = {}



# Parsed testcases at query #72
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'int'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 'str'
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = [var_2, var_4]
    var_7 = module_0.maybe_parse_user_type(var_6)
    var_8 = (var_2, var_4)
    var_9 = module_0.maybe_parse_user_type(var_8)
    var_10 = 'float'
    var_11 = 123
    var_12 = module_0.maybe_parse_user_type(var_11)
    var_13 = None
    var_14 = module_0.maybe_parse_user_type(var_13)



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = -3
    var_4 = 10
    var_5 = 8
    var_6 = 12
    var_7 = -5



# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = 4
    var_1 = -3
    var_2 = 0



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 10



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -1
    var_5 = 0
    var_6 = 100



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 10



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -3
    var_6 = -2



# Parsed testcases at query #79
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



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -2
    var_4 = -1
    var_5 = -1



# Parsed testcases at query #81
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'OK')
    var_1 = '__stored_invariants__'
    var_2 = '__invariant__'
    var_3 = 0
    var_4 = None
    var_5 = lambda self: (True, 'OK2')
    var_6 = 1
    var_7 = 'not callable'
    var_8 = '__stored_invariants__'
    var_9 = '__invariant__'
    var_10 = lambda self: [(True, 'OK1'), (False, 'ERROR1')]



# Parsed testcases at query #82
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 10



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = -3
    var_4 = 3
    var_5 = 10



# Parsed testcases at query #84
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'str_type'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)



# Parsed testcases at query #85
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1



# Parsed testcases at query #86
#--------------------------




# Parsed testcases at query #87
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'test_invariants'
    var_3 = module_0.store_invariants(var_0, var_1, var_2, var_2)
    var_4 = {}
    var_5 = ()
    var_6 = module_0.store_invariants(var_4, var_5, var_2, var_2)
    var_7 = var_4[var_2]
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = var_4[var_2][var_9]
    var_11 = callable(var_10)
    var_12 = {}
    var_13 = ()
    var_14 = module_0.store_invariants(var_12, var_13, var_2, var_2)
    var_15 = var_12[var_2]
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_12[var_2]
    var_18 = {}
    var_19 = 'invariant1'
    var_20 = module_0.store_invariants(var_18, var_13, var_2, var_19)
    var_21 = var_18[var_2]
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = var_18[var_2][var_9]
    var_24 = callable(var_23)
    var_25 = {}
    var_26 = ()
    var_27 = 'test_invariants'
    var_28 = module_0.store_invariants(var_25, var_26, var_27, var_27)
    var_29 = {}
    var_30 = ()
    var_31 = module_0.store_invariants(var_29, var_30, var_27, var_27)
    var_32 = var_29[var_27]
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = var_29[var_27][var_9]
    var_35 = callable(var_34)
    var_36 = var_29[var_27][var_9]



# Parsed testcases at query #88
#--------------------------




# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 10



# Parsed testcases at query #90
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 0
    var_6 = 10



# Parsed testcases at query #91
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
    var_8 = 'not_callable'
    var_9 = {}
    var_10 = 'invariants'
    var_11 = 'invariant'

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



# Parsed testcases at query #92
#--------------------------




# Parsed testcases at query #93
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 2
    var_3 = -1



# Parsed testcases at query #94
#--------------------------




# Parsed testcases at query #95
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



# Parsed testcases at query #96
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -3
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = 0
    var_6 = -3



# Parsed testcases at query #97
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 10



# Parsed testcases at query #98
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = -1
    var_5 = -1



# Parsed testcases at query #99
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = None
    var_7 = module_0.maybe_parse_user_type(var_6)



# Parsed testcases at query #100
#--------------------------




# Parsed testcases at query #101
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 3
    var_4 = -2
    var_5 = -1
    var_6 = 0
    var_7 = 1



# Parsed testcases at query #102
#--------------------------


import pyrsistent._checked_types as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = module_1.object()
    var_7 = module_0.maybe_parse_user_type(var_6)



# Parsed testcases at query #103
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = -1
    var_4 = 10



