####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __new__ method of _CheckedMapTypeMeta.\n    It verifies that:\n    1. __key_type__ is correctly extracted from the class dict.\n    2. __value_type__ is correctly extracted from the class dict.\n    3. __invariant__ is correctly extracted from the class dict.\n    4. __serializer__ is set with a default implementation.\n    5. Inheritance works (types and invariants are inherited from bases).\n    6. __slots__ is set to an empty tuple.\n    '
    var_1 = 'extra'
    var_2 = (var_1,)
    var_3 = 1
    var_4 = 'hello'
    var_5 = 0
    var_6 = None



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __new__ method of _CheckedMapTypeMeta, ensuring it correctly:\n    1. Inherits/stores key types from __key_type__.\n    2. Inherits/stores value types from __value_type__.\n    3. Inherits/stores invariants from __invariant__.\n    4. Sets a default __serializer__.\n    5. Sets __slots__ to an empty tuple.\n    '
    var_1 = lambda self, k, v: (True, None)
    var_2 = lambda self, k, v: (v > 0, 'Must be positive')
    var_3 = 0
    var_4 = None
    var_5 = 'key'
    var_6 = 10
    var_7 = 'value'



# Parsed testcases at query #4
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = 'some_source_key'
    var_3 = 'my_invariants'
    var_4 = ()
    var_5 = 'not_callable'
    var_6 = {var_2: var_5}
    var_7 = ()
    var_8 = 'my_invariants'
    var_9 = 'some_source_key'
    var_10 = module_0.store_invariants(var_6, var_7, var_8, var_9)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = lambda self: (True, 'No error')
    var_1 = lambda self: (True, 'Sub error')
    var_2 = 'key'
    var_3 = 123
    var_4 = 'plain_key'
    var_5 = 456



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = -1
    var_3 = 15
    var_4 = -5
    var_5 = 1
    var_6 = 2



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = -1
    var_2 = 0
    var_3 = 5
    var_4 = 4
    var_5 = -1
    var_6 = 11
    var_7 = -2
    var_8 = 2



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'my_invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = {}
    var_6 = 'inv1'
    var_7 = 0
    var_8 = 'ok'
    var_9 = 'fail'
    var_10 = 'not a callable'
    var_11 = {}
    var_12 = 'invariants'
    var_13 = 'check'
    var_14 = 'check'
    var_15 = 10
    var_16 = lambda x: x < var_15
    var_17 = {var_14: var_16}
    var_18 = var_17[var_12]
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = False
    var_21 = 'error'
    var_22 = (var_20, var_21)
    var_23 = lambda x: var_22
    var_24 = {var_14: var_23}
    var_25 = var_24[var_12]
    var_26 = var_25[var_20]
    var_27 = 'anything'
    var_28 = {}
    var_29 = 'non_existent'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'not callable'
    var_1 = 'dest'
    var_2 = 'bad_inv'
    var_3 = None



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = -1
    var_2 = 11
    var_3 = 101
    var_4 = -2
    var_5 = 'ok'
    var_6 = 'fail'
    var_7 = 3
    var_8 = 'string'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}
    var_5 = {var_0: var_1}
    var_6 = {var_0: var_1}



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}
    var_5 = {var_0: var_1}
    var_6 = {var_0: var_1}



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -5
    var_2 = 10
    var_3 = 2
    var_4 = 11
    var_5 = -1
    var_6 = 0
    var_7 = 1



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5
    var_6 = {}
    var_7 = lambda k, v: (len(str(v)) > 0, 'Empty string not allowed')
    var_8 = 'valid'
    var_9 = {var_0: var_8}
    var_10 = 1
    var_11 = ''
    var_12 = {var_10: var_11}



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}
    var_5 = {var_0: var_1}
    var_6 = {var_0: var_1}
    var_7 = {var_0: var_1}



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = lambda self, k, v: (v > 0, 'Must be positive')
    var_1 = 0
    var_2 = 'key'
    var_3 = 10
    var_4 = None
    var_5 = 'plain_key'
    var_6 = 123
    var_7 = '__slots__'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #3
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 'str'
    var_4 = [var_0, var_3]
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = [var_0]
    var_7 = [var_3]
    var_8 = [var_6, var_7]
    var_9 = module_0.maybe_parse_user_type(var_8)
    var_10 = 123
    var_11 = module_0.maybe_parse_user_type(var_10)
    var_12 = 1
    var_13 = 'int'
    var_14 = {var_12: var_13}
    var_15 = module_0.maybe_parse_user_type(var_14)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'builtins.str'
    var_3 = module_0.get_type(var_2)
    var_4 = 'builtins.list'
    var_5 = module_0.get_type(var_4)
    var_6 = 'datetime.datetime'
    var_7 = module_0.get_type(var_6)
    var_8 = 'datetime'
    var_9 = __import__(var_8)
    var_10 = var_9.datetime
    var_11 = 'non_existent_module.NonExistentClass'
    var_12 = module_0.get_type(var_11)
    var_13 = 'builtins.DoesNotExist'
    var_14 = module_0.get_type(var_13)
    var_15 = 'int'
    var_16 = module_0.get_type(var_15)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 123
    var_4 = module_0.maybe_parse_user_type(var_3)
    var_5 = None
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = 'bool'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = lambda self, k, v: (v >= 0, 'Negative Value')
    var_1 = lambda self, k, v: (k > 0, 'Non-positive key')
    var_2 = 0
    var_3 = 1
    var_4 = 10
    var_5 = -1
    var_6 = -5
    var_7 = 'key'
    var_8 = 123



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = [var_0, var_4, var_5]
    var_7 = 'inner1'
    var_8 = 'inner2'
    var_9 = [var_7, var_8]
    var_10 = 'iter2'
    var_11 = [var_7, var_10]
    var_12 = {var_11}
    var_13 = [var_0, var_1]
    var_14 = [var_4, var_5]



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = {}
    var_7 = 'ten'
    var_8 = {var_5: var_7}
    var_9 = 5



# Parsed testcases at query #9
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = {}
    var_7 = 10
    var_8 = 'ten'
    var_9 = {var_7: var_8}
    var_10 = module_0.CheckedPMap()
    var_11 = len(var_10)
    assert var_11 == 0



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = None
    var_3 = 'not a callable'
    var_4 = {}
    var_5 = 'check'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 'key'
    var_3 = 123
    var_4 = '_checked_types'
    var_5 = '_checked_invariants'
    var_6 = 'not a callable'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'builtins.str'
    var_3 = module_0.get_type(var_2)
    var_4 = 'builtins.list'
    var_5 = module_0.get_type(var_4)
    var_6 = 'collections.abc.Iterable'
    var_7 = module_0.get_type(var_6)
    var_8 = 'int'
    var_9 = module_0.get_type(var_8)
    var_10 = 'non_existent_module.SomeClass'
    var_11 = module_0.get_type(var_10)
    var_12 = 'builtins.NonExistentClass'
    var_13 = module_0.get_type(var_12)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = lambda self, k, v: (True, None)
    var_1 = '__serializer__'
    var_2 = None
    var_3 = 'plain_key'
    var_4 = 123



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'A'
    var_6 = 'B'
    var_7 = 10
    var_8 = 20
    var_9 = 'serialized_A'
    var_10 = 'serialized_B'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'x'
    var_13 = 'serialized_1'
    var_14 = 'x'
    var_15 = {var_13: var_14}
    var_16 = {var_13: var_9}
    var_17 = {}



# Parsed testcases at query #15
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 123
    var_4 = module_0.maybe_parse_user_type(var_3)
    var_5 = []
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = 'bool'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -5
    var_2 = 10
    var_3 = 2
    var_4 = -1
    var_5 = 1



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'val'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}
    var_5 = {var_0: var_1}
    var_6 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = -1
    var_2 = 11
    var_3 = 102
    var_4 = -2
    var_5 = 3
    var_6 = 6



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5
    var_6 = {}
    var_7 = 10
    var_8 = 'ten'
    var_9 = {var_7: var_8}
    var_10 = 'one'
    var_11 = {var_0: var_10}
    var_12 = 'not_an_int'
    var_13 = 'value'
    var_14 = {var_12: var_13}



