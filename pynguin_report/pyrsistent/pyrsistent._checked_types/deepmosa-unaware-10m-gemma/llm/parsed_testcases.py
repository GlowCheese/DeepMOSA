####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = 'hello'
    var_6 = True
    var_7 = [var_4, var_5, var_6]
    var_8 = '10'
    var_9 = 'True'
    var_10 = {var_8, var_5, var_9}
    var_11 = [var_6, var_2]
    var_12 = 'data'
    var_13 = []
    var_14 = set()



# Parsed testcases at query #2
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 1
    var_4 = module_0.maybe_parse_user_type(var_3)
    var_5 = None
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = 'float'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 1
    var_3 = 'str'
    var_4 = None
    var_5 = module_0.maybe_parse_user_type(var_4)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = 4
    var_3 = 12
    var_4 = 11
    var_5 = -2
    var_6 = 'magic'
    var_7 = 'not magic'



# Parsed testcases at query #6
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'positive'
    var_2 = lambda x: (x > var_0, var_1)
    var_3 = module_0.wrap_invariant(var_2)
    var_4 = 5
    var_5 = -5
    var_6 = 2
    var_7 = 1
    var_8 = -1
    var_9 = True
    var_10 = lambda x: x == var_9
    var_11 = module_0.wrap_invariant(var_10)
    var_12 = True
    var_13 = False
    var_14 = None



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}
    var_5 = {var_0: var_1}
    var_6 = {var_0: var_1}



# Parsed testcases at query #8
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'str'
    var_3 = [var_2]
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = [var_2]
    var_7 = []
    var_8 = module_0.maybe_parse_user_type(var_7)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 10



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)
    var_8 = {}
    var_9 = 'c_inv'
    var_10 = 'all_invs'
    var_11 = 'p_inv'
    var_12 = 'beta_check'
    var_13 = 'collected'
    var_14 = 'alpha_check'
    var_15 = None
    var_16 = 'delta_inv'
    var_17 = 'merged'
    var_18 = 'gamma_inv'
    var_19 = 'bad_inv'
    var_20 = 'not a callable'
    var_21 = {var_19: var_20}
    var_22 = 'dest'
    var_23 = 'bad_inv'
    var_24 = 'multi'
    var_25 = 'wrapped'
    var_26 = 'test'
    var_27 = {}
    var_28 = 'non_existent'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}
    var_5 = {var_0: var_1}
    var_6 = {var_0: var_1}



# Parsed testcases at query #12
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'my_invariants'
    var_3 = {}
    var_4 = 'check_val'
    var_5 = var_3[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_3[var_1][var_7]
    var_9 = 1
    var_10 = var_3[var_1][var_7]
    var_11 = 2
    var_12 = {}
    var_13 = 'all_invs'
    var_14 = 'inv2'
    var_15 = var_12[var_13]
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = {}
    var_18 = 'collected'
    var_19 = 'check'
    var_20 = var_17[var_18]
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = 'not a function'
    var_23 = {}
    var_24 = 'invariants'
    var_25 = 'check'
    var_26 = module_0.store_invariants(var_23, var_1, var_24, var_25)
    var_27 = {}
    var_28 = 'wrapped'
    var_29 = var_27[var_28][var_7]
    var_30 = {}
    var_31 = var_30[var_28][var_7]
    var_32 = 5
    var_33 = var_30[var_28][var_7]
    var_34 = -5



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = None



# Parsed testcases at query #14
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = [var_2]
    var_4 = 'validators'
    var_5 = 'check'
    var_6 = 0
    var_7 = 2
    var_8 = 'all_checks'
    var_9 = 'primary'
    var_10 = 'rules'
    var_11 = 'rule1'
    var_12 = 'checks'
    var_13 = 'check_a'
    var_14 = 'I am a string'
    var_15 = 'error_test'
    var_16 = 'not_a_callable'
    var_17 = module_0.store_invariants(var_1, var_2, var_15, var_16)
    var_18 = 'multi_wrapped'
    var_19 = 'multi'
    var_20 = None
    var_21 = 'empty'
    var_22 = 'non_existent'



# Parsed testcases at query #15
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = ''
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 'str'



# Parsed testcases at query #16
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'str'
    var_3 = 'Type'
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = None
    var_7 = module_0.maybe_parse_user_type(var_6)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}
    var_5 = {var_0: var_1}
    var_6 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = ''
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 'bool'
    var_5 = 123
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = None
    var_8 = module_0.maybe_parse_user_type(var_7)
    var_9 = 'float'
    var_10 = [var_9]
    var_11 = 'list_of_types'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'MyCustomType'
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 'string'
    var_5 = None
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = module_0.maybe_parse_user_type(var_5)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}
    var_5 = {var_0: var_1}
    var_6 = {var_0: var_1}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = 'not a callable'
    var_3 = 'stored'
    var_4 = 'bad_inv'

def test_case_0():
    var_0 = 'stored_inv'
    var_1 = 'target_inv'
    var_2 = 0
    var_3 = None



# Parsed testcases at query #22
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = 'source_invariant'
    var_3 = 'destination_invariants'
    var_4 = 'source_invariant'
    var_5 = 'not_a_callable'
    var_6 = {var_4: var_5}
    var_7 = ()
    var_8 = 'dest'
    var_9 = module_0.store_invariants(var_6, var_7, var_8, var_4)
    var_10 = lambda x: True



# Parsed testcases at query #23
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'stored_invariants'
    var_1 = 0
    var_2 = None
    var_3 = 'not_callable'
    var_4 = 'stored_invariants'
    var_5 = 'not_an_inv'
    var_6 = 'non_existent'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'some_value'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'json'
    var_5 = 123
    var_6 = 'text'
    var_7 = 'xml'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'id'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 'json'
    var_7 = 'hello'
    var_8 = 'text'
    var_9 = 42



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_value'
    var_1 = 'json'
    var_2 = 'text'
    var_3 = 'key'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = {var_3: var_7}
    var_9 = 'any_format'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = {}
    var_4 = 'src_inv'
    var_5 = var_3[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_3[var_1][var_7]
    var_9 = None
    var_10 = {}
    var_11 = var_10[var_1]
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = var_10[var_1]
    var_14 = [f(var_9) for f in var_13]
    var_15 = {}
    var_16 = var_15[var_1][var_7]
    var_17 = 'not a function'
    var_18 = {}
    var_19 = 'dest'
    var_20 = 'src_inv'
    var_21 = {}
    var_22 = var_21[var_19]
    var_23 = len(var_22)
    assert var_23 == 1



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 20



# Parsed testcases at query #6
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'builtins.str'
    var_3 = module_0.get_type(var_2)
    var_4 = 'collections.abc.Iterable'
    var_5 = module_0.get_type(var_4)
    var_6 = 'int'
    var_7 = module_0.get_type(var_6)
    var_8 = 'non_existent_module.SomeClass'
    var_9 = module_0.get_type(var_8)
    var_10 = 'builtins.Exception'
    var_11 = module_0.get_type(var_10)



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = -1
    var_3 = 10
    var_4 = 4
    var_5 = 11
    var_6 = -1
    var_7 = None



# Parsed testcases at query #9
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = ''
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = 'string'
    var_5 = 123
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = None
    var_8 = [var_7]
    var_9 = module_0.maybe_parse_user_type(var_8)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = {var_0, var_1, var_2}
    var_4 = 'upper'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = []
    var_1 = set()



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 10



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = {var_0: var_1}
    var_5 = {var_0: var_1}
    var_6 = {var_0: var_1}



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 10
    var_5 = 5
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6, var_7}
    var_9 = ()
    var_10 = 'apple'
    var_11 = 'banana'
    var_12 = {var_10, var_11}
    var_13 = [var_0, var_0, var_1]



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 10



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = {}
    var_3 = {}
    var_4 = 'dest'
    var_5 = 'src'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = -5



# Parsed testcases at query #19
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'str'
    var_3 = [var_2]
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)
    var_6 = 'Error occurred'
    var_7 = 'err1'
    var_8 = 'err2'
    var_9 = lambda : var_8
    var_10 = [var_7, var_9]
    var_11 = 'field1'
    var_12 = (var_11,)
    var_13 = module_0.InvariantException(var_10, var_12)
    var_14 = str(var_13)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.InvariantException()
    var_1 = str(var_0)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'data1'
    var_5 = 'data2'
    var_6 = 'serialized_data1'
    var_7 = 'serialized_data2'
    var_8 = {var_6, var_7}
    var_9 = []
    var_10 = set()
    var_11 = 1
    var_12 = 2
    var_13 = [var_11, var_12]
    var_14 = 'plain_string'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 10



# Parsed testcases at query #22
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'str'
    var_3 = [var_2]
    var_4 = 123
    var_5 = module_0.maybe_parse_user_type(var_4)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'float'
    var_3 = 'str'
    var_4 = [var_3]
    var_5 = 123
    var_6 = module_0.maybe_parse_user_type(var_5)
    var_7 = None
    var_8 = module_0.maybe_parse_user_type(var_7)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 10



# Parsed testcases at query #25
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = 'str'
    var_3 = 123
    var_4 = module_0.maybe_parse_user_type(var_3)
    var_5 = None
    var_6 = [var_5]
    var_7 = module_0.maybe_parse_user_type(var_6)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = lambda : var_0
    var_2 = 'Error2'
    var_3 = [var_1, var_2]
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = (var_4, var_5)
    var_7 = 'Base error'
    var_8 = module_0.InvariantException(var_3, var_6)
    var_9 = str(var_8)



