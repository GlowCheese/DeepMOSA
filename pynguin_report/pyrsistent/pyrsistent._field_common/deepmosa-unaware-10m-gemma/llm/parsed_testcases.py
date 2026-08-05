####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'original'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_6: var_7}
    var_9 = '_PField'
    var_10 = None



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'x_val'
    var_6 = 10
    var_7 = 'key1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'key2'
    var_11 = 'value2'
    var_12 = {var_10: var_11}
    var_13 = 'existing'
    var_14 = 'data'
    var_15 = {var_13: var_14}
    var_16 = 'val'
    var_17 = 'other'
    var_18 = {var_17: var_2}
    var_19 = 'base_key'
    var_20 = 'base_val'
    var_21 = 'test'
    var_22 = 'target'
    var_23 = {var_0: var_2}
    var_24 = []
    var_25 = 'new_key'
    var_26 = 'a'
    var_27 = 'b'
    var_28 = 1
    var_29 = 2
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = 'b'
    var_32 = 'c'
    var_33 = 3
    var_34 = 4
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = {}
    var_37 = 'overlap'



# Parsed testcases at query #4
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = 'not an int'
    var_3 = 'not a function'
    var_4 = module_0.field(invariant=var_3)
    var_5 = None
    var_6 = module_0.field(factory=var_5)
    var_7 = 'not a function'
    var_8 = module_0.field(serializer=var_7)
    var_9 = 123
    var_10 = [var_9]
    var_11 = module_0.field(var_10)
    var_12 = True
    var_13 = module_0.field(mandatory=var_12)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'my_inv_wrapped'
    var_3 = globals()
    var_4 = var_2 in var_3
    var_5 = 2
    var_6 = 'one'
    var_7 = 'two'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = 'field1'
    var_10 = 'not_an_int'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = '2'
    var_14 = 'a'
    var_15 = 'b'
    var_16 = {var_11: var_14, var_13: var_15}



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = '1'
    var_2 = 'a'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = None
    var_6 = 'b'
    var_7 = {var_1: var_6}
    var_8 = 10
    var_9 = {var_4: var_8}
    var_10 = 'string_key'
    var_11 = 10
    var_12 = {var_10: var_11}
    var_13 = {}



# Parsed testcases at query #7
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = True
    var_10 = 'error_code_1'
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_8, var_12]
    var_14 = module_0.check_global_invariants(var_4, var_13)
    var_15 = True
    var_16 = (var_15, var_6)
    var_17 = lambda x: var_16
    var_18 = module_0.check_global_invariants(var_4, var_13)
    var_19 = False
    var_20 = 'ERR001'
    var_21 = (var_19, var_20)
    var_22 = lambda x: var_21
    var_23 = True
    var_24 = (var_23, var_6)
    var_25 = lambda x: var_24
    var_26 = module_0.check_global_invariants(var_4, var_13)
    var_27 = []
    var_28 = module_0.check_global_invariants(var_4, var_27)
    var_29 = module_0.check_global_invariants(var_4, var_27)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'my_field'
    var_3 = 'hello'
    var_4 = {var_0: var_3}
    var_5 = 'my_field'
    var_6 = 'string_key'
    var_7 = 'value'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #9
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'new_field'
    var_7 = module_0.set_fields(var_4, var_5, var_6)
    var_8 = 'attr'
    var_9 = 'other'
    var_10 = 'k1'
    var_11 = 'v1'
    var_12 = {var_10: var_11}
    var_13 = 'k2'
    var_14 = 'v2'
    var_15 = {var_13: var_14}
    var_16 = {var_8: var_12, var_9: var_15}
    var_17 = 'extra'
    var_18 = 'k3'
    var_19 = 'v3'
    var_20 = {var_18: var_19}
    var_21 = 'k4'
    var_22 = 'v4'
    var_23 = {var_21: var_22}
    var_24 = {var_8: var_20, var_17: var_23}
    var_25 = {}
    var_26 = {}
    var_27 = 'p_field'
    var_28 = {}
    var_29 = 'target'
    var_30 = 'existing_pfield'
    var_31 = 'move_me'
    var_32 = 'new_key'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'get'
    var_1 = True
    var_2 = None
    var_3 = 'StringToIntPMap'
    var_4 = None

def test_case_0():
    var_0 = 'not a callable'
    var_1 = 'not a callable'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'age'
    var_1 = 25
    var_2 = 'name'
    var_3 = 'Alice'
    var_4 = 'age'
    var_5 = 'not_an_int'
    var_6 = 'count'
    var_7 = 10
    var_8 = 'count'
    var_9 = 1
    var_10 = 2
    var_11 = [var_9, var_10]
    var_12 = None
    var_13 = 'score'
    var_14 = 'items'
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = [var_15, var_16, var_17]
    var_19 = 'items'
    var_20 = 123



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'age'
    var_1 = 25
    var_2 = 'name'
    var_3 = 'Alice'
    var_4 = 'mixed'
    var_5 = 10
    var_6 = 'age'
    var_7 = 'twenty-five'
    var_8 = 'name'
    var_9 = 123
    var_10 = 'mixed'
    var_11 = 10.5



# Parsed testcases at query #13
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'not_an_int'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 123
    var_10 = {var_8: var_9}
    var_11 = True
    var_12 = '__getitem__'
    var_13 = None
    var_14 = 'a'
    var_15 = {var_11: var_14}
    var_16 = module_0.pmap(var_15)
    var_17 = module_0.pmap()
    var_18 = 10
    var_19 = 'ten'
    var_20 = {var_18: var_19}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'age'
    var_1 = 25
    var_2 = 'name'
    var_3 = 123
    var_4 = 'score'
    var_5 = 95.5
    var_6 = 'score'
    var_7 = 'A+'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #16
#--------------------------




# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 2
    var_3 = 'one'
    var_4 = 'two'
    var_5 = {var_0: var_3, var_2: var_4}
    var_6 = 'not_an_int'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'IntToStrPMap'



# Parsed testcases at query #18
#--------------------------




####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'new_key'
    var_7 = module_0.set_fields(var_4, var_5, var_6)
    var_8 = 'key1'
    var_9 = 'val1'
    var_10 = 'key2'
    var_11 = 'val2'
    var_12 = {}
    var_13 = 'merged'
    var_14 = 'field_attr'
    var_15 = 'other_attr'
    var_16 = 'internal_key'
    var_17 = 'internal_val'
    var_18 = {var_16: var_17}
    var_19 = 'not_a_field'
    var_20 = 'base_val'
    var_21 = 10
    var_22 = 'new_sub_dict'
    var_23 = 'v'
    var_24 = {}
    var_25 = 'overlap_test'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'shared'
    var_1 = 'f1'
    var_2 = 'original'
    var_3 = 0
    var_4 = False
    var_5 = 'only_a'
    var_6 = 'val_a'
    var_7 = 'a_only'
    var_8 = False
    var_9 = 'only_b'
    var_10 = 'f2'
    var_11 = 'val_b'
    var_12 = 'b_only'
    var_13 = ''
    var_14 = False
    var_15 = 'other'
    var_16 = False
    var_17 = 'keep_me'
    var_18 = 'new_class_name'



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = 'error_code_ignored'
    var_8 = (var_3, var_7)
    var_9 = lambda x: var_8
    var_10 = ''
    var_11 = (var_3, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_6, var_9, var_12]
    var_14 = module_0.check_global_invariants(var_2, var_13)
    var_15 = (var_3, var_4)
    var_16 = lambda x: var_15
    var_17 = False
    var_18 = 'ERR001'
    var_19 = (var_17, var_18)
    var_20 = lambda x: var_19
    var_21 = (var_3, var_4)
    var_22 = lambda x: var_21
    var_23 = [var_16, var_20, var_22]
    var_24 = module_0.check_global_invariants(var_2, var_23)
    var_25 = 'ERR_A'
    var_26 = (var_17, var_25)
    var_27 = lambda x: var_26
    var_28 = (var_3, var_4)
    var_29 = lambda x: var_28
    var_30 = 'ERR_B'
    var_31 = (var_17, var_30)
    var_32 = lambda x: var_31
    var_33 = 'ERR_C'
    var_34 = (var_17, var_33)
    var_35 = lambda x: var_34
    var_36 = [var_27, var_29, var_32, var_35]
    var_37 = module_0.check_global_invariants(var_2, var_36)
    var_38 = (var_17, var_10)
    var_39 = lambda x: var_38
    var_40 = (var_3, var_4)
    var_41 = lambda x: var_40
    var_42 = [var_39, var_41]
    var_43 = module_0.check_global_invariants(var_2, var_42)
    var_44 = []
    var_45 = module_0.check_global_invariants(var_2, var_44)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = dict()
    var_1 = True
    var_2 = None
    var_3 = 2
    var_4 = 'one'
    var_5 = 'two'
    var_6 = {var_1: var_4, var_3: var_5}
    var_7 = 'string_key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = dict()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not a callable'
    var_1 = 'not callable'
    var_2 = module_0.field(factory=var_1)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = ()
    var_1 = set()



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'get'
    var_1 = True
    var_2 = None
    var_3 = dict()
    var_4 = dict()
    var_5 = dict()
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = 2
    var_10 = 'one'
    var_11 = 'two'
    var_12 = {var_1: var_10, var_9: var_11}
    var_13 = 'test_field'
    var_14 = 'not_a_map'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'error_code_1'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = ''
    var_8 = (var_0, var_7)
    var_9 = lambda x: var_8
    var_10 = [var_3, var_6, var_9]
    var_11 = (var_0, var_1)
    var_12 = lambda x: var_11
    var_13 = False
    var_14 = 'ERR01'
    var_15 = (var_13, var_14)
    var_16 = lambda x: var_15
    var_17 = (var_0, var_7)
    var_18 = lambda x: var_17
    var_19 = [var_12, var_16, var_18]
    var_20 = (var_13, var_14)
    var_21 = lambda x: var_20
    var_22 = (var_0, var_1)
    var_23 = lambda x: var_22
    var_24 = 'ERR02'
    var_25 = (var_13, var_24)
    var_26 = lambda x: var_25
    var_27 = 'ERR03'
    var_28 = (var_13, var_27)
    var_29 = lambda x: var_28
    var_30 = [var_21, var_23, var_26, var_29]
    var_31 = []
    var_32 = (var_13, var_7)
    var_33 = lambda x: var_32
    var_34 = (var_0, var_1)
    var_35 = lambda x: var_34
    var_36 = [var_33, var_35]



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'custom_output'
    var_2 = 'plain_string'
    var_3 = 'xml'
    var_4 = 'plain_string'
    var_5 = 'custom_checked_output'
    var_6 = 'yaml'
    var_7 = 123
    var_8 = 'text'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'age'
    var_1 = 25
    var_2 = 'age'
    var_3 = 'twenty-five'
    var_4 = 'name'
    var_5 = 'Alice'
    var_6 = 'name'
    var_7 = 123
    var_8 = 'data'
    var_9 = 100
    var_10 = 'hello'
    var_11 = 'data'
    var_12 = 10.5



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'IntToStrPMap'
    var_3 = None
    var_4 = 2
    var_5 = 'one'
    var_6 = 'two'
    var_7 = {var_0: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'not a callable'
    var_1 = 'not a callable'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the serialize function for different scenarios including:\n    1. Using the default (no-op) serializer.\n    2. Using a custom serializer for non-CheckedType objects.\n    3. Using a custom serializer that triggers CheckedType.serialize.\n    '
    var_1 = 42
    var_2 = 'json'
    var_3 = 'hello'
    var_4 = 'text'
    var_5 = 'xml'
    var_6 = 'any_format'
    var_7 = 'some_value'
    var_8 = 'json'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 10
    var_5 = [var_3, var_3]
    var_6 = 'ERR_001'
    var_7 = []



