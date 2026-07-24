####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'existing'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_1, var_4: var_5}
    var_7 = 'c'
    var_8 = 3
    var_9 = 4
    var_10 = {var_4: var_8, var_7: var_9}
    var_11 = 'test_attr'
    var_12 = 'key1'
    var_13 = 'key2'
    var_14 = 'val1'
    var_15 = 'val2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = 'key3'
    var_18 = 'val3'
    var_19 = 'val4'
    var_20 = {var_13: var_18, var_17: var_19}
    var_21 = {}
    var_22 = '_PField'
    var_23 = None
    var_24 = 'p_field_attr'
    var_25 = 'other'
    var_26 = 100
    var_27 = 'internal_key'
    var_28 = 'internal_val'
    var_29 = {var_27: var_28}
    var_30 = '_PField'
    var_31 = '_PField'
    var_32 = {}
    var_33 = {}



# Parsed testcases at query #2
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
    var_12 = None
    var_13 = 'type'
    var_14 = []
    var_15 = 'a'
    var_16 = {var_11: var_15}
    var_17 = {var_11: var_15}
    var_18 = module_0.pmap(var_17)
    var_19 = 3
    var_20 = 4
    var_21 = 5
    var_22 = 'b'
    var_23 = 'c'
    var_24 = 'd'
    var_25 = 'e'
    var_26 = {var_11: var_15, var_1: var_22, var_19: var_23, var_20: var_24, var_21: var_25}
    var_27 = module_0.pmap(var_26)
    var_28 = {var_15: var_11}
    var_29 = module_0.pmap(var_28)
    var_30 = {var_11: var_29}



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = False
    var_3 = True
    var_4 = 2
    var_5 = 'one'
    var_6 = 'two'
    var_7 = {var_0: var_5, var_4: var_6}
    var_8 = 'my_field'
    var_9 = 'val'
    var_10 = {var_0: var_9}
    var_11 = 'my_field'
    var_12 = 'not_an_int'
    var_13 = 'val'
    var_14 = {var_12: var_13}
    var_15 = 'my_field'
    var_16 = 1
    var_17 = 123
    var_18 = {var_16: var_17}
    var_19 = {var_17: var_9}
    var_20 = f_opt.factory(var_19)[var_17]
    assert var_20 == 'val'



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'age'
    var_1 = 25
    var_2 = 'data'
    var_3 = 'hello'
    var_4 = 123
    var_5 = 'age'
    var_6 = 'not_an_int'
    var_7 = 'count'
    var_8 = 5
    var_9 = ()
    var_10 = 'empty_field'
    var_11 = 10



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_0: var_2}
    var_4 = 'test'
    var_5 = {var_0: var_4}
    var_6 = 2
    var_7 = 'string'
    var_8 = 123
    var_9 = {var_0: var_7, var_6: var_8}
    var_10 = 'not_an_int'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = {}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'simple_string'
    var_2 = 'xml'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 2
    var_4 = 'c'
    var_5 = 3
    var_6 = {}
    var_7 = 'shared_key'
    var_8 = 10
    var_9 = 'my_field'
    var_10 = 'x'
    var_11 = 100
    var_12 = 'y'
    var_13 = 200
    var_14 = {}
    var_15 = 'attr'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 150
    var_2 = []



# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '__name__'
    var_1 = True
    var_2 = None
    var_3 = 2
    var_4 = 'one'
    var_5 = 'two'
    var_6 = {var_1: var_4, var_3: var_5}
    var_7 = 'not_an_int'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = False
    var_11 = True
    assert var_11 is True



# Parsed testcases at query #13
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda _, value: value
    var_1 = 'json'
    var_2 = 'result'
    var_3 = 'some_data'
    var_4 = 'custom_output'
    var_5 = 'xml'
    var_6 = 'yaml'
    var_7 = 123
    var_8 = module_0.serialize(var_0, var_6, var_7)
    assert var_8 == 123



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'test_name'
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = 'val1'
    var_7 = 'val2'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = 'existing'
    var_11 = 'new'
    var_12 = 2
    var_13 = {var_10: var_1, var_11: var_12}
    var_14 = 'extra'
    var_15 = 10
    var_16 = 3
    var_17 = {var_10: var_15, var_14: var_16}
    var_18 = {}
    var_19 = 'shared'
    var_20 = 'only_base'
    var_21 = 'original'
    var_22 = 1
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = 'shared'
    var_25 = 'only_another'
    var_26 = 'overridden'
    var_27 = 2
    var_28 = {var_24: var_26, var_25: var_27}
    var_29 = {}
    var_30 = 'b'
    var_31 = {var_0: var_1, var_30: var_12}



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'xml'
    var_2 = 'data'
    var_3 = 'csv'
    var_4 = 123
    var_5 = 'json'
    var_6 = 'test'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 2
    var_4 = 3
    var_5 = 'c'
    var_6 = 4
    var_7 = {var_5: var_6}
    var_8 = 'x'
    var_9 = 10
    var_10 = {var_8: var_9}
    var_11 = 'Target'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'old_key'
    var_1 = 'old_val'
    var_2 = {var_0: var_1}
    var_3 = 'new_attr'
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = 'val1'
    var_7 = 'val2'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'existing'
    var_10 = 'data'
    var_11 = {var_9: var_10}
    var_12 = 'target_attr'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = []



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'existing'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 0
    var_6 = False
    var_7 = lambda x: x
    var_8 = lambda f, v: v
    var_9 = 'Base2'
    var_10 = 'attr3'
    var_11 = 'value3'
    var_12 = {var_10: var_11}
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = 'some_key'
    var_17 = 'field_to_move'
    var_18 = 'some_val'
    var_19 = True
    var_20 = lambda x: var_19
    var_21 = False
    var_22 = lambda x: x
    var_23 = lambda f, v: v
    var_24 = 1
    var_25 = 2
    var_26 = 'extra'
    var_27 = 'moved_field'
    var_28 = 'data'
    var_29 = 'new_entry'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'a_val'
    var_1 = 1
    var_2 = 'b_val'
    var_3 = 2
    var_4 = 'a_override'
    var_5 = 100
    var_6 = 0
    var_7 = False
    var_8 = 'my_field'
    var_9 = 'my_field'
    var_10 = 'other_key'
    var_11 = 'other_val'
    var_12 = 'not_a_field'
    var_13 = 123
    var_14 = {var_12: var_13}
    var_15 = 'nested'
    var_16 = True



# Parsed testcases at query #7
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
    var_9 = 'key2'
    var_10 = 'val1'
    var_11 = 'val2'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'base_val'
    var_2 = 'some_value'
    var_3 = 'test_field'
    var_4 = 'TargetClass'
    var_5 = 'value_a'
    var_6 = '_PField'
    var_7 = None
    var_8 = 'my_field'



