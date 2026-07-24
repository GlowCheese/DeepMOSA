####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = 'fields'
    var_7 = 'field1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'field2'
    var_11 = 'field3'
    var_12 = 'value2'
    var_13 = 'value3'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = {}
    var_16 = 'fields'
    var_17 = set()
    var_18 = False
    var_19 = set()
    var_20 = 'pf1'
    var_21 = 'pf2'
    var_22 = 'other'
    var_23 = 'value'
    var_24 = ()
    var_25 = 'fields'
    var_26 = module_0.set_fields(var_15, var_24, var_25)
    var_27 = 'inherited_field'
    var_28 = 'inherited_value'
    var_29 = {var_27: var_28}
    var_30 = set()
    var_31 = 'new_field'
    var_32 = 'fields'
    var_33 = module_0.set_fields(var_15, var_24, var_32)
    var_34 = {}
    var_35 = 'fields'
    var_36 = module_0.set_fields(var_34, var_24, var_35)
    var_37 = 'field1'
    var_38 = 'old_value'
    var_39 = {var_37: var_38}
    var_40 = set()
    var_41 = 'field1'
    var_42 = 'fields'
    var_43 = module_0.set_fields(var_34, var_24, var_42)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test pmap_field function for creating checked PMap fields.'
    var_1 = True
    var_2 = None
    var_3 = 2
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_1: var_4, var_3: var_5}
    var_7 = False
    var_8 = 'x'
    var_9 = {var_1: var_8}



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'xml'
    var_2 = 'test_value'
    var_3 = 'csv'
    var_4 = 'data'
    var_5 = None
    var_6 = 42
    var_7 = 3.14
    var_8 = True



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test pmap_field function for creating checked PMap fields.'
    var_1 = True
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_1, var_3: var_4}
    var_6 = None
    var_7 = 1.5
    var_8 = 2.5
    var_9 = {var_7: var_2, var_8: var_3}



# Parsed testcases at query #5
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'json'
    var_2 = 2
    var_3 = lambda fmt, val: val * var_2
    var_4 = 21
    var_5 = module_0.serialize(var_3, var_1, var_4)
    assert var_5 == 42
    assert var_5 == 'serialized_json'
    assert var_5 == 'custom_xml'
    var_6 = 'custom_'
    var_7 = lambda fmt, val: var_6 + fmt
    var_8 = 'xml'
    var_9 = lambda fmt, val: fmt.upper()
    var_10 = 'yaml'
    var_11 = 100
    var_12 = module_0.serialize(var_9, var_10, var_11)
    assert var_12 == 'YAML'
    assert var_12 is None
    var_13 = None
    var_14 = lambda fmt, val: str(val)
    var_15 = 123
    var_16 = module_0.serialize(var_14, var_1, var_15)
    assert var_16 == '123'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'key'
    var_3 = {var_2: var_0}
    var_4 = None
    var_5 = {var_2: var_0}
    var_6 = 2
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_0: var_7, var_6: var_8}



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 42
    var_2 = 'xml'
    var_3 = 'test_value'
    var_4 = None
    var_5 = 'csv'
    var_6 = 100
    var_7 = 'yaml'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 42
    var_2 = 21
    var_3 = 'xml'
    var_4 = 'hello'
    var_5 = None
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]



# Parsed testcases at query #9
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = module_0.object()
    var_8 = module_1.check_global_invariants(var_7, var_6)
    var_9 = False
    var_10 = 'error_code_1'
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_12]
    var_14 = module_1.check_global_invariants(var_7, var_13)
    var_15 = 'error_1'
    var_16 = (var_9, var_15)
    var_17 = lambda x: var_16
    var_18 = (var_14, var_1)
    var_19 = lambda x: var_18
    var_20 = 'error_2'
    var_21 = (var_9, var_20)
    var_22 = lambda x: var_21
    var_23 = [var_17, var_19, var_22]
    var_24 = module_1.check_global_invariants(var_7, var_23)
    var_25 = []
    var_26 = module_1.check_global_invariants(var_7, var_25)
    var_27 = 'ignored_1'
    var_28 = (var_24, var_27)
    var_29 = lambda x: var_28
    var_30 = 'code_a'
    var_31 = (var_9, var_30)
    var_32 = lambda x: var_31
    var_33 = 'ignored_2'
    var_34 = (var_24, var_33)
    var_35 = lambda x: var_34
    var_36 = 'code_b'
    var_37 = (var_9, var_36)
    var_38 = lambda x: var_37
    var_39 = 'code_c'
    var_40 = (var_9, var_39)
    var_41 = lambda x: var_40
    var_42 = [var_29, var_32, var_35, var_38, var_41]
    var_43 = module_1.check_global_invariants(var_7, var_42)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for check_type function.'
    var_1 = False
    var_2 = 'test_field'
    var_3 = 42
    var_4 = 'test_field'
    var_5 = 'string'
    var_6 = 'string'
    var_7 = 'test_field'
    var_8 = []
    var_9 = set()
    var_10 = 'anything'
    var_11 = 123
    var_12 = None
    var_13 = 'my_field'
    var_14 = 3.14



# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = module_0.object()
    var_8 = module_1.check_global_invariants(var_7, var_6)
    var_9 = False
    var_10 = 'error_code_1'
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_12]
    var_14 = module_0.object()
    var_15 = module_1.check_global_invariants(var_14, var_13)
    var_16 = (var_15, var_1)
    var_17 = lambda x: var_16
    var_18 = 'error_code_2'
    var_19 = (var_9, var_18)
    var_20 = lambda x: var_19
    var_21 = (var_15, var_1)
    var_22 = lambda x: var_21
    var_23 = [var_17, var_20, var_22]
    var_24 = module_0.object()
    var_25 = module_1.check_global_invariants(var_24, var_23)
    var_26 = 'error_1'
    var_27 = (var_9, var_26)
    var_28 = lambda x: var_27
    var_29 = 'error_2'
    var_30 = (var_9, var_29)
    var_31 = lambda x: var_30
    var_32 = (var_25, var_1)
    var_33 = lambda x: var_32
    var_34 = 'error_3'
    var_35 = (var_9, var_34)
    var_36 = lambda x: var_35
    var_37 = [var_28, var_31, var_33, var_36]
    var_38 = module_0.object()
    var_39 = module_1.check_global_invariants(var_38, var_37)
    var_40 = []
    var_41 = module_0.object()
    var_42 = module_1.check_global_invariants(var_41, var_40)
    var_43 = 'negative_value'
    var_44 = lambda x: (x.value > var_9, var_43)
    var_45 = [var_44]
    var_46 = 5
    var_47 = module_1.check_global_invariants(var_41, var_45)
    var_48 = -1
    var_49 = module_1.check_global_invariants(var_41, var_45)
    var_50 = 'test_error'
    var_51 = (var_9, var_50)
    var_52 = lambda x: var_51
    var_53 = [var_52]
    var_54 = module_0.object()
    var_55 = module_1.check_global_invariants(var_54, var_53)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = lambda x: x



# Parsed testcases at query #14
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = module_0.object()
    var_8 = module_1.check_global_invariants(var_7, var_6)
    var_9 = False
    var_10 = 'error_code_1'
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_12]
    var_14 = module_1.check_global_invariants(var_7, var_13)
    var_15 = (var_14, var_1)
    var_16 = lambda x: var_15
    var_17 = 'error_code_2'
    var_18 = (var_9, var_17)
    var_19 = lambda x: var_18
    var_20 = 'error_code_3'
    var_21 = (var_9, var_20)
    var_22 = lambda x: var_21
    var_23 = [var_16, var_19, var_22]
    var_24 = module_1.check_global_invariants(var_7, var_23)
    var_25 = []
    var_26 = module_1.check_global_invariants(var_7, var_25)
    var_27 = 'error_1'
    var_28 = (var_9, var_27)
    var_29 = lambda x: var_28
    var_30 = 'error_2'
    var_31 = (var_9, var_30)
    var_32 = lambda x: var_31
    var_33 = 'error_3'
    var_34 = (var_9, var_33)
    var_35 = lambda x: var_34
    var_36 = [var_29, var_32, var_35]
    var_37 = module_1.check_global_invariants(var_7, var_36)
    var_38 = False
    var_39 = 'test_error'
    var_40 = (var_38, var_39)
    var_41 = lambda x: var_40
    var_42 = [var_41]
    var_43 = module_1.check_global_invariants(var_7, var_42)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'xml'
    var_2 = 'test_value'
    var_3 = 'plain_string'
    var_4 = 'csv'
    var_5 = 42
    var_6 = 'binary'
    var_7 = 123
    var_8 = None



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'xml'
    var_2 = 'test_value'
    var_3 = 'plain_value'
    var_4 = 'csv'
    var_5 = 42
    var_6 = None
    var_7 = 'binary'
    var_8 = 'data'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test check_type function for type validation.'
    var_1 = False
    var_2 = 'test_field'
    var_3 = 42
    var_4 = 'test_field'
    var_5 = 'string'
    var_6 = 'string'
    var_7 = 'test_field'
    var_8 = 3.14
    var_9 = set()
    var_10 = 'anything'
    var_11 = 'my_field'
    var_12 = []
    var_13 = 5



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'xml'
    var_2 = 'test_value'
    var_3 = 42
    var_4 = 'csv'
    var_5 = 'data'
    var_6 = None
    var_7 = 'test'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'test_value'
    var_2 = 'xml'
    var_3 = 'plain_value'
    var_4 = 'csv'
    var_5 = 123



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test pmap_field function for creating checked PMap fields.'
    var_1 = 0
    var_2 = True
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 2
    var_6 = {var_3: var_2, var_4: var_5}
    var_7 = None
    var_8 = {var_3: var_2}
    var_9 = 'Map must not be empty'
    var_10 = lambda x: (len(x) > var_1, var_9)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test pmap_field function for creating checked PMap fields.'
    var_1 = False
    var_2 = True
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_2, var_4: var_5}
    var_7 = None
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 10
    var_11 = 20
    var_12 = {var_8: var_10, var_9: var_11}



# Parsed testcases at query #22
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Test pmap_field function creates proper checked PMap fields.'
    var_1 = module_0.pmap()
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = True
    var_8 = None
    var_9 = 'x'
    var_10 = 10
    var_11 = {var_9: var_10}
    var_12 = 'one'
    var_13 = 'two'
    var_14 = {var_7: var_12, var_5: var_13}



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = lambda x: x



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = None
    var_6 = 'x'
    var_7 = 10
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = {}
    var_11 = {var_0: var_1, var_3: var_2}



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'xml'
    var_2 = 'test_value'
    var_3 = 'yaml'
    var_4 = 42
    var_5 = 123
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'proto'
    var_10 = None
    var_11 = 'csv'
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 42
    var_2 = 21
    var_3 = 10
    var_4 = None
    var_5 = 'hello'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 42
    var_2 = 21
    var_3 = 'xml'
    var_4 = 100
    var_5 = None
    var_6 = 'test_string'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = None
    var_7 = 'x'
    var_8 = 10
    var_9 = {var_7: var_8}
    var_10 = {var_5: var_0, var_3: var_1}
    var_11 = False
    var_12 = {}



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = lambda x: x



# Parsed testcases at query #31
#--------------------------




# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = lambda x: x



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = lambda x: x



# Parsed testcases at query #34
#--------------------------




# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = lambda x: x
    var_3 = set()



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = 'json'
    var_2 = module_0.serialize(var_1)
    var_3 = 42
    var_4 = 'xml'
    var_5 = 'test'
    var_6 = 'any_format'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = lambda fmt, val: val * var_8
    var_12 = module_0.serialize(var_11, var_1, var_0)
    assert var_12 == 10



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test set_fields function for proper field extraction and organization.'
    var_1 = {}
    var_2 = {}
    var_3 = '__fields__'
    var_4 = var_2[var_3]
    var_5 = set()
    var_6 = False
    var_7 = set()
    var_8 = 'name'
    var_9 = 'age'
    var_10 = {}
    var_11 = set()
    var_12 = set()
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = {}
    var_16 = set()
    var_17 = 'field'
    var_18 = 'other_attr'
    var_19 = 'another'
    var_20 = 'value'
    var_21 = 123
    var_22 = {}
    var_23 = {}
    var_24 = set()
    var_25 = set()
    var_26 = 'shared'
    var_27 = 'shared'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = 'fields'
    var_5 = 'fields'
    var_6 = 'field1'
    var_7 = {}
    var_8 = 'fields'
    var_9 = module_0.set_fields(var_7, var_1, var_8)
    var_10 = set()
    var_11 = False
    var_12 = 'my_field'
    var_13 = ()
    var_14 = 'fields'
    var_15 = module_0.set_fields(var_7, var_13, var_14)
    var_16 = 'fields'
    var_17 = 'field2'
    var_18 = set()
    var_19 = 'field3'
    var_20 = 'fields'
    var_21 = module_0.set_fields(var_7, var_13, var_20)
    var_22 = 'regular_attr'
    var_23 = 'another'
    var_24 = 'value'
    var_25 = 123
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = ()
    var_28 = 'fields'
    var_29 = module_0.set_fields(var_26, var_27, var_28)
    var_30 = set()
    var_31 = 'field_obj'
    var_32 = 'regular'
    var_33 = 'attr'
    var_34 = ()
    var_35 = 'fields'
    var_36 = module_0.set_fields(var_26, var_34, var_35)



# Parsed testcases at query #4
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = module_0.object()
    var_8 = module_1.check_global_invariants(var_7, var_6)
    var_9 = False
    var_10 = 'error_1'
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_12]
    var_14 = module_0.object()
    var_15 = module_1.check_global_invariants(var_14, var_13)
    var_16 = (var_15, var_1)
    var_17 = lambda x: var_16
    var_18 = 'error_2'
    var_19 = (var_9, var_18)
    var_20 = lambda x: var_19
    var_21 = (var_15, var_1)
    var_22 = lambda x: var_21
    var_23 = [var_17, var_20, var_22]
    var_24 = module_0.object()
    var_25 = module_1.check_global_invariants(var_24, var_23)
    var_26 = (var_9, var_10)
    var_27 = lambda x: var_26
    var_28 = (var_9, var_18)
    var_29 = lambda x: var_28
    var_30 = 'error_3'
    var_31 = (var_9, var_30)
    var_32 = lambda x: var_31
    var_33 = [var_27, var_29, var_32]
    var_34 = module_0.object()
    var_35 = module_1.check_global_invariants(var_34, var_33)
    var_36 = []
    var_37 = module_0.object()
    var_38 = module_1.check_global_invariants(var_37, var_36)
    var_39 = []
    var_40 = (var_35, var_1)
    var_41 = lambda x: (received_subjects.append(x), var_40)[var_35]
    var_42 = [var_41]
    var_43 = 'test'
    var_44 = 'value'
    var_45 = {var_43: var_44}
    var_46 = module_1.check_global_invariants(var_45, var_42)
    var_47 = 'custom_error'
    var_48 = (var_9, var_47)
    var_49 = lambda x: var_48
    var_50 = [var_49]
    var_51 = module_0.object()
    var_52 = module_1.check_global_invariants(var_51, var_50)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'test_name'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = set()
    var_5 = False
    var_6 = set()
    var_7 = '_pfields'
    var_8 = 'field1'
    var_9 = '_pfields'
    var_10 = 'field2'
    var_11 = {}
    var_12 = '_pfields'
    var_13 = module_0.set_fields(var_11, var_1, var_12)
    var_14 = set()
    var_15 = 'my_field'
    var_16 = 'other_value'
    var_17 = 'not_a_field'
    var_18 = ()
    var_19 = module_0.set_fields(var_11, var_18, var_12)
    var_20 = set()
    var_21 = set()
    var_22 = '_pfields'
    var_23 = 'inherited_field'
    var_24 = 'own_field'
    var_25 = module_0.set_fields(var_11, var_18, var_12)



# Parsed testcases at query #6
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = module_0.object()
    var_8 = module_1.check_global_invariants(var_7, var_6)
    var_9 = False
    var_10 = 'error_code_1'
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_12]
    var_14 = module_0.object()
    var_15 = module_1.check_global_invariants(var_14, var_13)
    var_16 = 'error_1'
    var_17 = (var_9, var_16)
    var_18 = lambda x: var_17
    var_19 = 'error_2'
    var_20 = (var_9, var_19)
    var_21 = lambda x: var_20
    var_22 = (var_15, var_1)
    var_23 = lambda x: var_22
    var_24 = [var_18, var_21, var_23]
    var_25 = module_0.object()
    var_26 = module_1.check_global_invariants(var_25, var_24)
    var_27 = []
    var_28 = module_0.object()
    var_29 = module_1.check_global_invariants(var_28, var_27)
    var_30 = 42
    var_31 = (var_9, var_30)
    var_32 = lambda x: var_31
    var_33 = 'string_error'
    var_34 = (var_9, var_33)
    var_35 = lambda x: var_34
    var_36 = [var_32, var_35]
    var_37 = module_0.object()
    var_38 = module_1.check_global_invariants(var_37, var_36)
    var_39 = []
    var_40 = 'test'
    var_41 = 'object'
    var_42 = {var_40: var_41}
    var_43 = module_1.check_global_invariants(var_42, var_36)



# Parsed testcases at query #7
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for check_type function.'
    var_1 = 'test_field'
    var_2 = 42
    var_3 = 'test_field'
    var_4 = 'not an int'
    var_5 = 'hello'
    var_6 = 'test_field'
    var_7 = 3.14
    var_8 = 'anything'
    var_9 = None
    var_10 = 'custom_field'
    var_11 = 'custom_field'
    var_12 = 42
    var_13 = 'parent_field'
    var_14 = 'test_field'
    var_15 = []



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test pmap_field function.'
    var_1 = True
    var_2 = None
    var_3 = False
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = 2
    var_7 = {var_4: var_1, var_5: var_6}



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 'string_value'
    var_3 = 'test_field'
    var_4 = 'string_value'
    var_5 = 42
    var_6 = 'string'
    var_7 = set()
    var_8 = 'any_value'
    var_9 = 123
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = 'test_field'
    var_15 = None
    var_16 = None



# Parsed testcases at query #11
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = module_0.object()
    var_8 = module_1.check_global_invariants(var_7, var_6)
    var_9 = False
    var_10 = 'error_code_1'
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_12]
    var_14 = module_1.check_global_invariants(var_7, var_13)
    var_15 = (var_14, var_1)
    var_16 = lambda x: var_15
    var_17 = 'error_code_2'
    var_18 = (var_9, var_17)
    var_19 = lambda x: var_18
    var_20 = 'error_code_3'
    var_21 = (var_9, var_20)
    var_22 = lambda x: var_21
    var_23 = [var_16, var_19, var_22]
    var_24 = module_1.check_global_invariants(var_7, var_23)
    var_25 = []
    var_26 = module_1.check_global_invariants(var_7, var_25)
    var_27 = (var_24, var_1)
    var_28 = lambda x: var_27
    var_29 = 'some_code'
    var_30 = (var_24, var_29)
    var_31 = lambda x: var_30
    var_32 = [var_28, var_31]
    var_33 = module_1.check_global_invariants(var_7, var_32)
    var_34 = 'code_a'
    var_35 = (var_9, var_34)
    var_36 = lambda x: var_35
    var_37 = 'code_b'
    var_38 = (var_9, var_37)
    var_39 = lambda x: var_38
    var_40 = 'code_c'
    var_41 = (var_9, var_40)
    var_42 = lambda x: var_41
    var_43 = [var_36, var_39, var_42]
    var_44 = module_1.check_global_invariants(var_7, var_43)



# Parsed testcases at query #12
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'success'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test_subject'
    var_9 = module_0.check_global_invariants(var_8, var_7)
    var_10 = False
    var_11 = 'error_code_1'
    var_12 = (var_10, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_13]
    var_15 = 'test_subject'
    var_16 = module_0.check_global_invariants(var_15, var_14)
    var_17 = 'error_1'
    var_18 = (var_10, var_17)
    var_19 = lambda x: var_18
    var_20 = 'error_2'
    var_21 = (var_10, var_20)
    var_22 = lambda x: var_21
    var_23 = (var_15, var_16)
    var_24 = lambda x: var_23
    var_25 = [var_19, var_22, var_24]
    var_26 = 'test_subject'
    var_27 = module_0.check_global_invariants(var_26, var_25)
    var_28 = []
    var_29 = module_0.check_global_invariants(var_8, var_28)
    var_30 = 'not_positive'
    var_31 = (var_26, var_27)
    var_32 = 42
    var_33 = -5
    var_34 = 'test_error'
    var_35 = (var_10, var_34)
    var_36 = lambda x: var_35
    var_37 = [var_36]
    var_38 = 'subject'
    var_39 = module_0.check_global_invariants(var_38, var_37)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test is_field_ignore_extra_complaint function.'
    var_1 = False
    var_2 = True
    var_3 = lambda x: x



# Parsed testcases at query #14
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = var_0.type
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = var_0.type
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_0.type
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_0.type
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = True
    var_11 = 42
    var_12 = 2
    var_13 = lambda x: x * var_12
    var_14 = lambda fmt, val: str(val)
    var_15 = 0
    var_16 = 'Must be positive'
    var_17 = lambda x: (x > var_15, var_16)
    var_18 = var_0.invariant
    var_19 = callable(var_18)
    var_20 = 10
    var_21 = lambda : var_20
    var_22 = var_0.initial
    var_23 = callable(var_22)
    var_24 = 123
    var_25 = [var_24]
    var_26 = module_0.field(var_25)
    var_27 = 'not an int'
    var_28 = 'not callable'
    var_29 = module_0.field(invariant=var_28)
    var_30 = 'not callable'
    var_31 = module_0.field(factory=var_30)
    var_32 = 'not callable'
    var_33 = module_0.field(serializer=var_32)
    var_34 = 'SomeType'
    var_35 = module_0.field(var_34)
    var_36 = var_35.type
    var_37 = len(var_36)
    var_38 = 'hello'
    var_39 = set()
    var_40 = module_0.field(var_39)
    var_41 = set()



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = None
    var_7 = 'a'
    var_8 = {var_7: var_0}
    var_9 = 'x'
    var_10 = 'y'
    var_11 = 1.5
    var_12 = 2.5
    var_13 = {var_9: var_11, var_10: var_12}



# Parsed testcases at query #16
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda subject: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda subject: var_4
    var_6 = [var_3, var_5]
    var_7 = 'test_subject'
    var_8 = module_0.check_global_invariants(var_7, var_6)
    var_9 = False
    var_10 = 'error_1'
    var_11 = (var_9, var_10)
    var_12 = lambda subject: var_11
    var_13 = [var_12]
    var_14 = 'test_subject'
    var_15 = module_0.check_global_invariants(var_14, var_13)
    var_16 = (var_14, var_15)
    var_17 = lambda subject: var_16
    var_18 = 'error_2'
    var_19 = (var_9, var_18)
    var_20 = lambda subject: var_19
    var_21 = 'error_3'
    var_22 = (var_9, var_21)
    var_23 = lambda subject: var_22
    var_24 = [var_17, var_20, var_23]
    var_25 = 'test_subject'
    var_26 = module_0.check_global_invariants(var_25, var_24)
    var_27 = []
    var_28 = module_0.check_global_invariants(var_7, var_27)
    var_29 = 5
    var_30 = 'value_too_small'
    var_31 = (var_25, var_26)
    var_32 = 3
    var_33 = module_0.check_global_invariants(var_32, var_27)
    var_34 = (var_32, var_33)
    var_35 = 10
    var_36 = module_0.check_global_invariants(var_35, var_27)
    var_37 = 42
    var_38 = 'negative_value'
    var_39 = lambda s: (s.value > var_9, var_38)
    var_40 = 100
    var_41 = 'value_too_large'
    var_42 = lambda s: (s.value < var_40, var_41)
    var_43 = [var_39, var_42]
    var_44 = -5
    var_45 = lambda s: (s.value > var_9, var_38)
    var_46 = [var_45]
    var_47 = 'error_a'
    var_48 = (var_9, var_47)
    var_49 = lambda subject: var_48
    var_50 = 'error_b'
    var_51 = (var_9, var_50)
    var_52 = lambda subject: var_51
    var_53 = 'error_c'
    var_54 = (var_9, var_53)
    var_55 = lambda subject: var_54
    var_56 = (var_32, var_33)
    var_57 = lambda subject: var_56
    var_58 = [var_49, var_52, var_55, var_57]
    var_59 = 'subject'
    var_60 = module_0.check_global_invariants(var_59, var_58)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = None
    var_3 = 'key'
    var_4 = {var_3: var_1}
    var_5 = False
    var_6 = 2
    var_7 = {var_3: var_6}



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = set()
    var_1 = ()
    var_2 = ()
    var_3 = 'CheckedPVector'
    var_4 = (var_3,)



# Parsed testcases at query #19
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = var_0.type
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = var_0.type
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_0.type
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = 42
    var_9 = True
    var_10 = module_0.field(mandatory=var_9)
    var_11 = 2
    var_12 = lambda x: x * var_11
    var_13 = module_0.field(factory=var_12)
    var_14 = lambda fmt, val: str(val)
    var_15 = module_0.field(serializer=var_14)
    var_16 = 0
    var_17 = 'must be positive'
    var_18 = lambda x: (x > var_16, var_17)
    var_19 = module_0.field(invariant=var_18)
    var_20 = module_0.field(invariant=var_18)
    var_21 = var_20.invariant
    var_22 = callable(var_21)
    var_23 = ''
    var_24 = (var_9, var_23)
    var_25 = lambda x: var_24
    var_26 = 10
    var_27 = lambda x: int(x)
    var_28 = lambda fmt, val: str(val)
    var_29 = var_20.invariant
    var_30 = callable(var_29)
    var_31 = 123
    var_32 = 'invalid'
    var_33 = [var_31, var_32]
    var_34 = module_0.field(var_33)
    var_35 = 'not an int'
    var_36 = 'not callable'
    var_37 = module_0.field(invariant=var_36)
    var_38 = 'not callable'
    var_39 = module_0.field(factory=var_38)
    var_40 = 'not callable'
    var_41 = module_0.field(serializer=var_40)
    var_42 = lambda : var_8
    var_43 = var_20.initial
    var_44 = callable(var_43)
    var_45 = 'SomeClass'
    var_46 = module_0.field(var_45)
    var_47 = []
    var_48 = module_0.field(var_47)
    var_49 = set()
    var_50 = module_0.field()



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = ()
    var_1 = []
    var_2 = 'SomeType'
    var_3 = (var_2,)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test check_type function for type validation.'
    var_1 = False
    var_2 = 'test_field'
    var_3 = 42
    var_4 = 'test_field'
    var_5 = 'string'
    var_6 = 'string'
    var_7 = 'test_field'
    var_8 = 3.14
    var_9 = set()
    var_10 = 'anything'
    var_11 = None
    var_12 = 'test_field'
    var_13 = None



# Parsed testcases at query #22
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda subject: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda subject: var_4
    var_6 = [var_3, var_5]
    var_7 = 'test_subject'
    var_8 = module_0.check_global_invariants(var_7, var_6)
    var_9 = False
    var_10 = 'error_1'
    var_11 = (var_9, var_10)
    var_12 = lambda subject: var_11
    var_13 = [var_12]
    var_14 = 'test_subject'
    var_15 = module_0.check_global_invariants(var_14, var_13)
    var_16 = (var_14, var_15)
    var_17 = lambda subject: var_16
    var_18 = 'error_2'
    var_19 = (var_9, var_18)
    var_20 = lambda subject: var_19
    var_21 = [var_17, var_20]
    var_22 = 'test_subject'
    var_23 = module_0.check_global_invariants(var_22, var_21)
    var_24 = (var_9, var_10)
    var_25 = lambda subject: var_24
    var_26 = (var_9, var_18)
    var_27 = lambda subject: var_26
    var_28 = (var_22, var_23)
    var_29 = lambda subject: var_28
    var_30 = 'error_3'
    var_31 = (var_9, var_30)
    var_32 = lambda subject: var_31
    var_33 = [var_25, var_27, var_29, var_32]
    var_34 = 'test_subject'
    var_35 = module_0.check_global_invariants(var_34, var_33)
    var_36 = []
    var_37 = module_0.check_global_invariants(var_7, var_36)
    var_38 = 42
    var_39 = 'not_42'
    var_40 = lambda subject: (subject == var_38, var_39)
    var_41 = [var_40]
    var_42 = 100
    var_43 = module_0.check_global_invariants(var_42, var_41)
    var_44 = lambda subject: (subject == var_38, var_39)
    var_45 = [var_44]
    var_46 = module_0.check_global_invariants(var_38, var_45)
    var_47 = 'test_error'
    var_48 = (var_9, var_47)
    var_49 = lambda subject: var_48
    var_50 = [var_49]
    var_51 = 'subject'
    var_52 = module_0.check_global_invariants(var_51, var_50)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test check_type function for type validation.'
    var_1 = False
    var_2 = 'test_field'
    var_3 = 42
    var_4 = 'test_field'
    var_5 = 'string'
    var_6 = 'string'
    var_7 = 'test_field'
    var_8 = []
    var_9 = set()
    var_10 = 'anything'
    var_11 = 'my_field'
    var_12 = 3.14



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test pmap_field function creates proper checked PMap fields.'
    var_1 = 1
    var_2 = 2
    var_3 = 'one'
    var_4 = 'two'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = None
    var_8 = {var_6: var_3}
    var_9 = optional_field.factory(var_8)[var_6]
    assert var_9 == 'one'
    var_10 = False
    var_11 = {var_6: var_3}
    var_12 = True
    var_13 = {}
    var_14 = {}
    var_15 = 'pi'
    var_16 = 'e'
    var_17 = 3.14
    var_18 = 2.71
    var_19 = {var_15: var_17, var_16: var_18}



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 42
    var_3 = 'test_field'
    var_4 = 'invalid'
    var_5 = 'multi_field'
    var_6 = 'valid'
    var_7 = 'multi_field'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 'no_type_field'
    var_13 = 'anything'
    var_14 = 123
    var_15 = None
    var_16 = 'parent_field'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for check_type function.'
    var_1 = False
    var_2 = 'test_field'
    var_3 = 42
    var_4 = 'test_field'
    var_5 = 'invalid_string'
    var_6 = 'multi_field'
    var_7 = 'string'
    var_8 = 'multi_field'
    var_9 = 3.14
    var_10 = 'any_field'
    var_11 = 'anything'
    var_12 = 12345
    var_13 = None
    var_14 = 'parent_field'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = ()
    var_1 = 'test_field'
    var_2 = 'any_value'
    var_3 = 42
    var_4 = 'test_field'
    var_5 = 'not_an_int'
    var_6 = 'multi_field'
    var_7 = 'string_value'
    var_8 = 100
    var_9 = 'multi_field'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = ()
    var_15 = 'nullable_field'
    var_16 = None
    var_17 = 'custom_field'
    var_18 = 'my_field'
    var_19 = True



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_field'
    var_2 = 42
    var_3 = 'test_field'
    var_4 = 'invalid'
    var_5 = str(var_3)
    var_6 = 'multi_field'
    var_7 = 'string_value'
    var_8 = 'multi_field'
    var_9 = 123
    var_10 = 'multi_field'
    var_11 = []
    var_12 = ()
    var_13 = 'no_type_field'
    var_14 = 'anything'
    var_15 = 42
    var_16 = []
    var_17 = 'test_field'
    var_18 = None
    var_19 = None



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test pmap_field function for creating checked PMap fields.'
    var_1 = 0
    var_2 = True
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 2
    var_6 = {var_3: var_2, var_4: var_5}
    var_7 = None
    var_8 = ''
    var_9 = (var_2, var_8)
    var_10 = lambda x: var_9



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'xml'
    var_2 = 'custom_xml_'
    var_3 = 'plain_string'
    var_4 = 'csv'
    var_5 = 42
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = None
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = 'test'



# Parsed testcases at query #31
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = 'test_subject'
    var_8 = module_0.check_global_invariants(var_7, var_6)
    var_9 = False
    var_10 = 'error_code_1'
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_12]
    var_14 = module_0.check_global_invariants(var_7, var_13)
    var_15 = (var_9, var_10)
    var_16 = lambda x: var_15
    var_17 = 'error_code_2'
    var_18 = (var_9, var_17)
    var_19 = lambda x: var_18
    var_20 = (var_14, var_1)
    var_21 = lambda x: var_20
    var_22 = [var_16, var_19, var_21]
    var_23 = module_0.check_global_invariants(var_7, var_22)
    var_24 = []
    var_25 = module_0.check_global_invariants(var_7, var_24)
    var_26 = 'ok_1'
    var_27 = (var_23, var_26)
    var_28 = lambda x: var_27
    var_29 = 'error_1'
    var_30 = (var_9, var_29)
    var_31 = lambda x: var_30
    var_32 = 'error_2'
    var_33 = (var_9, var_32)
    var_34 = lambda x: var_33
    var_35 = 'ok_2'
    var_36 = (var_23, var_35)
    var_37 = lambda x: var_36
    var_38 = [var_28, var_31, var_34, var_37]
    var_39 = module_0.check_global_invariants(var_7, var_38)
    var_40 = 'test_error'
    var_41 = (var_9, var_40)
    var_42 = lambda x: var_41
    var_43 = [var_42]
    var_44 = module_0.check_global_invariants(var_7, var_43)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 42
    var_3 = 'hello'
    var_4 = 123
    var_5 = 'test_field'
    var_6 = 'not an int'
    var_7 = set()
    var_8 = 'any value'
    var_9 = None
    var_10 = 'test_field'
    var_11 = None



# Parsed testcases at query #33
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test check_type function for type validation.'
    var_1 = False
    var_2 = 'test_field'
    var_3 = 42
    var_4 = 'test_field'
    var_5 = 42
    var_6 = 'hello'
    var_7 = 'test_field'
    var_8 = 3.14
    var_9 = 'any value'
    var_10 = None
    var_11 = 'my_field'
    var_12 = 123



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = lambda x: x
    var_3 = set()



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'xml'
    var_2 = 'custom_xml_'
    var_3 = 'test_value'
    var_4 = 'csv'
    var_5 = 42
    var_6 = None
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 1
    var_10 = 2
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 3
    var_13 = [var_9, var_10, var_12]



# Parsed testcases at query #37
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'ok'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test_subject'
    var_9 = module_0.check_global_invariants(var_8, var_7)
    var_10 = False
    var_11 = 'error_code_1'
    var_12 = (var_10, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_13]
    var_15 = 'test_subject'
    var_16 = module_0.check_global_invariants(var_15, var_14)
    var_17 = 'error_1'
    var_18 = (var_10, var_17)
    var_19 = lambda x: var_18
    var_20 = (var_15, var_4)
    var_21 = lambda x: var_20
    var_22 = 'error_2'
    var_23 = (var_10, var_22)
    var_24 = lambda x: var_23
    var_25 = [var_19, var_21, var_24]
    var_26 = 'test_subject'
    var_27 = module_0.check_global_invariants(var_26, var_25)
    var_28 = []
    var_29 = module_0.check_global_invariants(var_8, var_28)
    var_30 = 'pass1'
    var_31 = (var_26, var_30)
    var_32 = lambda x: var_31
    var_33 = 'fail1'
    var_34 = (var_10, var_33)
    var_35 = lambda x: var_34
    var_36 = 'pass2'
    var_37 = (var_26, var_36)
    var_38 = lambda x: var_37
    var_39 = [var_32, var_35, var_38]
    var_40 = 'test_subject'
    var_41 = module_0.check_global_invariants(var_40, var_39)
    var_42 = []
    var_43 = 'key'
    var_44 = 'value'
    var_45 = {var_43: var_44}



# Parsed testcases at query #38
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test pmap_field function for creating checked PMap fields.'
    var_1 = False
    var_2 = True
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_2, var_4: var_5}
    var_7 = None
    var_8 = 'x'
    var_9 = 10
    var_10 = {var_8: var_9}



# Parsed testcases at query #40
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)



