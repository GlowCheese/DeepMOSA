####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
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
    var_5 = 'test_fields'
    var_6 = 'base1_field'
    var_7 = 'base1_value'
    var_8 = {var_6: var_7}
    var_9 = 'base2_field'
    var_10 = 'base2_value'
    var_11 = {var_9: var_10}
    var_12 = 'new_field'
    var_13 = 'new_value'
    var_14 = {var_12: var_13}
    var_15 = '_fields'
    var_16 = 'base_field'
    var_17 = 'base_value'
    var_18 = {var_16: var_17}
    var_19 = True
    var_20 = None
    var_21 = (var_19, var_20)
    var_22 = lambda x: var_21
    var_23 = 0
    var_24 = False
    var_25 = lambda x: x
    var_26 = lambda f, v: v
    var_27 = 'custom_field'
    var_28 = '_fields'
    var_29 = 'field1'
    var_30 = 'common_field'
    var_31 = 'value1'
    var_32 = 'base1_value'
    var_33 = {var_29: var_31, var_30: var_32}
    var_34 = 'field2'
    var_35 = 'common_field'
    var_36 = 'value2'
    var_37 = 'base2_value'
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = {}
    var_40 = '_fields'
    var_41 = {}
    var_42 = []
    var_43 = 'fields'
    var_44 = module_0.set_fields(var_41, var_42, var_43)



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = 'base1_field'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'base2_field'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = '_fields'
    var_9 = 'value1'
    var_10 = 'value2'
    var_11 = 'existing_field'
    var_12 = 'existing_value'
    var_13 = {var_11: var_12}
    var_14 = {var_6: var_9, var_7: var_10, var_8: var_13}
    var_15 = set()
    var_16 = True
    var_17 = None
    var_18 = (var_16, var_17)
    var_19 = lambda x: var_18
    var_20 = module_0.object()
    var_21 = False
    var_22 = lambda x: x
    var_23 = lambda f, v: v
    var_24 = module_1._PField(var_15, var_19, var_20, var_21, var_22, var_23)
    var_25 = set()
    var_26 = (var_16, var_17)
    var_27 = lambda x: var_26
    var_28 = module_0.object()
    var_29 = lambda x: x
    var_30 = lambda f, v: v
    var_31 = module_1._PField(var_25, var_27, var_28, var_21, var_29, var_30)
    var_32 = 'existing'
    var_33 = 'value'
    var_34 = {var_32: var_33}
    var_35 = {var_8: var_34}
    var_36 = set()
    var_37 = (var_16, var_17)
    var_38 = lambda x: var_37
    var_39 = module_0.object()
    var_40 = lambda x: x
    var_41 = lambda f, v: v
    var_42 = module_1._PField(var_36, var_38, var_39, var_21, var_40, var_41)
    var_43 = []
    var_44 = module_1.set_fields(var_35, var_43, var_8)
    var_45 = {}
    var_46 = {var_8: var_45}
    var_47 = set()
    var_48 = (var_16, var_17)
    var_49 = lambda x: var_48
    var_50 = module_0.object()
    var_51 = lambda x: x
    var_52 = lambda f, v: v
    var_53 = module_1._PField(var_47, var_49, var_50, var_21, var_51, var_52)
    var_54 = 'common_field'
    var_55 = 'unique_a'
    var_56 = 'from_base_a'
    var_57 = 'a'
    var_58 = {var_54: var_56, var_55: var_57}
    var_59 = 'common_field'
    var_60 = 'unique_b'
    var_61 = 'from_base_b'
    var_62 = 'b'
    var_63 = {var_59: var_61, var_60: var_62}
    var_64 = {var_32: var_33}
    var_65 = {var_8: var_64}
    var_66 = 'regular_attr'
    var_67 = 'another'
    var_68 = {}
    var_69 = 'not_moved'
    var_70 = 'also_not_moved'
    var_71 = {var_8: var_68, var_66: var_69, var_67: var_70}
    var_72 = []
    var_73 = module_1.set_fields(var_71, var_72, var_8)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = 5
    var_3 = -1
    var_4 = 10
    var_5 = 42
    var_6 = lambda : var_5
    var_7 = var_0.initial
    var_8 = callable(var_7)
    var_9 = True
    var_10 = module_0.field(mandatory=var_9)
    var_11 = 123
    var_12 = 'json'
    var_13 = 'hello'
    var_14 = 'test'
    var_15 = ''
    var_16 = 'any'
    var_17 = 'TEST'
    var_18 = 'int'
    var_19 = module_0.field(var_18)
    var_20 = var_19.type
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = var_19.type
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = 'field1'
    var_25 = 'regular_attr'
    var_26 = False
    var_27 = 'value'
    var_28 = []
    var_29 = '_precord_fields'
    var_30 = 123
    var_31 = module_0.field(var_30)
    var_32 = 'not an int'
    var_33 = 'not callable'
    var_34 = module_0.field(invariant=var_33)
    var_35 = 'not callable'
    var_36 = module_0.field(factory=var_35)
    var_37 = 'not callable'
    var_38 = module_0.field(serializer=var_37)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = None
    var_7 = False
    var_8 = {}
    var_9 = {}
    var_10 = {}
    var_11 = 'test'
    var_12 = 'data'
    var_13 = {var_1: var_11, var_2: var_12}
    var_14 = {}



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_6: var_7}



# Parsed testcases at query #7
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'OK'
    var_10 = (var_5, var_9)
    var_11 = lambda x: var_10
    var_12 = 123
    var_13 = (var_5, var_12)
    var_14 = lambda x: var_13
    var_15 = [var_8, var_11, var_14]
    var_16 = module_0.check_global_invariants(var_2, var_15)
    var_17 = False
    var_18 = 'error1'
    var_19 = (var_17, var_18)
    var_20 = lambda x: var_19
    var_21 = [var_20]
    var_22 = module_0.check_global_invariants(var_2, var_21)
    var_23 = 'ok1'
    var_24 = (var_5, var_23)
    var_25 = lambda x: var_24
    var_26 = (var_17, var_18)
    var_27 = lambda x: var_26
    var_28 = 'ok2'
    var_29 = (var_5, var_28)
    var_30 = lambda x: var_29
    var_31 = 'error2'
    var_32 = (var_17, var_31)
    var_33 = lambda x: var_32
    var_34 = 'error3'
    var_35 = (var_17, var_34)
    var_36 = lambda x: var_35
    var_37 = [var_25, var_27, var_30, var_33, var_36]
    var_38 = module_0.check_global_invariants(var_2, var_37)
    var_39 = (var_17, var_12)
    var_40 = lambda x: var_39
    var_41 = 'string_error'
    var_42 = (var_17, var_41)
    var_43 = lambda x: var_42
    var_44 = 'code'
    var_45 = 'dict_error'
    var_46 = {var_44: var_45}
    var_47 = (var_17, var_46)
    var_48 = lambda x: var_47
    var_49 = [var_40, var_43, var_48]
    var_50 = module_0.check_global_invariants(var_2, var_49)
    var_51 = None
    var_52 = 'key'
    var_53 = 'value'
    var_54 = {var_52: var_53}
    var_55 = module_0.check_global_invariants(var_54, var_49)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'plain_string'
    var_2 = 'xml'
    var_3 = 'data'
    var_4 = None
    var_5 = 42
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = lambda x: x
    var_5 = False
    var_6 = ()
    var_7 = lambda x: x
    var_8 = True
    var_9 = ()
    var_10 = lambda x: x
    var_11 = ()
    var_12 = ()
    var_13 = ()
    var_14 = ()
    var_15 = ()
    var_16 = ()
    var_17 = ()



# Parsed testcases at query #10
#--------------------------


import builtins as module_0

def test_case_0():
    var_0 = 'MockRecord'
    var_1 = 'Field'
    var_2 = ()
    var_3 = 'type'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = 'test_field'
    var_7 = 'any_value'
    var_8 = 123
    var_9 = module_0.object()
    var_10 = ()
    var_11 = ()
    var_12 = ()
    var_13 = 'test_field'
    var_14 = ()
    var_15 = ()
    var_16 = {var_3: var_15}
    var_17 = 'test_field'
    var_18 = 'any_value'
    var_19 = 'StringType'
    var_20 = 'Field'
    var_21 = ()
    var_22 = 'type'
    var_23 = (var_19,)
    var_24 = {var_22: var_23}
    var_25 = 'test_field'
    var_26 = 'test_field'
    var_27 = ()



# Parsed testcases at query #11
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'
    var_2 = 'json'
    var_3 = 'regular_string'
    var_4 = 'xml'
    var_5 = 'data'
    var_6 = 'custom_xml_{}'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'xml'
    var_5 = 'data'
    var_6 = 'test'
    var_7 = None
    var_8 = 2
    var_9 = lambda fmt, val: val * var_8
    var_10 = 'any'
    var_11 = 5
    var_12 = module_0.serialize(var_9, var_10, var_11)
    assert var_12 == 10
    var_13 = lambda fmt, val: len(val)
    var_14 = 1
    var_15 = 3
    var_16 = [var_14, var_8, var_15]
    var_17 = module_0.serialize(var_13, var_10, var_16)
    assert var_17 == 3



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = lambda x: x
    var_3 = lambda x, ignore_extra=False: x
    var_4 = lambda x, ignore_extra=False: x
    var_5 = lambda x, ignore_extra=False: x
    var_6 = lambda x, ignore_extra=False: x



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = lambda x: x
    var_5 = False
    var_6 = ()
    var_7 = lambda x: x
    var_8 = True
    var_9 = ()
    var_10 = lambda x: x
    var_11 = ()
    var_12 = ()
    var_13 = ()
    var_14 = ()
    var_15 = ()



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'plain_value'
    var_2 = 'xml'
    var_3 = 'data'
    var_4 = 'yaml'
    var_5 = None



# Parsed testcases at query #16
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
    var_9 = 'not a dict'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'test_value'
    var_2 = 'xml'
    var_3 = 'data'
    var_4 = 'yaml'
    var_5 = 'yaml:{}'
    var_6 = None
    var_7 = 42
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 'binary'
    var_13 = 'test'



# Parsed testcases at query #18
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda _: var_2
    var_4 = module_0.object()
    var_5 = False
    var_6 = lambda x: x
    var_7 = lambda _, value: value
    var_8 = 'test_field'
    var_9 = (var_0, var_1)
    var_10 = lambda _: var_9
    var_11 = module_0.object()
    var_12 = lambda x: x
    var_13 = lambda _, value: value
    var_14 = (var_0, var_1)
    var_15 = lambda _: var_14
    var_16 = module_0.object()
    var_17 = lambda x: x
    var_18 = lambda _, value: value
    var_19 = 'test_field'
    var_20 = ()
    var_21 = (var_0, var_1)
    var_22 = lambda _: var_21
    var_23 = module_0.object()
    var_24 = lambda x: x
    var_25 = lambda _, value: value
    var_26 = module_1._PField(var_20, var_22, var_23, var_5, var_24, var_25)
    var_27 = (var_0, var_1)
    var_28 = lambda _: var_27
    var_29 = module_0.object()
    var_30 = lambda x: x
    var_31 = lambda _, value: value
    var_32 = None
    var_33 = 'test_field'
    var_34 = 'MockType'
    var_35 = (var_34,)
    var_36 = True
    var_37 = None
    var_38 = (var_36, var_37)
    var_39 = lambda _: var_38
    var_40 = module_0.object()
    var_41 = False
    var_42 = lambda x: x
    var_43 = lambda _, value: value
    var_44 = module_1._PField(var_35, var_39, var_40, var_41, var_42, var_43)
    var_45 = 'test_field'
    var_46 = 'test_field'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._field_common as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'MockDestinationClass'
    var_1 = None
    var_2 = False
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = 'field3'
    var_6 = 'any value'
    var_7 = 'field4'
    var_8 = set()
    var_9 = module_0._PField(var_8, var_1, var_1, var_2, var_1, var_1)
    var_10 = module_1.object()
    var_11 = 'field5'
    var_12 = 'AllowedType1'
    var_13 = {var_12}
    var_14 = module_0._PField(var_13, var_1, var_1, var_2, var_1, var_1)
    var_15 = 'field6'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'test_field'
    var_2 = 123
    var_3 = 'test_field'
    var_4 = None
    var_5 = 'any_value'
    var_6 = 'test_field'
    var_7 = ()
    var_8 = 'any_value'
    var_9 = 'test_field'
    var_10 = 'not_an_int'
    var_11 = 'test_field'
    var_12 = 'not_a_number'
    var_13 = 'test_field'
    var_14 = 'test_field'
    var_15 = 'builtins.str'
    var_16 = (var_15,)
    var_17 = 'test_string'
    var_18 = 'test_field'



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'test_value'
    var_2 = 'xml'
    var_3 = 'data'
    var_4 = None
    var_5 = 'binary'
    var_6 = 42
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = 'yaml'



# Parsed testcases at query #23
#--------------------------


import builtins as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 42
    var_2 = 'test_field'
    var_3 = 'not_an_int'
    var_4 = 'multi_field'
    var_5 = 'string'
    var_6 = 'no_type_field'
    var_7 = 'any_value'
    var_8 = 123
    var_9 = module_0.object()
    var_10 = 'checked_field'
    var_11 = 'error_field'
    var_12 = 'wrong'
    var_13 = ()
    var_14 = 'empty_field'
    var_15 = 'anything'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'MockDestinationClass'
    var_1 = None
    var_2 = False
    var_3 = 'test_field'
    var_4 = 'test_field'
    var_5 = set()
    var_6 = module_0._PField(var_5, var_1, var_1, var_2, var_1, var_1)
    var_7 = 'any_value'
    var_8 = 'test_field'
    var_9 = 'test_field'
    var_10 = 'test_field'



# Parsed testcases at query #25
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)
    var_5 = module_0.check_global_invariants(var_2, var_3)
    var_6 = module_0.check_global_invariants(var_2, var_3)
    var_7 = module_0.check_global_invariants(var_2, var_3)
    var_8 = module_0.check_global_invariants(var_2, var_3)



# Parsed testcases at query #26
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)
    var_5 = module_0.check_global_invariants(var_2, var_0)
    var_6 = module_0.check_global_invariants(var_2, var_0)
    var_7 = module_0.check_global_invariants(var_2, var_0)
    var_8 = 42



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'other_param'
    var_3 = 'ignore_extra'
    var_4 = ()



# Parsed testcases at query #28
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)
    var_5 = module_0.check_global_invariants(var_2, var_0)
    var_6 = module_0.check_global_invariants(var_2, var_0)
    var_7 = module_0.check_global_invariants(var_2, var_0)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.check_global_invariants(var_11, var_0)



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'test_value'
    var_2 = 'xml'
    var_3 = 'data'
    var_4 = 'yaml'
    var_5 = None
    var_6 = 42
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
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
    var_5 = 'test_fields'
    var_6 = 'regular'
    var_7 = 'value'
    var_8 = 'pfields'
    var_9 = 'new_field'
    var_10 = 'new'
    var_11 = 'all_fields'
    var_12 = {}
    var_13 = 'empty_fields'
    var_14 = 'field'
    var_15 = 'test'
    var_16 = ()
    var_17 = 'no_base_fields'
    var_18 = module_0.set_fields(var_12, var_16, var_17)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'field2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = 'field3'
    var_7 = '_fields'
    var_8 = 'value3'
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = 'existing'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = 'new_field'
    var_17 = 'another_field'
    var_18 = 'value1'
    var_19 = 'value2'
    var_20 = {var_11: var_18, var_12: var_19}
    var_21 = []
    var_22 = module_0.set_fields(var_20, var_21, var_7)
    var_23 = 'field1'
    var_24 = 'common'
    var_25 = 'from_A'
    var_26 = 'A'
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = 'field2'
    var_29 = 'common'
    var_30 = 'from_B'
    var_31 = 'B'
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = 'from_dct'
    var_34 = {var_6: var_33}
    var_35 = {var_11: var_18}
    var_36 = 'base_field'
    var_37 = 'base_value'
    var_38 = {var_36: var_37}
    var_39 = 'dct_field'
    var_40 = 'dct_value'
    var_41 = {var_39: var_40}



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'json'
    var_2 = 'regular_string'
    var_3 = 'xml'
    var_4 = 'data'
    var_5 = None
    var_6 = 42
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]



# Parsed testcases at query #4
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)
    var_5 = module_0.check_global_invariants(var_2, var_0)
    var_6 = module_0.check_global_invariants(var_2, var_0)
    var_7 = module_0.check_global_invariants(var_2, var_0)
    var_8 = 42



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_1, var_3: var_4}
    var_6 = None
    var_7 = 'key1'
    var_8 = 'key2'
    var_9 = 'value1'
    var_10 = 'value2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'test'
    var_13 = 123
    var_14 = {var_12: var_13}



# Parsed testcases at query #6
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'field2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = 'field3'
    var_7 = '_fields'
    var_8 = 'value3'
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'field1'
    var_12 = module_0._PField(var_11)
    var_13 = 'field2'
    var_14 = module_0._PField(var_13)
    var_15 = {}
    var_16 = {var_11: var_12, var_13: var_14, var_7: var_15}
    var_17 = ()
    var_18 = module_0.set_fields(var_16, var_17, var_7)
    var_19 = 'grand_field'
    var_20 = 'grand_value'
    var_21 = {var_19: var_20}
    var_22 = 'parent_field'
    var_23 = 'parent_value'
    var_24 = {var_22: var_23}
    var_25 = 'child_field'
    var_26 = 'child_value'
    var_27 = {}
    var_28 = {var_25: var_26, var_7: var_27}
    var_29 = module_0.set_fields(var_28, var_17, var_7)
    var_30 = 'value1'
    var_31 = {}
    var_32 = {var_11: var_30, var_7: var_31}
    var_33 = ()
    var_34 = module_0.set_fields(var_32, var_33, var_7)
    var_35 = 'regular_field'
    var_36 = 'regular_value'
    var_37 = {}
    var_38 = {var_35: var_36, var_7: var_37}
    var_39 = ()
    var_40 = module_0.set_fields(var_38, var_39, var_7)
    var_41 = {}
    var_42 = {var_11: var_30, var_7: var_41}
    var_43 = module_0.set_fields(var_42, var_39, var_7)
    var_44 = 'base_field'
    var_45 = 'base_value'
    var_46 = {var_44: var_45}
    var_47 = 'dct_field'
    var_48 = 'dct_value'
    var_49 = {}
    var_50 = {var_47: var_48, var_7: var_49}
    var_51 = module_0.set_fields(var_50, var_39, var_7)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_6: var_7}
    var_9 = 'one'
    var_10 = 'two'
    var_11 = {var_0: var_9, var_1: var_10}



# Parsed testcases at query #8
#--------------------------


import builtins as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 42
    var_2 = 'hello'
    var_3 = ()
    var_4 = 'any_value'
    var_5 = 123
    var_6 = None
    var_7 = 'test_field'
    var_8 = 'string_value'
    var_9 = 'test_field'
    var_10 = 'test_field'
    var_11 = 3.14
    var_12 = ()
    var_13 = module_0.object()



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'test_value'
    var_2 = 'data'
    var_3 = 'xml'
    var_4 = 'test'
    var_5 = 'key'
    var_6 = 'number'
    var_7 = 'value'
    var_8 = 42
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'yaml'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = None
    var_6 = 'x'
    var_7 = 'y'
    var_8 = {var_6: var_0, var_7: var_3}
    var_9 = {var_1: var_0}
    var_10 = 'my_map'
    var_11 = {var_1: var_0, var_2: var_3}
    var_12 = {var_10: var_11}



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = {}
    var_3 = None
    var_4 = 'a'
    var_5 = {var_1: var_4}
    var_6 = {var_1: var_4}
    var_7 = {}
    var_8 = {}
    var_9 = {}
    var_10 = {}
    var_11 = {}
    var_12 = 2
    var_13 = 'one'
    var_14 = 'two'
    var_15 = {var_1: var_13, var_12: var_14}



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'test_value'
    var_2 = 'test'
    var_3 = 'xml'
    var_4 = None
    var_5 = 42
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]



# Parsed testcases at query #13
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)
    var_3 = module_0.check_global_invariants(var_0, var_1)
    var_4 = module_0.check_global_invariants(var_0, var_1)
    var_5 = module_0.check_global_invariants(var_0, var_1)
    var_6 = None
    var_7 = 'hello'
    var_8 = ''
    var_9 = module_0.check_global_invariants(var_8, var_5)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'ignore_extra'
    var_3 = ()



# Parsed testcases at query #15
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
    var_10 = 5
    var_11 = -1
    var_12 = 10
    var_13 = 42
    var_14 = lambda : var_13
    var_15 = var_0.initial
    var_16 = callable(var_15)
    var_17 = True
    var_18 = module_0.field(mandatory=var_17)
    var_19 = 123
    var_20 = 'json'
    var_21 = 0
    var_22 = 'Must be non-negative'
    var_23 = lambda x: (x >= var_21, var_22)
    var_24 = lambda x: float(x)
    var_25 = lambda fmt, val: str(val)
    var_26 = var_18.type
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = -1
    var_29 = 'any'
    var_30 = 3.14
    var_31 = 'not callable'
    var_32 = module_0.field(invariant=var_31)
    var_33 = 'not callable'
    var_34 = module_0.field(factory=var_33)
    var_35 = 'not callable'
    var_36 = module_0.field(serializer=var_35)
    var_37 = 'not a type'
    var_38 = 123
    var_39 = module_0.field(var_3)
    var_40 = 'string'
    var_41 = 'int'
    var_42 = module_0.field(var_41)
    var_43 = var_42.type
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = var_42.type
    var_46 = len(var_45)
    assert var_46 == 1



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = None



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'test_value'
    var_2 = 'xml'
    var_3 = 'data'
    var_4 = 'yaml'
    var_5 = None
    var_6 = 42
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]



# Parsed testcases at query #18
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = 'OK'
    var_8 = (var_3, var_7)
    var_9 = lambda x: var_8
    var_10 = 200
    var_11 = (var_3, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_6, var_9, var_12]
    var_14 = module_0.check_global_invariants(var_2, var_13)
    var_15 = 'value'
    var_16 = 5
    var_17 = {var_15: var_16}
    var_18 = False
    var_19 = 'Value too small'
    var_20 = (var_18, var_19)
    var_21 = lambda x: var_20
    var_22 = [var_21]
    var_23 = module_0.check_global_invariants(var_17, var_22)
    var_24 = 10
    var_25 = {var_15: var_24}
    var_26 = 'Error1'
    var_27 = (var_18, var_26)
    var_28 = lambda x: var_27
    var_29 = (var_3, var_7)
    var_30 = lambda x: var_29
    var_31 = 'Error2'
    var_32 = (var_18, var_31)
    var_33 = lambda x: var_32
    var_34 = 'Error3'
    var_35 = (var_18, var_34)
    var_36 = lambda x: var_35
    var_37 = [var_28, var_30, var_33, var_36]
    var_38 = module_0.check_global_invariants(var_25, var_37)
    var_39 = 'data'
    var_40 = {var_39: var_1}
    var_41 = 404
    var_42 = (var_18, var_41)
    var_43 = lambda x: var_42
    var_44 = 'Not found'
    var_45 = (var_18, var_44)
    var_46 = lambda x: var_45
    var_47 = (var_3, var_4)
    var_48 = lambda x: var_47
    var_49 = [var_43, var_46, var_48]
    var_50 = module_0.check_global_invariants(var_40, var_49)
    var_51 = 'empty'
    var_52 = {var_51: var_3}
    var_53 = []
    var_54 = module_0.check_global_invariants(var_52, var_53)
    var_55 = 'count'
    var_56 = 15
    var_57 = {var_55: var_56}
    var_58 = 'Count too low'
    var_59 = lambda x: (x[var_55] > var_24, var_58)
    var_60 = 20
    var_61 = 'Count too high'
    var_62 = lambda x: (x[var_55] < var_60, var_61)
    var_63 = [var_59, var_62]
    var_64 = module_0.check_global_invariants(var_57, var_63)
    var_65 = {var_55: var_16}
    var_66 = lambda x: (x[var_55] > var_24, var_58)
    var_67 = lambda x: (x[var_55] < var_60, var_61)
    var_68 = [var_66, var_67]
    var_69 = module_0.check_global_invariants(var_65, var_68)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'test_value'
    var_2 = 'test'
    var_3 = 'xml'
    var_4 = None
    var_5 = 42
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'regular_value'
    var_2 = 'xml'
    var_3 = 'data'
    var_4 = None
    var_5 = 'binary'
    var_6 = 42
    var_7 = 'yaml'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = {}
    var_3 = None
    var_4 = 2
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_1: var_5, var_4: var_6}
    var_8 = {var_1: var_5, var_4: var_6}
    var_9 = {}
    var_10 = {}
    var_11 = {}
    var_12 = {}



# Parsed testcases at query #22
#--------------------------


import pyrsistent._field_common as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'value'
    var_2 = 'test'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'OK'
    var_10 = (var_5, var_9)
    var_11 = lambda x: var_10
    var_12 = 200
    var_13 = (var_5, var_12)
    var_14 = lambda x: var_13
    var_15 = [var_8, var_11, var_14]
    var_16 = module_0.check_global_invariants(var_4, var_15)
    var_17 = 'data'
    var_18 = 'invalid'
    var_19 = {var_17: var_18}
    var_20 = False
    var_21 = 'ERROR_001'
    var_22 = (var_20, var_21)
    var_23 = lambda x: var_22
    var_24 = [var_23]
    var_25 = module_0.check_global_invariants(var_19, var_24)
    var_26 = -5
    var_27 = {var_1: var_26}
    var_28 = 'NEGATIVE'
    var_29 = (var_20, var_28)
    var_30 = lambda x: var_29
    var_31 = 'POSITIVE'
    var_32 = (var_5, var_31)
    var_33 = lambda x: var_32
    var_34 = 'TOO_SMALL'
    var_35 = (var_20, var_34)
    var_36 = lambda x: var_35
    var_37 = 'INVALID_TYPE'
    var_38 = (var_20, var_37)
    var_39 = lambda x: var_38
    var_40 = [var_30, var_33, var_36, var_39]
    var_41 = module_0.check_global_invariants(var_27, var_40)
    var_42 = module_1.object()
    var_43 = (var_5, var_6)
    var_44 = lambda x: var_43
    var_45 = 404
    var_46 = (var_20, var_45)
    var_47 = lambda x: var_46
    var_48 = 'Not Found'
    var_49 = (var_20, var_48)
    var_50 = lambda x: var_49
    var_51 = ''
    var_52 = (var_5, var_51)
    var_53 = lambda x: var_52
    var_54 = 'err'
    var_55 = 'code'
    var_56 = (var_54, var_55)
    var_57 = (var_20, var_56)
    var_58 = lambda x: var_57
    var_59 = [var_44, var_47, var_50, var_53, var_58]
    var_60 = module_0.check_global_invariants(var_42, var_59)
    var_61 = 'any'
    var_62 = 'thing'
    var_63 = {var_61: var_62}
    var_64 = []
    var_65 = module_0.check_global_invariants(var_63, var_64)
    var_66 = 'NO_DATA'
    var_67 = lambda x: (hasattr(x, var_17), var_66)
    var_68 = 'NO_NAME'
    var_69 = lambda x: (hasattr(x, var_60), var_68)
    var_70 = 3
    var_71 = 'WRONG_SIZE'
    var_72 = lambda x: (len(x.data) == var_70, var_71)
    var_73 = [var_67, var_69, var_72]
    var_74 = module_0.check_global_invariants(var_63, var_73)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)
    var_3 = module_0.check_global_invariants(var_0, var_1)
    var_4 = module_0.check_global_invariants(var_0, var_1)
    var_5 = module_0.check_global_invariants(var_0, var_1)
    var_6 = module_0.check_global_invariants(var_0, var_1)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'ignore_extra'
    var_3 = True
    var_4 = 'other_param'
    var_5 = True
    var_6 = True



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'json'
    var_1 = 'test_string'
    var_2 = 'xml'
    var_3 = 'data'
    var_4 = 'yaml'
    var_5 = None
    var_6 = 42
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]



# Parsed testcases at query #26
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)
    var_5 = module_0.check_global_invariants(var_2, var_0)
    var_6 = module_0.check_global_invariants(var_2, var_0)
    var_7 = module_0.check_global_invariants(var_2, var_0)
    var_8 = 42



