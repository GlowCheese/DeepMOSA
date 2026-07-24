####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'ro'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = True



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'req_field'
    var_2 = 'opt_field'
    var_3 = 'ro_field'
    var_4 = False
    var_5 = 'val1'
    var_6 = 'val2'
    var_7 = 'val3'
    var_8 = {var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = {var_1: var_5, var_3: var_7}
    var_10 = None
    var_11 = 'May not be null.'
    var_12 = 'not'
    var_13 = 'a'
    var_14 = 'dict'
    var_15 = [var_12, var_13, var_14]
    var_16 = 'Must be an object.'
    var_17 = 123
    var_18 = 'req_field'
    var_19 = 'value'
    var_20 = 'val1'
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = 'All object keys must be strings.'
    var_23 = 'opt_field'
    var_24 = 'val2'
    var_25 = {var_23: var_24}
    var_26 = 'This field is required.'
    var_27 = 'Child error'
    var_28 = 'child_err'
    var_29 = []
    var_30 = module_0.Message(text=var_27, code=var_28, index=var_29)
    var_31 = [var_30]
    var_32 = 'child'
    var_33 = 'child'
    var_34 = 'bad_value'
    var_35 = {var_33: var_34}
    var_36 = 'child: Child error'
    var_37 = True



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'req_field'
    var_1 = 'opt_field'
    var_2 = False
    var_3 = True
    var_4 = 'def_field'
    var_5 = 'ro_field'
    var_6 = {}
    var_7 = module_0.Schema(var_6)
    var_8 = {}
    var_9 = module_0.Schema(var_8)



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'read_only_field'
    var_3 = 18
    var_4 = 'John'
    var_5 = 30
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = {var_0: var_4}
    var_8 = None
    var_9 = 'May not be null'
    var_10 = 'not'
    var_11 = 'a'
    var_12 = 'dict'
    var_13 = [var_10, var_11, var_12]
    var_14 = 'Must be an object'
    var_15 = 123
    var_16 = 'integer key'
    var_17 = {var_15: var_16}
    var_18 = 'All object keys must be strings'
    var_19 = 'age'
    var_20 = 25
    var_21 = {var_19: var_20}
    var_22 = 'This field is required'
    var_23 = 'error_field'
    var_24 = None
    var_25 = 'Child error'
    var_26 = 'child_err'
    var_27 = []
    var_28 = module_0.Message(text=var_25, code=var_26, index=var_27)
    var_29 = [var_28]
    var_30 = 'error_field'
    var_31 = 'bad_data'
    var_32 = {var_30: var_31}
    var_33 = 'error_field: Child error'
    var_34 = True



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'target_key'
    var_1 = True
    var_2 = None
    var_3 = False
    var_4 = None
    var_5 = 'May not be null'
    var_6 = 'id'
    var_7 = {var_6: var_1}
    var_8 = 'Invalid value'
    var_9 = [var_8]



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'ro'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = True



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'val1'
    var_1 = None
    var_2 = 'val2'
    var_3 = 'Child error'
    var_4 = 'child_err'
    var_5 = module_0.Message(text=var_3, code=var_4)
    var_6 = 'req_field'
    var_7 = 'def_field'
    var_8 = 'ro_field'
    var_9 = 'fail_field'
    var_10 = None
    var_11 = 'May not be null'
    var_12 = 'not a dict'
    var_13 = 'Must be an object'
    var_14 = 123
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = 'All object keys must be strings'
    var_18 = 'def_field'
    var_19 = 'some_val'
    var_20 = {var_18: var_19}
    var_21 = 'This field is required'
    var_22 = 'bad_data'
    var_23 = {var_6: var_18, var_9: var_22}
    var_24 = 'fail_field.Child error'
    var_25 = 'val4'
    var_26 = {var_6: var_18, var_7: var_20, var_9: var_25}
    var_27 = {var_6: var_18, var_9: var_25}



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'val1'
    var_1 = None
    var_2 = 'val2'
    var_3 = 'val3'
    var_4 = 'Child error'
    var_5 = 'error_code'
    var_6 = module_0.Message(text=var_4, code=var_5)
    var_7 = 'required_field'
    var_8 = 'optional_field'
    var_9 = 'readonly_field'
    var_10 = 'failing_field'
    var_11 = {var_7: var_0, var_8: var_2, var_9: var_3}
    var_12 = None
    var_13 = 'May not be null'
    var_14 = 'not'
    var_15 = 'a'
    var_16 = 'dict'
    var_17 = [var_14, var_15, var_16]
    var_18 = 'Must be an object'
    var_19 = 123
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = 'All object keys must be strings'
    var_23 = {var_8: var_16}
    var_24 = 'This field is required'
    var_25 = {var_7: var_14}
    var_26 = 'bad_data'
    var_27 = {var_7: var_14, var_10: var_26}
    var_28 = 'failing_field.error'



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'ro'
    var_3 = True
    var_4 = {}
    var_5 = module_0.Schema(var_4)
    var_6 = False



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = 'id'
    var_2 = 'name'
    var_3 = 1
    var_4 = 'John'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = None
    var_8 = False
    var_9 = None
    var_10 = 'May not be null'
    var_11 = 'Invalid field'
    var_12 = 'error_code'
    var_13 = module_0.Message(text=var_11, code=var_12)
    var_14 = [var_13]
    var_15 = 'invalid'
    var_16 = 'data'
    var_17 = {var_15: var_16}



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'user_schema'
    var_1 = 'id'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = True
    var_6 = None
    var_7 = False
    var_8 = None
    var_9 = 'May not be null.'
    var_10 = 'Target error'
    var_11 = [var_10]
    var_12 = 'invalid'
    var_13 = 'data'
    var_14 = {var_12: var_13}



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = False
    var_3 = 'id'
    var_4 = True
    var_5 = 'role'
    var_6 = {}
    var_7 = module_0.Schema(var_6)



# Parsed testcases at query #13
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'default_val'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3}
    var_7 = False
    var_8 = None
    var_9 = 'May not be null'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = 'Must be an object'
    var_15 = 123
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'All object keys must be strings'
    var_19 = True
    var_20 = 'required_key'
    var_21 = 'other_key'
    var_22 = 1
    var_23 = {var_21: var_22}
    var_24 = 'This field is required'
    var_25 = 'Child Error'
    var_26 = 'child_err'
    var_27 = 'sub_key'
    var_28 = [var_27]
    var_29 = module_0.Message(text=var_25, code=var_26, index=var_28)
    var_30 = None
    var_31 = [var_29]
    var_32 = (var_30, var_31)
    var_33 = 'child'
    var_34 = 'child'
    var_35 = 'bad_data'
    var_36 = {var_34: var_35}
    var_37 = 'child: Child Error'
    var_38 = True
    var_39 = 'ro'
    var_40 = 'some_val'
    var_41 = {var_39: var_40}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'ro'
    var_3 = True



# Parsed testcases at query #15
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'ro'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = True



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = 'May not be null'
    var_3 = None
    var_4 = 'not'
    var_5 = 'a'
    var_6 = 'dict'
    var_7 = [var_4, var_5, var_6]
    var_8 = 'Must be an object'
    var_9 = 123
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 'All object keys must be string'
    var_13 = 'age'
    var_14 = 30
    var_15 = {var_13: var_14}
    var_16 = 'This field is required'
    var_17 = 'valid_child'
    var_18 = 'default_val'
    var_19 = 'age'
    var_20 = 'metadata'
    var_21 = 'read_only_field'
    var_22 = 25
    var_23 = True
    var_24 = 'John'
    var_25 = 'key'
    var_26 = 'val'
    var_27 = {var_25: var_26}
    var_28 = 'should_be_ignored'
    var_29 = {var_13: var_24, var_20: var_27, var_21: var_28}
    var_30 = 'Child error'
    var_31 = 'child_err'
    var_32 = []
    var_33 = module_0.Message(text=var_30, code=var_31, index=var_32)
    var_34 = 'child_err_prefix:Child error'
    var_35 = []
    var_36 = module_0.Message(text=var_34, code=var_31, index=var_35)
    var_37 = 'metadata'
    var_38 = 'bad'
    var_39 = 'data'
    var_40 = {var_38: var_39}
    var_41 = {var_37: var_40}



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'default_val'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'val'
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = {var_1: var_3}
    var_6 = True
    var_7 = None
    var_8 = False
    var_9 = None
    var_10 = 'May not be null'
    var_11 = 'not'
    var_12 = 'a'
    var_13 = 'dict'
    var_14 = [var_11, var_12, var_13]
    var_15 = 'Must be an object'
    var_16 = 123
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = 'All object keys must be string'
    var_20 = {}
    var_21 = 'This field is required'
    var_22 = 'Child Error'
    var_23 = 'child_err'
    var_24 = [var_17]
    var_25 = module_0.Message(text=var_22, code=var_23, index=var_24)
    var_26 = 'a'
    var_27 = 'bad_data'
    var_28 = {var_26: var_27}
    var_29 = 'a: Child Error'
    var_30 = 'ro'
    var_31 = 'other'
    var_32 = {var_31: var_6}



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'my_ref'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = True
    var_6 = None
    var_7 = False
    var_8 = None
    var_9 = 'May not be null'
    var_10 = 'Error in target'
    var_11 = 'err'
    var_12 = 'bad'
    var_13 = 'data'
    var_14 = {var_12: var_13}



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'default_val'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3}
    var_7 = False
    var_8 = None
    var_9 = 'May not be null'
    var_10 = True
    var_11 = None
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = 'Must be an object'
    var_17 = 123
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = 'All object keys must be strings'
    var_21 = 'b'
    var_22 = 1
    var_23 = {var_21: var_22}
    var_24 = 'This field is required'
    var_25 = 'Child Error'
    var_26 = 'child_err'
    var_27 = []
    var_28 = module_0.Message(text=var_25, code=var_26, index=var_27)
    var_29 = [var_28]
    var_30 = 'a'
    var_31 = 'bad_value'
    var_32 = {var_30: var_31}
    var_33 = 'a: Child Error'
    var_34 = True
    var_35 = {}



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'string_field'
    var_1 = 'int_field'
    var_2 = 'bool_field'
    var_3 = 'input_string'
    var_4 = 10
    var_5 = False
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'value1'
    var_8 = 123
    var_9 = True
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = 'obj_s'
    var_12 = 99
    var_13 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_14 = None
    var_15 = 'only_one'
    var_16 = {var_0: var_15}
    var_17 = {var_0: var_7}
    var_18 = []



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = len(var_0)
    assert var_1 == 1



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'meta'
    var_4 = 'John'
    var_5 = 30
    var_6 = 'ignored'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_1: var_4}
    var_9 = None
    var_10 = 'May not be null'
    var_11 = 'not'
    var_12 = 'a'
    var_13 = 'dict'
    var_14 = [var_11, var_12, var_13]
    var_15 = 'Must be an object'
    var_16 = 123
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = 'All object keys must be strings'
    var_20 = 'age'
    var_21 = 30
    var_22 = {var_20: var_21}
    var_23 = 'This field is required'
    var_24 = 'Invalid name'
    var_25 = 'err'
    var_26 = []
    var_27 = module_0.Message(text=var_24, code=var_25, index=var_26)
    var_28 = [var_27]
    var_29 = (var_20, var_28)
    var_30 = 'name'
    var_31 = 'age'
    var_32 = 'Invalid'
    var_33 = 30
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = 'name: Invalid name'



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'user'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = False
    var_7 = module_0.Reference(var_1, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = 'May not be null'
    var_11 = 'id'
    var_12 = 'name'
    var_13 = 'John'
    var_14 = {var_11: var_9, var_12: var_13}
    var_15 = var_3.validate(var_14)
    var_16 = 'Invalid user'
    var_17 = 'error'
    var_18 = 'invalid'
    var_19 = 'data'
    var_20 = {var_18: var_19}
    var_21 = var_3.validate(var_20)



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'val_a'
    var_1 = 'default_b'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0}
    var_5 = False
    var_6 = None
    var_7 = 'May not be null'
    var_8 = True
    var_9 = None
    var_10 = 'not'
    var_11 = 'a'
    var_12 = 'dict'
    var_13 = [var_10, var_11, var_12]
    var_14 = 'Must be an object'
    var_15 = 123
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'All object keys must be strings'
    var_19 = 'b'
    var_20 = 1
    var_21 = {var_19: var_20}
    var_22 = 'This field is required'
    var_23 = 'child_err'
    var_24 = []
    var_25 = module_0.Message(text=var_23, code=var_23, index=var_24)
    var_26 = [var_25]
    var_27 = module_0.ValidationError(messages=var_26)
    var_28 = 'a'
    var_29 = 'bad_value'
    var_30 = {var_28: var_29}
    var_31 = 'a: child_err'
    var_32 = 'ro_value'
    var_33 = {var_30: var_32}



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'ro'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = False
    var_8 = lambda : var_7
    var_9 = lambda : var_7



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = False
    var_3 = None
    var_4 = 'id'
    var_5 = True
    var_6 = 'role'
    var_7 = 'user'
    var_8 = 'guest'
    var_9 = {}
    var_10 = module_0.Schema(var_9)



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'my_ref'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    assert var_6 == 'valid_data'
    var_7 = {var_3: var_4}
    var_8 = True
    var_9 = module_0.Reference(var_1, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None
    var_12 = False
    var_13 = module_0.Reference(var_1, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = 'null'
    var_17 = 'May not be null'
    var_18 = 'Target error'
    var_19 = 'error_code'
    var_20 = module_1.Message(text=var_18, code=var_19)
    var_21 = [var_20]
    var_22 = module_1.ValidationError(messages=var_21)
    var_23 = 'bad'
    var_24 = 'data'
    var_25 = {var_23: var_24}
    var_26 = var_2.validate(var_25)



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'required_field'
    var_1 = 'optional_field'
    var_2 = 'readonly_field'
    var_3 = True
    var_4 = {}
    var_5 = module_0.Schema(var_4)



# Parsed testcases at query #15
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 18
    var_3 = 'Alice'
    var_4 = 25
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'Bob'
    var_7 = {var_0: var_6}
    var_8 = None
    var_9 = 'May not be null'
    var_10 = True
    var_11 = None
    var_12 = 'not'
    var_13 = 'a'
    var_14 = 'dict'
    var_15 = [var_12, var_13, var_14]
    var_16 = 'Must be an object'
    var_17 = 123
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = 'All object keys must be strings'
    var_21 = 'username'
    var_22 = 'email'
    var_23 = 'test@test.com'
    var_24 = {var_22: var_23}
    var_25 = 'This field is required'
    var_26 = 'Invalid age'
    var_27 = 'error'
    var_28 = []
    var_29 = module_0.Message(text=var_26, code=var_27, index=var_28)
    var_30 = [var_29]
    var_31 = (var_11, var_30)
    var_32 = 'age'
    var_33 = 'not_a_number'
    var_34 = {var_32: var_33}
    var_35 = 'age: Invalid age'
    var_36 = 'id'
    var_37 = {}



