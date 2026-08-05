####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = None
    var_4 = 'original1'
    var_5 = 'original2'
    var_6 = 'original3'
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6}
    var_8 = 'value1'
    var_9 = 123
    var_10 = {var_0: var_8, var_1: var_9, var_2: var_3}
    var_11 = 'attr1'
    var_12 = 'attr2'
    var_13 = 'attr3'
    var_14 = {var_0: var_8, var_1: var_9, var_2: var_3}
    var_15 = 'only_one'
    var_16 = {var_0: var_15}
    var_17 = {var_0: var_8}
    var_18 = 'attr_only'



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'my_ref'
    var_1 = False
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = True
    var_7 = None
    var_8 = None
    var_9 = 'May not be null'
    var_10 = 'Target Error'
    var_11 = 'target_err'
    var_12 = module_0.Message(text=var_10, code=var_11)
    var_13 = [var_12]
    var_14 = 'invalid'
    var_15 = 'data'
    var_16 = {var_14: var_15}



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'extra'
    var_3 = 'John'
    var_4 = 30
    var_5 = 'hidden'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_8 = {var_0: var_3}
    var_9 = None
    var_10 = 'nested'
    var_11 = 'simple'
    var_12 = 'a'
    var_13 = 1
    var_14 = {var_12: var_13}
    var_15 = {var_12: var_13}
    var_16 = 'one'
    var_17 = {var_12: var_13}
    var_18 = {var_10: var_17, var_11: var_13}
    var_19 = {var_12: var_13}
    var_20 = {var_10: var_19, var_11: var_16}



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'read_only_field'
    var_3 = 'nested'
    var_4 = True
    var_5 = None
    var_6 = 'John'
    var_7 = 30
    var_8 = "don't change me"
    var_9 = 'inner'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_11}
    var_13 = {var_9: var_10}
    var_14 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_13}
    var_15 = 'Jane'
    var_16 = 25
    var_17 = 'static'
    var_18 = {var_0: var_15, var_1: var_16, var_2: var_17}
    var_19 = 'Only Name'
    var_20 = {var_0: var_19}
    var_21 = 'key'



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'ro'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = True



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'meta'
    var_4 = 'John'
    var_5 = 25
    var_6 = 'some_meta'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'Jane'
    var_9 = {var_1: var_8}
    var_10 = False
    var_11 = None
    var_12 = 'May not be null'
    var_13 = 'not'
    var_14 = 'a'
    var_15 = 'dict'
    var_16 = [var_13, var_14, var_15]
    var_17 = 'Must be an object'
    var_18 = 'age'
    var_19 = 30
    var_20 = {var_18: var_19}
    var_21 = 'This field is required'
    var_22 = 123
    var_23 = 'value'
    var_24 = {var_22: var_23}
    var_25 = 'All object keys must be strings'
    var_26 = 'Child error'
    var_27 = 'child_err'
    var_28 = [var_23]
    var_29 = module_0.Message(text=var_26, code=var_27, index=var_28)
    var_30 = 'name'
    var_31 = 'bad_val'
    var_32 = {var_30: var_31}



# Parsed testcases at query #7
#--------------------------


import typesystem.schemas as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'my_ref'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'data'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = True
    var_8 = module_0.Reference(var_1, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = False
    var_12 = module_0.Reference(var_1, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = 'May not be null'
    var_16 = 'Invalid'
    var_17 = 'error'
    var_18 = module_1.Message(text=var_16, code=var_17)
    var_19 = [var_18]
    var_20 = 'wrong'
    var_21 = 'data'
    var_22 = {var_20: var_21}
    var_23 = var_2.validate(var_22)



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'required_field'
    var_1 = 'optional_field'
    var_2 = 'readonly_field'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = True



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'status'
    var_5 = 'Alice'
    var_6 = 25
    var_7 = 'active'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'Bob'
    var_10 = {var_2: var_9}
    var_11 = 'age'
    var_12 = 30
    var_13 = {var_11: var_12}
    var_14 = 'required'
    var_15 = [var_13]
    var_16 = 'not a dict'
    var_17 = 'type'
    var_18 = None
    var_19 = 'null'
    var_20 = None
    var_21 = 123
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = 'invalid_key'
    var_25 = 123
    var_26 = [var_25]
    var_27 = 'name'
    var_28 = 'age'
    var_29 = 'error'
    var_30 = 25
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = 'child_error'
    var_33 = 'Charlie'
    var_34 = 40
    var_35 = 'should_be_ignored'
    var_36 = {var_29: var_33, var_30: var_34, var_31: var_35}



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'my_ref'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = var_2.validate(var_5)
    var_8 = True
    var_9 = module_0.Reference(var_1, var_0)
    var_10 = var_9.validate(var_6)
    assert var_10 is None
    var_11 = False
    var_12 = module_0.Reference(var_1, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = 'May not be null.'
    var_16 = 'Invalid value'
    var_17 = 'error_code'
    var_18 = module_1.Message(text=var_16, code=var_17)
    var_19 = [var_18]
    var_20 = 'bad'
    var_21 = 'data'
    var_22 = {var_20: var_21}
    var_23 = var_2.validate(var_22)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'val_a'
    var_1 = 'def_b'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0}
    var_5 = False
    var_6 = None
    var_7 = 'May not be null.'
    var_8 = True
    var_9 = None
    var_10 = 'not'
    var_11 = 'a'
    var_12 = 'dict'
    var_13 = [var_10, var_11, var_12]
    var_14 = 'Must be an object.'
    var_15 = 123
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'All object keys must be strings.'
    var_19 = 'b'
    var_20 = 1
    var_21 = {var_19: var_20}
    var_22 = 'This field is required.'
    var_23 = 'a'
    var_24 = 'some_value'
    var_25 = {var_23: var_24}
    var_26 = 'should_not_appear'
    var_27 = 'val'
    var_28 = {var_25: var_27}



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 10
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = 25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3}
    var_7 = False
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
    var_21 = 'age'
    var_22 = 25
    var_23 = {var_21: var_22}
    var_24 = 'This field is required'
    var_25 = 'Child Error'
    var_26 = 'child_err'
    var_27 = module_0.Message(text=var_25, code=var_26)
    var_28 = 'child'
    var_29 = 'child'
    var_30 = 'bad_data'
    var_31 = {var_29: var_30}
    var_32 = 'child: Child Error'
    var_33 = 'const'
    var_34 = {}



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



# Parsed testcases at query #16
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = 'id'
    var_2 = 'name'
    var_3 = 1
    var_4 = 'John'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = True
    var_8 = None
    var_9 = False
    var_10 = None
    var_11 = 'May not be null.'
    var_12 = 'Invalid'
    var_13 = 'error'
    var_14 = module_0.Message(text=var_12, code=var_13)
    var_15 = [var_14]
    var_16 = 'id'
    var_17 = 'invalid'
    var_18 = {var_16: var_17}
    var_19 = 'nonexistent'
    var_20 = 'some'
    var_21 = 'data'
    var_22 = {var_20: var_21}



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'meta'
    var_4 = 'John'
    var_5 = 25
    var_6 = 'hidden'
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
    var_21 = 25
    var_22 = {var_20: var_21}
    var_23 = 'This field is required'
    var_24 = 'Bad string'
    var_25 = 'err'
    var_26 = []
    var_27 = module_0.Message(text=var_24, code=var_25, index=var_26)
    var_28 = [var_27]
    var_29 = (var_20, var_28)
    var_30 = 'name'
    var_31 = 123
    var_32 = {var_30: var_31}
    var_33 = 'name: Bad string'



# Parsed testcases at query #18
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'my_ref'
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
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.Reference(var_8, var_0)
    var_15 = var_14.validate(var_13)
    var_16 = 'error'
    var_17 = module_0.Reference(var_8, var_0)
    var_18 = var_17.validate(var_13)
    var_19 = 'some_input'
    var_20 = var_14.validate(var_19)
    assert var_20 == 'valid_result'



# Parsed testcases at query #19
#--------------------------


import typesystem.schemas as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'my_ref'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = False
    var_7 = module_0.Reference(var_1, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = module_0.Reference(var_8, var_0)
    var_14 = 'some'
    var_15 = 'data'
    var_16 = {var_14: var_15}
    var_17 = var_13.validate(var_16)
    var_18 = 'Error in target'
    var_19 = 'error'
    var_20 = module_1.Message(text=var_18, code=var_19)
    var_21 = [var_20]
    var_22 = module_1.ValidationError(messages=var_21)
    var_23 = module_0.Reference(var_8, var_0)
    var_24 = 'bad'
    var_25 = 'data'
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)



# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'my_ref'
    var_1 = True
    var_2 = None
    var_3 = False
    var_4 = None
    var_5 = 'May not be null.'
    var_6 = 'data'
    var_7 = 123
    var_8 = {var_6: var_7}
    var_9 = 'Error'
    var_10 = 'err'
    var_11 = 'bad'
    var_12 = 'data'
    var_13 = {var_11: var_12}



# Parsed testcases at query #22
#--------------------------


import typesystem.schemas as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'my_ref'
    var_2 = False
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = 'some'
    var_5 = 'data'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    assert var_7 == 'valid_data'
    var_8 = {var_4: var_5}
    var_9 = module_0.Reference(var_1, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = True
    var_13 = module_0.Reference(var_10, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = 'Target error'
    var_17 = 'target_err'
    var_18 = []
    var_19 = module_1.Message(text=var_16, code=var_17, index=var_18)
    var_20 = [var_19]
    var_21 = module_1.ValidationError(messages=var_20)
    var_22 = 'bad'
    var_23 = 'data'
    var_24 = {var_22: var_23}
    var_25 = var_3.validate(var_24)



# Parsed testcases at query #23
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
    var_7 = None
    var_8 = 'May not be null'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = 'Must be an object'
    var_14 = 'b'
    var_15 = 2
    var_16 = {var_14: var_15}
    var_17 = 'This field is required'
    var_18 = 123
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = 'All object keys must be strings'
    var_22 = 'Child Error'
    var_23 = 'child_err'
    var_24 = []
    var_25 = module_0.Message(text=var_22, code=var_23, index=var_24)
    var_26 = None
    var_27 = [var_25]
    var_28 = lambda add_prefix: var_27
    var_29 = 'child'
    var_30 = 'child'
    var_31 = 'invalid'
    var_32 = {var_30: var_31}
    var_33 = True
    var_34 = True
    var_35 = 'readonly'
    var_36 = 'some_val'
    var_37 = {var_35: var_36}



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'required_key'
    var_1 = 'optional_key'
    var_2 = 'readonly_key'
    var_3 = True



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'status'
    var_4 = 'Alice'
    var_5 = 25
    var_6 = 'active'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'Bob'
    var_9 = {var_1: var_8}
    var_10 = 'not'
    var_11 = 'a'
    var_12 = 'dict'
    var_13 = [var_10, var_11, var_12]
    var_14 = 'Must be an object'
    var_15 = None
    var_16 = 'May not be null'
    var_17 = 'age'
    var_18 = 30
    var_19 = {var_17: var_18}
    var_20 = 'This field is required'
    var_21 = 123
    var_22 = 'invalid'
    var_23 = {var_21: var_22}
    var_24 = 'All object keys must be strings'
    var_25 = 'Invalid string'
    var_26 = 'err'
    var_27 = []
    var_28 = module_0.Message(text=var_25, code=var_26, index=var_27)
    var_29 = [var_28]
    var_30 = (var_21, var_29)
    var_31 = 'name'
    var_32 = 123
    var_33 = {var_31: var_32}
    var_34 = 'name.Invalid string'



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'ro'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = True



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'ro'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = True



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'my_ref'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None
    var_5 = 'key'
    var_6 = 'nested'
    var_7 = 'value'
    var_8 = 123
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_2.serialize(var_9)
    var_11 = (var_5, var_7)
    var_12 = [var_11]
    var_13 = var_2.serialize(var_12)
    var_14 = 'a'
    var_15 = 1
    var_16 = {var_14: var_15}



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'metadata'
    var_4 = 'Alice'
    var_5 = 30
    var_6 = 'some_meta'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'Bob'
    var_9 = {var_1: var_8}
    var_10 = True
    var_11 = None
    var_12 = 'May not be null'
    var_13 = 'not'
    var_14 = 'a'
    var_15 = 'dict'
    var_16 = [var_13, var_14, var_15]
    var_17 = 'Must be an object'
    var_18 = 123
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = 'All object keys must be strings'
    var_22 = 'age'
    var_23 = 25
    var_24 = {var_22: var_23}
    var_25 = 'This field is required'
    var_26 = 'Child error'
    var_27 = 'child_err'
    var_28 = module_0.Message(text=var_26, code=var_27)
    var_29 = 'error_prop'
    var_30 = 'error_prop'
    var_31 = 'bad_data'
    var_32 = {var_30: var_31}
    var_33 = 'error_propChild error'



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'data'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = None
    var_6 = False
    var_7 = None
    var_8 = 'May not be null'
    var_9 = 'Invalid'
    var_10 = 'error'
    var_11 = module_0.Message(text=var_9, code=var_10)
    var_12 = [var_11]
    var_13 = 'bad'
    var_14 = 'data'
    var_15 = {var_13: var_14}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'status'
    var_4 = False
    var_5 = 'John'
    var_6 = 30
    var_7 = 'active'
    var_8 = {var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = {var_1: var_5}
    var_10 = 'age'
    var_11 = 30
    var_12 = {var_10: var_11}
    var_13 = 'This field is required.'
    var_14 = None
    var_15 = 'May not be null.'
    var_16 = True
    var_17 = 'not'
    var_18 = 'a'
    var_19 = 'dict'
    var_20 = [var_17, var_18, var_19]
    var_21 = 'Must be an object.'
    var_22 = 123
    var_23 = 'value'
    var_24 = {var_22: var_23}
    var_25 = 'All object keys must be strings.'
    var_26 = 'error_key'
    var_27 = 'error_key'
    var_28 = 'bad_data'
    var_29 = {var_27: var_28}
    var_30 = 'Error in error_key'



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'meta'
    var_4 = 'John'
    var_5 = 25
    var_6 = 'some_meta'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_1: var_4}
    var_9 = None
    var_10 = 'not'
    var_11 = 'a'
    var_12 = 'dict'
    var_13 = [var_10, var_11, var_12]
    var_14 = 123
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = 'All object keys must be strings.'
    var_18 = {var_12: var_5}
    var_19 = 'This field is required.'
    var_20 = 'Child Error'
    var_21 = 'child_err'
    var_22 = module_0.Message(text=var_20, code=var_21)
    var_23 = [var_22]
    var_24 = lambda add_prefix: var_23
    var_25 = 'child'
    var_26 = 'child'
    var_27 = 'bad_data'
    var_28 = {var_26: var_27}
    var_29 = True



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'required_key'
    var_1 = 'optional_key'
    var_2 = 'readonly_key'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = True



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'f1'
    var_1 = 'f2'
    var_2 = False
    var_3 = None
    var_4 = 'default_val'
    var_5 = True
    var_6 = 'req'
    var_7 = 'opt'
    var_8 = 'ro'
    var_9 = 'something'
    var_10 = {}
    var_11 = module_0.Schema(var_10)



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'my_ref'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = True
    var_8 = module_0.Reference(var_1, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = False
    var_12 = module_0.Reference(var_1, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = 'May not be null.'
    var_16 = 'some'
    var_17 = 'data'
    var_18 = {var_16: var_17}
    var_19 = var_2.validate(var_18)



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'required_key'
    var_1 = 'optional_key'
    var_2 = 'readonly_key'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = True



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'required_key'
    var_1 = 'optional_key'
    var_2 = 'readonly_key'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = True



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'user'
    var_1 = True
    var_2 = None
    var_3 = False
    var_4 = None
    var_5 = 'id'
    var_6 = 'name'
    var_7 = 'Test'
    var_8 = {var_5: var_1, var_6: var_7}
    var_9 = 'Invalid data'
    var_10 = [var_9]
    var_11 = 'invalid'
    var_12 = 'data'
    var_13 = {var_11: var_12}



# Parsed testcases at query #16
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'status'
    var_4 = 'extra'
    var_5 = 'John'
    var_6 = 30
    var_7 = 'active'
    var_8 = 'ignored'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = None
    var_11 = 'May not be null'
    var_12 = 'not'
    var_13 = 'a'
    var_14 = 'dict'
    var_15 = [var_12, var_13, var_14]
    var_16 = 'Must be an object'
    var_17 = 123
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = 'All object keys must be strings'
    var_21 = 'age'
    var_22 = 30
    var_23 = {var_21: var_22}
    var_24 = 'This field is required'
    var_25 = 'Invalid string'
    var_26 = 'error'
    var_27 = []
    var_28 = module_0.Message(text=var_25, code=var_26, index=var_27)
    var_29 = 'name.error'
    var_30 = []
    var_31 = module_0.Message(text=var_29, code=var_26, index=var_30)
    var_32 = 'name'
    var_33 = 'age'
    var_34 = 'wrong'
    var_35 = 30
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = {var_33: var_5}



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'my_ref'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'data'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    assert var_6 == 'validated_value'
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
    var_16 = module_0.Reference(var_14, var_0)
    var_17 = 'Error in target'
    var_18 = [var_17]
    var_19 = 'invalid'
    var_20 = 'data'
    var_21 = {var_19: var_20}
    var_22 = var_16.validate(var_21)



# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = True
    var_2 = None
    var_3 = False
    var_4 = None
    var_5 = 'May not be null'
    var_6 = 'id'
    var_7 = 'name'
    var_8 = 'Test'
    var_9 = {var_6: var_1, var_7: var_8}
    var_10 = {var_6: var_1}
    var_11 = 'Invalid type'
    var_12 = 'type'
    var_13 = module_0.Message(text=var_11, code=var_12)
    var_14 = [var_13]
    var_15 = 'id'
    var_16 = 'not_an_int'
    var_17 = {var_15: var_16}



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'status'
    var_4 = 'Alice'
    var_5 = 30
    var_6 = 'active'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = None
    var_9 = 'May not be null'
    var_10 = True
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
    var_25 = 'error'
    var_26 = [var_21]
    var_27 = module_0.Message(text=var_24, code=var_25, index=var_26)
    var_28 = [var_27]
    var_29 = lambda add_prefix: var_28
    var_30 = 'name'
    var_31 = 'age'
    var_32 = 'bad'
    var_33 = 30
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = 'Bob'
    var_36 = {var_31: var_35}



# Parsed testcases at query #20
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = 'opt'
    var_2 = 'ro'
    var_3 = True
    var_4 = {}
    var_5 = module_0.Schema(var_4)



# Parsed testcases at query #21
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'target_key'
    var_1 = True
    var_2 = None
    var_3 = False
    var_4 = None
    var_5 = 'May not be null.'
    var_6 = 'name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = 'some'
    var_10 = 'data'
    var_11 = {var_9: var_10}
    var_12 = {var_9: var_10}
    var_13 = 'Error'
    var_14 = 'err'
    var_15 = module_0.Message(text=var_13, code=var_14)
    var_16 = [var_15]
    var_17 = 'bad'
    var_18 = 'data'
    var_19 = {var_17: var_18}



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'my_ref'
    var_1 = False
    var_2 = 'key'
    var_3 = 'val'
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = True
    var_7 = None
    var_8 = None
    var_9 = 'May not be null'
    var_10 = 'Error in target'
    var_11 = 'error'
    var_12 = 'invalid'
    var_13 = 'data'
    var_14 = {var_12: var_13}



# Parsed testcases at query #23
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'required_key'
    var_1 = 'optional_key'
    var_2 = 'readonly_key'
    var_3 = True
    var_4 = {}
    var_5 = module_0.Schema(var_4)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'ok'
    var_1 = 10
    var_2 = 'readonly'
    var_3 = True
    var_4 = 'fail'
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'status'
    var_8 = 'error_trigger'
    var_9 = 'Alice'
    var_10 = 30
    var_11 = 'active'
    var_12 = {var_5: var_9, var_6: var_10, var_7: var_11, var_8: var_0}
    var_13 = 'Bob'
    var_14 = {var_5: var_13}
    var_15 = 'age'
    var_16 = 25
    var_17 = {var_15: var_16}
    var_18 = 'required'
    var_19 = 'not'
    var_20 = 'a'
    var_21 = 'dict'
    var_22 = [var_19, var_20, var_21]
    var_23 = 'type'
    var_24 = None
    var_25 = 'null'
    var_26 = 123
    var_27 = 'value'
    var_28 = {var_26: var_27}
    var_29 = 'invalid_key'
    var_30 = 'name'
    var_31 = 'error_trigger'
    var_32 = 'Alice'
    var_33 = 'fail'
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = 'error'
    var_36 = None



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'my_ref'
    var_4 = {var_0: var_1}
    var_5 = True
    var_6 = None
    var_7 = False
    var_8 = None
    var_9 = 'May not be null'
    var_10 = 'trigger_error'



# Parsed testcases at query #26
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'required_field'
    var_1 = 'optional_field'
    var_2 = 'readonly_field'
    var_3 = {}
    var_4 = module_0.Schema(var_3)
    var_5 = True



# Parsed testcases at query #27
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'metadata'
    var_4 = 'Alice'
    var_5 = 30
    var_6 = 'extra'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'Bob'
    var_9 = {var_1: var_8}
    var_10 = None
    var_11 = 'May not be null'
    var_12 = True
    var_13 = 'not'
    var_14 = 'a'
    var_15 = 'dict'
    var_16 = [var_13, var_14, var_15]
    var_17 = 'Must be an object'
    var_18 = 123
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = 'All object keys must be strings'
    var_22 = 'age'
    var_23 = 25
    var_24 = {var_22: var_23}
    var_25 = 'This field is required'
    var_26 = 'Invalid age'
    var_27 = 'error'
    var_28 = []
    var_29 = module_0.Message(text=var_26, code=var_27, index=var_28)
    var_30 = 'age'
    var_31 = 'not_a_number'
    var_32 = {var_30: var_31}



