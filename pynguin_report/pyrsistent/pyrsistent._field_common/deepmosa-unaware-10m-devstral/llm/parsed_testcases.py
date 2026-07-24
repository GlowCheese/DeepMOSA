####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'field2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = '__fields__'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = '__fields__'
    var_10 = 'field3'
    var_11 = {}
    var_12 = {var_6: var_11}
    var_13 = '__fields__'
    var_14 = 'field3'
    var_15 = var_12[var_6][var_14]
    var_16 = {}
    var_17 = {var_6: var_16}
    var_18 = ()
    var_19 = '__fields__'
    var_20 = module_0.set_fields(var_17, var_18, var_19)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = '__fields__'
    var_7 = 'c'
    var_8 = {}
    var_9 = 3
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = True
    var_12 = None
    var_13 = (var_11, var_12)
    var_14 = lambda x: var_13
    var_15 = 0
    var_16 = lambda x: x
    var_17 = lambda _, v: v
    var_18 = (var_11, var_12)
    var_19 = lambda x: var_18
    var_20 = ''
    var_21 = False
    var_22 = lambda x: x
    var_23 = lambda _, v: v
    var_24 = 'field1'
    var_25 = 'field2'
    var_26 = 'field1'
    var_27 = 'field2'
    var_28 = 'other'
    var_29 = {}
    var_30 = 'value'
    var_31 = 'a'
    var_32 = {}
    var_33 = {var_6: var_32, var_31: var_11}
    var_34 = ()
    var_35 = module_0.set_fields(var_33, var_34, var_6)
    var_36 = {}
    var_37 = {var_6: var_36, var_31: var_11}



# Parsed testcases at query #4
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'Value must be positive'
    var_2 = 5
    var_3 = -1
    var_4 = 10
    var_5 = module_0.field(initial=var_4)
    var_6 = True
    var_7 = module_0.field(mandatory=var_6)

def test_case_0():
    var_0 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = 123
    var_2 = module_0.field(var_1)
    var_3 = 'string'
    var_4 = 'not callable'
    var_5 = module_0.field(invariant=var_4)
    var_6 = 'not callable'
    var_7 = module_0.field(factory=var_6)
    var_8 = 'not callable'
    var_9 = module_0.field(serializer=var_8)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = '__fields__'
    var_7 = 'c'
    var_8 = {}
    var_9 = 3
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = '__fields__'
    var_12 = 'field1'
    var_13 = {}
    var_14 = True
    var_15 = None
    var_16 = (var_14, var_15)
    var_17 = lambda x: var_16
    var_18 = 0
    var_19 = False
    var_20 = lambda x: x
    var_21 = lambda _, v: v
    var_22 = ()
    var_23 = '__fields__'
    var_24 = module_0.set_fields(var_10, var_22, var_23)
    var_25 = var_10[var_6][var_12]
    var_26 = {}
    var_27 = {var_6: var_26}
    var_28 = ()
    var_29 = '__fields__'
    var_30 = module_0.set_fields(var_27, var_28, var_29)
    var_31 = 'a'
    var_32 = 'b'
    var_33 = 1
    var_34 = 2
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = 'b'
    var_37 = 'c'
    var_38 = 3
    var_39 = 4
    var_40 = {var_36: var_38, var_37: var_39}
    var_41 = {}
    var_42 = {var_6: var_41}
    var_43 = '__fields__'
    var_44 = module_0.set_fields(var_42, var_28, var_43)



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = '__fields__'
    var_7 = 'c'
    var_8 = {}
    var_9 = 3
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = '__fields__'
    var_12 = True
    var_13 = None
    var_14 = (var_12, var_13)
    var_15 = lambda x: var_14
    var_16 = 0
    var_17 = False
    var_18 = lambda x: x
    var_19 = lambda _, v: v
    var_20 = 'field1'
    var_21 = {}
    var_22 = ()
    var_23 = '__fields__'
    var_24 = module_0.set_fields(var_10, var_22, var_23)
    var_25 = {}
    var_26 = {var_6: var_25, var_20: var_12}
    var_27 = ()
    var_28 = '__fields__'
    var_29 = module_0.set_fields(var_26, var_27, var_28)
    var_30 = {}
    var_31 = {var_6: var_30, var_20: var_12}
    var_32 = '__fields__'
    var_33 = module_0.set_fields(var_31, var_27, var_32)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = 5
    var_3 = module_0.field(initial=var_2)
    var_4 = True
    var_5 = module_0.field(mandatory=var_4)
    var_6 = 'invalid'
    var_7 = module_0.field(var_6)
    var_8 = 'invalid'
    var_9 = 'invalid'
    var_10 = module_0.field(invariant=var_9)
    var_11 = 'invalid'
    var_12 = module_0.field(factory=var_11)
    var_13 = 'invalid'
    var_14 = module_0.field(serializer=var_13)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = {var_2: var_0}
    var_7 = {var_2: var_0, var_3: var_4}
    var_8 = 'test_field'
    var_9 = 'test_field'
    var_10 = 'a'
    var_11 = 'not_int'
    var_12 = {var_10: var_11}
    var_13 = 'Too small'
    var_14 = lambda pmap: (len(pmap) < var_4, var_13)
    var_15 = {var_2: var_0}



# Parsed testcases at query #6
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'format'
    var_1 = module_0.serialize(var_0)
    var_2 = 'test_value'



# Parsed testcases at query #8
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'test_field'
    var_4 = 'string'
    var_5 = 'string'
    var_6 = 'test_field'
    var_7 = 1.5
    var_8 = 1.5



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = lambda x: x
    var_1 = False
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = True
    var_6 = lambda x: x
    var_7 = lambda x: x



# Parsed testcases at query #11
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = ''
    var_3 = (var_0, var_2)
    var_4 = lambda x: var_3
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_5: var_0, var_6: var_7}



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'hello'
    var_4 = set()
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = 'test_field'
    var_9 = 'not an int'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'Value must be positive'
    var_2 = 5
    var_3 = -1
    var_4 = 10
    var_5 = module_0.field(initial=var_4)
    var_6 = True
    var_7 = module_0.field(mandatory=var_6)

def test_case_0():
    var_0 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'json'
    var_1 = 123
    var_2 = module_0.field(var_1)
    var_3 = 'string'
    var_4 = 'not a function'
    var_5 = module_0.field(invariant=var_4)
    var_6 = 'not a function'
    var_7 = module_0.field(factory=var_6)
    var_8 = 'not a function'
    var_9 = module_0.field(serializer=var_8)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'test_field'
    var_4 = 10
    var_5 = 10.5
    var_6 = set()
    var_7 = 'any_type'



# Parsed testcases at query #16
#--------------------------




# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'test_field'
    var_4 = 'not_an_int'
    var_5 = 'string'
    var_6 = 'test_field'
    var_7 = 10.5
    var_8 = 'anything'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = ''
    var_3 = (var_0, var_2)
    var_4 = lambda x: var_3
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_5: var_0, var_6: var_7}
    var_9 = {var_5: var_0}
    var_10 = 'not a type'
    var_11 = 'not a type'
    var_12 = 'test_field'
    var_13 = 'a'
    var_14 = 'not an int'
    var_15 = {var_13: var_14}
    var_16 = 'test_field'
    var_17 = 1
    var_18 = 2
    var_19 = {var_17: var_18}



# Parsed testcases at query #20
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 5
    var_4 = module_0.field(initial=var_3)
    var_5 = True
    var_6 = module_0.field(mandatory=var_5)

def test_case_0():
    pass

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(var_0)
    var_2 = 'string'
    var_3 = 'not callable'
    var_4 = module_0.field(invariant=var_3)
    var_5 = 'not callable'
    var_6 = module_0.field(factory=var_5)
    var_7 = 'not callable'
    var_8 = module_0.field(serializer=var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(var_0)
    var_2 = 'string'
    var_3 = 'not callable'
    var_4 = module_0.field(invariant=var_3)
    var_5 = 'not callable'
    var_6 = module_0.field(factory=var_5)
    var_7 = 'not callable'
    var_8 = module_0.field(serializer=var_7)



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 10.5
    var_4 = set()
    var_5 = 'any_value'
    var_6 = 'test_field'
    var_7 = 10



# Parsed testcases at query #23
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = module_0.field()
    var_4 = 'c'
    var_5 = module_0.field()
    var_6 = {var_4: var_5}
    var_7 = module_0.field()
    var_8 = '__fields__'
    var_9 = module_0.field()
    var_10 = ()
    var_11 = module_0.field()
    var_12 = module_0.field()
    var_13 = module_0.field()
    var_14 = module_0.field()
    var_15 = ()



# Parsed testcases at query #24
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'



# Parsed testcases at query #25
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #26
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 5
    var_4 = module_0.field(initial=var_3)
    var_5 = True
    var_6 = module_0.field(mandatory=var_5)

def test_case_0():
    pass

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.field(var_0)
    var_2 = 'not an int'
    var_3 = 'not callable'
    var_4 = module_0.field(invariant=var_3)
    var_5 = 'not callable'
    var_6 = module_0.field(factory=var_5)
    var_7 = 'not callable'
    var_8 = module_0.field(serializer=var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.field(var_0)
    var_2 = 'not an int'
    var_3 = 'not callable'
    var_4 = module_0.field(invariant=var_3)
    var_5 = 'not callable'
    var_6 = module_0.field(factory=var_5)
    var_7 = 'not callable'
    var_8 = module_0.field(serializer=var_7)



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = 'x'
    var_8 = 10
    var_9 = {var_7: var_8}
    var_10 = 'test_field'
    var_11 = 'a'
    var_12 = 'not_int'
    var_13 = {var_11: var_12}



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'hello'
    var_4 = 'test_field'
    var_5 = 'not_an_int'
    var_6 = 'anything_goes'



# Parsed testcases at query #29
#--------------------------




# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'hello'
    var_4 = 'test_field'
    var_5 = 'not_an_int'
    var_6 = set()
    var_7 = 'anything_goes'



# Parsed testcases at query #31
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #32
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'Value must be positive'
    var_2 = 5
    var_3 = -1
    var_4 = 10
    var_5 = module_0.field(initial=var_4)
    var_6 = True
    var_7 = module_0.field(mandatory=var_6)

def test_case_0():
    var_0 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = 123
    var_2 = module_0.field(var_1)
    var_3 = 'not an int'
    var_4 = 'not callable'
    var_5 = module_0.field(invariant=var_4)
    var_6 = 'not callable'
    var_7 = module_0.field(factory=var_6)
    var_8 = 'not callable'
    var_9 = module_0.field(serializer=var_8)



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = {var_2: var_0}
    var_7 = 'x'
    var_8 = 10
    var_9 = {var_7: var_8}



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'hello'
    var_4 = 'test_field'
    var_5 = 'hello'
    var_6 = set()
    var_7 = 'anything'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'test_field'
    var_4 = 'string'
    var_5 = 'string'
    var_6 = set()
    var_7 = 'anything'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'hello'
    var_4 = 'test_field'
    var_5 = 'not_an_int'
    var_6 = set()
    var_7 = 'anything_goes'



# Parsed testcases at query #37
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 10.5
    var_4 = 'test_field'
    var_5 = 'string'
    var_6 = set()
    var_7 = 'any_type'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = lambda x: x
    var_6 = lambda _, value: value
    var_7 = (var_0, var_1)
    var_8 = lambda x: var_7
    var_9 = lambda x: x
    var_10 = lambda _, value: value
    var_11 = (var_0, var_1)
    var_12 = lambda x: var_11
    var_13 = lambda _, value: value
    var_14 = (var_0, var_1)
    var_15 = lambda x: var_14
    var_16 = lambda _, value: value
    var_17 = ()
    var_18 = (var_0, var_1)
    var_19 = lambda x: var_18
    var_20 = lambda _, value: value



# Parsed testcases at query #40
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = set()
    var_1 = lambda x: x
    var_2 = module_0._PField(var_0, var_1)
    var_3 = False
    var_4 = lambda x: x
    var_5 = True
    var_6 = set()
    var_7 = lambda x: x
    var_8 = module_0._PField(var_6, var_7)
    var_9 = set()
    var_10 = set()
    var_11 = (var_10,)
    var_12 = set()
    var_13 = [var_12]



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_2: var_0, var_5: var_6}
    var_8 = {var_2: var_0, var_5: var_6}
    var_9 = 'test_field'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_2, var_4: var_5}
    var_7 = {var_3: var_2, var_4: var_5}
    var_8 = 'x'
    var_9 = 10
    var_10 = {var_8: var_9}
    var_11 = {var_8: var_9}
    var_12 = 'test_field'
    var_13 = 'a'
    var_14 = 'not_int'
    var_15 = {var_13: var_14}
    var_16 = 'a'
    var_17 = 1
    var_18 = {var_16: var_17}

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_2, var_4: var_5}
    var_7 = {var_3: var_2, var_4: var_5}
    var_8 = 'x'
    var_9 = 10
    var_10 = {var_8: var_9}
    var_11 = {var_8: var_9}
    var_12 = 'test_field'
    var_13 = 'a'
    var_14 = 'not_int'
    var_15 = {var_13: var_14}
    var_16 = 'a'
    var_17 = 1
    var_18 = {var_16: var_17}



# Parsed testcases at query #43
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 5
    var_4 = module_0.field(initial=var_3)
    var_5 = True
    var_6 = module_0.field(mandatory=var_5)

def test_case_0():
    pass

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.field(var_0)
    var_2 = 'not an int'
    var_3 = 'not callable'
    var_4 = module_0.field(invariant=var_3)
    var_5 = 'not callable'
    var_6 = module_0.field(factory=var_5)
    var_7 = 'not callable'
    var_8 = module_0.field(serializer=var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.field(var_0)
    var_2 = 'not an int'
    var_3 = 'not callable'
    var_4 = module_0.field(invariant=var_3)
    var_5 = 'not callable'
    var_6 = module_0.field(factory=var_5)
    var_7 = 'not callable'
    var_8 = module_0.field(serializer=var_7)



# Parsed testcases at query #44
#--------------------------




# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'test_field'
    var_4 = 'not_an_int'
    var_5 = 'now_its_ok'
    var_6 = 'anything_goes'
    var_7 = 'test_field'
    var_8 = 'not_custom_type'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = ''
    var_3 = (var_0, var_2)
    var_4 = lambda x: var_3
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_5: var_0, var_6: var_7}
    var_9 = 'x'
    var_10 = 10
    var_11 = {var_9: var_10}
    var_12 = {var_9: var_10}



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = {var_0: var_2}
    var_7 = 1
    var_8 = 'invalid_key'
    var_9 = 'one'
    var_10 = 'two'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 1
    var_13 = 2
    var_14 = 'one'
    var_15 = {var_12: var_14, var_13: var_13}
    var_16 = {var_14: var_2, var_1: var_3}



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'test_field'
    var_4 = 'string'
    var_5 = 'string'
    var_6 = 'test_field'
    var_7 = 10.5
    var_8 = 'anything'



# Parsed testcases at query #49
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #50
#--------------------------




# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = None
    var_7 = 'x'
    var_8 = 10
    var_9 = {var_7: var_8}
    var_10 = {var_7: var_8}
    var_11 = 'test_field'
    var_12 = 'a'
    var_13 = 'not_int'
    var_14 = {var_12: var_13}

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = None
    var_7 = 'x'
    var_8 = 10
    var_9 = {var_7: var_8}
    var_10 = {var_7: var_8}
    var_11 = 'test_field'
    var_12 = 'a'
    var_13 = 'not_int'
    var_14 = {var_12: var_13}



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = lambda x: var_3
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_5: var_0, var_6: var_7}
    var_9 = 1
    var_10 = 'a'
    var_11 = {var_9: var_10}
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_13}
    var_15 = 'x'
    var_16 = {var_15: var_0}
    var_17 = {var_15: var_0}
    var_18 = 'default'
    var_19 = 0
    var_20 = {var_18: var_19}
    var_21 = {var_18: var_19}
    var_22 = 'key'
    var_23 = 42
    var_24 = {var_22: var_23}



# Parsed testcases at query #53
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #54
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = None
    var_6 = {var_1: var_0}
    var_7 = 'x'
    var_8 = 10
    var_9 = {var_7: var_8}
    var_10 = 'test_field'
    var_11 = 'a'
    var_12 = 'not_int'
    var_13 = {var_11: var_12}
    var_14 = 'test_field'
    var_15 = 1
    var_16 = 2
    var_17 = {var_15: var_16}
    var_18 = {var_1: var_0}
    var_19 = 'test_format'
    var_20 = module_0.serialize(var_19)



# Parsed testcases at query #55
#--------------------------




# Parsed testcases at query #56
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'xml'



# Parsed testcases at query #57
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = '__fields__'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = True
    var_10 = None
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = 0
    var_14 = lambda x: x
    var_15 = lambda _, v: v
    var_16 = (var_9, var_10)
    var_17 = lambda x: var_16
    var_18 = ''
    var_19 = False
    var_20 = lambda x: x
    var_21 = lambda _, v: v
    var_22 = 'field1'
    var_23 = 'field2'
    var_24 = 'field1'
    var_25 = 'field2'
    var_26 = {}
    var_27 = {}
    var_28 = {var_6: var_27}
    var_29 = ()
    var_30 = module_0.set_fields(var_28, var_29, var_6)
    var_31 = {}
    var_32 = {var_6: var_31}
    var_33 = module_0.set_fields(var_32, var_29, var_6)



# Parsed testcases at query #58
#--------------------------




# Parsed testcases at query #59
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = None
    var_6 = {var_1: var_0}
    var_7 = field.factory(var_6)[var_1]
    assert var_7 == 1
    var_8 = 'TestClass'
    var_9 = ()
    var_10 = {}
    var_11 = 'test_field'
    var_12 = 'a'
    var_13 = 'not_int'
    var_14 = {var_12: var_13}
    var_15 = 'x'
    var_16 = 10
    var_17 = {var_15: var_16}
    var_18 = {var_15: var_16}
    var_19 = 0
    var_20 = False
    var_21 = 'Error'
    var_22 = (var_20, var_21)
    var_23 = (var_11, var_5)
    var_24 = lambda p: var_22 if len(p) > var_19 else var_23
    var_25 = 'a'
    var_26 = 1
    var_27 = {var_25: var_26}
    var_28 = [var_12]
    var_29 = module_0.check_global_invariants(var_11, var_28)



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = lambda x: x
    var_1 = False
    var_2 = lambda x: x
    var_3 = True
    var_4 = lambda x: x
    var_5 = lambda x: x



# Parsed testcases at query #61
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = '__fields__'
    var_7 = 'c'
    var_8 = {}
    var_9 = 3
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = '__fields__'
    var_12 = True
    var_13 = None
    var_14 = (var_12, var_13)
    var_15 = lambda x: var_14
    var_16 = 0
    var_17 = lambda x: x
    var_18 = lambda _, v: v
    var_19 = 'field1'
    var_20 = {}
    var_21 = ()
    var_22 = module_0.set_fields(var_10, var_21, var_11)
    var_23 = 'a'
    var_24 = {}
    var_25 = {var_6: var_24, var_23: var_12}
    var_26 = ()
    var_27 = '__fields__'
    var_28 = module_0.set_fields(var_25, var_26, var_27)
    var_29 = {}
    var_30 = {var_6: var_29, var_23: var_12}
    var_31 = '__fields__'
    var_32 = module_0.set_fields(var_30, var_26, var_31)



# Parsed testcases at query #62
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = 'value'



# Parsed testcases at query #63
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #64
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #65
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 0
    var_5 = False
    var_6 = lambda x: x
    var_7 = lambda _, v: v
    var_8 = False
    var_9 = False
    var_10 = (var_0, var_1)
    var_11 = lambda x: var_10
    var_12 = ''
    var_13 = False
    var_14 = lambda x: x
    var_15 = lambda _, v: v
    var_16 = (var_0, var_1)
    var_17 = lambda x: var_16
    var_18 = False
    var_19 = lambda x: x
    var_20 = lambda _, v: v
    var_21 = (var_0, var_1)
    var_22 = lambda x: var_21
    var_23 = False
    var_24 = lambda _, v: v
    var_25 = set()
    var_26 = (var_0, var_1)
    var_27 = lambda x: var_26
    var_28 = False
    var_29 = lambda x: x
    var_30 = lambda _, v: v
    var_31 = module_0._PField(var_25, var_27, var_1, var_28, var_29, var_30)



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = 'Too many items'
    var_6 = (var_4, var_5)
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 2
    var_10 = {var_7: var_3, var_8: var_9}
    var_11 = 'test_field'
    var_12 = 'not'
    var_13 = 'a'
    var_14 = 'map'
    var_15 = [var_12, var_13, var_14]

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = 'Too many items'
    var_6 = (var_4, var_5)
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 2
    var_10 = {var_7: var_3, var_8: var_9}
    var_11 = 'test_field'
    var_12 = 'not'
    var_13 = 'a'
    var_14 = 'map'
    var_15 = [var_12, var_13, var_14]



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = ''
    var_3 = (var_0, var_2)
    var_4 = lambda x: var_3
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_5: var_0, var_6: var_7}



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = {var_2: var_0, var_3: var_4}
    var_7 = 'x'
    var_8 = 10
    var_9 = {var_7: var_8}
    var_10 = {var_7: var_8}
    var_11 = {var_7: var_8}
    var_12 = {var_7: var_8}



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = {}
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 10
    var_6 = 20
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 1.5
    var_9 = 2.5
    var_10 = 100
    var_11 = 200
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'default'
    var_14 = 0
    var_15 = {var_13: var_14}
    var_16 = {var_13: var_14}

def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = {}
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 10
    var_6 = 20
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 1.5
    var_9 = 2.5
    var_10 = 100
    var_11 = 200
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'default'
    var_14 = 0
    var_15 = {var_13: var_14}
    var_16 = {var_13: var_14}



# Parsed testcases at query #70
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = 'x'
    var_7 = 10
    var_8 = {var_6: var_7}
    var_9 = 'TestClass'
    var_10 = ()
    var_11 = {}
    var_12 = 'test_field'
    var_13 = 'a'
    var_14 = 'not_int'
    var_15 = {var_13: var_14}
    var_16 = 'TestRecord'
    var_17 = ()
    var_18 = 'test_field'
    var_19 = 'Instance'
    var_20 = ()
    var_21 = {}
    var_22 = 'key'
    var_23 = 123
    var_24 = {var_22: var_23}
    var_25 = {}



# Parsed testcases at query #72
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_4: var_0, var_5: var_6}
    var_8 = None
    var_9 = {var_4: var_0}
    var_10 = 'test_field'
    var_11 = 'a'
    var_12 = 'not_int'
    var_13 = {var_11: var_12}
    var_14 = 3.0
    var_15 = {var_4: var_0, var_6: var_14}



# Parsed testcases at query #73
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #74
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = '__fields__'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'x'
    var_10 = None
    var_11 = False
    var_12 = module_0._PField(var_10, var_10, var_10, var_11, var_10, var_10)
    var_13 = {}
    var_14 = {var_9: var_12, var_6: var_13}
    var_15 = ()
    var_16 = module_0.set_fields(var_14, var_15, var_6)
    var_17 = {}
    var_18 = {var_6: var_17}
    var_19 = ()
    var_20 = module_0.set_fields(var_18, var_19, var_6)
    var_21 = {}
    var_22 = {}
    var_23 = {var_6: var_22}
    var_24 = module_0.set_fields(var_23, var_19, var_6)
    var_25 = 'a'
    var_26 = 1
    var_27 = {var_25: var_26}
    var_28 = 'a'
    var_29 = 2
    var_30 = {var_28: var_29}
    var_31 = {}
    var_32 = {var_6: var_31}
    var_33 = module_0.set_fields(var_32, var_19, var_6)



# Parsed testcases at query #75
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #76
#--------------------------




# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'Invalid types'
    var_5 = (var_3, var_4)
    var_6 = '1'
    var_7 = 0
    var_8 = 'Empty map'
    var_9 = lambda x: (len(x) > var_7, var_8)
    var_10 = {}

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'Invalid types'
    var_5 = (var_3, var_4)
    var_6 = '1'
    var_7 = 0
    var_8 = 'Empty map'
    var_9 = lambda x: (len(x) > var_7, var_8)
    var_10 = {}



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = lambda x: x



# Parsed testcases at query #80
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #81
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #82
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
    var_10 = (var_9, var_6)
    var_11 = lambda x: var_10
    var_12 = [var_8, var_11]
    var_13 = module_0.check_global_invariants(var_4, var_12)
    var_14 = {var_0: var_9, var_1: var_3}
    var_15 = False
    var_16 = 'Error1'
    var_17 = (var_15, var_16)
    var_18 = lambda x: var_17
    var_19 = True
    var_20 = (var_19, var_6)
    var_21 = lambda x: var_20
    var_22 = [var_18, var_21]
    var_23 = module_0.check_global_invariants(var_14, var_22)
    var_24 = {var_23: var_19, var_1: var_3}
    var_25 = (var_15, var_16)
    var_26 = lambda x: var_25
    var_27 = 'Error2'
    var_28 = (var_15, var_27)
    var_29 = lambda x: var_28
    var_30 = [var_26, var_29]
    var_31 = module_0.check_global_invariants(var_24, var_30)
    var_32 = {var_31: var_19, var_1: var_3}
    var_33 = []
    var_34 = module_0.check_global_invariants(var_32, var_33)



# Parsed testcases at query #83
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'Value must be positive'
    var_2 = 5
    var_3 = -1
    var_4 = 10
    var_5 = module_0.field(initial=var_4)
    var_6 = True
    var_7 = module_0.field(mandatory=var_6)

def test_case_0():
    var_0 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'json'
    var_1 = 123
    var_2 = module_0.field(var_1)
    var_3 = 'not an int'
    var_4 = 'not callable'
    var_5 = module_0.field(invariant=var_4)
    var_6 = 'not callable'
    var_7 = module_0.field(factory=var_6)
    var_8 = 'not callable'
    var_9 = module_0.field(serializer=var_8)



# Parsed testcases at query #84
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)
    var_6 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #85
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_0: var_2}
    var_4 = {var_0: var_2}
    var_5 = {var_0: var_2}
    var_6 = 10
    var_7 = range(var_6)
    var_8 = {i: 'a' for i in var_7}
    var_9 = {var_0: var_2}
    var_10 = {var_0: var_2}
    var_11 = {var_0: var_2}
    var_12 = {var_0: var_2}
    var_13 = 'TestClass'
    var_14 = ()
    var_15 = {}
    var_16 = 'test_field'
    var_17 = 1
    var_18 = 2
    var_19 = {var_17: var_18}
    var_20 = 'TestClass'
    var_21 = ()
    var_22 = {}
    var_23 = 'test_field'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = {var_24: var_25}
    var_27 = 'TestClass'
    var_28 = ()
    var_29 = {}
    var_30 = 'test_field'
    var_31 = 1
    var_32 = 2
    var_33 = 3
    var_34 = [var_31, var_32, var_33]
    var_35 = 'TestClass'
    var_36 = ()
    var_37 = {}
    var_38 = 'test_field'
    var_39 = 1
    var_40 = 'a'
    var_41 = {var_39: var_40}
    var_42 = ()
    var_43 = {}
    var_44 = {var_39: var_40}
    var_45 = 2
    var_46 = 'b'
    var_47 = {var_37: var_43, var_45: var_46}
    var_48 = 'test_format'
    var_49 = module_0.serialize(var_48)



# Parsed testcases at query #86
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 0
    var_5 = False
    var_6 = lambda x: x
    var_7 = lambda _, v: v
    var_8 = False
    var_9 = (var_0, var_1)
    var_10 = lambda x: var_9
    var_11 = {var_8}
    var_12 = False
    var_13 = lambda x: x
    var_14 = lambda _, v: v
    var_15 = (var_0, var_1)
    var_16 = lambda x: var_15
    var_17 = {var_12}
    var_18 = False
    var_19 = lambda _, v: v
    var_20 = (var_0, var_1)
    var_21 = lambda x: var_20
    var_22 = (var_18,)
    var_23 = False
    var_24 = lambda x: x
    var_25 = lambda _, v: v
    var_26 = (var_0, var_1)
    var_27 = lambda x: var_26
    var_28 = [var_23]
    var_29 = False
    var_30 = lambda x: x
    var_31 = lambda _, v: v



# Parsed testcases at query #87
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)
    var_6 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #88
#--------------------------




# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = lambda x: x
    var_1 = False
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = True
    var_6 = lambda x: x
    var_7 = set()



# Parsed testcases at query #90
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = None
    var_6 = {var_1: var_0}
    var_7 = 'test_field'
    var_8 = 'a'
    var_9 = 'not_int'
    var_10 = {var_8: var_9}
    var_11 = 'x'
    var_12 = 10
    var_13 = {var_11: var_12}
    var_14 = 'key'
    var_15 = 123
    var_16 = {var_14: var_15}



# Parsed testcases at query #91
#--------------------------




# Parsed testcases at query #92
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'
    var_9 = 'xml'



# Parsed testcases at query #93
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'format'
    var_2 = 'serialized_'



# Parsed testcases at query #94
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = None
    var_6 = {var_1: var_0}
    var_7 = 'test_field'
    var_8 = 'a'
    var_9 = 'not an int'
    var_10 = {var_8: var_9}
    var_11 = {var_0: var_0, var_1: var_3}



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = '__fields__'
    var_7 = 'c'
    var_8 = {}
    var_9 = 3
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = '__fields__'
    var_12 = 'field1'
    var_13 = {}
    var_14 = True
    var_15 = None
    var_16 = (var_14, var_15)
    var_17 = lambda x: var_16
    var_18 = 0
    var_19 = lambda x: x
    var_20 = lambda _, v: v
    var_21 = ()
    var_22 = '__fields__'
    var_23 = module_0.set_fields(var_10, var_21, var_22)
    var_24 = 'x'
    var_25 = {}
    var_26 = 10
    var_27 = {var_6: var_25, var_24: var_26}
    var_28 = ()
    var_29 = '__fields__'
    var_30 = module_0.set_fields(var_27, var_28, var_29)
    var_31 = 'a'
    var_32 = 'b'
    var_33 = 2
    var_34 = {var_31: var_14, var_32: var_33}
    var_35 = ()
    var_36 = '__fields__'
    var_37 = module_0.set_fields(var_34, var_35, var_36)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = '__fields__'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = '__fields__'
    var_10 = True
    var_11 = None
    var_12 = (var_10, var_11)
    var_13 = lambda x: var_12
    var_14 = 0
    var_15 = lambda x: x
    var_16 = lambda _, v: v
    var_17 = (var_10, var_11)
    var_18 = lambda x: var_17
    var_19 = ''
    var_20 = False
    var_21 = lambda x: x
    var_22 = lambda _, v: v
    var_23 = 'c'
    var_24 = 'd'
    var_25 = 'c'
    var_26 = 'd'
    var_27 = {}
    var_28 = '__fields__'
    var_29 = {}
    var_30 = {var_6: var_29}
    var_31 = '__fields__'
    var_32 = 'a'
    var_33 = 1
    var_34 = {var_32: var_33}
    var_35 = 'a'
    var_36 = 2
    var_37 = {var_35: var_36}
    var_38 = {}
    var_39 = {var_6: var_38}
    var_40 = '__fields__'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #4
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_4: var_0, var_5: var_6}
    var_8 = None
    var_9 = 'TestClass'
    var_10 = ()
    var_11 = {}
    var_12 = 'test_field'
    var_13 = 'a'
    var_14 = 'not_an_int'
    var_15 = {var_13: var_14}



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'hello'
    var_4 = 'test_field'
    var_5 = 'not_an_int'
    var_6 = set()
    var_7 = 'any_type_is_ok'



# Parsed testcases at query #7
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'format'
    var_1 = module_0.serialize(var_0)
    var_2 = 'test_value'



# Parsed testcases at query #9
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = ()



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 10.5
    var_4 = 'test_field'
    var_5 = 10
    var_6 = ()
    var_7 = 'any_type'



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'test_field'
    var_4 = 'not an int'
    var_5 = 'a string'
    var_6 = ()
    var_7 = 'a'
    var_8 = 'dict'
    var_9 = {var_7: var_8}



# Parsed testcases at query #16
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 5
    var_4 = module_0.field(initial=var_3)
    var_5 = True
    var_6 = module_0.field(mandatory=var_5)

def test_case_0():
    pass

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(var_0)
    var_2 = 'not an int'
    var_3 = 'not callable'
    var_4 = module_0.field(invariant=var_3)
    var_5 = 'not callable'
    var_6 = module_0.field(factory=var_5)
    var_7 = 'not callable'
    var_8 = module_0.field(serializer=var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(var_0)
    var_2 = 'not an int'
    var_3 = 'not callable'
    var_4 = module_0.field(invariant=var_3)
    var_5 = 'not callable'
    var_6 = module_0.field(factory=var_5)
    var_7 = 'not callable'
    var_8 = module_0.field(serializer=var_7)



# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = 'Values must be positive'
    var_6 = (var_4, var_5)
    var_7 = -1
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 10
    var_11 = 20
    var_12 = {var_8: var_10, var_9: var_11}

def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = 'Values must be positive'
    var_6 = (var_4, var_5)
    var_7 = -1
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 10
    var_11 = 20
    var_12 = {var_8: var_10, var_9: var_11}



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'test_field'
    var_4 = 'not_an_int'
    var_5 = 'string'
    var_6 = set()
    var_7 = 'any_type'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = {var_2: var_0}
    var_7 = 'not_a_type'
    var_8 = 'not_a_type'
    var_9 = 0
    var_10 = 'Empty map'
    var_11 = lambda x: (len(x) > var_9, var_10)
    var_12 = {var_2: var_0}
    var_13 = {}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'age'
    var_2 = 25
    var_3 = 'age'
    var_4 = '25'
    var_5 = 'value'
    var_6 = 25.5
    var_7 = 'value'
    var_8 = '25'
    var_9 = set()
    var_10 = 'anything'
    var_11 = 'parent'



# Parsed testcases at query #22
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'Value must be positive'
    var_2 = 5
    var_3 = -1
    var_4 = 10
    var_5 = module_0.field(initial=var_4)
    var_6 = True
    var_7 = module_0.field(mandatory=var_6)

def test_case_0():
    var_0 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'json'
    var_1 = 123
    var_2 = module_0.field(var_1)
    var_3 = 'string'
    var_4 = 'not a function'
    var_5 = module_0.field(invariant=var_4)
    var_6 = 'not a function'
    var_7 = module_0.field(factory=var_6)
    var_8 = 'not a function'
    var_9 = module_0.field(serializer=var_8)



# Parsed testcases at query #24
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = 'value'



# Parsed testcases at query #25
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_4: var_0, var_5: var_6}
    var_8 = {var_4: var_0, var_5: var_6}
    var_9 = 123
    var_10 = 123
    var_11 = 'test_field'
    var_12 = 'a'
    var_13 = 'not_an_int'
    var_14 = {var_12: var_13}
    var_15 = 'test_field'
    var_16 = {var_4: var_13, var_5: var_6}



# Parsed testcases at query #27
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = '_fields'
    var_7 = 'c'
    var_8 = {}
    var_9 = 3
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = '_fields'
    var_12 = True
    var_13 = None
    var_14 = (var_12, var_13)
    var_15 = lambda x: var_14
    var_16 = 0
    var_17 = lambda x: x
    var_18 = lambda _, v: v
    var_19 = (var_12, var_13)
    var_20 = lambda x: var_19
    var_21 = ''
    var_22 = False
    var_23 = lambda x: x
    var_24 = lambda _, v: v
    var_25 = 'd'
    var_26 = 'e'
    var_27 = 'f'
    var_28 = {}
    var_29 = '_fields'
    var_30 = 'g'
    var_31 = {}
    var_32 = 4
    var_33 = {var_6: var_31, var_30: var_32}
    var_34 = '_fields'
    var_35 = 'h'
    var_36 = 5
    var_37 = {var_35: var_36}
    var_38 = '_fields'



# Parsed testcases at query #29
#--------------------------




# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = 'test_field'
    var_2 = 10
    var_3 = 'test_field'
    var_4 = 'string'
    var_5 = 'string'
    var_6 = 'test_field'
    var_7 = 10.5
    var_8 = 10.5



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = 0.0
    var_8 = lambda x: x
    var_9 = lambda _, v: v
    var_10 = '__fields__'



# Parsed testcases at query #32
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = '_test_fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = 'field1'
    var_5 = 'value1'
    var_6 = {var_4: var_5}
    var_7 = 'field2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = module_0.set_fields(var_10, var_1, var_2)
    var_12 = 'field3'
    var_13 = True
    var_14 = None
    var_15 = (var_13, var_14)
    var_16 = lambda x: var_15
    var_17 = 0
    var_18 = False
    var_19 = lambda x: x
    var_20 = lambda _, v: v
    var_21 = ()
    var_22 = module_0.set_fields(var_10, var_21, var_2)
    var_23 = 'field4'
    var_24 = 'value4'
    var_25 = {var_23: var_24}
    var_26 = 'field5'
    var_27 = (var_13, var_14)
    var_28 = lambda x: var_27
    var_29 = ''
    var_30 = False
    var_31 = lambda x: x
    var_32 = lambda _, v: v
    var_33 = module_0.set_fields(var_10, var_21, var_2)



# Parsed testcases at query #33
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_6]
    var_8 = module_0.check_global_invariants(var_2, var_7)
    var_9 = {var_0: var_1}
    var_10 = False
    var_11 = 'error_code'
    var_12 = (var_10, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_13]
    var_15 = module_0.check_global_invariants(var_9, var_14)
    var_16 = {var_15: var_1}
    var_17 = (var_3, var_4)
    var_18 = lambda x: var_17
    var_19 = 'error_code1'
    var_20 = (var_10, var_19)
    var_21 = lambda x: var_20
    var_22 = 'error_code2'
    var_23 = (var_10, var_22)
    var_24 = lambda x: var_23
    var_25 = [var_18, var_21, var_24]
    var_26 = module_0.check_global_invariants(var_16, var_25)
    var_27 = {var_26: var_1}
    var_28 = []
    var_29 = module_0.check_global_invariants(var_27, var_28)



# Parsed testcases at query #34
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #35
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {}
    var_3 = '__fields__'
    var_4 = {}
    var_5 = '__fields__'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = 'x'
    var_9 = 'x'
    var_10 = {}



# Parsed testcases at query #37
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #38
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #39
#--------------------------




# Parsed testcases at query #40
#--------------------------




# Parsed testcases at query #41
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2



# Parsed testcases at query #43
#--------------------------




# Parsed testcases at query #44
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'c'
    var_4 = '__fields__'
    var_5 = 'a'
    var_6 = '__fields__'
    var_7 = 'b'



# Parsed testcases at query #46
#--------------------------




# Parsed testcases at query #47
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_2: var_0, var_5: var_6}
    var_8 = {var_0: var_2, var_6: var_5}
    var_9 = 1
    var_10 = 'a'
    var_11 = {var_9: var_10}
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_13}



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_4: var_0, var_5: var_6}
    var_8 = None
    var_9 = {var_4: var_0}
    var_10 = 'x'
    var_11 = 10
    var_12 = {var_10: var_11}
    var_13 = 'test_field'
    var_14 = 'a'
    var_15 = 'not_int'
    var_16 = {var_14: var_15}
    var_17 = 'test_field'
    var_18 = 1
    var_19 = 2
    var_20 = {var_18: var_19}
    var_21 = 3.0
    var_22 = {var_4: var_0, var_6: var_21}



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = {var_2: var_0}
    var_7 = field_obj.factory(var_6)[var_2]
    assert var_7 == 1
    var_8 = 'x'
    var_9 = 10
    var_10 = {var_8: var_9}
    var_11 = {var_8: var_9}
    var_12 = {var_2: var_0, var_3: var_4}
    var_13 = 'a'
    var_14 = 'not an int'
    var_15 = {var_13: var_14}



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_4: var_0, var_5: var_6}
    var_8 = {var_4: var_0, var_5: var_6}
    var_9 = 'not a type'
    var_10 = 'not a type'
    var_11 = 'not callable'



# Parsed testcases at query #52
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
    var_9 = 'a should be less than b'
    var_10 = lambda x: (x[var_0] < x[var_1], var_9)
    var_11 = [var_8, var_10]
    var_12 = module_0.check_global_invariants(var_4, var_11)
    var_13 = 3
    var_14 = {var_0: var_13, var_1: var_3}
    var_15 = lambda x: (x[var_0] < x[var_1], var_9)
    var_16 = [var_15]
    var_17 = module_0.check_global_invariants(var_14, var_16)
    var_18 = 'c'
    var_19 = {var_17: var_13, var_1: var_3, var_18: var_5}
    var_20 = lambda x: (x[var_17] < x[var_1], var_9)
    var_21 = 'b should be less than c'
    var_22 = lambda x: (x[var_1] < x[var_18], var_21)
    var_23 = [var_20, var_22]
    var_24 = module_0.check_global_invariants(var_19, var_23)



# Parsed testcases at query #53
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}
    var_5 = {var_2: var_0}
    var_6 = -1
    var_7 = {var_2: var_6}
    var_8 = {var_2: var_0}
    var_9 = {var_2: var_0}
    var_10 = {var_2: var_0}
    var_11 = {var_2: var_0}
    var_12 = False
    var_13 = {var_2: var_0}
    var_14 = {var_2: var_0}



# Parsed testcases at query #55
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
    var_10 = (var_9, var_6)
    var_11 = lambda x: var_10
    var_12 = [var_8, var_11]
    var_13 = module_0.check_global_invariants(var_4, var_12)
    var_14 = True
    var_15 = (var_14, var_6)
    var_16 = lambda x: var_15
    var_17 = False
    var_18 = 'error1'
    var_19 = (var_17, var_18)
    var_20 = lambda x: var_19
    var_21 = [var_16, var_20]
    var_22 = module_0.check_global_invariants(var_4, var_21)
    var_23 = (var_17, var_18)
    var_24 = lambda x: var_23
    var_25 = 'error2'
    var_26 = (var_17, var_25)
    var_27 = lambda x: var_26
    var_28 = [var_24, var_27]
    var_29 = module_0.check_global_invariants(var_4, var_28)
    var_30 = []
    var_31 = module_0.check_global_invariants(var_4, var_30)



# Parsed testcases at query #56
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = {var_4: var_0}
    var_7 = {var_4: var_0}
    var_8 = {var_4: var_0}
    var_9 = {var_4: var_0}
    var_10 = {var_4: var_0}
    var_11 = 'test_field'
    var_12 = 'a'
    var_13 = 'not an int'
    var_14 = {var_12: var_13}



# Parsed testcases at query #58
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #59
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}
    var_5 = 'test_field'
    var_6 = 'a'
    var_7 = 'not_an_int'
    var_8 = {var_6: var_7}
    var_9 = 'b'
    var_10 = 2
    var_11 = {var_2: var_0, var_9: var_10}
    var_12 = 'test_field'
    var_13 = False
    var_14 = 'test_field'
    var_15 = None



# Parsed testcases at query #61
#--------------------------




# Parsed testcases at query #62
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #63
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
    var_9 = 'test_field'
    var_10 = 'a'
    var_11 = 'not_int'
    var_12 = {var_10: var_11}
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = 'd'
    var_17 = 'e'
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = 4
    var_22 = 5
    var_23 = {var_13: var_18, var_14: var_19, var_15: var_20, var_16: var_21, var_17: var_22}



# Parsed testcases at query #64
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'data'
    var_8 = 'plain_value'



# Parsed testcases at query #65
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}
    var_5 = {var_2: var_0}
    var_6 = {var_2: var_0}
    var_7 = {var_2: var_0}
    var_8 = {var_2: var_0}
    var_9 = 'test'
    var_10 = 'a'
    var_11 = 'not an int'
    var_12 = {var_10: var_11}



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}
    var_5 = {var_2: var_0}
    var_6 = 10
    var_7 = range(var_6)
    var_8 = {str(i): i for i in var_7}
    var_9 = 'key1'
    var_10 = 'key2'
    var_11 = 2
    var_12 = {var_9: var_0, var_10: var_11}
    var_13 = 'TestClass'
    var_14 = ()
    var_15 = {}
    var_16 = 'test_field'
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]



# Parsed testcases at query #68
#--------------------------




# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = {}
    var_8 = 'test_field'
    var_9 = 'a'
    var_10 = 'not_int'
    var_11 = {var_9: var_10}
    var_12 = 'x'
    var_13 = {var_12: var_11}
    var_14 = {var_12: var_11}

def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = {}
    var_8 = 'test_field'
    var_9 = 'a'
    var_10 = 'not_int'
    var_11 = {var_9: var_10}
    var_12 = 'x'
    var_13 = {var_12: var_11}
    var_14 = {var_12: var_11}



# Parsed testcases at query #70
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = None
    var_6 = {var_1: var_0}
    var_7 = field.factory(var_6)[var_1]
    assert var_7 == 1
    var_8 = 'test_field'
    var_9 = 'a'
    var_10 = 'not_an_int'
    var_11 = {var_9: var_10}
    var_12 = 'x'
    var_13 = 10
    var_14 = {var_12: var_13}
    var_15 = {var_12: var_13}
    var_16 = [var_10]
    var_17 = module_0.check_global_invariants(var_9, var_16)



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = {var_2: var_0, var_3: var_4}
    var_7 = 'test_field'
    var_8 = 'a'
    var_9 = 'not_an_int'
    var_10 = {var_8: var_9}



# Parsed testcases at query #72
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #73
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #74
#--------------------------




# Parsed testcases at query #75
#--------------------------




# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = None
    var_6 = 'test_field'
    var_7 = 'a'
    var_8 = 'not_int'
    var_9 = {var_7: var_8}
    var_10 = 'test_field'
    var_11 = {var_1: var_0, var_2: var_3}
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = 'd'
    var_16 = 'e'
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = 4
    var_21 = 5
    var_22 = {var_12: var_17, var_13: var_18, var_14: var_19, var_15: var_20, var_16: var_21}



# Parsed testcases at query #77
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = {}
    var_3 = 'b'
    var_4 = 2
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = {}
    var_8 = 'test_field'
    var_9 = 'a'
    var_10 = 'not_int'
    var_11 = {var_9: var_10}

def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = {}
    var_3 = 'b'
    var_4 = 2
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = {}
    var_8 = 'test_field'
    var_9 = 'a'
    var_10 = 'not_int'
    var_11 = {var_9: var_10}



# Parsed testcases at query #79
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #80
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #81
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = 'a'
    var_3 = {}
    var_4 = 'b'
    var_5 = 2
    var_6 = 'not a type'
    var_7 = 'not a type'
    var_8 = 'json'
    var_9 = module_0.serialize(var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = 'a'
    var_3 = {}
    var_4 = 'b'
    var_5 = 2
    var_6 = 'not a type'
    var_7 = 'not a type'
    var_8 = 'json'
    var_9 = module_0.serialize(var_8)



# Parsed testcases at query #82
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 42
    var_8 = 'test'



# Parsed testcases at query #83
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #84
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #85
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #86
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_4: var_0, var_5: var_6}
    var_8 = {var_4: var_0}
    var_9 = field.factory(var_8)[var_4]
    assert var_9 == 1
    var_10 = 'x'
    var_11 = 10
    var_12 = {var_10: var_11}
    var_13 = 'test_field'
    var_14 = 'a'
    var_15 = 'not_int'
    var_16 = {var_14: var_15}
    var_17 = 'not_a_type'
    var_18 = 'not_a_type'



# Parsed testcases at query #87
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'



# Parsed testcases at query #88
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = 2
    var_3 = 'one'
    var_4 = 'two'
    var_5 = None

def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = 2
    var_3 = 'one'
    var_4 = 'two'
    var_5 = None



# Parsed testcases at query #90
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #91
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #92
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #93
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
    var_9 = [var_8]
    var_10 = module_0.check_global_invariants(var_4, var_9)
    var_11 = {var_0: var_5, var_1: var_3}
    var_12 = False
    var_13 = 'error_code'
    var_14 = (var_12, var_13)
    var_15 = lambda x: var_14
    var_16 = [var_15]
    var_17 = module_0.check_global_invariants(var_11, var_16)
    var_18 = {var_17: var_5, var_1: var_3}
    var_19 = True
    var_20 = (var_19, var_6)
    var_21 = lambda x: var_20
    var_22 = 'error_code1'
    var_23 = (var_12, var_22)
    var_24 = lambda x: var_23
    var_25 = 'error_code2'
    var_26 = (var_12, var_25)
    var_27 = lambda x: var_26
    var_28 = [var_21, var_24, var_27]
    var_29 = module_0.check_global_invariants(var_18, var_28)



# Parsed testcases at query #94
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = {}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = 1
    var_7 = 'a'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 0
    var_1 = 'Map must not be empty'
    var_2 = {}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = 1
    var_7 = 'a'
    var_8 = {var_6: var_7}



# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = None

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = None



# Parsed testcases at query #96
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #97
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)
    var_5 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #98
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_4: var_0, var_5: var_6}
    var_8 = False
    var_9 = 'test_field'
    var_10 = 'a'
    var_11 = 'not_int'
    var_12 = {var_10: var_11}



# Parsed testcases at query #99
#--------------------------


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedPMap()
    var_1 = True
    var_2 = None
    var_3 = module_0.CheckedPMap()
    var_4 = {}
    var_5 = 'a'
    var_6 = {var_5: var_1}
    var_7 = {var_5: var_1}
    var_8 = 'b'
    var_9 = 2
    var_10 = {var_5: var_1, var_8: var_9}
    var_11 = 'test_field'
    var_12 = 123
    var_13 = 'test_field'
    var_14 = 'a'
    var_15 = 'b'
    var_16 = {var_14: var_15}
    var_17 = 'test_field'
    var_18 = {var_5: var_1}



# Parsed testcases at query #100
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'format'
    var_5 = module_0.serialize(var_4)
    var_6 = 'json'
    var_7 = 'test_value'
    var_8 = 'plain_value'



# Parsed testcases at query #101
#--------------------------


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)
    var_3 = module_1.check_global_invariants(var_0, var_1)
    var_4 = module_1.check_global_invariants(var_0, var_1)



