####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'field'
    var_12 = 'invalid'
    var_13 = module_0.pmap()



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'invalid'
    var_12 = None
    var_13 = module_0.pmap()
    var_14 = None
    var_15 = module_0.pmap()
    var_16 = True



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 'x'
    var_7 = 'y'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = 'z'
    var_10 = 30
    var_11 = {var_6: var_2, var_7: var_3, var_9: var_10}
    var_12 = True
    var_13 = lambda : 1
    var_14 = 2



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'z'
    var_6 = 20
    var_7 = 0
    var_8 = lambda self: (self.x >= 0, 'x must be non-negative')
    var_9 = module_0.pmap()
    var_10 = -1
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = 'y'
    var_14 = 5
    var_15 = 0
    var_16 = module_0.pmap()
    var_17 = module_0.pmap()
    var_18 = True
    var_19 = 'z'
    var_20 = 20



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0.pmap()
    var_3 = 0
    var_4 = 1
    var_5 = module_0.pmap()
    var_6 = 0
    var_7 = module_0.pmap()
    var_8 = 5
    var_9 = 15
    var_10 = 0
    var_11 = 0
    var_12 = module_0.pmap()



# Parsed testcases at query #3
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = module_0.serialize()
    var_4 = 'custom'
    var_5 = module_0.serialize(var_4)
    var_6 = module_0.field()
    var_7 = module_0.field()
    var_8 = module_0.serialize()



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = 0
    var_13 = module_0.pmap()
    var_14 = 0
    var_15 = module_0.pmap()
    var_16 = 5
    var_17 = module_0.pmap()
    var_18 = True
    var_19 = 'invalid_field'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = lambda self: len(self.name) > 0
    var_1 = None
    var_2 = 0
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 2
    var_7 = 0
    var_8 = lambda : 1
    var_9 = 2
    var_10 = 0
    var_11 = 1
    var_12 = 'x'
    var_13 = 0
    var_14 = 1
    var_15 = True
    var_16 = 30
    var_17 = 0
    var_18 = 1
    var_19 = 2
    var_20 = (var_12, var_2)
    var_21 = 'y'
    var_22 = (var_21, var_3)
    var_23 = [var_20, var_22]



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = lambda : 2
    var_9 = 3
    var_10 = 1
    var_11 = 2
    var_12 = 'x'
    var_13 = 'y'
    var_14 = [var_12, var_13]
    var_15 = 1
    var_16 = 2
    var_17 = True
    var_18 = 30
    var_19 = 2
    var_20 = {var_12: var_2, var_13: var_3}
    var_21 = module_0.pmap(var_20)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 5
    var_7 = 0
    var_8 = lambda : 100
    var_9 = 0
    var_10 = 1
    var_11 = module_0.pmap()
    var_12 = 'x'
    var_13 = 0
    var_14 = 1
    var_15 = 30
    var_16 = True
    var_17 = 0
    var_18 = 1
    var_19 = 10
    var_20 = 0
    var_21 = 1



# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'invalid'
    var_12 = None
    var_13 = module_0.pmap()



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 0
    var_4 = 0
    var_5 = 0
    var_6 = module_0.pmap()
    var_7 = 0
    var_8 = 0
    var_9 = module_0.pmap()
    var_10 = 1
    var_11 = 2
    var_12 = 0
    var_13 = 0
    var_14 = module_0.pmap()
    var_15 = 0
    var_16 = 0
    var_17 = module_0.pmap()



# Parsed testcases at query #11
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = module_0.pmap()
    var_13 = module_0.pmap()
    var_14 = 5
    var_15 = module_0.pmap()
    var_16 = True
    var_17 = 'extra_field'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = lambda : 2
    var_9 = 3
    var_10 = 1
    var_11 = 2
    var_12 = module_0.pmap()
    var_13 = 'x'
    var_14 = 1
    var_15 = 2
    var_16 = module_0.pmap()
    var_17 = True
    var_18 = 1
    var_19 = 2
    var_20 = module_0.pmap()
    var_21 = None
    var_22 = False
    var_23 = 1
    var_24 = 2
    var_25 = module_0.pmap()



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'default'
    var_5 = 0
    var_6 = 'test'
    var_7 = 42



# Parsed testcases at query #15
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = module_0.pmap()
    var_13 = module_0.pmap()
    var_14 = 5
    var_15 = module_0.pmap()
    var_16 = True



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = lambda self: len(self.name) > 0
    var_1 = None
    var_2 = 0
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()



# Parsed testcases at query #17
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = lambda : 2
    var_9 = 3
    var_10 = 1
    var_11 = 2
    var_12 = module_0.pmap()
    var_13 = 'x'
    var_14 = 1
    var_15 = 2
    var_16 = module_0.pmap()
    var_17 = True
    var_18 = 1
    var_19 = 2
    var_20 = 2
    var_21 = module_0.pmap()
    var_22 = var_21._buckets



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = 1
    var_2 = 2
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()
    var_8 = lambda self: True
    var_9 = 1
    var_10 = True
    var_11 = (var_9, var_10)
    var_12 = lambda self: True
    var_13 = lambda : 1
    var_14 = 2
    var_15 = 'x'
    var_16 = lambda self: True
    var_17 = 1
    var_18 = 2



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'x'
    var_5 = 100
    var_6 = True
    var_7 = 30
    var_8 = lambda : 1
    var_9 = 2



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = 1
    var_2 = 2
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = '_precord_mandatory_fields'
    var_3 = '_precord_initial_values'
    var_4 = None
    var_5 = None
    var_6 = set()
    var_7 = None
    var_8 = None
    var_9 = lambda x: (True, None)
    var_10 = lambda x: (True, None)
    var_11 = set()
    var_12 = 1
    var_13 = lambda : 2
    var_14 = None
    var_15 = lambda x: (x > 0, 'must be positive')



# Parsed testcases at query #22
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = module_0.serialize()
    var_4 = None
    var_5 = module_0.serialize()



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 10
    var_5 = 20
    var_6 = None
    var_7 = 'data'
    var_8 = 'test'
    var_9 = None
    var_10 = 1
    var_11 = 2
    var_12 = True
    var_13 = None
    var_14 = None
    var_15 = -1
    var_16 = 'Bob'
    var_17 = 25



# Parsed testcases at query #24
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = module_0.pmap()
    var_4 = 1
    var_5 = 2
    var_6 = 0
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = 0
    var_12 = 0
    var_13 = module_0.pmap()



# Parsed testcases at query #25
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 5
    var_7 = 0
    var_8 = lambda : 100
    var_9 = 5
    var_10 = 0
    var_11 = 1
    var_12 = module_0.pmap()
    var_13 = 'x'
    var_14 = 'y'
    var_15 = 'extra'
    var_16 = 30
    var_17 = {var_13: var_2, var_14: var_3, var_15: var_16}
    var_18 = True
    var_19 = 2
    var_20 = 100
    var_21 = 200
    var_22 = {var_13: var_20, var_14: var_21}
    var_23 = module_0.pmap(var_22)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0



# Parsed testcases at query #27
#--------------------------


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_1.pmap()
    var_3 = module_0.field()
    var_4 = module_1.pmap()
    var_5 = module_1.pmap()
    var_6 = lambda self: (self.x != self.y, 'x != y')
    var_7 = module_0.field()
    var_8 = module_0.field()
    var_9 = module_1.pmap()
    var_10 = 10
    var_11 = 20



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #29
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 0
    var_4 = 0
    var_5 = module_0.pmap()
    var_6 = 0
    var_7 = module_0.pmap()
    var_8 = 0
    var_9 = 0
    var_10 = 1
    var_11 = 2
    var_12 = 0
    var_13 = 0
    var_14 = module_0.pmap()



# Parsed testcases at query #30
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = module_0.pmap()
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = 0
    var_10 = 0
    var_11 = module_0.pmap()
    var_12 = module_1.field()
    var_13 = module_1.field()
    var_14 = module_0.pmap()
    var_15 = 'x'



# Parsed testcases at query #31
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 20
    var_6 = 'y'
    var_7 = 30
    var_8 = 'z'
    var_9 = 40
    var_10 = 0
    var_11 = module_0.pmap()
    var_12 = -1
    var_13 = 0
    var_14 = 1
    var_15 = module_0.pmap()



# Parsed testcases at query #32
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 'localhost'
    var_5 = 8080
    var_6 = lambda : 1234567890
    var_7 = None
    var_8 = None
    var_9 = module_0.pmap()
    var_10 = 'x'
    var_11 = None
    var_12 = 1
    var_13 = 2
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 1
    var_17 = 2
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = True
    var_20 = 'Bob'
    var_21 = 25



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 3
    var_7 = 'x'
    var_8 = [var_7]
    var_9 = True
    var_10 = 30
    var_11 = 2
    var_12 = (var_7, var_2)
    var_13 = 'y'
    var_14 = (var_13, var_3)
    var_15 = [var_12, var_14]



# Parsed testcases at query #34
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 0
    var_4 = 0



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #36
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 42
    var_4 = module_0.serialize()
    var_5 = 'custom_format'
    var_6 = module_0.serialize(var_5)



# Parsed testcases at query #37
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = module_0.pmap()
    var_13 = 'x'
    var_14 = 'not an int'



# Parsed testcases at query #38
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 10
    var_5 = 'default'
    var_6 = 20
    var_7 = 'custom'
    var_8 = lambda : 1234567890
    var_9 = None
    var_10 = None
    var_11 = module_0.pmap()
    var_12 = 'a'
    var_13 = None
    var_14 = 'value1'
    var_15 = 'should_be_ignored'
    var_16 = True
    var_17 = 'Bob'
    var_18 = 25



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0



# Parsed testcases at query #40
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'z'
    var_6 = 20
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = -5
    var_10 = 0
    var_11 = None
    var_12 = module_0.pmap()
    var_13 = 0
    var_14 = module_0.pmap()
    var_15 = 'x'
    var_16 = 'not_an_int'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = None
    var_2 = None
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = None
    var_8 = None
    var_9 = None
    var_10 = 1
    var_11 = None



# Parsed testcases at query #42
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'default'
    var_2 = 42
    var_3 = 'test'
    var_4 = module_0.serialize()
    var_5 = 'json'
    var_6 = module_0.serialize(var_5)
    var_7 = 0
    var_8 = 'default'
    var_9 = module_0.serialize()



# Parsed testcases at query #43
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = module_0.pmap()
    var_7 = 0
    var_8 = lambda self: (self.x > 0, 'x must be positive')
    var_9 = module_0.pmap()
    var_10 = 0
    var_11 = 0
    var_12 = module_0.pmap()



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'default'
    var_5 = 0
    var_6 = 'test'
    var_7 = 42



# Parsed testcases at query #45
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = lambda : 2
    var_9 = 3
    var_10 = 1
    var_11 = 2
    var_12 = module_0.pmap()
    var_13 = 'x'
    var_14 = 1
    var_15 = 2
    var_16 = module_0.pmap()
    var_17 = True
    var_18 = 2
    var_19 = 'y'
    var_20 = {var_13: var_2, var_19: var_3}
    var_21 = module_0.pmap(var_20)
    var_22 = var_21._buckets



# Parsed testcases at query #46
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = module_0.pmap()
    var_6 = 'y'
    var_7 = 20
    var_8 = module_0.pmap()
    var_9 = module_0.pmap()
    var_10 = 'z'
    var_11 = 30
    var_12 = 0
    var_13 = 0
    var_14 = lambda self: (self.x >= 0, 'x must be non-negative')
    var_15 = module_0.pmap()
    var_16 = -1
    var_17 = module_0.pmap()
    var_18 = 1



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = 1
    var_2 = 2
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()
    var_8 = lambda self: True
    var_9 = 1
    var_10 = 2
    var_11 = lambda self: True
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = lambda self: True
    var_16 = lambda : 1
    var_17 = 2
    var_18 = 'x'



# Parsed testcases at query #48
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = module_0.pmap()
    var_6 = 'y'
    var_7 = 20
    var_8 = module_0.pmap()
    var_9 = 'z'
    var_10 = 30
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = -1
    var_14 = 0
    var_15 = module_0.pmap()



# Parsed testcases at query #49
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 0
    var_4 = 0
    var_5 = module_0.pmap()
    var_6 = 0
    var_7 = module_0.pmap()
    var_8 = 0
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = 1
    var_12 = 2



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = lambda self: len(self.name) > 0
    var_1 = None
    var_2 = 0
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()



# Parsed testcases at query #51
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'hello'
    var_2 = module_0.serialize()
    var_3 = 10
    var_4 = 'world'
    var_5 = 'custom'
    var_6 = module_0.serialize(var_5)
    var_7 = 7
    var_8 = 'test'
    var_9 = module_0.serialize()



# Parsed testcases at query #52
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = module_0.pmap()
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = 0
    var_10 = 0
    var_11 = module_0.pmap()
    var_12 = module_0.pmap()



# Parsed testcases at query #53
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'invalid'
    var_12 = module_0.pmap()



# Parsed testcases at query #54
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = -1



# Parsed testcases at query #55
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = module_0.pmap()
    var_6 = 'y'
    var_7 = 20
    var_8 = module_0.pmap()
    var_9 = 'z'
    var_10 = 30
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = -1
    var_14 = 0
    var_15 = module_0.pmap()



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 'x'
    var_5 = 30
    var_6 = True



# Parsed testcases at query #57
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'field'
    var_12 = 'invalid'
    var_13 = None
    var_14 = module_0.pmap()



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'



# Parsed testcases at query #59
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = 0
    var_11 = module_0.pmap()
    var_12 = module_0.pmap()
    var_13 = 30



# Parsed testcases at query #60
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 1
    var_5 = 'y'
    var_6 = 2
    var_7 = module_0.pmap()
    var_8 = 0
    var_9 = 0
    var_10 = 0
    var_11 = module_0.pmap()
    var_12 = 0
    var_13 = 0
    var_14 = lambda self: self.x + self.y > 0
    var_15 = module_0.pmap()
    var_16 = -1
    var_17 = -2
    var_18 = 0
    var_19 = 0
    var_20 = module_0.pmap()
    var_21 = -1
    var_22 = -2



# Parsed testcases at query #61
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()
    var_8 = None
    var_9 = None
    var_10 = 'json'
    var_11 = module_0.serialize(var_10)
    var_12 = module_0.serialize()
    var_13 = None
    var_14 = None
    var_15 = None
    var_16 = 'nested_value1'
    var_17 = 'nested_value2'
    var_18 = module_0.serialize()



# Parsed testcases at query #62
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = 2
    var_9 = module_0.pmap()
    var_10 = 'x'
    var_11 = 1
    var_12 = 2
    var_13 = 30
    var_14 = True
    var_15 = 1
    var_16 = 2
    var_17 = module_0.pmap()
    var_18 = var_17._size
    var_19 = var_17._buckets



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = set()



# Parsed testcases at query #64
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()



# Parsed testcases at query #65
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = 1
    var_2 = 2
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()
    var_8 = lambda self: True
    var_9 = 1
    var_10 = 2
    var_11 = None
    var_12 = lambda self: True
    var_13 = 1
    var_14 = 2
    var_15 = None
    var_16 = lambda self: True
    var_17 = 1
    var_18 = 2
    var_19 = lambda : 3
    var_20 = lambda self: True
    var_21 = 1
    var_22 = 2



# Parsed testcases at query #67
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = None
    var_8 = None
    var_9 = module_0.pmap()
    var_10 = 'a'
    var_11 = None
    var_12 = 1
    var_13 = 2
    var_14 = True



# Parsed testcases at query #68
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()



# Parsed testcases at query #69
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = None
    var_11 = module_0.pmap()
    var_12 = 'invalid'
    var_13 = None
    var_14 = module_0.pmap()
    var_15 = 'optional_field'
    var_16 = 'value'



# Parsed testcases at query #70
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'



# Parsed testcases at query #72
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = lambda : 1
    var_5 = 2
    var_6 = 'x'
    var_7 = 1
    var_8 = 30
    var_9 = True
    var_10 = 2
    var_11 = 'y'
    var_12 = {var_6: var_9, var_11: var_10}
    var_13 = module_0.pmap(var_12)



# Parsed testcases at query #73
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = 0
    var_13 = module_0.pmap()



# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = '_precord_mandatory_fields'
    var_3 = '_precord_initial_values'
    var_4 = 1
    var_5 = 2
    var_6 = set()
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 1
    var_11 = 2
    var_12 = lambda self: True



# Parsed testcases at query #75
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 1
    var_5 = module_0.pmap()
    var_6 = 'y'
    var_7 = 2
    var_8 = module_0.pmap()
    var_9 = 'z'
    var_10 = 3
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = -1
    var_14 = 0
    var_15 = module_0.pmap()



# Parsed testcases at query #76
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = 1
    var_10 = 2
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = 1
    var_15 = lambda self: (True, 'error') if self.x > 0 else (False, 'error')



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = lambda self: len(self) > 0
    var_1 = 1
    var_2 = 2
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = set()
    var_7 = 1
    var_8 = 2
    var_9 = lambda self: True



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'



# Parsed testcases at query #81
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #82
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = 1
    var_7 = 2
    var_8 = None
    var_9 = 42
    var_10 = lambda self: True



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = 1
    var_2 = 2
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()
    var_8 = lambda self: True
    var_9 = 1
    var_10 = 2
    var_11 = None
    var_12 = lambda self: True
    var_13 = 1
    var_14 = 2
    var_15 = None
    var_16 = lambda self: True
    var_17 = 1
    var_18 = 2



# Parsed testcases at query #84
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()



# Parsed testcases at query #85
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 3
    var_7 = 'x'
    var_8 = 'y'
    var_9 = 2
    var_10 = {var_7: var_2, var_8: var_3}
    var_11 = module_0.pmap(var_10)
    var_12 = var_11._buckets
    var_13 = 'z'
    var_14 = 30
    var_15 = {var_7: var_2, var_8: var_3, var_13: var_14}
    var_16 = True



# Parsed testcases at query #86
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = module_0.pmap()
    var_13 = 5
    var_14 = module_0.pmap()
    var_15 = True
    var_16 = 'extra_field'
    var_17 = 30



# Parsed testcases at query #87
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = lambda : 2
    var_9 = 3
    var_10 = module_0.pmap()
    var_11 = 'x'
    var_12 = module_0.pmap()
    var_13 = True
    var_14 = 2
    var_15 = module_0.pmap()
    var_16 = var_15._buckets



# Parsed testcases at query #88
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = None
    var_2 = None
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'



# Parsed testcases at query #89
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 5
    var_7 = 0
    var_8 = lambda : 100
    var_9 = 0
    var_10 = 1
    var_11 = module_0.pmap()
    var_12 = 'x'
    var_13 = 0
    var_14 = 1
    var_15 = 30
    var_16 = True
    var_17 = 0
    var_18 = 1
    var_19 = 2
    var_20 = 'y'
    var_21 = {var_12: var_2, var_20: var_3}
    var_22 = module_0.pmap(var_21)
    var_23 = var_22._buckets



# Parsed testcases at query #90
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = set()
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = set()
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = 1
    var_15 = 2
    var_16 = lambda : 3
    var_17 = 'z'
    var_18 = 1
    var_19 = 2
    var_20 = 1
    var_21 = 2



# Parsed testcases at query #91
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #92
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = lambda self: (self.x >= 0, 'x must be non-negative')
    var_11 = module_0.pmap()
    var_12 = -1
    var_13 = 0
    var_14 = 0
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = module_0.pmap()



# Parsed testcases at query #93
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'default'
    var_5 = 0
    var_6 = 'test'
    var_7 = 42



# Parsed testcases at query #94
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 3
    var_7 = 'x'
    var_8 = 'y'
    var_9 = {var_7: var_2, var_8: var_3}
    var_10 = 'extra'
    var_11 = 30
    var_12 = {var_7: var_2, var_8: var_3, var_10: var_11}
    var_13 = True
    var_14 = 2
    var_15 = {var_7: var_2, var_8: var_3}
    var_16 = module_0.pmap(var_15)
    var_17 = var_16._buckets
    var_18 = 0
    var_19 = lambda : 1
    var_20 = 3



# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = '_precord_mandatory_fields'
    var_3 = '_precord_initial_values'
    var_4 = None
    var_5 = 0
    var_6 = set()
    var_7 = None
    var_8 = 0
    var_9 = 42
    var_10 = None



# Parsed testcases at query #96
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #97
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = 2
    var_9 = module_0.pmap()
    var_10 = 'x'
    var_11 = 1
    var_12 = 2
    var_13 = module_0.pmap()
    var_14 = True
    var_15 = 1
    var_16 = 2
    var_17 = 2
    var_18 = module_0.pmap()
    var_19 = var_18._buckets



# Parsed testcases at query #98
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = 1
    var_2 = 2
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()
    var_8 = lambda self: True
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = set()
    var_13 = lambda self: True
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = lambda self: True
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = set()
    var_22 = lambda self: True
    var_23 = 1
    var_24 = 2
    var_25 = 3



# Parsed testcases at query #99
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = set()



# Parsed testcases at query #100
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = None
    var_13 = None
    var_14 = module_0.pmap()
    var_15 = None
    var_16 = None
    var_17 = module_0.pmap()



# Parsed testcases at query #101
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'invalid'
    var_12 = None
    var_13 = module_0.pmap()
    var_14 = None
    var_15 = module_0.pmap()
    var_16 = True
    var_17 = 'extra_field'
    var_18 = 'extra_value'



# Parsed testcases at query #102
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = module_0.pmap()
    var_6 = 'y'
    var_7 = 20
    var_8 = module_0.pmap()
    var_9 = 'z'
    var_10 = 30
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = -1
    var_14 = 0
    var_15 = module_0.pmap()



# Parsed testcases at query #103
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = 0
    var_13 = 1
    var_14 = module_0.pmap()
    var_15 = 0
    var_16 = module_0.pmap()
    var_17 = 5
    var_18 = 0
    var_19 = module_0.pmap()
    var_20 = True



# Parsed testcases at query #104
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 5



# Parsed testcases at query #105
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = lambda : 2
    var_9 = 3
    var_10 = 1
    var_11 = 2
    var_12 = module_0.pmap()
    var_13 = 'x'
    var_14 = 1
    var_15 = 2
    var_16 = module_0.pmap()
    var_17 = True
    var_18 = 1
    var_19 = 2
    var_20 = 2
    var_21 = module_0.pmap()
    var_22 = var_21._buckets



# Parsed testcases at query #106
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 'x'
    var_5 = 15
    var_6 = 0
    var_7 = 1
    var_8 = module_0.pmap()
    var_9 = 'optional_field'
    var_10 = 0
    var_11 = module_0.pmap()
    var_12 = 'positive'
    var_13 = -1
    var_14 = 5
    var_15 = 0
    var_16 = 0
    var_17 = 3
    var_18 = module_0.pmap()



# Parsed testcases at query #107
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()
    var_8 = None
    var_9 = None
    var_10 = 'upper'
    var_11 = module_0.serialize(var_10)



# Parsed testcases at query #108
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = 1
    var_7 = 2
    var_8 = 1
    var_9 = 2
    var_10 = 1
    var_11 = 2
    var_12 = 1



# Parsed testcases at query #109
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = module_0.serialize()
    var_4 = 'field2'
    var_5 = record.serialize()[var_4]
    assert var_5 == 'VALUE2'
    var_6 = 'test'
    var_7 = module_0.serialize(var_6)



# Parsed testcases at query #110
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'default'
    var_5 = 0
    var_6 = 'test'
    var_7 = 42



# Parsed testcases at query #111
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'z'
    var_6 = 20
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = -1
    var_10 = 0
    var_11 = None
    var_12 = module_0.pmap()
    var_13 = 0
    var_14 = module_0.pmap()
    var_15 = 5



# Parsed testcases at query #112
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 30



# Parsed testcases at query #113
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'z'
    var_6 = 20
    var_7 = 0
    var_8 = lambda self: ('x must be positive', self.x > 0)
    var_9 = module_0.pmap()
    var_10 = -1
    var_11 = 0
    var_12 = 0
    var_13 = 0
    var_14 = module_0.pmap()
    var_15 = module_0.pmap()
    var_16 = True



# Parsed testcases at query #114
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'invalid'
    var_12 = None
    var_13 = module_0.pmap()



# Parsed testcases at query #115
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'z'
    var_6 = 20
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = -1
    var_10 = 0
    var_11 = module_0.pmap()
    var_12 = 5
    var_13 = module_0.pmap()
    var_14 = True



# Parsed testcases at query #116
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = 1
    var_2 = 2
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = set()
    var_12 = 1
    var_13 = 2
    var_14 = 3



# Parsed testcases at query #117
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0.pmap()
    var_3 = module_0.pmap()
    var_4 = 1
    var_5 = module_0.pmap()
    var_6 = 1
    var_7 = 2
    var_8 = module_0.pmap()
    var_9 = 5
    var_10 = 15



# Parsed testcases at query #118
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()
    var_8 = None
    var_9 = None
    var_10 = 'upper'
    var_11 = module_0.serialize(var_10)



# Parsed testcases at query #119
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 0
    var_4 = 0
    var_5 = module_0.pmap()
    var_6 = 0
    var_7 = 0
    var_8 = lambda self: self.x + self.y > 0
    var_9 = module_0.pmap()
    var_10 = 1
    var_11 = 2



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 10
    var_5 = 20
    var_6 = 5
    var_7 = lambda : 12345
    var_8 = 100
    var_9 = None
    var_10 = None
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 1
    var_14 = 2
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = None
    var_17 = None
    var_18 = 'field1'
    var_19 = 'field2'
    var_20 = 'extra'
    var_21 = 3
    var_22 = {var_18: var_13, var_19: var_14, var_20: var_21}
    var_23 = True
    var_24 = 'Bob'
    var_25 = 25
    var_26 = None
    var_27 = 'x'
    var_28 = 100
    var_29 = (var_27, var_28)
    var_30 = [var_29]



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = 0
    var_5 = 0
    var_6 = None
    var_7 = module_0.pmap()
    var_8 = 0
    var_9 = 0
    var_10 = lambda self: (self.x >= 0, 'x must be non-negative')
    var_11 = module_0.pmap()
    var_12 = 0
    var_13 = 0
    var_14 = module_0.pmap()



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = 0
    var_13 = 0
    var_14 = module_0.pmap()



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 0
    var_4 = module_0.pmap()
    var_5 = 0
    var_6 = module_0.pmap()
    var_7 = 1
    var_8 = 2
    var_9 = 0
    var_10 = 0
    var_11 = module_0.pmap()
    var_12 = module_0.pmap()



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = lambda self: (self.x >= 0, 'x must be non-negative')
    var_11 = module_0.pmap()
    var_12 = -1
    var_13 = 0
    var_14 = 1
    var_15 = 'x'
    var_16 = 'y'
    var_17 = {var_15, var_16}
    var_18 = module_0.pmap()
    var_19 = 0
    var_20 = 'x'
    var_21 = module_0.pmap()
    var_22 = 'x'
    var_23 = 'not an int'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 5
    var_7 = 0
    var_8 = lambda : 100
    var_9 = 0
    var_10 = 1
    var_11 = module_0.pmap()
    var_12 = 'x'
    var_13 = 0
    var_14 = 1
    var_15 = 30
    var_16 = True
    var_17 = 0
    var_18 = 1
    var_19 = 2
    var_20 = 'y'
    var_21 = {var_12: var_2, var_20: var_3}
    var_22 = module_0.pmap(var_21)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'invalid'
    var_12 = None
    var_13 = module_0.pmap()



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = 1
    var_2 = 2
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'invalid'
    var_12 = module_0.pmap()



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = '_precord_mandatory_fields'
    var_3 = '_precord_initial_values'
    var_4 = 1
    var_5 = 2
    var_6 = set()
    var_7 = 1
    var_8 = 2
    var_9 = set()
    var_10 = 1
    var_11 = 2
    var_12 = lambda self: True
    var_13 = 1
    var_14 = 2



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 10
    var_3 = 'test'



# Parsed testcases at query #13
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_2: var_4, var_3: var_5}
    var_9 = module_0.pmap(var_8)
    var_10 = 10
    var_11 = 1
    var_12 = 2
    var_13 = None
    var_14 = {var_2: var_4, var_3: var_5}
    var_15 = module_0.pmap(var_14)
    var_16 = 1
    var_17 = 2
    var_18 = lambda self: (self.x > 0, 'x must be positive')
    var_19 = -1
    var_20 = {var_2: var_19, var_3: var_5}
    var_21 = module_0.pmap(var_20)
    var_22 = 1
    var_23 = 2
    var_24 = lambda self: (self.x + self.y > 0, 'sum must be positive')
    var_25 = -1
    var_26 = -2
    var_27 = {var_2: var_25, var_3: var_26}
    var_28 = module_0.pmap(var_27)
    var_29 = {var_2: var_4, var_3: var_5}
    var_30 = module_0.pmap(var_29)



# Parsed testcases at query #14
#--------------------------


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_1.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = module_1.pmap()
    var_10 = -1
    var_11 = module_1.pmap()
    var_12 = 'x'
    var_13 = 'not an int'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = 2
    var_9 = 'x'
    var_10 = 1
    var_11 = 2
    var_12 = True
    var_13 = 30
    var_14 = lambda : 1
    var_15 = 2
    var_16 = 1
    var_17 = 2



# Parsed testcases at query #16
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = 0
    var_13 = 0
    var_14 = module_0.pmap()
    var_15 = 0
    var_16 = ''
    var_17 = module_0.pmap()
    var_18 = 'y'
    var_19 = 123
    var_20 = 0
    var_21 = module_0.pmap()
    var_22 = True



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 30



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 1
    var_5 = 2
    var_6 = None



# Parsed testcases at query #19
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = module_0.pmap()
    var_6 = 'y'
    var_7 = 20
    var_8 = module_0.pmap()
    var_9 = 'z'
    var_10 = 30
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = -1
    var_14 = 0
    var_15 = module_0.pmap()
    var_16 = module_0.pmap()
    var_17 = 5
    var_18 = 0
    var_19 = module_0.pmap()
    var_20 = True
    var_21 = 'a'
    var_22 = 'b'
    var_23 = 2
    var_24 = {var_21: var_20, var_22: var_23}



# Parsed testcases at query #20
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 0
    var_4 = 0
    var_5 = 0
    var_6 = module_0.pmap()
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = 1
    var_10 = 2
    var_11 = 0
    var_12 = 0
    var_13 = module_0.pmap()
    var_14 = module_0.pmap()



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #23
#--------------------------


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_1.pmap()
    var_2 = 'field1'
    var_3 = 'value1'
    var_4 = 'field2'
    var_5 = 'value2'
    var_6 = module_1.pmap()
    var_7 = module_1.pmap()
    var_8 = -1
    var_9 = module_0.field()
    var_10 = module_0.field()
    var_11 = lambda self: (self.field1 != self.field2, 'fields must differ')
    var_12 = module_1.pmap()
    var_13 = 'same'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = None
    var_2 = None
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()
    var_8 = None
    var_9 = None
    var_10 = True
    var_11 = (var_9, var_10)
    var_12 = 42
    var_13 = lambda : 'initial'



# Parsed testcases at query #25
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = module_0.pmap()
    var_6 = 'y'
    var_7 = 20
    var_8 = module_0.pmap()
    var_9 = 'z'
    var_10 = 30
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = -1
    var_14 = 0
    var_15 = 1
    var_16 = module_0.pmap()



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 10
    var_5 = 20
    var_6 = 5
    var_7 = None
    var_8 = None
    var_9 = 'a'
    var_10 = 'b'



# Parsed testcases at query #27
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'z'
    var_6 = 20
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = -1
    var_10 = module_0.pmap()
    var_11 = 5
    var_12 = module_0.pmap()
    var_13 = True
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 2
    var_17 = {var_14: var_13, var_15: var_16}



# Parsed testcases at query #28
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 5
    var_7 = 0
    var_8 = lambda : 100
    var_9 = 'x'
    var_10 = 'y'
    var_11 = {var_9, var_10}
    var_12 = True
    var_13 = 30
    var_14 = 2
    var_15 = {var_9: var_2, var_10: var_3}
    var_16 = module_0.pmap(var_15)



# Parsed testcases at query #29
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = module_0.pmap()
    var_13 = module_0.pmap()
    var_14 = 5
    var_15 = module_0.pmap()
    var_16 = True
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 2
    var_20 = {var_17: var_16, var_18: var_19}



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'default'
    var_5 = None
    var_6 = 'test'
    var_7 = None



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = set()



# Parsed testcases at query #32
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = lambda : 2
    var_9 = 3
    var_10 = 1
    var_11 = 2
    var_12 = module_0.pmap()
    var_13 = 'x'
    var_14 = 1
    var_15 = 2
    var_16 = module_0.pmap()
    var_17 = True
    var_18 = 1
    var_19 = 2
    var_20 = 2
    var_21 = module_0.pmap()
    var_22 = var_21._buckets



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = None
    var_8 = None
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 1
    var_12 = 2
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = None
    var_15 = 'field1'
    var_16 = 'extra'
    var_17 = {var_15: var_11, var_16: var_12}
    var_18 = True
    var_19 = 0
    var_20 = ()



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'default'
    var_5 = 'test'



# Parsed testcases at query #35
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = lambda self: self.x >= 0
    var_11 = module_0.pmap()
    var_12 = -1
    var_13 = 0
    var_14 = 0
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = module_0.pmap()



# Parsed testcases at query #36
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = module_0.pmap()
    var_2 = module_0.pmap()
    var_3 = 5
    var_4 = 'original'



# Parsed testcases at query #37
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'name'
    var_4 = 'Alice'
    var_5 = 'age'
    var_6 = 30
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = None
    var_11 = lambda self: (self.age >= 0, 'Age must be non-negative')
    var_12 = module_0.pmap()
    var_13 = 'age'
    var_14 = -1



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'field1'
    var_3 = 'value1'
    var_4 = 'invalid_field'
    var_5 = 'value'
    var_6 = None
    var_7 = 'invalid'
    var_8 = 'field1'
    var_9 = 'not_an_int'



# Parsed testcases at query #39
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'invalid'
    var_12 = None
    var_13 = module_0.pmap()



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = 0
    var_3 = 0
    var_4 = 0
    var_5 = 0
    var_6 = 1
    var_7 = 2



# Parsed testcases at query #41
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = module_0.serialize()
    var_3 = 'json'
    var_4 = module_0.serialize(var_3)



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = set()
    var_7 = 1
    var_8 = 2
    var_9 = 1
    var_10 = lambda self: self.x > 0



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'default'
    var_5 = 0
    var_6 = 'test'
    var_7 = 42



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 30
    var_7 = 1
    var_8 = 2
    var_9 = 1
    var_10 = 2
    var_11 = 'x'
    var_12 = 1
    var_13 = 2
    var_14 = True



# Parsed testcases at query #46
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 5
    var_7 = 0
    var_8 = lambda : 100
    var_9 = 0
    var_10 = 1
    var_11 = module_0.pmap()
    var_12 = 'x'
    var_13 = 0
    var_14 = 1
    var_15 = 30
    var_16 = True
    var_17 = 2
    var_18 = 'y'
    var_19 = {var_12: var_2, var_18: var_3}
    var_20 = module_0.pmap(var_19)
    var_21 = var_20._buckets



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = 1
    var_13 = 2



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = set()
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = lambda self: len(self) > 0
    var_11 = set()
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = lambda self: len(self) > 0



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = set()
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = set()
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = set()



# Parsed testcases at query #50
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'x'
    var_5 = 100
    var_6 = True
    var_7 = 30
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = 1
    var_12 = lambda : 2
    var_13 = 3
    var_14 = 2
    var_15 = 'y'
    var_16 = {var_4: var_6, var_15: var_14}
    var_17 = module_0.pmap(var_16)



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = lambda self: len(self) > 0
    var_1 = None
    var_2 = 1
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()



# Parsed testcases at query #52
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.pmap()
    var_3 = module_0.pmap()
    var_4 = 10
    var_5 = 20
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = module_0.pmap()
    var_10 = 1
    var_11 = 2
    var_12 = lambda self: (self.x > 0, 'x must be positive')
    var_13 = module_0.pmap()
    var_14 = 1
    var_15 = 2
    var_16 = module_0.pmap()



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = 0
    var_7 = 1
    var_8 = 2
    var_9 = 0
    var_10 = 1
    var_11 = 2
    var_12 = 0
    var_13 = 1



# Parsed testcases at query #54
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = module_0.serialize()



# Parsed testcases at query #55
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'invalid'
    var_12 = None
    var_13 = module_0.pmap()
    var_14 = None
    var_15 = module_0.pmap()
    var_16 = 'typed_field'
    var_17 = 'invalid_type'



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = lambda self: self.field1 is not None
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()



# Parsed testcases at query #57
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = module_0.pmap()



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 0
    var_7 = lambda self: (self.x > 0, 'x must be positive')
    var_8 = 0
    var_9 = 0
    var_10 = lambda self: (self.x + self.y > 0, 'sum must be positive')
    var_11 = -5
    var_12 = 0
    var_13 = 1



# Parsed testcases at query #59
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = module_0.pmap()
    var_6 = 1
    var_7 = -1
    var_8 = module_0.pmap()
    var_9 = 1
    var_10 = 2
    var_11 = 1
    var_12 = module_0.pmap()



# Parsed testcases at query #60
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 5
    var_5 = 10
    var_6 = 15
    var_7 = lambda : 5
    var_8 = 10
    var_9 = module_0.pmap()
    var_10 = 2
    var_11 = module_0.pmap()
    var_12 = var_11._buckets
    var_13 = 'x'
    var_14 = 'y'
    var_15 = 'z'
    var_16 = 30
    var_17 = {var_13: var_2, var_14: var_3, var_15: var_16}
    var_18 = True



# Parsed testcases at query #61
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 0
    var_4 = 0
    var_5 = 0
    var_6 = module_0.pmap()
    var_7 = 0
    var_8 = 0
    var_9 = lambda self: self.x + self.y > 0
    var_10 = module_0.pmap()
    var_11 = 1
    var_12 = 2
    var_13 = 0
    var_14 = 0
    var_15 = module_0.pmap()



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'x'
    var_5 = 100
    var_6 = True
    var_7 = 30
    var_8 = 1
    var_9 = 2
    var_10 = lambda : 1
    var_11 = 2



# Parsed testcases at query #63
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'x'
    var_5 = 100
    var_6 = True
    var_7 = 30
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = lambda : 1
    var_12 = 2
    var_13 = 2
    var_14 = 'y'
    var_15 = {var_4: var_6, var_14: var_13}
    var_16 = module_0.pmap(var_15)



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = 0
    var_5 = 0
    var_6 = 0
    var_7 = 0
    var_8 = 0
    var_9 = 5



# Parsed testcases at query #65
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 20
    var_6 = 30
    var_7 = 'z'
    var_8 = 40
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = module_0.pmap()
    var_13 = module_0.pmap()
    var_14 = True
    var_15 = 'z'



# Parsed testcases at query #66
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 'test'
    var_4 = module_0.serialize()
    var_5 = module_0.field()
    var_6 = 5
    var_7 = 'hello'
    var_8 = module_0.serialize()
    var_9 = 42
    var_10 = 'str'
    var_11 = module_0.serialize(var_10)



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = 1
    var_7 = 2
    var_8 = 'with_callable'



# Parsed testcases at query #68
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = module_0.serialize()
    var_3 = module_0.serialize()
    var_4 = 'upper'
    var_5 = module_0.serialize(var_4)



# Parsed testcases at query #69
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = 2
    var_9 = module_0.pmap()
    var_10 = 'x'
    var_11 = 1
    var_12 = 2
    var_13 = 30
    var_14 = True
    var_15 = 1
    var_16 = 2
    var_17 = 2
    var_18 = 'y'
    var_19 = {var_10: var_2, var_18: var_3}
    var_20 = module_0.pmap(var_19)
    var_21 = var_20._buckets



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = lambda self: len(self) > 0
    var_1 = None
    var_2 = 1
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()



# Parsed testcases at query #71
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = module_0.pmap()
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = 0
    var_10 = 0
    var_11 = module_0.pmap()
    var_12 = module_0.pmap()



# Parsed testcases at query #72
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = lambda : 1
    var_5 = lambda : 2
    var_6 = 'x'
    var_7 = 'y'
    var_8 = True
    var_9 = 'z'
    var_10 = 30
    var_11 = 2
    var_12 = {var_6: var_2, var_7: var_3}
    var_13 = module_0.pmap(var_12)



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'x'
    var_5 = True
    var_6 = 30
    var_7 = lambda : 1
    var_8 = 2



# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #75
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = module_0.serialize()
    var_4 = 'custom'
    var_5 = module_0.serialize(var_4)



# Parsed testcases at query #76
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = None
    var_8 = None
    var_9 = module_0.pmap()
    var_10 = 'a'
    var_11 = None
    var_12 = 'value1'
    var_13 = 'ignored'
    var_14 = True
    var_15 = 'Bob'
    var_16 = 25



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = 0
    var_5 = 0



# Parsed testcases at query #78
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = module_0.serialize()
    var_4 = 'field2'
    var_5 = record.serialize()[var_4]
    assert var_5 == 'VALUE2'
    var_6 = 'custom'
    var_7 = module_0.serialize(var_6)



# Parsed testcases at query #79
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = module_0.pmap()
    var_6 = 0
    var_7 = module_0.pmap()
    var_8 = 0
    var_9 = 0
    var_10 = module_0.pmap()



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = 'x'
    var_3 = 10
    var_4 = 'y'
    var_5 = 20
    var_6 = 'z'
    var_7 = 30
    var_8 = 0
    var_9 = -1
    var_10 = 0
    var_11 = 0



# Parsed testcases at query #81
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = None
    var_2 = None
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = lambda self: True
    var_8 = None
    var_9 = lambda self: True
    var_10 = 42
    var_11 = lambda self: True
    var_12 = None
    var_13 = lambda self: True
    var_14 = None



# Parsed testcases at query #82
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'z'
    var_6 = 20
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = -5
    var_10 = 0
    var_11 = module_0.pmap()
    var_12 = 5
    var_13 = 0
    var_14 = module_0.pmap()
    var_15 = True



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'hello'
    var_5 = 'world'
    var_6 = None
    var_7 = 0



# Parsed testcases at query #84
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 5
    var_7 = 'x'
    var_8 = 'y'
    var_9 = 2
    var_10 = (var_7, var_2)
    var_11 = (var_8, var_3)
    var_12 = (var_10, var_11)
    var_13 = (var_12,)
    var_14 = 'z'
    var_15 = 30
    var_16 = {var_7: var_2, var_8: var_3, var_14: var_15}
    var_17 = True



# Parsed testcases at query #85
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 1
    var_4 = 2
    var_5 = 0
    var_6 = 0
    var_7 = 'x'
    var_8 = lambda x: (x > 0, 'x must be positive')
    var_9 = {var_7: var_8}
    var_10 = module_0.pmap()
    var_11 = 0
    var_12 = 0
    var_13 = 'x'
    var_14 = 'y'
    var_15 = lambda x: (x > 0, 'x must be positive')
    var_16 = lambda y: (y > 0, 'y must be positive')
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_0.pmap()



# Parsed testcases at query #86
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = module_0.serialize()
    var_5 = module_0.field()
    var_6 = module_0.serialize()



# Parsed testcases at query #87
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = module_0.pmap()
    var_6 = 'y'
    var_7 = 20
    var_8 = module_0.pmap()
    var_9 = 'z'
    var_10 = 30
    var_11 = 0
    var_12 = module_0.pmap()
    var_13 = -1
    var_14 = 0
    var_15 = module_0.pmap()



# Parsed testcases at query #88
#--------------------------


def test_case_0():
    var_0 = lambda self: True
    var_1 = None
    var_2 = None
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'



# Parsed testcases at query #89
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()
    var_8 = None
    var_9 = None
    var_10 = module_0.serialize()



# Parsed testcases at query #90
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'x'
    var_5 = True
    var_6 = 30
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 1
    var_11 = lambda : 2
    var_12 = 3
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = 10
    var_17 = 20
    var_18 = 1
    var_19 = 2
    var_20 = lambda self: (self.x > 0, 'x must be positive')
    var_21 = -1
    var_22 = 20
    var_23 = 1
    var_24 = 2
    var_25 = lambda self: (self.x + self.y > 0, 'sum must be positive')
    var_26 = -1
    var_27 = -2
    var_28 = 2
    var_29 = 'y'
    var_30 = {var_4: var_5, var_29: var_28}
    var_31 = module_0.pmap(var_30)
    var_32 = var_31._buckets



# Parsed testcases at query #91
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()
    var_8 = None
    var_9 = None
    var_10 = 'upper'
    var_11 = module_0.serialize(var_10)



# Parsed testcases at query #92
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'



# Parsed testcases at query #93
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = 40



# Parsed testcases at query #94
#--------------------------


def test_case_0():
    var_0 = lambda self: len(self.name) > 0
    var_1 = None
    var_2 = 0
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = set()



# Parsed testcases at query #95
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0.pmap()
    var_3 = module_0.pmap()
    var_4 = 10
    var_5 = 20
    var_6 = 0
    var_7 = 1
    var_8 = module_0.pmap()
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = 0
    var_12 = 0
    var_13 = module_0.pmap()



# Parsed testcases at query #96
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = None
    var_4 = None
    var_5 = module_0.pmap()
    var_6 = None
    var_7 = lambda self: (self.x > 0, 'x must be positive')
    var_8 = module_0.pmap()
    var_9 = 1
    var_10 = 2
    var_11 = None
    var_12 = None
    var_13 = module_0.pmap()



# Parsed testcases at query #97
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 3
    var_7 = 0
    var_8 = lambda : 2
    var_9 = 'x'
    var_10 = 'y'
    var_11 = [var_9, var_10]
    var_12 = True
    var_13 = 30



# Parsed testcases at query #98
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 3
    var_4 = 4
    var_5 = 0
    var_6 = module_0.pmap()
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = 0
    var_10 = 0
    var_11 = module_0.pmap()



# Parsed testcases at query #99
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = 0
    var_13 = module_0.pmap()
    var_14 = 'x'
    var_15 = 'not an int'



# Parsed testcases at query #100
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 5
    var_7 = 0
    var_8 = lambda : 1
    var_9 = 5
    var_10 = 0
    var_11 = 1
    var_12 = 30
    var_13 = True
    var_14 = 0
    var_15 = 1
    var_16 = 'x'
    var_17 = 0
    var_18 = 1



# Parsed testcases at query #101
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()
    var_5 = None
    var_6 = None
    var_7 = module_0.serialize()



# Parsed testcases at query #102
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'



# Parsed testcases at query #103
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 1
    var_4 = 2
    var_5 = 0
    var_6 = 0
    var_7 = module_0.pmap()
    var_8 = 0
    var_9 = module_0.pmap()
    var_10 = 0
    var_11 = 0
    var_12 = module_0.pmap()



# Parsed testcases at query #104
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'field1'
    var_3 = 'value1'
    var_4 = 'field2'
    var_5 = 'value2'
    var_6 = 'invalid_field'
    var_7 = 'value'
    var_8 = None
    var_9 = 'invalid'
    var_10 = None



# Parsed testcases at query #105
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.pmap()
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = 'invalid_field'
    var_8 = 'value'
    var_9 = None
    var_10 = module_0.pmap()
    var_11 = 'invalid'
    var_12 = module_0.pmap()
    var_13 = 'field1'
    var_14 = 'not_an_int'
    var_15 = module_0.pmap()
    var_16 = 'lowercase'
    var_17 = module_0.pmap()
    var_18 = True
    var_19 = 'extra'
    var_20 = 'actual'
    var_21 = 'value'
    var_22 = 'data'
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = module_0.pmap()



# Parsed testcases at query #106
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 2
    var_7 = 0
    var_8 = 1
    var_9 = lambda self: (self.x > 0, 'x must be positive')
    var_10 = 0
    var_11 = 1
    var_12 = lambda self: (self.x + self.y > 0, 'sum must be positive')



# Parsed testcases at query #107
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 0
    var_4 = 0
    var_5 = None
    var_6 = module_0.pmap()
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = 1
    var_10 = 2
    var_11 = 0
    var_12 = 0
    var_13 = module_0.pmap()
    var_14 = module_0.pmap()



# Parsed testcases at query #108
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = 1
    var_6 = 5
    var_7 = 'mandatory_field'
    var_8 = 0
    var_9 = 0
    var_10 = 0
    var_11 = 1



# Parsed testcases at query #109
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 0
    var_2 = module_0.pmap()
    var_3 = 'x'
    var_4 = 10
    var_5 = 'y'
    var_6 = 20
    var_7 = 'z'
    var_8 = 30
    var_9 = 0
    var_10 = module_0.pmap()
    var_11 = -1
    var_12 = module_0.pmap()
    var_13 = 5
    var_14 = module_0.pmap()
    var_15 = True
    var_16 = module_0.pmap()



# Parsed testcases at query #110
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 10
    var_3 = 20
    var_4 = 0
    var_5 = module_0.pmap()
    var_6 = module_0.pmap()
    var_7 = 0
    var_8 = module_0.pmap()
    var_9 = 'x'
    var_10 = 'y'
    var_11 = {var_9: var_2, var_10: var_3}
    var_12 = module_0.pmap(var_11)



# Parsed testcases at query #111
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 1
    var_8 = 2
    var_9 = 'x'
    var_10 = 1
    var_11 = 2
    var_12 = 30
    var_13 = True
    var_14 = 1
    var_15 = 2
    var_16 = 1
    var_17 = lambda : 2



