####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'Bob'
    var_3 = 'age'
    var_4 = 30
    var_5 = 'active'
    var_6 = False
    var_7 = '25'
    var_8 = 'age'
    var_9 = 'not_a_number'
    var_10 = 'nonexistent'
    var_11 = 'value'
    var_12 = 'value'
    var_13 = -5
    var_14 = 25
    var_15 = 'data'
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 1
    var_19 = 2
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = {var_16: var_18, var_17: var_19}
    var_22 = module_0.pmap(var_21)
    var_23 = 'Initial'
    var_24 = 40
    var_25 = 'Updated'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 25
    var_4 = 'Bob'
    var_5 = 'age'
    var_6 = 'not_an_int'
    var_7 = 'Charlie'
    var_8 = 30
    var_9 = 'score'
    var_10 = -5
    var_11 = 'nonexistent'
    var_12 = 'value'
    var_13 = 'items'
    var_14 = 'single'
    var_15 = 'a'
    var_16 = 1
    var_17 = 'b'
    var_18 = 2
    var_19 = 'David'
    var_20 = 'x'
    var_21 = -1
    var_22 = 'y'
    var_23 = 20



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = 'Charlie'
    var_5 = 35
    var_6 = False
    var_7 = 'David'
    var_8 = 'Eve'
    var_9 = 40
    var_10 = None
    var_11 = 'Frank'
    var_12 = 45
    var_13 = 2
    var_14 = 'name'
    var_15 = 'age'
    var_16 = 'Grace'
    var_17 = 50
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = var_19._buckets
    var_21 = 'Henry'
    var_22 = 'not_an_int'
    var_23 = []
    var_24 = len(var_23)
    assert var_24 == 1



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 25
    var_4 = 'score'
    var_5 = 95.5
    var_6 = 'age'
    var_7 = 'not_an_int'
    var_8 = 30
    var_9 = -5.0
    var_10 = 'nonexistent'
    var_11 = 'value'
    var_12 = 'items'
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 'Bob'
    var_18 = 80.0
    var_19 = 'value1'
    var_20 = -5
    var_21 = 'value2'
    var_22 = 200



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 25
    var_4 = 'score'
    var_5 = 95.5
    var_6 = 30
    var_7 = 'age'
    var_8 = 'thirty'
    var_9 = -5.0
    var_10 = 'nonexistent'
    var_11 = 'value'
    var_12 = 'Bob'
    var_13 = 80.0
    var_14 = 'items'
    var_15 = 42
    var_16 = 'value'
    var_17 = -5
    var_18 = 'other'
    var_19 = 15
    var_20 = 'Charlie'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = lambda self: (len(self.name) > 0, 'name_non_empty')
    var_1 = '_precord_fields'
    var_2 = 'active'
    var_3 = '_precord_invariants'
    var_4 = set()



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'active'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = '_precord_invariants'
    var_7 = set()



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'active'
    var_4 = '_precord_mandatory_fields'
    var_5 = '_precord_initial_values'
    var_6 = '_precord_invariants'
    var_7 = set()



# Parsed testcases at query #9
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Alice'
    var_2 = 30
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = module_0.serialize()
    var_7 = 'Bob'
    var_8 = 25
    var_9 = 'x'
    var_10 = 'y'
    var_11 = [var_9, var_10]
    var_12 = 'json'
    var_13 = module_0.serialize(var_12)
    var_14 = module_0.field()
    var_15 = 'test'
    var_16 = 5
    var_17 = module_0.serialize()
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = module_0.serialize()
    var_22 = 'c'
    var_23 = [var_3, var_4, var_22]
    var_24 = 19.99
    var_25 = module_0.serialize()
    var_26 = module_0.serialize()



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = 'age'
    var_5 = {var_4: var_1}
    var_6 = module_0.pmap(var_5)
    var_7 = 'Charlie'
    var_8 = 5
    var_9 = 'David'
    var_10 = 40
    var_11 = 'Eve'
    var_12 = 35
    var_13 = 3
    var_14 = 1
    var_15 = 2
    var_16 = [var_14, var_15, var_13]
    var_17 = 4
    var_18 = 6
    var_19 = {}
    var_20 = module_0.pmap(var_19)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 30
    var_4 = 'Bob'
    var_5 = 'age'
    var_6 = 'not_an_int'
    var_7 = 'Charlie'
    var_8 = -5
    var_9 = 'nonexistent'
    var_10 = 'value'
    var_11 = -1
    var_12 = -2
    var_13 = 'items'
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = (var_14, var_15, var_16)
    var_18 = True
    var_19 = 'data'
    var_20 = 'extra'
    var_21 = 'ignored'
    var_22 = {var_20: var_21}
    var_23 = 'Dave'
    var_24 = 25



# Parsed testcases at query #12
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.serialize()
    var_6 = 'test'
    var_7 = 'prefix'
    var_8 = module_0.serialize(var_7)
    var_9 = 5
    var_10 = module_0.serialize()
    var_11 = 10
    var_12 = 'hello'
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.serialize()
    var_18 = module_0.serialize()
    var_19 = 'anything'
    var_20 = module_0.serialize()



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = lambda self: (len(self.name) > 0, 'name_non_empty')
    var_1 = '_precord_fields'
    var_2 = '_precord_invariants'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = 0
    var_5 = 'test'
    var_6 = 'test2'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = 'items'
    var_12 = 'value1'
    var_13 = True
    var_14 = 'name'
    var_15 = 'age'
    var_16 = 'Charlie'
    var_17 = 40
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = var_19._buckets
    var_21 = 'test'
    var_22 = 'present'
    var_23 = 'not an int'
    var_24 = 'data'
    var_25 = 'key'
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = 42



# Parsed testcases at query #15
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = 'name'
    var_2 = 'Alice'
    var_3 = 'age'
    var_4 = 25
    var_5 = 'Bob'
    var_6 = 30
    var_7 = 31
    var_8 = module_0.pmap()
    var_9 = 'Charlie'
    var_10 = -5
    var_11 = module_0.pmap()
    var_12 = 20
    var_13 = module_0.pmap()
    var_14 = -10
    var_15 = module_0.pmap()
    var_16 = 'x'
    var_17 = 10
    var_18 = 'y'
    var_19 = 5
    var_20 = 'Dave'
    var_21 = 40
    var_22 = module_0.pmap()
    var_23 = 'items'
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'items'
    var_9 = 'x'
    var_10 = 'y'
    var_11 = 'z'
    var_12 = {var_9: var_4, var_10: var_5, var_11: var_6}
    var_13 = True
    var_14 = 0
    var_15 = 'key'
    var_16 = 'value'
    var_17 = (var_15, var_16)
    var_18 = (var_14, var_17)
    var_19 = (var_18,)
    var_20 = 5
    var_21 = 'not_an_int'
    var_22 = 10
    var_23 = -5
    var_24 = 5
    var_25 = 10
    var_26 = 5
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_24, var_28: var_22}
    var_30 = 'test'
    var_31 = 'should_fail'
    var_32 = 'Test'
    var_33 = 42



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 30
    var_4 = 25
    var_5 = 'Bob'
    var_6 = -5
    var_7 = 'value1'
    var_8 = -1
    var_9 = 'value2'
    var_10 = 20
    var_11 = 'Charlie'
    var_12 = 40
    var_13 = 'David'
    var_14 = 35
    var_15 = 36
    var_16 = 'x'
    var_17 = 'y'
    var_18 = 80
    var_19 = 'items'
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = -10



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = False
    var_5 = "O'Connor"
    var_6 = 40
    var_7 = ''
    var_8 = 'test'
    var_9 = 3
    var_10 = 1.5
    var_11 = 42
    var_12 = 3.14
    var_13 = True
    var_14 = 'Charlie'
    var_15 = 35
    var_16 = 36



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 'Charlie'
    var_4 = 'thirty'
    var_5 = 2
    var_6 = 1
    var_7 = 3
    var_8 = [var_6, var_5, var_7]
    var_9 = 'items'
    var_10 = True
    var_11 = 1
    var_12 = 2
    var_13 = 999.99
    var_14 = 100
    var_15 = 5
    var_16 = 10



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 25
    var_4 = 'score'
    var_5 = 95.5
    var_6 = 'age'
    var_7 = 'not_an_int'
    var_8 = 30
    var_9 = -5.0
    var_10 = 'nonexistent'
    var_11 = 'value'
    var_12 = 'items'
    var_13 = 42
    var_14 = 'value1'
    var_15 = -5
    var_16 = 'value2'
    var_17 = 15
    var_18 = 'Bob'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = 'Charlie'
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'David'
    var_8 = 35
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = None
    var_11 = 'extra'
    var_12 = 'Eve'
    var_13 = 28
    var_14 = 'ignored'
    var_15 = {var_5: var_12, var_6: var_13, var_11: var_14}
    var_16 = True
    var_17 = 2
    var_18 = 'Frank'
    var_19 = (var_5, var_18)
    var_20 = 40
    var_21 = (var_6, var_20)
    var_22 = [var_19, var_21]
    var_23 = 'Grace'
    var_24 = 'not_an_int'
    var_25 = 'test'
    var_26 = 20
    var_27 = 'override'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_mandatory_fields'
    var_2 = '_precord_initial_values'
    var_3 = 'active'
    var_4 = '_precord_invariants'
    var_5 = '__slots__'
    var_6 = set()



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 25
    var_4 = 'score'
    var_5 = 95.5
    var_6 = 30
    var_7 = -10.0
    var_8 = 'nonexistent'
    var_9 = 'value'
    var_10 = 'items'
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = (var_11, var_12, var_13)
    var_15 = 'Bob'
    var_16 = -5.0
    var_17 = -10.0



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Test\nLine'
    var_3 = 3.14159
    var_4 = True
    var_5 = None
    var_6 = 5
    var_7 = 'test'
    var_8 = 2.5
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 'Bob'
    var_13 = 25
    var_14 = 'Robert'
    var_15 = False
    var_16 = ''



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = lambda self: (len(self.name) > 0, 'NAME_EMPTY')
    var_1 = '_precord_fields'
    var_2 = 'active'
    var_3 = '_precord_invariants'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 25
    var_4 = 'score'
    var_5 = 95.5
    var_6 = 'age'
    var_7 = 'not_an_int'
    var_8 = 30
    var_9 = -5.0
    var_10 = 'Bob'
    var_11 = 'invalid_field'
    var_12 = 'value'
    var_13 = 'nested'
    var_14 = 'data'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = 'value1'
    var_20 = -5
    var_21 = 'value2'
    var_22 = 200
    var_23 = 10
    var_24 = 20
    var_25 = 'Charlie'
    var_26 = 40



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 0
    var_4 = 'test1'
    var_5 = 'test2'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = 'items'
    var_11 = 'name'
    var_12 = 'extra'
    var_13 = 'value'
    var_14 = {var_11: var_0, var_12: var_13}
    var_15 = True
    var_16 = 'Charlie'
    var_17 = 25
    var_18 = '_precord_size'
    var_19 = '_precord_buckets'
    var_20 = 'age'
    var_21 = 'Dave'
    var_22 = 40
    var_23 = {var_11: var_21, var_20: var_22}
    var_24 = module_0.pmap(var_23)
    var_25 = var_24._buckets
    var_26 = {var_18: var_7, var_19: var_25}
    var_27 = 'not_an_int'
    var_28 = 5
    var_29 = -5
    var_30 = 10



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = 'name'
    var_2 = 'Alice'
    var_3 = 'age'
    var_4 = 25
    var_5 = module_0.pmap()
    var_6 = 30
    var_7 = module_0.pmap()
    var_8 = 'score'
    var_9 = -5
    var_10 = module_0.pmap()
    var_11 = 'invalid_field'
    var_12 = 'value'
    var_13 = module_0.pmap()
    var_14 = 'items'
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = (var_15, var_16, var_17)
    var_19 = module_0.pmap()
    var_20 = set()
    var_21 = (var_15, var_16, var_17)
    var_22 = module_0.pmap()
    var_23 = 'a'
    var_24 = -1
    var_25 = 'b'
    var_26 = 20
    var_27 = module_0.pmap()
    var_28 = 5
    var_29 = 8
    var_30 = 'Bob'
    var_31 = {var_12: var_30, var_3: var_6}
    var_32 = 'Robert'
    var_33 = module_0.pmap()



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = 'name'
    var_2 = 'Alice'
    var_3 = 'age'
    var_4 = 25
    var_5 = 'Bob'
    var_6 = 30
    var_7 = module_0.pmap()
    var_8 = module_0.pmap()
    var_9 = 'Charlie'
    var_10 = -5
    var_11 = module_0.pmap()
    var_12 = 'value'
    var_13 = -1
    var_14 = 'other'
    var_15 = 15
    var_16 = module_0.pmap()
    var_17 = 'x'
    var_18 = 10
    var_19 = 'y'
    var_20 = 5
    var_21 = 'Dave'
    var_22 = 40
    var_23 = 41
    var_24 = module_0.pmap()
    var_25 = 'items'
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = module_0.pmap()



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 30
    var_4 = 'score'
    var_5 = 95.5
    var_6 = 25
    var_7 = -10.0
    var_8 = 'invalid_field'
    var_9 = 'value'
    var_10 = 'data'
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = 'a'
    var_15 = 5
    var_16 = 'b'
    var_17 = 7
    var_18 = 'x'
    var_19 = -5
    var_20 = 'y'
    var_21 = 15
    var_22 = 'Bob'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 30
    var_4 = 25
    var_5 = 'Bob'
    var_6 = -5
    var_7 = -10
    var_8 = 'items'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = lambda x: (len(x.get('name', '')) > 0, 'ERR_NAME_EMPTY')
    var_1 = '_precord_fields'
    var_2 = 'active'
    var_3 = '_precord_invariants'
    var_4 = 'Alice'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = 'name'
    var_2 = 'Alice'
    var_3 = 'age'
    var_4 = 25
    var_5 = 'score'
    var_6 = 95
    var_7 = module_0.pmap()
    var_8 = 'Bob'
    var_9 = module_0.pmap()
    var_10 = 30
    var_11 = module_0.pmap()
    var_12 = 'Charlie'
    var_13 = -5
    var_14 = module_0.pmap()
    var_15 = -10
    var_16 = 'David'
    var_17 = 40
    var_18 = 'Eve'
    var_19 = 35
    var_20 = 36
    var_21 = module_0.pmap()
    var_22 = 'items'
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.pmap()
    var_28 = 'field1'
    var_29 = 'value1'
    var_30 = module_0.pmap()
    var_31 = -1
    var_32 = 'value2'
    var_33 = -2



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = lambda self: (len(self.name) > 0, 'NAME_EMPTY')
    var_1 = '_precord_fields'
    var_2 = 'active'
    var_3 = '_precord_invariants'
    var_4 = 'John'



# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = False
    var_5 = 'Charlie'
    var_6 = 'David'
    var_7 = 35
    var_8 = None
    var_9 = 'Eve'
    var_10 = 40
    var_11 = True
    var_12 = 2
    var_13 = 'name'
    var_14 = 'age'
    var_15 = 'Frank'
    var_16 = 45
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = var_18._buckets
    var_20 = 'Grace'
    var_21 = 'not_an_int'
    var_22 = 'Henry'
    var_23 = 50
    var_24 = 'value'
    var_25 = 'timestamp'
    var_26 = 'custom'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 25
    var_4 = 'Bob'
    var_5 = 'not_an_int'
    var_6 = 'Charlie'
    var_7 = -5
    var_8 = 'nonexistent'
    var_9 = 'value'
    var_10 = 'items'
    var_11 = 'single'
    var_12 = 'nested'
    var_13 = 'value'
    var_14 = 'extra'
    var_15 = 1
    var_16 = 'should_be_ignored'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = -1
    var_19 = -2
    var_20 = 'optional'
    var_21 = 'present'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 30
    var_4 = 'score'
    var_5 = 95.5
    var_6 = 'not_an_int'
    var_7 = 'Bob'
    var_8 = -10.0
    var_9 = 'nonexistent'
    var_10 = 'value'
    var_11 = 25
    var_12 = 'invalid'
    var_13 = -5.0
    var_14 = 'Dave'
    var_15 = 50



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 'Bob'
    var_2 = 30
    var_3 = True
    var_4 = "TestRecord(name='Bob', age=30, active=True)"
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 2
    var_9 = 3
    var_10 = [var_3, var_8, var_9]
    var_11 = 'TestRecord('
    var_12 = 'Line1\nLine2\tTab'
    var_13 = 'Charlie'
    var_14 = None



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = 'active'
    var_2 = '_precord_invariants'
    var_3 = set()



# Parsed testcases at query #14
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.serialize()
    var_6 = 'test'
    var_7 = 'prefix'
    var_8 = module_0.serialize(var_7)
    var_9 = None
    var_10 = module_0.serialize(var_9)
    var_11 = 'hello'
    var_12 = 42
    var_13 = module_0.serialize()
    var_14 = 'text'
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = 4
    var_19 = [var_15, var_16, var_17, var_18]
    var_20 = module_0.serialize()
    var_21 = module_0.serialize()



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 25
    var_4 = 'score'
    var_5 = 95.5
    var_6 = 'age'
    var_7 = 'not_an_int'
    var_8 = 30
    var_9 = -5.0
    var_10 = 'nonexistent'
    var_11 = 'value'
    var_12 = 'nested'
    var_13 = 'data'
    var_14 = 'key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = {var_13: var_16}
    var_18 = {var_14: var_15}
    var_19 = {var_13: var_18}
    var_20 = set()
    var_21 = {var_14: var_15}
    var_22 = {var_13: var_21}
    var_23 = 'a'
    var_24 = -1
    var_25 = 'b'
    var_26 = 20
    var_27 = 'Bob'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 30
    var_4 = 'score'
    var_5 = 95.5
    var_6 = 'not_an_int'
    var_7 = 'Bob'
    var_8 = -10.0
    var_9 = 'nonexistent'
    var_10 = 'value'
    var_11 = 25
    var_12 = 80.0
    var_13 = 'items'
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = (var_14, var_15, var_16)
    var_18 = 'value'
    var_19 = -5
    var_20 = 'other'
    var_21 = 15



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = lambda x: (len(x.get('name', '')) > 0, 'name_non_empty')
    var_1 = '_precord_fields'
    var_2 = '_precord_invariants'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_initial_values'
    var_5 = '__slots__'



# Parsed testcases at query #18
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'Charlie'
    var_6 = 25
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = None
    var_9 = 'extra'
    var_10 = 'David'
    var_11 = 35
    var_12 = 'ignored'
    var_13 = {var_3: var_10, var_4: var_11, var_9: var_12}
    var_14 = True
    var_15 = '_precord_size'
    var_16 = '_precord_buckets'
    var_17 = 2
    var_18 = 'Eve'
    var_19 = 40
    var_20 = {var_3: var_18, var_4: var_19}
    var_21 = module_0.pmap(var_20)
    var_22 = var_21._buckets
    var_23 = {var_15: var_17, var_16: var_22}
    var_24 = 'Frank'
    var_25 = 45
    var_26 = 'Franklin'
    var_27 = 'Frankie'
    var_28 = 46
    var_29 = 'Grace'
    var_30 = 'not_an_int'
    var_31 = -5
    var_32 = 10



# Parsed testcases at query #19
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'dev'
    var_3 = 'python'
    var_4 = [var_2, var_3]
    var_5 = module_0.serialize()
    var_6 = 'raw'
    var_7 = module_0.serialize(var_6)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.serialize(var_6)
    var_13 = 'summary'
    var_14 = module_0.serialize(var_13)
    var_15 = 42
    var_16 = module_0.serialize()
    var_17 = 123
    var_18 = True
    var_19 = 98.5
    var_20 = module_0.serialize()
    var_21 = module_0.serialize()
    var_22 = 'Bob'
    var_23 = 25
    var_24 = 'test'
    var_25 = [var_24]
    var_26 = module_0.serialize()



# Parsed testcases at query #20
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = False
    var_5 = 'Charlie'
    var_6 = 2
    var_7 = 'name'
    var_8 = 'age'
    var_9 = 'Dave'
    var_10 = 40
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = var_12._buckets
    var_14 = 'Eve'
    var_15 = 35
    var_16 = {var_7: var_14, var_8: var_15}
    var_17 = set()
    var_18 = 'extra'
    var_19 = 'Frank'
    var_20 = 45
    var_21 = 'ignored'
    var_22 = {var_7: var_19, var_8: var_20, var_18: var_21}
    var_23 = True
    var_24 = 'Grace'
    var_25 = 50
    var_26 = 'not_allowed'
    var_27 = 0



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = 'age'
    var_3 = 25
    var_4 = 'not_an_int'
    var_5 = 'Bob'
    var_6 = -5
    var_7 = 'nonexistent'
    var_8 = 'value'
    var_9 = 'items'
    var_10 = 'single'
    var_11 = 'a'
    var_12 = 1
    var_13 = 'b'
    var_14 = 2
    var_15 = 'x'
    var_16 = -1
    var_17 = 'y'
    var_18 = 20
    var_19 = set()
    var_20 = 'data'
    var_21 = 'raw'
    var_22 = 'value'
    var_23 = {var_21: var_22}



# Parsed testcases at query #22
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 25
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = module_0.serialize()
    var_6 = 'json'
    var_7 = module_0.serialize(var_6)
    var_8 = 42
    var_9 = 'hello'
    var_10 = module_0.serialize()
    var_11 = None
    var_12 = module_0.serialize(var_11)
    var_13 = ''
    var_14 = 0
    var_15 = []
    var_16 = module_0.serialize()



# Parsed testcases at query #23
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'Charlie'
    var_6 = 25
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'extra'
    var_9 = 'David'
    var_10 = 35
    var_11 = 'ignored'
    var_12 = {var_3: var_9, var_4: var_10, var_8: var_11}
    var_13 = True
    var_14 = 'active'
    var_15 = 'Eve'
    var_16 = 40
    var_17 = False
    var_18 = {var_3: var_15, var_4: var_16, var_14: var_17}
    var_19 = module_0.pmap(var_18)
    var_20 = var_19._size
    var_21 = var_19._buckets
    var_22 = 'Frank'
    var_23 = 45
    var_24 = 'Grace'
    var_25 = 'not_an_int'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = False
    var_5 = 'Charlie'
    var_6 = 2
    var_7 = 'name'
    var_8 = 'age'
    var_9 = 'Dave'
    var_10 = 40
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = var_12._buckets
    var_14 = 'Eve'
    var_15 = 35
    var_16 = set()
    var_17 = 'Frank'
    var_18 = 45
    var_19 = 'should_be_ignored'
    var_20 = True
    var_21 = 'Grace'
    var_22 = 50
    var_23 = 'error'
    var_24 = 'timestamp'
    var_25 = 100.0
    var_26 = 5



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = "MyRecord(active=True, age=30, name='Alice')"
    var_3 = 'Bob'
    var_4 = 25
    var_5 = False
    var_6 = "MyRecord(active=False, age=25, name='Bob')"
    var_7 = 'EmptyRecord()'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = "NestedRecord(data={'key': 'value'}, items=[1, 2, 3])"
    var_16 = 'Test\nLine'
    var_17 = 40
    var_18 = "MyRecord(active=True, age=40, name='Test\\nLine')"
    var_19 = 'z'
    var_20 = 'a'
    var_21 = 'm'
    var_22 = "MultiFieldRecord(a_field='a', m_field='m', z_field='z')"
    var_23 = 10
    var_24 = 'OptionalRecord(count=10, value=None)'



