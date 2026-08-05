####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'John'
    var_1 = 30
    var_2 = 'name'
    var_3 = 'Jane'
    var_4 = 'age'
    var_5 = 25
    var_6 = 'age'
    var_7 = 'not_an_int'
    var_8 = 'non_existent'
    var_9 = True
    var_10 = 'Bob'
    var_11 = 40
    var_12 = module_0.pmap()
    var_13 = None
    var_14 = False
    var_15 = 'MockRecord.name'
    var_16 = any(var_4)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = 'name'
    var_5 = 'age'
    var_6 = 'tags'
    var_7 = 'Charlie'
    var_8 = 40
    var_9 = 'admin'
    var_10 = [var_9]
    var_11 = {var_4: var_7, var_5: var_8, var_6: var_10}
    var_12 = [var_9]
    var_13 = 'Alice'
    var_14 = 'error'
    var_15 = 10
    var_16 = 'MockRecord.name'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 'name'
    var_2 = 'NewName'
    var_3 = 'non_existent_field'
    var_4 = 123
    var_5 = 'ArgSet'
    var_6 = 10



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = 'name'
    var_2 = 'Alice'
    var_3 = module_0.pmap()
    var_4 = 'age'
    var_5 = 25
    var_6 = 'non_existent'
    var_7 = 'value'
    var_8 = 'name'
    var_9 = 123
    var_10 = 'Bob'
    var_11 = 0
    var_12 = {var_9: var_10, var_4: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = [var_9]
    var_15 = 30
    var_16 = module_0.pmap()
    var_17 = 'value'
    var_18 = -1
    var_19 = module_0.pmap()



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '_precord_invariants'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = '_precord_mandatory_fields'
    var_3 = '_precord_initial_values'
    var_4 = '__slots__'
    var_5 = [name for (name, f) in var_0 if f.mandatory]



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'name'
    var_3 = 'Bob'
    var_4 = 'age'
    var_5 = 31
    var_6 = 0
    var_7 = evolver3.persistent()[var_4]
    assert var_7 == 0
    var_8 = 'non_existent'
    var_9 = 'value'
    var_10 = 'age'
    var_11 = 'not_an_int'
    var_12 = 10
    var_13 = 'count'
    var_14 = -5
    var_15 = 'evologer'
    var_16 = evolver8.persistent()[var_2]
    assert var_16 == 'Charlie'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'age'
    var_3 = 31
    var_4 = 'name'
    var_5 = 'Bob'
    var_6 = 40
    var_7 = 'non_existent'
    var_8 = 'value'
    var_9 = 'age'
    var_10 = 'not_an_int'
    var_11 = 1
    var_12 = 2
    var_13 = 10
    var_14 = 'req'
    var_15 = 10



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 'Bob'
    var_2 = 30
    var_3 = False
    var_4 = 10
    var_5 = 'Charlie'
    var_6 = 25
    var_7 = True



# Parsed testcases at query #12
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 30
    var_7 = 'Bob'
    var_8 = 40
    var_9 = 'age'
    var_10 = 'not_an_int'
    var_11 = 'non_existent'
    var_12 = 'value'
    var_13 = 'Charlie'
    var_14 = 'val'
    var_15 = 10
    var_16 = {var_14: var_15}
    var_17 = module_0.pmap(var_16)
    var_18 = -5
    var_19 = 'NewName'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = False
    var_3 = "TestRecord(name='Alice', age=30, active=False)"
    var_4 = 'Bob'
    var_5 = 'TestRecord('
    var_6 = ')'
    var_7 = ''
    var_8 = -1
    var_9 = True



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'age'
    var_3 = 31
    var_4 = e.persistent()[var_2]
    assert var_4 == 31
    var_5 = 'custom'
    var_6 = 5
    var_7 = e.persistent()[var_5]
    assert var_7 == 10
    var_8 = 'age'
    var_9 = 'not_an_int'
    var_10 = 'non_existent'
    var_11 = 123
    var_12 = 'count'
    var_13 = 1
    var_14 = 2
    var_15 = e3.persistent()[var_12]
    assert var_15 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Test the _factory_fields logic in Evolver.'
    var_1 = 1
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.pmap(var_5)
    var_7 = [var_3]
    var_8 = 10
    var_9 = e_restricted.persistent()[var_3]
    assert var_9 == 10
    var_10 = 99
    var_11 = e_restricted.persistent()[var_4]
    assert var_11 == 2



# Parsed testcases at query #15
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 25
    var_7 = {var_1: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = 'value'
    var_10 = 10
    var_11 = {var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = -5
    var_14 = 'Alice'
    var_15 = 20
    var_16 = 'Bob'
    var_17 = {var_0: var_16}
    var_18 = module_0.pmap(var_17)
    var_19 = True
    var_20 = 'Charlie'
    var_21 = 40
    var_22 = {var_0: var_20, var_1: var_21}
    var_23 = module_0.pmap(var_22)



# Parsed testcases at query #16
#--------------------------




# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = '_precord_mandatory_fields'
    var_3 = '_precord_initial_values'
    var_4 = '__slots__'
    var_5 = 'Test'



# Parsed testcases at query #18
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 123
    var_2 = 'data'
    var_3 = module_0.serialize()
    var_4 = 'upper'
    var_5 = module_0.serialize(var_4)
    var_6 = 'none'
    var_7 = module_0.serialize(var_6)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = 'hello'
    var_2 = 'upper'
    var_3 = module_0.serialize(var_2)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Initial'
    var_1 = 10
    var_2 = 'name'
    var_3 = 'Updated'
    var_4 = 'age'
    var_5 = 'not_an_int'
    var_6 = 'non_existent_field'
    var_7 = 'value'
    var_8 = 'val'
    var_9 = -5
    var_10 = 'Positional'
    var_11 = 20
    var_12 = 'age'
    var_13 = 30
    var_14 = 'Keyword'
    var_15 = 'NewName'
    var_16 = 40



# Parsed testcases at query #20
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'hello'
    var_3 = module_0.serialize()
    var_4 = 'upper'
    var_5 = module_0.serialize(var_4)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.serialize()



# Parsed testcases at query #21
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 10
    var_4 = 'name'
    var_5 = 'age'
    var_6 = 'Charlie'
    var_7 = 25
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = var_9._size
    var_11 = var_9._buckets
    var_12 = 'Alice'
    var_13 = 'error'
    var_14 = 'unknown'
    var_15 = 'Dave'
    var_16 = 'ignored'
    var_17 = {var_4: var_15, var_14: var_16}
    var_18 = True
    var_19 = module_0.pmap()
    var_20 = None
    var_21 = 'TestRecord.name'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Charlie'
    var_1 = 25



# Parsed testcases at query #23
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = 'a'
    var_2 = 10
    var_3 = module_0.pmap()
    var_4 = 'c'
    var_5 = 'hello'
    var_6 = module_0.pmap()
    var_7 = 'invalid_data'
    var_8 = module_0.pmap()
    var_9 = 'non_existent'
    var_10 = 123
    var_11 = module_0.pmap()
    var_12 = module_0.pmap()
    var_13 = module_0.pmap()
    var_14 = [var_10]
    var_15 = module_0.pmap()
    var_16 = 1
    var_17 = 'b'
    var_18 = 2



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'counter'
    var_3 = module_0.PMap()
    var_4 = 1
    var_5 = 0
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'name'
    var_9 = 'age'
    var_10 = 'Bob'
    var_11 = 25
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = [var_8]
    var_14 = 'Charlie'
    var_15 = 40



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'name'
    var_3 = 'Bob'
    var_4 = 'age'
    var_5 = 25
    var_6 = evolver.persistent()[var_2]
    assert var_6 == 'Charlie'
    var_7 = 'non_existent'
    var_8 = 'value'
    var_9 = 'age'
    var_10 = 'not_an_int'
    var_11 = 10
    var_12 = 'value'
    var_13 = -5
    var_14 = 'original'
    var_15 = 'data'
    var_16 = 'new_value'
    var_17 = evolver_f.persistent()[var_15]
    assert var_17 == 'new_value'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Bob'
    var_1 = 40



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = 'name'
    var_2 = 'Alice'
    var_3 = module_0.pmap()
    var_4 = 'name'
    var_5 = 123
    var_6 = 'non_existent'
    var_7 = 'value'
    var_8 = module_0.pmap()
    var_9 = 'age'
    var_10 = [var_9]
    var_11 = 'Bob'
    var_12 = e4.persistent()[var_7]
    assert var_12 == 'Bob'
    var_13 = module_0.pmap()
    var_14 = 'val'
    var_15 = -1



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'Bob'
    var_7 = 25
    var_8 = 20
    var_9 = {var_1: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = 'value'
    var_12 = 10
    var_13 = {var_11: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = -5



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precroll_invariants'
    var_2 = '_precord_invariants'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_initial_values'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'John'
    var_1 = 30
    var_2 = 'age'
    var_3 = 31
    var_4 = 'name'
    var_5 = 'Jane'
    var_6 = 'non_existent_field'
    var_7 = 'value'
    var_8 = 'age'
    var_9 = 'not_an_int'
    var_10 = 'Original'
    var_11 = 10
    var_12 = 20
    var_13 = {var_4: var_8, var_2: var_9}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_4]
    var_16 = 99



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'initial_a'
    var_1 = 'initial_b'
    var_2 = 'field_a'
    var_3 = 'new_a'
    var_4 = 'low'
    var_5 = 'field_c'
    var_6 = 'up'
    var_7 = evolver_trans.persistent()[var_5]
    assert var_7 == 'UP'
    var_8 = 'good'
    var_9 = 'field_bad'
    var_10 = 'bad_value'
    var_11 = 'non_existent'
    var_12 = 'value'
    var_13 = 'orig'
    var_14 = 'field_x'
    var_15 = [var_14]
    var_16 = 'field_y'
    var_17 = 'new'
    var_18 = evolver_res.persistent()[var_16]
    assert var_18 == 'new'
    var_19 = 'val1'
    var_20 = 'field_b'
    var_21 = 'val2'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'age'
    var_3 = 31
    var_4 = 'age'
    var_5 = 'not_an_int'
    var_6 = 'non_existent'
    var_7 = True
    var_8 = 'val'
    var_9 = 10
    var_10 = {var_8: var_9}
    var_11 = True
    var_12 = 'unknown'
    var_13 = 5
    var_14 = 'error_code'
    var_15 = 20
    var_16 = 'MockRecord.name'



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = False
    var_3 = "TestRecord(name='Alice', age=30, active=False)"
    var_4 = 'Bob'
    var_5 = "TestRecord(name='Bob', age=0, active=True)"
    var_6 = ''
    var_7 = "TestException(name='', age=0, active=False)"
    var_8 = 'TestException'
    var_9 = 'TestRecord'
    var_10 = 'Charlie'
    var_11 = 50



# Parsed testcases at query #14
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = 25
    var_9 = {var_1: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = 'value'
    var_12 = -1
    var_13 = {var_11: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = 'Bob'
    var_16 = 20



# Parsed testcases at query #15
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'Bob'
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = module_0.pmap(var_7)
    var_9 = 30
    var_10 = {var_0: var_2}
    var_11 = module_0.pmap(var_10)
    var_12 = 'non_existent_field'
    var_13 = 'value'
    var_14 = 'val'
    var_15 = 1
    var_16 = {var_14: var_15}
    var_17 = module_0.pmap(var_16)
    var_18 = 2
    var_19 = e4.persistent()[var_14]
    assert var_19 == 2
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_15, var_21: var_18}
    var_23 = module_0.pmap(var_22)
    var_24 = [var_20]
    var_25 = 10
    var_26 = 20
    var_27 = {var_20: var_15}
    var_28 = module_0.pmap(var_27)
    var_29 = True
    var_30 = 5
    var_31 = e6.persistent()[var_20]
    assert var_31 == 5



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 25
    var_2 = 'name'
    var_3 = 'NewName'
    var_4 = 'age'
    var_5 = 30
    var_6 = 'non_existent_field'
    var_7 = 'value'
    var_8 = 'age'
    var_9 = 'not_an_int'
    var_10 = 'FinalName'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'age'
    var_3 = 31
    var_4 = 'non_existent_field'
    var_5 = 'value'
    var_6 = 'name'
    var_7 = 123
    var_8 = 10
    var_9 = 'value'
    var_10 = -5
    var_11 = 'initial'
    var_12 = 'attr'
    var_13 = [var_12]
    var_14 = 'new'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = {}
    var_5 = 2
    var_6 = 10
    var_7 = 'TestRecord.name'
    var_8 = 'Alice'
    var_9 = 'error'
    var_10 = 'name'
    var_11 = 'unknown'
    var_12 = 'Charlie'
    var_13 = 'ignored'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = True



# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 20



# Parsed testcases at query #21
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 10
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = {var_1: var_4}
    var_9 = module_0.pmap(var_8)
    var_10 = {var_0: var_3, var_1: var_4}
    var_11 = module_0.pmap(var_10)
    var_12 = -1
    var_13 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_14 = module_0.pmap(var_13)
    var_15 = 2



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'MockRecord.name'
    var_1 = 'Charlie'
    var_2 = 40



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = "TestRecord(name='Alice', age=0, active=True)"
    var_2 = 'Bob'
    var_3 = 30
    var_4 = False
    var_5 = "TestRecord(name='Bob', age=30, active=False)"
    var_6 = 'Charlie'
    var_7 = 25
    var_8 = "TestRecord(name='Charlie', age=25, active=True)"



