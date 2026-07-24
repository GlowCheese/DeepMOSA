####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'age'
    var_3 = 31
    var_4 = 'name'
    var_5 = 'Bob'
    var_6 = 25
    var_7 = 'non_existent_field'
    var_8 = True
    var_9 = 'age'
    var_10 = 'not_an_int'
    var_11 = 'Original'
    var_12 = 10
    var_13 = 'Changed'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Charlie'
    var_1 = 40



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = 'admin'
    var_5 = [var_4]
    var_6 = 'name'
    var_7 = 'age'
    var_8 = 'Charlie'
    var_9 = 40
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = 10
    var_13 = 'TestRecord.name'
    var_14 = 'Alice'
    var_15 = 'error'
    var_16 = 'Dave'
    var_17 = {var_6: var_16}
    var_18 = True



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'old_value'
    var_3 = 'val2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'new_value'
    var_7 = 'non_existent_field'
    var_8 = 123
    var_9 = 'bad_field'
    var_10 = 'original'
    var_11 = {var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = 'trigger_failure'
    var_14 = 'f'
    var_15 = 'start'
    var_16 = {var_14: var_15}
    var_17 = module_0.pmap(var_16)
    var_18 = 'lowercase'
    var_19 = e_factory.persistent()[var_14]
    assert var_19 == 'LOWERCASE'
    var_20 = [var_7]
    var_21 = 'new'
    var_22 = e_filtered.persistent()[var_7]
    assert var_22 == 'new'
    var_23 = 'num'
    var_24 = 1
    var_25 = {var_23: var_24}
    var_26 = module_0.pmap(var_25)
    var_27 = 'not_an_int'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'name'
    var_3 = 'Bob'
    var_4 = 'age'
    var_5 = 31
    var_6 = 'extra'
    var_7 = 100
    var_8 = 'age'
    var_9 = 'not_an_int'
    var_10 = 'non_existent'
    var_11 = 123
    var_12 = 1
    var_13 = 2
    var_14 = 10
    var_15 = 'value'
    var_16 = -5



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Alice'



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'python'
    var_3 = 'unit'
    var_4 = [var_2, var_3]
    var_5 = module_0.serialize()
    var_6 = 'upper'
    var_7 = module_0.serialize(var_6)
    var_8 = 'something_else'
    var_9 = module_0.serialize(var_8)
    var_10 = 10
    var_11 = module_0.serialize()



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'John'
    var_1 = 30
    var_2 = 'name'
    var_3 = 'Jane'
    var_4 = 'age'
    var_5 = 25
    var_6 = 'Bob'
    var_7 = evolver2.persistent()[var_2]
    assert var_7 == 'Bob'
    var_8 = 'non_existent_field'
    var_9 = 'value'
    var_10 = 'age'
    var_11 = 'not_an_int'
    var_12 = 'Alice'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Bob'
    var_1 = 40



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Bob'
    var_1 = 40



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Charlie'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Bob'
    var_1 = 20



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Charlie'
    var_1 = 40



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 'admin'
    var_4 = [var_3]
    var_5 = 'Charlie'
    var_6 = 2
    var_7 = {}
    var_8 = 'Dave'
    var_9 = 40
    var_10 = 'Eve'
    var_11 = 'error'
    var_12 = 25
    var_13 = 'TestRecord.name'



# Parsed testcases at query #18
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 123
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = module_0.serialize()
    var_6 = 'upper'
    var_7 = module_0.serialize(var_6)
    var_8 = 'json'
    var_9 = module_0.serialize(var_8)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = 'description'
    var_3 = 'Alice'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = 'active'
    var_2 = '_precord_invariants'
    var_3 = 'Test'



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'data'
    var_3 = 25
    var_4 = 'MockRecord.name'
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'tags'
    var_8 = 'Bob'
    var_9 = 40
    var_10 = 'admin'
    var_11 = [var_10]
    var_12 = {var_5: var_8, var_6: var_9, var_7: var_11}
    var_13 = module_0.pmap(var_12)
    var_14 = var_13._size
    var_15 = var_13._buckets
    var_16 = 'Charlie'
    var_17 = 'error'
    var_18 = 'extra'
    var_19 = 'Dave'
    var_20 = 50
    var_21 = 'ignored'
    var_22 = {var_5: var_19, var_6: var_20, var_18: var_21}
    var_23 = True
    var_24 = 'Eve'
    var_25 = 10



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = 'name'
    var_2 = 'Alice'
    var_3 = module_0.pmap()
    var_4 = 'age'
    var_5 = 25
    var_6 = 'age'
    var_7 = 'not_an_int'
    var_8 = 'non_existent'
    var_9 = 'value'
    var_10 = module_0.pmap()
    var_11 = 'val'
    var_12 = [var_11]
    var_13 = 10
    var_14 = module_0.pmap()
    var_15 = True
    var_16 = 'test'
    var_17 = 'Bob'
    var_18 = 30
    var_19 = 'Robert'



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'name'
    var_3 = 'Bob'
    var_4 = 'Charlie'
    var_5 = 'age'
    var_6 = 25
    var_7 = 'age'
    var_8 = 'not_an_int'
    var_9 = 'non_existent_field'
    var_10 = True
    var_11 = 'Base'
    var_12 = 'extra'
    var_13 = 'some_value'
    var_14 = 10
    var_15 = 'value'
    var_16 = -5
    var_17 = 'Dave'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Bob'
    var_1 = 40



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'tags'
    var_6 = 'Charlie'
    var_7 = 25
    var_8 = 'admin'
    var_9 = [var_8]
    var_10 = {var_3: var_6, var_4: var_7, var_5: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = 'Dan'
    var_13 = 40
    var_14 = 'new'
    var_15 = [var_14]
    var_16 = 'Eve'
    var_17 = 'error'
    var_18 = 'Frank'
    var_19 = 50
    var_20 = {var_3: var_18, var_4: var_19}
    var_21 = None



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'name'
    var_3 = 'Bob'
    var_4 = 'age'
    var_5 = 25
    var_6 = 'Charlie'
    var_7 = evolver2.persistent()[var_2]
    assert var_7 == 'Charlie'
    var_8 = 'non_existent_field'
    var_9 = True
    var_10 = 'age'
    var_11 = 'not_an_int'
    var_12 = 'Original'
    var_13 = 10
    var_14 = 20
    var_15 = 'FactoryTest'
    var_16 = 5
    var_17 = 'NewName'
    var_18 = evolver4.persistent()[var_2]
    assert var_18 == 'NewName'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = 'active'
    var_2 = '__slots__'
    var_3 = '_precord_invariants'



# Parsed testcases at query #9
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'hello'
    var_3 = module_0.serialize()
    var_4 = 'upper'
    var_5 = module_0.serialize(var_4)
    var_6 = 'Bob'
    var_7 = 25
    var_8 = 'world'
    var_9 = module_0.serialize(var_4)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = 'Alice'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = 'John'
    var_3 = 30



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'Bob'
    var_3 = 25
    var_4 = 'admin'
    var_5 = [var_4]
    var_6 = 'Charlie'
    var_7 = 40
    var_8 = 'Dave'
    var_9 = 'error'
    var_10 = 'name'
    var_11 = 'age'
    var_12 = 'Eve'
    var_13 = 20
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'extra'
    var_16 = 'Frank'
    var_17 = 'value'
    var_18 = {var_10: var_16, var_15: var_17}
    var_19 = True
    var_20 = 'name'
    var_21 = 'extra'
    var_22 = 'Frank'
    var_23 = 'value'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = False



# Parsed testcases at query #13
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
    var_7 = {var_0: var_2}
    var_8 = module_0.pmap(var_7)
    var_9 = 'Charlie'
    var_10 = e2.persistent()[var_0]
    assert var_10 == 'Charlie'
    var_11 = {var_0: var_2}
    var_12 = module_0.pmap(var_11)
    var_13 = 'non_existent_field'
    var_14 = 'value'
    var_15 = {var_13: var_2}
    var_16 = module_0.pmap(var_15)
    var_17 = 'name'
    var_18 = 123
    var_19 = 'val'
    var_20 = 10
    var_21 = {var_19: var_20}
    var_22 = module_0.pmap(var_21)
    var_23 = -5
    var_24 = {var_17: var_2}
    var_25 = module_0.pmap(var_24)
    var_26 = []
    var_27 = e6.persistent()[var_17]
    assert var_27 == 'Bob'
    var_28 = {var_17: var_2}
    var_29 = module_0.pmap(var_28)
    var_30 = e7.persistent()[var_18]
    assert var_30 == 30



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Charlie'
    var_1 = 40



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'John'
    var_1 = 30
    var_2 = 'name'
    var_3 = 'Jane'
    var_4 = 'age'
    var_5 = 25
    var_6 = 'Bob'
    var_7 = evolver2.persistent()[var_2]
    assert var_7 == 'Bob'
    var_8 = 'age'
    var_9 = 'not_an_int'
    var_10 = 'non_existent'
    var_11 = 'value'
    var_12 = 1
    var_13 = 2
    var_14 = 'a'
    var_15 = [var_14]
    var_16 = 10
    var_17 = 'b'
    var_18 = 20



# Parsed testcases at query #16
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'age'
    var_3 = 31
    var_4 = 'name'
    var_5 = 'Bob'
    var_6 = 25
    var_7 = 'Charlie'
    var_8 = 40
    var_9 = 'non_existent_field'
    var_10 = True
    var_11 = 'evoler'
    var_12 = 'age'
    var_13 = 'not_an_int'
    var_14 = {var_4: var_12, var_2: var_13}
    var_15 = module_0.pmap(var_14)
    var_16 = [var_4]
    var_17 = 35
    var_18 = 20
    var_19 = {var_2: var_18}
    var_20 = module_0.pmap(var_19)
    var_21 = 'MockRecord.name'



# Parsed testcases at query #17
#--------------------------


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = 'id'
    var_3 = 123
    var_4 = {var_2: var_3}
    var_5 = module_0.serialize()
    var_6 = 'upper'
    var_7 = module_0.serialize(var_6)



