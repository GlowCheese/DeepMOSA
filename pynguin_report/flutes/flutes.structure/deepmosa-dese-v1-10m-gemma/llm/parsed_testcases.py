####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = [var_0, var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2, var_0}
    var_4 = module_0.map_structure(var_1, var_3)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = 'a'
    var_6 = 4
    var_7 = {var_5: var_6}
    var_8 = [var_1, var_4, var_7]
    var_9 = module_0.map_structure(var_0, var_8)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_dict_predicate_true. Retrieved 12/14 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'a'
    var_7 = 4
    var_8 = {var_6: var_7}
    var_9 = {var_0: var_5, var_1: var_8}
    var_10 = lambda x: x
    var_11 = module_0.map_structure(var_10, var_9)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_map_structure_mixed_types. Retrieved 8/10 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = list(var_5)
    var_7 = sorted(var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = [var_0, var_3, var_6]

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = module_0.map_structure(var_1, var_6)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_zip_simple_values. Retrieved 5/6 statements.
# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 1
    var_7 = 2
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 10
    var_9 = {var_1: var_8}
    var_10 = 20
    var_11 = 30
    var_12 = (var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = [var_7, var_13]
    var_15 = module_0.map_structure_zip(var_0, var_14)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = [var_4]
    var_6 = module_0.map_structure_zip(var_1, var_5)



# Parsed testcases at query #5
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = lambda x, y: x + y
    var_8 = module_0.map_structure_zip(var_7, var_6)



# Parsed testcases at query #6
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = 5
    var_12 = [var_11]
    var_13 = 6
    var_14 = [var_13]
    var_15 = [var_12, var_14]
    var_16 = [var_5, var_10, var_15]
    var_17 = module_0.map_structure_zip(var_0, var_16)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 1
    var_7 = 2
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_map_structure_zip_no_type_check_decorator_not_triggered. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_tuple_predicate. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_map_structure_predicate_is_false_with_list. Retrieved 9/29 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = '_NO_MAP_TYPES'
    var_1 = '_NO_MAP_INSTANCE_ATTR'
    var_2 = set()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = lambda x: x
    var_8 = module_0.map_structure(var_7, var_6)



# Parsed testcases at query #10
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)



# Parsed testcases at query #11
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 9/11 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 'a'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = lambda x: x
    var_8 = module_0.map_structure(var_7, var_6)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_map_structure_predicate_false_with_list. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #14
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_map_structure_zip_no_type_check_predicate. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_map_structure_tuple. Retrieved 4/5 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = 4
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = [var_0, var_4, var_7]
    var_9 = module_0.map_structure(var_1, var_8)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 'world'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.map_structure(var_1, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_0, var_3)
    var_5 = 'a'
    var_6 = 4
    var_7 = {var_5: var_6}
    var_8 = [var_2, var_4, var_7]
    var_9 = module_0.map_structure(var_1, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #17
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)



# Parsed testcases at query #18
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_map_structure_list_is_instance_of_list. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 9/14 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = [var_0, var_4]
    var_6 = 3
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2, var_0}
    var_4 = module_0.map_structure(var_1, var_3)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = (var_3,)
    var_5 = [var_2, var_4]
    var_6 = {var_1: var_5}
    var_7 = (var_0, var_6)
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = (var_8,)
    var_11 = [var_3, var_10]
    var_12 = {var_1: var_11}
    var_13 = (var_2, var_12)
    var_14 = 5
    var_15 = [var_13, var_14]
    var_16 = lambda x: x + var_0
    var_17 = module_0.map_structure(var_16, var_9)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = lambda x: x * var_6
    var_8 = 6



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_map_structure_zip_dict_true_predicate. Retrieved 13/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = 0
    var_12 = var_9[var_11]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = lambda x, y: x + y
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_6]
    var_8 = [var_5, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_map_structure_zip_no_type_check_predicate. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #24
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = module_0.map_structure_zip(var_0, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = 2
    var_6 = [var_5]
    var_7 = {var_1: var_6}
    var_8 = [var_4, var_7]
    var_9 = module_0.map_structure_zip(var_0, var_8)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_map_structure. Retrieved 15/17 statements.
# Partially parsed test_map_structure_namedtuple. Retrieved 11/18 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 10
    var_5 = 20
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 1
    var_8 = 2
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 10
    var_11 = 20
    var_12 = {var_5: var_10, var_6: var_11}
    var_13 = [var_9, var_12]
    var_14 = module_0.map_structure_zip(var_4, var_13)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 10
    var_8 = 20
    var_9 = 11
    var_10 = 22

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_map_structure_zip_no_type_check_decorator. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = (var_4, var_4)
    var_6 = [var_3, var_5]
    var_7 = module_0.map_structure_zip(var_0, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_8}
    var_10 = [var_5, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = module_0.map_structure_zip(var_0, var_6)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate_true. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]



# Parsed testcases at query #30
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_1, var_5)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_map_structure_zip_triggers_list_branch. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #32
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = (var_4, var_4)
    var_6 = [var_3, var_5]
    var_7 = module_0.map_structure_zip(var_0, var_6)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_6]
    var_8 = [var_5, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 5
    var_7 = 2
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = [var_1, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = [var_6, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 10
    var_2 = 20
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 30

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = lambda x, y: x + y
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 10
    var_2 = 20
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 30

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 1
    var_7 = 2
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = 4
    var_7 = 5
    var_8 = [var_6, var_7]
    var_9 = 6
    var_10 = (var_8, var_9)
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = lambda x, y: x + y
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #35
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)



# Parsed testcases at query #36
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_map_structure_deeply_nested. Retrieved 29/31 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 'world'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = module_0.map_structure(var_1, var_10)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = {var_2, var_3, var_4}
    var_6 = module_0.map_structure(var_1, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = (var_3,)
    var_5 = [var_2, var_4]
    var_6 = {var_1: var_5}
    var_7 = (var_0, var_6)
    var_8 = 4
    var_9 = 5
    var_10 = {var_8, var_9}
    var_11 = [var_7, var_10]
    var_12 = '2'
    var_13 = '3'
    var_14 = '4'
    var_15 = (var_14,)
    var_16 = [var_13, var_15]
    var_17 = {var_1: var_16}
    var_18 = (var_12, var_17)
    var_19 = 6
    var_20 = 7
    var_21 = {var_19, var_20}
    var_22 = [var_18, var_21]
    var_23 = 0
    var_24 = lambda x: x * var_2
    var_25 = [var_0]
    var_26 = (var_2,)
    var_27 = [var_25, var_26]
    var_28 = module_0.map_structure(var_24, var_27)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 6



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 9/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 1
    var_7 = 2
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = module_0.map_structure_zip(var_1, var_3)
    assert var_4 == 10
    var_5 = lambda x: x * var_0
    var_6 = 10.5
    var_7 = [var_6]
    var_8 = module_0.map_structure_zip(var_5, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = lambda x, y: x + y

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_8]
    var_10 = lambda x, y: x + y
    var_11 = module_0.map_structure_zip(var_10, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_map_structure_predicate_true_via_no_map_type. Retrieved 5/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'map_structure_module'
    var_1 = 1
    var_2 = 1
    var_3 = lambda x: x + var_2
    var_4 = module_0.map_structure(var_3, var_1)
    assert var_4 == 2



# Parsed testcases at query #40
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_5, var_6]
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.map_structure(var_1, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = list(var_5)
    var_7 = sorted(var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'other'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = [var_4, var_7]
    var_9 = 5
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = (var_3, var_6)
    var_12 = 6
    var_13 = 8
    var_14 = {var_12, var_13}
    var_15 = [var_11, var_14]
    var_16 = 10
    var_17 = {var_0: var_15, var_1: var_16}
    var_18 = lambda x: x * var_3
    var_19 = module_0.map_structure(var_18, var_10)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_map_structure_evaluates_tuple_predicate. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5



# Parsed testcases at query #42
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = (var_2, var_5)
    var_7 = 3
    var_8 = [var_7]
    var_9 = 4
    var_10 = {var_3: var_9}
    var_11 = (var_8, var_10)
    var_12 = [var_6, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)



# Parsed testcases at query #44
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_map_structure_predicate_false_with_list. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #46
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_map_structure_zip_no_type_check_predicate. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_1, var_3]



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = 20
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]



# Parsed testcases at query #49
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = 'a'
    var_6 = 4
    var_7 = {var_5: var_6}
    var_8 = [var_1, var_4, var_7]
    var_9 = module_0.map_structure(var_0, var_8)



# Parsed testcases at query #50
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = module_0.map_structure_zip(var_0, var_6)



# Parsed testcases at query #51
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple_behavior. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = 3
    var_6 = [var_5]
    var_7 = [var_2, var_4, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = (var_2, var_5)
    var_7 = 3
    var_8 = [var_7]
    var_9 = 4
    var_10 = {var_3: var_9}
    var_11 = (var_8, var_10)
    var_12 = [var_6, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #53
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_map_structure_tuple. Retrieved 4/5 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 'world'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'other'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 4
    var_8 = [var_3, var_7]
    var_9 = 6
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = lambda x: x * var_3
    var_12 = module_0.map_structure(var_11, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = list(var_5)
    var_7 = sorted(var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = (var_0, var_5)
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = 6
    var_10 = [var_7, var_9]
    var_11 = {var_1: var_10}
    var_12 = (var_2, var_11)
    var_13 = 8
    var_14 = [var_12, var_13]
    var_15 = lambda x: x * var_2
    var_16 = module_0.map_structure(var_15, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_tuple. Retrieved 4/5 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 'world'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 4
    var_10 = [var_3, var_9]
    var_11 = 6
    var_12 = {var_5: var_11}
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = lambda x: x * var_3
    var_15 = module_0.map_structure(var_14, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = [var_0, var_3, var_6]
    var_8 = 6
    var_9 = (var_5, var_8)
    var_10 = 8
    var_11 = {var_4: var_10}
    var_12 = [var_1, var_9, var_11]
    var_13 = lambda x: x * var_1
    var_14 = module_0.map_structure(var_13, var_7)



# Parsed testcases at query #2
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda x, y: x + y
    var_6 = [var_1, var_2]
    var_7 = module_0.map_structure_zip(var_5, var_6)
    assert var_7 == 3

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = 5
    var_9 = 6
    var_10 = [var_8, var_9]
    var_11 = 7
    var_12 = 8
    var_13 = [var_11, var_12]
    var_14 = [var_10, var_13]
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = (var_4, var_4)
    var_6 = [var_3, var_5]
    var_7 = module_0.map_structure_zip(var_0, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x / y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 2
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_8}
    var_10 = [var_5, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_2]
    var_4 = (var_1, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = [var_6]
    var_8 = (var_5, var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_zip_dicts_nested. Retrieved 11/12 statements.
# Partially parsed test_map_structure_zip_namedtuple. Retrieved 9/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    assert var_6 == 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = 5
    var_9 = 6
    var_10 = [var_8, var_9]
    var_11 = 7
    var_12 = 8
    var_13 = [var_11, var_12]
    var_14 = [var_10, var_13]
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 10
    var_5 = 20
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_8}
    var_10 = [var_5, var_9]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 10
    var_8 = 20

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: str(x) + str(y)
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'ab'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_map_structure. Retrieved 20/23 statements.
# Partially parsed test_map_structure_zip_namedtuple. Retrieved 11/18 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    assert var_6 == 3

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x * y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = (var_9, var_10)
    var_12 = 6
    var_13 = 7
    var_14 = (var_12, var_13)
    var_15 = 8
    var_16 = 9
    var_17 = (var_15, var_16)
    var_18 = [var_11, var_14, var_17]
    var_19 = [var_8, var_18]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 1
    var_7 = 2
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 10
    var_8 = 20
    var_9 = 11
    var_10 = 22

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = [var_2]
    var_4 = 2
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 10
    var_6 = 20
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = module_0.map_structure_zip(var_0, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = [var_4]
    var_6 = module_0.map_structure_zip(var_1, var_5)



# Parsed testcases at query #6
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_map_structure_zip_evaluates_true_at_line_17. Retrieved 9/14 statements.


def test_case_0():
    var_0 = '_no_map_'
    var_1 = (var_0,)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 9/14 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = lambda x: x * var_6
    var_8 = 20



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/18 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_1, var_2, var_5]
    var_7 = module_0.map_structure_zip(var_0, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

def test_case_0():
    pass

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = [var_1, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = [var_6, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 50



# Parsed testcases at query #10
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)



# Parsed testcases at query #11
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 5
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 5



# Parsed testcases at query #12
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 'world'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = module_0.map_structure(var_1, var_10)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = list(var_5)
    var_7 = sorted(var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 3
    var_4 = 4
    var_5 = (var_4,)
    var_6 = [var_3, var_5]
    var_7 = {var_2: var_6}
    var_8 = (var_1, var_7)
    var_9 = [var_0, var_8]
    var_10 = 5
    var_11 = (var_10,)
    var_12 = [var_4, var_11]
    var_13 = {var_2: var_12}
    var_14 = (var_3, var_13)
    var_15 = [var_1, var_14]
    var_16 = lambda x: x + var_0
    var_17 = module_0.map_structure(var_16, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 11/18 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    assert var_6 == 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 10
    var_7 = [var_6]
    var_8 = 20
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 1
    var_7 = 2
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = 10
    var_6 = 20
    var_7 = [var_6]
    var_8 = [var_5, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = [var_4]
    var_6 = module_0.map_structure_zip(var_1, var_5)

def test_case_0():
    pass

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 10
    var_8 = 20
    var_9 = 11
    var_10 = 22

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = [var_1, var_1]
    var_3 = module_0.map_structure_zip(var_0, var_2)
    assert var_3 == 10



# Parsed testcases at query #14
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_map_structure_tuple_predicate. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = '_fields'
    var_7 = lambda x: x



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_map_structure_predicate_true_via_no_map_types. Retrieved 6/12 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = '_NO_MAP_TYPES'
    var_1 = '_NO_MAP_INSTANCE_ATTR'
    var_2 = None
    var_3 = lambda x: x
    var_4 = 1
    var_5 = module_0.map_structure(var_3, var_4)
    assert var_5 == 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_map_structure_tuple. Retrieved 4/5 statements.
# Partially parsed test_map_structure_deeply_nested. Retrieved 18/26 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'apple'
    var_4 = 'banana'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 4
    var_10 = [var_3, var_9]
    var_11 = 6
    var_12 = {var_5: var_11}
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = lambda x: x * var_3
    var_15 = module_0.map_structure(var_14, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = '2'
    var_8 = 4
    var_9 = 6
    var_10 = [var_9]
    var_11 = {var_8: var_10}
    var_12 = (var_7, var_11)
    var_13 = [var_12]
    var_14 = [var_2]
    var_15 = {var_1: var_14}
    var_16 = (var_0, var_15)
    var_17 = [var_16]

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_map_structure_zip_no_type_check_predicate. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_map_structure_predicate_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: x



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_map_structure_deeply_nested. Retrieved 19/22 statements.
# Partially parsed test_map_structure_namedtuple. Retrieved 9/15 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_5, var_6]
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.map_structure(var_1, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = {var_0, var_2}
    var_4 = module_0.map_structure(var_1, var_3)

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = '1'
    var_10 = '2'
    var_11 = (var_9, var_10)
    var_12 = {var_4, var_5}
    var_13 = [var_11, var_12]
    var_14 = {var_0: var_13}
    var_15 = 'a'
    var_16 = (var_2, var_4)
    var_17 = [var_1, var_16]
    var_18 = {var_15: var_17}

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = lambda x: x * var_6
    var_8 = 20

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = '1'
    var_8 = '2'
    var_9 = {var_3: var_8}
    var_10 = {var_0: var_7, var_1: var_9}
    var_11 = lambda x: str(x)
    var_12 = module_0.map_structure(var_11, var_6)



# Parsed testcases at query #21
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = 4
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = [var_0, var_4, var_7]
    var_9 = module_0.map_structure(var_1, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_0, var_3)
    var_5 = (var_2, var_4)
    var_6 = module_0.map_structure(var_1, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = {var_2: var_0, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = {var_0, var_2}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = list(var_4)
    var_6 = sorted(var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = (var_1, var_6)
    var_8 = [var_0, var_7]
    var_9 = 5
    var_10 = [var_4, var_9]
    var_11 = {var_2: var_10}
    var_12 = (var_3, var_11)
    var_13 = [var_1, var_12]
    var_14 = lambda x: x + var_0
    var_15 = module_0.map_structure(var_14, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'abc'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'ABC'

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 50



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = lambda x, y: x + y
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 3
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = [var_4, var_7]
    var_9 = 'a'
    var_10 = [var_1]
    var_11 = {var_9: var_10}
    var_12 = [var_11]
    var_13 = [var_2]
    var_14 = {var_9: var_13}
    var_15 = [var_14]
    var_16 = [var_12, var_15]
    var_17 = lambda x, y: x + y
    var_18 = module_0.map_structure_zip(var_17, var_16)



# Parsed testcases at query #23
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 'world'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = module_0.map_structure(var_1, var_10)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = {var_2, var_3, var_4}
    var_6 = module_0.map_structure(var_1, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 11

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = 'a'
    var_6 = 4
    var_7 = {var_5: var_6}
    var_8 = [var_1, var_4, var_7]
    var_9 = module_0.map_structure(var_0, var_8)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_map_structure_zip_no_decorator. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #25
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = 3
    var_6 = [var_5]
    var_7 = [var_2, var_4, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = 5
    var_9 = 6
    var_10 = [var_8, var_9]
    var_11 = 7
    var_12 = 8
    var_13 = [var_11, var_12]
    var_14 = [var_10, var_13]
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = (var_4, var_4)
    var_6 = [var_3, var_5]
    var_7 = module_0.map_structure_zip(var_0, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_6]
    var_8 = [var_5, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 10
    var_2 = 20
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 200

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = [var_6]
    var_8 = 4
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 1
    var_7 = 2
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: str(x) + str(y)
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 5
    var_11 = 6
    var_12 = [var_10, var_11]
    var_13 = 7
    var_14 = 8
    var_15 = (var_13, var_14)
    var_16 = {var_1: var_12, var_2: var_15}
    var_17 = [var_9, var_16]
    var_18 = module_0.map_structure_zip(var_0, var_17)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_map_structure_deeply_nested. Retrieved 20/22 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_5, var_6]
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.map_structure(var_1, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2, var_0}
    var_4 = module_0.map_structure(var_1, var_3)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 'c'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = (var_1, var_6)
    var_8 = {var_0: var_7}
    var_9 = '2'
    var_10 = 4
    var_11 = 6
    var_12 = {var_3: var_11}
    var_13 = [var_10, var_12]
    var_14 = (var_9, var_13)
    var_15 = {var_0: var_14}
    var_16 = {var_3: var_4}
    var_17 = [var_2, var_16]
    var_18 = (var_1, var_17)
    var_19 = {var_0: var_18}

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 9/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    assert var_6 == 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 10
    var_7 = [var_6]
    var_8 = 20
    var_9 = [var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 1
    var_7 = 2
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 10
    var_8 = 20

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 6



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_map_structure_tuple_predicate_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = '_fields'
    var_7 = lambda x: x



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_map_structure_deeply_nested. Retrieved 20/22 statements.
# Partially parsed test_map_structure_namedtuple. Retrieved 9/15 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_5, var_6]
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.map_structure(var_1, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = list(var_5)
    var_7 = sorted(var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 'c'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = (var_1, var_6)
    var_8 = {var_0: var_7}
    var_9 = '2'
    var_10 = '3'
    var_11 = '4'
    var_12 = {var_3: var_11}
    var_13 = [var_10, var_12]
    var_14 = (var_9, var_13)
    var_15 = {var_0: var_14}
    var_16 = {var_3: var_4}
    var_17 = [var_2, var_16]
    var_18 = (var_1, var_17)
    var_19 = {var_0: var_18}

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = lambda x: x * var_6
    var_8 = 20



# Parsed testcases at query #31
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 9/10 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 'inner'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = lambda x: x * var_4
    var_8 = module_0.map_structure(var_7, var_6)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate_true. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 9/14 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = 2
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_5, var_6]
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.map_structure(var_1, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = {var_0, var_2}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = list(var_4)
    var_6 = sorted(var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = (var_2, var_4)
    var_10 = 5
    var_11 = {var_5, var_10}
    var_12 = [var_9, var_11]
    var_13 = {var_0: var_12}
    var_14 = lambda x: x + var_1
    var_15 = module_0.map_structure(var_14, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'abc'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'ABC'

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = lambda x: x * var_6
    var_8 = 20



