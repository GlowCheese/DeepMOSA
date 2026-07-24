####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = 3
    var_9 = {var_1: var_8}
    var_10 = 4
    var_11 = {var_4: var_10}
    var_12 = [var_9, var_11]
    var_13 = [var_7, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [{'a': 4}, {'b': 6}])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == [10])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, [4, 6], 8])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = {var_5: var_0}
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': 2, 'b': {'c': 4}})
    assert var_9 is True

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
    var_10 = bool(var_9 == [2, (4, 6), {'a': 8}])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'abc'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'aaabbbccc'



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_map_structure_predicate_false.




# Parsed testcases at query #4
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_predicate_at_line_15_evaluates_to_false.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, [4, 6], 8])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_0, var_3)
    var_5 = 4
    var_6 = (var_2, var_4, var_5)
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == (2, (4, 6), 8))
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = {var_5: var_0}
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': 2, 'b': {'c': 4}})
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = []
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = ()
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == ())
    assert var_4 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = {}
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = set()
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = set()
    var_5 = bool(var_3 == var_4)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = bool(not (False or True))
    assert var_0 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 17/23 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_17 = bool(var_16 == [[6, 8], [10, 12]])
    assert var_17 is True

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
    var_9 = bool(var_8 == (4, 6))
    assert var_9 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

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
    var_12 = bool(var_11 == {'a': [4, 6]})
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 8

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = lambda x, y: x + y
    var_13 = (var_0, var_9)
    var_14 = 6
    var_15 = (var_3, var_14)
    var_16 = [var_13, var_15]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_isinstance_list_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 2/7 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 2/8 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 2
    var_5 = lambda x: x * var_4
    var_6 = 1
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, [4, 6], 8])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = {var_5: var_0}
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': 2, 'b': {'c': 4}})
    assert var_9 is True

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_isinstance_obj_tuple. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_no_type_check_decorator. Retrieved 1/2 statements.


def test_case_0():
    var_0 = '__annotations__'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 17/23 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = 3
    var_8 = {var_1: var_7}
    var_9 = 4
    var_10 = {var_1: var_9}
    var_11 = [var_8, var_10]
    var_12 = [var_6, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == [{'a': 4}, {'a': 6}])
    assert var_14 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == [10])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = lambda x, y: x + y
    var_13 = (var_0, var_9)
    var_14 = 6
    var_15 = (var_3, var_14)
    var_16 = [var_13, var_15]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_isinstance_list_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 1/5 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/8 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [2, [4, 6]])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': {'b': 2}})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [2, (4, {'a': 6})])
    assert var_9 is True

def test_case_0():
    var_0 = lambda x: x

def test_case_0():
    var_0 = '_no_map'
    var_1 = True
    var_2 = lambda x: x



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_predicate_at_line_1.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = var_3.__class__



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 12/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [2, [4, 6]])
    assert var_7 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: x * var_4
    var_8 = (var_0, var_4)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = 3
    var_8 = {var_1: var_7}
    var_9 = 4
    var_10 = {var_1: var_9}
    var_11 = [var_8, var_10]
    var_12 = [var_6, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == [{'a': 4}, {'a': 6}])
    assert var_14 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == [10])
    assert var_6 is True

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
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_isinstance_tuple_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, [4, 6], 8])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = {var_5: var_0}
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': 2, 'b': {'c': 4}})
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'hello'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'HELLO'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_no_map_types. Retrieved 1/6 statements.
# Partially parsed test_map_structure_zip_with_no_map_instance_attr. Retrieved 3/9 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 'a'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = {var_2: var_7}
    var_9 = [var_6, var_8]
    var_10 = [var_5, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [4, {'a': 6}])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == [10])
    assert var_6 is True

def test_case_0():
    var_0 = lambda x: x

def test_case_0():
    var_0 = '_no_map'
    var_1 = True
    var_2 = lambda x: x

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_predicate_at_line_1.




# Parsed testcases at query #25
#--------------------------

# Failed to parse test_no_map_types_predicate.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_map_structure_zip_with_dict. Retrieved 11/14 statements.


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
    var_9 = 6
    var_10 = {var_0: var_6, var_1: var_9}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = var_3.__class__



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = lambda x, y: x + y
    var_9 = [var_4, var_7]
    var_10 = module_0.map_structure_zip(var_8, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 5
    var_10 = 6
    var_11 = [var_9, var_10]
    var_12 = 7
    var_13 = 8
    var_14 = (var_12, var_13)
    var_15 = {var_0: var_11, var_1: var_14}
    var_16 = lambda x, y: x + y
    var_17 = [var_8, var_15]
    var_18 = module_0.map_structure_zip(var_16, var_17)
    var_19 = bool(var_18 == {'a': [6, 8], 'b': (10, 12)})
    assert var_19 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 8

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    var_0 = bool(not 1 == 0)
    assert var_0 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_predicate_at_line_1.




# Parsed testcases at query #32
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 1/5 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [2, [4, 6]])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': {'b': 2}})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [2, (4, {'a': 6})])
    assert var_9 is True

def test_case_0():
    var_0 = lambda x: x



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 17/23 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = 3
    var_9 = {var_1: var_8}
    var_10 = 4
    var_11 = {var_4: var_10}
    var_12 = [var_9, var_11]
    var_13 = [var_7, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [{'a': 4}, {'b': 6}])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == [10])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 8

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = lambda x, y: x + y
    var_13 = (var_0, var_9)
    var_14 = 6
    var_15 = (var_3, var_14)
    var_16 = [var_13, var_15]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_isinstance_list_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_isinstance_tuple_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, [4, 6], 8])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = {var_5: var_0}
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': 2, 'b': {'c': 4}})
    assert var_9 is True

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
    var_10 = bool(var_9 == [2, (4, 6), {'a': 8}])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_isinstance_tuple_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_isinstance_obj_tuple. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_no_type_check_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = lambda x: x



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #41
#--------------------------




def test_case_0():
    var_0 = bool(True)
    assert var_0 is True



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_map_structure_dict. Retrieved 7/8 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, [4, 6], 8])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = {var_5: var_0}
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': 2, 'b': {'c': 4}})
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

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
    var_10 = bool(var_9 == [2, (4, 6), {'a': 8}])
    assert var_10 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_map_structure_zip_with_set_raises_value_error. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = {var_4, var_5, var_6}
    var_8 = [var_3, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 17/23 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = [var_7, var_8]
    var_10 = 6
    var_11 = [var_10]
    var_12 = [var_9, var_11]
    var_13 = [var_6, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [[5, 7], [9]])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

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
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3

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
    var_13 = bool(var_12 == [5, (7, 9)])
    assert var_13 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = lambda x, y: x + y
    var_13 = (var_0, var_9)
    var_14 = 6
    var_15 = (var_3, var_14)
    var_16 = [var_13, var_15]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, [4, 6], 8])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = {var_5: var_0}
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': 2, 'b': {'c': 4}})
    assert var_9 is True

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
    var_10 = bool(var_9 == [2, (4, 6), {'a': 8}])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = []
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = {}
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = set()
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = set()
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = ()
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == ())
    assert var_4 is True



# Parsed testcases at query #48
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = 3
    var_9 = {var_1: var_8}
    var_10 = 4
    var_11 = {var_4: var_10}
    var_12 = [var_9, var_11]
    var_13 = [var_7, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [{'a': 4}, {'b': 6}])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == [10])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 8



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_isinstance_list_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [2, [4, 6]])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = {var_5: var_0}
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': 2, 'b': {'c': 4}})
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

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
    var_10 = bool(var_9 == [2, (4, 6), {'a': 8}])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




# Parsed testcases at query #3
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 17/23 statements.
# Partially parsed test_map_structure_zip_with_no_map_types. Retrieved 3/9 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = lambda x, y: x + y
    var_9 = [var_4, var_7]
    var_10 = module_0.map_structure_zip(var_8, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 5
    var_10 = 6
    var_11 = [var_9, var_10]
    var_12 = 7
    var_13 = 8
    var_14 = (var_12, var_13)
    var_15 = {var_0: var_11, var_1: var_14}
    var_16 = lambda x, y: x + y
    var_17 = [var_8, var_15]
    var_18 = module_0.map_structure_zip(var_16, var_17)
    var_19 = bool(var_18 == {'a': [6, 8], 'b': (10, 12)})
    assert var_19 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 8

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = lambda x, y: x + y
    var_13 = (var_0, var_9)
    var_14 = 6
    var_15 = (var_3, var_14)
    var_16 = [var_13, var_15]

def test_case_0():
    var_0 = '_no_map'
    var_1 = True
    var_2 = lambda x, y: x

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_predicate_at_line_15_evaluates_to_true.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_map_structure_zip_predicate_true.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_isinstance_list_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_map_structure_zip_predicate.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_map_structure_with_mixed_types. Retrieved 6/8 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 12/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [2, [4, 6]])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': {'b': 2}})
    assert var_8 is True

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_3: var_0}
    var_5 = [var_1, var_2, var_4]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: x * var_4
    var_8 = (var_0, var_4)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_map_structure_set_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_no_type_check_predicate.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_isinstance_tuple_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_isinstance_list_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_isinstance_tuple_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_map_structure_zip_with_no_map_types. Retrieved 1/6 statements.


def test_case_0():
    var_0 = lambda *x: x



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, [4, 6], 8])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = {var_5: var_0}
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': 2, 'b': {'c': 4}})
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'hello'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'HELLO'

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
    var_10 = bool(var_9 == ['1', ('2', '3'), {'a': '4'}])
    assert var_10 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_isinstance_list_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_no_type_check_predicate.




# Parsed testcases at query #20
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [2, [4, 6]])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 1
    var_6 = {var_3: var_5, var_4: var_0}
    var_7 = {var_2: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': {'b': 2, 'c': 4}})
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = [var_2, var_4, var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [2, {'a': 4}, (6, 8)])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hellohello'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 17/23 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = lambda x, y: x + y
    var_9 = [var_4, var_7]
    var_10 = module_0.map_structure_zip(var_8, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 5
    var_10 = 6
    var_11 = [var_9, var_10]
    var_12 = 7
    var_13 = 8
    var_14 = (var_12, var_13)
    var_15 = {var_0: var_11, var_1: var_14}
    var_16 = lambda x, y: x + y
    var_17 = [var_8, var_15]
    var_18 = module_0.map_structure_zip(var_16, var_17)
    var_19 = bool(var_18 == {'a': [6, 8], 'b': (10, 12)})
    assert var_19 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = lambda x, y: x + y
    var_13 = (var_0, var_9)
    var_14 = 6
    var_15 = (var_3, var_14)
    var_16 = [var_13, var_15]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = bool(True)
    assert var_0 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_no_map_type. Retrieved 2/7 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = [var_7, var_8]
    var_10 = 6
    var_11 = [var_10]
    var_12 = [var_9, var_11]
    var_13 = [var_6, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [[5, 7], [9]])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = (var_4,)
    var_6 = (var_3, var_5)
    var_7 = 4
    var_8 = 5
    var_9 = (var_7, var_8)
    var_10 = 6
    var_11 = (var_10,)
    var_12 = (var_9, var_11)
    var_13 = [var_6, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == ((-3, -3), (-3,)))
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

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
    var_0 = lambda x, y: str(x) + str(y)
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
    var_13 = bool(var_12 == ['14', ('25', '36')])
    assert var_13 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == [10])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'mapped'
    var_1 = lambda x, y: var_0

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_map_structure_zip_with_set_raises_value_error. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = {var_4, var_5, var_6}
    var_8 = [var_3, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_isinstance_dict_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = bool(True)
    assert var_0 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = 3
    var_8 = {var_1: var_7}
    var_9 = 4
    var_10 = {var_1: var_9}
    var_11 = [var_8, var_10]
    var_12 = [var_6, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == [{'a': 4}, {'a': 6}])
    assert var_14 is True

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
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == ['13', '24'])
    assert var_9 is True



# Parsed testcases at query #28
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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_isinstance_tuple_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_predicate_evaluates_to_true.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_isinstance_tuple_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #33
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_17 = bool(var_16 == [[6, 8], [10, 12]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = 5
    var_9 = 6
    var_10 = (var_8, var_9)
    var_11 = 7
    var_12 = 8
    var_13 = (var_11, var_12)
    var_14 = (var_10, var_13)
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == ((6, 8), (10, 12)))
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 3
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = 4
    var_11 = 5
    var_12 = {var_3: var_10, var_4: var_11}
    var_13 = 6
    var_14 = {var_1: var_12, var_2: var_13}
    var_15 = [var_9, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == {'a': {'c': 5, 'd': 7}, 'b': 9})
    assert var_17 is True

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
    var_1 = 5
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 8

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
    var_13 = bool(var_12 == [5, (7, 9)])
    assert var_13 is True



# Parsed testcases at query #35
#--------------------------




def test_case_0():
    var_0 = bool(True)
    assert var_0 is True



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    var_0 = bool(True)
    assert var_0 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate. Retrieved 11/12 statements.


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
    var_9 = 0
    var_10 = var_8[var_9]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, [4, 6], 8])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = []
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = {}
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = set()
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = set()
    var_5 = bool(var_3 == var_4)
    assert var_5 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 17/23 statements.


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
    var_9 = bool(var_8 == [4, 6])
    assert var_9 is True

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
    var_9 = bool(var_8 == (3, 8))
    assert var_9 is True

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
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = 3
    var_8 = {var_1: var_7}
    var_9 = 4
    var_10 = {var_1: var_9}
    var_11 = [var_8, var_10]
    var_12 = [var_6, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == [{'a': 4}, {'a': 6}])
    assert var_14 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 3
    var_9 = (var_1, var_8)
    var_10 = 4
    var_11 = (var_4, var_10)
    var_12 = [var_9, var_11]
    var_13 = (var_1, var_10)
    var_14 = 6
    var_15 = (var_4, var_14)
    var_16 = [var_13, var_15]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_map_structure_zip_with_set_raises_value_error. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = {var_4, var_5, var_6}
    var_8 = [var_3, var_7]



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_isinstance_tuple_predicate. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



