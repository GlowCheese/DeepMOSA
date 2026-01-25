####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_ordereddict. Retrieved 17/23 statements.
# Partially parsed test_map_structure_zip_no_map_instance_attr. Retrieved 3/10 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = 7
    var_12 = 9
    var_13 = [var_6, var_11, var_12]
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

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
    var_17 = 12
    var_18 = [var_8, var_17]
    var_19 = 21
    var_20 = 32
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = bool(var_16 == var_22)
    assert var_23 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = 15
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = (var_4, var_4)
    var_10 = bool(var_8 == var_9)
    assert var_10 is True

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
    var_17 = (var_9, var_12)
    var_18 = 10
    var_19 = 12
    var_20 = (var_18, var_19)
    var_21 = (var_17, var_20)
    var_22 = bool(var_16 == var_21)
    assert var_22 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda a, b: a + b
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 4
    var_7 = 5
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = 8
    var_12 = 15
    var_13 = {var_1: var_11, var_2: var_12}
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'key'
    var_2 = 'sub'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 3
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = 7
    var_12 = {var_2: var_11}
    var_13 = {var_1: var_12}
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

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
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = 9
    var_13 = 12
    var_14 = [var_12, var_13]
    var_15 = bool(var_11 == var_14)
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x ** y
    var_1 = 5
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = 125
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = 'helloworld'
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: x.val + y.val
    var_2 = 30

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x | y
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
    var_10 = 'cannot contain `set`'

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
    var_13 = 7
    var_14 = 9
    var_15 = (var_13, var_14)
    var_16 = [var_7, var_15]
    var_17 = bool(var_12 == var_16)
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = [var_4]
    var_6 = module_0.map_structure_zip(var_1, var_5)
    var_7 = 4
    var_8 = 6
    var_9 = [var_0, var_7, var_8]
    var_10 = bool(var_6 == var_9)
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true. Retrieved 19/42 statements.


def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = 3
    var_10 = (var_2, var_9)
    var_11 = 4
    var_12 = (var_5, var_11)
    var_13 = [var_10, var_12]
    var_14 = lambda x, y: x + y
    var_15 = (var_2, var_11)
    var_16 = 6
    var_17 = (var_5, var_16)
    var_18 = [var_15, var_17]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_list. Retrieved 9/16 statements.
# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_empty_list. Retrieved 3/10 statements.
# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_nested_list. Retrieved 15/22 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = (var_3, var_7)

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = (var_6, var_13)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 10/15 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/15 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/6 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 4
    var_6 = (var_0, var_5)
    var_7 = (var_4, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((3, 6), (9, 12)))
    assert var_9 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 5
    var_5 = 10
    var_6 = 2
    var_7 = lambda x: x - var_6
    var_8 = 3
    var_9 = 8

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
    var_8 = bool(var_7 == {'a': 10, 'b': 20})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 100
    var_1 = lambda x: x + var_0
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
    var_12 = bool(var_11 == {'a': [101, 102], 'b': {'c': 103}})
    assert var_12 is True

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = lambda x: x * var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 2
    var_3 = {var_0, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [{0, 1}, {2, 3}])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = {var_7, var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = lambda x: x * var_3
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == {'list': [2, (4, 6)], 'set': {8, 10}})
    assert var_13 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = set()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = set()
    var_12 = bool(var_10 == var_11)
    assert var_12 is True
    var_13 = lambda x: x
    var_14 = ()
    var_15 = module_0.map_structure(var_13, var_14)
    var_16 = bool(var_15 == ())
    assert var_16 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true_for_namedtuple. Retrieved 7/11 statements.
# Partially parsed test_predicate_at_line_19_evaluates_to_true_for_namedtuple_instance. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = '_fields'

def test_case_0():
    var_0 = 'Person'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = [var_1, var_2]
    var_4 = 'Alice'
    var_5 = 30
    var_6 = '_fields'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = '_fields'
    var_5 = hasattr(var_3, var_4)
    assert var_5 is False

def test_case_0():
    var_0 = ()
    var_1 = '_fields'
    var_2 = hasattr(var_0, var_1)
    assert var_2 is False

def test_case_0():
    var_0 = 42
    var_1 = (var_0,)
    var_2 = '_fields'
    var_3 = hasattr(var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_dict. Retrieved 7/8 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_ordereddict. Retrieved 9/15 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_nested_dict. Retrieved 9/10 statements.


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

def test_case_0():
    var_0 = 'x'
    var_1 = 3
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 1
    var_8 = lambda x: x + var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'a'
    var_4 = 'nested'
    var_5 = 'b'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.map_structure(var_0, var_7)
    var_9 = bool(var_8 == {'key1': 'A', 'key2': {'nested': 'B'}})
    assert var_9 is True



# Parsed testcases at query #7
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
    var_4 = lambda a, b: a + b
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 5
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 2
    var_7 = 3
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 3, 'b': 7})
    assert var_11 is True

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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = [var_4]
    var_6 = module_0.map_structure_zip(var_1, var_5)
    var_7 = bool(var_6 == [2, 4, 6])
    assert var_7 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == ['ac', 'bd'])
    assert var_9 is True

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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True



# Parsed testcases at query #8
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
    var_4 = lambda a, b: a + b
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 5
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 2
    var_7 = 3
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 3, 'b': 7})
    assert var_11 is True

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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = module_0.map_structure_zip(var_1, var_3)
    assert var_4 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 3/6 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = (var_2, var_0)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((2, 4), (6, 8)))
    assert var_9 is True

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
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 10
    var_5 = 20
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 9, 'b': 19})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [2, 4], 'b': [6, 8]})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 5
    var_10 = 6
    var_11 = {var_9, var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}
    var_13 = 10
    var_14 = lambda x: x + var_13
    var_15 = module_0.map_structure(var_14, var_12)
    var_16 = bool(var_15 == {'list': [11, 12], 'tuple': (13, 14), 'set': {15, 16}})
    assert var_16 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = lambda x: x * var_7

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = {}
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = set()
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = set()
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = {var_1: var_4}
    var_6 = 'c'
    var_7 = 3
    var_8 = 4
    var_9 = {var_7, var_8}
    var_10 = {var_6: var_9}
    var_11 = [var_5, var_10]
    var_12 = {var_0: var_11}
    var_13 = lambda x: x * var_3
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = bool(var_14 == {'a': [{'b': (2, 4)}, {'c': {6, 8}}]})
    assert var_15 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_list. Retrieved 11/29 statements.


def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = 0
    var_10 = var_8[var_9]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_no_map_types. Retrieved 3/6 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 8/14 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = (var_3, var_0)
    var_5 = (var_2, var_4)
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == (3, (6, 9)))
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

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 5
    var_5 = 10
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 4, 'b': 9})
    assert var_8 is True

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
    var_11 = bool(var_10 == {'a': [2, 4], 'b': {'c': 6}})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 3
    var_3 = 4
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {4, 9, 16})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 'dict'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = (var_7, var_8)
    var_10 = 5
    var_11 = {var_10}
    var_12 = 'nested'
    var_13 = 6
    var_14 = {var_12: var_13}
    var_15 = {var_0: var_6, var_1: var_9, var_2: var_11, var_3: var_14}
    var_16 = 10
    var_17 = lambda x: x * var_16
    var_18 = module_0.map_structure(var_17, var_15)
    var_19 = bool(var_18 == {'list': [10, 20], 'tuple': (30, 40), 'set': {50}, 'dict': {'nested': 60}})
    assert var_19 is True

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 100
    var_1 = lambda x: x + var_0
    var_2 = 50
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 150

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: x * var_4

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = ()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = bool(var_10 == ())
    assert var_11 is True
    var_12 = lambda x: x
    var_13 = set()
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = set()
    var_16 = bool(var_14 == var_15)
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [None, None, None])
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true_for_namedtuple. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = '_fields'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = '_fields'
    var_5 = hasattr(var_3, var_4)
    assert var_5 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_map_structure_zip_dict_ordereddict. Retrieved 17/24 statements.


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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/16 statements.
# Partially parsed test_map_structure_zip_ordered_dict. Retrieved 17/22 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

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
    var_17 = bool(var_16 == [[5, 12], [21, 32]])
    assert var_17 is True

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
    var_9 = bool(var_8 == (-2, -2))
    assert var_9 is True

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
    var_0 = lambda x, y: x * y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'x'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'y'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 4
    var_11 = {var_3: var_10}
    var_12 = 5
    var_13 = {var_6: var_12}
    var_14 = {var_1: var_11, var_2: var_13}
    var_15 = [var_9, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == {'a': {'x': 8}, 'b': {'y': 15}})
    assert var_17 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda a, b: a + b
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
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = [var_4]
    var_6 = module_0.map_structure_zip(var_1, var_5)
    var_7 = bool(var_6 == [2, 4, 6])
    assert var_7 is True

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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == ['ac', 'bd'])
    assert var_9 is True

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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 13/20 statements.


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
    var_4 = lambda a, b: a + b
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 5
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 2
    var_7 = 3
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 3, 'b': 7})
    assert var_11 is True

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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = module_0.map_structure_zip(var_1, var_3)
    assert var_4 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

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
    var_0 = lambda x, y: x * y
    var_1 = 'a'
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 4
    var_9 = (var_1, var_8)
    var_10 = 5
    var_11 = (var_4, var_10)
    var_12 = [var_9, var_11]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/6 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 9/15 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 4
    var_6 = (var_0, var_5)
    var_7 = (var_4, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((3, 6), (9, 12)))
    assert var_9 is True

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
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 5
    var_5 = 10
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 4, 'b': 9})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x / var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 4
    var_5 = [var_0, var_4]
    var_6 = 6
    var_7 = 8
    var_8 = [var_6, var_7]
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = frozenset(var_4)
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = frozenset(var_8)
    var_10 = {var_5, var_9}
    var_11 = module_0.map_structure(var_1, var_10)
    var_12 = 11
    var_13 = 12
    var_14 = {var_12, var_13}
    var_15 = frozenset(var_14)
    var_16 = 13
    var_17 = 14
    var_18 = {var_16, var_17}
    var_19 = frozenset(var_18)
    var_20 = {var_15, var_19}
    var_21 = bool(var_11 == var_20)
    assert var_21 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 42
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 84

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 5
    var_10 = 6
    var_11 = {var_9, var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}
    var_13 = lambda x: x * var_4
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = bool(var_14 == {'list': [2, 4], 'tuple': (6, 8), 'set': {10, 12}})
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = ()
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == ())
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = {}
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = bool(var_10 == {})
    assert var_11 is True
    var_12 = lambda x: x
    var_13 = set()
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = set()
    var_16 = bool(var_14 == var_15)
    assert var_16 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = lambda x: x * var_7



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_map_structure_zip_no_map_types. Retrieved 2/30 statements.
# Partially parsed test_map_structure_zip_primitive_type. Retrieved 4/25 statements.
# Partially parsed test_map_structure_zip_list. Retrieved 8/28 statements.


def test_case_0():
    var_0 = '_no_map'
    var_1 = True

def test_case_0():
    var_0 = '_no_map'
    var_1 = 5
    var_2 = 3
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = '_no_map'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/6 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 4
    var_6 = (var_0, var_5)
    var_7 = (var_4, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((3, 6), (9, 12)))
    assert var_9 is True

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
    var_0 = 5
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 10
    var_5 = 20
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 5, 'b': 15})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x / var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 4
    var_5 = [var_0, var_4]
    var_6 = 6
    var_7 = 8
    var_8 = [var_6, var_7]
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = frozenset(var_4)
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = frozenset(var_8)
    var_10 = {var_5, var_9}
    var_11 = module_0.map_structure(var_1, var_10)
    var_12 = 11
    var_13 = 12
    var_14 = {var_12, var_13}
    var_15 = frozenset(var_14)
    var_16 = 13
    var_17 = 14
    var_18 = {var_16, var_17}
    var_19 = frozenset(var_18)
    var_20 = {var_15, var_19}
    var_21 = bool(var_11 == var_20)
    assert var_21 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 100
    var_8 = lambda x: x * var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 42
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 43

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 5
    var_10 = 6
    var_11 = {var_9, var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}
    var_13 = lambda x: x * var_4
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = bool(var_14 == {'list': [2, 4], 'tuple': (6, 8), 'set': {10, 12}})
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = 'b'
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = [var_4, var_9]
    var_11 = lambda x: x - var_1
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == [{'a': (0, 1)}, {'b': [2, 3]}])
    assert var_13 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_true. Retrieved 6/27 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '_no_map'
    var_5 = True
    var_6 = 2
    var_7 = lambda x: x * var_6



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 1/27 statements.


def test_case_0():
    var_0 = '_no_map'



# Parsed testcases at query #21
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda *args: sum(args)
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = set(var_3)
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = set(var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_list. Retrieved 13/30 statements.
# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_empty_list. Retrieved 9/26 statements.
# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_nested_list. Retrieved 21/38 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = 0
    var_12 = var_8[var_11]
    var_13 = bool(var_10 == [4, 6])
    assert var_13 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = []
    var_3 = []
    var_4 = [var_2, var_3]
    var_5 = lambda x, y: x + y
    var_6 = module_0.map_structure_zip(var_5, var_4)
    var_7 = 0
    var_8 = var_4[var_7]
    var_9 = bool(var_6 == [])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = 5
    var_10 = 6
    var_11 = [var_9, var_10]
    var_12 = 7
    var_13 = 8
    var_14 = [var_12, var_13]
    var_15 = [var_11, var_14]
    var_16 = [var_8, var_15]
    var_17 = lambda x, y: x + y
    var_18 = module_0.map_structure_zip(var_17, var_16)
    var_19 = 0
    var_20 = var_16[var_19]
    var_21 = bool(var_18 == [[6, 8], [10, 12]])
    assert var_21 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordereddict. Retrieved 13/20 statements.


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
    var_4 = lambda a, b: a + b
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 5
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 2
    var_7 = 3
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 3, 'b': 7})
    assert var_11 is True

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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = module_0.map_structure_zip(var_1, var_3)
    assert var_4 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

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
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 5
    var_11 = 6
    var_12 = [var_10, var_11]
    var_13 = 7
    var_14 = 8
    var_15 = [var_13, var_14]
    var_16 = {var_1: var_12, var_2: var_15}
    var_17 = [var_9, var_16]
    var_18 = module_0.map_structure_zip(var_0, var_17)
    var_19 = bool(var_18 == {'a': [6, 8], 'b': [10, 12]})
    assert var_19 is True

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 'a'
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 4
    var_9 = (var_1, var_8)
    var_10 = 5
    var_11 = (var_4, var_10)
    var_12 = [var_9, var_11]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_dict. Retrieved 7/8 statements.


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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_map_structure_with_no_map_types. Retrieved 3/24 statements.


def test_case_0():
    var_0 = '_no_map'
    var_1 = 1
    var_2 = lambda x: x + var_1



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true_for_dict. Retrieved 9/14 statements.
# Partially parsed test_predicate_at_line_24_evaluates_to_true_for_ordereddict. Retrieved 12/23 statements.
# Partially parsed test_predicate_at_line_24_evaluates_to_true_for_nested_dict. Retrieved 15/20 statements.
# Partially parsed test_predicate_at_line_24_evaluates_to_true_for_empty_dict. Retrieved 3/8 statements.
# Partially parsed test_predicate_at_line_24_evaluates_to_true_for_dict_with_mixed_types. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'hello'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'world'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'foo'
    var_8 = (var_0, var_7)
    var_9 = 'bar'
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 2
    var_5 = 3
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 5
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = 4
    var_10 = 6
    var_11 = {var_2: var_9, var_3: var_10}
    var_12 = 7
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = [var_8, var_13]

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 'a'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_types. Retrieved 3/6 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/15 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = (var_3, var_0)
    var_5 = (var_2, var_4)
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == (3, (6, 9)))
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 20
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 10, 'b': 20})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: len(x)
    var_1 = 'k1'
    var_2 = 'k2'
    var_3 = 'ab'
    var_4 = 'cde'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)
    var_7 = bool(var_6 == {'k1': 2, 'k2': 3})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 3
    var_3 = 4
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {4, 9, 16})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 5
    var_10 = 6
    var_11 = {var_9, var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}
    var_13 = lambda x: x * var_4
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = bool(var_14 == {'a': [2, 4], 'b': (6, 8), 'c': {10, 12}})
    assert var_15 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 5
    var_5 = 10
    var_6 = 2
    var_7 = lambda x: x / var_6
    var_8 = 2.5

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 100
    var_1 = lambda x: x + var_0
    var_2 = 50
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 150

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 10
    var_8 = lambda x: x * var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = ()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = bool(var_10 == ())
    assert var_11 is True
    var_12 = lambda x: x
    var_13 = set()
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = set()
    var_16 = bool(var_14 == var_15)
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = {var_1: var_4}
    var_6 = 4
    var_7 = 5
    var_8 = {var_6, var_7}
    var_9 = [var_0, var_5, var_8]
    var_10 = lambda x: x
    var_11 = module_0.map_structure(var_10, var_9)
    var_12 = bool(var_11 == [1, {'a': (2, 3)}, {4, 5}])
    assert var_12 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_namedtuple_mapping. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4
    var_8 = '_fields'



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_no_map_types_predicate_false.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/13 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/6 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = (var_2, var_0)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((2, 4), (6, 8)))
    assert var_9 is True

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
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 5
    var_5 = 10
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 4, 'b': 9})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [2, 4], 'b': [6, 8]})
    assert var_11 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 10
    var_8 = lambda x: x + var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 100
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = frozenset(var_4)
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = frozenset(var_8)
    var_10 = {var_5, var_9}
    var_11 = module_0.map_structure(var_1, var_10)
    var_12 = 101
    var_13 = 102
    var_14 = {var_12, var_13}
    var_15 = frozenset(var_14)
    var_16 = 103
    var_17 = 104
    var_18 = {var_16, var_17}
    var_19 = frozenset(var_18)
    var_20 = {var_15, var_19}
    var_21 = bool(var_11 == var_20)
    assert var_21 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 'inner_dict'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = (var_7, var_8)
    var_10 = 5
    var_11 = 6
    var_12 = {var_10, var_11}
    var_13 = 'a'
    var_14 = 7
    var_15 = {var_13: var_14}
    var_16 = {var_0: var_6, var_1: var_9, var_2: var_12, var_3: var_15}
    var_17 = lambda x: x * var_5
    var_18 = module_0.map_structure(var_17, var_16)
    var_19 = [var_5, var_8]
    var_20 = 8
    var_21 = (var_11, var_20)
    var_22 = 10
    var_23 = 12
    var_24 = {var_22, var_23}
    var_25 = 14
    var_26 = {var_13: var_25}
    var_27 = {var_0: var_19, var_1: var_21, var_2: var_24, var_3: var_26}
    var_28 = bool(var_18 == var_27)
    assert var_28 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = set()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = set()
    var_12 = bool(var_10 == var_11)
    assert var_12 is True
    var_13 = lambda x: x
    var_14 = ()
    var_15 = module_0.map_structure(var_13, var_14)
    var_16 = bool(var_15 == ())
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [None, None, None])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = 5
    var_8 = {var_7}
    var_9 = [var_5, var_6, var_8]
    var_10 = [var_4, var_9]
    var_11 = lambda x: x
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == var_10)
    assert var_13 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = lambda x: x + var_0
    var_6 = module_0.map_structure(var_5, var_4)
    var_7 = bool(var_6 == [[[[2]]]])
    assert var_7 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 4/30 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '_no_map'
    var_5 = True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 10/15 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = (var_2, var_0)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((2, 4), (6, 8)))
    assert var_9 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = lambda x: x + var_6
    var_8 = 11
    var_9 = 12

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 5
    var_5 = 10
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 4, 'b': 9})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [2, 4], 'b': [6, 8]})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 3
    var_3 = 4
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {4, 9, 16})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 5
    var_10 = 6
    var_11 = {var_9, var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}
    var_13 = 10
    var_14 = lambda x: x + var_13
    var_15 = module_0.map_structure(var_14, var_12)
    var_16 = bool(var_15 == {'list': [11, 12], 'tuple': (13, 14), 'set': {15, 16}})
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = lambda x: x * var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = ()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = bool(var_10 == ())
    assert var_11 is True
    var_12 = lambda x: x
    var_13 = set()
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = set()
    var_16 = bool(var_14 == var_15)
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = {var_1: var_4}
    var_6 = 'c'
    var_7 = 3
    var_8 = 4
    var_9 = {var_7, var_8}
    var_10 = {var_6: var_9}
    var_11 = [var_5, var_10]
    var_12 = {var_0: var_11}
    var_13 = lambda x: x * var_3
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = bool(var_14 == {'a': [{'b': (2, 4)}, {'c': {6, 8}}]})
    assert var_15 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_true_for_namedtuple. Retrieved 7/11 statements.
# Partially parsed test_predicate_at_line_13_evaluates_true_for_namedtuple_with_multiple_fields. Retrieved 9/13 statements.
# Partially parsed test_predicate_at_line_13_evaluates_true_for_namedtuple_empty. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = '_fields'

def test_case_0():
    var_0 = 'Person'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'city'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Alice'
    var_6 = 30
    var_7 = 'NYC'
    var_8 = '_fields'

def test_case_0():
    var_0 = 'Empty'
    var_1 = []
    var_2 = '_fields'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_custom_no_map_type. Retrieved 2/10 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

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
    var_15 = (var_7, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[5, 12], [21, 32]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda a, b: a - b
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = 15
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == (5, 5))
    assert var_9 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda p, q: p + q
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda u, v: u + v
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = (var_5, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'k'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = (var_3, var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_8, var_9]
    var_11 = 6
    var_12 = {var_4: var_11}
    var_13 = (var_10, var_12)
    var_14 = (var_7, var_13)
    var_15 = module_0.map_structure_zip(var_0, var_14)
    var_16 = bool(var_15 == ([5, 7], {'k': 9}))
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = (var_3, var_6, var_9)
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 5
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 50

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'cannot contain `set`'

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: x + y



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 5/27 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '_no_map'
    var_5 = True
    var_6 = lambda x: x



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_types. Retrieved 2/6 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 2/7 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = (var_2, var_0)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((2, 4), (6, 8)))
    assert var_9 is True

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
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 5
    var_5 = 10
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 4, 'b': 9})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [2, 4], 'b': [6, 8]})
    assert var_11 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = lambda x: x * var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = [var_4, var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [{11, 12}, {13, 14}])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = 5
    var_8 = {var_6, var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_4, var_1: var_9}
    var_11 = lambda x: x * var_3
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == {'a': (2, 4), 'b': [6, {8, 10}]})
    assert var_13 is True

def test_case_0():
    var_0 = 'mapped'
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = 'transformed'
    var_1 = lambda x: var_0

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 'none'
    var_1 = lambda x: var_0
    var_2 = None
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'none'

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = set()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = set()
    var_12 = bool(var_10 == var_11)
    assert var_12 is True
    var_13 = lambda x: x
    var_14 = ()
    var_15 = module_0.map_structure(var_13, var_14)
    var_16 = bool(var_15 == ())
    assert var_16 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 5/26 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '_no_map'
    var_5 = True
    var_6 = lambda x: x



# Parsed testcases at query #3
#--------------------------




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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_custom_no_map_type. Retrieved 2/10 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

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
    var_15 = (var_7, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[5, 12], [21, 32]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda a, b: a - b
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = 15
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == (5, 5))
    assert var_9 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda p, q: p + q
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda u, v: u + v
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = (var_5, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 11, 'b': 22})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 'a'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = (var_5,)
    var_7 = 3
    var_8 = 4
    var_9 = {var_2: var_8}
    var_10 = [var_7, var_9]
    var_11 = (var_10,)
    var_12 = (var_6, var_11)
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == ([4, {'a': 6}],))
    assert var_14 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = (var_3, var_6, var_9)
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 5
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 50

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'cannot contain `set`'

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: x + y



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_dict. Retrieved 7/8 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_ordereddict. Retrieved 11/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 2, 'b': 2})
    assert var_7 is True

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
    var_9 = (var_3, var_4)
    var_10 = [var_8, var_9]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_true_for_namedtuple. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = '_fields'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 6/27 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '_no_map'
    var_5 = True
    var_6 = 2
    var_7 = lambda x: x * var_6



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordereddict. Retrieved 13/20 statements.


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
    var_17 = bool(var_16 == [[5, 12], [21, 32]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = 15
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == (5, 5))
    assert var_9 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda a, b: a + b
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

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 'a'
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 4
    var_9 = (var_1, var_8)
    var_10 = 5
    var_11 = (var_4, var_10)
    var_12 = [var_9, var_11]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = (var_3, var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_8, var_9]
    var_11 = 6
    var_12 = {var_4: var_11}
    var_13 = (var_10, var_12)
    var_14 = [var_7, var_13]
    var_15 = module_0.map_structure_zip(var_0, var_14)
    var_16 = bool(var_15 == ([5, 7], {'a': 9}))
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = [var_4]
    var_6 = module_0.map_structure_zip(var_1, var_5)
    var_7 = bool(var_6 == [2, 4, 6])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

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
    var_1 = 'hello'
    var_2 = ' world'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'hello world'

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
    var_10 = 'cannot contain `set`'

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == [[], []])
    assert var_9 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 11/49 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = ()
    var_5 = '_no_map'
    var_6 = True
    var_7 = lambda x: x
    var_8 = lambda x: x
    var_9 = lambda x: x
    var_10 = 'key'
    var_11 = lambda x: x
    var_12 = lambda x: x



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_true_for_set. Retrieved 6/7 statements.


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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 3/6 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3, var_0)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (3, 6, 9))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 2
    var_3 = (var_0, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((0, 1), (2, 3)))
    assert var_9 is True

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
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 'world'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)
    var_7 = bool(var_6 == {'a': 'HELLO', 'b': 'WORLD'})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [2, 4], 'b': [6, 8]})
    assert var_11 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 10
    var_8 = lambda x: x + var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 100
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = [var_4, var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [{101, 102}, {103, 104}])
    assert var_10 is True

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = 4
    var_8 = 5
    var_9 = {var_7, var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = lambda x: x * var_3
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == {'a': (2, [4, 6]), 'b': {8, 10}})
    assert var_13 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = set()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = set()
    var_12 = bool(var_10 == var_11)
    assert var_12 is True
    var_13 = lambda x: x
    var_14 = ()
    var_15 = module_0.map_structure(var_13, var_14)
    var_16 = bool(var_15 == ())
    assert var_16 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_map_structure_on_flat_tuple. Retrieved 4/5 statements.
# Partially parsed test_map_structure_on_namedtuple. Retrieved 10/15 statements.
# Partially parsed test_map_structure_on_nested_dict. Retrieved 8/9 statements.
# Partially parsed test_map_structure_on_custom_no_map_instance. Retrieved 3/6 statements.
# Partially parsed test_map_structure_preserves_ordered_dict. Retrieved 8/14 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x * x
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure(var_0, var_7)
    var_9 = bool(var_8 == ((1, 4), (9, 16)))
    assert var_9 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = lambda x: x + var_6
    var_8 = 11
    var_9 = 12

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 'world'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)
    var_7 = bool(var_6 == {'a': 'HELLO', 'b': 'WORLD'})
    assert var_7 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'ab'
    var_3 = 'cd'
    var_4 = [var_2, var_3]
    var_5 = 'efg'
    var_6 = (var_5,)
    var_7 = {var_0: var_4, var_1: var_6}

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x % var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = {var_2, var_0, var_3, var_4}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {1, 0})
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: x * var_4

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 5
    var_10 = 6
    var_11 = {var_9, var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}
    var_13 = lambda x: x - var_3
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = bool(var_14 == {'list': [0, 1], 'tuple': (2, 3), 'set': {4, 5}})
    assert var_15 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/6 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 8/14 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 4
    var_6 = (var_0, var_5)
    var_7 = (var_4, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((3, 6), (9, 12)))
    assert var_9 is True

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
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 5
    var_5 = 10
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 4, 'b': 9})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x / var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 4
    var_5 = [var_0, var_4]
    var_6 = 'c'
    var_7 = 6
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [1.0, 2.0], 'b': {'c': 3.0}})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = frozenset(var_4)
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = frozenset(var_8)
    var_10 = {var_5, var_9}
    var_11 = module_0.map_structure(var_1, var_10)
    var_12 = 11
    var_13 = 12
    var_14 = {var_12, var_13}
    var_15 = frozenset(var_14)
    var_16 = 13
    var_17 = 14
    var_18 = {var_16, var_17}
    var_19 = frozenset(var_18)
    var_20 = {var_15, var_19}
    var_21 = bool(var_11 == var_20)
    assert var_21 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 100
    var_1 = lambda x: x + var_0
    var_2 = 50
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 150

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: x * var_4

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = {var_7, var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = lambda x: x * var_3
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == {'list': [2, (4, 6)], 'set': {8, 10}})
    assert var_13 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [None, None, None])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = set()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = set()
    var_12 = bool(var_10 == var_11)
    assert var_12 is True
    var_13 = lambda x: x
    var_14 = ()
    var_15 = module_0.map_structure(var_13, var_14)
    var_16 = bool(var_15 == ())
    assert var_16 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true. Retrieved 14/36 statements.


def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = lambda x, y: x + y
    var_10 = (var_2, var_6)
    var_11 = 4
    var_12 = (var_5, var_11)
    var_13 = [var_10, var_12]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_list. Retrieved 13/30 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = 0
    var_12 = var_8[var_11]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 5/32 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = None
    var_5 = type(var_4)
    var_6 = '_no_map'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_map_structure_with_no_map_types. Retrieved 3/8 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/6 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 10/15 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 8/13 statements.


def test_case_0():
    var_0 = '_no_map'
    var_1 = 1
    var_2 = lambda x: x + var_1

def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = lambda x: x + var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_3, var_4]
    var_6 = lambda x: x + var_0
    var_7 = module_0.map_structure(var_6, var_5)
    var_8 = bool(var_7 == [2, [3, 4], 5])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = bool(var_5 == (2, 4, 6))
    assert var_6 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = lambda x: x + var_6
    var_8 = 11
    var_9 = 12

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = lambda x: x * var_5
    var_7 = module_0.map_structure(var_6, var_4)
    var_8 = bool(var_7 == {'a': 3, 'b': 6})
    assert var_8 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: x - var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = lambda x: x ** var_1
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 42
    var_1 = 2
    var_2 = lambda x: x / var_1
    var_3 = module_0.map_structure(var_2, var_0)
    var_4 = bool(var_3 == 21.0)
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_dict. Retrieved 7/8 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_ordereddict. Retrieved 8/14 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_nested_dict. Retrieved 10/11 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_empty_dict. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: x * var_4

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 'z'
    var_5 = 5
    var_6 = 10
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == {'x': {'y': 6, 'z': 11}})
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = {}
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

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
    var_15 = (var_7, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[5, 12], [21, 32]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = (var_5, var_6, var_7)
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == (-3, -3, -3))
    assert var_11 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda a, b: a + b
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
    var_9 = (var_5, var_8)
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
    var_13 = (var_7, var_12)
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [{'a': 4}, {'b': 6}])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = (var_3, var_6, var_9)
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_4, var_5]
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == ['ac', 'bd'])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 5/32 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = None
    var_5 = type(var_4)
    var_6 = '_no_map'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/6 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 4
    var_6 = (var_0, var_5)
    var_7 = (var_4, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((3, 6), (9, 12)))
    assert var_9 is True

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
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 5
    var_5 = 10
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 4, 'b': 9})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x / var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 4
    var_5 = [var_0, var_4]
    var_6 = 'c'
    var_7 = 6
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [1.0, 2.0], 'b': {'c': 3.0}})
    assert var_11 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 10
    var_8 = lambda x: x * var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = [var_4, var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [{11, 12}, {13, 14}])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = 'b'
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = {var_5: var_8}
    var_10 = [var_4, var_9]
    var_11 = lambda x: x * var_2
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == [{'a': (2, 4)}, {'b': {6, 8}}])
    assert var_13 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = set()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = set()
    var_12 = bool(var_10 == var_11)
    assert var_12 is True
    var_13 = lambda x: x
    var_14 = ()
    var_15 = module_0.map_structure(var_13, var_14)
    var_16 = bool(var_15 == ())
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [None, None, None])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = {var_1: var_4}
    var_6 = 4
    var_7 = 5
    var_8 = {var_6, var_7}
    var_9 = [var_0, var_5, var_8]
    var_10 = lambda x: x
    var_11 = module_0.map_structure(var_10, var_9)
    var_12 = bool(var_11 == [1, {'a': (2, 3)}, {4, 5}])
    assert var_12 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/16 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 17/22 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

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
    var_15 = (var_7, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[5, 12], [21, 32]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = (var_5, var_6, var_7)
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == (-3, -3, -3))
    assert var_11 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda a, b: a + b
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
    var_9 = (var_5, var_8)
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
    var_13 = (var_7, var_12)
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [{'a': 4}, {'b': 6}])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = (var_3, var_6, var_9)
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'helloworld'

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'cannot contain `set`'

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = (var_4,)
    var_6 = module_0.map_structure_zip(var_1, var_5)
    var_7 = bool(var_6 == [2, 4, 6])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 3
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = (var_5, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': {'b': 6}})
    assert var_11 is True

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
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = [var_4, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = (var_7, var_8)
    var_10 = {var_0: var_9}
    var_11 = 6
    var_12 = [var_10, var_11]
    var_13 = (var_6, var_12)
    var_14 = lambda a, b: a + b
    var_15 = module_0.map_structure_zip(var_14, var_13)
    var_16 = bool(var_15 == [{'x': (5, 7)}, 9])
    assert var_16 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 10/15 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/6 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = (var_2, var_0)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((2, 4), (6, 8)))
    assert var_9 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = lambda x: x + var_6
    var_8 = 11
    var_9 = 12

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 3, 'b': 6})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = [var_0, var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [0, 1], 'b': {'c': 2}})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 100
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = frozenset(var_4)
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = frozenset(var_8)
    var_10 = {var_5, var_9}
    var_11 = module_0.map_structure(var_1, var_10)
    var_12 = 101
    var_13 = 102
    var_14 = {var_12, var_13}
    var_15 = frozenset(var_14)
    var_16 = 103
    var_17 = 104
    var_18 = {var_16, var_17}
    var_19 = frozenset(var_18)
    var_20 = {var_15, var_19}
    var_21 = bool(var_11 == var_20)
    assert var_21 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 10
    var_8 = lambda x: x * var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'set'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = {var_7, var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = lambda x: x * var_3
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == {'list': [2, (4, 6)], 'set': {8, 10}})
    assert var_13 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = set()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = set()
    var_12 = bool(var_10 == var_11)
    assert var_12 is True
    var_13 = lambda x: x
    var_14 = ()
    var_15 = module_0.map_structure(var_13, var_14)
    var_16 = bool(var_15 == ())
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [None, None, None])
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/16 statements.
# Partially parsed test_map_structure_zip_with_custom_no_map_type. Retrieved 3/8 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

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
    var_15 = (var_7, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[5, 12], [21, 32]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = (var_5, var_6, var_7)
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == (-3, -3, -3))
    assert var_11 is True

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
    var_15 = (var_7, var_14)
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
    var_9 = (var_5, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'x'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'y'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 4
    var_11 = {var_3: var_10}
    var_12 = 5
    var_13 = {var_6: var_12}
    var_14 = {var_1: var_11, var_2: var_13}
    var_15 = (var_9, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == {'a': {'x': 8}, 'b': {'y': 15}})
    assert var_17 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda a, b: a + b
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
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = 3
    var_9 = {var_1: var_8}
    var_10 = 4
    var_11 = {var_4: var_10}
    var_12 = [var_9, var_11]
    var_13 = (var_7, var_12)
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [{'a': 4}, {'b': 6}])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = (var_4,)
    var_6 = module_0.map_structure_zip(var_1, var_5)
    var_7 = bool(var_6 == [2, 4, 6])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = (var_3, var_6, var_9)
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x and y
    var_1 = True
    var_2 = False
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 is False

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'cannot contain `set`'

def test_case_0():
    var_0 = True
    var_1 = 42
    var_2 = lambda x, y: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x, y: var_0
    var_2 = {}
    var_3 = {}
    var_4 = (var_2, var_3)
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == {})
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = set()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true_for_namedtuple. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = '_fields'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = '_fields'
    var_5 = hasattr(var_3, var_4)
    assert var_5 is False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordereddict. Retrieved 16/22 statements.


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
    var_9 = 6
    var_10 = [var_5, var_9]
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

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
    var_17 = 12
    var_18 = [var_8, var_17]
    var_19 = 21
    var_20 = 32
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = bool(var_16 == var_22)
    assert var_23 is True

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
    var_9 = -2
    var_10 = -2
    var_11 = (var_9, var_10)
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda a, b: a + b
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

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
    var_11 = 5.0
    var_12 = {var_1: var_11, var_2: var_11}
    var_13 = bool(var_10 == var_12)
    assert var_13 is True

def test_case_0():
    var_0 = lambda x, y: x ** y
    var_1 = 'a'
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = (var_1, var_5)
    var_9 = (var_4, var_2)
    var_10 = [var_8, var_9]
    var_11 = 8
    var_12 = (var_1, var_11)
    var_13 = 9
    var_14 = (var_4, var_13)
    var_15 = [var_12, var_14]

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = module_0.map_structure_zip(var_1, var_3)
    var_5 = 10
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    var_6 = 6
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: str(x) + str(y)
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
    var_12 = '13'
    var_13 = '24'
    var_14 = [var_12, var_13]
    var_15 = {var_1: var_14}
    var_16 = bool(var_11 == var_15)
    assert var_16 is True

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
    var_10 = 'cannot contain `set`'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_no_map_types. Retrieved 3/6 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = (var_2, var_0)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((2, 4), (6, 8)))
    assert var_9 is True

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
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 3, 'b': 6})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
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
    var_12 = bool(var_11 == {'a': [11, 12], 'b': {'c': 13}})
    assert var_12 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: x - var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 100
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = frozenset(var_4)
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = frozenset(var_8)
    var_10 = {var_5, var_9}
    var_11 = module_0.map_structure(var_1, var_10)
    var_12 = 101
    var_13 = 102
    var_14 = {var_12, var_13}
    var_15 = frozenset(var_14)
    var_16 = 103
    var_17 = 104
    var_18 = {var_16, var_17}
    var_19 = frozenset(var_18)
    var_20 = {var_15, var_19}
    var_21 = bool(var_11 == var_20)
    assert var_21 is True

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'set'
    var_3 = 'inner_dict'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = (var_7, var_8)
    var_10 = 5
    var_11 = 6
    var_12 = {var_10, var_11}
    var_13 = 'a'
    var_14 = 7
    var_15 = {var_13: var_14}
    var_16 = {var_0: var_6, var_1: var_9, var_2: var_12, var_3: var_15}
    var_17 = 10
    var_18 = lambda x: x * var_17
    var_19 = module_0.map_structure(var_18, var_16)
    var_20 = 20
    var_21 = [var_17, var_20]
    var_22 = 30
    var_23 = 40
    var_24 = (var_22, var_23)
    var_25 = 50
    var_26 = 60
    var_27 = {var_25, var_26}
    var_28 = 70
    var_29 = {var_13: var_28}
    var_30 = {var_0: var_21, var_1: var_24, var_2: var_27, var_3: var_29}
    var_31 = bool(var_19 == var_30)
    assert var_31 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = lambda x: x
    var_5 = {}
    var_6 = module_0.map_structure(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True
    var_8 = lambda x: x
    var_9 = set()
    var_10 = module_0.map_structure(var_8, var_9)
    var_11 = set()
    var_12 = bool(var_10 == var_11)
    assert var_12 is True
    var_13 = lambda x: x
    var_14 = ()
    var_15 = module_0.map_structure(var_13, var_14)
    var_16 = bool(var_15 == ())
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [None, None, None])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = 5
    var_8 = {var_6, var_7}
    var_9 = [var_5, var_8]
    var_10 = [var_4, var_9]
    var_11 = lambda x: x
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == [{'a': (1, 2)}, [3, {4, 5}]])
    assert var_13 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_true_for_set. Retrieved 6/7 statements.


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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 13/20 statements.
# Partially parsed test_map_structure_zip_with_no_map_types. Retrieved 3/8 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

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
    var_15 = (var_7, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[5, 12], [21, 32]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda a, b: a - b
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = 15
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == (5, 5))
    assert var_9 is True

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
    var_15 = (var_7, var_14)
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
    var_9 = (var_5, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 'a'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'b'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 4
    var_11 = {var_3: var_10}
    var_12 = 5
    var_13 = {var_6: var_12}
    var_14 = {var_1: var_11, var_2: var_13}
    var_15 = (var_9, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == {'x': {'a': 8}, 'y': {'b': 15}})
    assert var_17 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda a, b: a + b
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = (var_3, var_6, var_9)
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: str(x) + str(y)
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
    var_13 = (var_7, var_12)
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [{'a': '13'}, {'b': '24'}])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x > y
    var_1 = 5
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = (var_4,)
    var_6 = module_0.map_structure_zip(var_1, var_5)
    var_7 = bool(var_6 == [2, 4, 6])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True

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

def test_case_0():
    var_0 = True
    var_1 = 'combined'
    var_2 = lambda x, y: var_1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_ordered_dict. Retrieved 16/22 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

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
    var_15 = (var_7, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[5, 12], [21, 32]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda a, b: a - b
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == (5, 17))
    assert var_9 is True

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
    var_15 = (var_7, var_14)
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
    var_9 = (var_5, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 'a'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'b'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 4
    var_11 = {var_3: var_10}
    var_12 = 5
    var_13 = {var_6: var_12}
    var_14 = {var_1: var_11, var_2: var_13}
    var_15 = (var_9, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == {'x': {'a': 8}, 'y': {'b': 15}})
    assert var_17 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda a, b: a + b
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
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = 3
    var_9 = {var_1: var_8}
    var_10 = 4
    var_11 = {var_4: var_10}
    var_12 = [var_9, var_11]
    var_13 = (var_7, var_12)
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [{'a': 4}, {'b': 6}])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = (var_3, var_6, var_9)
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = (var_4,)
    var_6 = module_0.map_structure_zip(var_1, var_5)
    var_7 = bool(var_6 == [2, 4, 6])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = [var_1, var_2]
    var_4 = '!'
    var_5 = '?'
    var_6 = [var_4, var_5]
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == ['hello!', 'world?'])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = {}
    var_2 = {}
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 5
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 2
    var_9 = (var_1, var_8)
    var_10 = 1
    var_11 = (var_4, var_10)
    var_12 = [var_9, var_11]
    var_13 = (var_1, var_5)
    var_14 = (var_4, var_8)
    var_15 = [var_13, var_14]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_list. Retrieved 7/15 statements.
# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_empty_list. Retrieved 3/11 statements.
# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_nested_list. Retrieved 16/26 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = (var_6, var_13)
    var_15 = 0



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 10/15 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 2/7 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/15 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [[2, 3], [4, 5]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = (var_2, var_0)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == ((2, 4), (6, 8)))
    assert var_9 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = lambda x: x + var_6
    var_8 = 11
    var_9 = 12

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 3, 'b': 6})
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = [var_0, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [0, 1], 'b': [2, 3]})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 5
    var_10 = 6
    var_11 = {var_9, var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}
    var_13 = lambda x: x * var_4
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = bool(var_14 == {'a': [2, 4], 'b': (6, 8), 'c': {10, 12}})
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = 'mapped'
    var_1 = lambda x: var_0

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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = {}
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = set()
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = set()
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 10
    var_8 = lambda x: x * var_7

import flutes.structure as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [None, None, None])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_0, var_3, var_6]
    var_8 = lambda x: x
    var_9 = module_0.map_structure(var_8, var_7)
    var_10 = bool(var_9 == [1, {'a': 2}, (3, 4)])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = []
    var_4 = ()
    var_5 = {}
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'filled'
    var_8 = lambda x: var_7
    var_9 = module_0.map_structure(var_8, var_6)
    var_10 = bool(var_9 == {'a': [], 'b': (), 'c': {}})
    assert var_10 is True



