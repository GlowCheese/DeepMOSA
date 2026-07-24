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
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

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
    var_9 = bool(var_8 == [[2], [3, [4]]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('1', '2', '3'))
    assert var_6 is True

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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 3
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': [2, 4], 'b': 6})
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 3, 4})
    assert var_6 is True

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
    var_10 = bool(var_9 == ['1', ('2', '3'), {'a': '4'}])
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 9/15 statements.


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
    var_6 = [var_5]
    var_7 = [var_4, var_6]
    var_8 = [var_3, var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [[2, 3], [4, [5]]])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('1', '2', '3'))
    assert var_6 is True

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
    var_12 = bool(var_11 == {'a': [10, 20], 'b': {'c': 30}})
    assert var_12 is True

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
    var_8 = bool(var_7 == [2, 3, 4])
    assert var_8 is True

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



# Parsed testcases at query #3
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
    var_13 = bool(var_12 == [[[4], [6]]])
    assert var_13 is True

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
    var_11 = bool(var_10 == {'a': 9, 'b': 18})
    assert var_11 is True

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
    var_2 = [var_1]
    var_3 = (var_2,)
    var_4 = 2
    var_5 = [var_4]
    var_6 = (var_5,)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == [(3,)])
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
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_zip_dict_is_true. Retrieved 9/14 statements.


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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate_true. Retrieved 9/14 statements.


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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_map_structure_zip_triggers_line_17_via_list_instance. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #7
#--------------------------




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
    var_11 = bool(var_10 == {'a': 11, 'b': 22})
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 11/18 statements.


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
    var_9 = bool(var_8 == [[11, 22]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 6

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
    var_17 = bool(var_16 == [[[5, 12], [21, 32]]])
    assert var_17 is True

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
    var_8 = bool(var_7 == ((5,), (15,)))
    assert var_8 is True

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
    var_11 = bool(var_10 == {'a': 11, 'b': 22})
    assert var_11 is True

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
    var_8 = 10
    var_9 = 20
    var_10 = [var_8, var_9]
    var_11 = 30
    var_12 = {var_4: var_11}
    var_13 = (var_10, var_12)
    var_14 = [var_7, var_13]
    var_15 = module_0.map_structure_zip(var_0, var_14)
    var_16 = bool(var_15 == ([11, 22], {'a': 33}))
    assert var_16 is True

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
    var_3 = {var_1, var_2}
    var_4 = {var_1, var_2}
    var_5 = [var_3, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #9
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
    var_9 = bool(var_8 == [[4, 6]])
    assert var_9 is True

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
    var_13 = bool(var_12 == [[[3], [8]]])
    assert var_13 is True

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
    var_9 = bool(var_8 == [(1 - 3, 2 - 4)])
    assert var_9 is True



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
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True

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
    var_9 = bool(var_8 == [[2], [3, [4]]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('1', '2', '3'))
    assert var_6 is True

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
    var_10 = bool(var_9 == {'a': 10, 'b': [20, 30]})
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = {var_0, var_2}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = bool(var_4 == {2, 3})
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2: var_0}
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = [var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [({1: 4}, (6, 8))])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate_true. Retrieved 9/15 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_map_structure_zip_dict_is_true. Retrieved 9/15 statements.


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



# Parsed testcases at query #13
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
    var_10 = bool(var_9 == [2, [3, 4], [[5]]])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('1', '2', '3'))
    assert var_6 is True

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
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 3
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == {'a': [10, 20], 'b': 30})
    assert var_10 is True

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
    var_8 = bool(var_7 == [2, 3, 4])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = (var_1, var_6)
    var_8 = 'key'
    var_9 = 6
    var_10 = 7
    var_11 = (var_9, var_10)
    var_12 = {var_8: var_11}
    var_13 = [var_0, var_7, var_12]
    var_14 = 14
    var_15 = 15
    var_16 = {var_14, var_15}
    var_17 = [var_9, var_16]
    var_18 = (var_3, var_17)
    var_19 = 12
    var_20 = (var_19, var_14)
    var_21 = {var_8: var_20}
    var_22 = [var_1, var_18, var_21]
    var_23 = lambda x: x * var_1
    var_24 = (var_2,)
    var_25 = [var_1, var_24]
    var_26 = [var_0, var_25]
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = bool(var_27 == [2, [4, (6,)]])
    assert var_28 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 6



# Parsed testcases at query #14
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
    var_9 = bool(var_8 == [[2], [3, [4]]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('1', '2', '3'))
    assert var_6 is True

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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 3
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': [2, 4], 'b': 6})
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 3, 4})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2: var_0}
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = (var_3, var_6)
    var_8 = [var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [({1: 4}, [6, 8])])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #15
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True



# Parsed testcases at query #16
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
    var_9 = bool(var_8 == [[4, 6]])
    assert var_9 is True



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_map_structure_dict_predicate_true. Retrieved 6/7 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = lambda x: x.upper()
    var_4 = {var_0: var_1}
    var_5 = module_0.map_structure(var_3, var_4)
    var_6 = bool(var_5 == {'key': 'VALUE'})
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple_logic. Retrieved 11/18 statements.


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
    var_9 = bool(var_8 == [11, 22])
    assert var_9 is True

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
    var_13 = bool(var_12 == [[11], [22]])
    assert var_13 is True

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
    var_9 = bool(var_8 == ((3, 8),))
    assert var_9 is True

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
    var_11 = bool(var_10 == {'a': 9, 'b': 18})
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
    var_7 = [var_3, var_6]
    var_8 = 10
    var_9 = 20
    var_10 = (var_8, var_9)
    var_11 = 30
    var_12 = 40
    var_13 = (var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[(11, 22), (33, 44)]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 11

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
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)



# Parsed testcases at query #20
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 9/15 statements.


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
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = 4
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = [var_0, var_4, var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [2, [3, 4], [[5]]])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('1', '2', '3'))
    assert var_6 is True

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
    var_10 = bool(var_9 == {'a': 10, 'b': {'c': 20}})
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 3, 4})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = (var_1, var_5)
    var_7 = [var_0, var_6]
    var_8 = 6
    var_9 = 8
    var_10 = {var_9}
    var_11 = [var_8, var_10]
    var_12 = (var_3, var_11)
    var_13 = [var_1, var_12]
    var_14 = lambda x: x * var_1
    var_15 = [var_2]
    var_16 = (var_1, var_15)
    var_17 = [var_0, var_16]
    var_18 = module_0.map_structure(var_14, var_17)
    var_19 = bool(var_18 == [2, (4, [6])])
    assert var_19 is True

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
    var_6 = 3
    var_7 = lambda x: x * var_6
    var_8 = 6



# Parsed testcases at query #22
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
    var_10 = bool(var_9 == [2, [3, 4], [[5]]])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('1', '2', '3'))
    assert var_6 is True

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
    var_10 = bool(var_9 == {'a': 10, 'b': {'c': 20}})
    assert var_10 is True

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
    var_8 = bool(var_7 == [2, 3, 4])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = 'a'
    var_6 = 4
    var_7 = {var_5: var_6}
    var_8 = [var_1, var_4, var_7]
    var_9 = module_0.map_structure(var_0, var_8)
    var_10 = bool(var_9 == [1, (2, 3), {'a': 4}])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = (var_0, var_7)
    var_9 = [var_2, var_8]
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == [2, (4, [6, {4: 10}])])
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 6



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 11/18 statements.


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
    var_9 = bool(var_8 == [[11, 22]])
    assert var_9 is True

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
    var_13 = bool(var_12 == [[[11], [22]]])
    assert var_13 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 10
    var_5 = 20
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == [(10, 40)])
    assert var_9 is True

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
    var_11 = bool(var_10 == {'a': 11, 'b': 22})
    assert var_11 is True

import flutes.structure as module_0

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
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == {'a': [11, 22]})
    assert var_12 is True

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
    var_3 = {var_1, var_2}
    var_4 = {var_1, var_2}
    var_5 = [var_3, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    var_7 = 'Should have raised ValueError'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_map_structure_tuple_predicate_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x
    var_7 = '_fields'



# Parsed testcases at query #25
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_namedtuple_simulation. Retrieved 10/15 statements.


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
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = 4
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = [var_0, var_4, var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [2, [3, 4], [[5]]])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('1', '2', '3'))
    assert var_6 is True

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
    var_10 = bool(var_9 == {'a': 10, 'b': {'c': 20}})
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 3, 4})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = 'a'
    var_6 = 4
    var_7 = {var_5: var_6}
    var_8 = [var_1, var_4, var_7]
    var_9 = module_0.map_structure(var_0, var_8)
    var_10 = bool(var_9 == [1, (2, 3), {'a': 4}])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = [var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [[[2]]])
    assert var_7 is True

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
    var_1 = 'abc'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'ABC'



# Parsed testcases at query #2
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 10
    var_6 = 20
    var_7 = 30
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [11, 22, 33])
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
    var_17 = bool(var_16 == [[[5, 12], [21, 32]]])
    assert var_17 is True

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
    var_9 = bool(var_8 == (-9, -18))
    assert var_9 is True

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
    var_11 = bool(var_10 == {'a': 11, 'b': 22})
    assert var_11 is True

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
    var_11 = bool(var_10 == [('13', '24')])
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    var_7 = bool(var_6 == [3])
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
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(True)
    assert var_9 is True
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------




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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_no_type_check_predicate. Retrieved 1/6 statements.


def test_case_0():
    var_0 = '__wrapped__'



# Parsed testcases at query #5
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
    var_9 = bool(var_8 == [[11, 22]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 6

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
    var_17 = bool(var_16 == [[[5, 12], [21, 32]]])
    assert var_17 is True

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
    var_8 = bool(var_7 == ((5, 15),))
    assert var_8 is True

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
    var_11 = bool(var_10 == {'a': 11, 'b': 22})
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
    var_7 = [var_3, var_6]
    var_8 = 10
    var_9 = 20
    var_10 = (var_8, var_9)
    var_11 = 30
    var_12 = 40
    var_13 = (var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[(11, 22), (33, 44)]])
    assert var_17 is True

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
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 11/18 statements.


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
    var_9 = bool(var_8 == [11, 22])
    assert var_9 is True

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
    var_13 = bool(var_12 == [[[3], [8]]])
    assert var_13 is True

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
    var_8 = bool(var_7 == (5, 15))
    assert var_8 is True

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
    var_11 = bool(var_10 == {'a': 11, 'b': 22})
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
    var_7 = [var_3, var_6]
    var_8 = 10
    var_9 = 20
    var_10 = (var_8, var_9)
    var_11 = 30
    var_12 = 40
    var_13 = (var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[(11, 22), (33, 44)]])
    assert var_17 is True

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
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 10
    var_8 = 20
    var_9 = 11
    var_10 = 22

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = {var_1, var_2}
    var_5 = [var_3, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #7
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
    var_10 = bool(var_9 == [2, [3, 4], [[5]]])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('1', '2', '3'))
    assert var_6 is True

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
    var_10 = bool(var_9 == {'a': 10, 'b': [20, 30]})
    assert var_10 is True

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
    var_8 = bool(var_7 == [2, 3, 4])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 'key'
    var_5 = 4
    var_6 = 5
    var_7 = [var_5, var_6]
    var_8 = {var_4: var_7}
    var_9 = [var_0, var_3, var_8]
    var_10 = 6
    var_11 = (var_5, var_10)
    var_12 = 8
    var_13 = 10
    var_14 = [var_12, var_13]
    var_15 = {var_4: var_14}
    var_16 = [var_1, var_11, var_15]
    var_17 = lambda x: x * var_1
    var_18 = module_0.map_structure(var_17, var_9)
    var_19 = bool(var_18 == var_16)
    assert var_19 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 10/17 statements.


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
    var_11 = bool(var_10 == {'a': 5, 'b': 18})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: str(x) + str(y)
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
    var_16 = bool(var_15 == [(['14', '25'], {'a': '36'})])
    assert var_16 is True

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
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3

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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_map_structure_zip_list_branch. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate_true. Retrieved 9/15 statements.


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



# Parsed testcases at query #11
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_map_structure_zip_dict_structure_is_true. Retrieved 9/14 statements.
# Partially parsed test_map_structure_zip_ordered_dict_structure_is_true. Retrieved 17/26 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 10
    var_8 = (var_0, var_7)
    var_9 = 20
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = 11
    var_13 = (var_0, var_12)
    var_14 = 22
    var_15 = (var_3, var_14)
    var_16 = [var_13, var_15]



# Parsed testcases at query #13
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 3, 4])
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_map_structure_no_type_check_predicate. Retrieved 4/7 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = '_no_map'
    var_1 = lambda x: x
    var_2 = 1
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_map_structure_zip_dict_evaluates_true_at_line_27. Retrieved 9/14 statements.
# Partially parsed test_map_structure_zip_ordered_dict_evaluates_true_at_line_27. Retrieved 12/22 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 10
    var_8 = (var_0, var_7)
    var_9 = 20
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 11/18 statements.


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
    var_9 = bool(var_8 == [[11, 22]])
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
    var_17 = bool(var_16 == [[[5, 12], [21, 32]]])
    assert var_17 is True

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
    var_9 = bool(var_8 == (-9, -18))
    assert var_9 is True

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
    var_11 = bool(var_10 == {'a': 11, 'b': 22})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = (var_2, var_4)
    var_6 = 10
    var_7 = [var_6]
    var_8 = 20
    var_9 = [var_8]
    var_10 = (var_7, var_9)
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)
    var_13 = bool(var_12 == ([11], [22]))
    assert var_13 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 11

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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_map_structure_zip_list_predicate_true. Retrieved 12/34 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = '_NO_MAP_TYPES'
    var_1 = '_NO_MAP_INSTANCE_ATTR'
    var_2 = '_no_map_attr'
    var_3 = lambda x, y: x + y
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = [var_6, var_9]
    var_11 = module_0.map_structure_zip(var_3, var_10)
    var_12 = bool(var_11 == [4, 6])
    assert var_12 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_map_structure_zip_dict_is_true. Retrieved 9/15 statements.


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



# Parsed testcases at query #19
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True



# Parsed testcases at query #20
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

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = 4
    var_6 = [var_0, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, [3, 4], 5])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('1', '2', '3'))
    assert var_6 is True

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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_4, var_0]
    var_6 = 3
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': [2, 4], 'b': 6})
    assert var_9 is True

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
    var_8 = bool(var_7 == [2, 3, 4])
    assert var_8 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = {var_4}
    var_6 = [var_3, var_5]
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [2, (4, [6, {8}])])
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_map_structure_zip_evaluates_true_at_line_19. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



