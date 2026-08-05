####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_0, var_7)
    var_9 = (var_1, var_2)
    var_10 = (var_4, var_5)
    var_11 = [var_9, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)
    var_13 = bool(var_12 == (3, 8))
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
    var_0 = lambda x, y: str(x) + str(y)
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == ['13', '24'])
    assert var_9 is True

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
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_deeply_nested. Retrieved 19/22 statements.


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
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = list(var_5)
    var_7 = sorted(var_6)
    var_8 = bool(var_7 == [2, 3, 4])
    assert var_8 is True

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
    var_16 = [var_2]
    var_17 = (var_1, var_16)
    var_18 = {var_0: var_17}

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #3
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
    var_7 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_7 is True



# Parsed testcases at query #4
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
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [2, 4], 'b': {'c': 6}})
    assert var_11 is True

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5
    var_6 = {var_0: var_5, var_1: var_5}
    var_7 = [var_4, var_6]



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
    var_18 = bool(var_17 == [[9], [12]])
    assert var_18 is True

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
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = (var_3, var_6)
    var_8 = 10
    var_9 = 20
    var_10 = [var_8, var_9]
    var_11 = 4
    var_12 = {var_4: var_11}
    var_13 = (var_10, var_12)
    var_14 = [var_7, var_13]
    var_15 = module_0.map_structure_zip(var_0, var_14)
    var_16 = bool(var_15 == [[11, 22], {'a': 7}])
    assert var_16 is True

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
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)



# Parsed testcases at query #7
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
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == (5, 18))
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
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = 5
    var_9 = 6
    var_10 = [var_8, var_9]
    var_11 = 7
    var_12 = 8
    var_13 = (var_11, var_12)
    var_14 = (var_10, var_13)
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[6, 12], (10, 12)])
    assert var_17 is True

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
    var_0 = lambda x: x
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)

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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_tuple_int_to_str. Retrieved 5/6 statements.


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

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_0, var_2, var_5]
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == [2, 3, [4, 5]])
    assert var_8 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_2,)
    var_4 = (var_0, var_1, var_3)

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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = {var_0, var_2}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = bool(var_4 == {2, 3})
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = (var_1, var_5)
    var_7 = 5
    var_8 = [var_0, var_6, var_7]
    var_9 = {var_7}
    var_10 = [var_3, var_9]
    var_11 = (var_2, var_10)
    var_12 = 6
    var_13 = [var_1, var_11, var_12]
    var_14 = lambda x: x + var_0
    var_15 = module_0.map_structure(var_14, var_8)
    var_16 = bool(var_15 == var_13)
    assert var_16 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 50



# Parsed testcases at query #9
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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2, var_0}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = bool(var_4 == {2, 4})
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = {var_7, var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = 6
    var_12 = (var_7, var_11)
    var_13 = [var_3, var_12]
    var_14 = 8
    var_15 = 10
    var_16 = {var_14, var_15}
    var_17 = {var_0: var_13, var_1: var_16}
    var_18 = lambda x: x * var_3
    var_19 = module_0.map_structure(var_18, var_10)
    var_20 = var_19['a']
    var_21 = bool(var_19['a'] == [2, (4, 6)])
    assert var_21 is True
    var_22 = var_19['b']
    var_23 = bool(var_19['b'] == {8, 10})
    assert var_23 is True

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
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_map_structure_zip_dict_true. Retrieved 12/17 statements.


import flutes.structure as module_0

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
    var_9 = 0
    var_10 = lambda *args: args[var_9]
    var_11 = module_0.map_structure_zip(var_10, var_8)
    var_12 = var_11['a']
    assert var_12 == 1
    var_13 = var_11['b']
    assert var_13 == 2



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
    var_5 = 'c'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == {'a': 10, 'b': {'c': 20}})
    assert var_10 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2, var_0}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = bool(var_4 == {2, 4})
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'abc'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'ABC'



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_map_structure_zip_predicate_true. Retrieved 9/19 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = '_no_map'
    var_1 = '_NO_MAP_TYPES'
    var_2 = '_NO_MAP_INSTANCE_ATTR'
    var_3 = '_no_map'
    var_4 = lambda x: x
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = module_0.map_structure_zip(var_4, var_7)
    assert var_8 == 1



# Parsed testcases at query #15
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



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'abc'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'ABC'

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 3
    var_4 = {var_2: var_3}
    var_5 = (var_1, var_4)
    var_6 = 4
    var_7 = 5
    var_8 = {var_7}
    var_9 = [var_6, var_8]
    var_10 = [var_0, var_5, var_9]
    var_11 = 6
    var_12 = {var_2: var_11}
    var_13 = (var_6, var_12)
    var_14 = 8
    var_15 = 10
    var_16 = {var_15}
    var_17 = [var_14, var_16]
    var_18 = [var_1, var_13, var_17]
    var_19 = lambda x: x * var_1
    var_20 = module_0.map_structure(var_19, var_10)
    var_21 = bool(var_20 == var_18)
    assert var_21 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_map_structure_zip_dict_evaluates_true. Retrieved 11/17 statements.


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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_map_structure_zip_no_type_check. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #21
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = '2'
    var_3 = 3.0
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ['1', '2', '3.0'])
    assert var_6 is True



# Parsed testcases at query #22
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
    var_7 = bool(var_6 == [[2], [3]])
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_map_structure_zip_simple_lists. Retrieved 8/9 statements.
# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]

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
    var_4 = [var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [[(4, 6)]])
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
    var_0 = lambda x: x
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)

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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_map_structure_predicate_true_for_tuple. Retrieved 6/8 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = lambda x: x
    var_5 = module_0.map_structure(var_4, var_3)



# Parsed testcases at query #25
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda *args: sum(args)
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



# Parsed testcases at query #26
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
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {6, 7})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 11

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



# Parsed testcases at query #27
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
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 3, 4})
    assert var_6 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = {var_5, var_6}
    var_8 = [var_1, var_4, var_7]
    var_9 = {var_0: var_8}
    var_10 = (var_3, var_5)
    var_11 = 6
    var_12 = {var_6, var_11}
    var_13 = [var_2, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = lambda x: x + var_1
    var_16 = module_0.map_structure(var_15, var_9)
    var_17 = var_16['key'][0]
    assert var_17 == 2
    var_18 = var_16['key'][1]
    var_19 = bool(var_16['key'][1] == (3, 4))
    assert var_19 is True
    var_20 = var_16['key'][2]
    var_21 = bool(var_16['key'][2] == {5, 6})
    assert var_21 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #28
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
    var_4 = (var_3,)
    var_5 = (var_1, var_2, var_4)
    var_6 = module_0.map_structure(var_0, var_5)
    var_7 = bool(var_6 == ('1', '2', ('3',)))
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
    var_3 = {var_0, var_2}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = bool(var_4 == {2, 3})
    assert var_5 is True

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



# Parsed testcases at query #29
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ['1', '2', '3'])
    assert var_6 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_map_structure_zip_simple_integers. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

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
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = 10
    var_7 = 20
    var_8 = [var_6, var_7]
    var_9 = 4
    var_10 = (var_8, var_9)
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)
    var_13 = bool(var_12 == [(11, 13), 7])
    assert var_13 is True

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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == [6])
    assert var_6 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_map_structure_zip_evaluates_true_at_line_19. Retrieved 9/16 statements.


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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_map_structure_zip_no_type_check_is_false. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 15/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = lambda x: x.upper()
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 1
    var_7 = 2
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.map_structure(var_3, var_8)
    var_10 = bool(var_9 == {'a': 1, 'b': 2})
    assert var_10 is True
    var_11 = lambda x: x
    var_12 = 'test'
    var_13 = [var_6, var_7]
    var_14 = {var_12: var_13}
    var_15 = module_0.map_structure(var_11, var_14)



# Parsed testcases at query #34
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



# Parsed testcases at query #35
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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2, var_0}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = bool(var_4 == {2, 4})
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = 'inner'
    var_6 = 4
    var_7 = {var_5: var_6}
    var_8 = [var_1, var_4, var_7]
    var_9 = {var_0: var_8}
    var_10 = 6
    var_11 = (var_6, var_10)
    var_12 = 8
    var_13 = {var_5: var_12}
    var_14 = [var_2, var_11, var_13]
    var_15 = {var_0: var_14}
    var_16 = lambda x: x * var_2
    var_17 = module_0.map_structure(var_16, var_9)
    var_18 = bool(var_17 == var_15)
    assert var_18 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_map_structure_dict_predicate_true. Retrieved 10/11 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = lambda x: x.upper()
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 1
    var_7 = 2
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.map_structure(var_3, var_8)
    var_10 = bool(var_9 == {'a': 1, 'b': 2})
    assert var_10 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_map_structure_zip_tuple_branch. Retrieved 13/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = 3
    var_9 = 4
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = module_0.map_structure_zip(var_4, var_11)
    var_13 = var_12.x
    assert var_13 == 4
    var_14 = var_12.y
    assert var_14 == 6



# Parsed testcases at query #38
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
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == (5, 18))
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
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = 5
    var_9 = 6
    var_10 = [var_8, var_9]
    var_11 = 7
    var_12 = 8
    var_13 = (var_11, var_12)
    var_14 = (var_10, var_13)
    var_15 = [var_7, var_14]
    var_16 = [var_1, var_2]
    var_17 = [var_4, var_5]
    var_18 = [var_16, var_17]
    var_19 = module_0.map_structure_zip(var_0, var_18)
    var_20 = bool(var_19 == ['13', '24'])
    assert var_20 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = [var_1, var_2]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 3

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

# Partially parsed test_map_structure_zip_tuple_branch. Retrieved 13/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = 3
    var_9 = 4
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = module_0.map_structure_zip(var_4, var_11)
    var_13 = bool(var_12 == (4, 6))
    assert var_13 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_map_structure_list_predicate_true. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 8/13 statements.


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
    var_11 = bool(var_10 == {'a': [2, 3], 'b': {'c': 4}})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2, var_0}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = bool(var_4 == {2, 4})
    assert var_5 is True

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
    var_15 = bool(var_14 == var_12)
    assert var_15 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x + var_4
    var_7 = 3



# Parsed testcases at query #42
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
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [2, 4], 'b': {'c': 6}})
    assert var_11 is True

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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple_logic. Retrieved 10/17 statements.


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
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = (var_4,)
    var_6 = (var_3, var_5)
    var_7 = 4
    var_8 = 5
    var_9 = [var_7, var_8]
    var_10 = 6
    var_11 = (var_10,)
    var_12 = (var_9, var_11)
    var_13 = [var_6, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == ([5, 7], (9,)))
    assert var_15 is True

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
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #44
#--------------------------




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
    var_13 = bool(var_12 == [[4], [6]])
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
    var_4 = [var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [[(4, 6)]])
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 10
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == [20])
    assert var_6 is True

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
    var_9 = bool(True)
    assert var_9 is True
    var_10 = bool(False)
    assert var_10 is True



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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 'a'
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_zip_dict. Retrieved 10/11 statements.
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
    var_8 = bool(var_7 == (5, 15))
    assert var_8 is True

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
    var_10 = 10
    var_11 = 20
    var_12 = [var_10, var_11]
    var_13 = 30
    var_14 = 40
    var_15 = (var_13, var_14)
    var_16 = {var_1: var_12, var_2: var_15}
    var_17 = [var_9, var_16]
    var_18 = module_0.map_structure_zip(var_0, var_17)
    var_19 = var_18['a']
    var_20 = bool(var_18['a'] == [11, 22])
    assert var_20 is True
    var_21 = var_18['b']
    var_22 = bool(var_18['b'] == (33, 44))
    assert var_22 is True

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

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
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
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_map_structure_zip_skips_no_map_instance_attr. Retrieved 2/10 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = [var_2]
    var_6 = [var_0]
    var_7 = [var_5, var_6]
    var_8 = module_0.map_structure_zip(var_1, var_7)
    var_9 = bool(var_8 == [2])
    assert var_9 is True

def test_case_0():
    var_0 = lambda x: x
    var_1 = 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_zip_dict_branch. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_ordered_dict_branch. Retrieved 17/25 statements.


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
    var_0 = 'x'
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = (var_0, var_7)
    var_9 = 5
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = 8
    var_13 = (var_0, var_12)
    var_14 = 15
    var_15 = (var_3, var_14)
    var_16 = [var_13, var_15]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_map_structure_zip_tuple_branch. Retrieved 13/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = 3
    var_9 = 4
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = module_0.map_structure_zip(var_4, var_11)
    var_13 = var_12.x
    assert var_13 == 4
    var_14 = var_12.y
    assert var_14 == 6



# Parsed testcases at query #6
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
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [2, 4], 'b': {'c': 6}})
    assert var_11 is True

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
    var_15 = bool(var_14 == var_12)
    assert var_15 is True



# Parsed testcases at query #7
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
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = (var_3, var_6)
    var_8 = 10
    var_9 = 20
    var_10 = [var_8, var_9]
    var_11 = 40
    var_12 = {var_4: var_11}
    var_13 = (var_10, var_12)
    var_14 = [var_7, var_13]
    var_15 = module_0.map_structure_zip(var_0, var_14)
    var_16 = bool(var_15 == [([11, 22], {'a': 43})])
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
    var_7 = bool(var_6 == [[2, 4, 6]])
    assert var_7 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_deeply_nested. Retrieved 17/22 statements.
# Partially parsed test_map_structure_namedtuple. Retrieved 9/14 statements.


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
    var_9 = 'one'
    var_10 = 'two'
    var_11 = (var_9, var_10)
    var_12 = 'three'
    var_13 = 'four'
    var_14 = {var_12, var_13}
    var_15 = [var_11, var_14]
    var_16 = {var_0: var_15}

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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_dict_predicate_is_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 'a'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}



# Parsed testcases at query #3
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
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 10
    var_5 = 20
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == [(11, 22)])
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
    var_3 = 'a'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = (var_2, var_5)
    var_7 = 10
    var_8 = [var_7]
    var_9 = 20
    var_10 = {var_3: var_9}
    var_11 = (var_8, var_10)
    var_12 = [var_6, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == [([11], {'a': 22})])
    assert var_14 is True

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
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)



# Parsed testcases at query #4
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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2, var_0}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = bool(var_4 == {2, 4})
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'c'
    var_8 = 4
    var_9 = {var_7: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = 6
    var_12 = (var_8, var_11)
    var_13 = [var_3, var_12]
    var_14 = 8
    var_15 = {var_7: var_14}
    var_16 = {var_0: var_13, var_1: var_15}
    var_17 = lambda x: x * var_3
    var_18 = module_0.map_structure(var_17, var_10)
    var_19 = bool(var_18 == var_16)
    assert var_19 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_map_structure_zip_dict_true. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = 20
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]



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
    var_11 = bool(var_10 == {'a': 5.0, 'b': 5.0})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: f'{x}-{y}'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = (var_4,)
    var_6 = 'b'
    var_7 = (var_6,)
    var_8 = [var_5, var_7]
    var_9 = [var_3, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [['1-(', '2-[']])
    assert var_11 is True
    var_12 = (var_1, var_2)
    var_13 = 3
    var_14 = 4
    var_15 = (var_13, var_14)
    var_16 = [var_12, var_15]
    var_17 = 10
    var_18 = 20
    var_19 = (var_17, var_18)
    var_20 = 30
    var_21 = 40
    var_22 = (var_20, var_21)
    var_23 = [var_19, var_22]
    var_24 = [var_16, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = bool(var_26 == [[(11, 22), (33, 44)]])
    assert var_27 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = {var_1, var_2}
    var_5 = [var_3, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    var_7 = bool(False)
    assert var_7 is True

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



# Parsed testcases at query #7
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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2, var_0}
    var_4 = module_0.map_structure(var_1, var_3)
    var_5 = bool(var_4 == {2, 4})
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'c'
    var_8 = 4
    var_9 = {var_7: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = 6
    var_12 = (var_8, var_11)
    var_13 = [var_3, var_12]
    var_14 = 8
    var_15 = {var_7: var_14}
    var_16 = {var_0: var_13, var_1: var_15}
    var_17 = lambda x: x * var_3
    var_18 = module_0.map_structure(var_17, var_10)
    var_19 = bool(var_18 == var_16)
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



# Parsed testcases at query #8
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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = [var_0, var_4]
    var_6 = 3
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == {'a': [2, 3], 'b': 4})
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {3, 6})
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
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = (var_0, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = [var_4, var_7]
    var_9 = '2'
    var_10 = '4'
    var_11 = {var_1: var_10}
    var_12 = (var_9, var_11)
    var_13 = {var_5, var_6}
    var_14 = [var_12, var_13]
    var_15 = lambda x: x
    var_16 = {var_1: var_2}
    var_17 = (var_0, var_16)
    var_18 = {var_5, var_6}
    var_19 = [var_17, var_18]
    var_20 = module_0.map_structure(var_15, var_19)
    var_21 = bool(var_20 == [(1, {'a': 2}), {3, 4}])
    assert var_21 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'abc'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'ABC'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_map_structure_tuple_predicate. Retrieved 13/20 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 'Point'
    var_5 = 'x'
    var_6 = 'y'
    var_7 = [var_5, var_6]
    var_8 = 10
    var_9 = 20
    var_10 = lambda x: x
    var_11 = module_0.map_structure(var_10, var_3)
    var_12 = bool(var_11 == (1, 2, 3))
    assert var_12 is True
    var_13 = lambda x: x



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_map_structure_dict_predicate_true. Retrieved 7/8 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = module_0.map_structure(var_5, var_4)



# Parsed testcases at query #11
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = 4
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = module_0.map_structure(var_5, var_4)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True



# Parsed testcases at query #12
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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = {var_2, var_3}
    var_5 = (var_1, var_4)
    var_6 = 'key'
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = {var_6: var_9}
    var_11 = [var_0, var_5, var_10]
    var_12 = 8
    var_13 = {var_8, var_12}
    var_14 = (var_3, var_13)
    var_15 = 10
    var_16 = 12
    var_17 = [var_15, var_16]
    var_18 = {var_6: var_17}
    var_19 = [var_1, var_14, var_18]
    var_20 = lambda x: x * var_1
    var_21 = module_0.map_structure(var_20, var_11)
    var_22 = bool(var_21 == var_19)
    assert var_22 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 6



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_map_structure_no_type_check_predicate. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



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
    var_6 = lambda x: x
    var_7 = '_fields'



# Parsed testcases at query #16
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
    var_7 = lambda x, y: x + y
    var_8 = [var_1, var_2]
    var_9 = [var_3, var_4]
    var_10 = [var_8, var_9]
    var_11 = module_0.map_structure_zip(var_7, var_10)
    var_12 = bool(var_11 == [[4, 6]])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 10
    var_5 = 20
    var_6 = [var_4, var_5]
    var_7 = 100
    var_8 = 200
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [[111, 222]])
    assert var_12 is True

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
    var_9 = bool(var_8 == [(3, 8)])
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_map_structure_zip_evaluates_true_at_line_19. Retrieved 10/11 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda *args: args[var_0]
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_0.map_structure_zip(var_1, var_8)
    var_10 = bool(var_9 == (1, 3))
    assert var_10 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_map_structure_evaluates_tuple_predicate. Retrieved 6/7 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = lambda x: x
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = bool(var_5 == (1, 2, 3))
    assert var_6 is True



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate_true. Retrieved 15/24 statements.


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
    var_9 = 'val'
    var_10 = {var_9: var_2}
    var_11 = {var_0: var_10}
    var_12 = {var_9: var_3}
    var_13 = {var_0: var_12}
    var_14 = [var_11, var_13]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_map_structure_evaluates_list_predicate_true. Retrieved 6/7 statements.


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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_map_structure_evaluates_tuple_predicate. Retrieved 10/16 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 'TestTuple'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = lambda x: x
    var_9 = module_0.map_structure(var_8, var_3)
    var_10 = bool(var_9 == (1, 2, 3))
    assert var_10 is True



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_map_structure_zip_flat_list. Retrieved 13/18 statements.
# Partially parsed test_map_structure_zip_nested_lists. Retrieved 11/14 statements.
# Partially parsed test_map_structure_zip_tuples. Retrieved 7/10 statements.
# Partially parsed test_map_structure_zip_namedtuple. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_mixed_structures. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_set_raises_error. Retrieved 7/12 statements.
# Partially parsed test_map_structure_zip_single_element_leaf. Retrieved 3/6 statements.


import typing as module_0

def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = module_0.TypeVar(var_0, *var_1)
    var_3 = 'R'
    var_4 = []
    var_5 = module_0.TypeVar(var_3, *var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = 4
    var_11 = 5
    var_12 = 6
    var_13 = [var_10, var_11, var_12]
    var_14 = [var_9, var_13]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = 3
    var_6 = [var_5]
    var_7 = 4
    var_8 = [var_7]
    var_9 = [var_6, var_8]
    var_10 = [var_4, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'hello'
    var_3 = 'foo'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = ' '
    var_6 = 'bar'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]



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



# Parsed testcases at query #26
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
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == [(4, 6)])
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
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = lambda x, y: x + y
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



# Parsed testcases at query #27
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



# Parsed testcases at query #28
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
    var_9 = bool(var_8 == [11, 22])
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
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = (var_4, var_4)
    var_6 = [var_3, var_5]
    var_7 = module_0.map_structure_zip(var_0, var_6)
    var_8 = bool(var_7 == (5, 15))
    assert var_8 is True

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



# Parsed testcases at query #29
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 8/14 statements.


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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = [var_2, var_0]
    var_4 = 3
    var_5 = 4
    var_6 = [var_5]
    var_7 = [var_4, var_6]
    var_8 = [var_3, var_7]
    var_9 = module_0.map_structure(var_1, var_8)
    var_10 = bool(var_9 == [[2, 4], [6, [8]]])
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
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)
    var_8 = bool(var_7 == {'a': 10, 'b': 20})
    assert var_8 is True

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
    var_11 = bool(var_10 == {'a': [2, 3], 'b': {'c': 4}})
    assert var_11 is True

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
    var_8 = bool(var_7 == [2, 4, 6])
    assert var_8 is True

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
    var_8 = '4'
    var_9 = 6
    var_10 = 8
    var_11 = {var_10}
    var_12 = [var_9, var_11]
    var_13 = (var_8, var_12)
    var_14 = [var_1, var_13]
    var_15 = lambda x: x * var_1
    var_16 = {var_3}
    var_17 = [var_2, var_16]
    var_18 = (var_1, var_17)
    var_19 = [var_0, var_18]
    var_20 = module_0.map_structure(var_15, var_19)
    var_21 = bool(var_20 == [2, (4, [6, {8}])])
    assert var_21 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'abc'
    var_2 = 'def'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure(var_0, var_3)
    var_5 = bool(var_4 == ['ABC', 'DEF'])
    assert var_5 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x + var_4
    var_7 = 3



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/16 statements.


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
    var_13 = bool(var_12 == [[4], [6]])
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
    var_4 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_map_structure_no_type_check_predicate. Retrieved 1/10 statements.


def test_case_0():
    var_0 = '__call__'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_map_structure_mixed_types. Retrieved 13/15 statements.


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
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [2, 4], 'b': {'c': 6}})
    assert var_11 is True

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



# Parsed testcases at query #34
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
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 5
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 5

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
    var_9 = bool(var_8 == [[4, 6]])
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
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_map_structure_zip_tuple_branch. Retrieved 14/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = 3
    var_9 = 4
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = module_0.map_structure_zip(var_4, var_11)
    var_13 = 6



# Parsed testcases at query #36
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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate. Retrieved 12/20 statements.


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
    var_9 = {var_0: var_2, var_1: var_3}
    var_10 = {var_0: var_5, var_1: var_6}
    var_11 = [var_9, var_10]



# Parsed testcases at query #38
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 1



# Parsed testcases at query #39
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_map_structure_no_type_check_predicate. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #41
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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_map_structure_zip_no_type_check_is_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '__name__'
    var_1 = False



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
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == [3, 8])
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

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_9 = 6

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
    var_9 = bool(var_8 == [[4, 6]])
    assert var_9 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = {var_1}
    var_3 = 2
    var_4 = {var_3}
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)



# Parsed testcases at query #44
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



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_map_structure_dict_predicate_true. Retrieved 7/8 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = module_0.map_structure(var_5, var_4)



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_map_structure_zip_no_type_check_decorator_is_present. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #49
#--------------------------

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
    var_13 = bool(var_12 == [[4], [6]])
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
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_6]
    var_8 = [var_5, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [[4], [6]])
    assert var_11 is True

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
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

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



