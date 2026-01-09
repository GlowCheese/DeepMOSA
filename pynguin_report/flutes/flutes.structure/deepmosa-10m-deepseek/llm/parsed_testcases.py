####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
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
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True


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
    var_6 = lambda x: x * var_5
    var_7 = 4


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
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True

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


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10


def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'


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


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = []
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True
    var_5 = lambda x: x * var_0
    var_6 = {}
    var_7 = module_0.map_structure(var_5, var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True
    var_9 = lambda x: x * var_0
    var_10 = set()
    var_11 = module_0.map_structure(var_9, var_10)
    var_12 = set()
    var_13 = bool(var_11 == var_12)
    assert var_13 is True


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



# Parsed testcases at query #2
#--------------------------





def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_no_map_instance_attr. Retrieved 2/8 statements.
# Partially parsed test_map_structure_zip_ordered_dict. Retrieved 17/23 statements.



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


def test_case_0():
    var_0 = lambda x, y: x - y
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
    var_17 = -4
    var_18 = -4
    var_19 = [var_17, var_18]
    var_20 = -4
    var_21 = -4
    var_22 = [var_20, var_21]
    var_23 = [var_19, var_22]
    var_24 = bool(var_16 == var_23)
    assert var_24 is True


def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = (var_5, var_6, var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = 10
    var_12 = 18
    var_13 = (var_5, var_11, var_12)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

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
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 2
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 3
    var_9 = {var_3: var_8}
    var_10 = 4
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = [var_7, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = {var_3: var_10}
    var_15 = 6
    var_16 = {var_1: var_14, var_2: var_15}
    var_17 = bool(var_13 == var_16)
    assert var_17 is True


def test_case_0():
    var_0 = lambda x, y: str(x) + str(y)
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
    var_19 = '15'
    var_20 = '26'
    var_21 = [var_19, var_20]
    var_22 = '37'
    var_23 = '48'
    var_24 = (var_22, var_23)
    var_25 = {var_1: var_21, var_2: var_24}
    var_26 = bool(var_18 == var_25)
    assert var_26 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = 15
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: (x, y)

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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 7
    var_2 = [var_1]
    var_3 = 8
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    var_7 = 56
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_no_map_instance_attr. Retrieved 3/9 statements.



def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10


def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 3, 4])
    assert var_6 is True


def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_0, var_7)
    var_9 = bool(var_8 == [['A', 'B'], ['C', 'D']])
    assert var_9 is True


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
    var_6 = 10
    var_7 = lambda x: x * var_6
    var_8 = 20


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
    var_7 = 3
    var_8 = lambda x: x * var_7


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True


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
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'


def test_case_0():
    var_0 = b'!'
    var_1 = lambda x: x + var_0
    var_2 = b'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == b'hello!'

def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = lambda x: x.value * var_1


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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 5/42 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = ()
    var_5 = '_no_map'
    var_6 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordereddict. Retrieved 13/20 statements.



def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = [var_0]
    var_3 = [var_2]
    var_4 = module_0.map_structure_zip(var_1, var_3)
    var_5 = bool(var_4 == [2])
    assert var_5 is True


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'ab'
    var_2 = 'cd'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'abcd'


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 10
    var_2 = 20
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 30


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


def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = (var_2,)
    var_4 = [var_1, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_6,)
    var_8 = [var_5, var_7]
    var_9 = 5
    var_10 = 6
    var_11 = (var_10,)
    var_12 = [var_9, var_11]
    var_13 = [var_4, var_8, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [9, (12,)])
    assert var_15 is True



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/20 statements.
# Partially parsed test_map_structure_zip_with_ordereddict. Retrieved 14/24 statements.
# Partially parsed test_map_structure_zip_with_custom_no_map_instance. Retrieved 2/9 statements.



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
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = lambda u, v: u + v
    var_9 = 6


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = [var_5]
    var_7 = 3
    var_8 = 4
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = [var_9]
    var_11 = (var_6, var_10)
    var_12 = module_0.map_structure_zip(var_0, var_11)
    var_13 = bool(var_12 == {'a': 4, 'b': 6})
    assert var_13 is True

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 30
    var_8 = (var_0, var_7)
    var_9 = 40
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = lambda a, b: a * b
    var_13 = 0


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'helloworld'


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = 3
    var_6 = [var_5]
    var_7 = [var_4, var_6]
    var_8 = 4
    var_9 = 5
    var_10 = [var_9]
    var_11 = [var_8, var_10]
    var_12 = 6
    var_13 = [var_12]
    var_14 = [var_11, var_13]
    var_15 = (var_7, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[5, [7]], [9]])
    assert var_17 is True


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = {}
    var_2 = [var_1]
    var_3 = {}
    var_4 = [var_3]
    var_5 = (var_2, var_4)
    var_6 = module_0.map_structure_zip(var_0, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = [var_3]
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = [var_7]
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'cannot contain `set`'

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: (x, y)


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'k'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = [var_5]
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = {var_1: var_9}
    var_11 = [var_10]
    var_12 = (var_6, var_11)
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == {'k': [4, 6]})
    assert var_14 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true. Retrieved 16/45 statements.


def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = lambda x, y: x + y
    var_8 = 'x'
    var_9 = 10
    var_10 = (var_8, var_9)
    var_11 = 'y'
    var_12 = 20
    var_13 = (var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = lambda a, b: a + b



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true_for_list. Retrieved 11/28 statements.



def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = (var_4, var_7)
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_map_structure_with_flat_tuple. Retrieved 4/5 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/6 statements.



def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True


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
    var_6 = lambda x: x * var_5
    var_7 = 4


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


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True


def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = {var_0, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = bool(var_8 == [{2, 3}, {4, 5}])
    assert var_9 is True


def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 11/21 statements.
# Partially parsed test_map_structure_zip_with_custom_no_map_instance. Retrieved 2/7 statements.



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
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = lambda a, b: a + b
    var_9 = 6
    var_10 = 8


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
    var_9 = lambda x, y: x * y
    var_10 = (var_8, var_8)
    var_11 = module_0.map_structure_zip(var_9, var_10)
    var_12 = bool(var_11 == {'a': [1, 4], 'b': (9, 16)})
    assert var_12 is True


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'helloworld'

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: (x, y)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 13/20 statements.



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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15


def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 'hello'
    var_2 = ' world'
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'hello world'


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
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


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


def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 3
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = (var_5, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': {'b': 7}})
    assert var_11 is True

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = lambda a, b: a * b



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 10/15 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_types. Retrieved 3/6 statements.



def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True


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
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True


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


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 4, 'b': 9})
    assert var_7 is True


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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

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


def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'


def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = 'c'
    var_7 = 4
    var_8 = {var_6: var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_4, var_1: var_9}
    var_11 = lambda x: x * var_3
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == {'a': (2, 4), 'b': [6, {'c': 8}]})
    assert var_13 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = []
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = {}
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = set()
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = set()
    var_5 = bool(var_3 == var_4)
    assert var_5 is True


def test_case_0():
    var_0 = None
    var_1 = 'mapped'
    var_2 = lambda x: var_1 if x is var_0 else x
    var_3 = module_0.map_structure(var_2, var_0)
    assert var_3 == 'mapped'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_true. Retrieved 19/44 statements.



def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '_no_map'
    var_5 = True
    var_6 = 'hello'
    var_7 = b'world'
    var_8 = b'test'
    var_9 = bytearray(var_8)
    var_10 = lambda x: x.upper()
    var_11 = module_0.map_structure(var_10, var_6)
    assert var_11 == 'HELLO'
    var_12 = lambda x: x.upper()
    var_13 = module_0.map_structure(var_12, var_7)
    assert var_13 == b'WORLD'
    var_14 = lambda x: x.upper()
    var_15 = module_0.map_structure(var_14, var_9)
    var_16 = 'mapped'
    var_17 = lambda x: var_16
    var_18 = lambda x: var_16
    var_19 = b'TEST'
    var_20 = bytearray(var_19)
    var_21 = bool(var_15 == var_20)
    assert var_21 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_true. Retrieved 5/30 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '_no_map'
    var_5 = True
    var_6 = '_no_map'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_map_structure_zip_no_map_instance_attr. Retrieved 2/6 statements.
# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.



def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: x + y


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
    var_4 = lambda a, b: a + b
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6


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
    var_10 = bool(True)
    assert var_10 is True


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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_ordereddict. Retrieved 16/22 statements.
# Partially parsed test_map_structure_zip_custom_no_map_instance. Retrieved 2/7 statements.



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
    var_0 = lambda x, y: x - y
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
    var_17 = bool(var_16 == [[-4, -4], [-4, -4]])
    assert var_17 is True


def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = (var_5, var_6, var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == (4, 10, 18))
    assert var_11 is True


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
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 4
    var_9 = {var_3: var_8}
    var_10 = 5
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = [var_7, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == {'a': {'c': 8}, 'b': 15})
    assert var_14 is True

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 5
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 6
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 2
    var_9 = (var_1, var_8)
    var_10 = 3
    var_11 = (var_4, var_10)
    var_12 = [var_9, var_11]
    var_13 = (var_1, var_10)
    var_14 = (var_4, var_10)
    var_15 = [var_13, var_14]


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
    var_16 = bool(var_15 == (['14', '25'], {'a': '36'}))
    assert var_16 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = ' world'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'hello world'


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


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

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: x.val + y.val



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true. Retrieved 18/43 statements.


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
    var_14 = (var_2, var_11)
    var_15 = 6
    var_16 = (var_5, var_15)
    var_17 = [var_14, var_16]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 11/21 statements.



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
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = lambda a, b: a + b
    var_9 = 6
    var_10 = 8


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


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = {var_1: var_6}
    var_8 = [var_5, var_7]
    var_9 = (var_4, var_8)
    var_10 = lambda x, y: x * y
    var_11 = module_0.map_structure_zip(var_10, var_9)
    var_12 = bool(var_11 == [3, {'a': 8}])
    assert var_12 is True


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


def test_case_0():
    var_0 = lambda x, y: x / y
    var_1 = 5
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == 2.5)
    assert var_5 is True


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
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true_for_namedtuple. Retrieved 7/11 statements.
# Partially parsed test_predicate_at_line_19_evaluates_to_true_for_another_namedtuple. Retrieved 7/11 statements.


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
    var_0 = 5
    var_1 = (var_0,)
    var_2 = '_fields'
    var_3 = hasattr(var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_true_for_set. Retrieved 1/2 statements.


def test_case_0():
    var_0 = set()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_true_for_namedtuple. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = '_fields'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_ordered_dict. Retrieved 17/23 statements.



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
    var_0 = lambda x, y: x - y
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
    var_17 = bool(var_16 == [[-4, -4], [-4, -4]])
    assert var_17 is True


def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = (var_5, var_6, var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == (4, 10, 18))
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
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 4
    var_9 = {var_3: var_8}
    var_10 = 5
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = [var_7, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == {'a': {'c': 8}, 'b': 15})
    assert var_14 is True


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
    var_16 = bool(var_15 == (['14', '25'], {'a': '36'}))
    assert var_16 is True


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = ' world'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'hello world'


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = {}
    var_2 = {}
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True

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


def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    var_7 = bool(var_6 == ['A', 'B', 'C'])
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_no_map_instance_attr. Retrieved 3/8 statements.



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
    var_0 = lambda x, y: x - y
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
    var_17 = bool(var_16 == [[-4, -4], [-4, -4]])
    assert var_17 is True


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
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 5
    var_5 = {var_3: var_4}
    var_6 = 2
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 3
    var_9 = {var_3: var_8}
    var_10 = 1
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = [var_7, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == {'a': {'c': 2}, 'b': 1})
    assert var_14 is True


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
    var_9 = [var_5, var_8]
    var_10 = 5
    var_11 = {var_4: var_10}
    var_12 = (var_9, var_11)
    var_13 = [var_7, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == ([4, 6], {'a': 8}))
    assert var_15 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

def test_case_0():
    var_0 = True
    var_1 = 42
    var_2 = lambda x, y: var_1


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
    var_1 = 7
    var_2 = [var_1]
    var_3 = 8
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    var_7 = bool(var_6 == [56])
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 10/15 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/9 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 8/14 statements.



def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True


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
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True


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


def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'c'
    var_8 = {var_7: var_0}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [3, 6], 'b': {'c': 9}})
    assert var_11 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 3
    var_3 = 4
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {4, 9, 16})
    assert var_6 is True


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
    var_19 = 20
    var_20 = [var_16, var_19]
    var_21 = 30
    var_22 = 40
    var_23 = (var_21, var_22)
    var_24 = 50
    var_25 = {var_24}
    var_26 = 60
    var_27 = {var_12: var_26}
    var_28 = {var_0: var_20, var_1: var_23, var_2: var_25, var_3: var_27}
    var_29 = bool(var_18 == var_28)
    assert var_29 is True


def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'


def test_case_0():
    var_0 = 100
    var_1 = lambda x: x + var_0
    var_2 = 50
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 150

def test_case_0():
    var_0 = 42
    var_1 = 2
    var_2 = lambda x: x.value * var_1


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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: x * var_4


def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ['1', '2', '3'])
    assert var_6 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 13/20 statements.



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


def test_case_0():
    var_0 = lambda a, b: a + b
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
    var_4 = lambda p1, p2: p1 + p2
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6


def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 10
    var_4 = 20
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 5
    var_7 = 8
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 5, 'b': 12})
    assert var_11 is True


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


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = [var_2]
    var_4 = module_0.map_structure_zip(var_1, var_3)
    assert var_4 == 10


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = ' world'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'hello world'


def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 6

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
    var_10 = "Structures cannot contain `set` because it's unordered"



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 10/32 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = 'R'
    var_5 = ()
    var_6 = {}
    var_7 = type(var_4, var_5, var_6)
    var_8 = '_no_map'
    var_9 = True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/12 statements.
# Partially parsed test_map_structure_with_custom_no_map_instance. Retrieved 3/6 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.



def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True


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
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True


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
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 'a'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = 'b'
    var_8 = 2
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = module_0.map_structure(var_1, var_10)
    var_12 = bool(var_11 == {'x': {'a': 11}, 'y': {'b': 12}})
    assert var_12 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
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


def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'hello'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'HELLO'


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
    var_7 = 3
    var_8 = lambda x: x * var_7


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = []
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True
    var_5 = lambda x: x * var_0
    var_6 = {}
    var_7 = module_0.map_structure(var_5, var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True
    var_9 = lambda x: x * var_0
    var_10 = set()
    var_11 = module_0.map_structure(var_9, var_10)
    var_12 = set()
    var_13 = bool(var_11 == var_12)
    assert var_13 is True
    var_14 = lambda x: x * var_0
    var_15 = ()
    var_16 = module_0.map_structure(var_14, var_15)
    var_17 = bool(var_16 == ())
    assert var_17 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/13 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/6 statements.



def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True


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
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True


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
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == {'a': 4, 'b': 9})
    assert var_7 is True


def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 5
    var_5 = 6
    var_6 = [var_4, var_5]
    var_7 = 7
    var_8 = 8
    var_9 = [var_7, var_8]
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = module_0.map_structure(var_1, var_10)
    var_12 = bool(var_11 == {'x': [4, 5], 'y': [6, 7]})
    assert var_12 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x / var_0
    var_2 = 4
    var_3 = 6
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1.0, 2.0, 3.0})
    assert var_6 is True

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


def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15


def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'


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
    var_13 = 10
    var_14 = lambda x: x + var_13
    var_15 = module_0.map_structure(var_14, var_12)
    var_16 = bool(var_15 == {'a': [{'b': (11, 12)}, {'c': {13, 14}}]})
    assert var_16 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_ordered_dict. Retrieved 13/20 statements.
# Partially parsed test_map_structure_zip_no_map_instance_attr. Retrieved 2/7 statements.



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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'ab'
    var_2 = 'cd'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'abcd'


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: (x, y)


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 5
    var_9 = 6
    var_10 = (var_8, var_9)
    var_11 = 7
    var_12 = 8
    var_13 = (var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [(6, 8), (10, 12)])
    assert var_17 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_ordered_dict. Retrieved 16/22 statements.



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


def test_case_0():
    var_0 = lambda x, y: x - y
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
    var_17 = -4
    var_18 = -4
    var_19 = [var_17, var_18]
    var_20 = -4
    var_21 = -4
    var_22 = [var_20, var_21]
    var_23 = [var_19, var_22]
    var_24 = bool(var_16 == var_23)
    assert var_24 is True


def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = (var_5, var_6, var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = 10
    var_12 = 18
    var_13 = (var_5, var_11, var_12)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

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
    var_11 = 6
    var_12 = {var_1: var_7, var_2: var_11}
    var_13 = bool(var_10 == var_12)
    assert var_13 is True


def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 4
    var_9 = {var_3: var_8}
    var_10 = 5
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = [var_7, var_11]
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = 8
    var_15 = {var_3: var_14}
    var_16 = 15
    var_17 = {var_1: var_15, var_2: var_16}
    var_18 = bool(var_13 == var_17)
    assert var_18 is True


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
    var_16 = '14'
    var_17 = '25'
    var_18 = [var_16, var_17]
    var_19 = '36'
    var_20 = {var_4: var_19}
    var_21 = (var_18, var_20)
    var_22 = bool(var_15 == var_21)
    assert var_22 is True


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = 15
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = ' world'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = 'hello world'
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = {}
    var_2 = {}
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = {}
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_map_structure_with_no_map_types. Retrieved 3/8 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 12/16 statements.


def test_case_0():
    var_0 = '_no_map'
    var_1 = 1
    var_2 = lambda x: x + var_1

def test_case_0():
    var_0 = True
    var_1 = '_no_map'
    var_2 = 1
    var_3 = lambda x: x + var_2


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True


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
    var_6 = lambda x: x + var_4
    var_7 = 3


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = module_0.map_structure(var_5, var_4)
    var_7 = bool(var_6 == {'a': 2, 'b': 4})
    assert var_7 is True


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 2
    var_6 = 3
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_0: var_2, var_1: var_7}
    var_9 = lambda x: x + var_2
    var_10 = module_0.map_structure(var_9, var_8)
    var_11 = bool(var_10 == {'a': 2, 'b': {'c': 3, 'd': 4}})
    assert var_11 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = bool(var_5 == {2, 4, 6})
    assert var_6 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: x + var_1
    var_8 = (var_0, var_4)
    var_9 = 3
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 'c'
    var_7 = 4
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = {var_0: var_4, var_1: var_9}
    var_11 = lambda x: x * var_3
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = bool(var_12 == {'a': [2, 4], 'b': (6, {'c': 8})})
    assert var_13 is True


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = lambda x: x + var_1
    var_3 = module_0.map_structure(var_2, var_0)
    assert var_3 == 6



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 6/29 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '_no_map'
    var_5 = True
    var_6 = 2
    var_7 = lambda x: x * var_6



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/16 statements.
# Partially parsed test_map_structure_zip_ordereddict. Retrieved 17/22 statements.
# Partially parsed test_map_structure_zip_no_map_instance_attr. Retrieved 3/10 statements.



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
    var_0 = lambda x, y: x - y
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
    var_17 = bool(var_16 == [[-4, -4], [-4, -4]])
    assert var_17 is True


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
    var_12 = lambda x, y: x * y
    var_13 = (var_0, var_7)
    var_14 = 8
    var_15 = (var_3, var_14)
    var_16 = [var_13, var_15]


def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
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
    var_16 = lambda x, y: x - y
    var_17 = [var_8, var_15]
    var_18 = module_0.map_structure_zip(var_16, var_17)
    var_19 = bool(var_18 == {'list': [-4, -4], 'tuple': (-4, -4)})
    assert var_19 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

def test_case_0():
    var_0 = True
    var_1 = 42
    var_2 = lambda x, y: var_1


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


def test_case_0():
    var_0 = None
    var_1 = lambda x, y: var_0
    var_2 = {}
    var_3 = {}
    var_4 = [var_2, var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)
    var_6 = bool(var_5 == {})
    assert var_6 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'helloworld'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordereddict. Retrieved 18/24 statements.



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
    var_13 = 8
    var_14 = (var_1, var_13)
    var_15 = 15
    var_16 = (var_4, var_15)
    var_17 = [var_14, var_16]


def test_case_0():
    var_0 = lambda x, y: str(x) + str(y)
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = (var_4,)
    var_6 = [var_3, var_5]
    var_7 = 3
    var_8 = {var_1: var_7}
    var_9 = 4
    var_10 = (var_9,)
    var_11 = [var_8, var_10]
    var_12 = (var_6, var_11)
    var_13 = module_0.map_structure_zip(var_0, var_12)
    var_14 = bool(var_13 == [{'a': '13'}, ('24',)])
    assert var_14 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'helloworld'


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


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


def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 5
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = (var_5, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': {'b': 3}})
    assert var_11 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_dict. Retrieved 7/8 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_ordereddict. Retrieved 9/15 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_nested_dict. Retrieved 9/10 statements.



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
    var_1 = 5
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 10
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 1
    var_8 = lambda x: x + var_7


def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'k1'
    var_2 = 'k2'
    var_3 = 'a'
    var_4 = 'subk'
    var_5 = 'b'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.map_structure(var_0, var_7)
    var_9 = bool(var_8 == {'k1': 'A', 'k2': {'subk': 'B'}})
    assert var_9 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_types. Retrieved 3/6 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 2/7 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 9/14 statements.



def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True


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
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True


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


def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'c'
    var_8 = {var_7: var_0}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [3, 6], 'b': {'c': 9}})
    assert var_11 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 3
    var_3 = 4
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {4, 9, 16})
    assert var_6 is True

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 'transformed'
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = 'first'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'second'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 100
    var_8 = lambda x: x * var_7


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
    var_17 = 0.5
    var_18 = lambda x: x + var_17
    var_19 = module_0.map_structure(var_18, var_16)
    var_20 = 1.5
    var_21 = 2.5
    var_22 = [var_20, var_21]
    var_23 = 3.5
    var_24 = 4.5
    var_25 = (var_23, var_24)
    var_26 = 5.5
    var_27 = 6.5
    var_28 = {var_26, var_27}
    var_29 = 7.5
    var_30 = {var_13: var_29}
    var_31 = {var_0: var_22, var_1: var_25, var_2: var_28, var_3: var_30}
    var_32 = var_19['list']
    var_33 = bool(var_19['list'] == var_31['list'])
    assert var_33 is True
    var_34 = var_19['tuple']
    var_35 = bool(var_19['tuple'] == var_31['tuple'])
    assert var_35 is True
    var_36 = var_19['set']
    var_37 = bool(var_19['set'] == var_31['set'])
    assert var_37 is True
    var_38 = var_19['inner_dict']
    var_39 = bool(var_19['inner_dict'] == var_31['inner_dict'])
    assert var_39 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x / var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 5


def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'


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



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_to_true_for_namedtuple. Retrieved 7/11 statements.
# Partially parsed test_predicate_at_line_13_evaluates_to_true_for_namedtuple_with_multiple_fields. Retrieved 9/13 statements.
# Partially parsed test_predicate_at_line_13_evaluates_to_true_for_empty_namedtuple. Retrieved 3/7 statements.


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
    var_7 = 'Wonderland'
    var_8 = '_fields'

def test_case_0():
    var_0 = 'Empty'
    var_1 = []
    var_2 = '_fields'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = '_fields'
    var_5 = hasattr(var_3, var_4)
    assert var_5 is False

def test_case_0():
    var_0 = 42
    var_1 = (var_0,)
    var_2 = '_fields'
    var_3 = hasattr(var_1, var_2)
    assert var_3 is False

def test_case_0():
    var_0 = ()
    var_1 = '_fields'
    var_2 = hasattr(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_true_for_set. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)



# Parsed testcases at query #14
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_dict. Retrieved 7/8 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_ordereddict. Retrieved 9/15 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_nested_dict. Retrieved 7/8 statements.



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


def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'key'
    var_2 = 'inner'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.map_structure(var_0, var_5)
    var_7 = bool(var_6 == {'key': {'inner': 'VALUE'}})
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_no_map_types. Retrieved 2/7 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 17/23 statements.



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


def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = (var_3, var_6, var_9)
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == (9, 12))
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
    var_9 = (var_5, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 9, 'b': 18})
    assert var_11 is True


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


def test_case_0():
    var_0 = lambda x, y, z: x * y * z
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
    var_12 = bool(var_11 == [15, 48])
    assert var_12 is True


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
    var_1 = lambda x, y: (x, y)

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
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


def test_case_0():
    var_0 = None
    var_1 = lambda x, y: var_0
    var_2 = []
    var_3 = [var_2]
    var_4 = []
    var_5 = [var_4]
    var_6 = (var_3, var_5)
    var_7 = module_0.map_structure_zip(var_1, var_6)
    var_8 = bool(var_7 == [[]])
    assert var_8 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_dict. Retrieved 7/8 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_ordereddict. Retrieved 9/15 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_nested_dict. Retrieved 7/8 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_empty_dict. Retrieved 3/4 statements.



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


def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'key'
    var_2 = 'nested'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.map_structure(var_0, var_5)
    var_7 = bool(var_6 == {'key': {'nested': 'VALUE'}})
    assert var_7 is True


def test_case_0():
    var_0 = lambda x: x
    var_1 = {}
    var_2 = module_0.map_structure(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_map_structure_zip_with_single_non_collection. Retrieved 8/12 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 15/31 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 13/17 statements.
# Partially parsed test_map_structure_zip_with_nested_structures. Retrieved 22/26 statements.
# Partially parsed test_map_structure_zip_with_plain_tuple. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_single_element_non_collection. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 8/13 statements.


def test_case_0():
    var_0 = '_no_map'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = '_no_map'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 5
    var_10 = 6
    var_11 = 7
    var_12 = 8
    var_13 = 10
    var_14 = 12

def test_case_0():
    var_0 = '_no_map'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 'world'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'foo'
    var_7 = 'bar'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = 'hellofoo'
    var_11 = 'worldbar'
    var_12 = {var_1: var_10, var_2: var_11}

def test_case_0():
    var_0 = '_no_map'
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
    var_16 = 12
    var_17 = [var_8, var_16]
    var_18 = 21
    var_19 = 32
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]

def test_case_0():
    var_0 = '_no_map'
    var_1 = 10
    var_2 = 20
    var_3 = (var_1, var_2)
    var_4 = 5
    var_5 = 15
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = (var_4, var_4)

def test_case_0():
    var_0 = '_no_map'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '_no_map'
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_map_structure_zip_dict_ordered_dict. Retrieved 17/24 statements.


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



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordereddict. Retrieved 17/23 statements.



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

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 5
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 10
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 2
    var_9 = (var_1, var_8)
    var_10 = 3
    var_11 = (var_4, var_10)
    var_12 = [var_9, var_11]
    var_13 = (var_1, var_10)
    var_14 = 7
    var_15 = (var_4, var_14)
    var_16 = [var_13, var_15]


def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = 3
    var_6 = [var_5]
    var_7 = (var_2, var_4, var_6)
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == [6])
    assert var_9 is True


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
    var_10 = 5
    var_11 = 6
    var_12 = [var_11]
    var_13 = [var_10, var_12]
    var_14 = 7
    var_15 = 8
    var_16 = [var_15]
    var_17 = [var_14, var_16]
    var_18 = [var_13, var_17]
    var_19 = (var_9, var_18)
    var_20 = module_0.map_structure_zip(var_0, var_19)
    var_21 = bool(var_20 == [[6, [8]], [10, [12]]])
    assert var_21 is True


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
    var_0 = lambda x, y: x + y
    var_1 = 'ab'
    var_2 = 'cd'
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'abcd'


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15


def test_case_0():
    var_0 = lambda x, y, z: x * y * z
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
    var_12 = bool(var_11 == [15, 48])
    assert var_12 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = {}
    var_2 = {}
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_map_structure_zip_no_map_types. Retrieved 1/29 statements.
# Partially parsed test_map_structure_zip_no_map_instance_attr. Retrieved 3/29 statements.
# Partially parsed test_map_structure_zip_list. Retrieved 9/31 statements.
# Partially parsed test_map_structure_zip_tuple. Retrieved 9/31 statements.
# Partially parsed test_map_structure_zip_namedtuple. Retrieved 11/39 statements.
# Failed to parse test_map_structure_zip_dict.


def test_case_0():
    var_0 = '_no_map'

def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = True

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

def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = 'Point'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = [var_3, var_4]
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = 4
    var_10 = 6



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 10/15 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 3/6 statements.



def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True


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
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)
    var_6 = bool(var_5 == ('A', 'B', 'C'))
    assert var_6 is True


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


def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'c'
    var_8 = {var_7: var_0}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = bool(var_10 == {'a': [3, 6], 'b': {'c': 9}})
    assert var_11 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 3
    var_3 = 4
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {4, 9, 16})
    assert var_6 is True

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

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1


def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'


def test_case_0():
    var_0 = 5
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15


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
    var_13 = lambda x: x + var_2
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = bool(var_14 == {'a': [{'b': (2, 3)}, {'c': {4, 5}}]})
    assert var_15 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 3/17 statements.


def test_case_0():
    var_0 = '_no_map'
    var_1 = True
    var_2 = 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_dict. Retrieved 7/8 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_ordereddict. Retrieved 9/15 statements.
# Partially parsed test_predicate_at_line_18_evaluates_to_true_for_nested_dict. Retrieved 9/10 statements.



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


def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'k1'
    var_2 = 'k2'
    var_3 = 'a'
    var_4 = 'subk'
    var_5 = 'b'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.map_structure(var_0, var_7)
    var_9 = bool(var_8 == {'k1': 'A', 'k2': {'subk': 'B'}})
    assert var_9 is True



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_ordereddict. Retrieved 13/20 statements.
# Partially parsed test_map_structure_zip_with_custom_no_map_instance_attr. Retrieved 2/6 statements.



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

def test_case_0():
    var_0 = lambda x, y: x / y
    var_1 = 'a'
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 20
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 2
    var_9 = (var_1, var_8)
    var_10 = 5
    var_11 = (var_4, var_10)
    var_12 = [var_9, var_11]


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


def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 5
    var_11 = 6
    var_12 = (var_10, var_11)
    var_13 = 7
    var_14 = 8
    var_15 = [var_13, var_14]
    var_16 = {var_1: var_12, var_2: var_15}
    var_17 = [var_9, var_16]
    var_18 = module_0.map_structure_zip(var_0, var_17)
    var_19 = bool(var_18 == {'a': (5, 12), 'b': [21, 32]})
    assert var_19 is True


def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'hello'
    var_2 = [var_1]
    var_3 = [var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == ['HELLO'])
    assert var_5 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'ab'


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


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = {}
    var_2 = {}
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True


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

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: x + y



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/12 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/9 statements.



def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [2, 4, 6])
    assert var_6 is True


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
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3, var_0)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == (3, 6, 9))
    assert var_6 is True


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


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == {1, 4, 9})
    assert var_6 is True


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
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hello!'


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

def test_case_0():
    var_0 = 42
    var_1 = 2
    var_2 = lambda x: x.value * var_1


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


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = {var_1: var_4}
    var_6 = 4
    var_7 = {var_6}
    var_8 = [var_0, var_5, var_7]
    var_9 = lambda x: x
    var_10 = module_0.map_structure(var_9, var_8)
    var_11 = bool(var_10 == [1, {'a': (2, 3)}, {4}])
    assert var_11 is True



# Parsed testcases at query #29
#--------------------------





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



