####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'a'
    var_3 = 'c'
    var_4 = 'b'
    var_5 = {var_4: var_0}
    var_6 = 2
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/12 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 3/8 statements.


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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

def test_case_0():
    var_0 = '_no_map'
    var_1 = True
    var_2 = lambda x: x



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_no_map_instance. Retrieved 2/8 statements.


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

import flutes.structure as module_0

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 7/8 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = module_0.map_structure(var_5, var_4)



# Parsed testcases at query #5
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
    var_9 = 6
    var_10 = [var_5, var_9]

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
    var_9 = 8
    var_10 = (var_4, var_9)

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
    var_11 = 7
    var_12 = {var_1: var_7, var_2: var_11}

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
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = 50



# Parsed testcases at query #6
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
    var_3 = 5
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 7
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
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



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
    var_3 = 5
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 7
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = [var_1, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
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
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 2



# Parsed testcases at query #8
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = module_0.map_structure(var_5, var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_map_structure_no_map_types. Retrieved 2/16 statements.


def test_case_0():
    var_0 = '_no_map'
    var_1 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 4/9 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

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
    var_4 = 2
    var_5 = 3
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = '!'
    var_1 = lambda x: x + var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'hi'
    var_5 = 'c'
    var_6 = 'hello'
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.map_structure(var_1, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 3
    var_3 = 4
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

def test_case_0():
    var_0 = '_no_map'
    var_1 = True
    var_2 = 'mapped'
    var_3 = lambda x: var_2

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
    var_8 = 'd'
    var_9 = 4
    var_10 = 5
    var_11 = 6
    var_12 = {var_10, var_11}
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = {var_0: var_6, var_1: var_13}
    var_15 = lambda x: x * var_3
    var_16 = module_0.map_structure(var_15, var_14)



# Parsed testcases at query #11
#--------------------------




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
    var_8 = [var_4, var_7]
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)



# Parsed testcases at query #12
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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

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
    var_8 = (var_6, var_7)
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10



# Parsed testcases at query #13
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_map_structure_zip_with_list. Retrieved 9/10 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/16 statements.
# Partially parsed test_map_structure_zip_with_regular_tuple. Retrieved 9/10 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 11/12 statements.
# Partially parsed test_map_structure_zip_with_primitive. Retrieved 5/7 statements.


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
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_map_structure_handles_non_collection_types. Retrieved 1/4 statements.
# Partially parsed test_map_structure_maps_over_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_maps_over_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_maps_over_namedtuple. Retrieved 7/14 statements.
# Partially parsed test_map_structure_maps_over_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_maps_over_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_handles_nested_structures. Retrieved 11/14 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_map_structure_ordered_dict. Retrieved 8/14 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = module_0.map_structure(var_5, var_4)

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
    var_0 = 'a'
    var_1 = 'd'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = lambda x: x * var_5
    var_10 = module_0.map_structure(var_9, var_8)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_map_structure_zip_returns_correct_result_for_sets. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = {var_0, var_1, var_2}
    var_5 = [var_3, var_4]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_map_structure_zip_with_non_mappable_type. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_list. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_tuple. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 12/22 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = 5
    var_7 = 6
    var_8 = (var_6, var_7)
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 5
    var_9 = 6
    var_10 = 9
    var_11 = 12

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = 5
    var_9 = 6
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = [var_4, var_7, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 11/22 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = (var_3, var_6)
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
    var_15 = (var_7, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
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
    var_8 = lambda a, b: a + b
    var_9 = 6
    var_10 = 8

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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3

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
    var_8 = module_0.map_structure_zip(var_0, var_7)

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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_map_structure_with_named_tuple. Retrieved 8/13 statements.


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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'hello'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'HELLO'

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
    var_9 = lambda x: x * var_3
    var_10 = module_0.map_structure(var_9, var_8)



# Parsed testcases at query #23
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = {var_1: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_6, var_7}
    var_9 = {var_1: var_8}
    var_10 = [var_5, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)



# Parsed testcases at query #24
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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_no_map_types. Retrieved 3/9 statements.


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
    var_3 = (var_1, var_2)
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
    var_16 = [var_8, var_15]
    var_17 = lambda x, y: x + y
    var_18 = module_0.map_structure_zip(var_17, var_16)

def test_case_0():
    var_0 = '_no_map'
    var_1 = True
    var_2 = lambda x, y: x + y



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_map_structure_zip_returns_fn_result_when_obj_is_namedtuple. Retrieved 10/17 statements.


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
    var_0 = lambda x, y: x - y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 5
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 7
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
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/13 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 4/9 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = [var_2, var_0]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 5
    var_3 = 6
    var_4 = 7
    var_5 = (var_2, var_3, var_4)
    var_6 = module_0.map_structure(var_1, var_5)

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
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 'world'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 3
    var_3 = 4
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

def test_case_0():
    var_0 = '_no_map'
    var_1 = True
    var_2 = 'mapped'
    var_3 = lambda x: var_2



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_map_structure_zip_returns_false_for_set_input. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = set(var_6)
    var_8 = [var_3, var_7]



# Parsed testcases at query #29
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

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda a, b: Point(a.x + b.x, a.y + b.y)
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
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)



# Parsed testcases at query #30
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
    var_6 = 5
    var_7 = 15
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
    var_1 = 1
    var_2 = 2
    var_3 = (var_2,)
    var_4 = [var_1, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_6,)
    var_8 = [var_5, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.
# Partially parsed test_map_structure_zip_with_no_map_instance_attr. Retrieved 2/7 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = (var_3, var_6)
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
    var_15 = (var_7, var_14)
    var_16 = module_0.map_structure_zip(var_0, var_15)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3

def test_case_0():
    var_0 = True
    var_1 = lambda x, y: x + y

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



# Parsed testcases at query #32
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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x - y
    var_1 = 5
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 2
    var_5 = 1
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

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
    var_0 = lambda x, y: x / y
    var_1 = 10
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)



# Parsed testcases at query #33
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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'a'
    var_3 = 'c'
    var_4 = 'b'
    var_5 = {var_4: var_0}
    var_6 = 2
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 2
    var_6 = [var_0, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = (var_7, var_8)
    var_10 = 5
    var_11 = 6
    var_12 = {var_10, var_11}
    var_13 = {var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = module_0.map_structure(var_1, var_13)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_map_structure_returns_false_for_non_collection. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 42



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_map_structure_no_map_types. Retrieved 1/6 statements.
# Partially parsed test_map_structure_no_map_instance_attr. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '_no_map'

def test_case_0():
    var_0 = '_no_map'
    var_1 = True



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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 1
    var_6 = [var_5, var_0]
    var_7 = 3
    var_8 = 4
    var_9 = (var_7, var_8)
    var_10 = 5
    var_11 = 6
    var_12 = {var_10, var_11}
    var_13 = {var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = module_0.map_structure(var_1, var_13)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 11/18 statements.


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
    var_9 = 8
    var_10 = (var_4, var_9)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x - y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = -2
    var_10 = -2

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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x ** y
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = 8

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
    var_17 = [var_9, var_12]
    var_18 = 10
    var_19 = 12
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
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



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_map_structure_no_map_instance_attr. Retrieved 2/7 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0



# Parsed testcases at query #39
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

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: Point(x.x + y.x, x.y + y.y)
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



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 11/18 statements.


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

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x - y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = -2
    var_10 = -2

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



# Parsed testcases at query #41
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

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = (var_0, var_2, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = 'd'
    var_7 = 2
    var_8 = {var_5: var_7, var_6: var_0}
    var_9 = {var_2: var_4, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)

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
    var_0 = lambda x: x



# Parsed testcases at query #42
#--------------------------

# Failed to parse test_predicate_at_line_1_evaluates_to_false.




# Parsed testcases at query #43
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_3, var_4]
    var_6 = lambda x: x * var_1
    var_7 = module_0.map_structure(var_6, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = module_0.map_structure(var_5, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = lambda x: x * var_4
    var_8 = module_0.map_structure(var_7, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = lambda x: x * var_1
    var_3 = module_0.map_structure(var_2, var_0)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = lambda x: x.upper()
    var_2 = module_0.map_structure(var_1, var_0)
    assert var_2 == 'HELLO'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_map_structure_zip_namedtuple. Retrieved 10/17 statements.


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



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = module_0.map_structure(var_5, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

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
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_3, var_4]
    var_6 = lambda x: x * var_1
    var_7 = module_0.map_structure(var_6, var_5)

import flutes.structure as module_0

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
    var_9 = lambda x: x * var_5
    var_10 = module_0.map_structure(var_9, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = lambda x: x * var_1
    var_3 = module_0.map_structure(var_2, var_0)
    assert var_3 == 10



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_namedtuple. Retrieved 8/12 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = [var_2, var_0]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 5
    var_3 = 6
    var_4 = 7
    var_5 = (var_2, var_3, var_4)
    var_6 = module_0.map_structure(var_1, var_5)

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

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x + var_4
    var_7 = 3

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)

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
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = {var_0, var_2}
    var_4 = frozenset(var_3)
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = frozenset(var_7)
    var_9 = {var_4, var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = {var_2, var_5}
    var_12 = frozenset(var_11)
    var_13 = 5
    var_14 = {var_6, var_13}
    var_15 = frozenset(var_14)
    var_16 = {var_12, var_15}

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'hello'
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'HELLO'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/12 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 2/6 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 2/7 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 6

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
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_2, var_3]
    var_5 = 'd'
    var_6 = [var_1, var_4, var_5]
    var_7 = module_0.map_structure(var_0, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = (var_2, var_3, var_4)
    var_6 = module_0.map_structure(var_1, var_5)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 10
    var_7 = lambda x: x * var_6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: len(x)
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'apple'
    var_4 = 'banana'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x % var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = {var_2, var_0, var_3, var_4}
    var_6 = module_0.map_structure(var_1, var_5)

def test_case_0():
    var_0 = 'mapped'
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = 'mapped'
    var_1 = lambda x: var_0



# Parsed testcases at query #3
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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_custom_no_map_type. Retrieved 1/5 statements.
# Partially parsed test_map_structure_with_custom_no_map_instance_attr. Retrieved 1/6 statements.


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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_0, var_3]
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = module_0.map_structure(var_1, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_0, var_3}
    var_5 = frozenset(var_4)
    var_6 = 4
    var_7 = {var_2, var_5, var_6}
    var_8 = module_0.map_structure(var_1, var_7)
    var_9 = 6
    var_10 = {var_6, var_9}
    var_11 = frozenset(var_10)
    var_12 = 8
    var_13 = {var_0, var_11, var_12}

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

def test_case_0():
    var_0 = lambda x: x

def test_case_0():
    var_0 = lambda x: x



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 11/18 statements.


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

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x - y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = -2
    var_10 = -2

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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



# Parsed testcases at query #6
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
    var_3 = 5
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 2
    var_7 = 3
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
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 3/6 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

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
    var_4 = 2
    var_5 = 3
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.map_structure(var_1, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 3
    var_3 = 4
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1

import flutes.structure as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'd'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 'c'
    var_8 = 4
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = 5
    var_12 = 6
    var_13 = {var_11, var_12}
    var_14 = {var_0: var_5, var_1: var_10, var_2: var_13}
    var_15 = 10
    var_16 = lambda x: x + var_15
    var_17 = module_0.map_structure(var_16, var_14)
    var_18 = 11
    var_19 = 12
    var_20 = [var_18, var_19]
    var_21 = 13
    var_22 = 14
    var_23 = {var_7: var_22}
    var_24 = (var_21, var_23)
    var_25 = 15
    var_26 = 16
    var_27 = {var_25, var_26}
    var_28 = {var_0: var_20, var_1: var_24, var_2: var_27}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/16 statements.


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

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x - y
    var_5 = 10
    var_6 = 20
    var_7 = 5
    var_8 = 15

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
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_to_true_for_tuple. Retrieved 4/5 statements.
# Partially parsed test_predicate_at_line_13_evaluates_to_true_for_namedtuple. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_map_structure_zip_with_set. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = set(var_6)
    var_8 = [var_3, var_7]



# Parsed testcases at query #11
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = set(var_3)
    var_5 = [var_4, var_4]
    var_6 = lambda x, y: x + y
    var_7 = module_0.map_structure_zip(var_6, var_5)



# Parsed testcases at query #12
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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

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
    var_8 = (var_6, var_7)
    var_9 = {var_2: var_5, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 20

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 4



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_map_structure_zip_handles_namedtuple. Retrieved 10/17 statements.


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
    var_11 = 6
    var_12 = {var_1: var_7, var_2: var_11}

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
    var_9 = 6
    var_10 = (var_5, var_9)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = 3



# Parsed testcases at query #14
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = set(var_4)
    var_6 = 4
    var_7 = 5
    var_8 = 6
    var_9 = [var_6, var_7, var_8]
    var_10 = set(var_9)
    var_11 = [var_5, var_10]
    var_12 = module_0.map_structure_zip(var_0, var_11)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_map_structure_with_no_map_types. Retrieved 1/6 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 1/6 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


def test_case_0():
    var_0 = lambda x: x

def test_case_0():
    var_0 = lambda x: x

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = module_0.map_structure(var_5, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = lambda x: x * var_1
    var_3 = module_0.map_structure(var_2, var_0)
    assert var_3 == 10



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_map_structure_tuple. Retrieved 6/7 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_map_structure_with_set. Retrieved 6/7 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_map_structure_zip_with_no_map_types. Retrieved 6/10 statements.
# Partially parsed test_map_structure_zip_with_no_map_instance_attr. Retrieved 3/12 statements.


def test_case_0():
    var_0 = '_no_map'
    var_1 = 42
    var_2 = [var_1]
    var_3 = var_1.__class__
    var_4 = hasattr(var_1, var_0)
    var_5 = False

def test_case_0():
    var_0 = '_no_map'
    var_1 = True
    var_2 = False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_map_structure_zip_namedtuples. Retrieved 10/17 statements.


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
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
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
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = module_0.map_structure_zip(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_map_structure_tuple. Retrieved 6/7 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_map_structure_zip_list_predicate. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_map_structure_zip_returns_correct_result_for_namedtuple. Retrieved 10/17 statements.


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
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_map_structure_zip_with_no_map_types. Retrieved 5/6 statements.
# Partially parsed test_map_structure_zip_with_no_map_instance_attr. Retrieved 4/8 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = '_no_map'
    var_1 = 42
    var_2 = [var_1]
    var_3 = lambda x: x
    var_4 = module_0.map_structure_zip(var_3, var_2)
    assert var_4 == 42

def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = True
    var_3 = lambda x: x

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x, y: x + y
    var_8 = module_0.map_structure_zip(var_7, var_6)

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
    var_8 = [var_4, var_7]
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = lambda x, y: x + y
    var_8 = module_0.map_structure_zip(var_7, var_6)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1.5
    var_1 = 2.5
    var_2 = [var_0, var_1]
    var_3 = lambda x, y: x + y
    var_4 = module_0.map_structure_zip(var_3, var_2)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = lambda x: x + var_4
    var_7 = 3

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 6



# Parsed testcases at query #27
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
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 1/5 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 2/5 statements.


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
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 'c'
    var_6 = 'd'
    var_7 = 3
    var_8 = {var_5: var_0, var_6: var_7}
    var_9 = {var_2: var_4, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)

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
    var_0 = lambda x: x

def test_case_0():
    var_0 = True
    var_1 = lambda x: x



# Parsed testcases at query #29
#--------------------------




import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_map_structure_zip_with_list. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_map_structure_with_no_map_types. Retrieved 5/6 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = '_no_map'
    var_1 = 42
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.map_structure(var_3, var_1)
    assert var_4 == 84

def test_case_0():
    var_0 = ()
    var_1 = '_no_map'
    var_2 = True
    var_3 = lambda x: x

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda x: x * var_3
    var_6 = module_0.map_structure(var_5, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)

import flutes.structure as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = lambda x: x.upper()
    var_2 = module_0.map_structure(var_1, var_0)
    assert var_2 == 'TEST'



# Parsed testcases at query #32
#--------------------------




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



# Parsed testcases at query #33
#--------------------------




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
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_0.map_structure(var_1, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = (var_0, var_5)
    var_7 = 4
    var_8 = 5
    var_9 = {var_7, var_8}
    var_10 = [var_2, var_6, var_9]
    var_11 = module_0.map_structure(var_1, var_10)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 3/6 statements.


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
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.map_structure(var_0, var_4)

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
    var_4 = 'c'
    var_5 = 2
    var_6 = 3
    var_7 = {var_2: var_0, var_3: var_5, var_4: var_6}
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'c'
    var_4 = 'b'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = 2
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = module_0.map_structure(var_1, var_8)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 15

def test_case_0():
    var_0 = True
    var_1 = 'mapped'
    var_2 = lambda x: var_1



# Parsed testcases at query #35
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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

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
    var_3 = [var_2, var_0]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure(var_1, var_7)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 'a'
    var_3 = 'c'
    var_4 = 'b'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = 'd'
    var_8 = {var_7: var_0}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = module_0.map_structure(var_1, var_9)

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

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = {var_2, var_0}
    var_4 = frozenset(var_3)
    var_5 = 3
    var_6 = 4
    var_7 = {var_5, var_6}
    var_8 = frozenset(var_7)
    var_9 = {var_4, var_8}
    var_10 = module_0.map_structure(var_1, var_9)
    var_11 = {var_0, var_6}
    var_12 = frozenset(var_11)
    var_13 = 6
    var_14 = 8
    var_15 = {var_13, var_14}
    var_16 = frozenset(var_15)
    var_17 = {var_12, var_16}



# Parsed testcases at query #36
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
    var_3 = 5
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 2
    var_7 = 3
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
    var_0 = lambda x, y: x / y
    var_1 = 10
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)

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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line19_evaluates_to_true_for_namedtuple. Retrieved 7/11 statements.


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



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_map_structure_returns_fn_result_for_no_map_types. Retrieved 5/6 statements.
# Partially parsed test_map_structure_returns_fn_result_for_no_map_instance_attr. Retrieved 4/9 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = '_no_map'
    var_1 = 1
    var_2 = lambda x: x + var_1
    var_3 = 5
    var_4 = module_0.map_structure(var_2, var_3)
    assert var_4 == 6

def test_case_0():
    var_0 = set()
    var_1 = '_no_map'
    var_2 = 1
    var_3 = lambda x: x + var_2



# Parsed testcases at query #39
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
    var_0 = lambda x, y: x - y
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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 6



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 11/18 statements.


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

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x - y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = -2
    var_10 = -2

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

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = {var_1, var_2}
    var_4 = 'c'
    var_5 = 'd'
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)

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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_map_structure_zip_tuple_without_fields. Retrieved 9/10 statements.


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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_no_map_type. Retrieved 4/9 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = 4
    var_7 = 6
    var_8 = [var_0, var_6, var_7]

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
    var_9 = [var_2, var_4]
    var_10 = 5
    var_11 = [var_5, var_10]
    var_12 = [var_9, var_11]

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3, var_0)
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = 6
    var_7 = 9
    var_8 = (var_0, var_6, var_7)

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
    var_7 = 'HELLO'
    var_8 = 'WORLD'
    var_9 = {var_1: var_7, var_2: var_8}

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = 4
    var_7 = 9
    var_8 = {var_2, var_6, var_7}

import flutes.structure as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    var_4 = 15

def test_case_0():
    var_0 = '_no_map'
    var_1 = True
    var_2 = 'mapped'
    var_3 = lambda x: var_2



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/13 statements.
# Partially parsed test_map_structure_with_no_map_instance_attr. Retrieved 3/9 statements.


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
    var_4 = (var_0, var_2, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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
    var_3 = 'apple'
    var_4 = 'banana'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.map_structure(var_0, var_5)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = {var_2, var_0, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 4
    var_8 = 5
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = [var_0, var_4, var_9]
    var_11 = module_0.map_structure(var_1, var_10)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 10
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 20

import flutes.structure as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = 'hello'
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 'hellohellohello'

def test_case_0():
    var_0 = 10
    var_1 = 2
    var_2 = lambda x: x * var_1



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 6/7 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_map_structure_with_namedtuple. Retrieved 9/14 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = (var_2, var_0, var_3)
    var_5 = module_0.map_structure(var_1, var_4)

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
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 2
    var_3 = 3
    var_4 = {var_0, var_2, var_3}
    var_5 = module_0.map_structure(var_1, var_4)

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
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 6



