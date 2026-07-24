####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_0, var_5]
    var_7 = 0
    var_8 = module_0.get_in(var_6, var_4, var_7)
    assert var_8 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 10
    var_7 = [var_0, var_6]
    var_8 = None
    var_9 = module_0.get_in(var_7, var_5, var_8)
    assert var_9 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = [var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_3, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'string'
    var_1 = []
    var_2 = module_0.get_in(var_1, var_0)
    assert var_2 == 'string'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = [var_0, var_1, var_7]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 is None
    var_10 = [var_0, var_1, var_7]
    var_11 = 0
    var_12 = module_0.get_in(var_10, var_6, var_11)
    assert var_12 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'd'
    var_10 = [var_7, var_8, var_9]
    var_11 = True
    var_12 = module_0.get_in(var_10, var_6, no_default=var_11)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 10
    var_7 = [var_0, var_6]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 is None
    var_9 = [var_0, var_6]
    var_10 = 0
    var_11 = module_0.get_in(var_9, var_5, var_10)
    assert var_11 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'a'
    var_7 = 10
    var_8 = [var_6, var_7]
    var_9 = True
    var_10 = module_0.get_in(var_8, var_5, no_default=var_9)
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_0, var_5]
    var_7 = None
    var_8 = module_0.get_in(var_6, var_4, var_7)
    assert var_8 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_0, var_5]
    var_7 = 'default'
    var_8 = module_0.get_in(var_6, var_4, var_7)
    assert var_8 == 'default'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'c'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_0, var_2, var_4]
    var_10 = module_0.get_in(var_9, var_8)
    assert var_10 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None
    var_6 = [var_0, var_3]
    var_7 = 0
    var_8 = module_0.get_in(var_6, var_2, var_7)
    assert var_8 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_in_raises_keyerror_when_no_default. Retrieved 5/6 statements.


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)



# Parsed testcases at query #4
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_0, var_5]
    var_7 = 0
    var_8 = module_0.get_in(var_6, var_4, var_7)
    assert var_8 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'a'
    var_6 = 'c'
    var_7 = [var_5, var_6]
    var_8 = True
    var_9 = module_0.get_in(var_7, var_4, no_default=var_8)
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 10
    var_7 = [var_0, var_6]
    var_8 = None
    var_9 = module_0.get_in(var_7, var_5, var_8)
    assert var_9 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_0, var_5]
    var_7 = module_0.get_in(var_6, var_4)
    assert var_7 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0



# Parsed testcases at query #5
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = [var_0, var_1, var_7]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = [var_0, var_1, var_7]
    var_9 = 0
    var_10 = module_0.get_in(var_8, var_6, var_9)
    assert var_10 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'd'
    var_10 = [var_7, var_8, var_9]
    var_11 = True
    var_12 = module_0.get_in(var_10, var_6, no_default=var_11)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 10
    var_7 = [var_0, var_6]
    var_8 = -1
    var_9 = module_0.get_in(var_7, var_5, var_8)
    assert var_9 == -1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = None
    var_3 = 0
    var_4 = module_0.get_in(var_1, var_2, var_3)
    assert var_4 == 0



# Parsed testcases at query #6
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = [var_0, var_1, var_7]
    var_9 = 99
    var_10 = module_0.get_in(var_8, var_6, var_9)
    assert var_10 == 99

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = [var_0, var_1, var_7]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'd'
    var_10 = [var_7, var_8, var_9]
    var_11 = True
    var_12 = module_0.get_in(var_10, var_6, no_default=var_11)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = 0
    var_10 = [var_0, var_1, var_9]
    var_11 = module_0.get_in(var_10, var_8)
    assert var_11 == 3

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 42
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #7
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = None
    var_7 = False
    var_8 = module_0.get_in(var_2, var_5, var_6, var_7)
    assert var_8 == 1



# Parsed testcases at query #8
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = [var_0, var_1, var_7]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = [var_0, var_1, var_7]
    var_9 = 42
    var_10 = module_0.get_in(var_8, var_6, var_9)
    assert var_10 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'd'
    var_10 = [var_7, var_8, var_9]
    var_11 = True
    var_12 = module_0.get_in(var_10, var_6, no_default=var_11)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 10
    var_7 = [var_0, var_6]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = 0
    var_10 = [var_0, var_9, var_1, var_2]
    var_11 = module_0.get_in(var_10, var_8)
    assert var_11 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_0, var_5]
    var_7 = 42
    var_8 = module_0.get_in(var_6, var_4, var_7)
    assert var_8 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 5
    var_6 = [var_0, var_5]
    var_7 = 42
    var_8 = module_0.get_in(var_6, var_4, var_7)
    assert var_8 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = 0
    var_10 = [var_0, var_9, var_1, var_2]
    var_11 = module_0.get_in(var_10, var_8)
    assert var_11 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 42
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 42



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = bool(not (False or False or False))
    assert var_0 is True



# Parsed testcases at query #4
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = [var_0, var_1, var_7]
    var_9 = 2
    var_10 = module_0.get_in(var_8, var_6, var_9)
    assert var_10 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'd'
    var_10 = [var_7, var_8, var_9]
    var_11 = True
    var_12 = module_0.get_in(var_10, var_6, no_default=var_11)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 5
    var_7 = [var_0, var_6]
    var_8 = 0
    var_9 = module_0.get_in(var_7, var_5, var_8)
    assert var_9 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_0, var_5]
    var_7 = module_0.get_in(var_6, var_4)
    assert var_7 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0



# Parsed testcases at query #5
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 123
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 123

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.get_in(var_3, var_2)
    assert var_4 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 42
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 10
    var_7 = [var_0, var_6]
    var_8 = None
    var_9 = module_0.get_in(var_7, var_5, var_8)
    assert var_9 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 42
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 42



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = bool(not (False or False or False))
    assert var_0 is True



# Parsed testcases at query #7
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.get_in(var_2, var_5)
    assert var_6 == 1



# Parsed testcases at query #8
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = [var_0, var_1, var_7]
    var_9 = 0
    var_10 = module_0.get_in(var_8, var_6, var_9)
    assert var_10 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'd'
    var_10 = [var_7, var_8, var_9]
    var_11 = True
    var_12 = module_0.get_in(var_10, var_6, no_default=var_11)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 10
    var_7 = [var_0, var_6]
    var_8 = 0
    var_9 = module_0.get_in(var_7, var_5, var_8)
    assert var_9 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'a'
    var_7 = 10
    var_8 = [var_6, var_7]
    var_9 = True
    var_10 = module_0.get_in(var_8, var_5, no_default=var_9)
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'd'
    var_8 = [var_0, var_1, var_7]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True



