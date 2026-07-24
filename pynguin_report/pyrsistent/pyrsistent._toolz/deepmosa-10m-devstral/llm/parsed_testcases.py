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
    var_1 = [var_0]
    var_2 = None
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True

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



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #3
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    assert var_4 is False



# Parsed testcases at query #4
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'key'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = True
    var_5 = module_0.get_in(var_2, var_3, no_default=var_4)
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    assert var_4 is False



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = bool(not False)
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
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 5
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
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None

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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = [var_0, var_1, var_3, var_4]
    var_11 = module_0.get_in(var_10, var_9)
    assert var_11 == 3

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None



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
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None



# Parsed testcases at query #3
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    assert var_4 is None



# Parsed testcases at query #4
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_0, var_1, var_2]
    var_9 = module_0.get_in(var_8, var_7)
    assert var_9 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'c'
    var_9 = [var_0, var_8]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'c'
    var_9 = [var_0, var_8]
    var_10 = 42
    var_11 = module_0.get_in(var_9, var_7, var_10)
    assert var_11 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'a'
    var_9 = 'c'
    var_10 = [var_8, var_9]
    var_11 = True
    var_12 = module_0.get_in(var_10, var_7, no_default=var_11)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 10
    var_9 = [var_0, var_1, var_8]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 10
    var_11 = [var_8, var_9, var_10]
    var_12 = True
    var_13 = module_0.get_in(var_11, var_7, no_default=var_12)
    var_14 = bool(False)
    assert var_14 is True

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
    var_0 = 42
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = True
    var_4 = module_0.get_in(var_2, var_0, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    assert var_4 is None



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
    var_4 = 'c'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = [var_0, var_1, var_3, var_4]
    var_11 = module_0.get_in(var_10, var_9)
    assert var_11 == 3

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = module_0.get_in(var_4, var_2)
    assert var_5 is None



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #8
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



