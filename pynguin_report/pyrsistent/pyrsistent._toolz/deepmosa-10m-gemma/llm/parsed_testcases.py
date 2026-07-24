####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = module_0.get_in(var_5, var_4)
    assert var_6 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 20
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 1
    var_7 = [var_6, var_0]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 == 20

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_3, var_4]
    var_6 = 'missing'
    var_7 = module_0.get_in(var_5, var_2, var_6)
    assert var_7 == 'missing'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'z'
    var_4 = [var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 'error'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'error'

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
    var_3 = 42
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = 0
    var_10 = [var_0, var_9, var_1, var_2, var_9]
    var_11 = module_0.get_in(var_10, var_8)
    assert var_11 == 42



# Parsed testcases at query #2
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 'missing'
    var_6 = False
    var_7 = module_0.get_in(var_4, var_2, var_5, var_6)
    assert var_7 == 'missing'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = 'missing'
    var_6 = False
    var_7 = module_0.get_in(var_4, var_2, var_5, var_6)
    assert var_7 == 'missing'

def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'non_existent'
    var_4 = [var_3]
    var_5 = 'fallback'
    var_6 = False
    var_7 = module_0.get_in(var_4, var_2, var_5, var_6)
    assert var_7 == 'fallback'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = 'fallback'
    var_7 = False
    var_8 = module_0.get_in(var_5, var_3, var_6, var_7)
    assert var_8 == 'fallback'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 123
    var_1 = 'key'
    var_2 = [var_1]
    var_3 = 'fallback'
    var_4 = False
    var_5 = module_0.get_in(var_2, var_0, var_3, var_4)
    assert var_5 == 'fallback'



# Parsed testcases at query #4
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'non'
    var_1 = 'existent'
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 'fallback'
    var_7 = False
    var_8 = module_0.get_in(var_2, var_5, var_6, var_7)
    assert var_8 == 'fallback'



# Parsed testcases at query #5
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = module_0.get_in(var_5, var_4)
    assert var_6 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = [var_1, var_0]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 'missing'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'missing'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = 'KeyError not raised'
    var_9 = AssertionError(var_8)

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = 'IndexError not raised'
    var_9 = AssertionError(var_8)

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'sub'
    var_4 = [var_0, var_3]
    var_5 = None
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == {'a': 1})
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = [var_5]
    var_7 = {var_0: var_6}
    var_8 = 0
    var_9 = [var_0, var_8, var_1, var_2]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 == 42



# Parsed testcases at query #6
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = module_0.get_in(var_5, var_4)
    assert var_6 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = [var_1, var_0]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 'missing'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'missing'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'sub_key'
    var_4 = [var_0, var_3]
    var_5 = None
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 is None

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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = 10
    var_4 = 20
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = [var_7]
    var_9 = {var_0: var_8}
    var_10 = 0
    var_11 = 1
    var_12 = [var_0, var_10, var_1, var_2, var_11]
    var_13 = module_0.get_in(var_12, var_9)
    assert var_13 == 20



# Parsed testcases at query #7
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = module_0.get_in(var_5, var_4)
    assert var_6 == 1



# Parsed testcases at query #8
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = module_0.get_in(var_5, var_4)
    assert var_6 == 1



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




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
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_1, var_2]
    var_4 = 40
    var_5 = [var_0, var_3, var_4]
    var_6 = 1
    var_7 = 0
    var_8 = [var_6, var_7]
    var_9 = module_0.get_in(var_8, var_5)
    assert var_9 == 20

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'users'
    var_1 = 'id'
    var_2 = 'name'
    var_3 = 1
    var_4 = 'Alice'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 2
    var_7 = 'Bob'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_9}
    var_11 = [var_0, var_3, var_2]
    var_12 = module_0.get_in(var_11, var_10)
    assert var_12 == 'Bob'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 'missing'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'missing'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = None
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = 'KeyError not raised'
    var_8 = AssertionError(var_7)

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = 'IndexError not raised'
    var_8 = AssertionError(var_7)

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'sub_key'
    var_4 = [var_0, var_3]
    var_5 = 'fallback'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'fallback'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = None
    var_6 = False
    var_7 = module_0.get_in(var_4, var_2, var_5, var_6)
    var_8 = bool(var_7 == var_5)
    assert var_8 is True



# Parsed testcases at query #3
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 'missing'
    var_6 = False
    var_7 = module_0.get_in(var_4, var_2, var_5, var_6)
    assert var_7 == 'missing'



# Parsed testcases at query #4
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = module_0.get_in(var_5, var_4)
    assert var_6 == 10

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = [var_1, var_0]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 0
    var_2 = 'y'
    var_3 = 100
    var_4 = 200
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = 1
    var_10 = [var_0, var_9, var_2, var_1]
    var_11 = module_0.get_in(var_10, var_8)
    assert var_11 == 100

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 'missing'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'missing'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = None
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = 'error'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'error'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = 'KeyError not raised'
    var_8 = AssertionError(var_7)

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = 'IndexError not raised'
    var_8 = AssertionError(var_7)

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
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.get_in(var_3, var_2)
    assert var_4 == 'Alice'



# Parsed testcases at query #5
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 10
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = None
    var_7 = False
    var_8 = module_0.get_in(var_2, var_5, var_6, var_7)
    assert var_8 == 10



# Parsed testcases at query #6
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'non_existent_key'
    var_1 = [var_0]
    var_2 = 'existing_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'fallback'
    var_6 = False
    var_7 = module_0.get_in(var_1, var_4, var_5, var_6)
    assert var_7 == 'fallback'



# Parsed testcases at query #7
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = module_0.get_in(var_5, var_4)
    assert var_6 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 20
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = 1
    var_7 = [var_6, var_0]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 == 10

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'users'
    var_1 = 'id'
    var_2 = 'name'
    var_3 = 1
    var_4 = 'Alice'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 2
    var_7 = 'Bob'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_9}
    var_11 = [var_0, var_3, var_2]
    var_12 = module_0.get_in(var_11, var_10)
    assert var_12 == 'Bob'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.get_in(var_3, var_2)
    assert var_4 == 'Alice'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 'missing'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'missing'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = None
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'sub_key'
    var_4 = [var_0, var_3]
    var_5 = 'error'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'error'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = 'KeyError not raised'
    var_8 = AssertionError(var_7)

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = 'IndexError not raised'
    var_8 = AssertionError(var_7)

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == {'a': 1})
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = None
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 is None



# Parsed testcases at query #8
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = module_0.get_in(var_5, var_4)
    assert var_6 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 10
    var_2 = 20
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = [var_4]
    var_6 = 1
    var_7 = [var_0, var_6]
    var_8 = 0
    var_9 = var_5[var_8]
    var_10 = module_0.get_in(var_7, var_9)
    assert var_10 == 20

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 'missing'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'missing'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'sub_key'
    var_4 = [var_0, var_3]
    var_5 = None
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.get_in(var_3, var_2)
    assert var_4 == 'Alice'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True



