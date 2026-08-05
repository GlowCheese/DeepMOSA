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
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_1, var_0]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 1

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = 'found'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = [var_5]
    var_7 = {var_0: var_6}
    var_8 = 0
    var_9 = [var_0, var_8, var_1, var_2]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 == 'found'

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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'z'
    var_4 = [var_3]
    var_5 = 0
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 0



# Parsed testcases at query #2
#--------------------------




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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = 20
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = 1
    var_9 = [var_0, var_8, var_1]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 == 20

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
    var_5 = 'missing'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'missing'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 5
    var_6 = [var_0, var_5]
    var_7 = None
    var_8 = module_0.get_in(var_6, var_4, var_7)
    assert var_8 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'sub'
    var_4 = [var_0, var_3]
    var_5 = 'fallback'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'fallback'

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
    var_0 = 'a'
    var_1 = 1
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'a'
    var_5 = 5
    var_6 = [var_4, var_5]
    var_7 = True
    var_8 = module_0.get_in(var_6, var_3, no_default=var_7)
    var_9 = 'IndexError not raised'
    var_10 = AssertionError(var_9)

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
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.get_in(var_3, var_2)
    assert var_4 is None



# Parsed testcases at query #3
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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
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



# Parsed testcases at query #5
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



# Parsed testcases at query #6
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
    var_0 = 0
    var_1 = 10
    var_2 = 20
    var_3 = [var_1, var_2]
    var_4 = 30
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

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)

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
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_1, var_0]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 1

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
    var_7 = 'KeyError not raised'
    var_8 = AssertionError(var_7)

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
    var_5 = 'fallback'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'fallback'

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

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'non_existent'
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'fallback'
    var_6 = False
    var_7 = module_0.get_in(var_1, var_4, var_5, var_6)
    assert var_7 == 'fallback'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




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
    var_0 = 'a'
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 1
    var_7 = [var_0, var_6]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 == 20

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

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = True
    var_6 = module_0.get_in(var_4, var_2, no_default=var_5)

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
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
    var_5 = bool(var_4 == {'a': 1})
    assert var_5 is True

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



# Parsed testcases at query #2
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
    var_7 = 'KeyError not raised'
    var_8 = AssertionError(var_7)

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
    var_3 = 'sub-key'
    var_4 = [var_0, var_3]
    var_5 = 'fallback'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'fallback'

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



# Parsed testcases at query #3
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 1



# Parsed testcases at query #4
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'non_existent_key'
    var_1 = [var_0]
    var_2 = 'existing_key'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'fallback'
    var_6 = False
    var_7 = module_0.get_in(var_1, var_4, var_5, var_6)
    assert var_7 == 'fallback'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'fallback'
    var_7 = False
    var_8 = module_0.get_in(var_1, var_5, var_6, var_7)
    assert var_8 == 'fallback'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = [var_0]
    var_2 = None
    var_3 = 'fallback'
    var_4 = False
    var_5 = module_0.get_in(var_1, var_2, var_3, var_4)
    assert var_5 == 'fallback'



# Parsed testcases at query #5
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 1



# Parsed testcases at query #6
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
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_1, var_0]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 1

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
    var_4 = [var_0, var_3]
    var_5 = 'missing'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'missing'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'x'
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
    var_3 = 'a'
    var_4 = 'sub_key'
    var_5 = [var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_2, no_default=var_6)
    var_8 = bool(True)
    assert var_8 is True
    var_9 = 'TypeError not raised'
    var_10 = AssertionError(var_9)

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
    var_5 = None
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 is None



# Parsed testcases at query #8
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
    var_5 = None
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
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

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == {'a': 1})
    assert var_5 is True



