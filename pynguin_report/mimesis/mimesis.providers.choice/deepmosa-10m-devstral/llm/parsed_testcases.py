####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_choice_list_with_length. Retrieved 8/9 statements.
# Partially parsed test_choice_string_with_length. Retrieved 5/8 statements.
# Partially parsed test_choice_tuple_with_length. Retrieved 9/12 statements.
# Partially parsed test_choice_unique_elements. Retrieved 8/11 statements.


import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0(items=var_4)
    var_6 = bool(var_5 in ['a', 'b', 'c'])
    assert var_6 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0]
    var_9 = bool(var_6[0] in ['a', 'b', 'c'])
    assert var_9 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'abc'
    var_2 = 2
    var_3 = var_0(items=var_1, length=var_2)
    var_4 = len(var_3)
    assert var_4 == 2

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = 5
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 5
    var_8 = (var_1, var_2, var_3)

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'aabbbccccddddd'
    var_2 = 4
    var_3 = True
    var_4 = var_0(items=var_1, length=var_2, unique=var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = set(var_4)
    var_7 = len(var_6)
    assert var_7 == 4

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 123
    var_2 = var_0(items=var_1)
    var_3 = bool(False)
    assert var_3 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = []
    var_2 = var_0(items=var_1)
    var_3 = bool(False)
    assert var_3 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = -1
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = bool(False)
    assert var_7 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 5
    var_6 = True
    var_7 = var_0(items=var_4, length=var_5, unique=var_6)
    var_8 = bool(False)
    assert var_8 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_choice_call_with_list_and_length. Retrieved 9/12 statements.
# Partially parsed test_choice_call_with_tuple_and_length. Retrieved 9/12 statements.
# Partially parsed test_choice_call_with_string_and_length. Retrieved 5/8 statements.
# Partially parsed test_choice_call_with_unique_elements. Retrieved 14/17 statements.
# Partially parsed test_choice_call_with_unique_string_elements. Retrieved 8/11 statements.


import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = []
    var_2 = var_0(items=var_1)

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 123
    var_2 = var_0(items=var_1)

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = -1
    var_6 = var_0(items=var_4, length=var_5)

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 5
    var_6 = True
    var_7 = var_0(items=var_4, length=var_5, unique=var_6)

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 0
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = bool(var_6 in ['a', 'b', 'c'])
    assert var_7 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 2
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = [var_1, var_2, var_3]

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = 2
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = (var_1, var_2, var_3)

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'abc'
    var_2 = 2
    var_3 = var_0(items=var_1, length=var_2)
    var_4 = len(var_3)
    assert var_4 == 2

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 'e'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 3
    var_8 = True
    var_9 = var_0(items=var_6, length=var_7, unique=var_8)
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = set(var_9)
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = [var_1, var_2, var_3, var_4, var_5]

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'abcde'
    var_2 = 3
    var_3 = True
    var_4 = var_0(items=var_1, length=var_2, unique=var_3)
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = set(var_4)
    var_7 = len(var_6)
    assert var_7 == 3



