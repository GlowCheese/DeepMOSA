####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_choice_with_list_and_length_one. Retrieved 8/9 statements.
# Partially parsed test_choice_with_string_and_length. Retrieved 5/8 statements.
# Partially parsed test_choice_with_tuple_and_length. Retrieved 8/11 statements.
# Partially parsed test_choice_with_unique_true. Retrieved 8/11 statements.


import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0(items=var_4)
    var_6 = bool(var_5 in var_4)
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
    var_9 = bool(var_6[0] in var_4)
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
    var_4 = bool(True)
    assert var_4 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = []
    var_2 = var_0(items=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

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
    var_8 = bool(True)
    assert var_8 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'aa'
    var_2 = 3
    var_3 = True
    var_4 = var_0(items=var_1, length=var_2, unique=var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_choice_with_list_and_length_one. Retrieved 8/9 statements.
# Partially parsed test_choice_with_string_and_length. Retrieved 5/8 statements.
# Partially parsed test_choice_with_tuple_and_length. Retrieved 8/11 statements.
# Partially parsed test_choice_with_unique_elements. Retrieved 8/11 statements.


import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0(items=var_4)
    var_6 = bool(var_5 in var_4)
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
    var_9 = bool(var_6[0] in var_4)
    assert var_9 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'abc'
    var_2 = 2
    var_3 = var_0(items=var_1, length=var_2)
    var_4 = len(var_3)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

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
    var_8 = bool(var_7 == var_5)
    assert var_8 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'aabbbccccddddd'
    var_2 = 4
    var_3 = True
    var_4 = var_0(items=var_1, length=var_2, unique=var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True
    var_7 = set(var_4)
    var_8 = len(var_7)
    var_9 = bool(var_8 == var_2)
    assert var_9 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 123
    var_2 = var_0(items=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = []
    var_2 = var_0(items=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

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
    var_8 = bool(True)
    assert var_8 is True

import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_1, var_2]
    var_4 = 3
    var_5 = True
    var_6 = var_0(items=var_3, length=var_4, unique=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



