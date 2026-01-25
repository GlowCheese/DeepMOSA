####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.plist(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 4
    var_6 = [var_0, var_1, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = bool(not var_4 == var_7)
    assert var_8 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_0.plist(var_5)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = bool(not var_4 == var_5)
    assert var_6 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[0]
    assert var_7 == 1
    var_8 = var_6[2]
    assert var_8 == 3
    var_9 = var_6[-1]
    assert var_9 == 5
    var_10 = var_6[-3]
    assert var_10 == 3

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 3
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = -4
    var_9 = var_4[var_8]
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = [var_1, var_2]
    var_8 = module_0.plist(var_7)
    var_9 = var_6[1:3]
    var_10 = bool(var_6[1:3] == var_8)
    assert var_10 is True
    var_11 = [var_1, var_2, var_3, var_4]
    var_12 = module_0.plist(var_11)
    var_13 = var_6[1:]
    var_14 = bool(var_6[1:] == var_12)
    assert var_14 is True
    var_15 = [var_0, var_1, var_2]
    var_16 = module_0.plist(var_15)
    var_17 = var_6[:3]
    var_18 = bool(var_6[:3] == var_16)
    assert var_18 is True
    var_19 = [var_4, var_3, var_2, var_1, var_0]
    var_20 = module_0.plist(var_19)
    var_21 = var_6[::-1]
    var_22 = bool(var_6[::-1] == var_20)
    assert var_22 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 'invalid'
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_constructor.




# Parsed testcases at query #4
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = tuple(var_5)[var_0:var_2:var_1]
    var_7 = module_0.plist(var_6)
    var_8 = var_5[1:3:2]
    var_9 = bool(var_5[1:3:2] == var_7)
    assert var_9 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = tuple(var_5)[var_0:var_2]
    var_7 = module_0.plist(var_6)
    var_8 = var_5[1:3]
    var_9 = bool(var_5[1:3] == var_7)
    assert var_9 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = tuple(var_5)[:var_2]
    var_7 = module_0.plist(var_6)
    var_8 = var_5[:3]
    var_9 = bool(var_5[:3] == var_7)
    assert var_9 is True



# Parsed testcases at query #5
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1, var_2, var_3]
    var_7 = module_0.plist(var_6)
    var_8 = var_5[None:]
    var_9 = bool(var_5[None:] == var_7)
    assert var_9 is True
    var_10 = [var_0, var_1]
    var_11 = module_0.plist(var_10)
    var_12 = var_5[:2]
    var_13 = bool(var_5[:2] == var_11)
    assert var_13 is True
    var_14 = [var_0, var_2]
    var_15 = module_0.plist(var_14)
    var_16 = var_5[::2]
    var_17 = bool(var_5[::2] == var_15)
    assert var_17 is True
    var_18 = [var_1]
    var_19 = module_0.plist(var_18)
    var_20 = var_5[1:3:2]
    var_21 = bool(var_5[1:3:2] == var_19)
    assert var_21 is True



# Parsed testcases at query #6
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = slice(var_0, var_2, var_1)
    var_7 = var_5[var_6]
    var_8 = [var_1, var_3]
    var_9 = module_0.plist(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = slice(var_0, var_2)
    var_7 = var_5[var_6]
    var_8 = [var_1, var_2]
    var_9 = module_0.plist(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = None
    var_7 = slice(var_0, var_6, var_1)
    var_8 = var_5[var_7]
    var_9 = [var_1, var_3]
    var_10 = module_0.plist(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = None
    var_7 = slice(var_6, var_2, var_1)
    var_8 = var_5[var_7]
    var_9 = [var_0, var_2]
    var_10 = module_0.plist(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = None
    var_7 = slice(var_0, var_6)
    var_8 = var_5[var_7]
    var_9 = [var_1, var_2, var_3]
    var_10 = module_0.plist(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = None
    var_7 = slice(var_6, var_2)
    var_8 = var_5[var_7]
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.plist(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = None
    var_7 = slice(var_6, var_6, var_1)
    var_8 = var_5[var_7]
    var_9 = [var_0, var_2]
    var_10 = module_0.plist(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = None
    var_6 = slice(var_0, var_5)
    var_7 = var_6.start
    var_8 = bool(var_6.start is not None)
    assert var_8 is True
    var_9 = var_6.stop
    assert var_9 is None
    var_10 = bool(var_6.step is None or var_6.step == 1)
    assert var_10 is True



# Parsed testcases at query #8
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = [var_1, var_2, var_3, var_4]
    var_8 = module_0.plist(var_7)
    var_9 = var_6[1:]
    var_10 = bool(var_6[1:] == var_8)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = [var_1, var_2, var_3, var_4]
    var_8 = module_0.plist(var_7)
    var_9 = var_6[1::1]
    var_10 = bool(var_6[1::1] == var_8)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = [var_2, var_3, var_4]
    var_8 = module_0.plist(var_7)
    var_9 = var_6[2:]
    var_10 = bool(var_6[2:] == var_8)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = [var_2, var_3, var_4]
    var_8 = module_0.plist(var_7)
    var_9 = var_6[-3:]
    var_10 = bool(var_6[-3:] == var_8)
    assert var_10 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_split_empty_list. Retrieved 4/5 statements.
# Partially parsed test_split_single_element_list. Retrieved 7/8 statements.
# Partially parsed test_split_at_beginning. Retrieved 9/10 statements.
# Partially parsed test_split_at_end. Retrieved 8/9 statements.
# Partially parsed test_split_in_middle. Retrieved 10/11 statements.
# Partially parsed test_split_with_index_out_of_range. Retrieved 9/10 statements.
# Partially parsed test_split_with_negative_index. Retrieved 10/11 statements.


import pyrsistent._plist as module_0

def test_case_0():
    var_0 = module_0.plist()
    var_1 = 0
    var_2 = module_0.plist()
    var_3 = module_0.plist()

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 0
    var_4 = module_0.plist()
    var_5 = [var_0]
    var_6 = module_0.plist(var_5)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 0
    var_6 = module_0.plist()
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.plist(var_7)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.plist(var_5)
    var_7 = module_0.plist()

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1]
    var_7 = module_0.plist(var_6)
    var_8 = [var_2, var_3]
    var_9 = module_0.plist(var_8)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 5
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.plist(var_6)
    var_8 = module_0.plist()

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = -1
    var_6 = [var_0, var_1]
    var_7 = module_0.plist(var_6)
    var_8 = [var_2]
    var_9 = module_0.plist(var_8)



# Parsed testcases at query #2
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[0]
    assert var_7 == 1
    var_8 = var_6[2]
    assert var_8 == 3
    var_9 = var_6[-1]
    assert var_9 == 5
    var_10 = var_6[-3]
    assert var_10 == 3

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 3
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = -4
    var_9 = var_4[var_8]
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = [var_1, var_2]
    var_8 = module_0.plist(var_7)
    var_9 = var_6[1:3]
    var_10 = bool(var_6[1:3] == var_8)
    assert var_10 is True
    var_11 = [var_0, var_1, var_2]
    var_12 = module_0.plist(var_11)
    var_13 = var_6[:3]
    var_14 = bool(var_6[:3] == var_12)
    assert var_14 is True
    var_15 = [var_3, var_4]
    var_16 = module_0.plist(var_15)
    var_17 = var_6[3:]
    var_18 = bool(var_6[3:] == var_16)
    assert var_18 is True
    var_19 = [var_0, var_2, var_4]
    var_20 = module_0.plist(var_19)
    var_21 = var_6[::2]
    var_22 = bool(var_6[::2] == var_20)
    assert var_22 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 'invalid'
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_split_returns_original_list_and_empty_list_when_index_exceeds_length. Retrieved 6/7 statements.
# Partially parsed test_split_returns_original_list_and_empty_list_when_index_equals_length. Retrieved 5/6 statements.
# Partially parsed test_split_returns_original_list_and_empty_list_for_empty_list. Retrieved 3/4 statements.


import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 5

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.plist(var_0)
    var_2 = 0



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_constructor.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_remove_element_exists. Retrieved 8/9 statements.
# Partially parsed test_remove_element_not_exists. Retrieved 7/9 statements.
# Partially parsed test_remove_first_element. Retrieved 8/9 statements.
# Partially parsed test_remove_last_element. Retrieved 8/9 statements.
# Partially parsed test_remove_only_element. Retrieved 4/5 statements.
# Partially parsed test_remove_multiple_elements. Retrieved 7/8 statements.
# Partially parsed test_remove_empty_list. Retrieved 2/4 statements.


import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1, var_3]
    var_7 = module_0.plist(var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 5
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_1, var_2, var_3]
    var_7 = module_0.plist(var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.plist(var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_0, var_2, var_0]
    var_4 = module_0.plist(var_3)
    var_5 = [var_1, var_0, var_2, var_0]
    var_6 = module_0.plist(var_5)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_split_empty_list. Retrieved 1/3 statements.
# Partially parsed test_split_single_element_list_at_0. Retrieved 4/5 statements.
# Partially parsed test_split_single_element_list_at_1. Retrieved 3/4 statements.
# Partially parsed test_split_multi_element_list_at_0. Retrieved 6/7 statements.
# Partially parsed test_split_multi_element_list_at_middle. Retrieved 10/11 statements.
# Partially parsed test_split_multi_element_list_at_end. Retrieved 5/6 statements.
# Partially parsed test_split_multi_element_list_past_end. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 0

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 0

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 0

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1]
    var_7 = module_0.plist(var_6)
    var_8 = [var_2, var_3]
    var_9 = module_0.plist(var_8)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 4



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_split_empty_list. Retrieved 7/8 statements.
# Partially parsed test_split_single_element_list. Retrieved 8/9 statements.
# Partially parsed test_split_list_at_beginning. Retrieved 10/11 statements.
# Partially parsed test_split_list_at_middle. Retrieved 10/11 statements.
# Partially parsed test_split_list_at_end. Retrieved 9/10 statements.
# Partially parsed test_split_list_with_negative_index. Retrieved 6/8 statements.
# Partially parsed test_split_list_with_index_out_of_range. Retrieved 6/8 statements.


import pyrsistent._plist as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.plist(var_0)
    var_2 = 0
    var_3 = []
    var_4 = module_0.plist(var_3)
    var_5 = []
    var_6 = module_0.plist(var_5)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 0
    var_4 = []
    var_5 = module_0.plist(var_4)
    var_6 = [var_0]
    var_7 = module_0.plist(var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 0
    var_6 = []
    var_7 = module_0.plist(var_6)
    var_8 = [var_0, var_1, var_2]
    var_9 = module_0.plist(var_8)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1]
    var_7 = module_0.plist(var_6)
    var_8 = [var_2, var_3]
    var_9 = module_0.plist(var_8)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.plist(var_5)
    var_7 = []
    var_8 = module_0.plist(var_7)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = -1
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 4
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_split_empty_list. Retrieved 5/6 statements.
# Partially parsed test_split_single_element_list. Retrieved 8/9 statements.
# Partially parsed test_split_at_head. Retrieved 10/11 statements.
# Partially parsed test_split_at_tail. Retrieved 10/11 statements.
# Partially parsed test_split_middle. Retrieved 11/12 statements.
# Partially parsed test_split_index_greater_than_length. Retrieved 10/11 statements.
# Partially parsed test_split_index_negative. Retrieved 6/8 statements.
# Partially parsed test_split_empty_list_at_non_zero. Retrieved 5/6 statements.
# Partially parsed test_split_list_with_duplicates. Retrieved 10/11 statements.


import pyrsistent._plist as module_0

def test_case_0():
    var_0 = module_0.plist()
    var_1 = 0
    var_2 = module_0.plist()
    var_3 = module_0.plist()
    var_4 = (var_2, var_3)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 0
    var_4 = module_0.plist()
    var_5 = [var_0]
    var_6 = module_0.plist(var_5)
    var_7 = (var_4, var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 0
    var_6 = module_0.plist()
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.plist(var_7)
    var_9 = (var_6, var_8)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_0.plist(var_5)
    var_7 = [var_2]
    var_8 = module_0.plist(var_7)
    var_9 = (var_6, var_8)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1]
    var_7 = module_0.plist(var_6)
    var_8 = [var_2, var_3]
    var_9 = module_0.plist(var_8)
    var_10 = (var_7, var_9)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 5
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.plist(var_6)
    var_8 = module_0.plist()
    var_9 = (var_7, var_8)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = -1
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = module_0.plist()
    var_3 = module_0.plist()
    var_4 = (var_2, var_3)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_0.plist(var_5)
    var_7 = [var_1, var_2]
    var_8 = module_0.plist(var_7)
    var_9 = (var_6, var_8)



# Parsed testcases at query #9
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[0]
    assert var_7 == 1
    var_8 = var_6[2]
    assert var_8 == 3
    var_9 = var_6[4]
    assert var_9 == 5

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[-1]
    assert var_7 == 5
    var_8 = var_6[-3]
    assert var_8 == 3
    var_9 = var_6[-5]
    assert var_9 == 1

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 3
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = -4
    var_9 = var_4[var_8]
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = [var_1, var_2]
    var_8 = module_0.plist(var_7)
    var_9 = var_6[1:3]
    var_10 = bool(var_6[1:3] == var_8)
    assert var_10 is True
    var_11 = [var_1, var_2, var_3, var_4]
    var_12 = module_0.plist(var_11)
    var_13 = var_6[1:]
    var_14 = bool(var_6[1:] == var_12)
    assert var_14 is True
    var_15 = [var_0, var_1, var_2]
    var_16 = module_0.plist(var_15)
    var_17 = var_6[:3]
    var_18 = bool(var_6[:3] == var_16)
    assert var_18 is True
    var_19 = [var_4, var_3, var_2, var_1, var_0]
    var_20 = module_0.plist(var_19)
    var_21 = var_6[::-1]
    var_22 = bool(var_6[::-1] == var_20)
    assert var_22 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 'invalid'
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[var_0:var_2]
    var_8 = [var_1, var_2]
    var_9 = module_0.plist(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[var_0:var_3:var_1]
    var_8 = [var_1, var_3]
    var_9 = module_0.plist(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[:var_2]
    var_8 = [var_0, var_1, var_2]
    var_9 = module_0.plist(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[::var_1]
    var_8 = [var_0, var_2, var_4]
    var_9 = module_0.plist(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[var_0::var_1]
    var_8 = [var_1, var_3]
    var_9 = module_0.plist(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_constructor_empty.
# Partially parsed test_constructor_with_elements. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = len(var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_remove_element_from_plist. Retrieved 8/9 statements.
# Partially parsed test_remove_first_element. Retrieved 8/9 statements.
# Partially parsed test_remove_last_element. Retrieved 8/9 statements.
# Partially parsed test_remove_non_existing_element_raises_value_error. Retrieved 7/9 statements.
# Partially parsed test_remove_duplicate_element_removes_first_occurrence. Retrieved 7/8 statements.


import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1, var_3]
    var_7 = module_0.plist(var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_1, var_2, var_3]
    var_7 = module_0.plist(var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.plist(var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 5
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_0, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_1, var_0, var_2]
    var_6 = module_0.plist(var_5)



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_plistbase_constructor_empty.
# Partially parsed test_plistbase_constructor_single_element. Retrieved 1/6 statements.
# Partially parsed test_plistbase_constructor_multiple_elements. Retrieved 3/10 statements.
# Partially parsed test_plistbase_constructor_mcons. Retrieved 4/9 statements.
# Partially parsed test_plistbase_constructor_equality. Retrieved 5/9 statements.
# Partially parsed test_plistbase_constructor_inequality. Retrieved 6/10 statements.
# Partially parsed test_plistbase_constructor_hash. Retrieved 5/11 statements.
# Partially parsed test_plistbase_constructor_less_than. Retrieved 6/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 1

def test_case_0():
    var_0 = []
    var_1 = 3
    var_2 = 2
    var_3 = 1

def test_case_0():
    var_0 = []
    var_1 = 3
    var_2 = 2
    var_3 = 1
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = 4
    var_7 = [var_1, var_2, var_6]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = 4
    var_7 = [var_1, var_2, var_6]



# Parsed testcases at query #14
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[var_1:]
    var_8 = [var_2, var_3, var_4]
    var_9 = module_0.plist(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = var_6[var_0:]
    var_8 = [var_1, var_2, var_3, var_4]
    var_9 = module_0.plist(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = None
    var_8 = var_6[var_2:var_7:var_0]
    var_9 = [var_3, var_4]
    var_10 = module_0.plist(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_split_empty_list. Retrieved 1/3 statements.
# Partially parsed test_split_single_element_list. Retrieved 4/5 statements.
# Partially parsed test_split_at_beginning. Retrieved 6/7 statements.
# Partially parsed test_split_at_end. Retrieved 5/6 statements.
# Partially parsed test_split_middle. Retrieved 10/11 statements.
# Partially parsed test_split_out_of_bounds. Retrieved 6/7 statements.
# Partially parsed test_split_negative_index. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 0

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 0

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1]
    var_7 = module_0.plist(var_6)
    var_8 = [var_2, var_3]
    var_9 = module_0.plist(var_8)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 5

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = -1
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_constructor.




# Parsed testcases at query #17
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 10
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 10
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_remove_first_element. Retrieved 7/8 statements.
# Partially parsed test_remove_middle_element. Retrieved 7/8 statements.
# Partially parsed test_remove_last_element. Retrieved 7/8 statements.
# Partially parsed test_remove_first_occurrence_of_element. Retrieved 7/8 statements.
# Partially parsed test_raise_error_if_element_not_found. Retrieved 6/8 statements.


import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_1, var_2]
    var_6 = module_0.plist(var_5)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_0, var_2]
    var_6 = module_0.plist(var_5)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_0.plist(var_5)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_0, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_1, var_0, var_2]
    var_6 = module_0.plist(var_5)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 4
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_remove_element_from_plist. Retrieved 8/9 statements.
# Partially parsed test_remove_first_element. Retrieved 8/9 statements.
# Partially parsed test_remove_last_element. Retrieved 8/9 statements.
# Partially parsed test_remove_non_existing_element_raises_value_error. Retrieved 7/9 statements.
# Partially parsed test_remove_duplicate_element_removes_first_occurrence. Retrieved 7/8 statements.


import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1, var_3]
    var_7 = module_0.plist(var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_1, var_2, var_3]
    var_7 = module_0.plist(var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.plist(var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 5
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_0, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_1, var_0, var_2]
    var_6 = module_0.plist(var_5)



