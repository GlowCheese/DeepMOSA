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
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1]
    assert var_7 == 2
    var_8 = var_5[3]
    assert var_8 == 4

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = var_5[-1]
    assert var_6 == 4
    var_7 = var_5[-2]
    assert var_7 == 3
    var_8 = var_5[-4]
    assert var_8 == 1

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
    var_7 = [var_1, var_2, var_3]
    var_8 = module_0.plist(var_7)
    var_9 = var_6[1:4]
    var_10 = bool(var_6[1:4] == var_8)
    assert var_10 is True
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.plist(var_11)
    var_13 = var_6[2:]
    var_14 = bool(var_6[2:] == var_12)
    assert var_14 is True
    var_15 = [var_0, var_1, var_2]
    var_16 = module_0.plist(var_15)
    var_17 = var_6[:3]
    var_18 = bool(var_6[:3] == var_16)
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




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = var_4[var_0:var_1]
    var_6 = [var_1]
    var_7 = module_0.plist(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.plist(var_5)
    var_7 = 0
    var_8 = var_6[var_7:var_3:var_1]
    var_9 = [var_0, var_2]
    var_10 = module_0.plist(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = var_4[:var_1]
    var_6 = [var_0, var_1]
    var_7 = module_0.plist(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_6 = None
    var_7 = slice(var_0, var_1, var_6)
    var_8 = var_5[var_7]
    var_9 = [var_1]
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
    var_7 = slice(var_6, var_6, var_0)
    var_8 = var_5[var_7]
    var_9 = [var_0, var_1, var_2, var_3]
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
    var_7 = slice(var_0, var_6, var_1)
    var_8 = var_5[var_7]
    var_9 = [var_1, var_3]
    var_10 = module_0.plist(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_plist_single_element. Retrieved 6/8 statements.
# Partially parsed test_plist_multiple_elements. Retrieved 8/14 statements.
# Partially parsed test_plist_string_iterable. Retrieved 7/13 statements.
# Partially parsed test_plist_tuple_iterable. Retrieved 8/14 statements.


import pyrsistent._plist as module_0

def test_case_0():
    var_0 = module_0.plist()
    var_1 = []
    var_2 = module_0.plist(var_1)
    var_3 = []
    var_4 = True
    var_5 = module_0.plist(var_3, var_4)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = [var_0]
    var_4 = True
    var_5 = module_0.plist(var_3, var_4)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = True
    var_7 = module_0.plist(var_5, var_6)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.plist(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = True
    var_6 = module_0.plist(var_0, var_5)

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.plist(var_3)
    var_5 = (var_0, var_1, var_2)
    var_6 = True
    var_7 = module_0.plist(var_5, var_6)



# Parsed testcases at query #2
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = var_5[0]
    assert var_6 == 1
    var_7 = var_5[1]
    assert var_7 == 2
    var_8 = var_5[2]
    assert var_8 == 3
    var_9 = var_5[3]
    assert var_9 == 4

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = var_5[-1]
    assert var_6 == 4
    var_7 = var_5[-2]
    assert var_7 == 3
    var_8 = var_5[-3]
    assert var_8 == 2
    var_9 = var_5[-4]
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
    var_7 = [var_1, var_2, var_3]
    var_8 = module_0.plist(var_7)
    var_9 = var_6[1:4]
    var_10 = bool(var_6[1:4] == var_8)
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

# Partially parsed test_split_empty_list. Retrieved 1/3 statements.
# Partially parsed test_split_single_element_list_at_0. Retrieved 4/5 statements.
# Partially parsed test_split_single_element_list_at_1. Retrieved 3/4 statements.
# Partially parsed test_split_two_elements_list_at_1. Retrieved 8/9 statements.
# Partially parsed test_split_three_elements_list_at_2. Retrieved 9/10 statements.
# Partially parsed test_split_three_elements_list_at_0. Retrieved 6/7 statements.
# Partially parsed test_split_three_elements_list_at_3. Retrieved 5/6 statements.
# Partially parsed test_split_four_elements_list_at_2. Retrieved 10/11 statements.


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
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = [var_0]
    var_5 = module_0.plist(var_4)
    var_6 = [var_1]
    var_7 = module_0.plist(var_6)

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_getitem_slice_start_not_none_stop_not_none_step_1. Retrieved 3/4 statements.
# Partially parsed test_getitem_slice_start_none_stop_none_step_none. Retrieved 2/3 statements.
# Partially parsed test_getitem_slice_start_not_none_stop_none_step_not_1. Retrieved 4/5 statements.
# Partially parsed test_getitem_slice_start_not_none_stop_not_none_step_not_1. Retrieved 3/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = slice(var_1, var_2, var_1)
    var_4 = bool(not (var_3.start is not None and var_3.stop is None and (var_3.step is None or var_3.step == 1)))
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = slice(var_1, var_1, var_1)
    var_3 = bool(not (var_2.start is not None and var_2.stop is None and (var_2.step is None or var_2.step == 1)))
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = None
    var_3 = 2
    var_4 = slice(var_1, var_2, var_3)
    var_5 = bool(not (var_4.start is not None and var_4.stop is None and (var_4.step is None or var_4.step == 1)))
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = slice(var_1, var_2, var_2)
    var_4 = bool(not (var_3.start is not None and var_3.stop is None and (var_3.step is None or var_3.step == 1)))
    assert var_4 is True



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
    var_6 = slice(var_0, var_2, var_0)
    var_7 = bool(not (var_6.start is not None and var_6.stop is None and (var_6.step is None or var_6.step == 1)))
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = var_4[var_0:var_1]

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = var_4[var_0::var_1]

import pyrsistent._plist as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = var_4[:var_1]



# Parsed testcases at query #7
#--------------------------




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
    var_7 = slice(var_0, var_2, var_6)
    var_8 = var_5[var_7]
    var_9 = [var_1, var_2]
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
    var_6 = slice(var_0, var_2, var_1)
    var_7 = var_5[var_6]
    var_8 = [var_1]
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
    var_7 = slice(var_6, var_2, var_6)
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



