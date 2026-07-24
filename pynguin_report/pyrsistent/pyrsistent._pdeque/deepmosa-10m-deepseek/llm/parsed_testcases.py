####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_valid_inputs. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_raises_type_error_for_non_integer_maxlen. Retrieved 6/8 statements.
# Partially parsed test_constructor_raises_value_error_for_negative_maxlen. Retrieved 6/8 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_only_left_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_only_right_list. Retrieved 7/8 statements.


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()
    var_4 = 1
    var_5 = 'invalid'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()
    var_4 = 1
    var_5 = -1
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = module_0.plist()
    var_6 = 3
    var_7 = [var_4, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 3
    var_7 = [var_0, var_5, var_6]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_rotate_positive_steps. Retrieved 9/10 statements.
# Partially parsed test_rotate_negative_steps. Retrieved 10/11 statements.
# Partially parsed test_rotate_zero_steps. Retrieved 8/9 statements.
# Partially parsed test_rotate_steps_greater_than_length. Retrieved 8/9 statements.
# Partially parsed test_rotate_negative_steps_greater_than_length. Retrieved 8/9 statements.
# Partially parsed test_rotate_single_element. Retrieved 4/5 statements.
# Partially parsed test_rotate_empty. Retrieved 3/4 statements.
# Partially parsed test_rotate_with_maxlen. Retrieved 7/8 statements.
# Partially parsed test_rotate_negative_with_maxlen. Retrieved 8/9 statements.
# Partially parsed test_rotate_steps_equal_length. Retrieved 5/6 statements.


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_3, var_4, var_0, var_1, var_2]
    var_8 = module_0.pdeque(var_7)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = -2
    var_8 = [var_2, var_3, var_4, var_0, var_1]
    var_9 = module_0.pdeque(var_8)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = 0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 5
    var_6 = [var_1, var_2, var_0]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = -5
    var_6 = [var_2, var_0, var_1]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0.pdeque(var_1)
    var_3 = 3


def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = 10


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3, var_2)
    var_5 = [var_2, var_0, var_1]
    var_6 = module_0.pdeque(var_5, var_2)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3, var_2)
    var_5 = -1
    var_6 = [var_1, var_2, var_0]
    var_7 = module_0.pdeque(var_6, var_2)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_popleft_empty_deque. Retrieved 4/6 statements.
# Partially parsed test_popleft_single_element. Retrieved 5/7 statements.
# Partially parsed test_popleft_multiple_elements. Retrieved 7/9 statements.
# Partially parsed test_popleft_with_count. Retrieved 9/11 statements.
# Partially parsed test_popleft_count_exceeds_length. Retrieved 8/10 statements.
# Partially parsed test_popleft_negative_count. Retrieved 8/10 statements.
# Partially parsed test_popleft_on_bounded_deque. Retrieved 7/9 statements.
# Partially parsed test_popleft_on_bounded_deque_with_count. Retrieved 8/10 statements.
# Partially parsed test_popleft_preserves_maxlen. Retrieved 8/9 statements.
# Partially parsed test_popleft_identity_when_count_zero. Retrieved 6/8 statements.



def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = []
    var_3 = module_0.pdeque(var_2)


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.pdeque(var_1)
    var_3 = []
    var_4 = module_0.pdeque(var_3)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_1, var_2]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_3, var_4]
    var_8 = module_0.pdeque(var_7)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 5
    var_6 = []
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = -2
    var_6 = [var_2]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3, var_2)
    var_5 = [var_1, var_2]
    var_6 = module_0.pdeque(var_5, var_2)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.pdeque(var_4, var_3)
    var_6 = [var_2, var_3]
    var_7 = module_0.pdeque(var_6, var_3)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = module_0.pdeque(var_3, var_4)
    var_6 = [var_2]
    var_7 = module_0.pdeque(var_6, var_4)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 0



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_valid_maxlen. Retrieved 10/11 statements.
# Partially parsed test_constructor_without_maxlen. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_with_non_integer_maxlen_raises_typeerror. Retrieved 8/10 statements.
# Partially parsed test_constructor_with_negative_maxlen_raises_valueerror. Retrieved 8/10 statements.


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = 'invalid'
    var_8 = [var_2, var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = -1
    var_8 = [var_2, var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_with_valid_arguments. Retrieved 9/10 statements.
# Partially parsed test_constructor_without_maxlen. Retrieved 6/7 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_maxlen_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_non_integer_maxlen_raises_type_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_only_left_list. Retrieved 6/7 statements.
# Partially parsed test_constructor_with_only_right_list. Retrieved 6/7 statements.
# Partially parsed test_constructor_with_maxlen_equal_to_length. Retrieved 8/9 statements.
# Partially parsed test_constructor_with_maxlen_greater_than_length. Retrieved 7/8 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 5
    var_9 = [var_3, var_7, var_5]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_2, var_5, var_3]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = module_0.plist()
    var_6 = [var_4, var_5, var_2]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_0, var_5, var_3]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = [var_3, var_7, var_5]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 10
    var_7 = [var_2, var_5, var_3]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_constructor_with_valid_inputs. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_invalid_maxlen_type. Retrieved 10/12 statements.
# Partially parsed test_constructor_with_negative_maxlen. Retrieved 10/12 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_only_left_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_only_right_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_length_mismatch. Retrieved 9/10 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 0
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 'invalid'
    var_10 = [var_3, var_7, var_8, var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(True)
    assert var_12 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = -1
    var_10 = [var_3, var_7, var_8, var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(True)
    assert var_12 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = module_0.plist()
    var_6 = 3
    var_7 = [var_4, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 3
    var_7 = [var_0, var_5, var_6]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 10
    var_9 = [var_3, var_7, var_8]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_rotate_positive_steps. Retrieved 7/8 statements.
# Partially parsed test_rotate_negative_steps. Retrieved 8/9 statements.
# Partially parsed test_rotate_zero_steps. Retrieved 6/7 statements.
# Partially parsed test_rotate_more_than_length. Retrieved 8/9 statements.
# Partially parsed test_rotate_negative_more_than_length. Retrieved 8/9 statements.
# Partially parsed test_rotate_single_element. Retrieved 4/5 statements.
# Partially parsed test_rotate_empty. Retrieved 3/4 statements.
# Partially parsed test_rotate_with_maxlen. Retrieved 7/8 statements.
# Partially parsed test_rotate_negative_with_maxlen. Retrieved 8/9 statements.


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_2, var_0, var_1]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = -2
    var_6 = [var_2, var_0, var_1]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 5
    var_6 = [var_1, var_2, var_0]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = -4
    var_6 = [var_1, var_2, var_0]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0.pdeque(var_1)
    var_3 = 10


def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = 3


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3, var_2)
    var_5 = [var_2, var_0, var_1]
    var_6 = module_0.pdeque(var_5, var_2)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3, var_2)
    var_5 = -1
    var_6 = [var_1, var_2, var_0]
    var_7 = module_0.pdeque(var_6, var_2)



# Parsed testcases at query #5
#--------------------------





def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 0
    var_6 = var_4[var_5]
    var_7 = 10
    var_8 = bool(var_6 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = -1
    var_6 = var_4[var_5]
    var_7 = 30
    var_8 = bool(var_6 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 5
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = -5
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_6[:]
    var_8 = [var_0, var_1, var_2, var_3, var_4]
    var_9 = module_0.pdeque(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_6[var_1:]
    var_8 = [var_2, var_3, var_4]
    var_9 = module_0.pdeque(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_6[:var_2]
    var_8 = [var_0, var_1, var_2]
    var_9 = module_0.pdeque(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_6[var_0:var_3]
    var_8 = [var_1, var_2, var_3]
    var_9 = module_0.pdeque(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = -3
    var_8 = var_6[var_7:]
    var_9 = [var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = -2
    var_8 = var_6[:var_7]
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_6[::var_1]
    var_8 = [var_0, var_2, var_4]
    var_9 = module_0.pdeque(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = -1
    var_8 = var_6[::var_7]
    var_9 = [var_4, var_3, var_2, var_1, var_0]
    var_10 = module_0.pdeque(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = 7
    var_7 = 8
    var_8 = 9
    var_9 = 10
    var_10 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = module_0.pdeque(var_10)
    var_12 = var_11[var_0:var_7:var_2]
    var_13 = [var_1, var_4, var_7]
    var_14 = module_0.pdeque(var_13)
    var_15 = bool(var_12 == var_14)
    assert var_15 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_6[var_2:var_1]
    var_8 = []
    var_9 = module_0.pdeque(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5, var_2)
    var_7 = var_6[:]
    var_8 = [var_2, var_3, var_4]
    var_9 = module_0.pdeque(var_8, var_2)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 10
    var_6 = var_4[var_5:]
    var_7 = []
    var_8 = module_0.pdeque(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 10
    var_6 = var_4[:var_5]
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = -10
    var_6 = var_4[var_5:]
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = -10
    var_6 = var_4[:var_5]
    var_7 = []
    var_8 = module_0.pdeque(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 'invalid'
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = None
    var_8 = var_6[var_0:var_3:var_7]
    var_9 = [var_1, var_2, var_3]
    var_10 = module_0.pdeque(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_remove_existing_element_from_left. Retrieved 6/7 statements.
# Partially parsed test_remove_existing_element_from_right. Retrieved 7/8 statements.
# Partially parsed test_remove_only_element. Retrieved 5/6 statements.
# Partially parsed test_remove_first_occurrence. Retrieved 7/8 statements.
# Partially parsed test_remove_element_not_present_raises_value_error. Retrieved 6/8 statements.
# Partially parsed test_remove_from_empty_deque_raises_value_error. Retrieved 3/5 statements.
# Partially parsed test_remove_maintains_maxlen. Retrieved 7/8 statements.
# Partially parsed test_remove_element_with_multiple_occurrences. Retrieved 7/8 statements.
# Partially parsed test_remove_after_append. Retrieved 7/9 statements.
# Partially parsed test_remove_after_appendleft. Retrieved 7/9 statements.



def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.pdeque(var_2)
    var_4 = [var_1, var_0]
    var_5 = module_0.pdeque(var_4)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = module_0.pdeque(var_1)
    var_3 = []
    var_4 = module_0.pdeque(var_3)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_0, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_1, var_0, var_2]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 4
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True


def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3, var_2)
    var_5 = [var_0, var_2]
    var_6 = module_0.pdeque(var_5, var_2)


def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = [var_0, var_1, var_0, var_2, var_0]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_1, var_0, var_2, var_0]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.pdeque(var_2)
    var_4 = 3
    var_5 = [var_0, var_4]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.pdeque(var_2)
    var_4 = 0
    var_5 = [var_4, var_1]
    var_6 = module_0.pdeque(var_5)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_popleft_empty_deque. Retrieved 4/6 statements.
# Partially parsed test_popleft_single_element. Retrieved 5/7 statements.
# Partially parsed test_popleft_multiple_elements. Retrieved 7/9 statements.
# Partially parsed test_popleft_with_count. Retrieved 8/10 statements.
# Partially parsed test_popleft_all_elements. Retrieved 7/9 statements.
# Partially parsed test_popleft_more_than_length. Retrieved 7/9 statements.
# Partially parsed test_popleft_negative_count. Retrieved 8/10 statements.
# Partially parsed test_popleft_with_maxlen. Retrieved 7/9 statements.
# Partially parsed test_popleft_preserves_maxlen. Retrieved 5/6 statements.
# Partially parsed test_popleft_on_deque_with_single_list_side. Retrieved 7/10 statements.



def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = []
    var_3 = module_0.pdeque(var_2)


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.pdeque(var_1)
    var_3 = []
    var_4 = module_0.pdeque(var_3)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_1, var_2]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.pdeque(var_4)
    var_6 = [var_2, var_3]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = []
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.pdeque(var_2)
    var_4 = 5
    var_5 = []
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = -2
    var_6 = [var_0]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3, var_2)
    var_5 = [var_1, var_2]
    var_6 = module_0.pdeque(var_5, var_2)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = module_0.pdeque(var_2, var_3)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_2]
    var_6 = module_0.pdeque(var_5)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test___new___creates_instance_with_valid_arguments. Retrieved 10/12 statements.
# Partially parsed test___new___creates_instance_without_maxlen. Retrieved 7/9 statements.
# Partially parsed test___new___raises_type_error_for_non_integral_maxlen. Retrieved 4/7 statements.
# Partially parsed test___new___raises_value_error_for_negative_maxlen. Retrieved 4/7 statements.
# Partially parsed test___new___accepts_zero_maxlen. Retrieved 4/6 statements.
# Partially parsed test___new___accepts_positive_maxlen. Retrieved 8/10 statements.


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = [var_2, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = 10
    var_8 = [var_2, var_5, var_6, var_7]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_constructor_creates_pdeque_with_correct_attributes. Retrieved 10/11 statements.
# Partially parsed test_constructor_without_maxlen_sets_maxlen_to_none. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_maxlen_as_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_with_negative_maxlen_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_non_integer_maxlen_raises_type_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_maxlen_as_positive_integer. Retrieved 8/9 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = [var_2, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = 3
    var_8 = [var_2, var_5, var_6, var_7]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_constructor_creates_pdeque_with_correct_attributes. Retrieved 9/10 statements.
# Partially parsed test_constructor_without_maxlen. Retrieved 6/7 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 3/4 statements.
# Partially parsed test_constructor_raises_type_error_for_non_integer_maxlen. Retrieved 4/6 statements.
# Partially parsed test_constructor_raises_value_error_for_negative_maxlen. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_maxlen_positive. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 5/6 statements.
# Partially parsed test_constructor_length_zero_with_nonempty_lists. Retrieved 7/8 statements.
# Partially parsed test_constructor_assigns_weakref_slot. Retrieved 4/6 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 5
    var_9 = [var_3, var_7, var_5]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_2, var_5, var_3]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 10
    var_7 = [var_2, var_5, var_3]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist()
    var_5 = module_0.plist()


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 0
    var_7 = [var_2, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = '__weakref__'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_constructor_creates_valid_pdeque. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen. Retrieved 7/8 statements.
# Partially parsed test_constructor_maxlen_non_negative. Retrieved 3/4 statements.
# Partially parsed test_constructor_raises_type_error_for_invalid_maxlen. Retrieved 4/6 statements.
# Partially parsed test_constructor_raises_value_error_for_negative_maxlen. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 4/5 statements.
# Partially parsed test_constructor_length_zero_with_nonempty_lists. Retrieved 8/9 statements.
# Partially parsed test_constructor_preserves_weakref_slot. Retrieved 5/7 statements.
# Partially parsed test_constructor_maxlen_none_implies_unbounded. Retrieved 9/10 statements.
# Partially parsed test_constructor_sets_correct_length. Retrieved 10/12 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = None
    var_9 = [var_3, var_7, var_5, var_8]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 5
    var_7 = [var_2, var_5, var_3, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2, var_2]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = None
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 0
    var_7 = None
    var_8 = [var_2, var_5, var_6, var_7]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = None
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = '__weakref__'


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = None
    var_9 = [var_3, var_7, var_5, var_8]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 4
    var_6 = 5
    var_7 = [var_5, var_6]
    var_8 = module_0.plist(var_7)
    var_9 = None
    var_10 = [var_4, var_8, var_6, var_9]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_constructor_with_valid_arguments. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_with_non_integral_maxlen_raises_typeerror. Retrieved 6/8 statements.
# Partially parsed test_constructor_with_negative_maxlen_raises_valueerror. Retrieved 6/8 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_only_left_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_only_right_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_length_mismatch_but_still_constructs. Retrieved 8/9 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()
    var_4 = 1
    var_5 = 'invalid'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()
    var_4 = 1
    var_5 = -1
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = module_0.plist()
    var_6 = 3
    var_7 = [var_4, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 3
    var_7 = [var_0, var_5, var_6]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = [var_4]
    var_6 = module_0.plist(var_5)
    var_7 = 10
    var_8 = [var_3, var_6, var_7]



# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_constructor_with_valid_maxlen. Retrieved 10/11 statements.
# Partially parsed test_constructor_without_maxlen. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_raises_type_error_for_non_integer_maxlen. Retrieved 4/6 statements.
# Partially parsed test_constructor_raises_value_error_for_negative_maxlen. Retrieved 4/6 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_eq_same_elements_different_internal_structure. Retrieved 6/9 statements.


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_4 == var_6
    assert var_7 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 4
    var_6 = [var_0, var_1, var_5]
    var_7 = module_0.pdeque(var_6)
    var_8 = var_4 == var_7
    assert var_8 is False


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_4 == var_6
    assert var_7 is False


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = module_0.pdeque(var_3, var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = 10
    var_8 = module_0.pdeque(var_6, var_7)
    var_9 = var_5 == var_8
    assert var_9 is True


def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = []
    var_3 = module_0.pdeque(var_2)
    var_4 = var_1 == var_3
    assert var_4 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = var_4 == var_5


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 4


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_4 == var_6
    var_8 = hash(var_4)
    var_9 = hash(var_6)
    var_10 = var_8 == var_9
    var_11 = bool(var_7 == var_10)
    assert var_11 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_remove_elem_not_found_raises_value_error. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 4
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_constructor_with_valid_inputs. Retrieved 9/10 statements.
# Partially parsed test_constructor_without_maxlen. Retrieved 6/7 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_maxlen_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_non_integer_maxlen_raises_type_error. Retrieved 4/6 statements.


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 5
    var_9 = [var_3, var_7, var_5]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_2, var_5, var_3]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------

# Partially parsed test_eq_assertion_fails_when_lengths_differ_but_tuples_equal. Retrieved 14/28 statements.


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = iter(var_4)
    var_6 = [var_1, var_2, var_3]
    var_7 = module_0.pdeque(var_6)
    var_8 = [var_1, var_2, var_3]
    var_9 = iter(var_8)
    var_10 = []
    var_11 = iter(var_10)
    var_12 = []
    var_13 = iter(var_12)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_remove_elem_not_in_left_list_but_in_right_list. Retrieved 11/16 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = []
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_2]
    var_10 = module_0.pdeque(var_9)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_constructor_creates_valid_pdeque. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_raises_type_error_for_non_integer_maxlen. Retrieved 6/8 statements.
# Partially parsed test_constructor_raises_value_error_for_negative_maxlen. Retrieved 6/8 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_only_left_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_only_right_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_length_mismatch_handled. Retrieved 8/9 statements.
# Partially parsed test_constructor_maxlen_none_is_default. Retrieved 7/8 statements.


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()
    var_4 = 1
    var_5 = 'invalid'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()
    var_4 = 1
    var_5 = -1
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = module_0.plist()
    var_6 = 3
    var_7 = [var_4, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 3
    var_7 = [var_0, var_5, var_6]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = [var_4]
    var_6 = module_0.plist(var_5)
    var_7 = 3
    var_8 = [var_3, var_6, var_7]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = [var_2, var_5, var_6]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test___new___creates_instance_with_correct_attributes. Retrieved 10/11 statements.
# Partially parsed test___new___creates_instance_without_maxlen. Retrieved 9/10 statements.
# Partially parsed test___new___raises_type_error_for_non_integer_maxlen. Retrieved 10/12 statements.
# Partially parsed test___new___raises_value_error_for_negative_maxlen. Retrieved 10/12 statements.
# Partially parsed test___new___accepts_zero_maxlen. Retrieved 10/11 statements.
# Partially parsed test___new___accepts_positive_maxlen. Retrieved 10/11 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 'invalid'
    var_10 = [var_3, var_7, var_8, var_9]
    var_11 = bool(False)
    assert var_11 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = -1
    var_10 = [var_3, var_7, var_8, var_9]
    var_11 = bool(False)
    assert var_11 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 0
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 10
    var_10 = [var_3, var_7, var_8, var_9]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_constructor_with_valid_inputs. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_with_non_integer_maxlen_raises_typeerror. Retrieved 6/8 statements.
# Partially parsed test_constructor_with_negative_maxlen_raises_valueerror. Retrieved 6/8 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_only_left_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_only_right_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_length_mismatch_but_still_creates. Retrieved 8/9 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()
    var_4 = 1
    var_5 = 'invalid'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()
    var_4 = 1
    var_5 = -1
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = module_0.plist()
    var_6 = 3
    var_7 = [var_4, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 3
    var_7 = [var_0, var_5, var_6]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = [var_4]
    var_6 = module_0.plist(var_5)
    var_7 = 10
    var_8 = [var_3, var_6, var_7]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_constructor_creates_pdeque_with_correct_attributes. Retrieved 10/11 statements.
# Partially parsed test_constructor_without_maxlen. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_raises_type_error_for_non_integer_maxlen. Retrieved 4/6 statements.
# Partially parsed test_constructor_raises_value_error_for_negative_maxlen. Retrieved 4/6 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = [var_2, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------

# Partially parsed test_remove_elem_not_found_in_left_list_but_found_in_right_list. Retrieved 11/16 statements.


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = []
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_2]
    var_10 = module_0.pdeque(var_9)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_constructor_creates_pdeque_with_correct_attributes. Retrieved 9/10 statements.
# Partially parsed test_constructor_without_maxlen. Retrieved 6/7 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 3/4 statements.
# Partially parsed test_constructor_raises_type_error_for_non_integer_maxlen. Retrieved 4/6 statements.
# Partially parsed test_constructor_raises_value_error_for_negative_maxlen. Retrieved 4/6 statements.


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 5
    var_9 = [var_3, var_7, var_5]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = [var_2, var_5, var_3]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_remove_existing_element_from_left. Retrieved 6/7 statements.
# Partially parsed test_remove_existing_element_from_right. Retrieved 7/8 statements.
# Partially parsed test_remove_only_element. Retrieved 5/6 statements.
# Partially parsed test_remove_element_not_present. Retrieved 6/8 statements.
# Partially parsed test_remove_from_empty_deque. Retrieved 3/5 statements.
# Partially parsed test_remove_first_occurrence. Retrieved 7/8 statements.
# Partially parsed test_remove_with_maxlen. Retrieved 7/8 statements.
# Partially parsed test_remove_preserves_maxlen. Retrieved 8/9 statements.
# Partially parsed test_remove_element_at_right_end. Retrieved 8/9 statements.
# Partially parsed test_remove_element_at_left_end. Retrieved 8/9 statements.


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.pdeque(var_2)
    var_4 = [var_1, var_0]
    var_5 = module_0.pdeque(var_4)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = module_0.pdeque(var_1)
    var_3 = []
    var_4 = module_0.pdeque(var_3)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 4
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_0, var_2, var_0]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_1, var_0, var_2, var_0]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3, var_2)
    var_5 = [var_0, var_2]
    var_6 = module_0.pdeque(var_5, var_2)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.pdeque(var_4, var_3)
    var_6 = [var_0, var_1, var_3]
    var_7 = module_0.pdeque(var_6, var_3)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.pdeque(var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.pdeque(var_4)
    var_6 = [var_1, var_2, var_3]
    var_7 = module_0.pdeque(var_6)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_constructor_with_valid_inputs. Retrieved 10/11 statements.
# Partially parsed test_constructor_without_maxlen. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_maxlen_none. Retrieved 8/9 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_with_non_integral_maxlen_raises_typeerror. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_maxlen_raises_valueerror. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_only_left_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_only_right_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_maxlen_equal_to_length. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_maxlen_greater_than_length. Retrieved 8/9 statements.
# Partially parsed test_constructor_with_maxlen_less_than_length. Retrieved 11/12 statements.


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = [var_2, var_5, var_6]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = None
    var_8 = [var_2, var_5, var_6, var_7]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = module_0.plist()
    var_6 = 3
    var_7 = [var_4, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 3
    var_7 = [var_0, var_5, var_6]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 4
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = 10
    var_8 = [var_2, var_5, var_6, var_7]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 4
    var_6 = 5
    var_7 = [var_5, var_6]
    var_8 = module_0.plist(var_7)
    var_9 = 5
    var_10 = 3
    var_11 = [var_4, var_8, var_9, var_10]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_assertion_failure_on_unequal_lengths. Retrieved 8/9 statements.


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_4 == var_6



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_eq_sanity_check_fails. Retrieved 8/11 statements.


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = 2
    var_4 = [var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 2
    var_7 = [var_2, var_5, var_6]
    var_8 = var_6 + var_0
    var_9 = [var_2, var_5, var_8]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_constructor_with_valid_arguments. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_invalid_maxlen_type. Retrieved 10/12 statements.
# Partially parsed test_constructor_with_negative_maxlen. Retrieved 10/12 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_only_left_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_only_right_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_length_mismatch. Retrieved 9/10 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 0
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 'invalid'
    var_10 = [var_3, var_7, var_8, var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(True)
    assert var_12 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = -1
    var_10 = [var_3, var_7, var_8, var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(True)
    assert var_12 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = module_0.plist()
    var_6 = 3
    var_7 = [var_4, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 3
    var_7 = [var_0, var_5, var_6]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 10
    var_9 = [var_3, var_7, var_8]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_remove_existing_element_from_left. Retrieved 6/7 statements.
# Partially parsed test_remove_existing_element_from_right. Retrieved 7/8 statements.
# Partially parsed test_remove_only_element. Retrieved 5/6 statements.
# Partially parsed test_remove_element_from_empty_deque_raises_error. Retrieved 3/5 statements.
# Partially parsed test_remove_non_existing_element_raises_error. Retrieved 6/8 statements.
# Partially parsed test_remove_first_occurrence_from_left. Retrieved 7/8 statements.
# Partially parsed test_remove_with_duplicates_in_right_list. Retrieved 8/9 statements.
# Partially parsed test_remove_maintains_maxlen. Retrieved 7/8 statements.
# Partially parsed test_remove_element_from_single_element_bounded_queue. Retrieved 6/7 statements.
# Partially parsed test_remove_preserves_hash. Retrieved 8/10 statements.


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.pdeque(var_2)
    var_4 = [var_1, var_0]
    var_5 = module_0.pdeque(var_4)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = module_0.pdeque(var_1)
    var_3 = []
    var_4 = module_0.pdeque(var_3)


def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 4
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_0, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_1, var_0, var_2]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_1, var_3]
    var_5 = module_0.pdeque(var_4)
    var_6 = [var_0, var_2, var_1, var_3]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3, var_2)
    var_5 = [var_0, var_2]
    var_6 = module_0.pdeque(var_5, var_2)


def test_case_0():
    var_0 = 7
    var_1 = [var_0]
    var_2 = 1
    var_3 = module_0.pdeque(var_1, var_2)
    var_4 = []
    var_5 = module_0.pdeque(var_4, var_2)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_2]
    var_6 = module_0.pdeque(var_5)
    var_7 = hash(var_6)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_constructor_with_valid_inputs. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_with_invalid_maxlen_type. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_maxlen. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_only_left_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_only_right_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_length_mismatch. Retrieved 8/9 statements.


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 'invalid'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = -1
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = module_0.plist()
    var_6 = 3
    var_7 = [var_4, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 3
    var_7 = [var_0, var_5, var_6]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = [var_4]
    var_6 = module_0.plist(var_5)
    var_7 = 10
    var_8 = [var_3, var_6, var_7]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_valid_inputs. Retrieved 9/10 statements.
# Partially parsed test_constructor_with_maxlen. Retrieved 10/11 statements.
# Partially parsed test_constructor_with_maxlen_zero. Retrieved 4/5 statements.
# Partially parsed test_constructor_with_invalid_maxlen_type. Retrieved 6/8 statements.
# Partially parsed test_constructor_with_negative_maxlen. Retrieved 6/8 statements.
# Partially parsed test_constructor_with_empty_lists. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_only_left_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_with_only_right_list. Retrieved 7/8 statements.
# Partially parsed test_constructor_length_mismatch. Retrieved 8/9 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = [var_3, var_7, var_8]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = [var_3, var_7, var_8, var_9]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = 0
    var_4 = [var_0, var_1, var_2, var_3]


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()
    var_4 = 1
    var_5 = 'invalid'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.plist(var_1)
    var_3 = module_0.plist()
    var_4 = 1
    var_5 = -1
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.plist()
    var_1 = module_0.plist()
    var_2 = 0
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = module_0.plist()
    var_6 = 3
    var_7 = [var_4, var_5, var_6]


def test_case_0():
    var_0 = module_0.plist()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.plist(var_4)
    var_6 = 3
    var_7 = [var_0, var_5, var_6]


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = [var_4]
    var_6 = module_0.plist(var_5)
    var_7 = 10
    var_8 = [var_3, var_6, var_7]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pop_empty_deque. Retrieved 4/6 statements.
# Partially parsed test_pop_single_element. Retrieved 5/7 statements.
# Partially parsed test_pop_multiple_elements. Retrieved 7/9 statements.
# Partially parsed test_pop_with_count. Retrieved 8/10 statements.
# Partially parsed test_pop_all_elements. Retrieved 7/9 statements.
# Partially parsed test_pop_more_than_length. Retrieved 7/9 statements.
# Partially parsed test_pop_negative_count. Retrieved 8/10 statements.
# Partially parsed test_pop_on_bounded_deque. Retrieved 7/9 statements.
# Partially parsed test_pop_on_bounded_deque_with_count. Retrieved 8/10 statements.
# Partially parsed test_pop_maintains_maxlen. Retrieved 8/9 statements.
# Partially parsed test_pop_zero_count. Retrieved 8/10 statements.


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = []
    var_3 = module_0.pdeque(var_2)


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.pdeque(var_1)
    var_3 = []
    var_4 = module_0.pdeque(var_3)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.pdeque(var_4)
    var_6 = [var_0, var_1]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = []
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.pdeque(var_2)
    var_4 = 5
    var_5 = []
    var_6 = module_0.pdeque(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = -2
    var_6 = [var_2]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3, var_2)
    var_5 = [var_0, var_1]
    var_6 = module_0.pdeque(var_5, var_2)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.pdeque(var_4, var_3)
    var_6 = [var_0, var_1]
    var_7 = module_0.pdeque(var_6, var_3)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = module_0.pdeque(var_3, var_4)
    var_6 = [var_0]
    var_7 = module_0.pdeque(var_6, var_4)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 0
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pdeque(var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_eq_after_operations. Retrieved 8/11 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_4 == var_6
    assert var_7 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = 4
    var_6 = [var_0, var_1, var_5]
    var_7 = module_0.pdeque(var_6)
    var_8 = var_4 == var_7
    assert var_8 is False


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_0.pdeque(var_5)
    var_7 = var_4 == var_6
    assert var_7 is False


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = module_0.pdeque(var_3, var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = 10
    var_8 = module_0.pdeque(var_6, var_7)
    var_9 = var_5 == var_8
    assert var_9 is True


def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = []
    var_3 = module_0.pdeque(var_2)
    var_4 = var_1 == var_3
    assert var_4 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = var_4 == var_5


def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = module_0.pdeque(var_1)
    var_3 = [var_0]
    var_4 = module_0.pdeque(var_3)
    var_5 = var_2 == var_4
    assert var_5 is True


def test_case_0():
    var_0 = 1000
    var_1 = range(var_0)
    var_2 = module_0.pdeque(var_1)
    var_3 = range(var_0)
    var_4 = module_0.pdeque(var_3)
    var_5 = var_2 == var_4
    assert var_5 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.pdeque(var_2)
    var_4 = 3
    var_5 = 0
    var_6 = [var_5, var_0, var_1, var_4]
    var_7 = module_0.pdeque(var_6)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = var_4 == var_4
    assert var_5 is True



