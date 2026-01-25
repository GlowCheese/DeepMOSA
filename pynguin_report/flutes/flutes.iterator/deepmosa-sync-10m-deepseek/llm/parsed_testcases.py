####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = bool(not var_1.exhausted)
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = iter(var_2)
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = bool(var_4.list == [])
    assert var_7 is True
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_constructor_no_args.
# Partially parsed test_constructor_one_arg. Retrieved 1/2 statements.
# Partially parsed test_constructor_two_args. Retrieved 2/3 statements.
# Partially parsed test_constructor_three_args. Retrieved 3/4 statements.
# Partially parsed test_constructor_four_args. Retrieved 4/6 statements.
# Partially parsed test_constructor_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_zero_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_start_equal_stop. Retrieved 1/2 statements.
# Partially parsed test_constructor_start_greater_than_stop_positive_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_start_less_than_stop_negative_step. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 7
    var_1 = [var_0, var_0]

def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = -1
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #4
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list is var_4)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list == var_1)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.MapList(var_1, var_3)
    var_5 = var_4.func
    var_6 = bool(var_4.func is var_1)
    assert var_6 is True
    var_7 = var_4.list
    var_8 = bool(var_4.list == var_3)
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x * x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.MapList(var_0, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_0)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_scanl_with_initial_value. Retrieved 10/12 statements.
# Partially parsed test_scanl_without_initial_value. Retrieved 10/12 statements.
# Partially parsed test_scanl_empty_iterable_with_initial. Retrieved 4/6 statements.
# Partially parsed test_scanl_single_element_without_initial. Retrieved 4/6 statements.
# Partially parsed test_scanl_single_element_with_initial. Retrieved 6/8 statements.
# Partially parsed test_scanl_too_many_arguments. Retrieved 5/8 statements.
# Partially parsed test_scanl_with_initial_as_none. Retrieved 7/9 statements.
# Partially parsed test_scanl_using_operator_mul. Retrieved 8/12 statements.
# Partially parsed test_scanl_using_operator_sub. Retrieved 7/11 statements.


def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = 0
    var_7 = [var_6]
    var_8 = 6
    var_9 = 10
    var_10 = [var_6, var_1, var_3, var_8, var_9]

def test_case_0():
    var_0 = lambda s, x: x + s
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = []
    var_7 = 'ba'
    var_8 = 'cba'
    var_9 = 'dcba'
    var_10 = [var_1, var_7, var_8, var_9]

def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = []
    var_2 = 5
    var_3 = [var_2]
    var_4 = [var_2]

def test_case_0():
    var_0 = []
    var_1 = iter(var_0)
    var_2 = next(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 10
    var_2 = [var_1]
    var_3 = []
    var_4 = [var_1]

def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 10
    var_2 = [var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = 15
    var_6 = [var_3, var_5]

def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 0
    var_5 = [var_4, var_1]
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = None
    var_1 = lambda a, b: a if a is not var_0 else b
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_0]
    var_7 = [var_0, var_2, var_2, var_2]

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 6
    var_6 = 24
    var_7 = [var_4, var_0, var_5, var_6]

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = 2
    var_6 = [var_4, var_0, var_5, var_2]



# Parsed testcases at query #6
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = [var_0, var_9, var_10]
    var_12 = 6
    var_13 = 7
    var_14 = 8
    var_15 = [var_12, var_13, var_14]
    var_16 = 9
    var_17 = [var_16]
    var_18 = [var_8, var_11, var_15, var_17]
    var_19 = bool(var_4 == var_18)
    assert var_19 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = 4
    var_4 = [var_1, var_0, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = [var_1, var_0]
    var_8 = [var_2, var_3]
    var_9 = [var_7, var_8]
    var_10 = bool(var_6 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 6
    var_3 = 7
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = [var_1]
    var_8 = [var_2]
    var_9 = [var_3]
    var_10 = [var_7, var_8, var_9]
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'hello'
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = 'h'
    var_5 = 'e'
    var_6 = [var_4, var_5]
    var_7 = 'l'
    var_8 = [var_7, var_7]
    var_9 = 'o'
    var_10 = [var_9]
    var_11 = [var_6, var_8, var_10]
    var_12 = bool(var_3 == var_11)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = iter(var_1)
    var_3 = 2
    var_4 = module_0.chunk(var_3, var_2)
    var_5 = list(var_4)
    var_6 = 0
    var_7 = 1
    var_8 = [var_6, var_7]
    var_9 = 3
    var_10 = [var_3, var_9]
    var_11 = 4
    var_12 = [var_11]
    var_13 = [var_8, var_10, var_12]
    var_14 = bool(var_5 == var_13)
    assert var_14 is True
    var_15 = list(var_2)
    var_16 = bool(var_15 == [])
    assert var_16 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 100
    var_1 = 30
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = range(var_1)
    var_6 = list(var_5)
    var_7 = [var_6]
    var_8 = bool(var_4 == var_7)
    assert var_8 is True



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_constructor_no_args_raises_value_error.
# Partially parsed test_constructor_four_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_one_arg. Retrieved 1/2 statements.
# Partially parsed test_constructor_two_args. Retrieved 2/3 statements.
# Partially parsed test_constructor_three_args. Retrieved 3/4 statements.
# Partially parsed test_constructor_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_step_zero_raises_no_error_but_length_calculation. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_len_with_single_argument. Retrieved 2/4 statements.
# Partially parsed test_len_with_start_and_stop. Retrieved 3/5 statements.
# Partially parsed test_len_with_start_stop_and_step. Retrieved 4/6 statements.
# Partially parsed test_len_with_negative_step. Retrieved 4/6 statements.
# Partially parsed test_len_with_zero_length. Retrieved 2/4 statements.
# Partially parsed test_len_with_step_causing_zero_length. Retrieved 4/6 statements.
# Partially parsed test_len_with_large_range. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 5

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]
    var_3 = 6

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 4

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]
    var_4 = 5

def test_case_0():
    var_0 = 5
    var_1 = [var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 20
    var_3 = [var_0, var_1, var_2]
    var_4 = 0

def test_case_0():
    var_0 = 0
    var_1 = 1000
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = 143



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_take_with_generator. Retrieved 3/6 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.take(var_0, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0.take(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = 3

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = iter(var_4)
    var_6 = module_0.take(var_1, var_5)
    var_7 = list(var_6)
    var_8 = module_0.take(var_1, var_5)
    var_9 = list(var_8)
    var_10 = bool(var_7 == [1, 2])
    assert var_10 is True
    var_11 = bool(var_9 == [3, 4])
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 4
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3, var_0]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2, 3, 4])
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_four_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_error_but_length_calculation. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = bool(var_4.list == [])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.exhausted
    assert var_3 is False
    var_4 = var_2.list
    var_5 = bool(var_2.list == [])
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_raises_no_error_but_length_calculation. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * x
    var_7 = module_0.MapList(var_6, var_5)
    var_8 = var_7[var_1]
    var_9 = 9
    var_10 = bool(var_8 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = 10
    var_7 = lambda x: x + var_6
    var_8 = module_0.MapList(var_7, var_5)
    var_9 = -1
    var_10 = var_8[var_9]
    var_11 = 15
    var_12 = bool(var_10 == var_11)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * var_1
    var_7 = module_0.MapList(var_6, var_5)
    var_8 = var_7[var_0:var_3]
    var_9 = 6
    var_10 = 8
    var_11 = [var_3, var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x - var_0
    var_7 = module_0.MapList(var_6, var_5)
    var_8 = var_7[:]
    var_9 = 0
    var_10 = [var_9, var_0, var_1, var_2, var_3]
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x ** var_1
    var_7 = module_0.MapList(var_6, var_5)
    var_8 = var_7[::var_1]
    var_9 = 9
    var_10 = 25
    var_11 = [var_0, var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = 10
    var_7 = lambda x: x * var_6
    var_8 = module_0.MapList(var_7, var_5)
    var_9 = 20
    var_10 = var_8[var_6:var_9]
    var_11 = []
    var_12 = bool(var_10 == var_11)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda s: s.upper()
    var_5 = module_0.MapList(var_4, var_3)
    var_6 = 1
    var_7 = var_5[var_6]
    var_8 = 'B'
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = lambda s: s * var_5
    var_7 = module_0.MapList(var_6, var_4)
    var_8 = 0
    var_9 = 3
    var_10 = var_7[var_8:var_9]
    var_11 = 'aa'
    var_12 = 'bb'
    var_13 = 'cc'
    var_14 = [var_11, var_12, var_13]
    var_15 = bool(var_10 == var_14)
    assert var_15 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x + var_0
    var_5 = module_0.MapList(var_4, var_3)
    var_6 = 5
    var_7 = var_5[var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x + var_0
    var_5 = module_0.MapList(var_4, var_3)
    var_6 = -5
    var_7 = var_5[var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_immediately. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #15
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list is var_4)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list == var_1)
    assert var_6 is True
    var_7 = len(var_2)
    assert var_7 == 0

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = (var_2, var_3, var_4)
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = var_6.func
    var_8 = bool(var_6.func is var_1)
    assert var_8 is True
    var_9 = var_6.list
    var_10 = bool(var_6.list == var_5)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.MapList(var_1, var_3)
    var_5 = var_4.func
    var_6 = bool(var_4.func is var_1)
    assert var_6 is True
    var_7 = var_4.list
    var_8 = bool(var_4.list == var_3)
    assert var_8 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_getitem_positive_index. Retrieved 4/6 statements.
# Partially parsed test_getitem_negative_index. Retrieved 4/6 statements.
# Partially parsed test_getitem_slice_with_start_stop. Retrieved 4/6 statements.
# Partially parsed test_getitem_slice_with_step. Retrieved 5/7 statements.
# Partially parsed test_getitem_slice_negative_indices. Retrieved 5/7 statements.
# Partially parsed test_getitem_slice_out_of_bounds. Retrieved 3/5 statements.
# Partially parsed test_getitem_slice_empty. Retrieved 4/6 statements.
# Partially parsed test_getitem_slice_no_start. Retrieved 3/5 statements.
# Partially parsed test_getitem_slice_no_stop. Retrieved 3/5 statements.
# Partially parsed test_getitem_slice_no_start_no_stop. Retrieved 3/5 statements.
# Partially parsed test_getitem_index_out_of_range_positive. Retrieved 4/7 statements.
# Partially parsed test_getitem_index_out_of_range_negative. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_step_one. Retrieved 2/4 statements.
# Partially parsed test_getitem_slice_with_step_one. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_negative_step_range. Retrieved 4/6 statements.
# Partially parsed test_getitem_slice_with_negative_step_range. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 4

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -3
    var_5 = -1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 5

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -10
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 3

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 1
    var_3 = 4

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]
    var_4 = 2

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 3



# Parsed testcases at query #17
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = bool(var_4.list == [])
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = 6
    var_7 = 7
    var_8 = 8
    var_9 = 9
    var_10 = [var_6, var_7, var_8, var_9]
    var_11 = bool(var_5 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x >= var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = [var_2, var_3, var_4]
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = []
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'b'
    var_1 = lambda s: s.startswith(var_0)
    var_2 = 'a'
    var_3 = 'aa'
    var_4 = 'bb'
    var_5 = 'c'
    var_6 = [var_2, var_3, var_0, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = [var_0, var_4, var_5]
    var_10 = bool(var_8 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5, var_0]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = [var_0]
    var_10 = bool(var_8 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = iter(var_5)
    var_7 = lambda x: x > var_1
    var_8 = module_0.drop_until(var_7, var_6)
    var_9 = list(var_8)
    var_10 = [var_2, var_3, var_4]
    var_11 = bool(var_9 == var_10)
    assert var_11 is True
    var_12 = list(var_6)
    var_13 = bool(var_12 == [])
    assert var_13 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_immediately. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #20
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 3
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    var_6 = list(var_5)
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = 4
    var_11 = 5
    var_12 = [var_10, var_11]
    var_13 = 7
    var_14 = 8
    var_15 = [var_13, var_14]
    var_16 = [var_9, var_12, var_15]
    var_17 = bool(var_6 == var_16)
    assert var_17 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = True
    var_3 = 3
    var_4 = 0
    var_5 = lambda x: x % var_3 == var_4
    var_6 = module_0.split_by(var_1, var_2, criterion=var_5)
    var_7 = list(var_6)
    var_8 = []
    var_9 = 2
    var_10 = [var_2, var_9]
    var_11 = 4
    var_12 = 5
    var_13 = [var_11, var_12]
    var_14 = 7
    var_15 = 8
    var_16 = [var_14, var_15]
    var_17 = []
    var_18 = [var_8, var_10, var_13, var_16, var_17]
    var_19 = bool(var_7 == var_18)
    assert var_19 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a.b.c'
    var_1 = '.'
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = [var_6]
    var_8 = 'c'
    var_9 = [var_8]
    var_10 = [var_5, var_7, var_9]
    var_11 = bool(var_3 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = '..a..b..'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = []
    var_7 = 'a'
    var_8 = [var_7]
    var_9 = []
    var_10 = 'b'
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = [var_5, var_6, var_8, var_9, var_11, var_12, var_13]
    var_15 = bool(var_4 == var_14)
    assert var_15 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = lambda x: var_4
    var_6 = module_0.split_by(var_3, criterion=var_5)
    var_7 = list(var_6)
    var_8 = [var_0, var_1, var_2]
    var_9 = [var_8]
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = module_0.split_by(var_3, separator=var_4)
    var_6 = list(var_5)
    var_7 = [var_0, var_1, var_2]
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = module_0.split_by(var_3, criterion=var_5)
    var_7 = list(var_6)
    var_8 = []
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.split_by(var_1, separator=var_0)
    var_3 = list(var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = module_0.split_by(var_3, var_4, criterion=var_6)
    var_8 = list(var_7)
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = bool(var_8 == var_13)
    assert var_14 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = True
    var_3 = module_0.split_by(var_1, var_2, separator=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = module_0.split_by(var_0, criterion=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = module_0.split_by(var_0, var_1, criterion=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.split_by(var_0, var_1, separator=var_1)
    var_3 = list(var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.split_by(var_3)
    var_5 = list(var_4)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = module_0.split_by(var_3, criterion=var_5, separator=var_4)
    var_7 = list(var_6)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = lambda x: x == var_1
    var_6 = module_0.split_by(var_4, criterion=var_5)
    var_7 = list(var_6)
    var_8 = [var_0]
    var_9 = [var_2, var_3]
    var_10 = [var_8, var_9]
    var_11 = bool(var_7 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'sep'
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_1, var_2, var_1, var_3]
    var_5 = module_0.split_by(var_4, separator=var_1)
    var_6 = list(var_5)
    var_7 = [var_0]
    var_8 = [var_2]
    var_9 = [var_3]
    var_10 = [var_7, var_8, var_9]
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 0
    var_6 = lambda x: x % var_1 == var_5
    var_7 = module_0.split_by(var_4, criterion=var_6)
    var_8 = list(var_7)
    var_9 = [var_0]
    var_10 = [var_2]
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_0, var_1, var_0, var_0, var_2, var_0]
    var_4 = module_0.split_by(var_3, separator=var_0)
    var_5 = list(var_4)
    var_6 = [var_1]
    var_7 = [var_2]
    var_8 = [var_6, var_7]
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = True
    var_6 = 0
    var_7 = lambda x: x % var_1 == var_6
    var_8 = module_0.split_by(var_4, var_5, criterion=var_7)
    var_9 = list(var_8)
    var_10 = [var_5]
    var_11 = []
    var_12 = [var_2]
    var_13 = []
    var_14 = [var_10, var_11, var_12, var_13]
    var_15 = bool(var_9 == var_14)
    assert var_15 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_0, var_1, var_0, var_0, var_2, var_0]
    var_4 = True
    var_5 = module_0.split_by(var_3, var_4, separator=var_0)
    var_6 = list(var_5)
    var_7 = []
    var_8 = []
    var_9 = [var_4]
    var_10 = []
    var_11 = []
    var_12 = [var_2]
    var_13 = []
    var_14 = [var_7, var_8, var_9, var_10, var_11, var_12, var_13]
    var_15 = bool(var_6 == var_14)
    assert var_15 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_constructor_with_generator. Retrieved 2/4 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False

def test_case_0():
    var_0 = 3
    var_1 = range(var_0)

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.list
    var_4 = bool(var_2.list == [])
    assert var_4 is True
    var_5 = var_2.exhausted
    assert var_5 is False



# Parsed testcases at query #22
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True



# Parsed testcases at query #23
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list is var_4)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list == var_1)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = (var_2, var_3, var_4)
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = var_6.func
    var_8 = bool(var_6.func is var_1)
    assert var_8 is True
    var_9 = var_6.list
    var_10 = bool(var_6.list == var_5)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x * x
    var_1 = 5
    var_2 = range(var_1)
    var_3 = module_0.MapList(var_0, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func is var_0)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = list(var_6)
    var_8 = range(var_1)
    var_9 = list(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_immediately. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_constructor_with_generator. Retrieved 2/4 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False

def test_case_0():
    var_0 = 3
    var_1 = range(var_0)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.iter
    var_9 = next(var_8)
    assert var_9 == 1

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False
    var_5 = var_1.iter
    var_6 = next(var_5)
    var_7 = bool(False)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = 7
    var_3 = [var_0, var_1, var_2]
    var_4 = iter(var_3)
    var_5 = module_0.LazyList(var_4)
    var_6 = var_5.list
    var_7 = bool(var_5.list == [])
    assert var_7 is True
    var_8 = var_5.exhausted
    assert var_8 is False
    var_9 = var_5.iter
    var_10 = next(var_9)
    assert var_10 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 300
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.iter
    var_9 = next(var_8)
    assert var_9 == 100

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False
    var_5 = var_1.iter
    var_6 = next(var_5)
    assert var_6 == 'a'



# Parsed testcases at query #27
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = 6
    var_7 = 7
    var_8 = 8
    var_9 = 9
    var_10 = [var_2, var_3, var_4, var_5, var_0, var_6, var_7, var_8, var_9]
    var_11 = module_0.drop_until(var_1, var_10)
    var_12 = list(var_11)
    var_13 = bool(var_12 == [6, 7, 8, 9])
    assert var_13 is True



# Parsed testcases at query #28
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = bool(var_4.list == [])
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list is var_4)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list == var_1)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = (var_2, var_3, var_4)
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = var_6.func
    var_8 = bool(var_6.func is var_1)
    assert var_8 is True
    var_9 = var_6.list
    var_10 = bool(var_6.list == var_5)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x * x
    var_1 = 5
    var_2 = range(var_1)
    var_3 = module_0.MapList(var_0, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func is var_0)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = list(var_6)
    var_8 = range(var_1)
    var_9 = list(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda c: ord(c)
    var_1 = 'abc'
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    assert var_5 == 'abc'

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 5
    var_2 = 6
    var_3 = 7
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.MapList(var_0, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_0)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list is var_4)
    assert var_9 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_slice_with_start_stop_step. Retrieved 8/10 statements.
# Partially parsed test_slice_with_negative_start. Retrieved 7/9 statements.
# Partially parsed test_slice_with_stop_only. Retrieved 6/8 statements.
# Partially parsed test_slice_with_step_only. Retrieved 8/10 statements.
# Partially parsed test_slice_with_all_negative. Retrieved 10/12 statements.
# Partially parsed test_slice_with_large_step. Retrieved 9/11 statements.
# Partially parsed test_slice_with_zero_step_raises_error. Retrieved 2/5 statements.
# Partially parsed test_slice_out_of_bounds. Retrieved 4/6 statements.
# Partially parsed test_slice_with_start_greater_than_stop_negative_step. Retrieved 9/11 statements.
# Partially parsed test_slice_on_empty_range. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 1
    var_3 = 8
    var_4 = 2
    var_5 = 3
    var_6 = 5
    var_7 = 7
    var_8 = [var_2, var_5, var_6, var_7]

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = [var_0, var_1]
    var_3 = -3
    var_4 = 12
    var_5 = 13
    var_6 = 14
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 5
    var_6 = [var_0, var_4, var_5]

def test_case_0():
    var_0 = 0
    var_1 = 20
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 6
    var_6 = 12
    var_7 = 18
    var_8 = [var_0, var_5, var_6, var_7]

def test_case_0():
    var_0 = 100
    var_1 = [var_0]
    var_2 = -10
    var_3 = -20
    var_4 = -2
    var_5 = 90
    var_6 = 88
    var_7 = 86
    var_8 = 84
    var_9 = 82
    var_10 = [var_5, var_6, var_7, var_8, var_9]

def test_case_0():
    var_0 = 0
    var_1 = 50
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 8
    var_6 = 3
    var_7 = 10
    var_8 = 25
    var_9 = [var_7, var_8]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 0
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 10
    var_3 = 20
    var_4 = []

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = -1
    var_6 = 4
    var_7 = 3
    var_8 = 2
    var_9 = [var_4, var_6, var_7, var_8]

def test_case_0():
    var_0 = 0
    var_1 = -5
    var_2 = -1
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 4
    var_6 = -1
    var_7 = -2
    var_8 = -3
    var_9 = [var_6, var_7, var_8]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_getitem_with_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_and_step. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_full_slice. Retrieved 7/9 statements.
# Partially parsed test_getitem_with_negative_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_out_of_bounds. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_and_negative_step. Retrieved 7/9 statements.
# Partially parsed test_getitem_with_empty_slice. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = 5
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_4, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 5
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_2, var_3, var_4, var_5, var_6]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -3
    var_3 = -1
    var_4 = 7
    var_5 = 8
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 2
    var_3 = 10
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_4, var_5]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 5
    var_3 = 2
    var_4 = -1
    var_5 = 4
    var_6 = 3
    var_7 = [var_2, var_5, var_6]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 5
    var_3 = 2
    var_4 = []



# Parsed testcases at query #32
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list is var_4)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list == var_1)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = (var_2, var_3, var_4)
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = var_6.func
    var_8 = bool(var_6.func is var_1)
    assert var_8 is True
    var_9 = var_6.list
    var_10 = bool(var_6.list == var_5)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.MapList(var_1, var_3)
    var_5 = var_4.func
    var_6 = bool(var_4.func is var_1)
    assert var_6 is True
    var_7 = var_4.list
    var_8 = list(var_7)
    var_9 = range(var_2)
    var_10 = list(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda c: c * var_0
    var_2 = 'abc'
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func is var_1)
    assert var_5 is True
    var_6 = var_3.list
    assert var_6 == 'abc'



# Parsed testcases at query #33
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True
    var_7 = 0
    var_8 = lambda x: x == var_7
    var_9 = 1
    var_10 = 2
    var_11 = [var_7, var_9, var_10]
    var_12 = module_0.drop_until(var_8, var_11)
    var_13 = list(var_12)
    var_14 = bool(var_13 == [0, 1, 2])
    assert var_14 is True
    var_15 = lambda x: x == var_0
    var_16 = 3
    var_17 = 4
    var_18 = 6
    var_19 = [var_9, var_10, var_16, var_17, var_0, var_18]
    var_20 = module_0.drop_until(var_15, var_19)
    var_21 = list(var_20)
    var_22 = bool(var_21 == [5, 6])
    assert var_22 is True
    var_23 = lambda x: x
    var_24 = False
    var_25 = False
    var_26 = True
    var_27 = False
    var_28 = [var_24, var_25, var_26, var_27]
    var_29 = module_0.drop_until(var_23, var_28)
    var_30 = list(var_29)
    var_31 = bool(var_30 == [True, False])
    assert var_31 is True
    var_32 = lambda x: len(x) > var_10
    var_33 = 'a'
    var_34 = 'ab'
    var_35 = 'abc'
    var_36 = 'd'
    var_37 = [var_33, var_34, var_35, var_36]
    var_38 = module_0.drop_until(var_32, var_37)
    var_39 = list(var_38)
    var_40 = bool(var_39 == ['abc', 'd'])
    assert var_40 is True
    var_41 = lambda x: x % var_10 == var_27
    var_42 = 7
    var_43 = 8
    var_44 = [var_26, var_16, var_0, var_18, var_42, var_43]
    var_45 = module_0.drop_until(var_41, var_44)
    var_46 = list(var_45)
    var_47 = bool(var_46 == [6, 7, 8])
    assert var_47 is True
    var_48 = None
    var_49 = lambda x: x is var_48
    var_50 = [var_48, var_26, var_10]
    var_51 = module_0.drop_until(var_49, var_50)
    var_52 = list(var_51)
    var_53 = bool(var_52 == [None, 1, 2])
    assert var_53 is True
    var_54 = lambda x: x > var_2
    var_55 = [var_26, var_10, var_16]
    var_56 = module_0.drop_until(var_54, var_55)
    var_57 = list(var_56)
    var_58 = bool(var_57 == [])
    assert var_58 is True
    var_59 = lambda x: x
    var_60 = []
    var_61 = module_0.drop_until(var_59, var_60)
    var_62 = list(var_61)
    var_63 = bool(var_62 == [])
    assert var_63 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_getitem_with_positive_index. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_negative_index. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_full_slice. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_step_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_reverse_slice. Retrieved 9/11 statements.
# Partially parsed test_getitem_with_slice_and_negative_indices. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_slice_out_of_range. Retrieved 7/9 statements.
# Partially parsed test_getitem_with_single_argument_range. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_single_argument_range_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_two_argument_range. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_two_argument_range_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_negative_step_range. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_negative_step_range_slice. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -1
    var_5 = 9

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 5
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 5
    var_6 = 7
    var_7 = 9
    var_8 = [var_0, var_4, var_5, var_6, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 9
    var_6 = [var_0, var_4, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -1
    var_5 = 9
    var_6 = 7
    var_7 = 5
    var_8 = 3
    var_9 = [var_5, var_6, var_7, var_8, var_0]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -3
    var_5 = -1
    var_6 = 5
    var_7 = 7
    var_8 = [var_6, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 7
    var_6 = 9
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 3
    var_3 = 3

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 1
    var_3 = 4
    var_4 = 2
    var_5 = 3
    var_6 = [var_2, var_4, var_5]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]
    var_3 = 4

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 6

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 4
    var_6 = 8
    var_7 = 6
    var_8 = [var_6, var_7, var_5]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_constructor_with_empty_list. Retrieved 1/3 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list is var_4)
    assert var_9 is True

def test_case_0():
    var_0 = []

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = (var_2, var_3, var_4)
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = var_6.func
    var_8 = bool(var_6.func is var_1)
    assert var_8 is True
    var_9 = var_6.list
    var_10 = bool(var_6.list == var_5)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.MapList(var_1, var_3)
    var_5 = var_4.func
    var_6 = bool(var_4.func is var_1)
    assert var_6 is True
    var_7 = var_4.list
    var_8 = bool(var_4.list == var_3)
    assert var_8 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_immediately. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_getitem_with_int_index_negative. Retrieved 6/7 statements.
# Partially parsed test_getitem_with_slice_negative_stop. Retrieved 7/8 statements.
# Partially parsed test_getitem_after_exhaustion. Retrieved 6/7 statements.
# Partially parsed test_getitem_slice_after_exhaustion. Retrieved 7/8 statements.
# Partially parsed test_getitem_index_error. Retrieved 6/8 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 3
    var_4 = var_2[var_3]
    assert var_4 == 3
    var_5 = var_2.list
    var_6 = len(var_5)
    assert var_6 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = -1
    var_5 = var_2[var_4]
    assert var_5 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 2
    var_4 = 5
    var_5 = var_2[var_3:var_4]
    var_6 = bool(var_5 == [2, 3, 4])
    assert var_6 is True
    var_7 = var_2.list
    var_8 = len(var_7)
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 2
    var_4 = var_2[var_3:]
    var_5 = bool(var_4 == [2, 3, 4])
    assert var_5 is True
    var_6 = var_2.list
    var_7 = len(var_6)
    assert var_7 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 2
    var_5 = -2
    var_6 = var_2[var_4:var_5]
    var_7 = bool(var_6 == [2, 3, 4, 5, 6, 7])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 4
    var_4 = var_2[var_3]
    var_5 = var_2.list
    var_6 = len(var_5)
    assert var_6 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 2
    var_4 = 7
    var_5 = var_2[var_3:var_4]
    var_6 = var_2.list
    var_7 = len(var_6)
    assert var_7 == 7

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 3
    var_5 = var_2[var_4]
    assert var_5 == 3

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 1
    var_5 = 4
    var_6 = var_2[var_4:var_5]
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 5
    var_5 = var_2[var_4]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 0
    var_4 = 10
    var_5 = var_2[var_3:var_4]
    var_6 = bool(var_5 == [0, 1, 2])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 1
    var_4 = 8
    var_5 = 2
    var_6 = var_2[var_3:var_4:var_5]
    var_7 = bool(var_6 == [1, 3, 5, 7])
    assert var_7 is True
    var_8 = var_2.list
    var_9 = len(var_8)
    assert var_9 == 8



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_raises_error. Retrieved 3/5 statements.
# Partially parsed test_constructor_length_calculation_positive_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_length_calculation_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_initial_val_set_to_start. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 3
    var_1 = 7
    var_2 = [var_0, var_1]



# Parsed testcases at query #39
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True
    var_7 = len(var_5)
    var_8 = bool(var_7 > 0)
    assert var_8 is True
    var_9 = 0
    var_10 = var_5[var_9]
    var_11 = bool(var_10 > 5)
    assert var_11 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_raises_error. Retrieved 3/5 statements.
# Partially parsed test_constructor_length_calculation_positive_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_length_calculation_negative_step. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_slice_indexing_returns_list. Retrieved 7/9 statements.
# Partially parsed test_slice_with_negative_start. Retrieved 6/8 statements.
# Partially parsed test_slice_with_negative_stop. Retrieved 8/10 statements.
# Partially parsed test_slice_with_step. Retrieved 8/10 statements.
# Partially parsed test_full_slice. Retrieved 7/9 statements.
# Partially parsed test_slice_out_of_bounds. Retrieved 6/8 statements.
# Partially parsed test_slice_with_only_step. Retrieved 7/9 statements.
# Partially parsed test_slice_with_negative_step. Retrieved 8/10 statements.
# Partially parsed test_slice_on_empty_range. Retrieved 2/4 statements.
# Partially parsed test_slice_with_start_stop_equal. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 3
    var_6 = 5
    var_7 = [var_0, var_5, var_6]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -3
    var_3 = 7
    var_4 = 8
    var_5 = 9
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = [var_0, var_1]
    var_3 = -5
    var_4 = 6
    var_5 = 7
    var_6 = 8
    var_7 = 9
    var_8 = [var_0, var_4, var_5, var_6, var_7]

def test_case_0():
    var_0 = 0
    var_1 = 20
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 5
    var_6 = 2
    var_7 = 9
    var_8 = [var_2, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 6
    var_2 = [var_0, var_1]
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = 5
    var_7 = [var_0, var_3, var_4, var_5, var_6]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 2
    var_3 = 10
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_4, var_5]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 6
    var_6 = 8
    var_7 = [var_0, var_2, var_4, var_5, var_6]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 5
    var_3 = 1
    var_4 = -1
    var_5 = 4
    var_6 = 3
    var_7 = 2
    var_8 = [var_2, var_5, var_6, var_7]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = []

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = []



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_getitem_with_negative_index_after_exhaustion. Retrieved 7/8 statements.
# Partially parsed test_getitem_with_slice_and_step. Retrieved 12/13 statements.
# Partially parsed test_getitem_index_out_of_range_raises_index_error. Retrieved 6/8 statements.
# Partially parsed test_getitem_slice_out_of_range_returns_empty_list. Retrieved 8/9 statements.
# Partially parsed test_getitem_slice_with_negative_start_after_exhaustion. Retrieved 10/11 statements.
# Partially parsed test_getitem_after_exhaustion_uses_internal_list. Retrieved 8/9 statements.
# Partially parsed test_getitem_slice_after_exhaustion_uses_internal_list. Retrieved 11/12 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = var_2[var_3]
    var_5 = 5
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 0
    var_4 = var_2[var_3]
    var_5 = 0
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = iter(var_1)
    var_3 = module_0.LazyList(var_2)
    var_4 = 3
    var_5 = var_3[var_4]
    var_6 = list(var_2)
    var_7 = 4
    var_8 = range(var_7, var_0)
    var_9 = list(var_8)
    var_10 = bool(var_6 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = -1
    var_5 = var_2[var_4]
    var_6 = 4
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = var_2[:var_3]
    var_5 = 0
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = 4
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = bool(var_4 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 2
    var_4 = 6
    var_5 = var_2[var_3:var_4]
    var_6 = 3
    var_7 = 4
    var_8 = 5
    var_9 = [var_3, var_6, var_7, var_8]
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 1
    var_5 = 8
    var_6 = 2
    var_7 = var_2[var_4:var_5:var_6]
    var_8 = 3
    var_9 = 5
    var_10 = 7
    var_11 = [var_4, var_8, var_9, var_10]
    var_12 = bool(var_7 == var_11)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = iter(var_1)
    var_3 = module_0.LazyList(var_2)
    var_4 = 10
    var_5 = 20
    var_6 = var_3[var_4:var_5]
    var_7 = list(var_2)
    var_8 = range(var_5, var_0)
    var_9 = list(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 10
    var_5 = var_2[var_4]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 10
    var_5 = 20
    var_6 = var_2[var_4:var_5]
    var_7 = []
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = iter(var_1)
    var_3 = module_0.LazyList(var_2)
    var_4 = -1
    var_5 = var_3[var_4]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True
    var_8 = var_3.list
    var_9 = len(var_8)
    assert var_9 == 0

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = -5
    var_4 = var_2[:var_3]
    var_5 = 0
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = 4
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = bool(var_4 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = -3
    var_5 = var_2[var_4:]
    var_6 = 7
    var_7 = 8
    var_8 = 9
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = 0
    var_3 = var_1[var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = 0
    var_3 = 5
    var_4 = var_1[var_2:var_3]
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 2
    var_5 = var_2[var_4]
    var_6 = var_2[var_4]
    var_7 = 2
    var_8 = bool(var_5 == var_7)
    assert var_8 is True
    var_9 = bool(var_6 == var_7)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 1
    var_5 = 4
    var_6 = var_2[var_4:var_5]
    var_7 = var_2[var_4:var_5]
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = bool(var_6 == var_10)
    assert var_11 is True
    var_12 = bool(var_7 == var_10)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1000
    var_1 = range(var_0)
    var_2 = iter(var_1)
    var_3 = module_0.LazyList(var_2)
    var_4 = 999
    var_5 = var_3[var_4]
    var_6 = list(var_2)
    var_7 = bool(var_6 == [])
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_take_with_generator. Retrieved 3/7 statements.
# Partially parsed test_take_large_n_with_infinite_generator. Retrieved 1/5 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.take(var_0, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2, var_0]
    var_4 = module_0.take(var_0, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0.take(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = iter(var_5)
    var_7 = module_0.take(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [1, 2])
    assert var_9 is True
    var_10 = list(var_6)
    var_11 = bool(var_10 == [3, 4, 5])
    assert var_11 is True

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 4

def test_case_0():
    var_0 = 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 'hello'
    var_2 = module_0.take(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['h', 'e', 'l'])
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 3
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    var_6 = list(var_5)
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = 4
    var_11 = 5
    var_12 = [var_10, var_11]
    var_13 = 7
    var_14 = 8
    var_15 = [var_13, var_14]
    var_16 = [var_9, var_12, var_15]
    var_17 = bool(var_6 == var_16)
    assert var_17 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = True
    var_3 = 3
    var_4 = 0
    var_5 = lambda x: x % var_3 == var_4
    var_6 = module_0.split_by(var_1, var_2, criterion=var_5)
    var_7 = list(var_6)
    var_8 = []
    var_9 = 2
    var_10 = [var_2, var_9]
    var_11 = 4
    var_12 = 5
    var_13 = [var_11, var_12]
    var_14 = 7
    var_15 = 8
    var_16 = [var_14, var_15]
    var_17 = []
    var_18 = [var_8, var_10, var_13, var_16, var_17]
    var_19 = bool(var_7 == var_18)
    assert var_19 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a.b.c'
    var_1 = '.'
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = [var_6]
    var_8 = 'c'
    var_9 = [var_8]
    var_10 = [var_5, var_7, var_9]
    var_11 = bool(var_3 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = ' Split by: '
    var_1 = True
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = 'S'
    var_7 = 'p'
    var_8 = 'l'
    var_9 = 'i'
    var_10 = 't'
    var_11 = [var_6, var_7, var_8, var_9, var_10]
    var_12 = 'b'
    var_13 = 'y'
    var_14 = ':'
    var_15 = [var_12, var_13, var_14]
    var_16 = []
    var_17 = [var_5, var_11, var_15, var_16]
    var_18 = bool(var_4 == var_17)
    assert var_18 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = lambda x: x > var_4
    var_6 = module_0.split_by(var_3, criterion=var_5)
    var_7 = list(var_6)
    var_8 = [var_0, var_1, var_2]
    var_9 = [var_8]
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = module_0.split_by(var_3, separator=var_4)
    var_6 = list(var_5)
    var_7 = [var_0, var_1, var_2]
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = lambda x: x == var_0
    var_3 = module_0.split_by(var_1, criterion=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.split_by(var_1, separator=var_0)
    var_3 = list(var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = True
    var_3 = lambda x: x == var_2
    var_4 = module_0.split_by(var_1, var_2, criterion=var_3)
    var_5 = list(var_4)
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = [var_6, var_7, var_8, var_9]
    var_11 = bool(var_5 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0, var_0, var_0]
    var_2 = True
    var_3 = module_0.split_by(var_1, var_2, separator=var_0)
    var_4 = list(var_3)
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = lambda x: x is var_1
    var_3 = module_0.split_by(var_0, criterion=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = lambda x: x is var_2
    var_4 = module_0.split_by(var_0, var_1, criterion=var_3)
    var_5 = list(var_4)
    var_6 = []
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 0
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_1, var_1, var_2, var_1, var_3]
    var_5 = lambda x: x == var_1
    var_6 = module_0.split_by(var_4, criterion=var_5)
    var_7 = list(var_6)
    var_8 = [var_0]
    var_9 = [var_2]
    var_10 = [var_3]
    var_11 = [var_8, var_9, var_10]
    var_12 = bool(var_7 == var_11)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_1, var_1, var_2, var_1, var_3]
    var_5 = module_0.split_by(var_4, separator=var_1)
    var_6 = list(var_5)
    var_7 = [var_0]
    var_8 = [var_2]
    var_9 = [var_3]
    var_10 = [var_7, var_8, var_9]
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_1, var_1, var_2, var_1, var_3]
    var_5 = True
    var_6 = lambda x: x == var_1
    var_7 = module_0.split_by(var_4, var_5, criterion=var_6)
    var_8 = list(var_7)
    var_9 = [var_5]
    var_10 = []
    var_11 = [var_2]
    var_12 = [var_3]
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = bool(var_8 == var_13)
    assert var_14 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_1, var_1, var_2, var_1, var_3]
    var_5 = True
    var_6 = module_0.split_by(var_4, var_5, separator=var_1)
    var_7 = list(var_6)
    var_8 = [var_5]
    var_9 = []
    var_10 = [var_2]
    var_11 = [var_3]
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = bool(var_7 == var_12)
    assert var_13 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = lambda x: x == var_0
    var_5 = module_0.split_by(var_3, criterion=var_4)
    var_6 = list(var_5)
    var_7 = [var_1, var_2]
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = module_0.split_by(var_3, separator=var_0)
    var_5 = list(var_4)
    var_6 = [var_1, var_2]
    var_7 = [var_6]
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = True
    var_5 = lambda x: x == var_0
    var_6 = module_0.split_by(var_3, var_4, criterion=var_5)
    var_7 = list(var_6)
    var_8 = []
    var_9 = [var_4, var_2]
    var_10 = []
    var_11 = [var_8, var_9, var_10]
    var_12 = bool(var_7 == var_11)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = True
    var_5 = module_0.split_by(var_3, var_4, separator=var_0)
    var_6 = list(var_5)
    var_7 = []
    var_8 = [var_4, var_2]
    var_9 = []
    var_10 = [var_7, var_8, var_9]
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.split_by(var_3)
    var_5 = list(var_4)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x > var_0
    var_5 = module_0.split_by(var_3, criterion=var_4, separator=var_1)
    var_6 = list(var_5)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_scanl_with_initial_value. Retrieved 10/12 statements.
# Partially parsed test_scanl_without_initial_value. Retrieved 10/12 statements.
# Partially parsed test_scanl_empty_iterable_with_initial. Retrieved 4/6 statements.
# Partially parsed test_scanl_empty_iterable_without_initial. Retrieved 7/10 statements.
# Partially parsed test_scanl_single_element_without_initial. Retrieved 4/6 statements.
# Partially parsed test_scanl_single_element_with_initial. Retrieved 6/8 statements.
# Partially parsed test_scanl_too_many_arguments. Retrieved 5/8 statements.
# Partially parsed test_scanl_with_different_func. Retrieved 9/11 statements.
# Partially parsed test_scanl_with_iterator. Retrieved 9/11 statements.


def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = 0
    var_7 = [var_6]
    var_8 = 6
    var_9 = 10
    var_10 = [var_6, var_1, var_3, var_8, var_9]

def test_case_0():
    var_0 = lambda s, x: x + s
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = []
    var_7 = 'ba'
    var_8 = 'cba'
    var_9 = 'dcba'
    var_10 = [var_1, var_7, var_8, var_9]

def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = []
    var_2 = 5
    var_3 = [var_2]
    var_4 = [var_2]

def test_case_0():
    var_0 = []
    var_1 = iter(var_0)
    var_2 = next(var_1)
    var_3 = lambda a, b: a + b
    var_4 = []
    var_5 = iter(var_4)
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 10
    var_2 = [var_1]
    var_3 = []
    var_4 = [var_1]

def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 10
    var_2 = [var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = 15
    var_6 = [var_3, var_5]

def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 0
    var_5 = [var_4, var_1]

def test_case_0():
    var_0 = lambda a, b: a * b
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = [var_1]
    var_7 = 6
    var_8 = 24
    var_9 = [var_1, var_1, var_2, var_7, var_8]

def test_case_0():
    var_0 = lambda a, b: a + b
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = iter(var_4)
    var_6 = 0
    var_7 = [var_6]
    var_8 = 6
    var_9 = [var_6, var_1, var_3, var_8]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_iter_single_arg. Retrieved 7/10 statements.
# Partially parsed test_iter_two_args. Retrieved 7/10 statements.
# Partially parsed test_iter_three_args. Retrieved 8/11 statements.
# Partially parsed test_iter_negative_step. Retrieved 8/11 statements.
# Partially parsed test_iter_empty_range. Retrieved 2/5 statements.
# Partially parsed test_iter_reverse_range. Retrieved 8/11 statements.
# Partially parsed test_iter_after_indexing. Retrieved 12/16 statements.
# Partially parsed test_iter_multiple_calls. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_2, var_3, var_4, var_5, var_6]

def test_case_0():
    var_0 = 2
    var_1 = 7
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_0, var_3, var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 5
    var_6 = 7
    var_7 = 9
    var_8 = [var_0, var_4, var_5, var_6, var_7]

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 3
    var_6 = 2
    var_7 = 1
    var_8 = [var_0, var_4, var_5, var_6, var_7]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = []

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]
    var_4 = 8
    var_5 = 6
    var_6 = 4
    var_7 = 2
    var_8 = [var_0, var_4, var_5, var_6, var_7]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 3
    var_3 = 0
    var_4 = 1
    var_5 = 2
    var_6 = 4
    var_7 = 5
    var_8 = 6
    var_9 = 7
    var_10 = 8
    var_11 = 9
    var_12 = [var_3, var_4, var_5, var_2, var_6, var_7, var_8, var_9, var_10, var_11]

def test_case_0():
    var_0 = 3
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = 2
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_raises_no_error_but_length_calculation. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_with_empty_list. Retrieved 1/3 statements.
# Partially parsed test_constructor_with_tuple. Retrieved 4/6 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list is var_4)
    assert var_9 is True

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'bb'
    var_2 = 'ccc'
    var_3 = (var_0, var_1, var_2)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.MapList(var_1, var_3)
    var_5 = var_4.func
    var_6 = bool(var_4.func is var_1)
    assert var_6 is True
    var_7 = var_4.list
    var_8 = list(var_7)
    var_9 = list(var_3)
    var_10 = bool(var_8 == var_9)
    assert var_10 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_but_length_calculation. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_drop_generator. Retrieved 3/6 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [4, 5])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = []
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = iter(var_1)
    var_3 = 5
    var_4 = module_0.drop(var_3, var_2)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [5, 6, 7, 8, 9])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 1000000
    var_2 = range(var_1)
    var_3 = module_0.drop(var_0, var_2)
    var_4 = next(var_3)
    assert var_4 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'hello'
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['l', 'l', 'o'])
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = 2



# Parsed testcases at query #10
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = 0
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = 4
    var_10 = 5
    var_11 = [var_0, var_9, var_10]
    var_12 = 6
    var_13 = 7
    var_14 = 8
    var_15 = [var_12, var_13, var_14]
    var_16 = 9
    var_17 = [var_16]
    var_18 = [var_8, var_11, var_15, var_17]
    var_19 = bool(var_4 == var_18)
    assert var_19 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = 4
    var_4 = [var_1, var_0, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = [var_1, var_0]
    var_8 = [var_2, var_3]
    var_9 = [var_7, var_8]
    var_10 = bool(var_6 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 6
    var_3 = 7
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = [var_1]
    var_8 = [var_2]
    var_9 = [var_3]
    var_10 = [var_7, var_8, var_9]
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = []
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 4
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3, var_0]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = [var_1, var_2, var_3, var_0]
    var_8 = [var_7]
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'abcde'
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = 'c'
    var_8 = 'd'
    var_9 = [var_7, var_8]
    var_10 = 'e'
    var_11 = [var_10]
    var_12 = [var_6, var_9, var_11]
    var_13 = bool(var_3 == var_12)
    assert var_13 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = iter(var_1)
    var_3 = 2
    var_4 = module_0.chunk(var_3, var_2)
    var_5 = list(var_4)
    var_6 = 0
    var_7 = 1
    var_8 = [var_6, var_7]
    var_9 = 3
    var_10 = [var_3, var_9]
    var_11 = 4
    var_12 = [var_11]
    var_13 = [var_8, var_10, var_12]
    var_14 = bool(var_5 == var_13)
    assert var_14 is True
    var_15 = list(var_2)
    var_16 = bool(var_15 == [])
    assert var_16 is True



# Parsed testcases at query #11
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5[var_2]
    assert var_6 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = 6
    var_4 = 7
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = -1
    var_8 = var_6[var_7]
    assert var_8 == 17

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_2, var_0, var_3, var_4, var_5]
    var_7 = module_0.MapList(var_1, var_6)
    var_8 = var_7[var_2:var_4]
    var_9 = bool(var_8 == [4, 9, 16])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.MapList(var_0, var_4)
    var_6 = var_5[:]
    var_7 = bool(var_6 == ['10', '20', '30'])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x / var_0
    var_2 = 4
    var_3 = 6
    var_4 = 8
    var_5 = 10
    var_6 = [var_0, var_2, var_3, var_4, var_5]
    var_7 = module_0.MapList(var_1, var_6)
    var_8 = var_7[::var_0]
    var_9 = bool(var_8 == [1.0, 3.0, 5.0])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = 0
    var_5 = 5
    var_6 = var_3[var_4:var_5]
    var_7 = bool(var_6 == [])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.MapList(var_0, var_4)
    var_6 = 10
    var_7 = var_5[var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: (x, x * var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = 0
    var_8 = var_6[var_7]
    var_9 = bool(var_8 == ('a', 'aa'))
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_2
    var_5 = module_0.MapList(var_4, var_3)
    var_6 = 0
    var_7 = var_5[var_6]
    var_8 = bool(var_3 == [1, 2, 3])
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_getitem_with_single_index. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_negative_index. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_and_step. Retrieved 7/9 statements.
# Partially parsed test_getitem_with_full_slice. Retrieved 7/9 statements.
# Partially parsed test_getitem_with_slice_negative_indices. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_out_of_range. Retrieved 4/6 statements.
# Partially parsed test_getitem_with_slice_and_negative_step. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_start_stop_step. Retrieved 4/6 statements.
# Partially parsed test_getitem_with_slice_no_start. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_no_stop. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_slice_step. Retrieved 7/9 statements.
# Partially parsed test_getitem_index_error. Retrieved 2/5 statements.
# Partially parsed test_getitem_with_negative_index_out_of_range. Retrieved 2/5 statements.
# Partially parsed test_getitem_with_slice_indices_method. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 0
    var_3 = 0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -1
    var_3 = 9

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = 5
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_4, var_5]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 4
    var_6 = 6
    var_7 = [var_2, var_5, var_6]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_2, var_3, var_4, var_5, var_6]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -3
    var_3 = -1
    var_4 = 7
    var_5 = 8
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 10
    var_3 = 20
    var_4 = []

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 5
    var_3 = 1
    var_4 = -1
    var_5 = 4
    var_6 = 3
    var_7 = 2
    var_8 = [var_2, var_5, var_6, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 5

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 3
    var_3 = 0
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 7
    var_3 = 8
    var_4 = 9
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = 0
    var_4 = 4
    var_5 = 6
    var_6 = 8
    var_7 = [var_3, var_2, var_4, var_5, var_6]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 10
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = -10
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = 5
    var_4 = 1
    var_5 = slice(var_2, var_3, var_4)
    var_6 = 3
    var_7 = 4
    var_8 = [var_2, var_6, var_7]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_iter_single_arg. Retrieved 1/8 statements.
# Partially parsed test_iter_two_args. Retrieved 2/9 statements.
# Partially parsed test_iter_three_args. Retrieved 3/10 statements.
# Partially parsed test_iter_negative_step. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 7
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_getitem_with_slice_returns_list. Retrieved 3/6 statements.
# Partially parsed test_getitem_with_slice_and_step. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_full_slice. Retrieved 1/4 statements.
# Partially parsed test_getitem_with_negative_slice. Retrieved 3/6 statements.
# Partially parsed test_getitem_with_slice_out_of_bounds. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = 5

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3

def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -3
    var_3 = -1

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_negative_index_out_of_range. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -11



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_negative_index_handling. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -1
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_0, var_4]
    var_6 = -2
    var_7 = 5
    var_8 = 0
    var_9 = -1
    var_10 = [var_7, var_8, var_9]
    var_11 = -1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_constructor_with_tuple. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_string. Retrieved 1/3 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list is var_4)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list == var_1)
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.MapList(var_1, var_3)
    var_5 = var_4.func
    var_6 = bool(var_4.func is var_1)
    assert var_6 is True
    var_7 = var_4.list
    var_8 = bool(var_4.list == var_3)
    assert var_8 is True

def test_case_0():
    var_0 = 'abc'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_immediately. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #19
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func is var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list is var_4)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list == var_1)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = (var_2, var_3, var_4)
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = var_6.func
    var_8 = bool(var_6.func is var_1)
    assert var_8 is True
    var_9 = var_6.list
    var_10 = bool(var_6.list == var_5)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.MapList(var_1, var_3)
    var_5 = var_4.func
    var_6 = bool(var_4.func is var_1)
    assert var_6 is True
    var_7 = var_4.list
    var_8 = list(var_7)
    var_9 = range(var_2)
    var_10 = list(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: ord(x)
    var_1 = 'abc'
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    assert var_5 == 'abc'



# Parsed testcases at query #20
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = [var_2, var_3, var_4]
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = [var_2, var_3, var_4]
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = []
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = 6
    var_7 = 7
    var_8 = [var_2, var_3, var_4, var_5, var_0, var_6, var_7]
    var_9 = module_0.drop_until(var_1, var_8)
    var_10 = list(var_9)
    var_11 = [var_0, var_6, var_7]
    var_12 = bool(var_10 == var_11)
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'b'
    var_1 = lambda s: s.startswith(var_0)
    var_2 = 'a'
    var_3 = 'aa'
    var_4 = 'bb'
    var_5 = 'c'
    var_6 = [var_2, var_3, var_0, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = [var_0, var_4, var_5]
    var_10 = bool(var_8 == var_9)
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = 1
    var_4 = 3
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = iter(var_8)
    var_10 = module_0.drop_until(var_2, var_9)
    var_11 = list(var_10)
    var_12 = [var_5, var_6, var_7]
    var_13 = bool(var_11 == var_12)
    assert var_13 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0
    var_2 = 1
    var_3 = 2
    var_4 = [var_0, var_0, var_2, var_3, var_0]
    var_5 = module_0.drop_until(var_1, var_4)
    var_6 = list(var_5)
    var_7 = [var_2, var_3, var_0]
    var_8 = bool(var_6 == var_7)
    assert var_8 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_iter_single_arg. Retrieved 1/8 statements.
# Partially parsed test_iter_two_args. Retrieved 2/9 statements.
# Partially parsed test_iter_three_args. Retrieved 3/10 statements.
# Partially parsed test_iter_negative_step. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 7
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_negative_index_handling. Retrieved 5/7 statements.
# Partially parsed test_negative_index_handling_with_single_arg. Retrieved 3/5 statements.
# Partially parsed test_negative_index_handling_with_start_stop. Retrieved 4/6 statements.
# Partially parsed test_negative_index_handling_step_not_one. Retrieved 5/7 statements.
# Partially parsed test_negative_index_handling_zero_length. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -1
    var_5 = 9

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -3
    var_3 = 7

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = [var_0, var_1]
    var_3 = -2
    var_4 = 13

def test_case_0():
    var_0 = 0
    var_1 = 20
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = -1
    var_5 = 18

def test_case_0():
    var_0 = 5
    var_1 = [var_0, var_0]
    var_2 = -1



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_constructor_with_empty_iterable. Retrieved 5/7 statements.
# Failed to parse test_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = 0
    var_6 = var_1.exhausted
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = len(var_6)
    assert var_7 == 0

import flutes.iterator as module_0

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = [var_0, var_1, var_2]
    var_4 = iter(var_3)
    var_5 = module_0.LazyList(var_4)
    var_6 = var_5.exhausted
    assert var_6 is False
    var_7 = var_5.list
    var_8 = len(var_7)
    assert var_8 == 0

import flutes.iterator as module_0

def test_case_0():
    var_0 = 9
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.LazyList(var_2)
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = len(var_6)
    assert var_7 == 0



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = bool(var_4.list == [])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = iter(var_3)
    var_5 = module_0.LazyList(var_4)
    var_6 = var_5.exhausted
    assert var_6 is False
    var_7 = var_5.list
    var_8 = bool(var_5.list == [])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_error_immediately. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_start_equal_stop_and_positive_step. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_equal_stop_and_negative_step. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_greater_than_stop_and_positive_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_start_less_than_stop_and_negative_step. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = [var_0, var_0, var_1]

def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = [var_0, var_0, var_1]

def test_case_0():
    var_0 = 8
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = -1
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_raises_no_explicit_error_but_length_calculation. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_getitem_with_positive_index. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_negative_index. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_full_slice. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_step_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_negative_step. Retrieved 9/11 statements.
# Partially parsed test_getitem_with_slice_out_of_bounds. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_slice_start_only. Retrieved 7/9 statements.
# Partially parsed test_getitem_with_slice_stop_only. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_slice_negative_indices. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_index_out_of_range_positive. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_index_out_of_range_negative. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_single_arg_constructor. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_two_arg_constructor. Retrieved 4/6 statements.
# Partially parsed test_getitem_with_negative_step_constructor. Retrieved 5/7 statements.
# Partially parsed test_getitem_slice_with_negative_step_constructor. Retrieved 8/10 statements.
# Partially parsed test_getitem_slice_with_zero_step_constructor. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -1
    var_5 = 9

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 5
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 5
    var_6 = 7
    var_7 = 9
    var_8 = [var_0, var_4, var_5, var_6, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 9
    var_6 = [var_0, var_4, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -1
    var_5 = 9
    var_6 = 7
    var_7 = 5
    var_8 = 3
    var_9 = [var_5, var_6, var_7, var_8, var_0]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 5
    var_6 = 7
    var_7 = 9
    var_8 = [var_4, var_5, var_6, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 7
    var_6 = 9
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -3
    var_5 = -1
    var_6 = 5
    var_7 = 7
    var_8 = [var_6, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -10
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 2
    var_3 = 2

def test_case_0():
    var_0 = 2
    var_1 = 7
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 5

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 6

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 3
    var_6 = 8
    var_7 = 6
    var_8 = [var_6, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = []



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = bool(var_4.list == [])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.exhausted
    assert var_3 is False
    var_4 = var_2.list
    var_5 = bool(var_2.list == [])
    assert var_5 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_during_init. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_immediately. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_start_equal_stop_and_positive_step. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_equal_stop_and_negative_step. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_large_numbers. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = [var_0, var_0, var_1]

def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = [var_0, var_0, var_1]

def test_case_0():
    var_0 = 1000
    var_1 = 2000
    var_2 = 100
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_slice_handling. Retrieved 7/9 statements.
# Partially parsed test_slice_with_step. Retrieved 7/9 statements.
# Partially parsed test_slice_negative_indices. Retrieved 7/9 statements.
# Partially parsed test_slice_full_range. Retrieved 5/7 statements.
# Partially parsed test_slice_out_of_bounds. Retrieved 6/8 statements.
# Partially parsed test_slice_with_negative_step. Retrieved 8/10 statements.
# Partially parsed test_slice_empty_result. Retrieved 3/5 statements.
# Partially parsed test_slice_start_none. Retrieved 5/7 statements.
# Partially parsed test_slice_stop_none. Retrieved 5/7 statements.
# Partially parsed test_slice_step_none. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_2, var_4, var_5, var_6]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 4
    var_6 = 6
    var_7 = [var_2, var_5, var_6]

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = [var_0, var_1]
    var_3 = -3
    var_4 = -1
    var_5 = 12
    var_6 = 13
    var_7 = [var_5, var_6]

def test_case_0():
    var_0 = 3
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = 2
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 2
    var_3 = 10
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_4, var_5]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 5
    var_3 = 1
    var_4 = -1
    var_5 = 4
    var_6 = 3
    var_7 = 2
    var_8 = [var_2, var_5, var_6, var_7]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 5
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = 6
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 2
    var_5 = [var_0, var_4, var_3]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 6
    var_5 = 9
    var_6 = [var_0, var_2, var_4, var_5]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_during_init. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_raises_no_explicit_error_but_length_calculation. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_during_init. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_equal_stop. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_greater_than_stop_and_positive_step. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_less_than_stop_and_negative_step. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = [var_0, var_1, var_0]

def test_case_0():
    var_0 = 5
    var_1 = [var_0, var_0]

def test_case_0():
    var_0 = 8
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = -1
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_division_by_zero. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = [var_0, var_1, var_0]



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_constructor_no_args.
# Partially parsed test_constructor_one_arg. Retrieved 1/2 statements.
# Partially parsed test_constructor_two_args. Retrieved 2/3 statements.
# Partially parsed test_constructor_three_args. Retrieved 3/4 statements.
# Partially parsed test_constructor_four_args. Retrieved 4/6 statements.
# Partially parsed test_constructor_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_step_zero. Retrieved 3/5 statements.
# Partially parsed test_constructor_length_calculation_positive_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_length_calculation_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_length_calculation_fractional_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_same_start_stop_positive_step. Retrieved 2/3 statements.
# Partially parsed test_constructor_same_start_stop_negative_step. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = [var_0, var_0, var_1]

def test_case_0():
    var_0 = 5
    var_1 = -1
    var_2 = [var_0, var_0, var_1]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_constructor_with_generator. Retrieved 2/4 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = bool(not var_1.exhausted)
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.list
    var_4 = bool(var_2.list == [])
    assert var_4 is True
    var_5 = bool(not var_2.exhausted)
    assert var_5 is True

def test_case_0():
    var_0 = 3
    var_1 = range(var_0)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = bool(not var_1.exhausted)
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_getitem_with_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_and_step. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_full_slice. Retrieved 7/9 statements.
# Partially parsed test_getitem_with_negative_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_out_of_bounds. Retrieved 4/6 statements.
# Partially parsed test_getitem_with_slice_and_negative_step. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = 5
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_4, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 5
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 0
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_2, var_3, var_4, var_5, var_6]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -3
    var_3 = -1
    var_4 = 7
    var_5 = 8
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 10
    var_3 = 20
    var_4 = []

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 5
    var_3 = 1
    var_4 = -1
    var_5 = 4
    var_6 = 3
    var_7 = 2
    var_8 = [var_2, var_5, var_6, var_7]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_raises_no_error_but_length_calculation. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



