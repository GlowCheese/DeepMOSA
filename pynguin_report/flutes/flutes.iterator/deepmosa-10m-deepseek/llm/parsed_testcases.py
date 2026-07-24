####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_getitem_with_positive_index. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_negative_index. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_full_slice. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_slice_and_step. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_negative_start. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_slice_negative_stop. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_slice_negative_step. Retrieved 9/11 statements.
# Partially parsed test_getitem_with_index_out_of_range. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_negative_index_out_of_range. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_single_argument_range. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_two_argument_range. Retrieved 4/6 statements.
# Partially parsed test_getitem_with_step_one. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_large_step. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_slice_empty_result. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_slice_out_of_bounds. Retrieved 5/7 statements.


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
    var_4 = -3
    var_5 = 5
    var_6 = 7
    var_7 = 9
    var_8 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -1
    var_5 = 3
    var_6 = 5
    var_7 = 7
    var_8 = [var_0, var_5, var_6, var_7]

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
    var_3 = 3

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]
    var_3 = 4
    var_4 = 6

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 2

def test_case_0():
    var_0 = 0
    var_1 = 20
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 15

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 20
    var_5 = []



# Parsed testcases at query #3
#--------------------------





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


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_0, var_0, var_3, var_4, var_0]
    var_6 = True
    var_7 = lambda x: x == var_0
    var_8 = module_0.split_by(var_5, var_6, criterion=var_7)
    var_9 = list(var_8)
    var_10 = []
    var_11 = [var_6, var_2]
    var_12 = []
    var_13 = [var_3, var_4]
    var_14 = []
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = bool(var_9 == var_15)
    assert var_16 is True


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_0, var_0, var_3, var_4, var_0]
    var_6 = False
    var_7 = lambda x: x == var_6
    var_8 = module_0.split_by(var_5, var_6, criterion=var_7)
    var_9 = list(var_8)
    var_10 = [var_1, var_2]
    var_11 = [var_3, var_4]
    var_12 = [var_10, var_11]
    var_13 = bool(var_9 == var_12)
    assert var_13 is True


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


def test_case_0():
    var_0 = ' Split by: '
    var_1 = True
    var_2 = ' '
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


def test_case_0():
    var_0 = ' Split by: '
    var_1 = False
    var_2 = ' '
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = 'S'
    var_6 = 'p'
    var_7 = 'l'
    var_8 = 'i'
    var_9 = 't'
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = 'b'
    var_12 = 'y'
    var_13 = ':'
    var_14 = [var_11, var_12, var_13]
    var_15 = [var_10, var_14]
    var_16 = bool(var_4 == var_15)
    assert var_16 is True


def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = lambda x: x is var_1
    var_3 = module_0.split_by(var_0, criterion=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda x: x == var_4
    var_6 = module_0.split_by(var_3, criterion=var_5)
    var_7 = list(var_6)
    var_8 = [var_0, var_1, var_2]
    var_9 = [var_8]
    var_10 = bool(var_7 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 0
    var_1 = [var_0, var_0, var_0]
    var_2 = lambda x: x == var_0
    var_3 = module_0.split_by(var_1, criterion=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = 0
    var_1 = [var_0, var_0, var_0]
    var_2 = True
    var_3 = lambda x: x == var_0
    var_4 = module_0.split_by(var_1, var_2, criterion=var_3)
    var_5 = list(var_4)
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = [var_6, var_7, var_8, var_9]
    var_11 = bool(var_5 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.split_by(var_3)
    var_5 = list(var_4)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x > var_0
    var_5 = module_0.split_by(var_3, criterion=var_4, separator=var_1)
    var_6 = list(var_5)


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_1, var_2, var_1, var_3]
    var_5 = module_0.split_by(var_4, separator=var_1)
    var_6 = list(var_5)
    var_7 = [var_0]
    var_8 = [var_2]
    var_9 = [var_3]
    var_10 = [var_7, var_8, var_9]
    var_11 = bool(var_6 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = -1
    var_1 = 0
    var_2 = 1
    var_3 = -2
    var_4 = 2
    var_5 = [var_0, var_1, var_2, var_3, var_1, var_4]
    var_6 = lambda x: x <= var_1
    var_7 = module_0.split_by(var_5, criterion=var_6)
    var_8 = list(var_7)
    var_9 = [var_2]
    var_10 = [var_4]
    var_11 = [var_9, var_10]
    var_12 = bool(var_8 == var_11)
    assert var_12 is True


def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_1, var_2, var_1, var_3]
    var_5 = iter(var_4)
    var_6 = lambda x: x == var_1
    var_7 = module_0.split_by(var_5, criterion=var_6)
    var_8 = list(var_7)
    var_9 = [var_0]
    var_10 = [var_2]
    var_11 = [var_3]
    var_12 = [var_9, var_10, var_11]
    var_13 = bool(var_8 == var_12)
    assert var_13 is True



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_getitem_with_negative_index. Retrieved 7/8 statements.
# Partially parsed test_getitem_with_slice_with_negative_stop. Retrieved 15/16 statements.
# Partially parsed test_getitem_with_slice_with_start_only. Retrieved 13/14 statements.
# Partially parsed test_getitem_with_slice_on_exhausted_list. Retrieved 11/12 statements.
# Partially parsed test_getitem_after_exhaustion. Retrieved 7/8 statements.



def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = var_2[var_3]
    var_5 = 5
    var_6 = bool(var_4 == var_5)
    assert var_6 is True
    var_7 = var_2.list
    var_8 = len(var_7)
    assert var_8 == 6


def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 0
    var_4 = var_2[var_3]
    var_5 = 0
    var_6 = bool(var_4 == var_5)
    assert var_6 is True
    var_7 = var_2.list
    var_8 = len(var_7)
    assert var_8 == 1


def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 4
    var_4 = var_2[var_3]
    var_5 = 4
    var_6 = bool(var_4 == var_5)
    assert var_6 is True
    var_7 = var_2.list
    var_8 = len(var_7)
    assert var_8 == 5
    var_9 = var_2.exhausted
    assert var_9 is True


def test_case_0():
    var_0 = 3
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = var_2[var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = var_2.list
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = var_2.exhausted
    assert var_8 is True


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = -1
    var_5 = var_2[var_4]
    var_6 = 9
    var_7 = bool(var_5 == var_6)
    assert var_7 is True


def test_case_0():
    var_0 = 20
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 2
    var_4 = 10
    var_5 = var_2[var_3:var_4:var_3]
    var_6 = 4
    var_7 = 6
    var_8 = 8
    var_9 = [var_3, var_6, var_7, var_8]
    var_10 = bool(var_5 == var_9)
    assert var_10 is True
    var_11 = var_2.list
    var_12 = len(var_11)
    assert var_12 == 10


def test_case_0():
    var_0 = 15
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
    var_12 = var_2.list
    var_13 = len(var_12)
    assert var_13 == 5


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = -2
    var_5 = var_2[:var_4]
    var_6 = 0
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = 4
    var_11 = 5
    var_12 = 6
    var_13 = 7
    var_14 = [var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13]
    var_15 = bool(var_5 == var_14)
    assert var_15 is True


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 3
    var_5 = var_2[var_4:]
    var_6 = 4
    var_7 = 5
    var_8 = 6
    var_9 = 7
    var_10 = 8
    var_11 = 9
    var_12 = [var_4, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = bool(var_5 == var_12)
    assert var_13 is True


def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 10
    var_4 = var_2[:var_3]
    var_5 = 0
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = 4
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = bool(var_4 == var_10)
    assert var_11 is True
    var_12 = var_2.list
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = var_2.exhausted
    assert var_14 is True


def test_case_0():
    var_0 = 7
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 1
    var_5 = 6
    var_6 = 2
    var_7 = var_2[var_4:var_5:var_6]
    var_8 = 3
    var_9 = 5
    var_10 = [var_4, var_8, var_9]
    var_11 = bool(var_7 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 0
    var_4 = var_2[var_3]
    assert var_4 == 0
    var_5 = 1
    var_6 = var_2[var_5]
    assert var_6 == 1
    var_7 = 2
    var_8 = var_2[var_7]
    assert var_8 == 2
    var_9 = var_2.list
    var_10 = len(var_9)
    assert var_10 == 3


def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = 0
    var_3 = var_1[var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = var_1.exhausted
    assert var_5 is True
    var_6 = var_1.list
    var_7 = len(var_6)
    assert var_7 == 0


def test_case_0():
    var_0 = 3
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 2
    var_5 = var_2[var_4]
    var_6 = 2
    var_7 = bool(var_5 == var_6)
    assert var_7 is True



# Parsed testcases at query #6
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
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_getitem_with_positive_index. Retrieved 4/6 statements.
# Partially parsed test_getitem_with_negative_index. Retrieved 4/6 statements.
# Partially parsed test_getitem_with_slice. Retrieved 4/6 statements.
# Partially parsed test_getitem_with_slice_and_step. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_full_slice. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_slice_negative_indices. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_slice_out_of_bounds. Retrieved 4/6 statements.
# Partially parsed test_getitem_with_index_out_of_bounds_positive. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_index_out_of_bounds_negative. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_step_one_range. Retrieved 2/4 statements.
# Partially parsed test_getitem_with_step_one_range_slice. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_negative_step_range. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_negative_step_range_slice. Retrieved 5/7 statements.


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
    var_4 = 0

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

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 3



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_division. Retrieved 3/4 statements.


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
    var_0 = 0
    var_1 = 5
    var_2 = 2
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #9
#--------------------------





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


def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = 3
    var_4 = 4
    var_5 = [var_0, var_3, var_4]
    var_6 = module_0.drop_until(var_2, var_5)
    var_7 = list(var_6)
    var_8 = [var_0, var_3, var_4]
    var_9 = bool(var_7 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x < var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = []
    var_9 = bool(var_7 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = 'l'
    var_1 = lambda c: c == var_0
    var_2 = 'hello'
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = 'o'
    var_6 = [var_0, var_0, var_5]
    var_7 = bool(var_4 == var_6)
    assert var_7 is True


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


def test_case_0():
    var_0 = lambda x: x
    var_1 = False
    var_2 = True
    var_3 = [var_1, var_1, var_2, var_1]
    var_4 = module_0.drop_until(var_0, var_3)
    var_5 = list(var_4)
    var_6 = [var_2, var_1]
    var_7 = bool(var_5 == var_6)
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_constructor_with_generator.



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
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False


def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False


def test_case_0():
    var_0 = 7
    var_1 = 8
    var_2 = 9
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_getitem_with_positive_index. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_negative_index. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_and_step. Retrieved 7/9 statements.
# Partially parsed test_getitem_with_full_slice. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_out_of_range_index_positive. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_out_of_range_index_negative. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_single_argument_range. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_two_argument_range. Retrieved 3/5 statements.
# Partially parsed test_getitem_with_negative_step_slice. Retrieved 11/13 statements.


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
    var_4 = 0
    var_5 = 4
    var_6 = 5
    var_7 = [var_0, var_6]

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
    var_3 = 3

def test_case_0():
    var_0 = 2
    var_1 = 8
    var_2 = [var_0, var_1]
    var_3 = 4

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 0
    var_6 = -1
    var_7 = 9
    var_8 = 7
    var_9 = 5
    var_10 = 3
    var_11 = [var_7, var_8, var_9, var_10]



# Parsed testcases at query #12
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

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_take_with_generator. Retrieved 8/11 statements.
# Partially parsed test_take_large_n_with_infinite_iterator. Retrieved 7/11 statements.



def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.take(var_0, var_5)
    var_7 = list(var_6)
    var_8 = [var_1, var_2, var_0]
    var_9 = bool(var_7 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = []
    var_8 = bool(var_6 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = bool(var_6 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0.take(var_0, var_1)
    var_3 = list(var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True


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
    var_2 = iter(var_1)
    var_3 = 3
    var_4 = module_0.take(var_3, var_2)
    var_5 = list(var_4)
    var_6 = 0
    var_7 = 1
    var_8 = 2
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(var_5 == var_9)
    assert var_10 is True
    var_11 = list(var_2)
    var_12 = bool(var_11 == [3, 4])
    assert var_12 is True

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 4
    var_3 = 0
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_3, var_4, var_5, var_6]


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4, var_0]
    var_6 = module_0.take(var_0, var_5)
    var_7 = list(var_6)
    var_8 = [var_1, var_2, var_3, var_4, var_0]
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_1, var_2, var_3, var_4, var_5]



# Parsed testcases at query #14
#--------------------------





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



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_constructor_no_args_raises_value_error.
# Partially parsed test_constructor_four_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_one_arg. Retrieved 1/2 statements.
# Partially parsed test_constructor_two_args. Retrieved 2/3 statements.
# Partially parsed test_constructor_three_args. Retrieved 3/4 statements.
# Partially parsed test_constructor_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_zero_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_length_calculation_positive_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_length_calculation_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_length_calculation_step_one. Retrieved 2/3 statements.
# Partially parsed test_constructor_val_initialized_to_start. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

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
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 3
    var_1 = 7
    var_2 = [var_0, var_1]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_range_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_range_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_range_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Partially parsed test_range_constructor_with_negative_step. Retrieved 3/4 statements.
# Failed to parse test_range_constructor_with_zero_args_raises_value_error.
# Partially parsed test_range_constructor_with_four_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_range_constructor_with_step_zero_should_not_raise_error_but_length_calculation. Retrieved 3/4 statements.


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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_here_but_length_calculation. Retrieved 3/4 statements.


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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_error_but_length_calculation. Retrieved 3/4 statements.


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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_negative_index_out_of_range. Retrieved 2/4 statements.
# Partially parsed test_slice_with_negative_start_and_stop. Retrieved 6/8 statements.
# Partially parsed test_slice_with_negative_step. Retrieved 8/10 statements.
# Partially parsed test_slice_start_negative_out_of_range. Retrieved 7/9 statements.
# Partially parsed test_slice_stop_negative_out_of_range. Retrieved 4/6 statements.
# Partially parsed test_slice_with_all_negative_indices. Retrieved 7/9 statements.
# Partially parsed test_slice_start_negative_stop_positive. Retrieved 5/7 statements.
# Partially parsed test_slice_start_positive_stop_negative. Retrieved 8/10 statements.
# Partially parsed test_slice_with_large_negative_step. Retrieved 7/9 statements.
# Partially parsed test_slice_negative_start_exceeds_length. Retrieved 7/9 statements.
# Partially parsed test_slice_negative_stop_exceeds_length. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -11

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -3
    var_3 = -1
    var_4 = 7
    var_5 = 8
    var_6 = [var_4, var_5]

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
    var_0 = 5
    var_1 = [var_0]
    var_2 = -10
    var_3 = -2
    var_4 = 0
    var_5 = 1
    var_6 = 2
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 1
    var_3 = -10
    var_4 = []

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -5
    var_3 = -2
    var_4 = 5
    var_5 = 6
    var_6 = 7
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -3
    var_3 = 8
    var_4 = 7
    var_5 = [var_4]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 3
    var_3 = -2
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = 7
    var_8 = [var_2, var_4, var_5, var_6, var_7]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 8
    var_3 = 2
    var_4 = -2
    var_5 = 6
    var_6 = 4
    var_7 = [var_2, var_5, var_6]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = -10
    var_3 = 3
    var_4 = 0
    var_5 = 1
    var_6 = 2
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 2
    var_3 = -10
    var_4 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_but_allow. Retrieved 3/4 statements.
# Partially parsed test_constructor_start_equal_stop_with_positive_step. Retrieved 2/3 statements.
# Partially parsed test_constructor_start_equal_stop_with_negative_step. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_large_numbers. Retrieved 3/4 statements.


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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_getitem_with_slice. Retrieved 17/28 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = 5
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = 1
    var_8 = [var_7, var_0, var_2]
    var_9 = [var_4, var_3]
    var_10 = [var_3]
    var_11 = 0
    var_12 = [var_11, var_7, var_2]
    var_13 = [var_4, var_5]
    var_14 = [var_11, var_2, var_5]
    var_15 = -1
    var_16 = [var_5, var_4, var_2, var_7, var_11]
    var_17 = [var_7, var_0]
    var_18 = 8
    var_19 = 7
    var_20 = [var_4, var_3, var_19]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_raises_error. Retrieved 3/5 statements.
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_constructor_with_tuple. Retrieved 4/6 statements.



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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)


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
    var_8 = bool(var_4.list is var_3)
    assert var_8 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_getitem_with_slice_returns_list_of_indices. Retrieved 6/8 statements.
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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_slice_returns_list. Retrieved 3/6 statements.
# Partially parsed test_slice_with_start_only. Retrieved 2/5 statements.
# Partially parsed test_slice_with_stop_only. Retrieved 2/5 statements.
# Partially parsed test_slice_with_step. Retrieved 4/7 statements.
# Partially parsed test_slice_negative_indices. Retrieved 3/6 statements.
# Partially parsed test_slice_reverse. Retrieved 2/5 statements.
# Partially parsed test_slice_empty. Retrieved 3/6 statements.
# Partially parsed test_slice_full_range. Retrieved 1/4 statements.
# Partially parsed test_slice_with_step_negative. Retrieved 4/7 statements.
# Partially parsed test_slice_with_large_range. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = 5

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 5

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 1
    var_3 = 8
    var_4 = 2

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -3
    var_3 = -1

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -1

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 5
    var_3 = 2

def test_case_0():
    var_0 = 10
    var_1 = [var_0]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 8
    var_3 = 1
    var_4 = -2

def test_case_0():
    var_0 = 1
    var_1 = 100
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = 20



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_raises_no_error_but_length_calculation. Retrieved 3/4 statements.
# Partially parsed test_constructor_start_equal_stop_with_positive_step. Retrieved 2/3 statements.
# Partially parsed test_constructor_start_equal_stop_with_negative_step. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_large_numbers. Retrieved 3/4 statements.


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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_getitem_with_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_and_step. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_full_slice. Retrieved 7/9 statements.
# Partially parsed test_getitem_with_negative_slice. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_out_of_bounds. Retrieved 4/6 statements.
# Partially parsed test_getitem_with_slice_and_negative_step. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_slice_on_empty_range. Retrieved 2/4 statements.
# Partially parsed test_getitem_with_slice_and_step_zero. Retrieved 2/5 statements.
# Partially parsed test_getitem_with_slice_indices. Retrieved 7/9 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0]
    var_2 = []

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 3
    var_6 = 7
    var_7 = [var_5, var_6]



# Parsed testcases at query #28
#--------------------------





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
    var_10 = bool(var_6.list is var_5)
    assert var_10 is True


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
    var_8 = bool(var_4.list is var_3)
    assert var_8 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda c: c * var_0
    var_2 = 'abc'
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func is var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list is var_2)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_constructor_with_generator.



def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True


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


def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True


def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.exhausted
    assert var_3 is False
    var_4 = var_2.list
    var_5 = bool(var_2.list == [])
    assert var_5 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_take_with_generator. Retrieved 3/6 statements.



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


def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0.take(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True


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
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = iter(var_3)
    var_5 = 2
    var_6 = module_0.take(var_5, var_4)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [10, 20])
    assert var_8 is True

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = 3


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
    var_9 = list(var_6)
    var_10 = bool(var_8 == [1, 2])
    assert var_10 is True
    var_11 = bool(var_9 == [3, 4, 5])
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_constructor_with_generator.



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
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False


def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False


def test_case_0():
    var_0 = 7
    var_1 = 8
    var_2 = 9
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False


def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False



# Parsed testcases at query #3
#--------------------------





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


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda x: x == var_4
    var_6 = module_0.split_by(var_3, criterion=var_5)
    var_7 = list(var_6)
    var_8 = [var_0, var_1, var_2]
    var_9 = [var_8]
    var_10 = bool(var_7 == var_9)
    assert var_10 is True


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


def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = lambda x: x == var_0
    var_3 = module_0.split_by(var_1, criterion=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.split_by(var_1, separator=var_0)
    var_3 = list(var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True


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


def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = lambda x: x is var_1
    var_3 = module_0.split_by(var_0, criterion=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = []
    var_5 = bool(var_3 == var_4)
    assert var_5 is True


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


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 0
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = lambda x: x == var_1
    var_5 = module_0.split_by(var_3, criterion=var_4)
    var_6 = list(var_5)
    var_7 = [var_0]
    var_8 = [var_2]
    var_9 = [var_7, var_8]
    var_10 = bool(var_6 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = module_0.split_by(var_3, separator=var_1)
    var_5 = list(var_4)
    var_6 = [var_0]
    var_7 = [var_2]
    var_8 = [var_6, var_7]
    var_9 = bool(var_5 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = True
    var_5 = lambda x: x == var_1
    var_6 = module_0.split_by(var_3, var_4, criterion=var_5)
    var_7 = list(var_6)
    var_8 = [var_4]
    var_9 = []
    var_10 = []
    var_11 = [var_2]
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = bool(var_7 == var_12)
    assert var_13 is True


def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = True
    var_5 = module_0.split_by(var_3, var_4, separator=var_1)
    var_6 = list(var_5)
    var_7 = [var_4]
    var_8 = []
    var_9 = []
    var_10 = [var_2]
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = bool(var_6 == var_11)
    assert var_12 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.split_by(var_3)
    var_5 = list(var_4)


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

# Partially parsed test_drop_generator. Retrieved 3/6 statements.



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


def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True


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


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = iter(var_1)
    var_3 = 3
    var_4 = module_0.drop(var_3, var_2)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [3, 4, 5, 6, 7, 8, 9])
    assert var_6 is True
    var_7 = list(var_2)
    var_8 = bool(var_7 == [])
    assert var_8 is True


def test_case_0():
    var_0 = 5
    var_1 = 1000000
    var_2 = range(var_1)
    var_3 = module_0.drop(var_0, var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 999995
    var_6 = var_4[0]
    assert var_6 == 5


def test_case_0():
    var_0 = 2
    var_1 = 'hello'
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['l', 'l', 'o'])
    assert var_4 is True


def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = (var_1, var_2, var_3)
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [20, 30])
    assert var_7 is True

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = 2



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_getitem_with_single_index_positive. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_single_index_negative. Retrieved 5/7 statements.
# Partially parsed test_getitem_with_slice_full. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_slice_partial. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_negative_indices. Retrieved 8/10 statements.
# Partially parsed test_getitem_with_slice_step. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_slice_out_of_bounds. Retrieved 7/9 statements.
# Partially parsed test_getitem_with_slice_empty. Retrieved 5/7 statements.
# Partially parsed test_getitem_index_error_positive. Retrieved 4/7 statements.
# Partially parsed test_getitem_index_error_negative. Retrieved 4/7 statements.
# Partially parsed test_getitem_with_step_one. Retrieved 3/5 statements.
# Partially parsed test_getitem_slice_with_step_one. Retrieved 6/8 statements.
# Partially parsed test_getitem_with_negative_step_range. Retrieved 5/7 statements.
# Partially parsed test_getitem_slice_with_negative_step_range. Retrieved 8/10 statements.


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
    var_6 = 7
    var_7 = 9
    var_8 = [var_0, var_4, var_5, var_6, var_7]

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
    var_5 = 9
    var_6 = [var_0, var_4, var_5]

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
    var_4 = 5
    var_5 = []

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
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 3

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 4
    var_6 = 3
    var_7 = 2
    var_8 = [var_5, var_6, var_7]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_during_init. Retrieved 3/4 statements.
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_four_args_raises_error. Retrieved 4/6 statements.
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



# Parsed testcases at query #8
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



# Parsed testcases at query #9
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
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_four_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_error_but_length_calculation. Retrieved 3/4 statements.


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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_constructor_with_builtin_function_and_tuple. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_custom_function_and_range. Retrieved 2/5 statements.



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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)


def test_case_0():
    var_0 = None
    var_1 = lambda x: var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func is var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == var_2)
    assert var_7 is True


def test_case_0():
    var_0 = lambda x: x
    var_1 = 'abc'
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list == var_1)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------





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



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_getitem_with_int_index_negative. Retrieved 7/8 statements.
# Partially parsed test_getitem_with_slice_stop_beyond_exhausted. Retrieved 11/12 statements.
# Partially parsed test_getitem_int_index_raises_index_error_when_exhausted. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = var_2[var_3]
    var_5 = 5
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 3
    var_4 = var_2[var_3]
    var_5 = var_2.list
    var_6 = len(var_5)
    assert var_6 == 4


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = -1
    var_5 = var_2[var_4]
    var_6 = 9
    var_7 = bool(var_5 == var_6)
    assert var_7 is True


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 2
    var_4 = 5
    var_5 = var_2[var_3:var_4]
    var_6 = 3
    var_7 = 4
    var_8 = [var_3, var_6, var_7]
    var_9 = bool(var_5 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = None
    var_4 = 1
    var_5 = 10
    var_6 = var_2[var_4:var_5]
    var_7 = 2
    var_8 = 3
    var_9 = 4
    var_10 = [var_4, var_7, var_8, var_9]
    var_11 = bool(var_6 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 1
    var_4 = -2
    var_5 = var_2[var_3:var_4]
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 5
    var_10 = 6
    var_11 = 7
    var_12 = [var_3, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = bool(var_5 == var_12)
    assert var_13 is True


def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 2
    var_4 = var_2[var_3:]
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_5, var_6]
    var_8 = bool(var_4 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 1
    var_4 = 8
    var_5 = 2
    var_6 = var_2[var_3:var_4:var_5]
    var_7 = 3
    var_8 = 5
    var_9 = 7
    var_10 = [var_3, var_7, var_8, var_9]
    var_11 = bool(var_6 == var_10)
    assert var_11 is True


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


def test_case_0():
    var_0 = 3
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 5
    var_4 = 10
    var_5 = var_2[var_3:var_4]
    var_6 = []
    var_7 = bool(var_5 == var_6)
    assert var_7 is True


def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = 0
    var_3 = var_1[var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = 0
    var_3 = 5
    var_4 = var_1[var_2:var_3]
    var_5 = []
    var_6 = bool(var_4 == var_5)
    assert var_6 is True


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 1
    var_4 = -2
    var_5 = var_2[var_3:var_4]
    var_6 = var_2.exhausted
    assert var_6 is True


def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 2
    var_4 = var_2[var_3:]
    var_5 = var_2.exhausted
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_getitem_ensures_lazy_evaluation_per_call. Retrieved 6/13 statements.



def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5[var_2]
    assert var_6 == 4


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


def test_case_0():
    var_0 = 1
    var_1 = lambda x: x - var_0
    var_2 = 5
    var_3 = 10
    var_4 = 15
    var_5 = 20
    var_6 = 25
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.MapList(var_1, var_7)
    var_9 = 2
    var_10 = var_8[::var_9]
    var_11 = bool(var_10 == [4, 14, 24])
    assert var_11 is True


def test_case_0():
    var_0 = 3
    var_1 = lambda x: x * var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = 0
    var_5 = 5
    var_6 = var_3[var_4:var_5]
    var_7 = bool(var_6 == [])
    assert var_7 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x / var_0
    var_2 = 100
    var_3 = 200
    var_4 = [var_2, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = 5
    var_7 = var_5[var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: (x, x * var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = 0
    var_7 = var_5[var_6]
    var_8 = bool(var_7 == ('a', 'aa'))
    assert var_8 is True


def test_case_0():
    var_0 = lambda x: len(x)
    var_1 = 'apple'
    var_2 = 'banana'
    var_3 = 'cherry'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.MapList(var_0, var_4)
    var_6 = -2
    var_7 = -1
    var_8 = var_5[var_6:var_7]
    var_9 = bool(var_8 == [6])
    assert var_9 is True

def test_case_0():
    var_0 = 0
    assert var_0 == 2
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_constructor_with_tuple_and_function. Retrieved 4/7 statements.
# Partially parsed test_constructor_with_range. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_string_sequence. Retrieved 1/3 statements.



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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)


def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list is var_1)
    assert var_6 is True

def test_case_0():
    var_0 = 'abc'



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_constructor_with_generator.



def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True


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


def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True


def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.exhausted
    assert var_3 is False
    var_4 = var_2.list
    var_5 = bool(var_2.list == [])
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_constructor_with_generator. Retrieved 2/4 statements.



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


def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 300
    var_3 = {var_0, var_1, var_2}
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False


def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False


def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.list
    var_4 = bool(var_2.list == [])
    assert var_4 is True
    var_5 = var_2.exhausted
    assert var_5 is False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_error_during_init. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_equal_to_stop. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_greater_than_stop_and_positive_step. Retrieved 2/3 statements.


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

def test_case_0():
    var_0 = -5
    var_1 = 0
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 7
    var_1 = [var_0, var_0]

def test_case_0():
    var_0 = 8
    var_1 = 3
    var_2 = [var_0, var_1]



# Parsed testcases at query #21
#--------------------------





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


def test_case_0():
    var_0 = lambda x: x * x
    var_1 = 5
    var_2 = range(var_1)
    var_3 = module_0.MapList(var_0, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func is var_0)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == var_2)
    assert var_7 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda c: c * var_0
    var_2 = 'abc'
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func is var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == var_2)
    assert var_7 is True



# Parsed testcases at query #22
#--------------------------





def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x >= var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [0, 1, 2, 3, 4])
    assert var_6 is True


def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True


def test_case_0():
    var_0 = 'b'
    var_1 = lambda s: s == var_0
    var_2 = 'a'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_2, var_0, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['b', 'c', 'd'])
    assert var_8 is True


def test_case_0():
    var_0 = 4
    var_1 = lambda x: x == var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [4])
    assert var_6 is True


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
    var_10 = bool(var_9 == [3, 4, 5])
    assert var_10 is True
    var_11 = list(var_6)
    var_12 = bool(var_11 == [])
    assert var_12 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 7/9 statements.



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
    assert var_7 == 4



# Parsed testcases at query #24
#--------------------------





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
    var_8 = lambda x: x > var_7
    var_9 = 1
    var_10 = 2
    var_11 = [var_7, var_9, var_10]
    var_12 = module_0.drop_until(var_8, var_11)
    var_13 = list(var_12)
    var_14 = bool(var_13 == [1, 2])
    assert var_14 is True
    var_15 = 'a'
    var_16 = lambda x: x == var_15
    var_17 = 'b'
    var_18 = 'c'
    var_19 = 'd'
    var_20 = [var_17, var_18, var_15, var_19]
    var_21 = module_0.drop_until(var_16, var_20)
    var_22 = list(var_21)
    var_23 = bool(var_22 == ['a', 'd'])
    assert var_23 is True
    var_24 = None
    var_25 = lambda x: x is var_24
    var_26 = 3
    var_27 = [var_9, var_10, var_26]
    var_28 = module_0.drop_until(var_25, var_27)
    var_29 = list(var_28)
    var_30 = bool(var_29 == [])
    assert var_30 is True
    var_31 = lambda x: x
    var_32 = False
    var_33 = False
    var_34 = True
    var_35 = False
    var_36 = [var_32, var_33, var_34, var_35]
    var_37 = module_0.drop_until(var_31, var_36)
    var_38 = list(var_37)
    var_39 = bool(var_38 == [True, False])
    assert var_39 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_division. Retrieved 3/4 statements.
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
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_constructor_with_named_function_and_tuple. Retrieved 4/7 statements.
# Partially parsed test_constructor_with_string_method_and_range. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_empty_list. Retrieved 4/5 statements.



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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]


def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = 5
    var_4 = var_2.list
    var_5 = bool(var_2.list == [])
    assert var_5 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = var_5[var_0:var_3]
    var_7 = -1
    var_8 = lambda x: x * var_7
    var_9 = module_0.MapList(var_8, var_6)
    var_10 = var_9.list
    var_11 = bool(var_9.list == [2, 3, 4])
    assert var_11 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_getitem_with_slice. Retrieved 22/34 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = 5
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = 1
    var_8 = [var_7, var_0, var_2]
    var_9 = [var_4, var_3]
    var_10 = [var_3]
    var_11 = 0
    var_12 = [var_11, var_7, var_2]
    var_13 = [var_4, var_5]
    var_14 = [var_11, var_2, var_5]
    var_15 = [var_7, var_0]
    var_16 = -3
    var_17 = 7
    var_18 = 8
    var_19 = 9
    var_20 = [var_17, var_18, var_19]
    var_21 = -2
    var_22 = 6
    var_23 = [var_4, var_5, var_3, var_22, var_17]
    var_24 = -1
    var_25 = [var_19, var_18, var_17, var_22, var_3, var_5, var_4, var_2, var_7]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_negative_index_conversion. Retrieved 3/5 statements.
# Partially parsed test_negative_index_with_start. Retrieved 4/6 statements.
# Partially parsed test_negative_index_with_step. Retrieved 5/7 statements.
# Partially parsed test_negative_index_zero_length. Retrieved 2/5 statements.
# Partially parsed test_negative_index_out_of_bounds. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -1
    var_3 = 9

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = [var_0, var_1]
    var_3 = -3
    var_4 = 12

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = -2
    var_5 = 7

def test_case_0():
    var_0 = 5
    var_1 = [var_0, var_0]
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -11
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #29
#--------------------------





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


def test_case_0():
    var_0 = lambda x: x * x
    var_1 = 5
    var_2 = range(var_1)
    var_3 = module_0.MapList(var_0, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func is var_0)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == var_2)
    assert var_7 is True


def test_case_0():
    var_0 = lambda x: ord(x)
    var_1 = 'abc'
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list == var_1)
    assert var_6 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_negative_index_out_of_range_raises_index_error. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = False
    var_3 = -11
    var_4 = True
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_constructor_with_tuple. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_string_sequence. Retrieved 1/3 statements.



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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)


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
    var_8 = bool(var_4.list is var_3)
    assert var_8 is True


def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list == var_1)
    assert var_6 is True

def test_case_0():
    var_0 = 'abc'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_error_but_length_calculation. Retrieved 3/4 statements.


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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_error.
# Partially parsed test_constructor_with_four_args_raises_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_error_here. Retrieved 3/4 statements.


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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_negative_index_out_of_range. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -11



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
    var_0 = 10
    var_1 = 0
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_constructor_with_generator. Retrieved 2/4 statements.



def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True


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

def test_case_0():
    var_0 = 3
    var_1 = range(var_0)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_negative_index_handling. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -1
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_0, var_4]
    var_6 = -2



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True



# Parsed testcases at query #40
#--------------------------





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


def test_case_0():
    var_0 = lambda x: ord(x)
    var_1 = 'abc'
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    assert var_5 == 'abc'



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_constructor_with_generator.



def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True


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


def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.exhausted
    assert var_2 is False
    var_3 = var_1.list
    var_4 = bool(var_1.list == [])
    assert var_4 is True


def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.exhausted
    assert var_3 is False
    var_4 = var_2.list
    var_5 = bool(var_2.list == [])
    assert var_5 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_constructor_with_stop_only. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_start_stop_and_step. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_step. Retrieved 3/4 statements.
# Failed to parse test_constructor_with_zero_args_raises_value_error.
# Partially parsed test_constructor_with_more_than_three_args_raises_value_error. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_step_zero_should_not_raise_immediately. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_start_and_stop. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_all_negative. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_large_numbers. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = -5
    var_1 = 0
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = -10
    var_1 = -20
    var_2 = -2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1000
    var_1 = 2000
    var_2 = 100
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #43
#--------------------------





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
    var_11 = 3
    var_12 = [var_9, var_10, var_7, var_11]
    var_13 = module_0.drop_until(var_8, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [0, 3])
    assert var_15 is True
    var_16 = lambda x: x
    var_17 = False
    var_18 = False
    var_19 = True
    var_20 = False
    var_21 = [var_17, var_18, var_19, var_20]
    var_22 = module_0.drop_until(var_16, var_21)
    var_23 = list(var_22)
    var_24 = bool(var_23 == [True, False])
    assert var_24 is True
    var_25 = lambda x: x % var_10 == var_20
    var_26 = 6
    var_27 = 7
    var_28 = 8
    var_29 = [var_19, var_11, var_0, var_26, var_27, var_28]
    var_30 = module_0.drop_until(var_25, var_29)
    var_31 = list(var_30)
    var_32 = bool(var_31 == [6, 7, 8])
    assert var_32 is True
    var_33 = 'a'
    var_34 = lambda x: x == var_33
    var_35 = 'b'
    var_36 = 'c'
    var_37 = 'd'
    var_38 = [var_35, var_36, var_33, var_37]
    var_39 = module_0.drop_until(var_34, var_38)
    var_40 = list(var_39)
    var_41 = bool(var_40 == ['a', 'd'])
    assert var_41 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_getitem_with_slice. Retrieved 20/31 statements.


def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = 5
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_4, var_5]
    var_7 = 1
    var_8 = [var_7, var_0, var_2]
    var_9 = [var_4, var_3]
    var_10 = [var_3]
    var_11 = 0
    var_12 = [var_11, var_7, var_2]
    var_13 = [var_4, var_5]
    var_14 = [var_11, var_2, var_5]
    var_15 = [var_7, var_0]
    var_16 = -3
    var_17 = 7
    var_18 = 8
    var_19 = 9
    var_20 = [var_17, var_18, var_19]
    var_21 = -2
    var_22 = 6
    var_23 = [var_4, var_5, var_3, var_22, var_17]



# Parsed testcases at query #45
#--------------------------





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


def test_case_0():
    var_0 = lambda s: len(s)
    var_1 = 'hello'
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2.func
    var_4 = bool(var_2.func is var_0)
    assert var_4 is True
    var_5 = var_2.list
    var_6 = bool(var_2.list is var_1)
    assert var_6 is True


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


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = (var_2, var_0, var_3, var_4)
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = var_6.func
    var_8 = bool(var_6.func is var_1)
    assert var_8 is True
    var_9 = var_6.list
    var_10 = bool(var_6.list == var_5)
    assert var_10 is True



