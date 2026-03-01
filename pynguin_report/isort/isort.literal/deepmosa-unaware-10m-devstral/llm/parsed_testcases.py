####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 1]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 1)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #2
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]    '
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]    '



# Parsed testcases at query #3
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = module_0.assignments(var_0)
    assert var_1 == 'a = 1\nb = 2\nc = 3'
    var_2 = 'b = 2\n\na = 1\n\nc = 3'
    var_3 = module_0.assignments(var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = 'b = 2  \na = 1  \nc = 3  '
    var_5 = module_0.assignments(var_4)
    assert var_5 == 'a = 1\nb = 2\nc = 3'
    var_6 = "z = [3, 2, 1]\na = {'b': 2, 'a': 1}"
    var_7 = module_0.assignments(var_6)
    assert var_7 == "a = {'b': 2, 'a': 1}\nz = [3, 2, 1]"
    var_8 = "print('hello')"
    var_9 = module_0.assignments(var_8)
    var_10 = 'x := 1'
    var_11 = module_0.assignments(var_10)



# Parsed testcases at query #4
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'a': 2, 'b': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #5
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = module_0.assignments(var_0)
    assert var_1 == 'a = 1\nb = 2\nc = 3'
    var_2 = 'b = 2\n\na = 1\n\nc = 3'
    var_3 = module_0.assignments(var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = 'b = 2\na = 1\nc = 3\n'
    var_5 = module_0.assignments(var_4)
    assert var_5 == 'a = 1\nb = 2\nc = 3\n'
    var_6 = 'b = 2\na = 1\nc = 3'
    var_7 = module_0.assignments(var_6)
    assert var_7 == 'a = 1\nb = 2\nc = 3'
    var_8 = 'b = 2\na = 1\nc'
    var_9 = module_0.assignments(var_8)
    var_10 = ''
    var_11 = module_0.assignments(var_10)
    assert var_11 == ''
    var_12 = 'a = 1'
    var_13 = module_0.assignments(var_12)
    assert var_13 == 'a = 1'



# Parsed testcases at query #6
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #7
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]    '
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]    '



# Parsed testcases at query #8
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = 'my_list = [1, 2, 3, 4, 5]\n'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'my_list = [1, 2, 3, 4, 5]\n'



# Parsed testcases at query #9
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = invalid_literal\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [1, 2, 3]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #10
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #11
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 2, 'a': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #12
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #13
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = ' '
    var_35 = ''
    var_36 = lambda code, ext, cfg: code.replace(var_34, var_35)
    var_37 = module_1.Config()
    var_38 = 'my_list = [3, 1, 2]  '
    var_39 = module_0.assignment(var_38, var_8, var_32, var_37)
    assert var_39 == 'my_list=[1,2,3]'
    var_40 = 'my_list = [3, 1, 2]  \n'
    var_41 = module_0.assignment(var_40, var_8, var_32)
    assert var_41 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #14
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {1: 'a', 2: 'b', 3: 'c'}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {1: 'a', 2: 'b', 3: 'c'}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #15
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'c': 2, 'a': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 1, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 1, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #16
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {1: 'a', 2: 'b', 3: 'c'}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #17
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'a': 2, 'b': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'a': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #18
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #19
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'a': 2, 'b': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #20
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = 'my_dict = {1: 3, 2: 1, 3: 2}'
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == 'my_dict = {1: 3, 2: 1, 3: 2}'
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #21
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]   '
    var_23 = module_0.assignment(var_22, var_8, var_2)
    assert var_23 == 'my_list = [1, 2, 3]   '
    var_24 = 'my_list = [3, 1, 2]\n'
    var_25 = module_0.assignment(var_24, var_8, var_2)
    assert var_25 == 'my_list = [1, 2, 3]\n'
    var_26 = lambda x, y, z: x.upper()
    var_27 = module_1.Config()
    var_28 = 'my_list = [3, 1, 2]'
    var_29 = module_0.assignment(var_28, var_8, var_2, var_27)
    assert var_29 == 'MY_LIST = [1, 2, 3]'
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'invalid'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'my_list = [3, 1, invalid]'
    var_35 = 'list'
    var_36 = 'py'
    var_37 = module_0.assignment(var_34, var_35, var_36)
    var_38 = 'my_list = [3, 1, 2]'
    var_39 = 'dict'
    var_40 = 'py'
    var_41 = module_0.assignment(var_38, var_39, var_40)



# Parsed testcases at query #22
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'c': 2, 'a': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 1]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 1)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #23
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "x = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "x = {'a': 3, 'b': 1, 'c': 2}"
    var_7 = 'x = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'x = [1, 2, 3]'
    var_10 = 'x = [3, 1, 2, 1, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'x = [1, 2, 3]'
    var_13 = 'x = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'x = {1, 2, 3}'
    var_16 = 'x = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'x = (1, 2, 3)'
    var_19 = 'x = (3, 1, 2, 1, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'x = (1, 2, 3)'
    var_22 = 'x = 1'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'x = 1'
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #24
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #25
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 2, 'a': 1}"
    var_7 = 'my_list = [2, 1]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2]'
    var_10 = 'my_list = [2, 1, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2]'
    var_13 = 'my_set = {2, 1}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2}'
    var_16 = 'my_tuple = (2, 1)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2)'
    var_19 = 'my_tuple = (2, 1, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2)'
    var_22 = 'my_list = [2, 1]'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [2, 1, ]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [2, 1]'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #26
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'a = 1\nb = 2\nc = 3'
    var_2 = 'assignments'
    var_3 = 'py'
    var_4 = module_0.assignment(var_0, var_2, var_3)
    var_5 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_6 = "my_dict = {1: 'a', 2: 'b', 3: 'c'}"
    var_7 = 'dict'
    var_8 = module_0.assignment(var_5, var_7, var_3)
    var_9 = 'my_list = [3, 1, 2]'
    var_10 = 'my_list = [1, 2, 3]'
    var_11 = 'list'
    var_12 = module_0.assignment(var_9, var_11, var_3)
    var_13 = 'my_list = [3, 1, 2, 2, 3]'
    var_14 = 'my_list = [1, 2, 3]'
    var_15 = 'unique-list'
    var_16 = module_0.assignment(var_13, var_15, var_3)
    var_17 = 'my_set = {3, 1, 2}'
    var_18 = 'my_set = {1, 2, 3}'
    var_19 = 'set'
    var_20 = module_0.assignment(var_17, var_19, var_3)
    var_21 = 'my_tuple = (3, 1, 2)'
    var_22 = 'my_tuple = (1, 2, 3)'
    var_23 = 'tuple'
    var_24 = module_0.assignment(var_21, var_23, var_3)
    var_25 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_26 = 'my_tuple = (1, 2, 3)'
    var_27 = 'unique-tuple'
    var_28 = module_0.assignment(var_25, var_27, var_3)
    var_29 = 'my_list = [3, 1, 2]'
    var_30 = 'invalid'
    var_31 = 'py'
    var_32 = module_0.assignment(var_29, var_30, var_31)
    var_33 = 'my_list = [3, 1, invalid]'
    var_34 = 'list'
    var_35 = 'py'
    var_36 = module_0.assignment(var_33, var_34, var_35)
    var_37 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_38 = 'list'
    var_39 = 'py'
    var_40 = module_0.assignment(var_37, var_38, var_39)
    var_41 = lambda code, ext, cfg: code.upper()
    var_42 = module_1.Config()
    var_43 = 'my_list = [3, 1, 2]'
    var_44 = 'MY_LIST = [1, 2, 3]'
    var_45 = module_0.assignment(var_43, var_11, var_38, var_42)



# Parsed testcases at query #27
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda x, y, z: x.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #28
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_var = [1, 2, 3]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_var = [1, 2, 3]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #29
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'x = 1'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid_literal'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "x = {'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda x, _, __: x.upper()
    var_35 = module_1.Config()
    var_36 = 'x = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'X = [1, 2, 3]'
    var_38 = 'x = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'x = [1, 2, 3]  \n'



# Parsed testcases at query #30
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'a': 2, 'b': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}"
    var_7 = 'my_list = [2, 1, 3]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [2, 1, 3, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {2, 1, 3}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (2, 1, 3)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (2, 1, 3, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [2, 1, 3]'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [2, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #31
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #32
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'c': 2, 'a': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_var = 1'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_var = [1, 2, 3]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #33
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_var = [1, 2, 3]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_var = [1, 2, 3]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #34
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #35
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #36
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {1: 'a', 2: 'b', 3: 'c'}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {1: 'a', 2: 'b', 3: 'c'}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #37
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'c': 2, 'a': 3}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2, 3]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_var = 1\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_var = {'a': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #38
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #39
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'c': 2, 'a': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 1, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 1, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   '
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   '



# Parsed testcases at query #40
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'my_list = [3, 1, 2]   \n'
    var_35 = module_0.assignment(var_34, var_8, var_32)
    assert var_35 == 'my_list = [1, 2, 3]   \n'
    var_36 = 40
    var_37 = module_1.Config()
    var_38 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_39 = module_0.assignment(var_38, var_5, var_32, var_37)
    assert var_39 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"



# Parsed testcases at query #41
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 1]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 1)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #42
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #43
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'my_list = [1, 2, 3]'



# Parsed testcases at query #44
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'my_list = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #45
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #46
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {1: 'a', 2: 'b', 3: 'c'}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_37 = module_0.assignment(var_36, var_33, var_31, var_35)
    assert var_37 == "my_dict = {1: 'a', 2: 'b', 3: 'c'}"



# Parsed testcases at query #47
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [2, 1, 3]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [2, 1, 3, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {2, 1, 3}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (2, 1, 3)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (2, 1, 3, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [2, 1, 3]\n'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [2, 1, 3\n'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #48
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #49
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #50
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'a = 1\nb = 2\nc = 3'
    var_2 = 'assignments'
    var_3 = 'py'
    var_4 = module_0.assignment(var_0, var_2, var_3)
    var_5 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_6 = "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'dict'
    var_8 = module_0.assignment(var_5, var_7, var_3)
    var_9 = 'my_list = [3, 1, 2]'
    var_10 = 'my_list = [1, 2, 3]'
    var_11 = 'list'
    var_12 = module_0.assignment(var_9, var_11, var_3)
    var_13 = 'my_list = [3, 1, 2, 2, 3]'
    var_14 = 'my_list = [1, 2, 3]'
    var_15 = 'unique-list'
    var_16 = module_0.assignment(var_13, var_15, var_3)
    var_17 = 'my_set = {3, 1, 2}'
    var_18 = 'my_set = {1, 2, 3}'
    var_19 = 'set'
    var_20 = module_0.assignment(var_17, var_19, var_3)
    var_21 = 'my_tuple = (3, 1, 2)'
    var_22 = 'my_tuple = (1, 2, 3)'
    var_23 = 'tuple'
    var_24 = module_0.assignment(var_21, var_23, var_3)
    var_25 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_26 = 'my_tuple = (1, 2, 3)'
    var_27 = 'unique-tuple'
    var_28 = module_0.assignment(var_25, var_27, var_3)
    var_29 = 'my_list = [3, 1, 2]'
    var_30 = 'invalid_type'
    var_31 = 'py'
    var_32 = module_0.assignment(var_29, var_30, var_31)
    var_33 = 'my_list = [3, 1, invalid]'
    var_34 = 'list'
    var_35 = 'py'
    var_36 = module_0.assignment(var_33, var_34, var_35)
    var_37 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_38 = 'list'
    var_39 = 'py'
    var_40 = module_0.assignment(var_37, var_38, var_39)
    var_41 = lambda code, ext, cfg: code.upper()
    var_42 = module_1.Config()
    var_43 = 'my_list = [3, 1, 2]'
    var_44 = 'MY_LIST = [1, 2, 3]'
    var_45 = module_0.assignment(var_43, var_11, var_38, var_42)



# Parsed testcases at query #51
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #52
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = ''
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = ''
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = ''
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = ''
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = 'py'
    var_38 = module_0.assignment(var_36, var_8, var_37, var_35)
    assert var_38 == 'MY_LIST = [1, 2, 3]'
    var_39 = 'my_list = [3, 1, 2]   \n'
    var_40 = module_0.assignment(var_39, var_8, var_31)
    assert var_40 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #53
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = invalid\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #54
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_var = [1, 2, 3]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_var = [1, 2, 3]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'my_list = [3, 1, 2]   '
    var_35 = module_0.assignment(var_34, var_8, var_31)
    assert var_35 == 'my_list = [1, 2, 3]   '
    var_36 = 40
    var_37 = module_1.Config()
    var_38 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_39 = module_0.assignment(var_38, var_33, var_31, var_37)
    assert var_39 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"



# Parsed testcases at query #55
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 2, 'a': 1}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'x = 1'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid_literal'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "x = {'a': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #56
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'z = 2\nb = 1\na = 3'
    var_1 = 'a = 3\nb = 1\nz = 2'
    var_2 = 'assignments'
    var_3 = 'py'
    var_4 = module_0.assignment(var_0, var_2, var_3)
    var_5 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_6 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_7 = 'dict'
    var_8 = module_0.assignment(var_5, var_7, var_3)
    var_9 = 'my_list = [3, 1, 2]'
    var_10 = 'my_list = [1, 2, 3]'
    var_11 = 'list'
    var_12 = module_0.assignment(var_9, var_11, var_3)
    var_13 = 'my_list = [3, 1, 2, 1, 3]'
    var_14 = 'my_list = [1, 2, 3]'
    var_15 = 'unique-list'
    var_16 = module_0.assignment(var_13, var_15, var_3)
    var_17 = 'my_set = {3, 1, 2}'
    var_18 = 'my_set = {1, 2, 3}'
    var_19 = 'set'
    var_20 = module_0.assignment(var_17, var_19, var_3)
    var_21 = 'my_tuple = (3, 1, 2)'
    var_22 = 'my_tuple = (1, 2, 3)'
    var_23 = 'tuple'
    var_24 = module_0.assignment(var_21, var_23, var_3)
    var_25 = 'my_tuple = (3, 1, 2, 1, 3)'
    var_26 = 'my_tuple = (1, 2, 3)'
    var_27 = 'unique-tuple'
    var_28 = module_0.assignment(var_25, var_27, var_3)
    var_29 = 'my_list = [3, 1, 2]'
    var_30 = 'invalid_type'
    var_31 = 'py'
    var_32 = module_0.assignment(var_29, var_30, var_31)
    var_33 = 'my_list = [3, 1, invalid]'
    var_34 = 'list'
    var_35 = 'py'
    var_36 = module_0.assignment(var_33, var_34, var_35)
    var_37 = "my_dict = {'a': 3, 'b': 1}"
    var_38 = 'list'
    var_39 = 'py'
    var_40 = module_0.assignment(var_37, var_38, var_39)



# Parsed testcases at query #57
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_var = 1'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_var = {'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 50
    var_35 = module_1.Config()
    var_36 = "my_dict = {'b': 2, 'a': 1}"
    var_37 = module_0.assignment(var_36, var_33, var_31, var_35)
    assert var_37 == "my_dict = {'a': 1, 'b': 2}"
    var_38 = 'my_list = [3, 1, 2]   '
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   '



# Parsed testcases at query #58
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'a = 1\nb = 2\nc = 3'
    var_2 = 'assignments'
    var_3 = 'py'
    var_4 = module_0.assignment(var_0, var_2, var_3)
    var_5 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_6 = "my_dict = {1: 'a', 2: 'b', 3: 'c'}"
    var_7 = 'dict'
    var_8 = module_0.assignment(var_5, var_7, var_3)
    var_9 = 'my_list = [3, 1, 2]'
    var_10 = 'my_list = [1, 2, 3]'
    var_11 = 'list'
    var_12 = module_0.assignment(var_9, var_11, var_3)
    var_13 = 'my_list = [3, 1, 2, 2, 3]'
    var_14 = 'my_list = [1, 2, 3]'
    var_15 = 'unique-list'
    var_16 = module_0.assignment(var_13, var_15, var_3)
    var_17 = 'my_set = {3, 1, 2}'
    var_18 = 'my_set = {1, 2, 3}'
    var_19 = 'set'
    var_20 = module_0.assignment(var_17, var_19, var_3)
    var_21 = 'my_tuple = (3, 1, 2)'
    var_22 = 'my_tuple = (1, 2, 3)'
    var_23 = 'tuple'
    var_24 = module_0.assignment(var_21, var_23, var_3)
    var_25 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_26 = 'my_tuple = (1, 2, 3)'
    var_27 = 'unique-tuple'
    var_28 = module_0.assignment(var_25, var_27, var_3)
    var_29 = 'my_list = [3, 1, 2]'
    var_30 = 'invalid'
    var_31 = 'py'
    var_32 = module_0.assignment(var_29, var_30, var_31)
    var_33 = 'my_list = [3, 1, invalid]'
    var_34 = 'list'
    var_35 = 'py'
    var_36 = module_0.assignment(var_33, var_34, var_35)
    var_37 = 'my_dict = [3, 1, 2]'
    var_38 = 'dict'
    var_39 = 'py'
    var_40 = module_0.assignment(var_37, var_38, var_39)
    var_41 = ' '
    var_42 = ''
    var_43 = lambda code, ext, cfg: code.replace(var_41, var_42)
    var_44 = module_1.Config()
    var_45 = 'my_list = [3, 1, 2]'
    var_46 = 'my_list=[1,2,3]'
    var_47 = module_0.assignment(var_45, var_11, var_38, var_44)



# Parsed testcases at query #59
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'my_list = [1, 2, 3]\n'



# Parsed testcases at query #60
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #61
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]    '
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]    '



# Parsed testcases at query #62
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "d = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "d = {'a': 2, 'b': 1}\n"
    var_7 = 'l = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'l = [1, 2, 3]\n'
    var_10 = 'ul = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'ul = [1, 2, 3]\n'
    var_13 = 's = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 's = {1, 2, 3}\n'
    var_16 = 't = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 't = (1, 2, 3)\n'
    var_19 = 'ut = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'ut = (1, 2, 3)\n'
    var_22 = 'x = 1'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'x = [1, 2, 3]'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = 'l = [1, 2, 3, 4, 5]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'l = [1, 2, 3, 4, 5]\n'
    var_38 = 'l = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'l = [1, 2, 3]   \n'



# Parsed testcases at query #63
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2, 3]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #64
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 3, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'a': 3}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 1]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 1)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = invalid\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 3, 'b': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #65
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #66
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #67
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'c': 2, 'a': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #68
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {2: 'b', 1: 'a', 3: 'c'}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {2: 'b', 1: 'a', 3: 'c'}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'my_list = [3, 1, 2]   '
    var_35 = module_0.assignment(var_34, var_8, var_32)
    assert var_35 == 'my_list = [1, 2, 3]   '
    var_36 = 50
    var_37 = module_1.Config()
    var_38 = 'my_list = [3, 1, 2]'
    var_39 = module_0.assignment(var_38, var_8, var_32, var_37)
    assert var_39 == 'my_list = [1, 2, 3]'



# Parsed testcases at query #69
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #70
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #71
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #72
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = invalid_literal\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [1, 2, 3]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #2
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #3
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda x, y, z: x.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #4
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = module_0.assignments(var_0)
    assert var_1 == 'a = 1\nb = 2\nc = 3'
    var_2 = 'b = 2\n\na = 1\n\nc = 3'
    var_3 = module_0.assignments(var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = 'b = 2 \na = 1 \nc = 3 '
    var_5 = module_0.assignments(var_4)
    assert var_5 == 'a = 1 \nb = 2 \nc = 3 '
    var_6 = ''
    var_7 = module_0.assignments(var_6)
    assert var_7 == ''
    var_8 = 'invalid code'
    var_9 = module_0.assignments(var_8)
    var_10 = 'a = 1; b = 2\nc = 3'
    var_11 = module_0.assignments(var_10)
    assert var_11 == 'a = 1; b = 2\nc = 3'
    var_12 = "z = [1, 2, 3]\na = {'a': 1}\nb = (1, 2)"
    var_13 = module_0.assignments(var_12)
    assert var_13 == "a = {'a': 1}\nb = (1, 2)\nz = [1, 2, 3]"



# Parsed testcases at query #5
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = module_0.assignments(var_0)
    assert var_1 == 'a = 1\nb = 2\nc = 3'
    var_2 = 'b = 2\n\na = 1\n\nc = 3'
    var_3 = module_0.assignments(var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = 'b = 2  \na = 1\nc = 3  '
    var_5 = module_0.assignments(var_4)
    assert var_5 == 'a = 1\nb = 2  \nc = 3  '
    var_6 = "z = [1, 2, 3]\na = {'a': 1}\nb = (1, 2)"
    var_7 = module_0.assignments(var_6)
    assert var_7 == "a = {'a': 1}\nb = (1, 2)\nz = [1, 2, 3]"
    var_8 = "print('hello')\nx = 1"
    var_9 = module_0.assignments(var_8)



# Parsed testcases at query #6
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #7
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'a': 2, 'b': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_unique_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_unique_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_unique_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_unique_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #8
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'c': 3, 'a': 1, 'b': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_var = [1, 2, 3]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_var = {'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'invalid_code'
    var_35 = 'assignments'
    var_36 = 'py'
    var_37 = module_0.assignment(var_34, var_35, var_36)



# Parsed testcases at query #9
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'a': 2, 'b': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'x = 1'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid_literal'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "x = {'a': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #10
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #11
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'a': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #12
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "x = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "x = {'a': 1, 'b': 2}\n"
    var_7 = 'x = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'x = [1, 2, 3]\n'
    var_10 = 'x = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'x = [1, 2, 3]\n'
    var_13 = 'x = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'x = {1, 2, 3}\n'
    var_16 = 'x = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'x = (1, 2, 3)\n'
    var_19 = 'x = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'x = (1, 2, 3)\n'
    var_22 = 'x = [1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'x = [1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #13
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]    '
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]    '



# Parsed testcases at query #14
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   '
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   '



# Parsed testcases at query #15
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'c': 2, 'a': 3}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 1, 3]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 1, 3)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #16
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 2, 'a': 1}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #17
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'my_list = [3, 1, 2]   '
    var_35 = module_0.assignment(var_34, var_8, var_32)
    assert var_35 == 'my_list = [1, 2, 3]   '
    var_36 = 40
    var_37 = module_1.Config()
    var_38 = 'my_list = [3, 1, 2]'
    var_39 = module_0.assignment(var_38, var_8, var_32, var_37)
    assert var_39 == 'my_list = [1, 2, 3]'
    var_40 = lambda x, y, z: x.upper()
    var_41 = module_1.Config()
    var_42 = 'my_list = [3, 1, 2]'
    var_43 = module_0.assignment(var_42, var_8, var_32, var_41)
    assert var_43 == 'MY_LIST = [1, 2, 3]'



# Parsed testcases at query #18
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #19
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda x, y, z: x.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   '
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   '



# Parsed testcases at query #20
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'



# Parsed testcases at query #21
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'x = 1'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid_literal'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "x = {'a': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda x, y, z: x.upper()
    var_35 = module_1.Config()
    var_36 = 'x = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'X = [1, 2, 3]'
    var_38 = 'x = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #22
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_var = 1'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_var = {'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #23
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #24
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [2, 1, 3]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [2, 1, 3, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {2, 1, 3}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (2, 1, 3)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (2, 1, 3, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [2, 1, 3]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [2, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [2, 1, 3]\n'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [2, 1, 3]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #25
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'c': 2, 'a': 3}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2, 3]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = invalid_literal\n'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #26
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "x = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "x = {'a': 3, 'b': 1, 'c': 2}"
    var_7 = 'y = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'y = [1, 2, 3]'
    var_10 = 'z = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'z = [1, 2, 3]'
    var_13 = 's = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 's = {1, 2, 3}'
    var_16 = 't = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 't = (1, 2, 3)'
    var_19 = 'u = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'u = (1, 2, 3)'
    var_22 = 'x = 1'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'x = [1, 2, 3]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'x = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'X = [1, 2, 3]'
    var_38 = 'x = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'x = [1, 2, 3]  \n'



# Parsed testcases at query #27
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'a': 2, 'b': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    var_40 = '    \n'



# Parsed testcases at query #28
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'a = 1\nb = 2\nc = 3'
    var_2 = 'assignments'
    var_3 = 'py'
    var_4 = module_0.assignment(var_0, var_2, var_3)
    var_5 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_6 = "my_dict = {'b': 1, 'c': 2, 'a': 3}"
    var_7 = 'dict'
    var_8 = module_0.assignment(var_5, var_7, var_3)
    var_9 = 'my_list = [3, 1, 2]'
    var_10 = 'my_list = [1, 2, 3]'
    var_11 = 'list'
    var_12 = module_0.assignment(var_9, var_11, var_3)
    var_13 = 'my_list = [3, 1, 2, 2, 3]'
    var_14 = 'my_list = [1, 2, 3]'
    var_15 = 'unique-list'
    var_16 = module_0.assignment(var_13, var_15, var_3)
    var_17 = 'my_set = {3, 1, 2}'
    var_18 = 'my_set = {1, 2, 3}'
    var_19 = 'set'
    var_20 = module_0.assignment(var_17, var_19, var_3)
    var_21 = 'my_tuple = (3, 1, 2)'
    var_22 = 'my_tuple = (1, 2, 3)'
    var_23 = 'tuple'
    var_24 = module_0.assignment(var_21, var_23, var_3)
    var_25 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_26 = 'my_tuple = (1, 2, 3)'
    var_27 = 'unique-tuple'
    var_28 = module_0.assignment(var_25, var_27, var_3)
    var_29 = 'my_list = [3, 1, 2]'
    var_30 = 'invalid'
    var_31 = 'py'
    var_32 = module_0.assignment(var_29, var_30, var_31)
    var_33 = 'my_list = [3, 1, invalid]'
    var_34 = 'list'
    var_35 = 'py'
    var_36 = module_0.assignment(var_33, var_34, var_35)
    var_37 = 'my_list = [3, 1, 2]'
    var_38 = 'dict'
    var_39 = 'py'
    var_40 = module_0.assignment(var_37, var_38, var_39)
    var_41 = lambda code, ext, cfg: code.upper()
    var_42 = module_1.Config()
    var_43 = 'my_list = [3, 1, 2]'
    var_44 = 'MY_LIST = [1, 2, 3]'
    var_45 = module_0.assignment(var_43, var_11, var_39, var_42)
    var_46 = 'my_list = [3, 1, 2]   '
    var_47 = 'my_list = [1, 2, 3]   '
    var_48 = module_0.assignment(var_46, var_11, var_39)



# Parsed testcases at query #29
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 2, 'a': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'my_list = [1, 2, 3]\n'



# Parsed testcases at query #30
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #31
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'x = 1'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid_literal'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "x = {'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda x, y, z: x.upper()
    var_35 = module_1.Config()
    var_36 = 'x = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'X = [1, 2, 3]\n'
    var_38 = 'x = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'x = [1, 2, 3]  \n'



# Parsed testcases at query #32
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #33
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'c': 2, 'a': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #34
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #35
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 2, 'a': 1}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'x = 1'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid_literal'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "x = {'a': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #36
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'



# Parsed testcases at query #37
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #38
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'invalid_code'
    var_35 = 'assignments'
    var_36 = 'py'
    var_37 = module_0.assignment(var_34, var_35, var_36)



# Parsed testcases at query #39
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_var = [1, 2, 3]\n'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal\n'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_var = [1, 2, 3]\n'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #40
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2, 3]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_var = [1, 2, 3]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_var = [1, 2, 3]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'



# Parsed testcases at query #41
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'my_list = [3, 1, 2]   \n'
    var_35 = module_0.assignment(var_34, var_8, var_31)
    assert var_35 == 'my_list = [1, 2, 3]   \n'
    var_36 = 40
    var_37 = module_1.Config()
    var_38 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_39 = module_0.assignment(var_38, var_33, var_31, var_37)
    assert var_39 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"



# Parsed testcases at query #42
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #43
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 2, 'a': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'x = 1'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'x = 1'
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda x, y, z: x.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #44
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = {3, 1, 2}'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #45
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #46
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'my_list = [3, 1, 2]   \n'
    var_35 = module_0.assignment(var_34, var_8, var_31)
    assert var_35 == 'my_list = [1, 2, 3]   \n'
    var_36 = 40
    var_37 = module_1.Config()
    var_38 = 'my_list = [3, 1, 2]'
    var_39 = module_0.assignment(var_38, var_8, var_31, var_37)
    assert var_39 == 'my_list = [1, 2, 3]'



# Parsed testcases at query #47
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 50
    var_35 = module_1.Config()
    var_36 = "my_dict = {'b': 2, 'a': 1}\n"
    var_37 = module_0.assignment(var_36, var_33, var_31, var_35)
    assert var_37 == "my_dict = {'a': 1, 'b': 2}\n"



# Parsed testcases at query #48
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "x = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "x = {'a': 3, 'b': 1, 'c': 2}"
    var_7 = 'x = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'x = [1, 2, 3]'
    var_10 = 'x = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'x = [1, 2, 3]'
    var_13 = 'x = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'x = {1, 2, 3}'
    var_16 = 'x = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'x = (1, 2, 3)'
    var_19 = 'x = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'x = (1, 2, 3)'
    var_22 = 'x = [1, 2, 3]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'x = [1, 2, 3]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'x = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'X = [1, 2, 3]'
    var_38 = 'x = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #49
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #50
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'x = 1'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid_literal'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "x = {'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda x, y, z: x.upper()
    var_35 = module_1.Config()
    var_36 = 'x = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'X = [1, 2, 3]'
    var_38 = 'x = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'x = [1, 2, 3]  \n'



# Parsed testcases at query #51
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #52
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2'
    var_4 = "my_dict = {'a': 2, 'b': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   '
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   '



# Parsed testcases at query #53
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #54
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #55
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 1]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 1)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #56
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 1]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 1)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'x = 1'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "x = {'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda x, _, __: x.upper()
    var_35 = module_1.Config()
    var_36 = 'x = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'X = [1, 2, 3]'
    var_38 = 'x = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'x = [1, 2, 3]  \n'



# Parsed testcases at query #57
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 50
    var_35 = module_1.Config()
    var_36 = "my_dict = {'b': 2, 'a': 1}"
    var_37 = module_0.assignment(var_36, var_33, var_31, var_35)
    assert var_37 == "my_dict = {'a': 1, 'b': 2}"
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #58
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_dict = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #59
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'my_list = [3, 1, 2]   \n'
    var_35 = module_0.assignment(var_34, var_8, var_32)
    assert var_35 == 'my_list = [1, 2, 3]   \n'
    var_36 = 40
    var_37 = module_1.Config()
    var_38 = "my_dict = {'b': 2, 'a': 1}\n"
    var_39 = module_0.assignment(var_38, var_5, var_32, var_37)
    assert var_39 == "my_dict = {'a': 1, 'b': 2}\n"



# Parsed testcases at query #60
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #61
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 2, 'b': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #62
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'a = 1\nb = 2\nc = 3'
    var_2 = 'assignments'
    var_3 = 'py'
    var_4 = module_0.assignment(var_0, var_2, var_3)
    var_5 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_6 = "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'dict'
    var_8 = module_0.assignment(var_5, var_7, var_3)
    var_9 = 'my_list = [3, 1, 2]'
    var_10 = 'my_list = [1, 2, 3]'
    var_11 = 'list'
    var_12 = module_0.assignment(var_9, var_11, var_3)
    var_13 = 'my_list = [3, 1, 2, 2, 3]'
    var_14 = 'my_list = [1, 2, 3]'
    var_15 = 'unique-list'
    var_16 = module_0.assignment(var_13, var_15, var_3)
    var_17 = 'my_set = {3, 1, 2}'
    var_18 = 'my_set = {1, 2, 3}'
    var_19 = 'set'
    var_20 = module_0.assignment(var_17, var_19, var_3)
    var_21 = 'my_tuple = (3, 1, 2)'
    var_22 = 'my_tuple = (1, 2, 3)'
    var_23 = 'tuple'
    var_24 = module_0.assignment(var_21, var_23, var_3)
    var_25 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_26 = 'my_tuple = (1, 2, 3)'
    var_27 = 'unique-tuple'
    var_28 = module_0.assignment(var_25, var_27, var_3)
    var_29 = 'my_list = [3, 1, 2]'
    var_30 = 'invalid_type'
    var_31 = 'py'
    var_32 = module_0.assignment(var_29, var_30, var_31)
    var_33 = 'my_list = [3, 1, invalid]'
    var_34 = 'list'
    var_35 = 'py'
    var_36 = module_0.assignment(var_33, var_34, var_35)
    var_37 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_38 = 'list'
    var_39 = 'py'
    var_40 = module_0.assignment(var_37, var_38, var_39)
    var_41 = ' '
    var_42 = ''
    var_43 = lambda code, ext, cfg: code.replace(var_41, var_42)
    var_44 = module_1.Config()
    var_45 = 'my_list = [3, 1, 2]  '
    var_46 = 'my_list=[1,2,3]'
    var_47 = module_0.assignment(var_45, var_11, var_38, var_44)



# Parsed testcases at query #63
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #64
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'my_list = [1, 2, 3]'



# Parsed testcases at query #65
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "x = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "x = {'a': 3, 'b': 1, 'c': 2}"
    var_7 = 'x = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'x = [1, 2, 3]'
    var_10 = 'x = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'x = [1, 2, 3]'
    var_13 = 'x = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'x = {1, 2, 3}'
    var_16 = 'x = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'x = (1, 2, 3)'
    var_19 = 'x = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'x = (1, 2, 3)'
    var_22 = 'x = [1, 2, 3]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'x = [1, 2, 3]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda x, y, z: x.upper()
    var_35 = module_1.Config()
    var_36 = 'x = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'X = [1, 2, 3]'
    var_38 = 'x = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #66
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 2, 'b': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 1, 'a': 2}\n"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]   '
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   '



# Parsed testcases at query #67
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'b': 2, 'a': 1}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #68
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {2: 'b', 1: 'a'}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {1: 'a', 2: 'b'}\n"
    var_7 = 'my_list = [2, 1, 3]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [2, 1, 3, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {2, 1, 3}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (2, 1, 3)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (2, 1, 3, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_list = [2, 1, 3]\n'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [2, 1, 3\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [2, 1, 3]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [2, 1, 3]\n'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [2, 1, 3]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #69
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]    '
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]    '
    var_40 = 10
    var_41 = module_1.Config()
    var_42 = "my_dict = {'bbbb': 2, 'aaaa': 1}"
    var_43 = module_0.assignment(var_42, var_5, var_32, var_41)
    assert var_43 == "my_dict = {\n    'aaaa': 1,\n    'bbbb': 2\n}"



# Parsed testcases at query #70
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   '
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   '



# Parsed testcases at query #71
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}"
    var_7 = 'my_list = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]'
    var_10 = 'my_list = [3, 1, 2, 2]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]'
    var_13 = 'my_set = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}'
    var_16 = 'my_tuple = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)'
    var_19 = 'my_tuple = (3, 1, 2, 2)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)'
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #72
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'b': 2, 'a': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 1, 'b': 2}\n"
    var_7 = 'my_list = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'my_list = [1, 2, 3]\n'
    var_10 = 'my_list = [3, 1, 2, 2]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'my_list = [1, 2, 3]\n'
    var_13 = 'my_set = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 'my_set = {1, 2, 3}\n'
    var_16 = 'my_tuple = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 'my_tuple = (1, 2, 3)\n'
    var_19 = 'my_tuple = (3, 1, 2, 2)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'my_tuple = (1, 2, 3)\n'
    var_22 = 'my_var = [1, 2, 3]\n'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_var = [1, 2, 3]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 50
    var_35 = module_1.Config()
    var_36 = "my_dict = {'b': 2, 'a': 1}\n"
    var_37 = module_0.assignment(var_36, var_33, var_31, var_35)
    assert var_37 == "my_dict = {'a': 1, 'b': 2}\n"



