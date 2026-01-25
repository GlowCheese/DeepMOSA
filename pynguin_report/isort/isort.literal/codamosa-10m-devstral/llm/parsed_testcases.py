####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = "my_dict = {'b': 2, 'a': 1}\n"
    var_37 = module_0.assignment(var_36, var_33, var_31, var_35)
    assert var_37 == "my_dict = {'a': 1, 'b': 2}\n"



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
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = {3, 1, 2}'
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



# Parsed testcases at query #3
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
    var_30 = 'my_dict = [3, 1, 2]'
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



# Parsed testcases at query #4
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
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #5
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = module_0.assignments(var_0)
    assert var_1 == 'a = 1\nb = 2'
    var_2 = 'b = 2\n\na = 1'
    var_3 = module_0.assignments(var_2)
    assert var_3 == 'a = 1\n\nb = 2'
    var_4 = 'z = 26\na = 1\nm = 13'
    var_5 = module_0.assignments(var_4)
    assert var_5 == 'a = 1\nm = 13\nz = 26'
    var_6 = "b = 'a = 1'\na = 2"
    var_7 = module_0.assignments(var_6)
    assert var_7 == "a = 2\nb = 'a = 1'"
    var_8 = 'invalid code'
    var_9 = module_0.assignments(var_8)
    var_10 = ''
    var_11 = module_0.assignments(var_10)
    assert var_11 == ''
    var_12 = '\n'
    var_13 = module_0.assignments(var_12)
    assert var_13 == '\n'



# Parsed testcases at query #6
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
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #7
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = '.py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = "d = {'a': 3, 'b': 1, 'c': 2}"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "d = {'b': 1, 'c': 2, 'a': 3}"
    var_7 = 'l = [3, 1, 2]'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'l = [1, 2, 3]'
    var_10 = 'ul = [3, 1, 2, 2, 3]'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'ul = [1, 2, 3]'
    var_13 = 's = {3, 1, 2}'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 's = {1, 2, 3}'
    var_16 = 't = (3, 1, 2)'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 't = (1, 2, 3)'
    var_19 = 'ut = (3, 1, 2, 2, 3)'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'ut = (1, 2, 3)'
    var_22 = 'x = 1'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "x = {'a': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'l = [3, 1, 2]  \n'
    var_35 = module_0.assignment(var_34, var_8, var_31)
    assert var_35 == 'l = [1, 2, 3]  \n'
    var_36 = 50
    var_37 = module_1.Config()
    var_38 = "d = {'a': 3, 'b': 1, 'c': 2}"
    var_39 = module_0.assignment(var_38, var_33, var_31, var_37)
    assert var_39 == "d = {'b': 1, 'c': 2, 'a': 3}"



# Parsed testcases at query #8
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
    var_10 = 'my_list = [3, 1, 2, 1]'
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
    var_19 = 'my_tuple = (3, 1, 2, 1)'
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
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #9
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
    var_34 = lambda x, y, z: x.upper()
    var_35 = module_1.Config()
    var_36 = 'x = [2, 1]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'X = [1, 2]'
    var_38 = 'x = [2, 1]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'x = [1, 2]   \n'



# Parsed testcases at query #10
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'z = 1\na = 2\nb = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 2\nb = 3\nz = 1'
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
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'a = 1\nb = 2\nc = 3'
    var_2 = 'assignments'
    var_3 = 'py'
    var_4 = module_0.assignment(var_0, var_2, var_3)
    var_5 = "my_dict = {1: 'a', 2: 'b'}"
    var_6 = "my_dict = {1: 'a', 2: 'b'}"
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
    var_37 = 'my_dict = [3, 1, 2]'
    var_38 = 'dict'
    var_39 = 'py'
    var_40 = module_0.assignment(var_37, var_38, var_39)



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



# Parsed testcases at query #14
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
    assert var_6 == "my_dict = {'b': 2, 'a': 1, 'c': 3}"
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
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   '
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   '



# Parsed testcases at query #15
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



# Parsed testcases at query #16
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
    var_34 = 50
    var_35 = module_1.Config()
    var_36 = "my_dict = {'b': 2, 'a': 1}\n"
    var_37 = module_0.assignment(var_36, var_33, var_31, var_35)
    assert var_37 == "my_dict = {'a': 1, 'b': 2}\n"



# Parsed testcases at query #17
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
    var_23 = 'invalid'
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
    var_34 = 50
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'my_list = [1, 2, 3]'



# Parsed testcases at query #18
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = module_0.assignment(var_0, var_1)
    assert var_2 == 'a = 1\nb = 2'
    var_3 = "my_dict = {'b': 2, 'a': 1}"
    var_4 = 'dict'
    var_5 = module_0.assignment(var_3, var_4)
    assert var_5 == "my_dict = {'a': 1, 'b': 2}"
    var_6 = 'my_list = [3, 1, 2]'
    var_7 = 'list'
    var_8 = module_0.assignment(var_6, var_7)
    assert var_8 == 'my_list = [1, 2, 3]'
    var_9 = 'my_list = [3, 1, 2, 2]'
    var_10 = 'unique-list'
    var_11 = module_0.assignment(var_9, var_10)
    assert var_11 == 'my_list = [1, 2, 3]'
    var_12 = 'my_set = {3, 1, 2}'
    var_13 = 'set'
    var_14 = module_0.assignment(var_12, var_13)
    assert var_14 == 'my_set = {1, 2, 3}'
    var_15 = 'my_tuple = (3, 1, 2)'
    var_16 = 'tuple'
    var_17 = module_0.assignment(var_15, var_16)
    assert var_17 == 'my_tuple = (1, 2, 3)'
    var_18 = 'my_tuple = (3, 1, 2, 2)'
    var_19 = 'unique-tuple'
    var_20 = module_0.assignment(var_18, var_19)
    assert var_20 == 'my_tuple = (1, 2, 3)'
    var_21 = 'my_list = [3, 1, 2]'
    var_22 = 'invalid_type'
    var_23 = module_0.assignment(var_21, var_22)
    var_24 = 'my_list = [3, 1, invalid]'
    var_25 = 'list'
    var_26 = module_0.assignment(var_24, var_25)
    var_27 = "my_dict = {'b': 2, 'a': 1}"
    var_28 = 'list'
    var_29 = module_0.assignment(var_27, var_28)



# Parsed testcases at query #19
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
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [2, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = 'my_list = [2, 1, 3, 4, 5]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'my_list = [1, 2, 3, 4, 5]'
    var_38 = 'my_list = [2, 1, 3]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



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
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #21
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
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #22
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
    assert var_6 == "my_dict = {'a': 3, 'b': 1, 'c': 2}"
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



# Parsed testcases at query #23
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
    var_23 = 'invalid_type'
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
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #24
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
    var_30 = 'my_list = [3, 1, 2]'
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



# Parsed testcases at query #25
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
    var_22 = 'x = 1'
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'x = invalid_literal'
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



# Parsed testcases at query #26
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
    var_30 = "my_dict = {'b': 2, 'a': 1}"
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
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



# Parsed testcases at query #28
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
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



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
    var_38 = 'my_list = [3, 1, 2]   '
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   '



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
    assert var_6 == "my_dict = {'b': 2, 'a': 1, 'c': 3}"
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



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



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



# Parsed testcases at query #3
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
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
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



# Parsed testcases at query #4
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
    var_22 = 'my_list = [3, 1, 2]'
    var_23 = 'invalid'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #5
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
    var_22 = 'x = 1'
    var_23 = 'invalid'
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
    var_38 = 'x = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'x = [1, 2, 3]   \n'



# Parsed testcases at query #6
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
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #7
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
    var_34 = 50
    var_35 = module_1.Config()
    var_36 = "my_dict = {'b': 2, 'a': 1}"
    var_37 = module_0.assignment(var_36, var_33, var_31, var_35)
    assert var_37 == "my_dict = {'a': 1, 'b': 2}"



# Parsed testcases at query #8
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
    var_4 = "my_dict = {'a': 3, 'b': 1, 'c': 2}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "my_dict = {'a': 3, 'b': 1, 'c': 2}\n"
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
    var_30 = 'my_list = [3, 1, 2]\n'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #9
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
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #10
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1'
    var_1 = 'assignments'
    var_2 = module_0.assignment(var_0, var_1)
    assert var_2 == 'a = 1\nb = 2'
    var_3 = 'my_dict = {1: 3, 2: 1}'
    var_4 = 'dict'
    var_5 = module_0.assignment(var_3, var_4)
    assert var_5 == 'my_dict = {1: 3, 2: 1}'
    var_6 = 'my_list = [3, 1, 2]'
    var_7 = 'list'
    var_8 = module_0.assignment(var_6, var_7)
    assert var_8 == 'my_list = [1, 2, 3]'
    var_9 = 'my_list = [3, 1, 2, 2]'
    var_10 = 'unique-list'
    var_11 = module_0.assignment(var_9, var_10)
    assert var_11 == 'my_list = [1, 2, 3]'
    var_12 = 'my_set = {3, 1, 2}'
    var_13 = 'set'
    var_14 = module_0.assignment(var_12, var_13)
    assert var_14 == 'my_set = {1, 2, 3}'
    var_15 = 'my_tuple = (3, 1, 2)'
    var_16 = 'tuple'
    var_17 = module_0.assignment(var_15, var_16)
    assert var_17 == 'my_tuple = (1, 2, 3)'
    var_18 = 'my_tuple = (3, 1, 2, 2)'
    var_19 = 'unique-tuple'
    var_20 = module_0.assignment(var_18, var_19)
    assert var_20 == 'my_tuple = (1, 2, 3)'
    var_21 = 'my_list = [3, 1, 2]'
    var_22 = 'invalid_type'
    var_23 = module_0.assignment(var_21, var_22)
    var_24 = 'my_list = [3, 1, 2'
    var_25 = 'list'
    var_26 = module_0.assignment(var_24, var_25)
    var_27 = 'my_dict = {1: 3, 2: 1}'
    var_28 = 'list'
    var_29 = module_0.assignment(var_27, var_28)



# Parsed testcases at query #11
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\n'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\n'
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
    var_30 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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
    var_22 = 'my_list = [3, 1, 2]\n'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = [3, 1, invalid]\n'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}\n"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]\n'
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]\n'
    var_38 = 'my_list = [3, 1, 2]    \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]    \n'



# Parsed testcases at query #14
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
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #15
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
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #16
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
    var_30 = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
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



# Parsed testcases at query #17
#--------------------------


import isort.literal as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'z = 1\na = 2\nb = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 2\nb = 3\nz = 1'
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
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   '
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   '



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
    var_26 = 'my_list = [3, 1, 2'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'b': 2, 'a': 1}"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #20
#--------------------------


import isort.literal as module_0

def test_case_0():
    var_0 = 'b = 2\na = 1\nc = 3'
    var_1 = 'assignments'
    var_2 = 'py'
    var_3 = module_0.assignment(var_0, var_1, var_2)
    assert var_3 == 'a = 1\nb = 2\nc = 3'
    var_4 = 'my_dict = {1: 3, 2: 2, 3: 1}'
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == 'my_dict = {1: 3, 2: 2, 3: 1}'
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



# Parsed testcases at query #21
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
    var_26 = 'my_list = [3, 1, invalid]'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = 'my_list = [3, 1, 2]'
    var_31 = 'dict'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #22
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
    var_23 = 'invalid_type'
    var_24 = 'py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_list = invalid_literal\n'
    var_27 = 'list'
    var_28 = 'py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_dict = {'a': 2, 'b': 1}\n"
    var_31 = 'list'
    var_32 = 'py'
    var_33 = module_0.assignment(var_30, var_31, var_32)
    var_34 = 'my_list = [3, 1, 2]   \n'
    var_35 = module_0.assignment(var_34, var_8, var_32)
    assert var_35 == 'my_list = [1, 2, 3]   \n'
    var_36 = 40
    var_37 = module_1.Config()
    var_38 = 'my_list = [3, 1, 2]\n'
    var_39 = module_0.assignment(var_38, var_8, var_32, var_37)
    assert var_39 == 'my_list = [1, 2, 3]\n'



# Parsed testcases at query #23
#--------------------------


import isort.literal as module_0

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



# Parsed testcases at query #24
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
    assert var_6 == "my_dict = {'a': 3, 'b': 1, 'c': 2}"
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



# Parsed testcases at query #25
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
    assert var_6 == "my_dict = {1: 'a', 2: 'b', 3: 'c'}"
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
    var_37 = module_0.assignment(var_36, var_8, var_32, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]   \n'
    var_39 = module_0.assignment(var_38, var_8, var_32)
    assert var_39 == 'my_list = [1, 2, 3]   \n'



# Parsed testcases at query #26
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
    var_34 = 40
    var_35 = module_1.Config()
    var_36 = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    var_37 = module_0.assignment(var_36, var_33, var_31, var_35)
    assert var_37 == "my_dict = {'a': 1, 'b': 2, 'c': 3}"



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
    var_34 = lambda code, extension, config: code.upper()
    var_35 = module_1.Config()
    var_36 = 'my_list = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'MY_LIST = [1, 2, 3]'
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



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
    var_4 = "x = {'a': 2, 'b': 1}\n"
    var_5 = 'dict'
    var_6 = module_0.assignment(var_4, var_5, var_2)
    assert var_6 == "x = {'a': 2, 'b': 1}\n"
    var_7 = 'y = [3, 1, 2]\n'
    var_8 = 'list'
    var_9 = module_0.assignment(var_7, var_8, var_2)
    assert var_9 == 'y = [1, 2, 3]\n'
    var_10 = 'z = [1, 2, 2, 3]\n'
    var_11 = 'unique-list'
    var_12 = module_0.assignment(var_10, var_11, var_2)
    assert var_12 == 'z = [1, 2, 3]\n'
    var_13 = 's = {3, 1, 2}\n'
    var_14 = 'set'
    var_15 = module_0.assignment(var_13, var_14, var_2)
    assert var_15 == 's = {1, 2, 3}\n'
    var_16 = 't = (3, 1, 2)\n'
    var_17 = 'tuple'
    var_18 = module_0.assignment(var_16, var_17, var_2)
    assert var_18 == 't = (1, 2, 3)\n'
    var_19 = 'u = (1, 2, 2, 3)\n'
    var_20 = 'unique-tuple'
    var_21 = module_0.assignment(var_19, var_20, var_2)
    assert var_21 == 'u = (1, 2, 3)\n'
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
    var_34 = lambda code, ext, cfg: code.upper()
    var_35 = module_1.Config()
    var_36 = 'x = [3, 1, 2]'
    var_37 = module_0.assignment(var_36, var_8, var_31, var_35)
    assert var_37 == 'X = [1, 2, 3]'



# Parsed testcases at query #29
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
    var_22 = 'my_var = 1'
    var_23 = 'invalid_type'
    var_24 = '.py'
    var_25 = module_0.assignment(var_22, var_23, var_24)
    var_26 = 'my_var = invalid_literal'
    var_27 = 'list'
    var_28 = '.py'
    var_29 = module_0.assignment(var_26, var_27, var_28)
    var_30 = "my_var = {'a': 1}"
    var_31 = 'list'
    var_32 = '.py'
    var_33 = module_0.assignment(var_30, var_31, var_32)



# Parsed testcases at query #30
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
    var_38 = 'my_list = [3, 1, 2]  \n'
    var_39 = module_0.assignment(var_38, var_8, var_31)
    assert var_39 == 'my_list = [1, 2, 3]  \n'



