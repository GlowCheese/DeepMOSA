####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pypara.commons.functional as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True
    var_5 = []
    var_6 = 2
    var_7 = module_0.chunk(var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.chunk(var_4, var_0)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1], [2], [3], [4]])
    assert var_7 is True
    var_8 = [var_0, var_1, var_2, var_3]
    var_9 = module_0.chunk(var_8, var_1)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [[1, 2], [3, 4]])
    assert var_11 is True
    var_12 = [var_0, var_1, var_2, var_3]
    var_13 = module_0.chunk(var_12, var_3)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [[1, 2, 3, 4]])
    assert var_15 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.chunk(var_5, var_1)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [[1, 2], [3, 4], [5]])
    assert var_8 is True
    var_9 = 6
    var_10 = 7
    var_11 = [var_0, var_1, var_2, var_3, var_4, var_9, var_10]
    var_12 = module_0.chunk(var_11, var_2)
    var_13 = list(var_12)
    var_14 = bool(var_13 == [[1, 2, 3], [4, 5, 6], [7]])
    assert var_14 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = module_0.chunk(var_3, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2, 3]])
    assert var_7 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pypara.commons.functional as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = []
    var_1 = 2
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.chunk(var_4, var_0)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1], [2], [3], [4]])
    assert var_7 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.chunk(var_4, var_1)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2], [3, 4]])
    assert var_7 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.chunk(var_5, var_1)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [[1, 2], [3, 4], [5]])
    assert var_8 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = module_0.chunk(var_2, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [[1, 2]])
    assert var_6 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = module_0.chunk(var_1, var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [[1]])
    assert var_4 is True

import pypara.commons.functional as module_0

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
    var_11 = module_0.chunk(var_10, var_3)
    var_12 = list(var_11)
    var_13 = bool(var_12 == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]])
    assert var_13 is True



