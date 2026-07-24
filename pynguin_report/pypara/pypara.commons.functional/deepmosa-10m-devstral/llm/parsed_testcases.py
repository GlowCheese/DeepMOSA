####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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



