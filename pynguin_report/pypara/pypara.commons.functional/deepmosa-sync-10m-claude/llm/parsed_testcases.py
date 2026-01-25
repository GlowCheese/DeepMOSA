####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
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
    var_1 = [var_0]
    var_2 = module_0.chunk(var_1, var_0)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [[1]])
    assert var_4 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 5
    var_3 = module_0.chunk(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[1]])
    assert var_5 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = module_0.chunk(var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [['a', 'b'], ['c', 'd']])
    assert var_8 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = module_0.chunk(var_3, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2, 3]])
    assert var_7 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.chunk(var_4, var_3)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2, 3, 4]])
    assert var_7 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
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
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = module_0.chunk(var_3, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2, 3]])
    assert var_7 is True

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
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.chunk(var_3, var_2)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [[1, 2, 3]])
    assert var_6 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 2
    var_6 = module_0.chunk(var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [['a', 'b'], ['c', 'd']])
    assert var_8 is True

import pypara.commons.functional as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 'b'
    var_4 = 3
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.chunk(var_5, var_2)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [[1, 'a'], [2, 'b'], [3]])
    assert var_8 is True



