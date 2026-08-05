####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '# noqa:? (?P<code>.*)'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = []
    var_4 = module_1.parse_noqa(var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True

import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '# noqa:? (?P<code>.*)'
    var_1 = module_0.compile(var_0)
    var_2 = 'E402'
    var_3 = 'Import error'
    var_4 = {var_2: var_3}
    var_5 = 'import os'
    var_6 = 'import sys  # noqa: E402, F401'
    var_7 = "print('hello')  # noqa"
    var_8 = 'x = 1  # noqa: E701'
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = module_1.parse_noqa(var_9)
    var_11 = var_10['Import error']
    var_12 = bool(var_10['Import error'] == {2})
    assert var_12 is True
    var_13 = var_10['F401']
    var_14 = bool(var_10['F401'] == {2})
    assert var_14 is True
    var_15 = var_10['all']
    var_16 = bool(var_10['all'] == {3})
    assert var_16 is True
    var_17 = var_10['E701']
    var_18 = bool(var_10['E701'] == {4})
    assert var_18 is True



# Parsed testcases at query #2
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 10
    var_3 = 20
    var_4 = [var_2, var_3]
    var_5 = 5
    var_6 = [var_5]
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 10
    var_3 = [var_2]
    var_4 = 5
    var_5 = 15
    var_6 = [var_4, var_5]
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_4, var_0)
    assert var_8 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 10
    var_3 = [var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 20
    var_8 = module_0.ignore_line(var_6, var_7, var_0)
    assert var_8 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 10
    var_3 = [var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 99
    var_8 = module_0.ignore_line(var_6, var_7, var_0)
    assert var_8 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'all'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 'E501'
    var_5 = module_0.ignore_line(var_2, var_3, var_4)
    assert var_5 is False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 10
    var_3 = 20
    var_4 = [var_2, var_3]
    var_5 = 5
    var_6 = [var_5]
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 10
    var_3 = [var_2]
    var_4 = 5
    var_5 = 15
    var_6 = [var_4, var_5]
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_4, var_0)
    assert var_8 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 10
    var_3 = [var_2]
    var_4 = 5
    var_5 = [var_4]
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 20
    var_8 = module_0.ignore_line(var_6, var_7, var_0)
    assert var_8 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'all'
    var_1 = 5
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'E501'
    var_5 = module_0.ignore_line(var_3, var_1, var_4)
    assert var_5 is True

def test_case_0():
    var_0 = {}

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'F401'
    var_1 = 'all'
    var_2 = 1
    var_3 = [var_2]
    var_4 = []
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_0.ignore_line(var_5, var_2, var_0)
    assert var_6 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E225'
    var_1 = 'all'
    var_2 = 100
    var_3 = [var_2]
    var_4 = 200
    var_5 = [var_4]
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 300
    var_8 = module_0.ignore_line(var_6, var_7, var_0)
    assert var_8 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_parse_noqa_with_no_matches. Retrieved 5/7 statements.


import vulture.noqa as module_0

def test_case_0():
    var_0 = 'import os  # noqa: E401, F401'
    var_1 = 'import sys  # noqa: E401'
    var_2 = 'import math'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_noqa(var_3)
    var_5 = var_4['Import error']
    var_6 = bool(var_4['Import error'] == {1, 2})
    assert var_6 is True
    var_7 = var_4['Unused import']
    var_8 = bool(var_4['Unused import'] == {1})
    assert var_8 is True
    var_9 = len(var_4)
    assert var_9 == 2

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'import os  # noqa'
    var_1 = 'import sys  # noqa: all'
    var_2 = 'import math'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_noqa(var_3)
    var_5 = var_4['all']
    var_6 = bool(var_4['all'] == {1, 2})
    assert var_6 is True
    var_7 = len(var_4)
    assert var_7 == 1

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E701'
    var_1 = 'y = 2  # noqa: E702'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = var_3['E701']
    var_5 = bool(var_3['E701'] == {1})
    assert var_5 is True
    var_6 = var_3['E702']
    var_7 = bool(var_3['E702'] == {2})
    assert var_7 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import sys'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = len(var_3)
    assert var_4 == 0

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'import os  # noqa: E401 ,  F401 '
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    var_3 = var_2['Import error']
    var_4 = bool(var_2['Import error'] == {1})
    assert var_4 is True
    var_5 = var_2['Unused import']
    var_6 = bool(var_2['Unused import'] == {1})
    assert var_6 is True



