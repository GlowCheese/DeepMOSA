####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E123'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    var_3 = bool(var_2 == {'E123': {1}})
    assert var_3 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E123, F456'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    var_3 = bool(var_2 == {'E123': {1}, 'F456': {1}})
    assert var_3 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    var_3 = bool(var_2 == {'all': {1}})
    assert var_3 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E123'
    var_1 = 'y = 2  # noqa: F456, G789'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = bool(var_3 == {'E123': {1}, 'F456': {2}, 'G789': {2}})
    assert var_4 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = 'y = 2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'x = 1  # noqa: E123'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = bool(var_3 == {'E123': {2}})
    assert var_4 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: W123'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)



# Parsed testcases at query #2
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 1
    var_6 = 2
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.ignore_line(var_8, var_2, var_0)
    assert var_9 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 1
    var_6 = 2
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.ignore_line(var_8, var_5, var_0)
    assert var_9 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 1
    var_6 = 2
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 3
    var_10 = module_0.ignore_line(var_8, var_9, var_0)
    assert var_10 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = set()
    var_3 = set()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5
    var_6 = module_0.ignore_line(var_4, var_5, var_0)
    assert var_6 is False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '(?P<codes>)'
    var_1 = ''
    var_2 = module_0.match(var_0, var_1)
    var_3 = module_1._parse_error_codes(var_2)
    var_4 = bool(var_3 == ['all'])
    assert var_4 is True

import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '(?P<codes>E001)'
    var_1 = 'E001'
    var_2 = module_0.match(var_0, var_1)
    var_3 = module_1._parse_error_codes(var_2)
    var_4 = bool(var_3 == ['E001'])
    assert var_4 is True

import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '(?P<codes>E001,E002,E003)'
    var_1 = 'E001,E002,E003'
    var_2 = module_0.match(var_0, var_1)
    var_3 = module_1._parse_error_codes(var_2)
    var_4 = bool(var_3 == ['E001', 'E002', 'E003'])
    assert var_4 is True

import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '(?P<codes> E001 , E002 , E003 )'
    var_1 = ' E001 , E002 , E003 '
    var_2 = module_0.match(var_0, var_1)
    var_3 = module_1._parse_error_codes(var_2)
    var_4 = bool(var_3 == ['E001', 'E002', 'E003'])
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 2
    var_6 = 7
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.ignore_line(var_8, var_2, var_0)
    assert var_9 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 2
    var_6 = 7
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.ignore_line(var_8, var_5, var_0)
    assert var_9 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 2
    var_6 = 7
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 3
    var_10 = module_0.ignore_line(var_8, var_9, var_0)
    assert var_10 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 2
    var_6 = 7
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 'E456'
    var_10 = module_0.ignore_line(var_8, var_2, var_9)
    assert var_10 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = set()
    var_3 = set()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5
    var_6 = module_0.ignore_line(var_4, var_5, var_0)
    assert var_6 is False



# Parsed testcases at query #3
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa'
    var_1 = 'y = 2  # noqa'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = bool(var_3 == {'all': {1, 2}})
    assert var_4 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: F401'
    var_1 = 'y = 2  # noqa: F841'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = bool(var_3 == {'F401': {1}, 'F841': {2}})
    assert var_4 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: F401, F841'
    var_1 = 'y = 2  # noqa: E711, E712'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = bool(var_3 == {'F401': {1}, 'F841': {1}, 'E711': {2}, 'E712': {2}})
    assert var_4 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa'
    var_1 = 'y = 2  # noqa: F401'
    var_2 = 'z = 3  # noqa: F841, E711'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_noqa(var_3)
    var_5 = bool(var_4 == {'all': {1}, 'F401': {2}, 'F841': {3}, 'E711': {3}})
    assert var_5 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = 'y = 2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: F'
    var_1 = 'y = 2  # noqa: E'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = bool(var_3 == {'F': {1}, 'E': {2}})
    assert var_4 is True



