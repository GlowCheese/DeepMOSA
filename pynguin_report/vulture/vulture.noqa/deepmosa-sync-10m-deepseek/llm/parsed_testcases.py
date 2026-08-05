####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




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
    var_0 = 'x = 1  # noqa: E301'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    var_3 = bool(var_2 == {'E301': {1}})
    assert var_3 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E301, E302'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    var_3 = bool(var_2 == {'E301': {1}, 'E302': {1}})
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
    var_0 = 'x = 1  # noqa: E301'
    var_1 = 'y = 2  # noqa: E302'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = bool(var_3 == {'E301': {1}, 'E302': {2}})
    assert var_4 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E301'
    var_1 = 'y = 2  # noqa: E301'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)
    var_4 = bool(var_3 == {'E301': {1, 2}})
    assert var_4 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: F401'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    var_3 = bool(var_2 == {'F401': {1}})
    assert var_3 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_noqa(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = '  x = 1  # noqa: E301  '
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    var_3 = bool(var_2 == {'E301': {1}})
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = {var_2, var_3, var_4}
    var_6 = set()
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = set()
    var_3 = 5
    var_4 = 6
    var_5 = {var_3, var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'W292'
    var_8 = module_0.ignore_line(var_6, var_4, var_7)
    assert var_8 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 1
    var_3 = 2
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 3
    var_8 = module_0.ignore_line(var_6, var_7, var_0)
    assert var_8 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = 10
    var_3 = {var_2}
    var_4 = 20
    var_5 = {var_4}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 15
    var_8 = 'F401'
    var_9 = module_0.ignore_line(var_6, var_7, var_8)
    assert var_9 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E501'
    var_1 = 'all'
    var_2 = set()
    var_3 = set()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 1
    var_6 = module_0.ignore_line(var_4, var_5, var_0)
    assert var_6 is False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = {var_2, var_3, var_4}
    var_6 = set()
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = set()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_3, var_4, var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = 'E456'
    var_9 = module_0.ignore_line(var_7, var_4, var_8)
    assert var_9 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 1
    var_3 = {var_2}
    var_4 = set()
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 2
    var_7 = module_0.ignore_line(var_5, var_6, var_0)
    assert var_7 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = set()
    var_3 = set()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 'E789'
    var_7 = module_0.ignore_line(var_4, var_5, var_6)
    assert var_7 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E111'
    var_1 = 'all'
    var_2 = 10
    var_3 = 20
    var_4 = {var_2, var_3}
    var_5 = 30
    var_6 = {var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 1
    var_3 = {var_2}
    var_4 = {var_2}
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_0.ignore_line(var_5, var_2, var_0)
    assert var_6 is True

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



# Parsed testcases at query #2
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1\ny = 2\n'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa\ny = 2\n'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = bool(var_1 == {'all': {1}})
    assert var_2 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E501\ny = 2\n'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = bool(var_1 == {'E501': {1}})
    assert var_2 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E501, W503\ny = 2\n'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = bool(var_1 == {'E501': {1}, 'W503': {1}})
    assert var_2 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E501\ny = 2  # noqa: W503\n'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = bool(var_1 == {'E501': {1}, 'W503': {2}})
    assert var_2 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E501\ny = 2  # noqa: E501\n'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = bool(var_1 == {'E501': {1, 2}})
    assert var_2 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: F401\n'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = bool(var_1 == {'F401': {1}})
    assert var_2 is True

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa:\ny = 2\n'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = bool(var_1 == {'all': {1}})
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------




import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '(?P<codes>.*)'
    var_1 = 'all'
    var_2 = module_0.match(var_0, var_1)
    var_3 = module_1._parse_error_codes(var_2)
    var_4 = bool(var_3 == ['all'])
    assert var_4 is True

import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '(?P<codes>.*)'
    var_1 = 'E001'
    var_2 = module_0.match(var_0, var_1)
    var_3 = module_1._parse_error_codes(var_2)
    var_4 = bool(var_3 == ['E001'])
    assert var_4 is True

import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '(?P<codes>.*)'
    var_1 = 'E001, E002, E003'
    var_2 = module_0.match(var_0, var_1)
    var_3 = module_1._parse_error_codes(var_2)
    var_4 = bool(var_3 == ['E001', 'E002', 'E003'])
    assert var_4 is True

import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '(?P<codes>.*)'
    var_1 = '  E001  ,  E002  '
    var_2 = module_0.match(var_0, var_1)
    var_3 = module_1._parse_error_codes(var_2)
    var_4 = bool(var_3 == ['E001', 'E002'])
    assert var_4 is True

import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '(?P<codes>.*)'
    var_1 = ''
    var_2 = module_0.match(var_0, var_1)
    var_3 = module_1._parse_error_codes(var_2)
    var_4 = bool(var_3 == ['all'])
    assert var_4 is True

import re as module_0
import vulture.noqa as module_1

def test_case_0():
    var_0 = '(?P<codes>.*)'
    var_1 = None
    var_2 = module_0.match(var_0, var_1)
    var_3 = module_1._parse_error_codes(var_2)
    var_4 = bool(var_3 == ['all'])
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parse_noqa_with_no_noqa_comments. Retrieved 2/3 statements.
# Partially parsed test_parse_noqa_with_single_error_code. Retrieved 6/7 statements.
# Partially parsed test_parse_noqa_with_multiple_error_codes. Retrieved 8/9 statements.
# Partially parsed test_parse_noqa_with_all_error_codes. Retrieved 6/7 statements.
# Partially parsed test_parse_noqa_with_multiple_lines. Retrieved 9/10 statements.
# Partially parsed test_parse_noqa_with_mapped_error_code. Retrieved 6/7 statements.


import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1\ny = 2'
    var_1 = module_0.parse_noqa(var_0)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E501'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'E501'
    var_3 = 1
    var_4 = {var_3}
    var_5 = {var_2: var_4}

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E501, W601'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'E501'
    var_3 = 'W601'
    var_4 = 1
    var_5 = {var_4}
    var_6 = {var_4}
    var_7 = {var_2: var_5, var_3: var_6}

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'all'
    var_3 = 1
    var_4 = {var_3}
    var_5 = {var_2: var_4}

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E501\ny = 2  # noqa: W601'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'E501'
    var_3 = 'W601'
    var_4 = 1
    var_5 = {var_4}
    var_6 = 2
    var_7 = {var_6}
    var_8 = {var_2: var_5, var_3: var_7}

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E501'
    var_1 = module_0.parse_noqa(var_0)
    var_2 = 'E501'
    var_3 = 1
    var_4 = {var_3}
    var_5 = {var_2: var_4}



# Parsed testcases at query #5
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'import os  # noqa'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)
    var_3 = bool(var_2)
    assert var_3 is True



