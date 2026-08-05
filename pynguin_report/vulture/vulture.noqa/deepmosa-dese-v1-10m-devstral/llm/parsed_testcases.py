####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parse_noqa_empty_code. Retrieved 2/3 statements.
# Partially parsed test_parse_noqa_no_noqa_lines. Retrieved 4/5 statements.
# Partially parsed test_parse_noqa_single_all. Retrieved 7/8 statements.
# Partially parsed test_parse_noqa_single_specific_code. Retrieved 7/8 statements.
# Partially parsed test_parse_noqa_multiple_codes. Retrieved 9/10 statements.
# Partially parsed test_parse_noqa_multiple_lines. Retrieved 11/12 statements.
# Partially parsed test_parse_noqa_mixed_all_and_specific. Retrieved 11/12 statements.
# Partially parsed test_parse_noqa_with_code_mapping. Retrieved 9/10 statements.


import vulture.noqa as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_noqa(var_0)

import vulture.noqa as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = 'x = 1'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa'
    var_1 = [var_0]
    var_2 = 'all'
    var_3 = 1
    var_4 = {var_3}
    var_5 = {var_2: var_4}
    var_6 = module_0.parse_noqa(var_1)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E123'
    var_1 = [var_0]
    var_2 = 'E123'
    var_3 = 1
    var_4 = {var_3}
    var_5 = {var_2: var_4}
    var_6 = module_0.parse_noqa(var_1)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E123, F456'
    var_1 = [var_0]
    var_2 = 'E123'
    var_3 = 'F456'
    var_4 = 1
    var_5 = {var_4}
    var_6 = {var_4}
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = module_0.parse_noqa(var_1)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E123'
    var_1 = 'y = 2  # noqa: F456'
    var_2 = [var_0, var_1]
    var_3 = 'E123'
    var_4 = 'F456'
    var_5 = 1
    var_6 = {var_5}
    var_7 = 2
    var_8 = {var_7}
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = module_0.parse_noqa(var_2)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa'
    var_1 = 'y = 2  # noqa: E123'
    var_2 = [var_0, var_1]
    var_3 = 'all'
    var_4 = 'E123'
    var_5 = 1
    var_6 = {var_5}
    var_7 = 2
    var_8 = {var_7}
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = module_0.parse_noqa(var_2)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1  # noqa: E123'
    var_1 = [var_0]
    var_2 = 'E123'
    var_3 = 'E999'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = {var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.parse_noqa(var_1)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parse_error_codes_with_no_codes. Retrieved 8/10 statements.
# Partially parsed test_parse_error_codes_with_single_code. Retrieved 8/10 statements.
# Partially parsed test_parse_error_codes_with_multiple_codes. Retrieved 8/10 statements.
# Partially parsed test_parse_error_codes_with_whitespace. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 'Match'
    var_1 = ()
    var_2 = 'groupdict'
    var_3 = 'codes'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = lambda : var_5
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'Match'
    var_1 = ()
    var_2 = 'groupdict'
    var_3 = 'codes'
    var_4 = 'E001'
    var_5 = {var_3: var_4}
    var_6 = lambda : var_5
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'Match'
    var_1 = ()
    var_2 = 'groupdict'
    var_3 = 'codes'
    var_4 = 'E001, E002, E003'
    var_5 = {var_3: var_4}
    var_6 = lambda : var_5
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'Match'
    var_1 = ()
    var_2 = 'groupdict'
    var_3 = 'codes'
    var_4 = '  E001  ,  E002  '
    var_5 = {var_3: var_4}
    var_6 = lambda : var_5
    var_7 = {var_2: var_6}



# Parsed testcases at query #2
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 3
    var_6 = {var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = module_0.ignore_line(var_7, var_2, var_0)
    assert var_8 is True
    var_9 = module_0.ignore_line(var_7, var_3, var_0)
    assert var_9 is True
    var_10 = 7
    var_11 = module_0.ignore_line(var_7, var_10, var_0)
    assert var_11 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = 10
    var_4 = {var_2, var_3}
    var_5 = 3
    var_6 = 8
    var_7 = {var_5, var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.ignore_line(var_8, var_5, var_0)
    assert var_9 is True
    var_10 = 'E456'
    var_11 = module_0.ignore_line(var_8, var_6, var_10)
    assert var_11 is True
    var_12 = module_0.ignore_line(var_8, var_2, var_10)
    assert var_12 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = set()
    var_3 = set()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 1
    var_6 = module_0.ignore_line(var_4, var_5, var_0)
    assert var_6 is False
    var_7 = 2
    var_8 = 'E456'
    var_9 = module_0.ignore_line(var_4, var_7, var_8)
    assert var_9 is False

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'E123'
    var_1 = 'all'
    var_2 = 5
    var_3 = {var_2}
    var_4 = 3
    var_5 = {var_4}
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'E456'
    var_8 = module_0.ignore_line(var_6, var_2, var_7)
    assert var_8 is False
    var_9 = module_0.ignore_line(var_6, var_4, var_7)
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parse_error_codes_with_empty_codes. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 'Match'
    var_1 = ()
    var_2 = 'groupdict'
    var_3 = 'codes'
    var_4 = ''
    var_5 = {var_3: var_4}
    var_6 = lambda : var_5
    var_7 = {var_2: var_6}



# Parsed testcases at query #4
#--------------------------




import vulture.noqa as module_0

def test_case_0():
    var_0 = '# noqa: E123'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)

import vulture.noqa as module_0

def test_case_0():
    var_0 = '# noqa: E123, F456'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)

import vulture.noqa as module_0

def test_case_0():
    var_0 = '# noqa'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)

import vulture.noqa as module_0

def test_case_0():
    var_0 = '# noqa: E123'
    var_1 = 'x = 1  # noqa: F456'
    var_2 = 'y = 2'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.parse_noqa(var_3)

import vulture.noqa as module_0

def test_case_0():
    var_0 = '# noqa: W123'
    var_1 = [var_0]
    var_2 = module_0.parse_noqa(var_1)

import vulture.noqa as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.parse_noqa(var_0)

import vulture.noqa as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = 'y = 2'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_noqa(var_2)



