####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'V29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'World'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8tV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello-World'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YQ=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = 123
    var_2 = module_0.base64_decode(var_1)



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = 'invalid_chars_!@#$'
    var_2 = module_0.base64_decode(var_1)



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = module_0.base64_decode(var_1)



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.base64_decode(var_3)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = 'invalid_char_@'
    var_2 = module_0.base64_decode(var_1)



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    pass

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.base64_decode(var_0)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YQ=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'Ym9i'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'bob'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YV9iLWMA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a_b-c'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = 'Should have raised BadData'
    var_3 = AssertionError(var_2)



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'V29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'World'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YV9iLWNfZA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a_b-c_d'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YmFzZTY0'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'base64'



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YQ=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'YQ=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YV9i'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a_b'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YWI'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'ab'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_base64_decode_url_safe_chars. Retrieved 6/7 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YmFzZTY0'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'base64'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'YmFzZTY0'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'base64'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YWJ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'ab'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YV9iLWM'
    var_1 = module_0.base64_decode(var_0)
    var_2 = b'a_b-c'
    var_3 = b'a\x96\xbc'
    var_4 = 'YWE'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'aa'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.base64_decode(var_3)



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.base64_decode(var_0)



