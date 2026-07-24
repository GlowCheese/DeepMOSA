####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YV9iLWNfZA=='
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
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid base64-encoded data'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



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
    var_0 = 'YQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '-_'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'\xff'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_base64_decode_url_safe_hyphen. Retrieved 5/6 statements.


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
    var_0 = 'YV9iY19k'
    var_1 = module_0.base64_decode(var_0)
    var_2 = b'a_bc_d'
    var_3 = b'd'
    var_4 = b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YWJj'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'abc'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YWJjZA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'abcd'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #5
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
    var_0 = 'YWJj'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'abc'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #6
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
    var_0 = 'YV9iLWM='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a_b-c'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid base64-encoded data'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



# Parsed testcases at query #7
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
    var_0 = 'YV9iLWM='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a_b-c'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YW55'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'any'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YQ=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a'



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YV9i'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a\xbe'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YWJj'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'abc'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YWJjZA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'abcd'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



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
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '-_=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'\xfb\xff'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = 'Should have raised BadData'
    var_3 = AssertionError(var_2)
    var_4 = 'Invalid base64-encoded data'



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YVgk'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'???'
    var_1 = b'!!!'
    var_2 = module_0.base64_decode(var_1)



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'YmFzZTY0'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'base64'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YWJj'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'abc'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YV9i'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a_b'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid base64-encoded data'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



# Parsed testcases at query #5
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
    var_0 = 'YV9i'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a_b'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YWJj'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'abc'

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
    var_4 = 'Invalid base64-encoded data'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8=©'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'\xff'
    var_1 = 'SGVsbG8='
    var_2 = b'Hello'
    var_3 = module_0.base64_decode(var_1)
    var_4 = bool(var_3 == var_2)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'abc%123'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(str(e).find('Invalid base64-encoded data') != -1)
    assert var_3 is True



