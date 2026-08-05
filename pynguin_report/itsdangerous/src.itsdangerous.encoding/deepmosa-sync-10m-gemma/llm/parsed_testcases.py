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
    var_0 = b'SGVsbG8gd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YV9i'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a_b'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!NotBase64!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



# Parsed testcases at query #2
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
    var_0 = None
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #3
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
    var_0 = 'YV9iLWNfZA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a_b-c_d'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YW55'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'any'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YW5'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'an'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid base64-encoded data'



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8gd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YV9iLWNfZA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'a_b-c_d'

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
    var_3 = bool(True)
    assert var_3 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    pass

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YmFzZTY0'
    var_1 = module_0.base64_decode(var_0)
    var_2 = b'base64'
    var_3 = var_1 == var_2



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #7
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
    var_0 = 'YW55'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'any'
    var_2 = module_0.base64_decode(var_0)
    assert var_2 == b'any'

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
    var_3 = bool(True)
    assert var_3 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.base64_decode(var_0)

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
    var_0 = 'YmFzZTY0-'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'base64\xee'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'YW55'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'any'

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
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.base64_decode(var_0)



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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_base64_decode_raises_bad_data_on_invalid_input. Retrieved 2/7 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(var_0)
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = b'\x00\xff\xfe'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #8
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
    var_0 = '-_'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'\xfa'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(str(e).find('Invalid base64-encoded data') != -1)
    assert var_3 is True



