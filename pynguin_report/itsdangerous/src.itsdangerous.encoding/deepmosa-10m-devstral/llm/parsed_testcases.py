####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
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
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8_'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'\xff\xfe'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_base64_decode_with_non_ascii_string. Retrieved 3/4 statements.


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
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'InvalidBase64!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'InvalidBase64!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = 'ascii'
    var_2 = 'ignore'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_base64_decode_with_non_ascii_string. Retrieved 3/4 statements.


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
    var_0 = 'SGVsbG8_'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello?'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@Base64'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = 'ascii'
    var_2 = 'ignore'



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #6
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
    var_0 = 'SGVsbG8_'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello?'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@Base64!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
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
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8_'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello?'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '8J+YgA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'\xf0\x9f\x98\x8a'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



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
    var_0 = 'SGVsbG8_'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello?'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '8J+YgA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'\xe4\xbd\xa0\xe5\xa5\xbd'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
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
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'Invalid!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '8J+YgA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'\xe4\xbd\xa0\x00'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'8J+YgA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'\xe4\xbd\xa0\x00'



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64_input'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_base64_decode_with_non_ascii_string. Retrieved 3/4 statements.


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
    var_0 = 'SGVsbG8gd29ybGQh'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world!'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@Base64!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = 'ascii'
    var_2 = 'ignore'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_base64_decode_with_non_ascii_string. Retrieved 3/4 statements.


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
    var_0 = 'SGVsbG8gd29ybGQh'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world!'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@Base64!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = 'ascii'
    var_2 = 'ignore'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_base64_decode_with_non_ascii_string. Retrieved 3/4 statements.


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
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@@@'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = 'ascii'
    var_2 = 'ignore'



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64!'
    var_1 = module_0.base64_decode(var_0)



