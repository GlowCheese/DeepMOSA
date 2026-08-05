####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
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



# Parsed testcases at query #2
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
    var_0 = 'Invalid@@@'
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
    assert var_1 == b'\x1f\x8b\x08\x00'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'\xe2\x82\xac'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_base64_decode_with_non_ascii_string. Retrieved 3/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQh'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World!'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8gV29ybGQh'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World!'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8_V29ybGQh'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello_World!'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'InvalidBase64!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQh'
    var_1 = 'ascii'
    var_2 = 'ignore'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'



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
    var_0 = 'SGVsbG8td29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello-world'

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

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = 'ascii'
    var_2 = 'ignore'



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64_string'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #7
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
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@Base64'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'Invalid@Base64'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'SGVsbG8='
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_base64_decode_with_non_ascii_string_input. Retrieved 3/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8_V29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8_V29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'

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

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = 'ascii'
    var_2 = 'ignore'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
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
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'

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

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = 'ascii'
    var_2 = 'ignore'



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #3
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
    var_0 = 'Invalid@@@'
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
    assert var_1 == b'\x00\x00\x00\x00'



# Parsed testcases at query #4
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
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8=='
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
    var_0 = 'invalid_base64_string'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64!'
    var_1 = module_0.base64_decode(var_0)



