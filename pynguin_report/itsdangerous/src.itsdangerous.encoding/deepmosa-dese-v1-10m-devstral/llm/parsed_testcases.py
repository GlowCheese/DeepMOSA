####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




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
    var_0 = 'SGVsbG8gV29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8_V29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@Base64!'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'Invalid@Base64!'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '8J+YgA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'\xff\xfe'



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
    var_0 = 'SGVsbG8gd29ybGQ-'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8gd29ybGQ-'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@@@'
    var_1 = module_0.base64_decode(var_0)

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
    var_0 = 'SGVsbG8gd29ybGQ=©'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'



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
    var_0 = 'invalid_base64_input'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #7
#--------------------------




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
    var_0 = 'SGVsbG8_V29ybGQh'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World!'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@@Base64'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQh'
    var_1 = 'ascii'
    var_2 = 'ignore'
    var_3 = module_0.base64_decode(var_0)
    assert var_3 == b'Hello World!'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '8J+YgCBIZWxsbyBXb3JsZCE='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'\xe4\xbd\xa0 \xe5\x93\x88 Hello World!'



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64!'
    var_1 = module_0.base64_decode(var_0)



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
    assert var_1 == b'Hello?'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = 'ascii'
    var_2 = 'ignore'
    var_3 = module_0.base64_decode(var_0)
    assert var_3 == b'Hello'



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64_string'
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
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid@Base64'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '8J+YgA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'\x00\x00'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64_input'
    var_1 = module_0.base64_decode(var_0)



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
    var_0 = 'invalid_base64!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid_base64_input'
    var_1 = module_0.base64_decode(var_0)



