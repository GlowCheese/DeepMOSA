####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA_-'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
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
    var_0 = b'dGVzdA\xff'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA\x80'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'ZA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'd'



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!invalid!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_base64_decode_valid_input. Retrieved 2/3 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_base64_decode_with_valid_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_base64_decode_with_valid_string_input. Retrieved 2/3 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA-_'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA\x80'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!invalid!!!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdC11cmw='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test-url'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!invalid!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b't'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA\x80'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA\n\t'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

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
    var_0 = 'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

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
    var_0 = 'dGVzdA\x80'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA+/'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test\xfb\xff'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_base64_decode_valid_data. Retrieved 2/3 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_base64_decode_valid_input_does_not_raise_exception. Retrieved 2/3 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!invalid_base64!!!'
    var_1 = None
    var_2 = module_0.base64_decode(var_0)
    assert var_2 is None



