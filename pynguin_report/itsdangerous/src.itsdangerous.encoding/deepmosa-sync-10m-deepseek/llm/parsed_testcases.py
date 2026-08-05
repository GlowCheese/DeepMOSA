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
    var_0 = b''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'



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
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8_d29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello?world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = '!!!'
    var_2 = module_0.base64_decode(var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'aGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

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
    var_0 = 'dGVzdA-_'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test\xef\xbc\x9f'

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
    var_0 = b'd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8td29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello-world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'



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
    var_0 = b''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA\x80'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



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
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'!!!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8gd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello world'



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



