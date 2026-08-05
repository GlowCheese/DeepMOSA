####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_base64_decode_returns_bytes. Retrieved 2/3 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)

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
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '====='
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
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
    var_0 = '!!!invalid!!!'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA==é'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA+/'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'ultra'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA\n=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



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
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Pj4_Pz8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'>>??'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Zm9vYmFy'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'foobar'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!invalid!!!'
    var_1 = module_0.base64_decode(var_0)



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
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8tV29ybGQ_'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello-World?'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!invalid!!!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #8
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
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Pz4_Pz4_Pz4_Pz4_Pz4_Pz4_Pz4_Pz4='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'>?>?>?>?>?>?>?>?'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!invalid!!!'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA==\x80'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'



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
    var_0 = 'PDw_Pz8-Pg=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'<<??>>'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'ZA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'd'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'ISFAIyQlXiYqKCk'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'!@#$%^&*()'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = 'ÿ'
    var_2 = var_0 + var_1
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!invalid!!!'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8$'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #2
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA!@#'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA_-'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA_-'
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
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #3
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
    var_0 = 'aGVsbG8td29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello-world'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!invalid!!!'
    var_1 = module_0.base64_decode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'a GVs\nbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'



# Parsed testcases at query #4
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
    var_0 = 'dGVzdA!!'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
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




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
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
    var_0 = 'dGVzdA__'
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
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '!!!'
    var_1 = module_0.base64_decode(var_0)



# Parsed testcases at query #7
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
    var_0 = 'aGVsbG8t'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello-'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b''

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'invalid!'
    var_1 = module_0.base64_decode(var_0)
    var_2 = 'Expected BadData exception'
    var_3 = AssertionError(var_2)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'not base64'
    var_1 = module_0.base64_decode(var_0)
    var_2 = 'Expected BadData exception'
    var_3 = AssertionError(var_2)



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'



