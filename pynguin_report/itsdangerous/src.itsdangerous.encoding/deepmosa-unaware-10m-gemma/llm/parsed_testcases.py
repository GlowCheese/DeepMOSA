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
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'Y29uY29y'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'concor'
    var_6 = 'YW55'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'any'
    var_8 = 'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = b'dGVzdA'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YS1i'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a/b'
    var_6 = b'V29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = 'SGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'Python_is_Awesome_123'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = module_0.want_bytes(var_16)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'dGVzdF9kYXRh'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test_data'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!Invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = b'YmFzZTY0'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'base64'



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'V29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'YS1i'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a/b'
    var_6 = 'YWJj'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'abc'
    var_8 = 'YWJjZA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'abcd'
    var_10 = '4pyo'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xe2\x9c\x94'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9i'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a_b'
    var_6 = 'YV8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a_'
    var_8 = 'SGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8gd29ybGQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello world'
    var_12 = b'Ym9i'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'bob'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YmFzZTY0'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'base64'
    var_6 = module_0.base64_decode(var_4)
    assert var_6 == b'base64'
    var_7 = 'SGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = b'\xff\xfe\xfd'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xfe\xfd'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'V29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'World'
    var_6 = 'Yy1i'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'c/b'
    var_8 = 'some_string_with_-chars'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_1.decode()
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '-_8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\xfb\xff'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YmI'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'bb'
    var_14 = 'Y2Ji'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'cbb'
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = b'SGVsbG8gd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = 'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'YmFzZTY0LS1f'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'base64-+_'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'V29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'World'
    var_6 = 'YS1iPw'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a/b?'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'\x00\xff\xfe\x01\x02\x03'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    var_13 = '!!!NotBase64!!!'
    var_14 = module_0.base64_decode(var_13)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YS1i'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a/b'
    var_6 = 'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'any carnal pleasure.'
    var_8 = b'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'any carnal pleasure.'
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = ''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9iLWM'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a_b-c'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'V29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'World'
    var_12 = 'YWJj'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abc'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8=🚀'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'SGVsbG8gd29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = b'YmFzZTY0'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'base64'
    var_6 = 'dGVzdF9kYXRh'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test_data'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = '\x00\x01\x02'
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YW55LWNvbmZpZ3VyYXRpb24='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'any-configuration'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'V29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'YWJjሴZGVm'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abcdef'



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '__4'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\xff\xfe'
    var_6 = b'Ym9i'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'bob'
    var_8 = 'Ym9i'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'bob'
    var_10 = 'Ym9'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'bo'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YV9i'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a_b'
    var_4 = b'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YWI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'ab'
    var_10 = 'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'any carnal pleasure.'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = b'YmFzZTY0'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'base64'
    var_4 = 'YWJj'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'abc'
    var_6 = '-_'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xff\xef'
    var_8 = 'SGVsbG8gd29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello world'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '-_'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\xfb\xff'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'Ym9i'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'bob'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'Ym9i'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'bob'
    var_16 = 'Ym9'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'bo'



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'a-b_c'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a+b/c'
    var_4 = b'VGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Test'
    var_6 = 'Ym9v'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'boo'
    var_8 = 'Ym9'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'bo'
    var_10 = 'SGVsbG8\n'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x00\xff\x00\xff'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = 'YWJjZA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'abcd'
    var_10 = 'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'YQ'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = 'YWI'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'ab'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x00\xff\xde\xad\xbe\xef'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = 'YWJjXw'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'abc/'
    var_10 = 'YWJj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'abc'
    var_12 = 'SGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = '!!!Invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'c3ViamVjdD8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'subject?'
    var_4 = b'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'any carnal pleasure.'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'Yg'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'b'
    var_10 = 'Yw'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'c'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'testing-123_abc'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = module_0.want_bytes(var_16)



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YS1iX2M='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a-b_c'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YWI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'ab'
    var_10 = module_0.base64_decode(var_0)
    var_11 = b'SGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    var_13 = '!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YmFzZTY0LS1f'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'base64-_'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'base64-_'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = '!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = '123'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'123'



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'c3ViamVjdD8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'subject?'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8=🚀'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '_v4'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\xff\xfe'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'YmFzZTY0'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'base64'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'YWJjZGU'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abcde'



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'c3ViamVjdD8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'subject?'
    var_4 = b'Ym95'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'boy'
    var_6 = 'Ym95'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'boy'
    var_8 = 'c3ViamVjdD'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'subject'
    var_10 = 'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'any carnal pleasure.'
    var_12 = '!!!not_base64!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YQ--'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a\xff'
    var_6 = 'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b'dGVzdA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = 'YWJj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'abc'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YS1i'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a/b'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'V29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'World'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'testing_123_special_chars_!@#'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'YW55IGNhcm5hbCBwbGVhc3VyZS4'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'any carnal pleasure.'
    var_4 = b'YmFzZTY0'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'base64'
    var_6 = '-_'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xfb\xff'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!not_base64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'VGVzdA'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Test'



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YWFfYmI'
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'aab_b'
    var_7 = b'_'
    var_8 = b'/'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = 'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'any carnal pleasure.'
    var_15 = 'SGVsbG8$'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = '!!!'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = '__4'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'\xff\xfe'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8====='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '-_'
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'-_'
    var_7 = b'=='
    var_8 = var_6 + var_7
    var_9 = module_1.urlsafe_b64decode(var_8)
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YmI'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'bb'
    var_14 = 'dGVzdA'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test'
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = ''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'c3ViamVjdD8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'subject?'
    var_4 = b'Ym9i'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'bob'
    var_6 = 'YW55'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'any'
    var_8 = 'YW55YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'anya'
    var_10 = '!!!not_base64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'a-b_'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a+b/'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = 'abc'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'abc'
    var_12 = 'YW55'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'any'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x00\xff\x00\xff'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = b'YW4'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'an'
    var_10 = b'test+data/with_special_chars'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = b''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'c3ViamVjdD9mb289YmFy'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'subject?foo=bar'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'vz__'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xfb\xbf\xbf'



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'i_8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\x8b\xff'
    var_6 = b'YmFzZTY0'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'base64'
    var_8 = 'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'YWJjMTIzXy1f'
    var_15 = b'abc123_-'
    var_16 = module_0.base64_decode(var_14)



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9i'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a_b'
    var_6 = 'YV-i'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a\xbe'
    var_8 = 'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = 'YWI'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = b'Some'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Some'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x01\x02\x03\xff'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = b'-_+'
    var_9 = 'YQ'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = module_0.base64_decode(var_9)
    assert var_11 == b'a'
    var_12 = 'YW55'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'any'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = b'SGVsbG8='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #41
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '-_8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\xfb\xff'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = 'YQ=='
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'YQ=='
    var_16 = module_0.base64_decode(var_15)
    var_17 = '!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'invalid@charset'
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = b'SGVsbG8gd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = 'SGVsbG8gd29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello world'
    var_6 = 'YS1i'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a/b'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = '?'
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'SGVsbG8gd29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = b'SGVsbG8gd29ybGQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello world'
    var_6 = 'YS1iXw'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a-b_'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!not_base64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'YWJj😀'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abc'



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x00\xff\xde\xad\xbe\xef'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = b'-_8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xfb\xff'
    var_10 = 'YWI'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = 'SGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = '!!!not_base64!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #45
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = '-_8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xff\xbf'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'YWJjሴ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'Yv8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a\xbf'
    var_6 = 'YW55'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'any'
    var_8 = 'YW55YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'anya'
    var_10 = 'YW55YWJj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'anyabc'
    var_12 = b'Ym9i'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'bob'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'\xff\xff\xff'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x00\xff\xde\xad\xbe\xef'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = b'\xff\xef'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = b'SGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = b'SGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = b'!@#$%^&*'
    var_16 = module_0.base64_decode(var_15)
    var_17 = ''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''
    var_19 = b''
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b''



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '-_7'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\xfb\xff'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'V29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'World'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'not_base64_@#$'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'U3ViamVjdD8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Subject?'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'Ym'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'b'
    var_10 = 'YmI='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'bb'
    var_12 = 'YmI'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'YmI'
    var_15 = module_0.base64_decode(var_14)
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'invalid_chars_@#$'
    var_19 = module_0.base64_decode(var_18)
    var_20 = ''
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b''



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'test'
    var_6 = module_1.urlsafe_b64encode(var_5)
    var_7 = b'='
    var_8 = b'dGVzdA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = b'\xff\xfe\xfd'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    var_13 = '!!!not_base64!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = b''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'YQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a'
    var_4 = 'YmI'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'bb'
    var_6 = b'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'any carnal pleasure.'
    var_8 = 'dGVzdC13aXRoX3NwZWNpYWw'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-with_special'
    var_10 = '!!!not_base64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #52
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x00\xff\xde\xad\xbe\xef'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = b'\xfb\xff\xbf'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = 'SGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = 'SGVsbG8'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = '!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = ''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''



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
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YS1iX2M'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a-b_c'
    var_6 = 'YQ=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YWI='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'ab'
    var_10 = 'YWJj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'abc'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'VGVzdA'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Test'



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9i'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a\xbe'
    var_6 = 'YV8b'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a\xbf'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'YWJj'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abc'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'invalid-base64-content-\x00'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'c3ViamVjdD9hPTE'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'subject?a=1'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'Ym9i'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'bob'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8©'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'Y29uZmlybV90ZXN0'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'confirm_test'
    var_4 = b'R29vZA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Good'
    var_6 = 'YWJjZGU'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'abcde'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!not_base64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'YWJjZGUÿ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abcde'



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = b'SGVsbG8gd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = 'YW55IHN0cmluZw'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'any string'
    var_6 = 'YW55IHN0cmluZw=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'any string'
    var_8 = '-_8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xfa\xff'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'YWJj😀'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'YQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a'
    var_4 = 'YmI'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'bb'
    var_6 = b'YmFzZTY0'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'base64'
    var_8 = 'YV9i'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a_b'
    var_10 = '!!!not_base64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YV9iLWM'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a_b-c'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = 'SGVsbG8\n'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9iLWM'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a_b-c'
    var_6 = 'YW55IGNhcm5hbCBwbGVhc3VyZS4'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'any carnal pleasure.'
    var_8 = b'dGVzdA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = '!!!not_base64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'YmFzZTY0'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'base64'
    var_6 = 'c3ViamVjdD8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'subject?'
    var_8 = 'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'any carnal pleasure.'
    var_10 = '!!!NotBase64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = b'YmFzZTY0'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'base64'
    var_4 = 'YS1iXw'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a-b_'
    var_6 = 'SGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8🚀'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'ascii'
    var_4 = module_1.decode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'teststring'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = b'\xbe\xef'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'abc!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'\x00\x01\x02'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = b'SGVsbG8='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9iLWM='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a_b-c'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b'\x00\x01\x02'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\x00\x01\x02'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'SGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'Hello'
    var_7 = b'Unittest'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Unittest'
    var_9 = b'YQ-'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a/'
    var_11 = b'!!!not_base64!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'c3ViamVjdD8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'subject?'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'YmFzZTY0'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'base64'
    var_10 = '!!!Invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8====='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YWJj'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'abc'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YWI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'ab'
    var_10 = module_0.base64_decode(var_2)
    assert var_10 == b'abc'
    var_11 = 'u_8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'\xb8\xbe'
    var_13 = '!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = b'SGVsbG8gd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = 'SGVsbG8gd29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello world'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'\xff\xef'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = '!!!NotBase64!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = 'A'
    var_14 = 100
    var_15 = var_13 * var_14
    var_16 = module_0.base64_decode(var_15)
    var_17 = '='
    var_18 = len(var_15)
    var_19 = 4



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YS1i'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a/b'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = 'YWI'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = 'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'any carnal pleasure.'
    var_14 = b'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'any carnal pleasure.'
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'invalid_chars_@#$%^&*'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'abc/def+'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'abc\xfe\xef'
    var_6 = '-_8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xfb\xff'
    var_8 = 'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = 'YmI'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'bb'
    var_12 = 'Y2Nj'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ccc'
    var_14 = b'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = 'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = '!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = ''
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b''
    var_22 = b''
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b''



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'u_4'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\xff\xfe'
    var_6 = b'YmFzZTY0'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'base64'
    var_8 = 'YW55'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'any'
    var_10 = 'YW55YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'anya'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YV9iLWNfZA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a_b-c_d'
    var_4 = b'R29vZA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Good'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YmI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'bb'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8=🚀'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x01\x02\x03\xff'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = 'YWJj'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'abc'
    var_10 = 'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'YQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YWI'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = ''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = b'\x00\x01\x02'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YW55LWNhcm5hbCBwb3N0'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'any-carnal post'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!Invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'Ym9i'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'bob'



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '_w'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\xff\xfe'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YmI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'bb'
    var_10 = b'VGVzdA'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Test'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x01\x02\x03\xff'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = b'-_8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xfb\xff'
    var_10 = -1
    var_11 = 'testing'
    var_12 = base64_encode(var_11)[:var_10]
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'testing'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YWJj'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'abc'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_2)
    assert var_6 == b'abc'
    var_7 = 'YQ'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = '__8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'\xff\xef'
    var_11 = '!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = b'SGVsbG8gd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = 'YQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a'
    var_6 = 'Ym'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'b'
    var_8 = 'Y2M'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'cc'
    var_10 = 'i__'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'i\xff'
    var_12 = 'aa--'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a\xee'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'SGVsbG8gd29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = '-_'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xf8'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'pytest unit testing'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = 'utf-8'
    var_7 = module_1.encode(var_6)
    var_8 = b'\xff\xfe\xff'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = b'YWJj'
    var_12 = b'YWI'
    var_13 = module_0.base64_decode(var_11)
    assert var_13 == b'abc'
    var_14 = module_0.base64_decode(var_12)
    assert var_14 == b'ab'
    var_15 = b'!@#$%^&*'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9i'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a_b'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'c3ViamVjdD'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'subject'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'YmFzZTY0'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'base64'



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'cHl0aG9u'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'python'
    var_7 = b'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'any carnal pleasure.'
    var_9 = b'_-ab'
    var_10 = module_0.base64_decode(var_9)
    var_11 = '!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = '_v4'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'\xff\xfe'
    var_4 = b'V29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'World'
    var_6 = 'YmFzZTY0'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'base64'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!NotBase64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8=🚀'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'c3ViamVjdD8gZGF0YQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'subject? data'
    var_6 = b'Ym9v'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'boo'
    var_8 = 'Ym9v'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'boo'
    var_10 = 'YWI'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9i'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a_b'
    var_6 = 'YW55IGNhcm5hbCBwbGVhc3VyZS4'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'any carnal pleasure.'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'dGVzdA'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YWJj'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'abc'
    var_6 = '-_8'
    var_7 = '-_'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'\xf1\xbd'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'abc'
    var_10 = 'YWI'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = '8J+Zjw=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xf0\x9f\x98\x98'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'not_base64_at_all'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'ZGF0YS10ZXN0Xw'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'data-test_'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YmI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'bb'
    var_10 = 'Y2Nj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ccc'
    var_12 = b'ABC'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ABC'
    var_14 = '!!!not_base64!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '-_'
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'-_=='
    var_7 = module_1.urlsafe_b64decode(var_6)
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'Ym9v'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'boo'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'A'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'YQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a'
    var_4 = 'YWI'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'ab'
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'YV8tYg'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a_b'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'a-b_'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a+b/'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8=🚀'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'V29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'World'
    var_6 = 'YV9iX2M-'
    var_7 = b'a_b_c'
    var_8 = 'YV9iX2M'
    var_9 = module_0.base64_decode(var_8)
    var_10 = 'YV9iX2M='
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'Python_is_fun_123'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = module_1.encode()
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = ''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YS9iKw'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a/b+'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'SGVsbG'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hel'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8=©'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'



# Parsed testcases at query #41
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x00\xff\xfe\x12'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = 'YWJj'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'abc'
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'c3ViamVjdD8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'subject?'
    var_4 = b'Ym9uam91cg'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'bonjour'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YWI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'ab'
    var_10 = 'YWJj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'abc'
    var_12 = 'VGVzdA=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Test'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YV9i'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a_b'
    var_4 = b'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YWI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'ab'
    var_10 = 'YWJj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'abc'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'SGVsbG8=\xff'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9iLWNfZA'
    var_5 = module_0.base64_decode(var_4)
    var_6 = 'a_b-c_d'
    var_7 = module_0.base64_encode(var_6)
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'YWI'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #45
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'Pj4_'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'>>\xff'
    var_6 = 'Pj4-'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'>>\xfe'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'YQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YWI'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'YWJj😀'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abc'



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'V29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'World'
    var_6 = '____'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xff\xfe'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'YQ=='
    var_13 = 'ascii'
    var_14 = module_0.base64_decode(var_12)
    assert var_14 == b'a'



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = b'SGVsbG8gd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = 'YQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a'
    var_6 = 'Ym'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'b'
    var_8 = 'YmFzZTY0LQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'base64-'
    var_10 = 'YmFzZTY0Xw'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'base64_'
    var_12 = '8J+Zjw=='
    var_13 = '='
    var_14 = ''
    var_15 = '!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = module_0.base64_decode(var_14)
    assert var_17 == b''



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = b'SGVsbG8gd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = 'YWJjZGU'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'abcde'
    var_6 = '-_=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xff\xef'
    var_8 = 'YV9i'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a_b'
    var_10 = 'YV-i'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a-b'
    var_12 = 'SGVsbG8gd29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello world'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'c3ViamVjdD8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'subject?'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YmI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'bb'
    var_10 = 'Y2Jj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ccc'
    var_12 = b'VGVzdA'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Test'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'pythoo_n'
    var_4 = 'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = module_0.base64_decode(var_4)
    assert var_8 == b'test'
    var_9 = b'\xff\xfe\xfd\xfc'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = '!!!not_base64!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YmFzZTY0KC0='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'base64(+)'
    var_4 = 'YmFzZTY0LS1f'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'base64-+_'
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'SGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'YmFzZTY0'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'base64'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8©'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



# Parsed testcases at query #52
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9iLWNfZA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a_b-c_d'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = b'YmFzZTY0'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'base64'
    var_11 = '!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''



# Parsed testcases at query #53
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9iLWM'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a_b-c'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'V29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'World'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8======'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'



