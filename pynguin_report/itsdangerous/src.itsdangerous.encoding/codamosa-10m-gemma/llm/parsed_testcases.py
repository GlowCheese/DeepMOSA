####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'z_4'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'\xff\xfe'
    var_4 = b'V29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'World'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = '!!!'
    var_9 = module_0.base64_decode(var_8)
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'YWI'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'ab'
    var_7 = b'c29tZSBkYXRh'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'some data'
    var_9 = 'Y29kZQ=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'code'
    var_11 = b'c3ViLWl0ZW1fMQ'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'sub-item_1'
    var_13 = '!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'\xff\xfe\xfd!"'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = 'python testing'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = 'utf-8'
    var_10 = module_1.encode(var_9)
    var_11 = b'SGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = b'SGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = b'!!!not_base64!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''
    var_19 = ''
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b''



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'U3ViamVjdD8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Subject?'
    var_4 = b'R29vZA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Good'
    var_6 = 'Y29kZQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'code'
    var_8 = 'Y29kZ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'code'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!Invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'YQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = 'YQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'



# Parsed testcases at query #5
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
    var_8 = 'testing_special-chars'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_1.encode(var_3)
    var_12 = b'a'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'a'
    var_15 = b'!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = ''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''
    var_19 = b''
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b''
    var_21 = 'Ym9i'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'bob'



# Parsed testcases at query #6
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
    var_6 = 'test-abc_'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'test-abc_'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = '!!!not_base64!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'simple'
    var_16 = 'complex string with spaces'
    var_17 = b'\x00\x01\x02'
    var_18 = [var_15, var_16, var_17]
    var_19 = module_0.base64_decode(var_7)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.want_bytes(var_0)
    assert var_1 == b'hello'
    var_2 = b'hello'
    var_3 = module_0.want_bytes(var_2)
    assert var_3 == b'hello'
    var_4 = 'abc'
    var_5 = 'ascii'
    var_6 = module_0.want_bytes(var_4, var_5)
    assert var_6 == b'abc'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 123456789
    var_1 = module_0.int_to_bytes(var_0)
    var_2 = module_0.bytes_to_int(var_1)
    var_3 = 1
    var_4 = 64
    var_5 = var_3 << var_4
    var_6 = var_5 - var_3
    var_7 = module_0.int_to_bytes(var_6)
    var_8 = module_0.bytes_to_int(var_7)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YV9iLWNfZA'
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'YV9iLWNfZA'
    var_7 = b'=='
    var_8 = var_6 + var_7
    var_9 = module_1.urlsafe_b64decode(var_8)
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'Y2Fi'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'cab'
    var_16 = b'dGVzdA'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'
    var_18 = '!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = ''
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b''



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'\x01\x02\x03\xff'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'data\xff\xfe'
    var_7 = b'\xfb\xef'
    var_8 = module_1.urlsafe_b64encode(var_7)
    var_9 = '--8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'\xfb\xef'
    var_11 = 'SGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''



# Parsed testcases at query #9
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
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YWI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'ab'
    var_10 = b'dGVzd'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



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
    var_6 = 'YS1iXw'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a/b+'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!not_base64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8🚀'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'



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
    var_4 = 'Ky8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'+/'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'YQ=='
    var_9 = module_0.base64_decode(var_8)
    var_10 = b'YQ'
    var_11 = module_0.base64_decode(var_10)
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'YWJj\x00'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'Zm9vX2JhcHM'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'foo_bahms'
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
    var_13 = b'Ym9i'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'bob'
    var_15 = 'Ym9i'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'bob'
    var_17 = '!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8😀'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



# Parsed testcases at query #13
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
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YmI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'bb'
    var_10 = 'Y2Nj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ccc'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b'SGVsbG8'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = '!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'invalid_chars_!@#$'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'YV9iLWNfZA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'a_b-c_d'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = b'\x00\xff\x00\xaa\xbb\xcc'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a'
    var_6 = 'v7__'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xfe\xff'
    var_8 = b'V29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'World'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #16
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
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'TWFu'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Man'
    var_12 = 'YWI'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = '!!!NotBase64!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'\xff\xfe\xfd'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'Y29uZmlybV90ZXN0'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'confirm_test'
    var_4 = 'Y29uZmlybS10ZXN0'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'confirm-test'
    var_6 = b'YWJj'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'abc'
    var_8 = 'YWJj'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'abc'
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = '????'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'c3ViamVjdD8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'subject?'
    var_4 = b'Ym9vaQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'booi'
    var_6 = 'Ym9vaQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'booi'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'YW55IGNhcm5hbCBwbGVhc3VyZS4='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'any carnal pleasure.'



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'c3ViamVjdD9xdWVyeT0x'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'subject?query=1'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YmI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'bb'
    var_10 = 'Y2Jj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ccc'
    var_12 = b'dGVzdA'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test'
    var_14 = 'abc'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = ''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''



# Parsed testcases at query #21
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
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'Ym'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'b'
    var_10 = module_0.base64_decode(var_6)
    assert var_10 == b'a'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = '!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'abc-123_XYZ'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    var_18 = module_0.want_bytes(var_15)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'\x00\xff\xfe\xfd'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = 'YWJj'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'abc'
    var_8 = 'YWJjZA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'abcd'
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = '!!!NotBase64!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #23
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
    var_6 = '-_'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xf8'
    var_8 = 'python_is_great-123'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_1.encode()
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #24
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
    var_8 = 'YQ=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '____'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xef'
    var_14 = '----'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xfb\xbf'
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8'
    var_19 = '🚀'
    var_20 = var_18 + var_19
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'\x00\xff\xfe\x01'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'\xff\xef'
    var_7 = module_0.base64_encode(var_6)
    var_8 = len(var_7)
    var_9 = 0
    var_10 = var_8 > var_9
    var_11 = module_0.base64_decode(var_7)
    var_12 = 'YQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = b'YQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = ''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = b''
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b''



# Parsed testcases at query #26
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
    var_6 = 'c3ViamVjdD9wYXJhbT12YWx1ZQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'subject?param=value'
    var_8 = 'YWJjZA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'abcd'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)



# Parsed testcases at query #27
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
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'YQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YmI'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'bb'
    var_16 = 'Y2Ji'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'cbb'



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'ZGF0YS10ZXN0'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'data-test'
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
    var_12 = 'YQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'Yg'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'b'



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
    var_4 = 'YV9iLWNfZA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a_b-c_d'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b'd29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'world'
    var_10 = '!!!not_base64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'testing 123'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = 'utf-8'
    var_7 = module_1.encode(var_6)
    var_8 = b'YWI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'ab'
    var_10 = b'\xff\xef'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'python-is-fun'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'python\xbe\x9a\xc5'
    var_5 = 'YWI'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'ab'
    var_7 = b'YWI'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'ab'
    var_9 = b'\xff\xef\xbe'
    var_10 = b'\xff\xef\xbe'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'\xff\xef\xbe'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = '!!!not_base64!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'abcሴ'
    var_20 = 'abcሴ'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'abc'
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #2
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
    var_8 = 'testing-with_underscores'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_1.decode(var_3)
    var_12 = b'YW4'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'an'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'\x00\xff\xfe\x01'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'-_'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xfb'
    var_8 = 'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



# Parsed testcases at query #4
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
    var_8 = b'\xff\xef'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = b'YW55IGNhcm5hbCBwbGVhc3VyZQ'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'any carnal pleasure'
    var_13 = 'SGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = b'!!!not_base64!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = ''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''
    var_19 = b''
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b''



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
    var_4 = 'c3ViamVjdHM_'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'subjects?'
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
    var_14 = 'YWJjÿ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.decode(var_3)
    var_5 = b'pytest testing'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = 'testing-chars_123'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_1.decode(var_3)
    var_12 = 'SGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    var_14 = module_1.decode(var_3)
    assert var_14 == 'Hello'
    var_15 = b'YmFzZTY0'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'base64'
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = ''
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b''



# Parsed testcases at query #7
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
    var_6 = b'YmFzZTY0'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'base64'
    var_8 = 'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = 'YWI'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
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
    var_2 = '__4='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'\xff\xfe'
    var_4 = b'V29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'World'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'V29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'World'
    var_10 = '!!!Invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'\x00\xff\xfe\x12'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'YWJjZA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'abcd'
    var_8 = 'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'utf-8'
    var_4 = module_1.encode(var_3)
    var_5 = b'\x00\xff\xfe'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    var_8 = b'\xfb\xff'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = b'YWJjZA'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abcd'
    var_13 = '!!!NotBase64!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'YmFzZTY0'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'base64'
    var_6 = 'testing_123-abc'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = 'utf-8'
    var_10 = module_1.encode(var_9)
    var_11 = '!!!NotBase64!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8ÿ'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = '-_++'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'\xfb\xff\xbe'
    var_4 = b'V29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'World'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'py5m'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xa9\xcef'
    var_10 = '!!!Invalid!!!'
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
    var_5 = 'YQ'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'a'
    var_7 = b'SGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = b'\xff\xef'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'c3ViamVjdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'subject'
    var_6 = 'YQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = 'YWI'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'ab'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '-_8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xfb\xff'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8ሴ'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'cHk'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'py'
    var_6 = 'dGVzdF9kYXRhLTEyMw'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test_data-123'
    var_8 = b'Ym90'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'bot'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'Y29uZ29fcmF0'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'congo_rat'
    var_6 = 'Y29uZ28tdmF0'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'congo-vat'
    var_8 = 'SGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hel'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'SGVsbG8\xff'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'YWF_YmI'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'aa/bb'
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
    var_14 = 'Y21i'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'cmi'
    var_16 = b'SGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = '!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'invalid_base64_chars_#$'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'\x00\xff\x00\xff'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'\xff\xfe\xfd\xfc'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = 'SGVsbG8='
    var_10 = 'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'YW4'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'an'
    var_14 = '!!!not_base64!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''



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
    var_4 = 'YWJjX2RlZg'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'abc_def'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'VGVzdA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Test'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8=🔥'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'



# Parsed testcases at query #20
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
    var_6 = b'VGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = '!!!'
    var_12 = module_0.base64_decode(var_11)



# Parsed testcases at query #21
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
    var_6 = b'\xff\xfe\xfd'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = 'SGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = 'YWJj'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = 'YWJjZA'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'abcd'
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = ''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'\x00\xff\xde\xad\xbe\xef'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = b'\xfb\xff'
    var_7 = module_0.base64_encode(var_6)
    var_8 = b'-'
    var_9 = b'_'
    var_10 = [var_8, var_9]
    var_11 = module_0.base64_decode(var_7)
    var_12 = b'YWJj'
    var_13 = b'abc'
    var_14 = module_1.urlsafe_b64encode(var_13)
    var_15 = b'='
    var_16 = '!!!not_base64!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = ''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'python-is-fun'
    var_4 = b'python-is-fun'
    var_5 = module_0.base64_encode(var_4)
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'python-is-fun'
    var_7 = b'abc'
    var_8 = b'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = b'testing_123-abc'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    var_13 = '!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = b''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'V29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'dGVzdC1jYXNlXw'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test-case_'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'python_is_awesome_123'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = module_0.want_bytes(var_14)



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'c3ViamVjdD8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'subject?'
    var_4 = b'Ym90'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'bot'
    var_6 = 'Ym90'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'bot'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!not-base64!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'Pj4'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>'



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
    var_4 = 'c3ViamVjdD9wYXJhbT12YWx1ZQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'subject?param=value'
    var_6 = b'Ym9i'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'bob'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'a'
    var_13 = 100
    var_14 = var_12 * var_13
    var_15 = module_0.base64_decode(var_14)
    var_16 = None
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = module_0.want_bytes(var_0)
    var_4 = b'\x00\xff\xfe\x01'
    var_5 = module_0.base64_encode(var_4)
    var_6 = module_0.base64_decode(var_5)
    var_7 = b'abc/def+'
    var_8 = module_0.base64_encode(var_7)
    var_9 = module_0.base64_decode(var_8)
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = b'SGVsbG8===='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = b'!!!not_base64!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = b''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''



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
    var_4 = 'YQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a'
    var_6 = b'\xff\xef'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = b'V29ybGQ='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'World'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = '!!!'
    var_14 = module_0.base64_decode(var_13)



# Parsed testcases at query #29
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
    var_6 = 'YW55'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'any'
    var_8 = 'YW55YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'anya'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'dGVzdA'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'Y29tZS13b3JsZA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'come-world'
    var_6 = 'YW55'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'any'
    var_8 = 'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'Ynlic3Rlc3Q='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'bytestest'



