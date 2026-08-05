####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b''
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b''
    var_5 = b'a'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = b'ab'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'ab'
    var_11 = b'abc'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abc'
    var_14 = b'\x00\x01\x02'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'\x00\x01\x02'
    var_17 = 'aGVsbG8='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'hello'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'aGVsbG8='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'hello'
    var_23 = b'aGVsbG8'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'hello'



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'Pj4-Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>'
    var_10 = 'Pj4-Pg'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'>>>'
    var_12 = 256
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'AAECAw=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'\x00\x01\x02\x03'
    var_19 = 'AAECAw'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\x00\x01\x02\x03'
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'ABC'
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'x'
    var_26 = 10000
    var_27 = var_25 * var_26
    var_28 = module_0.base64_encode(var_27)
    var_29 = module_0.base64_decode(var_28)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'Pj4-Pz8_'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>??'
    var_10 = 256
    var_11 = range(var_10)
    var_12 = bytes(var_11)
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = '$$$'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = 'test'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = 'aGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'\x00\x01\x02'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x00\x01\x02'
    var_16 = b'a'
    var_17 = 1000
    var_18 = var_16 * var_17
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = module_0.base64_decode(var_9)
    assert var_23 == b'hello'



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '!!!invalid!!!'
    var_9 = module_0.base64_decode(var_8)
    var_10 = 'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'Pz8_Pz8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'?????'
    var_14 = 'SGVsbG8gV29ybGQ=Ã'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello World'



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'VGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Test'
    var_6 = b'VGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '_-xq'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xbe'
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YWI='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = 'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8='
    var_21 = 2
    var_22 = var_20 * var_21
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'Pj4-Pg=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'>>>'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'\xff\xfe'
    var_16 = module_0.base64_decode(var_15)



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'YQ=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = b'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'a-_w'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'k\xef'
    var_18 = b'aGVsbG8=\xff'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = b'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = 'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'aGVsbG8gd29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello world'
    var_10 = b'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = module_0.base64_decode(var_4)
    assert var_14 == b'test'
    var_15 = b'a'
    var_16 = 100
    var_17 = var_15 * var_16
    var_18 = module_1.b64encode(var_17)
    var_19 = b'='



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'YQ=='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'a'
    var_7 = b'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b'YQ'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = b'\xff\xfe\xfd\xfc'
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'\xff\xfe\xfd\xfc'
    var_20 = b'!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'aGVsbG8='
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'\x00\x01\x02'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'aGVsbG8=\x80'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'hello'



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8gd29ybGQ='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello world'
    var_9 = b'\xff\xfe\x00\x01'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xfe\x00\x01'
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'aA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVsbG8'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #12
#--------------------------


import base64 as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.b64encode(var_0)
    var_2 = b'='
    var_3 = b'test'
    var_4 = module_0.b64encode(var_3)
    var_5 = module_1.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'aGVsbG8='
    var_7 = module_1.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = ''
    var_9 = module_1.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '!!!invalid!!!'
    var_11 = module_1.base64_decode(var_10)
    var_12 = 'aGVsbG8'
    var_13 = module_1.base64_decode(var_12)
    var_14 = b'd29ybGQ='
    var_15 = module_1.base64_decode(var_14)
    assert var_15 == b'world'



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = 'world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'world'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'aGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'aGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'-_0'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\xfb\xff'
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'hello\x80'
    var_24 = 'latin-1'
    var_25 = module_1.encode(var_24)
    var_26 = module_0.base64_decode(var_25)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8gd29ybGQ'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'YQ=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = 'YQ'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'a'
    var_13 = 'YWI='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'ab'
    var_15 = 'YWI'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'ab'
    var_17 = 'YWJj'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'abc'
    var_19 = b'\x00\x01\x02'
    var_20 = b'\xff\xfe\xfd'
    var_21 = b'test data with spaces and !@#$%^&*()'
    var_22 = 256
    var_23 = range(var_22)
    var_24 = bytes(var_23)
    var_25 = [var_19, var_20, var_21, var_24]
    var_26 = module_0.base64_decode(var_1)
    var_27 = '!!!invalid!!!'
    var_28 = module_0.base64_decode(var_27)
    var_29 = b'\xff\xff'
    var_30 = module_0.base64_decode(var_29)
    var_31 = 'invalid base64!@#$'
    var_32 = module_0.base64_decode(var_31)



# Parsed testcases at query #15
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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'aGVsbG8_d29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello?world'
    var_10 = 'YSBi'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a b'
    var_12 = 'MTIzNDU2Nzg5MA=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'1234567890'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVs'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello world'
    var_6 = 'héllo wörld'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = 'utf-8'
    var_10 = module_1.encode(var_9)
    var_11 = b''
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'\x00'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'\x00'
    var_17 = 256
    var_18 = range(var_17)
    var_19 = bytes(var_18)
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'test'
    var_23 = module_0.base64_encode(var_22)
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'test'
    var_25 = '!!!invalid!!!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'bytes input'
    var_28 = module_0.base64_encode(var_27)
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'bytes input'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\xfb\xff\xff\xff\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'aGVsbG8=\x80'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'hello'
    var_23 = b'aA=='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'h'
    var_25 = b'aA'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'h'



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b''
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b''
    var_5 = b'test'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = 'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'hello+world'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello+world'
    var_15 = b'x'
    var_16 = module_0.base64_decode(var_1)
    var_17 = b'!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'aGVsbG8=\x80'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'hello'



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'VGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = '_-xq'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xfb\xc6\xa8'
    var_10 = 'dGVzdC11cmw='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test-url'
    var_12 = b'Hello, World! This is a test.'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'Hello World'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\xff\xfb\x00\x01'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'8J+YgQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xf0\x9f\x98\x81'
    var_18 = b'x'
    var_19 = 1000
    var_20 = var_18 * var_19
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'YQ'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'a'
    var_25 = b'YWI'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'ab'
    var_27 = b'YWJj'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'abc'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'WA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'X'
    var_8 = 'dGVzdC1f'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-_'
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'YQ=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = 'YQ'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = 'YWI='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'ab'
    var_20 = 'YWI'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'ab'
    var_22 = 'YWJj'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'abc'



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = 'YWJjZGVmZw=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'abcdefg'
    var_10 = 'MTIzNDU2'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'123456'
    var_12 = 'YSBiIGM='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a b c'
    var_14 = 'aGVsbG8'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'w6llbMOz'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'ëeló'
    var_21 = 'utf-8'
    var_22 = module_1.encode(var_21)



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'VGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Test'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'VGVzdA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Test'
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = '_-w='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xec'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'SGVsbG8=\xff'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8gd29ybGQ'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = b'YQ=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = b'YWI='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'ab'
    var_11 = b'YWJj'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = b'_-_w'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\xff\xef\xc0'
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'aGVsbG8='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'hello'
    var_19 = bytes(var_15)
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'YQ=='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'a'
    var_7 = b'YWI='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'ab'
    var_9 = b'YWJj'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'abc'
    var_11 = b'd29ybGQ='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'world'
    var_13 = 'd29ybGQ='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'world'
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = b'aGVsbG8gd29ybGQh'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'hello world!'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'\xff\xfe\x00'
    var_22 = module_0.base64_decode(var_21)



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test string'
    var_6 = 'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = 'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = b'\x00\x01\x02'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'\x00\x01\x02'
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'aGVs\x00bG8='
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'Pj4-Pg=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'>>>'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'Pj4_Pg=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'>>\xfe>'
    var_16 = module_0.base64_decode(var_10)
    assert var_16 == b'>>\xfb>'



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = 'aGVsbG8td29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello-world'
    var_10 = 'YSBi'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a b'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'SGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = 'dGVzdC5wYXRoL3NvbWV0aGluZw=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test.path/something'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8$'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'MTIzNDU2Nzg5MA=='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'1234567890'
    var_24 = 256
    var_25 = range(var_24)
    var_26 = bytes(var_25)
    var_27 = module_0.base64_encode(var_26)
    var_28 = module_0.base64_decode(var_27)



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'd29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'world'
    var_7 = b'test'
    var_8 = module_0.base64_encode(var_7)
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'Zg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'f'
    var_14 = b'\xff\xfe'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'\xff\xfe'
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'Test base64_decode function with various inputs.'
    var_1 = 'SGVsbG8='
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'Hello'
    var_3 = 'SGVsbG8'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'Hello'
    var_5 = b'V29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'World'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 256
    var_10 = range(var_9)
    var_11 = bytes(var_10)
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = '_-x'
    var_15 = module_0.base64_decode(var_14)
    var_16 = len(var_15)
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'MTIz'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'123'
    var_21 = 'w6TDtsO8'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'äöü'
    var_24 = 'utf-8'
    var_25 = module_1.encode(var_24)



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello world'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'aGVsbG8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'aGVsbG8=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello'
    var_15 = 256
    var_16 = range(var_15)
    var_17 = bytes(var_16)
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'a'
    var_23 = module_0.base64_encode(var_22)
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'a'
    var_25 = b'\x00\x01\x02\xff\xfe\xfd'
    var_26 = module_0.base64_encode(var_25)
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'héllo'
    var_29 = 'utf-8'
    var_30 = module_1.encode(var_29)
    var_31 = module_0.base64_encode(var_30)
    var_32 = module_0.base64_decode(var_31)
    var_33 = b'\xff\xfe\xfd'
    var_34 = module_0.base64_decode(var_33)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b''
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'test data with spaces and !@#$%^&*()'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = 'héllo wörld'
    var_10 = 'utf-8'
    var_11 = module_1.encode(var_10)
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = '!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'not valid base64!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'test'
    var_24 = module_0.base64_encode(var_23)
    var_25 = 'ascii'
    var_26 = module_1.decode(var_25)
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'test'
    var_28 = b'short'
    var_29 = module_0.base64_encode(var_28)
    var_30 = module_0.base64_decode(var_29)
    var_31 = b'a'
    var_32 = 1000
    var_33 = var_31 * var_32
    var_34 = module_0.base64_encode(var_33)
    var_35 = module_0.base64_decode(var_34)



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'test?data=123'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test?data=123'
    var_16 = b'a'
    var_17 = b'ab'
    var_18 = b'abc'
    var_19 = b'test data with spaces'
    var_20 = b'binary\x00data'
    var_21 = b'unicode text'
    var_22 = [var_7, var_16, var_17, var_18, var_19, var_20, var_21]
    var_23 = module_0.base64_decode(var_11)



# Parsed testcases at query #34
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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'YQ=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = 'YWI='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = 'YWJj'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abc'
    var_14 = '_-w='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xff\xec'
    var_16 = 'AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8='
    var_17 = module_0.base64_decode(var_16)
    var_18 = 32
    var_19 = range(var_18)
    var_20 = bytes(var_19)
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = module_0.base64_decode(var_21)
    assert var_23 == b'Hello'



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'test data with bytes'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'test!@#$%^&*()_+-=[]{}|;\':",./<>?`~'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = b'!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'dGVzdA'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'test'
    var_15 = 'Hello, 世界'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    var_18 = module_1.encode()



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Test base64_decode function.'
    var_1 = b'hello world'
    var_2 = module_0.base64_encode(var_1)
    var_3 = module_0.base64_decode(var_2)
    var_4 = 'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'aGVsbG8gd29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello world'
    var_10 = 'aGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'aGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'\xff\xfe\xfd\xfc'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'\xff\xfe\xfd\xfc'



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'Pjw_Pg=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'><>?'
    var_8 = b'VGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Test'
    var_10 = 'dGVzdA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = 'SGVsbG8gV29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello World'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'Hello World'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b'aGVsbG8td29ybGQ='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello-world'
    var_11 = b''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'\x00\x01\x02\xff\xfe'
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'MTIzNDU2Nzg5MA=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'1234567890'



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'dGVzdGluZy11cmwtc2FmZQ=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'testing-url-safe'
    var_8 = b'd29ya3M='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'works'
    var_10 = 'dGVzdC13aXRoLXNwZWNpYWwtY2hhcnM/'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test-with-special-chars?'
    var_12 = 'dGVzdF91bmRlcnNjb3Jl'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test_underscore'
    var_14 = 'dGVzdC1kYXNo'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test-dash'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'dGVzdA=='
    var_19 = '\x80\x81'
    var_20 = var_18 + var_19
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'test'



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'data with + and /'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'a'
    var_15 = b'ab'
    var_16 = b'abc'
    var_17 = b'abcd'
    var_18 = b'test@#$%^&*()'
    var_19 = b'\x00\x01\x02\xff\xfe'
    var_20 = b'data with spaces and\nnewlines'
    var_21 = [var_6, var_14, var_15, var_16, var_17, var_18, var_19, var_20]
    var_22 = module_0.base64_decode(var_12)
    var_23 = '!!!invalid!!!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'aGVsbG8='
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'bytes input'
    var_28 = module_0.base64_encode(var_27)
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'bytes input'



# Parsed testcases at query #41
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = b'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'aGVsbG8=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'Pj4-Pg=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'>>>'
    var_15 = b'PDw8PA=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'<<<<'
    var_17 = 256
    var_18 = range(var_17)
    var_19 = bytes(var_18)
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'AAECAw=='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'\x00\x01\x02\x03'
    var_24 = b'!!!invalid!!!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = b'\xff\xfe\xfd'
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b'Pj4-Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>> >'
    var_10 = 'Pj4-Pg=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'>> >'
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b'YQ'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = b'YWI'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'ab'
    var_20 = b'Lg=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'.'
    var_22 = b'Lg'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'.'
    var_24 = b'!!!invalid!!!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = '!!!invalid!!!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = b'\xff\xfe\xfd'
    var_29 = module_0.base64_decode(var_28)
    var_30 = b'\xffSGVsbG8='
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'
    var_32 = 'SGVs\x80bG8='
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'Hello'



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b'\xff\xfe\x00\x01'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'not valid base64'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'a'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = b'hello\nworld'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello\nworld'
    var_15 = b'\x00\x01\x02\xff'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\x00\x01\x02\xff'
    var_18 = b'test'
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'test'
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'not base64'
    var_24 = module_0.base64_decode(var_23)
    var_25 = -1
    var_26 = base64_encode(var_18)[:var_25]
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #45
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'aGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = 'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'dGVzdA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = '_-x'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xeb'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'abc'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = '_-w'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xec'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'abc'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'dGVzdC11cmwtdG9rZW4'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-url-token'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'not valid base64'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'dGVzdA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test'



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8gd29ybGQ'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'YQ'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'SGVsbG8='
    var_14 = b'Hello'
    var_15 = (var_13, var_14)
    var_16 = b'V29ybGQ='
    var_17 = b'World'
    var_18 = (var_16, var_17)
    var_19 = b'MTIzNDU2'
    var_20 = b'123456'
    var_21 = (var_19, var_20)
    var_22 = [var_15, var_18, var_21]
    var_23 = module_0.base64_decode(var_1)
    var_24 = 256
    var_25 = range(var_24)
    var_26 = bytes(var_25)
    var_27 = module_0.base64_encode(var_26)
    var_28 = module_0.base64_decode(var_27)
    var_29 = b'!!!invalid!!!'
    var_30 = module_0.base64_decode(var_29)
    var_31 = b'aGVsbG8$'
    var_32 = module_0.base64_decode(var_31)



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = module_1.decode()
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b''
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'a'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'='
    var_11 = module_0.base64_decode(var_9)
    assert var_11 == b'a'
    var_12 = b'!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'QQ=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'A'
    var_10 = 'Pj4-Pg=='
    var_11 = module_0.base64_decode(var_10)



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello world'
    var_6 = b'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'test\x00\xff'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'test\x00\xff'
    var_17 = bytes(var_12)
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #52
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'dGVzdC11cmw'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-url'
    var_10 = b'dGVzdF8t'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test_-'
    var_12 = 'YWJjMTIz'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abc123'
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'w7zDvMO8'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xc3\xbc\xc3\xbc\xc3\xbc'



# Parsed testcases at query #53
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b'a'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'\x00\xff\x7f'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\x00\xff\x7f'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'\xff\xff\xff\xff'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #54
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'aGVsbG8\xff'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'a'
    var_16 = b'ab'
    var_17 = b'abc'
    var_18 = b'test data here'
    var_19 = 256
    var_20 = range(var_19)
    var_21 = bytes(var_20)
    var_22 = [var_7, var_15, var_16, var_17, var_18, var_21]
    var_23 = module_0.base64_decode(var_1)
    var_24 = b'aGVs_bG8='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'hel\xbb\x18'
    var_26 = b'aGVsbG8-'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'hell\xf8'



# Parsed testcases at query #55
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
    var_6 = 'aGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello world'
    var_8 = 'Zm9vLmJhcg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'foo.bar'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = module_0.base64_decode(var_12)
    assert var_14 == b'Hello'
    var_15 = 'a'
    var_16 = 1000
    var_17 = var_15 * var_16
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    var_20 = module_1.encode()



# Parsed testcases at query #56
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = b'a'
    var_8 = module_0.base64_encode(var_7)
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = 256
    var_11 = range(var_10)
    var_12 = bytes(var_11)
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'aGVsbG8$'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #57
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'VGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = 'aGVsbG8td29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello-world'
    var_10 = 'VGhpcyBpcyBhIHRlc3Q='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'This is a test'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'ABC123!@#'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'a'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'héllo'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #58
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\x00\x01\x02'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\x00\x01\x02'
    var_14 = b'test'
    var_15 = 100
    var_16 = var_14 * var_15
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'aGVsbG8\n'
    var_22 = module_0.base64_decode(var_21)
    var_23 = None
    var_24 = module_0.base64_decode(var_23)



# Parsed testcases at query #59
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'aGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'world'
    var_8 = 'dGVzdC11cmw'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-url'
    var_10 = 'ISQlJio='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'!$%&*'
    var_12 = 'invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'dGVzdA'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test'
    var_16 = 'YQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = 'aGVsbG8=ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'



# Parsed testcases at query #60
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = '_-xq'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xf1\xaa'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8=\x80'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'



# Parsed testcases at query #61
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'Pj4_Pz8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'>>???'
    var_12 = 'Pj4_Pz8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>???'
    var_14 = module_0.base64_decode(var_8)
    assert var_14 == b''
    var_15 = module_0.base64_decode(var_8)
    assert var_15 == b''
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'\xff\xff\xff'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'hello'
    var_21 = 'world'
    var_22 = 'test123'
    var_23 = 'a'
    var_24 = 100
    var_25 = var_23 * var_24
    var_26 = 'data:image/png;base64,'
    var_27 = [var_20, var_21, var_22, var_25, var_26]
    var_28 = module_1.encode()
    var_29 = b'hello'
    var_30 = b'world'
    var_31 = b'test123'
    var_32 = b'\x00\x01\x02\xff'
    var_33 = [var_29, var_30, var_31, var_32]
    var_34 = 'YQ=='
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'a'
    var_36 = 'YWE='
    var_37 = module_0.base64_decode(var_36)
    assert var_37 == b'aa'
    var_38 = 'YWFh'
    var_39 = module_0.base64_decode(var_38)
    assert var_39 == b'aaa'
    var_40 = 'YWFhYQ=='
    var_41 = module_0.base64_decode(var_40)
    assert var_41 == b'aaaa'



# Parsed testcases at query #62
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = '_-x'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xeb'
    var_10 = '=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'ABC@123'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #63
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'Zg=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'f'
    var_13 = b'_-A='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\xfb\xe0'
    var_15 = '!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'aGVsbG8='
    var_18 = 255
    var_19 = chr(var_18)
    var_20 = var_17 + var_19
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'aGVsbG8$'
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #64
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = 'SGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'_-x'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xfb\xe7'
    var_14 = b'\x00\x01\x02\xff\xfe\xfd'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'\xff\xfe\xfd'
    var_20 = module_0.base64_decode(var_19)
    var_21 = module_0.base64_decode(var_10)
    assert var_21 == b''



# Parsed testcases at query #65
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b'aGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b'test data 123'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test data 123'
    var_16 = b'\x00\x01\x02\xff'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'\x00\x01\x02\xff'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'\xff\xff\xff'
    var_22 = module_0.base64_decode(var_21)



# Parsed testcases at query #66
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8gd29ybGQ='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b'aGVsbG8gd29ybGQ'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = 'aGVsbG8gd29ybGQ='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello world'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'a'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = b'test with spaces!@#$%^&*()'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'\x00\x01\x02\xff'
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #67
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = 'aGVsbG8_d29ybGQ-'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello?world>'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'WA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'X'
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'AAAA'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x00\x00\x00'



# Parsed testcases at query #68
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test string'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = 'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'aGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = 'Pj4-Pj4_'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'>>>>>>?'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'aGVsbG8!='
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #69
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'aGVs'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hel'
    var_10 = b''
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b'\x00\x01\x02'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x00\x01\x02'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVsbG8='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'



# Parsed testcases at query #70
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'test+data=='
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = 'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = b'\x00\x01\x02\xff'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'héllo'
    var_16 = 'utf-8'
    var_17 = module_1.encode(var_16)
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'aGVsbG8'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'hello'
    var_22 = b'aGVsbG8='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'hello'
    var_24 = '!!!invalid!!!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'aGVsbG8!!!'
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #71
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'Test123!@#'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Test123!@#'
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'héllo'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'dGVzdA=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'test'
    var_20 = 256
    var_21 = range(var_20)
    var_22 = bytes(var_21)
    var_23 = module_0.base64_encode(var_22)
    var_24 = module_0.base64_decode(var_23)



# Parsed testcases at query #72
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = 'aGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'd29ybGQ='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'world'
    var_16 = 'd29ybGQ='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'world'



# Parsed testcases at query #73
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'aGVsbG8_d29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello?world'
    var_14 = 'aGVsbG8td29ybGQ='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello-world'
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8='
    var_24 = 'invalid'
    var_25 = var_23 + var_24
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8=\x80'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'



# Parsed testcases at query #74
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8gd29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'test_data+with/special?chars'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = b'\xff\xfe\x00\x01'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #75
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'dGVzdC13aXRoLXVybC1zYWZlLWNoYXJz'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test-with-url-safe-chars'
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = 'dGVzdC13aXRoLXNwZWNpYWwtY2hhcnM_'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test-with-special-chars?'
    var_12 = module_0.base64_decode(var_2)
    assert var_12 == b'test'
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'ñ'
    var_16 = var_13 + var_15
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'
    var_18 = 'MTIzNDU2Nzg5MA=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'1234567890'



# Parsed testcases at query #76
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'test?data=1&more=2'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test?data=1&more=2'
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'aGVs@bG8='
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'héllo'
    var_26 = module_0.base64_decode(var_25)



# Parsed testcases at query #77
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = module_0.base64_decode(var_7)
    assert var_9 == b''
    var_10 = 'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = b'YWI='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = b'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'aGVs'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #78
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'dGVzdC11cmw='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test-url'
    var_6 = 'dGVzdC11cmw'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test-url'
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = b'dGVzdA'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'YSBi'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a b'
    var_18 = module_0.base64_decode(var_16)
    assert var_18 == b'a b'
    var_19 = 256
    var_20 = range(var_19)
    var_21 = bytes(var_20)
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'YQ=='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'a'
    var_26 = 'YWI='
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'ab'
    var_28 = 'YWJj'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'abc'
    var_30 = module_0.base64_decode(var_24)
    assert var_30 == b'a'
    var_31 = module_0.base64_decode(var_26)
    assert var_31 == b'ab'
    var_32 = module_0.base64_decode(var_28)
    assert var_32 == b'abc'
    var_33 = 'U8Oww7zDtg=='
    var_34 = module_0.base64_decode(var_33)
    assert var_34 == b'S\xc3\xb0\xc3\xbc\xc3\xb6'
    var_35 = '!!!invalid!!!'
    var_36 = module_0.base64_decode(var_35)
    var_37 = b'!!!invalid!!!'
    var_38 = module_0.base64_decode(var_37)
    var_39 = 'test'
    var_40 = module_0.base64_decode(var_39)
    assert var_40 == b'\xb5\xeb'
    var_41 = None
    var_42 = module_0.base64_decode(var_41)



# Parsed testcases at query #79
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'aGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'world'
    var_8 = 'Pj4-Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>'
    var_10 = '_-5-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xee\xe7'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'aGVsbG8$'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'dGVzdA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'



# Parsed testcases at query #80
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = 'aGVsbG8'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'YQ'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'a'
    var_7 = 'YWI'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'ab'
    var_9 = 'YWJj'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'abc'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b'dGVzdA'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'test'
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'aGVsbG8$'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #81
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test string'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'\x00\x01\x02\xff'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'\x00\x01\x02\xff'
    var_11 = b'aGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'aGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello'
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'héllo'
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'utf-8'
    var_21 = module_1.encode(var_20)
    var_22 = b'aGVsbG8='
    var_23 = bytearray(var_22)
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'hello'



# Parsed testcases at query #82
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'aGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = 'dGVzdF91cmw'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test_url'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'd29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'world'
    var_10 = 'dGVzdC0t'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test--'
    var_12 = 'dGVzdF8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test_'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = module_0.base64_decode(var_14)
    assert var_16 == b'hello'



# Parsed testcases at query #83
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'd29ybGQ='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'world'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'\xfb\xff\xff\xff\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'\xff\xff\xff'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #84
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'YQ'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = 'YWI'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'ab'
    var_13 = 'YWJj'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'abc'
    var_15 = b'_-w'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'\xff\xef'
    var_17 = 'é'
    var_18 = var_5 + var_17
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'
    var_20 = 'aGVsbG8=\n\t'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'hello'
    var_22 = '!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'\xff\xff'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'YQ=='
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'a'
    var_28 = 'YWI='
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'ab'
    var_30 = bytes(var_24)
    var_31 = module_0.base64_encode(var_30)
    var_32 = module_0.base64_decode(var_31)



# Parsed testcases at query #85
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = b'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = 256
    var_10 = range(var_9)
    var_11 = bytes(var_10)
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'dGVzdF91cmw'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'test_url'



# Parsed testcases at query #86
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'd29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'world'
    var_4 = 'aGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = 'd29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = 'Pj4-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>>'
    var_14 = 'PDw8PA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<<<<'
    var_16 = '_-8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xff\xef'
    var_18 = '!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'aGVs bG8='
    var_21 = module_0.base64_decode(var_20)
    var_22 = 200
    var_23 = chr(var_22)
    var_24 = var_20 + var_23
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'hello'



# Parsed testcases at query #87
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\x00\x01\x02'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\x00\x01\x02'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'aGVsbG8=\x80\x81'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = module_0.base64_decode(var_3)
    assert var_18 == b'hello'



# Parsed testcases at query #88
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = 'aGVsbG8gd29ybGQ'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'YQ'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = b'YWI'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'ab'
    var_13 = b'YWJj'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'abc'
    var_15 = b'YQ=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'a'
    var_17 = b'YWI='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'ab'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'_-w'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'\xff\xeb'
    var_23 = b'1234567890'
    var_24 = module_0.base64_encode(var_23)
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'1234567890'
    var_26 = 256
    var_27 = range(var_26)
    var_28 = bytes(var_27)
    var_29 = module_0.base64_encode(var_28)
    var_30 = module_0.base64_decode(var_29)
    var_31 = 'aGVsbG8gd29ybGQ='
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'hello world'



# Parsed testcases at query #89
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b''
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b''
    var_5 = b'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b'YQ==.A'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a?\x00'
    var_11 = b'test data with numbers 123'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = 'aGVsbG8'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = module_0.base64_decode(var_5)
    assert var_18 == b'hello'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'invalid'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'x'
    var_24 = 1000
    var_25 = var_23 * var_24
    var_26 = module_0.base64_encode(var_25)
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #90
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = 'aGVsbG8gd29ybGQ'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello world'
    var_11 = 'Pz8_Pz8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'???\x00\x00'
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'aGVs bG8='
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'\xff\xfe\xfd'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'x'
    var_20 = module_0.base64_decode(var_18)
    var_21 = b'aGVsbG8='
    var_22 = module_0.base64_decode(var_21)



# Parsed testcases at query #91
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'\x00\x01\x02\xff'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = b'test?data=123'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test?data=123'
    var_12 = b'SGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = b'SGVsbG8'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'\xff\xff\xff\xff'
    var_19 = module_0.base64_decode(var_18)
    var_20 = '_-A='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\xfb\xff'



# Parsed testcases at query #92
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = '_-xK'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\x12'
    var_10 = 'PDw_Pz8-Pg=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<<??>>'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'a'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8='
    var_17 = '\x80\x81'
    var_18 = var_16 + var_17
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #93
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'aA=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'h'
    var_9 = 'aA'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'h'
    var_11 = b'dGVzdA=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'test'
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'aGVsbG8='
    var_16 = 128
    var_17 = chr(var_16)
    var_18 = var_15 + var_17
    var_19 = module_0.base64_decode(var_18)
    var_20 = b''
    var_21 = b'a'
    var_22 = b'ab'
    var_23 = b'abc'
    var_24 = b'test data with spaces'
    var_25 = 256
    var_26 = range(var_25)
    var_27 = bytes(var_26)
    var_28 = [var_20, var_21, var_22, var_23, var_24, var_27]
    var_29 = module_0.base64_decode(var_1)



# Parsed testcases at query #94
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '_-x'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xfb\xe7'
    var_10 = 'aGVsbG8gd29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello world'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8=\x80\x81'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'



# Parsed testcases at query #95
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'YQ=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = 'YWI='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'ab'
    var_11 = 'YWJj'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = b'hello+world/foo'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello+world/foo'
    var_16 = b'aGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'aGVsbG8$'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'aGVsbG8\xff'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'hello'



# Parsed testcases at query #96
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'YQ=='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'a'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = 'héllo wörld'
    var_10 = module_1.encode()
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b'YQ'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'a'
    var_17 = module_0.base64_decode(var_5)
    assert var_17 == b'a'
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 256
    var_21 = range(var_20)
    var_22 = bytes(var_21)
    var_23 = module_0.base64_encode(var_22)
    var_24 = module_0.base64_decode(var_23)



# Parsed testcases at query #97
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'\x00\x01\x02\xff'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = b'SGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = '!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = 'SGVs bG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'



# Parsed testcases at query #98
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = b'test data with + and /'
    var_8 = module_0.base64_encode(var_7)
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test data with + and /'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVsbG8='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'



# Parsed testcases at query #99
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = b'SGVsbG8gV29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = 'SGVsbG8gV29ybGQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '_-x'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xeb'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'SGVsbG8gV29ybGQ=\xff'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello World'



# Parsed testcases at query #100
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8gd29ybGQ'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = 'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8\x80'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = '!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'dGVzdA=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'test'



# Parsed testcases at query #101
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'aGVsbG8tX3dvcmxk'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello_world'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'dGVzdD8+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test?>'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'test\x00data'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #102
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'dGVzdC11cmw'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test-url'
    var_6 = 256
    var_7 = range(var_6)
    var_8 = bytes(var_7)
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = 'YQ=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'a'
    var_15 = b'SGVsbG8='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'abc$def'
    var_20 = module_0.base64_decode(var_19)
    var_21 = module_0.base64_decode(var_19)
    assert var_21 == b'Hello'



# Parsed testcases at query #103
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'aGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = 'PDw_Pz8-Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<<??>>'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YWI'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = 'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'aGVsbG8=\x80\x81'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'hello'
    var_22 = '-__-'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'\xfb\xbf\xbe'



# Parsed testcases at query #104
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'YQ=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = 'YWI='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'ab'
    var_11 = 'YWJj'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = '_-w='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\xfe\xec'
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'abcde!@#$%'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #105
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'Hello, World!'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'Hello, World!'
    var_3 = 'SGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'Hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = 'aGVsbG8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'test data with +/'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test data with +/'
    var_14 = b'SGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8=ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = b'\x00\x01\x02'
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'\x00\x01\x02'



# Parsed testcases at query #106
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'\x00\x01\x02'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\x00\x01\x02'
    var_12 = b'\xff\xfe\xfd'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\xff\xfe\xfd'
    var_15 = 'aGVsbG8'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'hello'
    var_17 = 'aGVsbG8='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'hello'
    var_19 = 'aGVsbG8=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'hello'
    var_21 = '_-'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'\xfb\xff'
    var_23 = ''
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b''
    var_25 = '!!!invalid!!!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'\xff\xff\xff\xff'
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'not base64'
    var_30 = module_0.base64_decode(var_29)



# Parsed testcases at query #107
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '_-x4'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\xff\x1e'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'AAECAwQFBgcICQoLDA0ODw=='
    var_11 = module_0.base64_decode(var_10)
    var_12 = 16
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'Hello World'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'é'
    var_20 = var_17 + var_19
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #108
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8gd29ybGQ='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b'aGVsbG8gd29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = b'test data with +/='
    var_8 = module_0.base64_encode(var_7)
    var_9 = module_0.base64_decode(var_8)
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'aGVsbG8gd29ybGQ=\x80\x81'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'hello world'
    var_22 = 'aGVsbG8'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'hello'
    var_24 = module_0.base64_decode(var_18)
    assert var_24 == b'hello world'
    var_25 = b''
    var_26 = b'a'
    var_27 = b'ab'
    var_28 = b'abc'
    var_29 = b'test'
    var_30 = b'\x00\x01\x02\xff'
    var_31 = b'binary\x00data'
    var_32 = [var_25, var_26, var_27, var_28, var_29, var_30, var_31]
    var_33 = module_0.base64_decode(var_8)



# Parsed testcases at query #109
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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dGVzdC11cmw'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-url'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'hello world'
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #110
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = 'world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'world'
    var_6 = ''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'test'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = b'longer string'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'longer string'
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 256
    var_18 = range(var_17)
    var_19 = bytes(var_18)
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'x'
    var_23 = module_0.base64_decode(var_16)
    var_24 = 'héllo wörld'
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'utf-8'
    var_28 = module_1.encode(var_27)



# Parsed testcases at query #111
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'\x00\x01\x02\xff'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\x00\x01\x02\xff'
    var_12 = 'aGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'aGVsbG8==='
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #112
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = 'utf-8'
    var_7 = module_1.encode(var_6)
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = b'dGVzdA'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'\xfb\xff\xff\xff\xff'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #113
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = '_-x.'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xed\xc7'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'dGVzdA==\x80'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test'



# Parsed testcases at query #114
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'ZA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'd'
    var_10 = 'dGVzdA_-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'hello world'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'Hello, World! This is a longer test string.'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = 256
    var_20 = range(var_19)
    var_21 = bytes(var_20)
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #115
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b''
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b''
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'test data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test data'
    var_10 = b'a'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'a'
    var_13 = b'x'
    var_14 = 100
    var_15 = var_13 * var_14
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'\x00\x01\x02\xff\xfe'
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'héllo'
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'utf-8'
    var_25 = module_1.encode(var_24)
    var_26 = b'\x00'
    var_27 = 10
    var_28 = var_26 * var_27
    var_29 = module_0.base64_encode(var_28)
    var_30 = module_0.base64_decode(var_29)
    var_31 = b'test'
    var_32 = module_0.base64_encode(var_31)
    var_33 = 'ascii'
    var_34 = module_1.decode(var_33)
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'test'
    var_36 = module_0.base64_encode(var_31)
    var_37 = b'='
    var_38 = 4
    var_39 = len(var_36)
    var_40 = var_39 % var_38
    var_41 = var_38 - var_40
    var_42 = var_37 * var_41
    var_43 = var_36 + var_42
    var_44 = module_0.base64_decode(var_43)
    assert var_44 == b'test'
    var_45 = b'!!!invalid!!!'
    var_46 = module_0.base64_decode(var_45)
    var_47 = b'abc$%^'
    var_48 = module_0.base64_decode(var_47)
    var_49 = bytes(var_47)
    var_50 = module_0.base64_encode(var_49)
    var_51 = module_0.base64_decode(var_50)



# Parsed testcases at query #116
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'test'
    var_4 = module_1.b64encode(var_3)
    var_5 = 'ascii'
    var_6 = module_1.decode(var_5)
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'a'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'a'
    var_13 = b'\x00\x01\x02'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x00\x01\x02'
    var_16 = 'aGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'aGVsbG8$'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #117
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'test data'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'test data'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello world'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'a'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 256
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'test'
    var_20 = module_0.base64_encode(var_19)
    var_21 = b'='
    var_22 = module_0.base64_decode(var_15)
    assert var_22 == b'test'



# Parsed testcases at query #118
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\x00\x01\x02\xff\xfe'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'\xfb\xff\xff\xff\xff'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)



# Parsed testcases at query #119
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'\xfb\xff\xff\xff\xff'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'aGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'aGVs'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'_-A='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'\xfb\xff'



# Parsed testcases at query #120
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'aGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'world'
    var_4 = b'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'dXNlcm5hbWU'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'username'
    var_14 = 'cGFzc3dvcmQxMjM'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'password123'
    var_16 = 'Pz4_Pz4'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'>?>?>'
    var_18 = b'Ynl0ZXM'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'bytes'
    var_20 = module_0.base64_decode(var_6)
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'\xff\xfe\xfd'
    var_24 = module_0.base64_decode(var_23)
    var_25 = '123'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'w6TDtsO8'
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'äöü'
    var_30 = 'utf-8'
    var_31 = module_1.encode(var_30)
    var_32 = b'test data with spaces and symbols!@#$%^&*()'
    var_33 = module_0.base64_encode(var_32)
    var_34 = module_0.base64_decode(var_33)



# Parsed testcases at query #121
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = 'dGVzdC11cmwtc2FmZQ=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test-url-safe'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'dGVzdC1ieXRlcw=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-bytes'
    var_10 = 'dGVzdC13aXRoICQjQCE='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test-with $#@!'
    var_12 = 'invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = '=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x00'



# Parsed testcases at query #122
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'test'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'\x00\xff\xfe\xfd'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'\x00\xff\xfe\xfd'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'aGVsbG8=ÿ'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'hello'
    var_17 = b'aGVsbG8'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'hello'



# Parsed testcases at query #123
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'a'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = b'ab'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'ab'
    var_19 = b'abc'
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'abc'
    var_22 = b'abcd'
    var_23 = module_0.base64_encode(var_22)
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'abcd'
    var_25 = b'\x00\xff\xfe\xfd'
    var_26 = module_0.base64_encode(var_25)
    var_27 = module_0.base64_decode(var_26)
    var_28 = b'!!!invalid!!!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = b'\x00\x00\x00'
    var_31 = module_0.base64_decode(var_30)



# Parsed testcases at query #124
#--------------------------


import base64 as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.b64encode(var_0)
    var_2 = b'='
    var_3 = b'aGVsbG8='
    var_4 = module_1.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b''
    var_6 = module_1.base64_decode(var_5)
    assert var_6 == b''
    var_7 = b'dGVzdC11cmwtdmFsdWU'
    var_8 = module_1.base64_decode(var_7)
    assert var_8 == b'test-url-value'
    var_9 = b'dGVzdC11cmwtdmFsdWU_'
    var_10 = module_1.base64_decode(var_9)
    assert var_10 == b'test-url-value_'
    var_11 = 'aGVsbG8='
    var_12 = module_1.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'!!!invalid!!!'
    var_14 = module_1.base64_decode(var_13)
    var_15 = b'this is not base64'
    var_16 = module_1.base64_decode(var_15)



# Parsed testcases at query #125
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YWI='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = 'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = 'Pj4-Pg=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'>>?>'
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'\xff\xfe\xfd'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=ÿ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'



# Parsed testcases at query #126
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
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'VGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = 'a-_w'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'k\xef\xc0'
    var_10 = 'd29yaw=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'work'
    var_12 = 'w6TDtsOc'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'äöü'
    var_15 = module_1.encode()
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'ab cd'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #127
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'YQ=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = 'YWI='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'ab'
    var_11 = 'YWJj'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = 'aGVsbG8gd29ybGQ='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello world'
    var_15 = '!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'dGVzdA=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'test'
    var_19 = module_0.base64_decode(var_3)
    assert var_19 == b'hello'
    var_20 = bytes(var_15)
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)



# Parsed testcases at query #128
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'dGVzdD4_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test>?'
    var_14 = 'dGVzdD4_'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test>?'
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'\xff\xff\xff'
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'a'
    var_26 = 1000
    var_27 = var_25 * var_26
    var_28 = module_0.base64_encode(var_27)
    var_29 = module_0.base64_decode(var_28)



# Parsed testcases at query #129
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'\x00\xff\xfe\xfd'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #130
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'dGVzdF91cmw'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test_url'
    var_6 = 'dGVzdC11cmw'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test-url'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'ZA=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'd'
    var_14 = 'Pj4+Pg=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'>>>>'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = '123'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #131
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test string'
    var_6 = b'a'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = b'x'
    var_10 = 100
    var_11 = var_9 * var_10
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'\x00\x01\x02\xff'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'YWJj'
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #132
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'aA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'h'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = '_-w='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xec'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8$'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #133
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8gd29ybGQ'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = 'aGVsbG8gd29ybGQ'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = b'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'\xfb\xff\xff\xff\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #134
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = 'aA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'h'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'abc'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'dGVzdA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'
    var_18 = 'Pj4-Pg'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'>>>'



# Parsed testcases at query #135
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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'YT4+Yg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a>b'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = b'\x00\x01\x02\x03'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = '!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'abc!def'
    var_22 = module_0.base64_decode(var_21)



# Parsed testcases at query #136
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '_-x'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xfb\xe7'
    var_10 = module_0.base64_decode(var_2)
    assert var_10 == b'test'
    var_11 = '!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = 'test\x00'
    var_14 = module_0.base64_decode(var_13)



# Parsed testcases at query #137
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = '_-_w'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xef\xc0'
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'WA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'X'
    var_18 = 'WA'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'X'
    var_20 = 'PDw/Pz4+'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'<<??>>'
    var_22 = '!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8\x00'
    var_25 = module_0.base64_decode(var_24)
    var_26 = b'a'
    var_27 = 1000
    var_28 = var_26 * var_27
    var_29 = module_0.base64_encode(var_28)
    var_30 = module_0.base64_decode(var_29)
    var_31 = 'YQ=='
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'a'
    var_33 = 'YWI='
    var_34 = module_0.base64_decode(var_33)
    assert var_34 == b'ab'
    var_35 = 'YWJj'
    var_36 = module_0.base64_decode(var_35)
    assert var_36 == b'abc'
    var_37 = 'YWJjZA=='
    var_38 = module_0.base64_decode(var_37)
    assert var_38 == b'abcd'



# Parsed testcases at query #138
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b''
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'\x00\x01\x02\xff'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'\x00\x01\x02\xff'
    var_9 = 'hello'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = 'aGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = 'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = bytes(var_16)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #139
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = b'dGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'Hello, World! This is a test with more data.'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    var_9 = 256
    var_10 = range(var_9)
    var_11 = bytes(var_10)
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = b'V29ybGQ='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'World'



# Parsed testcases at query #140
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'hello world'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'\x00'
    var_16 = b'\xff'
    var_17 = b'test'
    var_18 = b'1234567890'
    var_19 = b'!@#$%^&*()'
    var_20 = [var_7, var_15, var_16, var_17, var_18, var_19]
    var_21 = module_0.base64_decode(var_1)
    var_22 = 'SGVsbG8gV29ybGQ='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello World'



# Parsed testcases at query #141
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'aGVsbG8_LT4_'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello?->?'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'not valid base64!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVs\nbG8g\nV29y\nbGQ='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello World'



# Parsed testcases at query #142
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = 'dGVzdA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = b'aGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = 'aGVsbG8'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'_-w'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'\xfb\xc3'
    var_25 = '_-w'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'\xfb\xc3'



# Parsed testcases at query #143
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b'dGVzdA=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'test'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'hello world!\n\t'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello world!\n\t'
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = '!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'abc$%^'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'w7zDpMO8'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'üäß'
    var_26 = 'utf-8'
    var_27 = module_1.encode(var_26)



# Parsed testcases at query #144
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'Pj4-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>>'
    var_14 = 'PDw8PA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<<<<'
    var_16 = '-__-'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xfb\xff\xfe'
    var_18 = 'WA=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'X'
    var_20 = '//8='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\xff\xff'
    var_22 = '!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'Hello World'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8==='
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #145
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = ''
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b''
    var_5 = 'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b'\x00\x01\x02\xff\xfe'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8gV29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello World'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'SGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



# Parsed testcases at query #146
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = '_-x'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xfb\xdf'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'dGVzdA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = 'dGVzdA'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test'
    var_14 = 'AAECAwQFBgcI'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x00\x01\x02\x03\x04\x05\x06\x07\x08'
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'ABC\x00DEF'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #147
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8gd29ybGQ'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'a'
    var_10 = b'ab'
    var_11 = b'abc'
    var_12 = b'abcd'
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = module_0.base64_decode(var_1)
    var_15 = b'\xff\xfe\x00\x01'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'not valid base64!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'aGVsbG8='
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #148
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'Pj4-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>>'
    var_14 = b'PDw8PA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<<<<'
    var_16 = b'AAECAwQFBgcI'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\x00\x01\x02\x03\x04\x05\x06\x07\x08'
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'not valid base64'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'\xffdGVzdA=='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'test'



# Parsed testcases at query #149
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'aGVsbG8_d29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello?world'
    var_6 = b'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'aGVsbG8_d29ybGQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello?world'
    var_16 = 'aGVsbG8td29ybGQ='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello-world'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8$'
    var_21 = module_0.base64_decode(var_20)
    var_22 = module_0.base64_decode(var_6)
    assert var_22 == b'test'



# Parsed testcases at query #150
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'aGVsbG8t'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello-'
    var_8 = b'aGVsbG9f'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello_'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'\x00\x01\x02'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\x00\x01\x02'
    var_15 = b'\xff\xfe\xfd'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xff\xfe\xfd'
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'SGVs@G8='
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #151
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = 'world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'world'
    var_6 = b'a'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = b'ab'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = b'!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'test123!@#'
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'test123!@#'
    var_21 = 'héllo'
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'utf-8'
    var_25 = module_1.encode(var_24)



# Parsed testcases at query #152
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'_-w'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xeb'
    var_14 = '_-w'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xff\xeb'
    var_16 = b'\x00\x01\x02'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'\x00\x01\x02'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'dGVzdA\xff'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'test'
    var_25 = b'x'
    var_26 = 1000
    var_27 = var_25 * var_26
    var_28 = module_0.base64_encode(var_27)
    var_29 = module_0.base64_decode(var_28)



# Parsed testcases at query #153
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'VGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Test'
    var_6 = 'Pj4_Pz8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'>>???'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'dGVzdCB3aXRoIHNwYWNl'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test with space'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'VGhpcyBpcyBpbnZhbGlk'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'YQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = 'YQ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'a'
    var_20 = 'YWJj'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'abc'
    var_22 = 'YWJjZA=='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'abcd'
    var_24 = 'YWJjZGU='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'abcde'



# Parsed testcases at query #154
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b''
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b'\x00\x01\x02'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x00\x01\x02'
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'dGVzdA=='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'test'
    var_23 = b'dGVzdA'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'test'
    var_25 = b'!!!invalid!!!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'\x00\x01\x02'
    var_28 = module_0.base64_decode(var_27)



# Parsed testcases at query #155
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'aGVsbG8t'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello-'
    var_12 = 'aGVsbG9f'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello_'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'aGVsbG8='
    var_17 = 'ascii'
    var_18 = module_1.encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'



# Parsed testcases at query #156
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'https://example.com/path?query=value'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'https://example.com/path?query=value'
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'aGVsbG8$'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'AA=='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'\x00'
    var_25 = b'AAA='
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'\x00\x00'
    var_27 = b'AAAA'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'\x00\x00\x00'



# Parsed testcases at query #157
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'test'
    var_4 = module_1.urlsafe_b64encode(var_3)
    var_5 = module_1.decode()
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'test'
    var_7 = module_1.urlsafe_b64encode(var_3)
    var_8 = b'='
    var_9 = module_1.decode()
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'test'
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b'aGVsbG8='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'hello'
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'aGVsbG8=\x80'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'hello'



# Parsed testcases at query #158
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = 'aGVsbG8td29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello-world'
    var_10 = 'aGVsbG8_d29ybGQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello?world'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'not valid base64'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #159
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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '_-x'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xeb'
    var_10 = b'This is a test message!'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'Not base64 data!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'AAEBAg=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'\x00\x01\x01\x02'



# Parsed testcases at query #160
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'WA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'X'
    var_10 = 'AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8='
    var_11 = module_0.base64_decode(var_10)
    var_12 = 32
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8=\x80'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = b'dGVzdA=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'test'



# Parsed testcases at query #161
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'Pz4_Pz4_Pz4_Pz4'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'>?>?>?>?>'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'YQ=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'a'
    var_13 = 'YWI='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'ab'
    var_15 = 'YWJj'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'abc'
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'aGVsbG8$'
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #162
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'Pj4_Pz8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'>>???'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'dGVzdA'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test'
    var_14 = b'dGVzdA'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test'
    var_16 = module_0.base64_decode(var_2)
    assert var_16 == b'test'
    var_17 = 'YQ'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'a'
    var_19 = 'YWI'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'ab'
    var_21 = 'YWJj'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'abc'
    var_23 = 'YWJjZA'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'abcd'
    var_25 = 'YWJjZGU'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'abcde'
    var_27 = 'dGVzdA==\x80'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'test'
    var_29 = '!!!'
    var_30 = module_0.base64_decode(var_29)
    var_31 = 'invalid!'
    var_32 = module_0.base64_decode(var_31)
    var_33 = 'dGV zdA=='
    var_34 = module_0.base64_decode(var_33)



# Parsed testcases at query #163
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'a'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a'
    var_6 = b'ab'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'ab'
    var_9 = b'abc'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'abc'
    var_12 = b''
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b'aGVsbG8='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'hello'
    var_17 = 'aGVsbG8='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'hello'
    var_19 = b'test'
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'test'
    var_22 = b'!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'not base64 at all'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 256
    var_27 = range(var_26)
    var_28 = bytes(var_27)
    var_29 = module_0.base64_encode(var_28)
    var_30 = module_0.base64_decode(var_29)
    var_31 = 'aGVsbG8=\x80\x81'
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'hello'



# Parsed testcases at query #164
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'SGVsbG8gV29ybGQ='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello World'
    var_11 = b'!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'hello world'
    var_14 = module_0.base64_decode(var_13)
    var_15 = bytes(var_13)
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #165
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'Hello World!'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'Hello World!'
    var_3 = 'SGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'Hello'
    var_5 = 'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'YQ=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'test data with \x00\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test data with \x00\xff'
    var_14 = b'SGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'\xff\xff\xff'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'test'
    var_21 = b'dGVzdA=='
    var_22 = (var_20, var_21)
    var_23 = b'a'
    var_24 = b'YQ=='
    var_25 = (var_23, var_24)
    var_26 = b'ab'
    var_27 = b'YWI='
    var_28 = (var_26, var_27)
    var_29 = b'abc'
    var_30 = b'YWJj'
    var_31 = (var_29, var_30)
    var_32 = b'abcd'
    var_33 = b'YWJjZA=='
    var_34 = (var_32, var_33)
    var_35 = [var_22, var_25, var_28, var_31, var_34]



# Parsed testcases at query #166
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'Pj4-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>>'
    var_14 = 'PDw8PA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<<<<'
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8$'
    var_24 = module_0.base64_decode(var_23)
    var_25 = '='
    var_26 = module_0.base64_decode(var_25)
    var_27 = '=='
    var_28 = module_0.base64_decode(var_27)



# Parsed testcases at query #167
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
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'V29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = 'dGVzdC11cmwtdmFsdWU'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-url-value'
    var_10 = module_0.base64_decode(var_8)
    assert var_10 == b'test-url-value'
    var_11 = '!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = 'w6TDtsO8'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'äöü'
    var_16 = 'utf-8'
    var_17 = module_1.encode(var_16)



# Parsed testcases at query #168
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'Pj4-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>>?'
    var_14 = b'Pj4-Pg'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'>>>?'
    var_16 = b'YQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = b'YWI='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'ab'
    var_20 = b'YWJj'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'abc'
    var_22 = 256
    var_23 = range(var_22)
    var_24 = bytes(var_23)
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'!!!invalid!!!'
    var_28 = module_0.base64_decode(var_27)
    var_29 = b'Hello World!'
    var_30 = module_0.base64_decode(var_29)
    var_31 = module_0.base64_decode(var_29)
    var_32 = module_0.base64_decode(var_2)



# Parsed testcases at query #169
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'V29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = 'Pz4_'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'?>'
    var_10 = 'PDw_Pz4-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<<??>'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'abc'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #170
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'aGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = b'YQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = b'dGVzdC91cmw='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test/url'
    var_18 = b'dGVzdC91cmw'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'test/url'
    var_20 = b'dGVzdC11cmw='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'test-url'
    var_22 = b'dGVzdC11cmw'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'test-url'
    var_24 = 'aGVsbG8gd29ybGQ='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'hello world'
    var_26 = b'!!!invalid!!!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = b'aGVs\x00bG8='
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'not valid base64!'
    var_31 = module_0.base64_decode(var_30)



# Parsed testcases at query #171
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '_-w'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xef'
    var_14 = 'MTIzNDU2Nzg5MA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'1234567890'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'not base64 at all'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=Ã('
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello('
    var_22 = 256
    var_23 = range(var_22)
    var_24 = bytes(var_23)
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)



# Parsed testcases at query #172
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = b'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '_-x'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xfb\xe7'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8gV29ybGQ\x80\x81'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello World'



# Parsed testcases at query #173
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'SGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'Pj4-Pz8_'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'>>>???'
    var_16 = 'SGVsbG8=ÿ'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'A'
    var_23 = 1000
    var_24 = var_22 * var_23
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)
    var_27 = 256
    var_28 = range(var_27)
    var_29 = bytes(var_28)
    var_30 = module_0.base64_encode(var_29)
    var_31 = module_0.base64_decode(var_30)



# Parsed testcases at query #174
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'Pj4-Pg=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'>>>'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'AA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\x00'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'SGVsbG8=\xff'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



# Parsed testcases at query #175
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b''
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'\x00\x01\x02\xff'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'\x00\x01\x02\xff'
    var_9 = b'test'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = 'hello'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello'
    var_15 = b'hello\x80world'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #176
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'test data 123'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data 123'
    var_6 = 'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = 'dGVzdC1f'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test-_'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'a\x00GVsbG8='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'hello'
    var_22 = b'a\xffGVsbG8='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'hello'



# Parsed testcases at query #177
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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'Pz4_'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'?>?'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'\xffSGVsbG8='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #178
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = 'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = module_0.base64_decode(var_10)
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'SGVsbG8='
    var_18 = 100
    var_19 = var_17 * var_18
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'test data'
    var_22 = module_1.b64encode(var_21)
    var_23 = module_1.decode()
    var_24 = '+'
    var_25 = '-'
    var_26 = '/'
    var_27 = '_'
    var_28 = module_1.encode()
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'test data'



# Parsed testcases at query #179
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'aGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = 'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'dGVzdA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = 'Pj4-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>>'
    var_14 = 'PDw8PA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<<<<'
    var_16 = 200
    var_17 = chr(var_16)
    var_18 = var_0 + var_17
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'test'
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'dGVzdA==='
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'x'
    var_25 = 1000
    var_26 = var_24 * var_25
    var_27 = module_0.base64_encode(var_26)
    var_28 = module_0.base64_decode(var_27)
    var_29 = 256
    var_30 = range(var_29)
    var_31 = bytes(var_30)
    var_32 = module_0.base64_encode(var_31)
    var_33 = module_0.base64_decode(var_32)
    var_34 = 'AA=='
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'\x00'
    var_36 = '/w=='
    var_37 = module_0.base64_decode(var_36)
    assert var_37 == b'\xff'



# Parsed testcases at query #180
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'aGVsbG8vd29ybGQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello/world'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = '+/8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xfb\xff'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'\xff\xfe\xfd'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #181
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = b'd29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'world'
    var_4 = 'aGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dGVzdF91cmw'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test_url'
    var_10 = 'dGVzdC12YWx1ZQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test-value'
    var_12 = 'invalid!@#'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'w7xiw7xsZ8Ok'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'übülsgä'
    var_17 = 'utf-8'
    var_18 = module_1.encode(var_17)



# Parsed testcases at query #182
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'Pj4-Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>'
    var_10 = 'PDw8PA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<<<<'
    var_12 = b'SGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 200
    var_17 = chr(var_16)
    var_18 = var_14 + var_17
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #183
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b''
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b''
    var_5 = b'a'
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = b'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'aGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'\x00\x01\x02'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\x00\x01\x02'
    var_15 = b'\xff\xfe\xfd'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xff\xfe\xfd'
    var_18 = b'12345'
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'12345'
    var_21 = b'!@#$%^&*()'
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'!@#$%^&*()'
    var_24 = 'hello'
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'hello'
    var_27 = 'aGVsbG8='
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'hello'
    var_29 = b'!!!invalid!!!'
    var_30 = module_0.base64_decode(var_29)
    var_31 = b'aGVsbG8!!!'
    var_32 = module_0.base64_decode(var_31)
    var_33 = b'YQ=='
    var_34 = module_0.base64_decode(var_33)
    assert var_34 == b'a'
    var_35 = b'YWI='
    var_36 = module_0.base64_decode(var_35)
    assert var_36 == b'ab'
    var_37 = b'YWJj'
    var_38 = module_0.base64_decode(var_37)
    assert var_38 == b'abc'



# Parsed testcases at query #184
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = 'aGVsbG8t'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello-'
    var_6 = 'aGVsbG9f'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello_'
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = module_0.base64_decode(var_10)
    assert var_14 == b''
    var_15 = '!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'\xff\xff'
    var_18 = module_0.base64_decode(var_17)
    var_19 = module_0.base64_decode(var_17)
    assert var_19 == b'test'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b'data with \x00 null'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'data with \x00 null'
    var_9 = b''
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 256
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'aGVsbG8'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'hello'
    var_19 = b'aGVsbG8='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'hello'
    var_21 = b'aGVsbG8=='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'hello'
    var_23 = b'dGVzdA=='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'test'
    var_25 = b'!!!invalid!!!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'aGVs\x00bG8='
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'dGVzdA=='
    var_30 = 'utf-16'
    var_31 = module_1.encode(var_30)
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'test'
    var_33 = b'x'
    var_34 = module_0.base64_decode(var_15)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = b'd29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'world'
    var_4 = 'aGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'WA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'X'
    var_10 = 'Pz4_Pz4_Pz4-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'?>?>?>?>'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'hello world'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'a'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'w6TDtsO8'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'äöü'
    var_21 = 'utf-8'
    var_22 = module_1.encode(var_21)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'YQ=='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'a'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'aGVsbG8gd29ybGQ'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello world'
    var_11 = b'dGVzdC0t'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'test--'
    var_13 = b'dGVzdF8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'test_'
    var_15 = 'aGVsbG8='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'hello'
    var_17 = 'dGVzdA=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'test'
    var_19 = 'SGVsbG8='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'hello world!@#$%'
    var_24 = module_0.base64_decode(var_23)
    var_25 = module_0.base64_decode(var_7)
    assert var_25 == b''
    var_26 = b'YQ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'a'
    var_28 = b'YWI'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'ab'
    var_30 = b'YWJj'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'abc'
    var_32 = b'YWJjZA'
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'abcd'
    var_34 = b'YWJjZGU'
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'abcde'
    var_36 = b'YWJjZGVm'
    var_37 = module_0.base64_decode(var_36)
    assert var_37 == b'abcdef'
    var_38 = b'a'
    var_39 = b'ab'
    var_40 = b'abc'
    var_41 = b'test'
    var_42 = b'hello world'
    var_43 = b'12345'
    var_44 = [var_7, var_38, var_39, var_40, var_41, var_42, var_43]
    var_45 = module_0.base64_decode(var_1)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'dGVzdA=='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'test'
    var_7 = b'dGVzdA'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'test'
    var_9 = module_0.base64_decode(var_7)
    assert var_9 == b'test'
    var_10 = 'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'hello+world'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'hello+world'
    var_17 = 256
    var_18 = range(var_17)
    var_19 = bytes(var_18)
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'aGVsbG8'
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b'aGVsbG8_d29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello?world'
    var_8 = b'aGVsbG8td29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello-world'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = module_0.base64_decode(var_10)
    assert var_14 == b''
    var_15 = 'dGVzdA==\x80'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'test'
    var_17 = b'!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'not base64'
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\x00\x01\x02\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\x00\x01\x02\xff'
    var_14 = b'x'
    var_15 = 1000
    var_16 = var_14 * var_15
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'aGVsbG8!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 256
    var_24 = range(var_23)
    var_25 = bytes(var_24)
    var_26 = module_0.base64_encode(var_25)
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8gd29ybGQ'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b'aGVsbG8gd29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = b'aGVsbG8gd29ybGQ'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello world'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'aA=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'h'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 256
    var_16 = range(var_15)
    var_17 = bytes(var_16)
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'12345'
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'12345'
    var_23 = b'x'
    var_24 = module_0.base64_encode(var_17)
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #8
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
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'VGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = b'VGVzdA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Test'
    var_10 = '_-w=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xeb'
    var_12 = 'VGhpcyBpcyBhIHRlc3Q='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'This is a test'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'\xff\xff\xff'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'w6TDtsOcw7w='
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'äöü'
    var_21 = 'utf-8'
    var_22 = module_1.encode(var_21)



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = 'aGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = 'd29ybGQ='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'world'
    var_16 = 'd29ybGQ'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'world'
    var_18 = 'invalid!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'\xff\xfe\x00\x01'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'aGVsbG8='
    var_23 = 128
    var_24 = chr(var_23)
    var_25 = var_22 + var_24
    var_26 = module_0.base64_decode(var_25)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = 'aGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = 'dGVzdC93aXRoL3NwZWNpYWw='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'test/with/special'
    var_15 = b'dGVzdA=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'test'
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'not valid base64'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'héllo'
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'utf-8'
    var_25 = module_1.encode(var_24)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello world'
    var_6 = b'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'\xff\xfb\x00'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'\xff\xfb\x00'
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b'=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = b'==='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'\xff\x80aGVsbG8='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'hello'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'test+data/with=special'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = 'aGVsbG8=\x80'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello world'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'a'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = b'\x00\x01\x02'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\x00\x01\x02'
    var_15 = 256
    var_16 = range(var_15)
    var_17 = bytes(var_16)
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'aGVsbG8='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'hello'
    var_22 = b'!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'aGVs'
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'V29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = 'Pj4-Pz8_'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>??'
    var_10 = 'AAECAwQFBgcICQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'aGVsbG8t'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello-'
    var_10 = 'aGVsbG9f'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello_'
    var_12 = 256
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'YQ'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'a'
    var_19 = 'YWI'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'ab'
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = module_0.base64_decode(var_21)
    assert var_23 == b'Hello World'



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'aGVsbG8vd29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello/world'
    var_14 = 'aGVsbG8rd29ybGQ='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello+world'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'not-base64!'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = '_-w'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xeb'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = 'MTIz'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'123'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'abc$%^'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8gV29ybGQ=\x00'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello World'



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = module_0.base64_decode(var_2)
    assert var_6 == b'test'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'Pj4-Pg=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'>>>'
    var_11 = 'AAECAwQFBgcI'
    var_12 = module_0.base64_decode(var_11)
    var_13 = 9
    var_14 = range(var_13)
    var_15 = bytes(var_14)
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = module_0.base64_decode(var_7)
    assert var_18 == b''



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8gd29ybGQ'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b'aGVsbG8gd29ybGQ'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'YQ=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'a'
    var_13 = b'YQ'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'a'
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'hello world'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'\xff\xfe\xfd'
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\xff\xfe\xfd'



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = 'dGVzdA_-'
    var_5 = module_0.base64_decode(var_4)
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = 'aGVsbG8gd29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello world'
    var_12 = b'\x00\x01\x02\xff\xfe\xfd'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'dGVzdA===='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'test'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = 'Pj4-Pj4'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>>'
    var_10 = 'Lg=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'.'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'dGVzdA==\x80'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test'



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'aGVsbG8+'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello>'
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'test'
    var_11 = b'w6Rh'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'\xe4a'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'test\x00'
    var_16 = module_0.base64_decode(var_15)



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'VGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = b'VGVzdA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Test'
    var_10 = 'aGVsbG8gd29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello world'
    var_12 = 'dGVzdC0t'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test--'
    var_14 = 'dGVzdF8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test_'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8'
    var_19 = module_0.base64_decode(var_18)
    var_20 = '123'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'aGVsbG8td29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello-world'
    var_10 = 'YSBiCg=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a b\n'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'Hello World'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #25
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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'Pj4-Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>'
    var_10 = 'PDw8PA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<<<<'
    var_12 = 'YWJjZGVmZw=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abcdefg'
    var_14 = 'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = 'YQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = 'YWI='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'ab'
    var_20 = 'YWJj'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'abc'
    var_22 = '!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = ' '
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'abc123!!!'
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = '!!!invalid!!!'
    var_9 = module_0.base64_decode(var_8)
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = '_-w='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xec'
    var_14 = 'MTIzNA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'1234'



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test string'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = ''
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'\x00'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\x00'
    var_15 = b'\xff\xfe\xfd\xfc'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xff\xfe\xfd\xfc'
    var_18 = 256
    var_19 = range(var_18)
    var_20 = bytes(var_19)
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    var_23 = '!!!invalid base64!!!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = ''
    var_26 = module_0.base64_decode(var_25)
    var_27 = 256
    var_28 = [i % var_27 for i in var_25]
    var_29 = bytes(var_28)
    var_30 = module_0.base64_encode(var_29)
    var_31 = module_0.base64_decode(var_30)
    var_32 = b'direct bytes'
    var_33 = module_0.base64_encode(var_32)
    var_34 = module_0.base64_decode(var_33)
    assert var_34 == b'direct bytes'



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'\xff\xfe\x00\x01'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = module_0.base64_decode(var_14)
    assert var_16 == b'hello'



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aA=='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'h'
    var_7 = 'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = 'invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'a'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b''
    var_18 = b'a'
    var_19 = b'test'
    var_20 = b'\x00\x01\x02'
    var_21 = b'data with spaces'
    var_22 = 256
    var_23 = range(var_22)
    var_24 = bytes(var_23)
    var_25 = [var_17, var_18, var_19, var_15, var_20, var_21, var_24]
    var_26 = module_0.base64_decode(var_1)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello world'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = ''
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'a'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'a'
    var_15 = b'\x00\xff\xfe'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\x00\xff\xfe'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'12345'
    var_23 = module_0.base64_encode(var_22)
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'12345'
    var_25 = 'héllo wörld'
    var_26 = module_0.base64_encode(var_25)
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'utf-8'
    var_29 = module_1.encode(var_28)



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = 'world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'world'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'test\x00data'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'aGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = 'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'héllo'
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'utf-8'
    var_22 = module_1.encode(var_21)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'dGVzdA=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'test'
    var_11 = 'Pz4_'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'?>'
    var_13 = 256
    var_14 = range(var_13)
    var_15 = bytes(var_14)
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'aGVsbG8\x80'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'hello'



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'test\x00data\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'aGVsbG8'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = module_0.base64_decode(var_14)
    assert var_18 == b'hello'



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'V29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = 'Pj4-Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = '_-5v'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xff\xeeo'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'abcd\x80ef=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'i\xb7\x1d\xef'



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = b'_-4='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xef\xb8'
    var_14 = '_-4='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xff\xef\xb8'
    var_16 = b'Test data with special chars: \x00\xff\x01'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = '!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'aGVsbG8g\udce3world'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 256
    var_24 = range(var_23)
    var_25 = bytes(var_24)
    var_26 = module_0.base64_encode(var_25)
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #36
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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '_-w='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xec'
    var_10 = 'VGhpcyBpcyBhIHRlc3Q='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'This is a test'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'not@base64'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'dGVzdA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'aGVsbG8td29ybGQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello-world'
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'dGVzdC1zdHJpbmctd2l0aC1zcGVjaWFsLWNoYXJz'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test-string-with-special-chars'



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'YQ=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = 'YWI='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'ab'
    var_11 = 'YWJj'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = b'aGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello'
    var_15 = b'\xff\xfe'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xff\xfe'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'aGVsbG8$'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'w7xibGVy'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'\xc3\xbcbler'



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8gd29ybGQ'
    var_4 = module_0.base64_decode(var_3)
    var_5 = b'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b'aGVs'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hel'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'test_data+with/special~chars'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test_data+with/special~chars'
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'aGVsbG8\xff'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'hello'



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = module_0.base64_decode(var_3)
    assert var_9 == b'hello'
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'aGVsbG8_d29ybGQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello?world'
    var_14 = 'aGVsbG8-d29ybGQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello>world'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVsbG8='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'
    var_20 = b'\x00\x01\x02\xff\xfe'
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'x'
    var_24 = 1000
    var_25 = var_23 * var_24
    var_26 = module_0.base64_encode(var_25)
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #41
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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'YWJj'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'abc'
    var_10 = 'YWJjZA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'abcd'
    var_12 = 'Pj4-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>>'
    var_14 = 'PDw8PA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<<<<'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'hello world'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dGVzdC91cmw'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test/url'
    var_10 = 'dGVzdF91cmw='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test_url'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'VGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = '_-x'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xeb'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = '!'
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'YQ=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'a'
    var_13 = b'YWI='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'ab'
    var_15 = b'YWJj'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'abc'
    var_17 = b'test data with + and /'
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'test data with + and /'
    var_20 = b'!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'not-base64-characters!'
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #45
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = '_-w'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xe0'
    var_10 = module_0.base64_decode(var_4)
    assert var_10 == b''
    var_11 = '!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'SGVsbG8=\xff'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'test'
    var_4 = module_1.b64encode(var_3)
    var_5 = module_1.decode()
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'test'
    var_7 = b'bytes'
    var_8 = module_0.base64_encode(var_7)
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'bytes'
    var_10 = 'string'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'string'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b'\x00\x01\x02\xff'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = module_0.base64_encode(var_18)
    var_21 = b'='



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'YQ=='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'a'
    var_7 = b'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b'YQ'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = b''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = 'aGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello'
    var_15 = b'invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'a'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'\xff\xfe\xfd\xfc'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xff\xfe\xfd\xfc'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'abc'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'dGVzdA=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'test'
    var_22 = 'dGVzdA=='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'test'



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'aGVsbG8v'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello/'
    var_16 = 'aGVsbG8t'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello-'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = '123'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 256
    var_23 = range(var_22)
    var_24 = bytes(var_23)
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)
    var_27 = module_0.base64_decode(var_10)
    assert var_27 == b''
    var_28 = 'Zg=='
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'f'
    var_30 = 'Zm8='
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'fo'
    var_32 = 'Zm9v'
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'foo'



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'YQ=='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'a'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'YQ'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = module_0.base64_decode(var_9)
    assert var_11 == b'a'
    var_12 = module_0.base64_decode(var_5)
    assert var_12 == b'a'
    var_13 = 'aGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello'
    var_15 = b'aGVsbG8'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'hello'
    var_17 = b'test data with spaces'
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'test data with spaces'
    var_20 = 256
    var_21 = range(var_20)
    var_22 = bytes(var_21)
    var_23 = module_0.base64_encode(var_22)
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'!!!invalid!!!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'aGVsbG8\x00'
    var_28 = module_0.base64_decode(var_27)
    var_29 = b'aGVsbG8\xff'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'hello'



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'aGVsbG8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\x00\x01\x02\xff\xfe'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = '!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'aGVsbG8=\x80'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'hello'



# Parsed testcases at query #52
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = module_0.base64_decode(var_3)
    assert var_7 == b'hello'
    var_8 = 'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'Hello, World! This is a test with more characters.'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'\x00\x01\x02\xff\xfe'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'aGVs#bG8='
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'dGVzdA=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'test'
    var_22 = 'YXNjaWk='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'ascii'



# Parsed testcases at query #53
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'Pj4-Pg'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'>>>'
    var_6 = b'VGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'WA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'X'
    var_12 = 'YSBi'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a b'
    var_14 = 'Pj4-Pg=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'>>>'
    var_16 = 'X19f'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'___'
    var_18 = 'LS0t'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'---'
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'ABCD$%^'
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #54
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YWI='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = 'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = '_-A='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'\xfb\xe0'
    var_20 = 'w7xsw6TDtmzDtsO2'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\xc3\xbcl\xc3\xa4d\xc3\xb6l\xc3\xb6\xc3\xb6'
    var_22 = 'invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'test\x00'
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #55
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\x00\x01\x02\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'not valid base64@#$%'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #56
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b'a'
    var_7 = module_0.base64_decode(var_1)
    var_8 = b'\x00\x01\x02\xff\xfe'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = b''
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'test'
    var_15 = module_0.base64_encode(var_14)
    var_16 = b'='
    var_17 = var_15 + var_16
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'test'
    var_19 = '!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'\xff\xfe\xfd'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'not base64 data!'
    var_24 = module_0.base64_decode(var_23)



# Parsed testcases at query #57
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'abc'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = b'hello+world/test?query=1'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello+world/test?query=1'
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'héllo wörld'
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'utf-8'
    var_25 = module_1.encode(var_24)
    var_26 = b'!!!invalid!!!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = b'abc$%^'
    var_29 = module_0.base64_decode(var_28)
    var_30 = b'abc'
    var_31 = module_0.base64_decode(var_30)



# Parsed testcases at query #58
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'Pj4-Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>?'
    var_10 = 'AAECAwQFBgc='
    var_11 = module_0.base64_decode(var_10)
    var_12 = 8
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8=ÿ'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'



# Parsed testcases at query #59
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'dGVzdF91cmw'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test_url'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'YQ=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = 'YWI='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = 'YWJj'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abc'
    var_14 = 'AAECAwQFBgcI'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 9
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = 'Lg=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'.'
    var_21 = 'Lw=='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'/'
    var_23 = 'w6TDpMO8'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'ääü'
    var_26 = module_1.encode()
    var_27 = '!!!'
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'not valid!'
    var_30 = module_0.base64_decode(var_29)



# Parsed testcases at query #60
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = 'aGVsbG8'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'aGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'not valid base64!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'\x00\xff\xfe'
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'\x00\xff\xfe'
    var_20 = '_-w'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\xfb\xc0'
    var_22 = 'YQ'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'a'
    var_24 = 'YWI'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'ab'
    var_26 = 'YWJj'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'abc'



# Parsed testcases at query #61
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8gd29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'aA=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'h'
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'aGVsbG8$'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'Pj4_Pj4'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'>>?>'
    var_19 = 256
    var_20 = range(var_19)
    var_21 = bytes(var_20)
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #62
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'hello\nworld\t!'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello\nworld\t!'
    var_13 = 'héllo wörld'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'utf-8'
    var_17 = module_1.encode(var_16)
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'aGVsbG8='
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #63
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'YQ=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = 'YWI='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'ab'
    var_11 = 'YWJj'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = b'\x00\x01\x02\xff\xfe'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x00\x01\x02\xff\xfe'
    var_16 = b'dGVzdA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'aGVsbG8$'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'aGVsbG8\x80'
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #64
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b''
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b''
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = b'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = 'aGVsbG8gd29ybGQ='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello world'
    var_13 = b'\xff\xfe\x00\x01'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVsbG8$'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'???'
    var_21 = module_0.base64_decode(var_20)
    var_22 = bytes(var_20)
    var_23 = module_0.base64_encode(var_22)
    var_24 = module_0.base64_decode(var_23)



# Parsed testcases at query #65
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = b'_-x'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xff\xeb'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'Zg=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'f'
    var_12 = b'Zm8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'fo'
    var_14 = b'Zm9v'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'foo'
    var_16 = b'a'
    var_17 = 100
    var_18 = var_16 * var_17
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'YQ=='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'a'
    var_23 = b'YWI='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'ab'
    var_25 = b'YWJj'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'abc'
    var_27 = b'!!!invalid!!!'
    var_28 = module_0.base64_decode(var_27)
    var_29 = b'dGVzdA==\xff'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'test'



# Parsed testcases at query #66
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'aGVsbG8g8J+YgQ=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello \xf0\x9f\x98\x81'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8gV29ybGQ=='
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #67
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'test'
    var_4 = module_1.b64encode(var_3)
    var_5 = module_1.decode()
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'test'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'not base64 at all'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'\xff\xfe\x00\x01'
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'aGVsbG8'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'hello'
    var_22 = b'aGVs'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'hes'



# Parsed testcases at query #68
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'test'
    var_4 = module_1.b64encode(var_3)
    var_5 = 'ascii'
    var_6 = module_1.decode(var_5)
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'a'
    var_13 = module_1.urlsafe_b64encode(var_12)
    var_14 = module_1.decode(var_5)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'abcxyz123!!!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'abc def'
    var_26 = module_0.base64_decode(var_25)
    var_27 = module_1.urlsafe_b64encode(var_3)
    var_28 = b'='
    var_29 = module_0.base64_encode(var_12)
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'a'
    var_31 = b'\x00\x01\x02\xff\xfe'
    var_32 = module_0.base64_encode(var_31)
    var_33 = module_0.base64_decode(var_32)



# Parsed testcases at query #69
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'VGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Test'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'VGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'aGVsbG8td29ybGQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello-world'
    var_14 = 'aGVsbG8vd29ybGQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello/world'
    var_16 = 'MTIzNDU2'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'123456'
    var_18 = 'aGVsbG8_d29ybGQ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello?world'
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = '1234'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'\xff\xfe'
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #70
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = 'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = 'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'd29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'world'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'aGVsbG8=\x80'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'



# Parsed testcases at query #71
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'Pj4_Pz8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>???\xbf\xbf'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8gV29ybGQ!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'YQ=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = 'YWI='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'ab'
    var_18 = 'YWJj'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abc'



# Parsed testcases at query #72
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'Zg=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'f'
    var_9 = 'YQ=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = 'YWI='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'ab'
    var_13 = 'YWJj'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'abc'
    var_15 = '_-w='
    var_16 = module_0.base64_decode(var_15)
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 256
    var_20 = range(var_19)
    var_21 = bytes(var_20)
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'a G V s b G 8 ='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'hello'
    var_26 = b'x'
    var_27 = 10000
    var_28 = var_26 * var_27
    var_29 = module_0.base64_encode(var_28)
    var_30 = module_0.base64_decode(var_29)



# Parsed testcases at query #73
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
    var_6 = b'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'Pj4+Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>>>'
    var_14 = 'Pj4-Pg=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'>>>>'
    var_16 = 'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = 'YWJjZA=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abcd'
    var_20 = 'YQ=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'a'
    var_22 = 'YWI='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'ab'
    var_24 = module_0.base64_decode(var_16)
    assert var_24 == b'abc'
    var_25 = module_0.base64_decode(var_18)
    assert var_25 == b'abcd'
    var_26 = 'YWJjZGU='
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'abcde'
    var_28 = 'YWJjZGVm'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'abcdef'
    var_30 = '!!!invalid!!!'
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'abc!'
    var_33 = module_0.base64_decode(var_32)
    var_34 = '===='
    var_35 = module_0.base64_decode(var_34)
    var_36 = 'SGVsbG8gV29ybGQ=\x00'
    var_37 = module_0.base64_decode(var_36)
    assert var_37 == b'Hello World'



# Parsed testcases at query #74
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'test data with spaces and special chars!@#$%'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'YQ=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'a'
    var_21 = b'YQ'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'a'
    var_23 = b'!!!invalid!!!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'aGVsbG8=\xff'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'hello'



# Parsed testcases at query #75
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '_-x0='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\x1d'
    var_10 = 'SGVsbG8gV29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello World'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'dGVzdA==\x80'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test'



# Parsed testcases at query #76
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'Zg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'f'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = b'SGVsbG8g\x80V29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello World'
    var_14 = 'Pj4-Pg=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'>>> '
    var_16 = 'Pj4-Pg'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'>>> '



# Parsed testcases at query #77
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8gV29ybGQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = '_-x'
    var_7 = module_0.base64_decode(var_6)
    var_8 = len(var_7)
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'WA=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'X'
    var_13 = 'SGVsbG8gV29ybGQ='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello World'
    var_15 = 'SGVsbG8gV29ybGQ\x80'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello World'



# Parsed testcases at query #78
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'VGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Test'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'Pz4_'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'?>\xef'
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = '!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'x'
    var_19 = 1000
    var_20 = var_18 * var_19
    var_21 = module_1.b64encode(var_20)
    var_22 = module_1.decode()
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'AA=='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'\x00'
    var_26 = 'MTIzNDU2Nzg5MA=='
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'1234567890'



# Parsed testcases at query #79
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = '_-w'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xeb'
    var_10 = 'dGVzdGluZyB3aXRoIHNwZWNpYWwgY2hhcmFjdGVycyAhQCMkJV4mKigp'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'testing with special characters !@#$%^&*()'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = '\x00\x01\x02'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'dGVzdA'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'
    var_18 = 'dGVzdDEyMw'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'test123'



# Parsed testcases at query #80
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'aGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = 'd29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'world'
    var_6 = 'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = 'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = 'd29ybGQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'world'
    var_12 = b'dGVzdA=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test'
    var_14 = b'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'Pz4_'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'?>?'
    var_20 = module_0.base64_decode(var_18)
    assert var_20 == b'?>?'
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'not base64 valid chars'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'dGVzdA\x80\x81'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'test'



# Parsed testcases at query #81
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b'test data with + and /'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'test data with + and /'
    var_11 = b''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b'YQ=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'a'
    var_17 = b'YWI='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'ab'
    var_19 = b'YWJj'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'abc'
    var_21 = module_0.base64_decode(var_2)
    assert var_21 == b'Hello'
    var_22 = b'SGVsbG8=\xff'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = 'SGVsbG8=ÿ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'



# Parsed testcases at query #82
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'aGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = 'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = 'Pz4_'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'?>\xff'
    var_10 = b'dGVzdA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'YQ=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = 'YWI='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'ab'
    var_18 = 'YWJj'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abc'
    var_20 = '+/8='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\xfb\xff'
    var_22 = '!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'not valid base64!!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'w7xrw7xs'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'\xc3\xbc\xc3\xbc\xc3\xbc'



# Parsed testcases at query #83
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'Pj4-Pg=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'>>>'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'YWJj'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'abc'
    var_12 = 'MTIz'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'123'
    var_14 = 'dGVzdC11cmwtc2FmZQ=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test-url-safe'



# Parsed testcases at query #84
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'test'
    var_4 = module_1.urlsafe_b64encode(var_3)
    var_5 = 'ascii'
    var_6 = module_1.decode(var_5)
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = 'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'aGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b'_-w'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xff\xec'
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'aGVsbG8=\x80'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'hello'



# Parsed testcases at query #85
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'aGVsbG8tX3dvcmxk'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello_world'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'w7Zsw6Rmw6bDtsO2'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'öläfööö'
    var_15 = 'utf-8'
    var_16 = module_1.encode(var_15)



# Parsed testcases at query #86
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8gd29ybGQ'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'dGVzdA=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'test'
    var_15 = b'dGVzdA'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'test'
    var_17 = b'dA=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b't'
    var_19 = b'dA'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b't'
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'aGVsbG8$'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'not base64 at all'
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'a'
    var_28 = b'ab'
    var_29 = b'abc'
    var_30 = b'test data with spaces'
    var_31 = b'binary\x00data'
    var_32 = b'1234567890'
    var_33 = 256
    var_34 = range(var_33)
    var_35 = bytes(var_34)
    var_36 = [var_7, var_27, var_28, var_29, var_30, var_31, var_32, var_35]
    var_37 = module_0.base64_decode(var_1)



# Parsed testcases at query #87
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8gd29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = b'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = 'aGVsbG8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b'\x00\x01\x02'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x00\x01\x02'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'aGVsbG8$'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #88
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'd29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'world'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'YQ=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = b'\x00\x01\x02\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = module_0.base64_decode(var_9)
    assert var_14 == b'a'
    var_15 = 'YWI='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'ab'
    var_17 = 'YWJj'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'abc'
    var_19 = '_-w='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\xff\xec'
    var_21 = '!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'YQ'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'a'
    var_25 = 'YWI'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'ab'



# Parsed testcases at query #89
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'test_data_123'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'test_data_123'
    var_11 = b'dGVzdF9kYXRh'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'test_data'
    var_13 = b'dA=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b't'
    var_15 = b'dGU='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'te'
    var_17 = b'dGVz'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'tes'
    var_19 = b'AA=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\x00'
    var_21 = b'_w=='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'\xff'
    var_23 = b'!!!invalid!!!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = module_0.base64_decode(var_2)
    assert var_25 == b'Hello'
    var_26 = b'a'
    var_27 = b'ab'
    var_28 = b'abc'
    var_29 = b'test string with spaces'
    var_30 = b'\x00\x01\x02\xff\xfe'
    var_31 = b'1234567890'
    var_32 = b'data_with_underscores'
    var_33 = 1000
    var_34 = var_26 * var_33
    var_35 = [var_6, var_26, var_27, var_28, var_29, var_30, var_31, var_32, var_34]
    var_36 = module_0.base64_decode(var_9)
    var_37 = b'SGVsbG8'
    var_38 = module_0.base64_decode(var_37)
    var_39 = b'SGVs\nbG8='
    var_40 = module_0.base64_decode(var_39)



# Parsed testcases at query #90
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dGVzdC13aXRoLXVybC1zYWZlLWNoYXJz'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-with-url-safe-chars'
    var_10 = 'dGVzdC13aXRoX3VuZGVyc2NvcmU='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test-with_underscore'
    var_12 = 'invalid!!!'
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #91
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b''
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'test'
    var_7 = module_1.b64encode(var_6)
    var_8 = 'ascii'
    var_9 = module_1.decode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'test'
    var_11 = b'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'123'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'123'
    var_16 = b'data with spaces'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'data with spaces'
    var_19 = '!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'\xff\xff\xff'
    var_22 = module_0.base64_decode(var_21)
    var_23 = ''
    var_24 = module_0.base64_decode(var_23)



# Parsed testcases at query #92
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\x00\x01\x02'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\x00\x01\x02'
    var_14 = 'héllo'
    var_15 = 'utf-8'
    var_16 = module_1.encode(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = module_1.encode(var_15)
    var_20 = b'!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'aGVsbG8$'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'aGVsbG8\x80'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'hello'



# Parsed testcases at query #93
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'WA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'X'
    var_10 = 'Pj4-Pg=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'>>>'
    var_12 = '_-w='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xff'
    var_14 = module_0.base64_decode(var_0)
    assert var_14 == b'Hello World'
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'a'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 50



# Parsed testcases at query #94
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = module_0.base64_decode(var_3)
    assert var_9 == b'hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YWI='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = 'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = 'YWJjZA=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abcd'
    var_20 = b'hello+world/'
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'hello+world/'
    var_23 = '!!!invalid!!!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'aGVsbG8$'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'aGVsbG8=ÿ'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'hello'
    var_29 = 'aGVsbG8=é'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'hello'



# Parsed testcases at query #95
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b'YQ=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'a'
    var_15 = b'YWI='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'ab'
    var_17 = b'YWJj'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'abc'
    var_19 = b'YWJjZA=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'abcd'
    var_21 = b'Lg=='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'.'
    var_23 = b'Lw=='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'/'
    var_25 = b'Kw=='
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'+'
    var_27 = b'!!!invalid!!!'
    var_28 = module_0.base64_decode(var_27)
    var_29 = b'aGVs?bG8='
    var_30 = module_0.base64_decode(var_29)
    var_31 = b'\x00'
    var_32 = b'\xff\xff'
    var_33 = b'test'
    var_34 = 100
    var_35 = var_33 * var_34
    var_36 = 256
    var_37 = range(var_36)
    var_38 = bytes(var_37)
    var_39 = [var_9, var_31, var_32, var_35, var_38]
    var_40 = module_0.base64_decode(var_1)



# Parsed testcases at query #96
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
    var_6 = b'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = '_-x4'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\x1e'
    var_10 = '_-x4='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\x1e'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'YQ=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = 'YWI='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'ab'
    var_18 = 'YWJj'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abc'
    var_20 = 'invalid!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'not valid base64!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'\xff\xfe\xfd'
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #97
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'aGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = 'd29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'world'
    var_6 = 'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = 'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'dGVzdA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = b'aGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = module_0.base64_decode(var_6)
    assert var_14 == b'test'
    var_15 = module_0.base64_decode(var_8)
    assert var_15 == b'hello'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = b''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'not-base64'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'\xff\xfe\xfd'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'test data with spaces and symbols!@#$%'
    var_27 = module_0.base64_encode(var_26)
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'utf-8'
    var_30 = module_1.encode(var_29)
    var_31 = b'\x00\x01\x02\xff\xfe'
    var_32 = module_0.base64_encode(var_31)
    var_33 = module_0.base64_decode(var_32)
    var_34 = module_0.base64_encode(var_24)
    var_35 = module_0.base64_decode(var_34)



# Parsed testcases at query #98
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = '_-w='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xeb'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'abc123!@#'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'YQ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'a'
    var_20 = 'YWI'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'ab'



# Parsed testcases at query #99
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'_-A'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'\xff\xe0'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 256
    var_16 = range(var_15)
    var_17 = bytes(var_16)
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'x'
    var_21 = module_0.base64_decode(var_1)



# Parsed testcases at query #100
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b'\xff\xfe\xfd\xfc'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = b'!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'a'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'a'
    var_17 = b'YQ'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'a'
    var_19 = b'YWI'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'ab'



# Parsed testcases at query #101
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'dGVzdA=='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'test'
    var_5 = b'dGVzdA'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'test'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'PDw_Pz8-Pg=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'<<??>>'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'not valid base64'
    var_16 = module_0.base64_decode(var_15)



# Parsed testcases at query #102
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'Hello World'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'SGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'Hello'
    var_5 = 'SGVsbG8gV29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'Hello World'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'\x00\x01\x02\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'SGVsbG8'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



# Parsed testcases at query #103
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
    var_6 = b'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 256
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'dGVzdA=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'test'
    var_19 = 'dGVzdA'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'test'
    var_21 = module_0.base64_decode(var_19)
    assert var_21 == b'test'
    var_22 = 'YSBi'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'a b'
    var_24 = module_0.base64_decode(var_22)
    assert var_24 == b'a b'
    var_25 = 'MTIz'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'123'
    var_27 = module_0.base64_decode(var_25)
    assert var_27 == b'123'
    var_28 = '!!!invalid!!!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = b'!!!invalid!!!'
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'abc$%^'
    var_33 = module_0.base64_decode(var_32)
    var_34 = module_0.base64_decode(var_8)
    assert var_34 == b''
    var_35 = 10
    var_36 = [var_33]
    var_37 = bytes(var_3)
    var_38 = module_0.base64_encode(var_37)
    var_39 = module_0.base64_decode(var_38)



# Parsed testcases at query #104
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello world'
    var_6 = b'a'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = b''
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'test data'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'test data'
    var_17 = b'\x00\x01\x02\xff'
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'\x00\x01\x02\xff'



# Parsed testcases at query #105
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'aGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'aGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = b'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'dGVzdC0xMjM'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-123'
    var_10 = b'!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'dGVzdA=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test'
    var_14 = b'This is a longer test string with multiple characters!'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)



# Parsed testcases at query #106
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
    var_4 = 'V29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YWI='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = 'YWI'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'ab'
    var_18 = 'YWJj'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abc'
    var_20 = 'Pz4_'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'?>?'
    var_22 = module_0.base64_decode(var_20)
    var_23 = 'Pz4/'
    var_24 = module_0.base64_decode(var_23)
    var_25 = '//8='
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'\xff\xff'
    var_27 = '__8='
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'\xff\xff'
    var_29 = '!!!'
    var_30 = module_0.base64_decode(var_29)
    var_31 = ' '
    var_32 = module_0.base64_decode(var_31)
    var_33 = 'abcde'
    var_34 = module_0.base64_decode(var_33)
    var_35 = 'test@'
    var_36 = module_0.base64_decode(var_35)
    var_37 = bytes(var_35)
    var_38 = module_1.b64encode(var_37)
    var_39 = b'='
    var_40 = b'\x00\x01\x02\xff\xfe'
    var_41 = module_1.b64encode(var_40)
    var_42 = b'='
    var_43 = module_0.base64_decode(var_10)
    assert var_43 == b'a'
    var_44 = module_0.base64_decode(var_12)
    assert var_44 == b'a'
    var_45 = 'YQ==='
    var_46 = module_0.base64_decode(var_45)
    assert var_46 == b'a'



# Parsed testcases at query #107
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'd29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'world'
    var_4 = 'aGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = 'd29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'world'
    var_8 = b'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'Pj4+Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>>>'
    var_14 = 'PDw8PA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<<<<'
    var_16 = b'\x00\x01\x02\xff\xfe\xfd'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'héllo wörld'
    var_20 = module_1.encode()
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    var_23 = module_1.encode()
    var_24 = '!!!invalid!!!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'aGVsbG8='
    var_27 = 2
    var_28 = var_26 * var_27
    var_29 = module_0.base64_decode(var_28)



# Parsed testcases at query #108
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test string'
    var_6 = b'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'aGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'YQ=='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'a'
    var_23 = b'YQ'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'a'



# Parsed testcases at query #109
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'not@base64!'
    var_14 = module_0.base64_decode(var_13)



# Parsed testcases at query #110
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'VGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Test'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'VGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = '_-xr'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xeb'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'abc'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #111
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b'aGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'_-A'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'\xfb\xff'
    var_13 = 'aGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello'
    var_15 = module_0.base64_decode(var_5)
    assert var_15 == b'hello'
    var_16 = b'!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVs\x00bG8='
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'AQ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\x01'
    var_22 = b'a'
    var_23 = b'ab'
    var_24 = b'abc'
    var_25 = b'abcd'
    var_26 = b'\x00\x01\x02'
    var_27 = 256
    var_28 = range(var_27)
    var_29 = bytes(var_28)
    var_30 = [var_9, var_22, var_23, var_24, var_25, var_26, var_29]
    var_31 = module_0.base64_decode(var_1)



# Parsed testcases at query #112
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'YQ=='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'a'
    var_7 = 'YWI='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'ab'
    var_9 = 'YWJj'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'abc'
    var_11 = b'\xff\xfb\x00\x01'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xfb\x00\x01'
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b'dGVzdA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'abc'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #113
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'test bytes'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test bytes'
    var_6 = 'hello'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b''
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'data with \x00 null byte'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'YWJjZA'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'aGVsbG8='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'hello'
    var_21 = 'aGVsbG8='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'hello'



# Parsed testcases at query #114
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '_-x'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xfb\xd7'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8\x80'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'



# Parsed testcases at query #115
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'Pj4-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>>'
    var_14 = b'Pj4-Pg'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'>>>'
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'not base64 at all'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'cHLDpG3DpMOk'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'prämä'
    var_28 = 'utf-8'
    var_29 = module_1.encode(var_28)



# Parsed testcases at query #116
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'hello world'
    var_5 = module_0.base64_encode(var_4)
    var_6 = module_0.base64_decode(var_5)
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'\x00\x01\x02\xff'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = b'aGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'aGVsbG8'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = b'aGVsbG8===='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'



# Parsed testcases at query #117
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dGVzdC11cmw'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-url'
    var_10 = 'MA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'0'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'not-base64!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'dGVzdA==\x80'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'



# Parsed testcases at query #118
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '_-w='
    var_9 = module_0.base64_decode(var_8)
    var_10 = len(var_9)
    var_11 = 'YQ=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'a'
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'not base64!'
    var_16 = module_0.base64_decode(var_15)



# Parsed testcases at query #119
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = 'ascii'
    var_6 = module_1.decode(var_5)
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test string'
    var_8 = b'YQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = b'YWI'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = '你好'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'\x00'
    var_19 = b'\xff'
    var_20 = 10
    var_21 = var_19 * var_20
    var_22 = b'Hello, World!'
    var_23 = 256
    var_24 = range(var_23)
    var_25 = bytes(var_24)
    var_26 = [var_12, var_18, var_21, var_22, var_25]
    var_27 = module_0.base64_decode(var_1)
    var_28 = b'YQ=='
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'a'
    var_30 = b'YWI='
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'ab'



# Parsed testcases at query #120
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'aGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = 'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'dGVzdA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = 'Pj4-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'>>?>'
    var_14 = 'invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'dGVzdA==\x80'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'



# Parsed testcases at query #121
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'test string'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test string'
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'\x00\xff\x7f\x80'
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'AAAA!AAA'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'héllo'
    var_27 = 'aMOp'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'h\xc3\xa9'



# Parsed testcases at query #122
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = 'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = module_0.base64_decode(var_2)
    assert var_6 == b'test'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'\x00\x01\x02\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = '_-w'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xfb\xc0'
    var_16 = 'YQ'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = 'YWI'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'ab'
    var_20 = 'YWJj'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'abc'
    var_22 = 'YWJjZA'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'abcd'
    var_24 = '!!!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = b'\xff\xff\xff\xff'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'not valid base64!'
    var_29 = module_0.base64_decode(var_28)



# Parsed testcases at query #123
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'\x00\x01\x02\xff\xfe'
    var_11 = module_1.b64encode(var_10)
    var_12 = b'='
    var_13 = b'YQ=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'a'
    var_15 = b'YWI='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'ab'
    var_17 = b'YWJj'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'abc'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'\xff\xff\xff\xff'
    var_22 = module_0.base64_decode(var_21)



# Parsed testcases at query #124
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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dGVzdC11cmw'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-url'
    var_10 = 'dGVzdF9zdHJpbmc='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test_string'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'ÿ'
    var_15 = var_12 + var_14
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'



# Parsed testcases at query #125
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'world'
    var_4 = b''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'd29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'world'
    var_10 = b'_-x.'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xeb'
    var_12 = b'_-w.'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xeb'
    var_14 = b'YQ=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = b'YWI'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'ab'
    var_18 = b'YWJj'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abc'
    var_20 = 'héllo'
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'utf-8'
    var_24 = module_1.encode(var_23)
    var_25 = b'!!!invalid!!!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'SGVsbG8='
    var_28 = b'\x00'
    var_29 = var_27 + var_28
    var_30 = module_0.base64_decode(var_29)



# Parsed testcases at query #126
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = 'text input'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'text input'
    var_9 = b''
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 256
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'\x00\xff\x7f\x80'
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'test'
    var_21 = module_0.base64_decode(var_18)
    var_22 = b'!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'invalid base64!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'héllo wörld'
    var_27 = module_0.base64_encode(var_26)
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'utf-8'
    var_30 = module_1.encode(var_29)
    var_31 = '你好世界'
    var_32 = module_0.base64_encode(var_31)
    var_33 = module_0.base64_decode(var_32)
    var_34 = module_1.encode(var_29)



# Parsed testcases at query #127
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8gd29ybGQ='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b'aGVsbG8gd29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = 'aGVsbG8gd29ybGQ'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello world'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'\x00\x01\x02\xff\xfe'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'aGVsbG8gd29ybGQ=\x80'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello world'



# Parsed testcases at query #128
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'YQ=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = 'YWI='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'ab'
    var_11 = 'YWJj'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = 'Pj4-Pg=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'>>>>'
    var_15 = b'aGVsbG8='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'hello'
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = '//8='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\xff\xff'
    var_21 = '__8='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'\xff\xff'



# Parsed testcases at query #129
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = '_-x='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'\xff\xf1'
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'YQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = 'YWI'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ab'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = 'AAAA'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\x00\x00\x00'
    var_18 = 'SGVsbG8=\x80'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = '!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'ABCD\x01\x02'
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #130
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b'a'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = b'ab'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'ab'
    var_15 = b'abc'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = b'abcd'
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'abcd'
    var_21 = ''
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b''
    var_23 = b''
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b''
    var_25 = b'\x00\x01\x02\xff\xfe\xfd'
    var_26 = module_0.base64_encode(var_25)
    var_27 = module_0.base64_decode(var_26)
    var_28 = b'!!!invalid!!!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = b'\xff\xff\xff\xff'
    var_31 = module_0.base64_decode(var_30)
    var_32 = module_0.base64_decode(var_30)
    assert var_32 == b'hello'
    var_33 = b'Pj4-Pj4-'
    var_34 = module_0.base64_decode(var_33)
    assert var_34 == b'>>>>'



# Parsed testcases at query #131
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\xff\xfb'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xfb'
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'abc123!@#'
    var_22 = module_0.base64_decode(var_21)



# Parsed testcases at query #132
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = 'Pj4_Pz8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'>>???\x00'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'WA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'X'
    var_10 = b'dGVzdA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = 'w7zDtsO4w6I='
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'üöäß'
    var_15 = 'utf-8'
    var_16 = module_1.encode(var_15)
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'dGVzdA==\x80'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'test'



# Parsed testcases at query #133
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8gV29ybGQ'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8gV29ybGQh'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World!'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = b''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'_-4'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xfb\xef'
    var_12 = b'!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'\xff\xfe'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #134
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test string'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'\x00\x01\x02\xff'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\x00\x01\x02\xff'
    var_12 = b'a'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'a'
    var_15 = b'dGVzdA=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'test'
    var_17 = b'!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'\xff\xfe\xff\xfd'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\xff\xfe\xff\xfd'



# Parsed testcases at query #135
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = '_-w='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xec'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'test\x00'
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #136
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVs'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hel'
    var_7 = b'aGVsbG8_d29ybGQ='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello?world'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'YQ=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'a'
    var_15 = b'YWI='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'ab'
    var_17 = b'YWJj'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'abc'
    var_19 = b'YWJjZA=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'abcd'
    var_21 = 256
    var_22 = range(var_21)
    var_23 = bytes(var_22)
    var_24 = module_0.base64_encode(var_23)
    var_25 = module_0.base64_decode(var_24)
    var_26 = b'!!!invalid!!!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = b'aGVsbG8'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'héllo'
    var_31 = module_0.base64_encode(var_30)
    var_32 = module_0.base64_decode(var_31)
    var_33 = 'utf-8'
    var_34 = module_1.encode(var_33)



# Parsed testcases at query #137
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'VGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Test'
    var_6 = 'VGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'aGVsbG8_d29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello?world'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'Hello World'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''



# Parsed testcases at query #138
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b'\x00\x01\x02\xff'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'\x00\x01\x02\xff'
    var_9 = b'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'aGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'Pj4_Pz8-'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'>>???\xff'
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''
    var_19 = b'a'
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'a'
    var_22 = b'x'
    var_23 = 1000
    var_24 = var_22 * var_23
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)



# Parsed testcases at query #139
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = 'aGVsbG8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'Pj4-Pg=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'>>>'
    var_17 = b'test'
    var_18 = b'dGVzdA=='
    var_19 = (var_17, var_18)
    var_20 = b'a'
    var_21 = b'YQ=='
    var_22 = (var_20, var_21)
    var_23 = b'ab'
    var_24 = b'YWI='
    var_25 = (var_23, var_24)
    var_26 = b'abc'
    var_27 = b'YWJj'
    var_28 = (var_26, var_27)
    var_29 = 256
    var_30 = range(var_29)
    var_31 = bytes(var_30)
    var_32 = range(var_29)
    var_33 = bytes(var_32)
    var_34 = module_0.base64_encode(var_33)
    var_35 = (var_31, var_34)
    var_36 = [var_19, var_22, var_25, var_28, var_35]
    var_37 = b'!!!invalid!!!'
    var_38 = module_0.base64_decode(var_37)
    var_39 = 'invalid!'
    var_40 = module_0.base64_decode(var_39)
    var_41 = module_0.base64_decode(var_13)
    assert var_41 == b''
    var_42 = 'dGVzdA=='
    var_43 = module_0.base64_decode(var_42)
    assert var_43 == b'test'
    var_44 = b'dGVzdA===='
    var_45 = module_0.base64_decode(var_44)
    assert var_45 == b'test'



# Parsed testcases at query #140
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'test data'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'test data'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello world'
    var_6 = 'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = 'dGVzdA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'aGVsbG8t'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello-'
    var_14 = 'aGVsbG9f'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello_'
    var_16 = b'dGVzdA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'test\x00data'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #141
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'aGVsbG8-d29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello>world'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'dGVzdGluZw=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'testing'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'hello world'
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #142
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b'a'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = b'ab'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = b'hello\nworld'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello\nworld'
    var_15 = b'\x00\x01\x02\xff'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\x00\x01\x02\xff'
    var_18 = '!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b''
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b''
    var_23 = b'test'
    var_24 = module_0.base64_encode(var_23)
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'test'



# Parsed testcases at query #143
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = 'dGVzdC11cmw='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-url'
    var_10 = 'AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8='
    var_11 = module_0.base64_decode(var_10)
    var_12 = 32
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = 'invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'a'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'dGVzdA\x80'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'test'



# Parsed testcases at query #144
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = 'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'dGVzdA=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'test'
    var_11 = '_-w='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'\xff\xc0'
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'SGVsbG8gV29ybGQ='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello World'



# Parsed testcases at query #145
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'AA=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'\x00'
    var_13 = 256
    var_14 = range(var_13)
    var_15 = bytes(var_14)
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'\xff\xfe\xfd'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #146
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'dGVzdA=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'test'
    var_11 = 'dGVzdA=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'test'
    var_13 = b'YQ=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'a'
    var_15 = b'YWI='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'ab'
    var_17 = b'YWJj'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'abc'
    var_19 = b'YWJjZA=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'abcd'
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'aGVsbG8$'
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'a'
    var_26 = b'ab'
    var_27 = b'abc'
    var_28 = b'abcd'
    var_29 = b'test data'
    var_30 = b'12345'
    var_31 = b'\x00\x01\x02'
    var_32 = [var_7, var_25, var_26, var_27, var_28, var_29, var_30, var_31]
    var_33 = module_0.base64_decode(var_1)



# Parsed testcases at query #147
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'test'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'aGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'\xff\xfe\x00'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'aGVsbG8=ÿ'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'hello'



# Parsed testcases at query #148
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8gd29ybGQ'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'\x00\x01\x02\xff'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVsbG8='
    var_19 = b'\x00'
    var_20 = var_18 + var_19
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'YQ'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'a'
    var_24 = b'YWI'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'ab'
    var_26 = b'YWJj'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'abc'
    var_28 = module_0.base64_decode(var_11)
    assert var_28 == b'hello'



# Parsed testcases at query #149
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b''
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'a'
    var_7 = b'ab'
    var_8 = b'abc'
    var_9 = b'abcd'
    var_10 = b'test data here'
    var_11 = [var_6, var_7, var_8, var_9, var_10]
    var_12 = module_0.base64_decode(var_2)
    var_13 = 'aGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello'
    var_15 = b'test!@#$%^&*()'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    var_18 = 256
    var_19 = range(var_18)
    var_20 = bytes(var_19)
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    var_23 = '!!!invalid!!!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'YQ'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'a'
    var_27 = 'YWI'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'ab'
    var_29 = 'aGVsbG8=\x80'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'hello'
    var_31 = ''
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b''



# Parsed testcases at query #150
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8gd29ybGQ'
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'aGVsbG8gd29ybGQ='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello world'
    var_9 = b'test data with +/='
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = b'!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'aGVsbG8'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = 'aGVsbG8gd29ybGQ=\x80'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello world'
    var_18 = module_0.bytes_to_int(var_10)



# Parsed testcases at query #151
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b'\xff\xfb\x00'
    var_8 = module_0.base64_encode(var_7)
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xfb\x00'
    var_10 = 'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'aGVsbG8===='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVs$G8='
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #152
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'test'
    var_4 = module_1.urlsafe_b64encode(var_3)
    var_5 = module_1.decode()
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'test'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'YQ=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'a'
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'\xff\xfe'
    var_16 = module_1.urlsafe_b64encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xff\xfe'



# Parsed testcases at query #153
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8gd29ybGQ='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'aA=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'h'
    var_9 = 'YQ'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = 'YQ='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'a'
    var_13 = b'\xfb\xff\xff\xff'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xfb\xff\xff\xff'
    var_16 = b'aGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = 256
    var_19 = range(var_18)
    var_20 = bytes(var_19)
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    var_23 = '!!!invalid!!!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'aGVs\tbG8='
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'aGVs\x80bG8='
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'hello'



# Parsed testcases at query #154
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dGVzdC91cmw='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test/url'
    var_10 = 'dGVzdC91cmw'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test/url'
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = 'YWI'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'ab'
    var_18 = 'YWJj'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abc'
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'abc123!@#'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'áéíóú'
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #155
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'aGVsbG8+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello>'
    var_12 = 'aGVsbG8/'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello?'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8\x00'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'a'
    var_19 = 1000
    var_20 = var_18 * var_19
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    var_23 = module_1.encode()
    var_24 = b'\x00\x01\x02\xff\xfe\xfd'
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)



# Parsed testcases at query #156
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'aGVsbG8_d29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello?world'
    var_14 = 'aGVsbG8td29ybGQ='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello-world'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVs\nbG8='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'é'
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #157
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\x00\x01\x02\xff'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\x00\x01\x02\xff'
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'aGVsbG8\xff'
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #158
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'YQ=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = 'YWI='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'ab'
    var_11 = 'YWJj'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = b'\xff\xfe\xfd\xfc'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xff\xfe\xfd\xfc'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVsbG8='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'



# Parsed testcases at query #159
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = '_-w='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xeb'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'a'
    var_17 = 1000
    var_18 = var_16 * var_17
    var_19 = module_1.b64encode(var_18)
    var_20 = module_1.decode()
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8=\x00'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'



# Parsed testcases at query #160
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'Test base64_decode function with various inputs.'
    var_1 = b'hello world'
    var_2 = module_0.base64_encode(var_1)
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello world'
    var_4 = b''
    var_5 = module_0.base64_encode(var_4)
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = b'\x00\x01\x02\xff\xfe'
    var_8 = module_0.base64_encode(var_7)
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\x00\x01\x02\xff\xfe'
    var_10 = b'test'
    var_11 = module_1.urlsafe_b64encode(var_10)
    var_12 = b'='
    var_13 = module_0.base64_decode(var_8)
    assert var_13 == b'test'
    var_14 = 'unicode test'
    var_15 = module_0.base64_encode(var_14)
    var_16 = 'ascii'
    var_17 = module_1.decode(var_16)
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'unicode test'
    var_19 = b'\xff\xfe\x00\x01'
    var_20 = module_0.base64_encode(var_19)
    var_21 = 'latin-1'
    var_22 = module_1.decode(var_21)
    var_23 = module_0.base64_decode(var_22)
    var_24 = '!!!invalid!!!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 123
    var_27 = module_0.base64_decode(var_26)
    var_28 = b'A'
    var_29 = 1000
    var_30 = var_28 * var_29
    var_31 = module_0.base64_encode(var_30)
    var_32 = module_0.base64_decode(var_31)
    var_33 = 256
    var_34 = range(var_33)
    var_35 = bytes(var_34)
    var_36 = module_0.base64_encode(var_35)
    var_37 = module_0.base64_decode(var_36)



# Parsed testcases at query #161
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'VGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Test'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'VGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '_-x'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xeb'
    var_12 = '_-x='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xeb'
    var_14 = '_-x=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xff\xeb'
    var_16 = b'SGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = 'YQ=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'a'
    var_20 = 'YQ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'a'
    var_22 = 'dGVzdGluZw=='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'testing'
    var_24 = 'dGVzdGluZw'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'testing'
    var_26 = b'\x00\x01\x02\xff'
    var_27 = module_1.b64encode(var_26)
    var_28 = module_1.decode()
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'\x00\x01\x02\xff'



# Parsed testcases at query #162
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'\x00\x01\x02\xff'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\x00\x01\x02\xff'
    var_12 = b'aGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = b'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = b'aGVsbG8=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = b'test'
    var_19 = module_1.b64encode(var_18)
    var_20 = b'='
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'hello world'
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'MixedCase123'
    var_26 = module_0.base64_encode(var_25)
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'MixedCase123'
    var_28 = b'1234567890'
    var_29 = module_0.base64_encode(var_28)
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'1234567890'
    var_31 = '你好'
    var_32 = module_0.base64_decode(var_31)



# Parsed testcases at query #163
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'V29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'World'
    var_8 = '_-w='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xeb'
    var_10 = '_-w'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xeb'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = '=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b'SGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = 'YQ=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'a'
    var_20 = 'YWI='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'ab'
    var_22 = 'YWJj'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'abc'
    var_24 = '+/8='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'\xfb\xff'
    var_26 = '!!!invalid!!!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'abcde'
    var_29 = module_0.base64_decode(var_28)



# Parsed testcases at query #164
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test data'
    var_6 = 'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = 'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'dGVzdA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'
    var_18 = '_-w'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'\xff\xec'



# Parsed testcases at query #165
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test string'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = 'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = 'aGVsbG8=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'YQ=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'a'
    var_21 = 'YQ'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'a'
    var_23 = 'ISFAIyQlXiYqKCk='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'!@#$%^&*()'
    var_25 = '!!!invalid!!!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'dGVzdA=='
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'test'
    var_29 = 'héllo'
    var_30 = module_0.base64_encode(var_29)
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'utf-8'
    var_33 = module_1.encode(var_32)



# Parsed testcases at query #166
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = 'utf-8'
    var_7 = module_1.encode(var_6)
    var_8 = b'a'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = b''
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'test+data/with=special_chars'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'\xff\xfe\xfd'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'binary data \x00\x01\x02'
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'binary data \x00\x01\x02'
    var_24 = b'x'
    var_25 = module_0.base64_decode(var_1)
    var_26 = 'Hello, World! 123'
    var_27 = module_0.base64_encode(var_26)
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'ascii'
    var_30 = module_1.encode(var_29)



# Parsed testcases at query #167
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'='
    var_4 = 'aGVsbG8gd29ybGQ'
    var_5 = module_0.base64_decode(var_4)
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'WA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'X'
    var_12 = b'\xff\xfe\x00\x01'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'aGVsbG8=invalid'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'aGVsbG8gd29ybGQ\x00\x01\x02'
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #168
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'Hello, World!'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'test'
    var_4 = module_1.urlsafe_b64encode(var_3)
    var_5 = module_1.decode()
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'test'
    var_7 = b'SGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'SGVsbG8=\x80'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'



# Parsed testcases at query #169
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Test base64_decode function.'
    var_1 = b'hello world'
    var_2 = module_0.base64_encode(var_1)
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello world'
    var_4 = 'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = 'aGVsbG8+'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello>'
    var_8 = module_0.base64_decode(var_4)
    assert var_8 == b'hello'
    var_9 = b'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = '8J+YgQ=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\xf0\x9f\x98\x81'
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'aGVs$G8='
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'aGVsbG8=\xc3\xa9'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'hello'
    var_21 = b''
    var_22 = b'test'
    var_23 = 256
    var_24 = range(var_23)
    var_25 = bytes(var_24)
    var_26 = b'data with spaces and symbols!@#$%'
    var_27 = [var_21, var_22, var_18, var_25, var_26]
    var_28 = module_0.base64_decode(var_2)



# Parsed testcases at query #170
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVs'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hel'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'dGVzdC0t'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'test--'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'aGVsbG8\xff'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'hello'
    var_17 = b'hello'
    var_18 = (var_13, var_17)
    var_19 = b'aGVsbG8'
    var_20 = (var_19, var_17)
    var_21 = b'd29ybGQ='
    var_22 = b'world'
    var_23 = (var_21, var_22)
    var_24 = b'd29ybGQ'
    var_25 = (var_24, var_22)
    var_26 = (var_9, var_9)
    var_27 = b'YQ=='
    var_28 = b'a'
    var_29 = (var_27, var_28)
    var_30 = b'YWI='
    var_31 = b'ab'
    var_32 = (var_30, var_31)
    var_33 = b'YWJj'
    var_34 = b'abc'
    var_35 = (var_33, var_34)
    var_36 = [var_18, var_20, var_23, var_25, var_26, var_29, var_32, var_35]
    var_37 = module_0.base64_decode(var_1)
    var_38 = module_0.base64_encode(var_0)
    var_39 = module_0.base64_decode(var_38)



# Parsed testcases at query #171
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'dGVzdGluZw=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'testing'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'aGVsbG8$'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'aGVs bG8='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'hello'



# Parsed testcases at query #172
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b'Pj4_Pj4_'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'>>?>'
    var_11 = b'!!!invalid!!!'
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'aGVsbG8=\x80'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello'
    var_15 = b'a'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = b'ab'
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'ab'
    var_21 = b'abc'
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'abc'
    var_24 = b'abcd'
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'abcd'
    var_27 = b'\x00\x01\x02\xff'
    var_28 = module_0.base64_encode(var_27)
    var_29 = module_0.base64_decode(var_28)



# Parsed testcases at query #173
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'aGVsbG8_d29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello?world'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'aGVsbG8td29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello-world'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8=\x80'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'



# Parsed testcases at query #174
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = 'dGVzdA'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'test'
    var_4 = b'dGVzdA=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '_-x'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xeb'
    var_10 = 'dGVzdA!@#'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #175
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'héllo wörld'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    var_6 = 'utf-8'
    var_7 = module_1.encode(var_6)
    var_8 = b''
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'dGVzdA=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'test'
    var_13 = b'dGVzdA'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'test'
    var_15 = b'a'
    var_16 = b'ab'
    var_17 = b'abc'
    var_18 = b'abcd'
    var_19 = b'abcde'
    var_20 = [var_15, var_16, var_17, var_18, var_19]
    var_21 = module_0.base64_decode(var_4)
    var_22 = 256
    var_23 = range(var_22)
    var_24 = bytes(var_23)
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)
    var_27 = b'!!!invalid!!!'
    var_28 = module_0.base64_decode(var_27)
    var_29 = b'not base64!'
    var_30 = module_0.base64_decode(var_29)
    var_31 = 'aGVsbG8='
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'hello'



# Parsed testcases at query #176
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Test base64_decode function.'
    var_1 = b'SGVsbG8gV29ybGQ='
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'Hello World'
    var_3 = 'SGVsbG8gV29ybGQ='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'Hello World'
    var_5 = 'SGVsbG8gV29ybGQ'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'Hello World'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'YWJjZGVmZw=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'abcdefg'
    var_11 = 'MTIzNDU2'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'123456'
    var_13 = 'ISFAQCMkJV4mKigp'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'!@#$%^&*()'
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'not valid base64'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #177
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test string'
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'a'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = b'\x00\x01\x02'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\x00\x01\x02'
    var_15 = b'\xff\xfe\xfd'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xff\xfe\xfd'
    var_18 = b'!!!invalid!!!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'YWJj'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'test data'
    var_23 = module_0.base64_encode(var_22)
    var_24 = module_1.encode()
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_23)
    var_27 = module_0.base64_decode(var_25)



# Parsed testcases at query #178
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b'aGVsbG8td29ybGQ'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello-world'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'aGVsbG8='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'aGVsbG8v_d29ybGQ'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello/ world'
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'invalid\x00data'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #179
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'a'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'a'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'test'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'test'
    var_11 = b'\x00\x01\x02'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\x00\x01\x02'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'aGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = b'aGVsbG8'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'
    var_20 = b'x'
    var_21 = 100
    var_22 = var_20 * var_21
    var_23 = module_0.base64_encode(var_22)
    var_24 = module_0.base64_decode(var_23)



# Parsed testcases at query #180
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'test data'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'test data'
    var_3 = 'hello world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello world'
    var_6 = b'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b'dGVzdA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'aGVsbG8tX3dvcmxk'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello_world'
    var_14 = b'aGVsbG8rL3dvcmxk'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello+/world'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'\xff\xfe\x00\x01'
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\xff\xfe\x00\x01'



# Parsed testcases at query #181
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b'aGVsbG8'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 256
    var_10 = range(var_9)
    var_11 = bytes(var_10)
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'aGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVs\xffbG8='
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'Zg=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'f'
    var_22 = module_0.base64_decode(var_1)



# Parsed testcases at query #182
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = 'world'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'world'
    var_6 = b'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'\x00\x01\x02\xff\xfe'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'aGVsbG8$'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'a'
    var_20 = b'ab'
    var_21 = b'abc'
    var_22 = b'abcd'
    var_23 = b'test data'
    var_24 = 256
    var_25 = range(var_24)
    var_26 = bytes(var_25)
    var_27 = b'\x00'
    var_28 = 10
    var_29 = var_27 * var_28
    var_30 = [var_10, var_19, var_20, var_21, var_22, var_23, var_26, var_29]
    var_31 = module_0.base64_encode(var_12)
    var_32 = module_0.base64_decode(var_31)



# Parsed testcases at query #183
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'AAECAwQFBgcI'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\x00\x01\x02\x03\x04\x05\x06\x07\x08'
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = b'dGVzdA'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test'
    var_12 = '_-x4Zw=='
    var_13 = module_0.base64_decode(var_12)
    var_14 = len(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'abc def'
    var_18 = module_0.base64_decode(var_17)
    var_19 = module_0.base64_decode(var_4)
    assert var_19 == b''



# Parsed testcases at query #184
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'VGVzdA=='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Test'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'VGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Test'
    var_8 = b'hello+world'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello+world'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b'SGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'not-base64'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #185
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = b'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dGVzdC11cmwtc2FmZQ=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-url-safe'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'abc'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'dGVzdA=='
    var_15 = 'ascii'
    var_16 = 'ignore'
    var_17 = module_0.base64_decode(var_14)
    assert var_17 == b'test'



# Parsed testcases at query #186
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = 'test string'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_1.decode()
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'test string'
    var_7 = 'aGVsbG8='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = 'aGVsbG8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b'\xfb\xff\xff\xff\xff'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'\x00\x01\x02\xff\xfe'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = '!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'aGVsbG8$'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'dGVzdA=='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'test'
    var_25 = 256
    var_26 = range(var_25)
    var_27 = bytes(var_26)
    var_28 = module_0.base64_encode(var_27)
    var_29 = module_0.base64_decode(var_28)



# Parsed testcases at query #187
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = 'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = b'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'aGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8gV29ybGQ=\x80'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello World'



