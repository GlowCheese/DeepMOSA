####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
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
    var_12 = b'PDw_Pz8-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<<??>>'
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'dGVzdA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'test'
    var_18 = 'dGVzdA=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'test'



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'V29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'World'
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = '_-w='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xec'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8=\x80'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'aGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = b'aGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = 'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'dGVzdC11cmw'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test-url'
    var_10 = 'dGVzdC11cmw='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test-url'
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b'YQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = 'YQ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'a'
    var_20 = b'dGVzdC8_'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'test/?'
    var_22 = 'dGVzdC8_'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'test/?'
    var_24 = b'dGVzdC1f'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'test-_'
    var_26 = 'dGVzdC1f'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'test-_'
    var_28 = b'!!!invalid!!!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = '!!!invalid!!!'
    var_31 = module_0.base64_decode(var_30)
    var_32 = b'aGVsbG8='
    var_33 = 2
    var_34 = var_32 * var_33
    var_35 = module_0.base64_decode(var_34)
    var_36 = module_0.base64_decode(var_34)
    assert var_36 == b'hello'
    var_37 = module_0.base64_decode(var_6)
    assert var_37 == b'hello'



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'aGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = 'dGVzdC11cmwtc2FmZQ=='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test-url-safe'
    var_6 = b'd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'YQ=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'a'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'dGVzdC1f'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test-'



# Parsed testcases at query #5
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
    var_16 = 'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = 'MTIz'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'123'
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = '\x00\x01\x02'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'ñ'
    var_25 = var_22 + var_24
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'
    var_27 = b'x'
    var_28 = 100
    var_29 = var_27 * var_28
    var_30 = module_0.base64_encode(var_29)
    var_31 = module_0.base64_decode(var_30)



# Parsed testcases at query #6
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
    var_8 = b'aGVsbG8td29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello-world'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'YQ=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = b'\x00\x01\x02\xff'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8=\x80'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = b'dGVzdA=='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'test'



# Parsed testcases at query #7
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
    var_10 = '_-x5Zw=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xbcye'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8gV29ybGQ='
    var_15 = '!'
    var_16 = var_14 + var_15
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'YQ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'a'



# Parsed testcases at query #8
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
    var_6 = b'dGVzdA'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b'hello world'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello world'
    var_11 = 'aGVsbG8gd29ybGQ'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello world'
    var_13 = '!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'not base64 chars!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'aGVsbG8='
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'aGVsbG8\x80d29ybGQ'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'hello world'



# Parsed testcases at query #9
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
    var_6 = 'Pj4_Pz8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'>>???'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = '_-x'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xeb'
    var_12 = '!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'not-base64'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 100
    var_17 = b''
    var_18 = b'a'
    var_19 = b'ab'
    var_20 = b'abc'
    var_21 = b'abcd'
    var_22 = 256
    var_23 = range(var_22)
    var_24 = bytes(var_23)
    var_25 = [var_17, var_18, var_19, var_20, var_21, var_24]



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.encoding as module_0

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
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b't'
    var_10 = b'\x00\x01\x02\x03'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    var_13 = b'hello+world'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello+world'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'dGVzdA@'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'dGVzdA== '
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'test'



# Parsed testcases at query #11
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
    var_9 = b'dGVzdC11cmwtc2FmZQ=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'test-url-safe'
    var_11 = b'Ynl0ZXM='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'bytes'
    var_13 = 'dGVzdC1zdHJpbmc='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'test-string'
    var_15 = 'dGVzdC1zdHJpbmc'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'test-string'
    var_17 = b'!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'dGVzdA=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'test'
    var_21 = 256
    var_22 = range(var_21)
    var_23 = bytes(var_22)
    var_24 = module_0.base64_encode(var_23)
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #12
#--------------------------


import base64 as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.b64encode(var_0)
    var_2 = module_1.base64_decode(var_1)
    var_3 = b'test data with +/='
    var_4 = module_0.urlsafe_b64encode(var_3)
    var_5 = b'='
    var_6 = module_1.base64_decode(var_1)
    assert var_6 == b'test'
    var_7 = b'test'
    var_8 = module_0.urlsafe_b64encode(var_7)
    var_9 = 'aGVsbG8='
    var_10 = module_1.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b''
    var_12 = module_1.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b'!!!invalid!!!'
    var_14 = module_1.base64_decode(var_13)
    var_15 = 'aA=='
    var_16 = module_1.base64_decode(var_15)
    assert var_16 == b'h'



# Parsed testcases at query #13
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
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '8J+Zgw=='
    var_9 = module_0.base64_decode(var_8)
    var_10 = '🎃'
    var_11 = module_1.encode()
    var_12 = 'Kysv'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'++/'
    var_14 = b'!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'Invalid base64!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'dGVzdA=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'test'
    var_20 = 'dGVzdA==='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'test'



# Parsed testcases at query #14
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
    var_9 = b'aGVsbG8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'aGVsbG8==='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = b'!!!invalid!!!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'not-valid-base64!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'dGVzdA=='
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #15
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
    var_5 = b'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'aGVsbG8'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = b'\xff\xfe\xff'
    var_12 = module_1.b64encode(var_11)
    var_13 = 'ascii'
    var_14 = module_1.decode(var_13)
    var_15 = '+'
    var_16 = '-'
    var_17 = '/'
    var_18 = '_'
    var_19 = module_1.encode(var_13)
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\xff\xfe\xff'
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'abc def'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 1
    var_26 = 256
    var_27 = var_24 % var_26
    var_28 = 2
    var_29 = var_6 % var_26
    var_30 = bytes(var_8)
    var_31 = module_0.base64_encode(var_30)
    var_32 = module_0.base64_decode(var_31)



# Parsed testcases at query #16
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
    var_11 = b'\xff\xfe\x00\x01'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = bytes(var_2)
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'aGVsbG8=\x80'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'hello'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'test data'
    var_4 = module_0.base64_encode(var_3)
    var_5 = b'='
    var_6 = b'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'aGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello'
    var_14 = 256
    var_15 = range(var_14)
    var_16 = bytes(var_15)
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'_-A'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'\xff\xe8'



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'dGVzdA=='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'test'
    var_2 = b'aGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = b''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = 'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = b'dGVzdA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = b'aGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'Pz4_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'?>?'
    var_14 = b'Pz4-'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'?>?'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'ascii'
    var_19 = 'ignore'
    var_20 = module_0.base64_decode(var_6)
    assert var_20 == b'test'



# Parsed testcases at query #19
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
    var_8 = b'hello+world'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = 256
    var_12 = range(var_11)
    var_13 = bytes(var_12)
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'_-9-Zg=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xff\xfe\xfd'
    var_18 = module_0.base64_decode(var_2)
    assert var_18 == b'test'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'not valid base64'
    var_22 = module_0.base64_decode(var_21)



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
    var_4 = 'aGVsbG8td29ybGQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello-world'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = 'dA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b't'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'dGVzdA\x80\x81'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test'



# Parsed testcases at query #21
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
    var_13 = 'aGVsbG8'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello'
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = ''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''
    var_19 = b'\xff\xfe\x00\x01'
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'hello world'
    var_25 = module_0.base64_decode(var_24)
    var_26 = module_0.base64_decode(var_9)
    assert var_26 == b'a'
    var_27 = b'YWI'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'ab'
    var_29 = b'YWJj'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'abc'
    var_31 = b'YWJjZA=='
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'abcd'



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = 'aGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = b'aGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dGVzdC91cmw='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test/url'
    var_10 = b'dGVzdC91cmw'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'test/url'
    var_12 = 256
    var_13 = range(var_12)
    var_14 = bytes(var_13)
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'MTIzNDU='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'12345'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'aGVsbG8'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 123
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'utf-16'
    var_26 = module_1.encode(var_25)
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'hello'



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = b'd29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'world'
    var_4 = 'dGVzdA'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'dGhpcyBpcyBhIHRlc3Q='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'this is a test'
    var_10 = 'invalid!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = module_0.base64_decode(var_10)
    assert var_12 == b'hello'



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
    var_5 = b'YQ=='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'a'
    var_7 = b'YWI='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'ab'
    var_9 = b'YWJj'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'abc'
    var_11 = 256
    var_12 = range(var_11)
    var_13 = bytes(var_12)
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = ''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = b'YQ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'a'
    var_22 = b'YWI'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'ab'
    var_24 = b'!!!invalid!!!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'not base64!!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = b'\xff\xfe\xfd'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'aGVsbG8='
    var_31 = module_0.base64_decode(var_30)



# Parsed testcases at query #25
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
    var_8 = 'dGVzdC1fdXJsX3NhZmU='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test_url_safe'
    var_10 = 256
    var_11 = range(var_10)
    var_12 = bytes(var_11)
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'YQ=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'a'
    var_19 = 'ISQlJiYnKCk='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b"!$%&&'()"



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
    var_6 = b'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = b'aGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'dGVzdA'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test'
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b'YQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = b'_-w='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'\xfe\xec'
    var_20 = b'!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'aGVsbG8!!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'\xff\xfe\xfd'
    var_25 = module_0.base64_decode(var_24)
    var_26 = b'a'
    var_27 = b'ab'
    var_28 = b'abc'
    var_29 = b'test data'
    var_30 = b'\x00\x01\x02'
    var_31 = b'1234567890'
    var_32 = b'!@#$%^&*()'
    var_33 = [var_14, var_26, var_27, var_28, var_29, var_30, var_31, var_32]
    var_34 = module_0.base64_decode(var_1)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.encoding as module_0

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
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'Pj4_Pz8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>???'
    var_10 = b'AQID'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\x01\x02\x03'
    var_12 = b'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = b'YQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'\xff\xfe\xfd'
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #28
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
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = b'\x00\x01\x02\xff'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'\x00\x01\x02\xff'
    var_15 = b'hello'
    var_16 = module_0.base64_encode(var_15)
    var_17 = 'aGVsbG8'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'hello'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'not valid base64'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 256
    var_24 = range(var_23)
    var_25 = bytes(var_24)
    var_26 = module_0.base64_encode(var_25)
    var_27 = module_0.base64_decode(var_26)
    var_28 = 256
    var_29 = [i % var_28 for i in var_21]
    var_30 = bytes(var_29)
    var_31 = module_0.base64_encode(var_30)
    var_32 = module_0.base64_decode(var_31)



# Parsed testcases at query #29
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
    var_17 = 'héllo wörld'
    var_18 = 'utf-8'
    var_19 = module_1.encode(var_18)
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'test'
    var_23 = module_0.base64_encode(var_22)
    var_24 = 'ascii'
    var_25 = module_1.decode(var_24)
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'test'
    var_27 = b'!!!invalid!!!'
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'héllo'
    var_30 = 'utf-8'
    var_31 = module_1.encode(var_30)
    var_32 = module_0.base64_decode(var_31)
    var_33 = b'dGVzdA=='
    var_34 = module_0.base64_decode(var_33)
    assert var_34 == b'test'
    var_35 = b'dGVzdA'
    var_36 = module_0.base64_decode(var_35)
    assert var_36 == b'test'
    var_37 = b'dG V zdA'
    var_38 = module_0.base64_decode(var_37)
    assert var_38 == b'test'



# Parsed testcases at query #30
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
    var_8 = 'Pj4-Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>'
    var_10 = 256
    var_11 = range(var_10)
    var_12 = bytes(var_11)
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'a'
    var_18 = module_0.base64_decode(var_17)



# Parsed testcases at query #31
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
    assert var_14 == b'\xff\xec'
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'dGVzdA=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'test'
    var_19 = '//8='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\xff\xff'



# Parsed testcases at query #32
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
    var_8 = '!!!invalid!!!'
    var_9 = module_0.base64_decode(var_8)
    var_10 = 'aGVsbG8gd29ybGQ_'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello world'
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YWI='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = 'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'SGVsbG8gV29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello World'
    var_2 = b'SGVsbG8gV29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = b''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'YQ=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'a'
    var_10 = b'YWI='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'ab'
    var_12 = b'YWJj'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abc'
    var_14 = b'Pj4-Pg=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'>>>'
    var_16 = b'AAECAwQFBgcICQoLDA0ODw=='
    var_17 = module_0.base64_decode(var_16)
    var_18 = 16
    var_19 = range(var_18)
    var_20 = bytes(var_19)
    var_21 = b'!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'not-base64'
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'\xff\xfe\xfd'
    var_26 = module_0.base64_decode(var_25)



# Parsed testcases at query #34
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
    var_9 = 'YQ=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = 'YWI='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'ab'
    var_13 = 'YWJj'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'abc'
    var_15 = b'\x00\x01\x02'
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\x00\x01\x02'
    var_18 = b'\xff\xfe'
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\xff\xfe'
    var_21 = '!!!invalid!!!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = '\x00\x01\x02'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 256
    var_26 = range(var_25)
    var_27 = bytes(var_26)
    var_28 = module_0.base64_encode(var_27)
    var_29 = module_0.base64_decode(var_28)
    var_30 = b''
    var_31 = b'a'
    var_32 = b'ab'
    var_33 = b'abc'
    var_34 = b'abcd'
    var_35 = b'test'
    var_36 = b'\x00'
    var_37 = 8
    var_38 = var_36 * var_37
    var_39 = b'\xff'
    var_40 = var_39 * var_37
    var_41 = range(var_25)
    var_42 = bytes(var_41)
    var_43 = [var_30, var_31, var_32, var_33, var_34, var_35, var_23, var_38, var_40, var_42]
    var_44 = module_0.base64_decode(var_28)



# Parsed testcases at query #35
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
    var_9 = 'dGVzdA=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'test'
    var_11 = b'dGVzdA=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'test'
    var_13 = 'Pj4-Pg=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'>>>'
    var_15 = '!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'dGVzdA==\x80\x81'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'test'



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
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
    var_8 = 'aGVsbG8gd29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello world'
    var_10 = 'MTIzNA=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'1234'
    var_12 = 'aGVsbG8tXzE='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello-_1'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8#'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'Test data with spaces and special chars!@#$%^&*()'
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #2
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
    var_6 = b'Pj4-Pg=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'>>>'
    var_8 = b'PDw8PA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<<<<'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = b'YWI='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ab'
    var_16 = b'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc'
    var_18 = b'YWJjZA=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abcd'
    var_20 = b'//8='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\xff\xff'
    var_22 = b'+/8='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'\xfb\xff'
    var_24 = 'SGVsbG8gV29ybGQ='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello World'
    var_26 = b'!!!invalid!!!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = b'abc_def'
    var_29 = module_0.base64_decode(var_28)
    var_30 = module_0.base64_decode(var_24)
    assert var_30 == b'Hello World'



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'V29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'World'
    var_4 = ''
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b''
    var_6 = '_-xw=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'\xff\xec'
    var_8 = 'AAAA'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\x00\x00\x00'
    var_10 = 'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'V29ybGQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'World'
    var_14 = b'SGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = b'V29ybGQ='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'World'
    var_18 = 'SGVsbG8=ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'not base64 at all'
    var_23 = module_0.base64_decode(var_22)
    var_24 = module_0.base64_decode(var_4)
    assert var_24 == b''
    var_25 = 'AA=='
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'\x00'
    var_27 = '/w=='
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'\xff'



# Parsed testcases at query #4
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
    var_8 = 'Pj4-Pg'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>> >'
    var_10 = '!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'not valid base64!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'AQIDBAUGBwgJ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x01\x02\x03\x04\x05\x06\x07\x08\t'
    var_16 = b'Hello, World! This is a test.'
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    var_19 = 256
    var_20 = range(var_19)
    var_21 = bytes(var_20)
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #5
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
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'YQ=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = 'YWI='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'ab'
    var_13 = 'YWJj'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'abc'
    var_15 = 'aGVsbG8_d29ybGQ='
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'aGVsbG8_d29ybGQ'
    var_18 = module_0.base64_decode(var_17)
    var_19 = '!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'\xff\xfe\xfd'
    var_22 = module_0.base64_decode(var_21)
    var_23 = module_0.base64_decode(var_5)
    assert var_23 == b'hello'



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'test'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b''
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 256
    var_12 = range(var_11)
    var_13 = bytes(var_12)
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'Hello, 世界!'
    var_17 = 'utf-8'
    var_18 = module_1.encode(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = module_1.encode(var_17)
    var_22 = '!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'abc123!@#'
    var_25 = module_0.base64_decode(var_24)
    var_26 = b'bytes input'
    var_27 = module_0.base64_encode(var_26)
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'bytes input'



# Parsed testcases at query #7
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
    var_6 = b'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b'hello+world/'
    var_13 = module_0.base64_encode(var_12)
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello+world/'
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'héllo'
    var_18 = 'utf-8'
    var_19 = module_1.encode(var_18)
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    var_22 = module_1.encode(var_18)



# Parsed testcases at query #8
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
    var_8 = 'Pj4-Pz8_'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>??'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8=\x80'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = 'WA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'X'
    var_16 = 'WA'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'X'



# Parsed testcases at query #9
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
    var_6 = 'aGVsbG8td29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello-world'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8gd29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello world'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = -1
    var_15 = 'SGVsbG8='[:var_14]
    var_16 = module_0.base64_decode(var_15)
    var_17 = '////'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'\xff\xff\xff'
    var_19 = 'AA=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\x00'
    var_21 = 'AA'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'\x00'



# Parsed testcases at query #10
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
    var_13 = 256
    var_14 = range(var_13)
    var_15 = bytes(var_14)
    var_16 = module_0.base64_encode(var_15)
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #11
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
    var_9 = 'héllo wörld'
    var_10 = module_0.base64_encode(var_9)
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'utf-8'
    var_13 = module_1.encode(var_12)
    var_14 = b'a'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'x'
    var_18 = 100
    var_19 = var_17 * var_18
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    var_22 = 256
    var_23 = range(var_22)
    var_24 = bytes(var_23)
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)
    var_27 = '!!!invalid!!!'
    var_28 = module_0.base64_decode(var_27)
    var_29 = b'\xff\xfe\xfd'
    var_30 = module_0.base64_decode(var_29)
    var_31 = 'héllo'
    var_32 = module_0.base64_decode(var_31)
    var_33 = b'a'
    var_34 = module_0.base64_decode(var_32)



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
    var_5 = b'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = 'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = 'Pj4-Pg=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'>>>'
    var_11 = 'Pj4-Pg'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'>>>'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = '=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'aGVsbG8\x00'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b''
    var_22 = b'a'
    var_23 = b'ab'
    var_24 = b'abc'
    var_25 = b'test data'
    var_26 = b'binary\x00data'
    var_27 = b'special_chars!@#$%^&*()'
    var_28 = b'\xff\xfe\xfd\xfc'
    var_29 = [var_21, var_22, var_23, var_24, var_25, var_26, var_27, var_28]
    var_30 = module_0.base64_decode(var_1)



# Parsed testcases at query #13
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
    var_10 = 'aGVsbG8gd29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello world'



# Parsed testcases at query #14
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
    var_9 = b'!'
    var_10 = 100
    var_11 = var_9 * var_10
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'\x00\xff\xfe\xfd'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'\x00\xff\xfe\xfd'
    var_17 = b''
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = b'!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = '!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'\xff'
    var_25 = module_0.base64_encode(var_24)
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'\xff'



# Parsed testcases at query #15
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
    var_11 = b'dGVzdA=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'test'
    var_13 = b'dGVzdA'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'test'
    var_15 = b'!!!invalid!!!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = b'not valid base64!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'\x00\x01\x02\xff'
    var_20 = module_0.base64_encode(var_19)
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\x00\x01\x02\xff'



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello'
    var_3 = b'aGVsbG8='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello'
    var_5 = b''
    var_6 = module_0.base64_encode(var_5)
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'test data with \x00 null bytes'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    var_11 = 'héllo'
    var_12 = 'utf-8'
    var_13 = module_1.encode(var_12)
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = module_1.encode(var_12)
    var_17 = b'!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'aGVsbG8'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'test'
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'test'



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
    var_11 = b'\xff\xfe\x00\x01'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'aGVsbG8\xff'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'aGVs=bG8='
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'a'
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'a'
    var_23 = b'abcdefgh'
    var_24 = module_0.base64_encode(var_23)
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = module_0.base64_encode(var_0)
    var_4 = module_1.decode()
    var_5 = module_0.base64_decode(var_4)
    var_6 = b''
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 256
    var_10 = range(var_9)
    var_11 = bytes(var_10)
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'a'
    var_15 = b'ab'
    var_16 = b'abc'
    var_17 = b'abcd'
    var_18 = b'abcde'
    var_19 = [var_14, var_15, var_16, var_17, var_18]
    var_20 = module_0.base64_decode(var_1)
    var_21 = b'test'
    var_22 = module_0.base64_encode(var_21)
    var_23 = b'='
    var_24 = b'!!!invalid!!!'
    var_25 = b'not base64 at all'
    var_26 = b'123'
    var_27 = b'===='
    var_28 = [var_24, var_25, var_26, var_27]



# Parsed testcases at query #19
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
    var_17 = b'!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'aGVsbG8!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'\x00\x01\x02\xff\xfe\xfd'
    var_22 = module_0.base64_encode(var_21)
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'a'
    var_25 = b'ab'
    var_26 = b'abc'
    var_27 = b'abcd'
    var_28 = b'\x00'
    var_29 = b'\x00\x00'
    var_30 = b'\xff\xff\xff\xff\xff\xff\xff\xff'
    var_31 = b'test with spaces and punctuation!@#$%^&*()'
    var_32 = [var_24, var_25, var_26, var_27, var_28, var_29, var_30, var_31]
    var_33 = module_0.base64_decode(var_22)



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = b'd29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'world'
    var_4 = 'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = b'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = b'd29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'world'
    var_10 = b'aGVsbG8t'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello-'
    var_12 = b'aGVsbG9f'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'hello_'
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 256
    var_17 = range(var_16)
    var_18 = bytes(var_17)
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)
    var_21 = module_0.base64_decode(var_14)
    assert var_21 == b''
    var_22 = b'AA=='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'\x00'
    var_24 = b'_w=='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'\xff'
    var_26 = b'!!!invalid!!!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = b'aGVsbG8='
    var_29 = b'\x00'
    var_30 = var_28 + var_29
    var_31 = module_0.base64_decode(var_30)
    var_32 = b'YQ=='
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'a'
    var_34 = b'YWI='
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'ab'
    var_36 = b'YWJj'
    var_37 = module_0.base64_decode(var_36)
    assert var_37 == b'abc'
    var_38 = b'YWJjZA=='
    var_39 = module_0.base64_decode(var_38)
    assert var_39 == b'abcd'
    var_40 = module_0.base64_decode(var_4)
    assert var_40 == b'hello'
    var_41 = 'aGVsbG8=ÿ'
    var_42 = module_0.base64_decode(var_41)
    assert var_42 == b'hello'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'd29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'world'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = '='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = 'YQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = 'YWI='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'ab'
    var_18 = 'YWI'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'ab'
    var_20 = 'YWJj'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'abc'
    var_22 = '!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'not base64'
    var_25 = module_0.base64_decode(var_24)
    var_26 = module_0.base64_decode(var_24)
    assert var_26 == b'Hello'
    var_27 = b'test data with spaces and special chars!@#$%'
    var_28 = module_0.base64_encode(var_27)
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'MA=='
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'0'
    var_32 = 'MQ=='
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'1'



# Parsed testcases at query #22
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
    var_5 = b'aGVs'
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hel'
    var_7 = b'aGVsbG8'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = b''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = module_0.base64_decode(var_3)
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
    var_18 = 'aGVsbG8='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'
    var_20 = b'!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'\x00\x01\x02'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'_-w'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'\xff\xc0'
    var_26 = b'a'
    var_27 = b'ab'
    var_28 = b'abc'
    var_29 = b'test data here'
    var_30 = 256
    var_31 = range(var_30)
    var_32 = bytes(var_31)
    var_33 = [var_9, var_26, var_27, var_28, var_29, var_32]
    var_34 = module_0.base64_decode(var_1)



# Parsed testcases at query #23
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
    var_8 = b'_-A='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'\xff\xe0'
    var_10 = '_-A='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\xff\xe0'
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b'AA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\x00'
    var_18 = 'AA=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'\x00'
    var_20 = b'QQ=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'A'
    var_22 = b'QUE='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'AA'
    var_24 = b'!!!invalid!!!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = b'test$%^'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVs bG8='
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    assert var_2 == b'hello world'
    var_3 = b'test'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'test'
    var_6 = 'hello'
    var_7 = module_0.base64_encode(var_6)
    var_8 = module_1.decode()
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'python'
    var_11 = module_0.base64_encode(var_10)
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'python'
    var_13 = b'\x00\x01\x02\xff'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x00\x01\x02\xff'
    var_16 = b''
    var_17 = module_0.base64_encode(var_16)
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''
    var_19 = '!!!invalid base64!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'\xff\xff\xff\xff'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'aGVsbG8g d29ybGQ='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'hello world'



# Parsed testcases at query #25
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
    var_8 = b'a'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'a'
    var_11 = b'SGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = b'SGVsbG8===='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = b'test?data=123&more'
    var_18 = module_0.base64_encode(var_17)
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'test?data=123&more'
    var_20 = b'!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'not base64 at all'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b' SGVsbG8gV29ybGQ='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello World'



# Parsed testcases at query #26
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
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b'a-_w'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'k\xef\xc0'
    var_10 = 'a-_w'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'k\xef\xc0'
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b'Zw=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'g'
    var_18 = 'Zw=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'g'
    var_20 = b'YQ=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'a'
    var_22 = b'YWI='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'ab'
    var_24 = b'YWJj'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'abc'
    var_26 = b'test data'
    var_27 = module_1.b64encode(var_26)
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'test data'
    var_29 = b'!!!invalid!!!'
    var_30 = module_0.base64_decode(var_29)
    var_31 = b'\xff\xfeSGVsbG8='
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'Hello'
    var_33 = None
    var_34 = module_0.base64_decode(var_33)
    var_35 = 'ascii'
    var_36 = module_0.base64_decode(var_2)
    assert var_36 == b'Hello'



# Parsed testcases at query #27
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
    var_5 = b'aGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello'
    var_7 = b'aGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'hello'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = b'\xff\xfe'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xfe'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'aGVsbG8\x80'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = 256
    var_19 = range(var_18)
    var_20 = bytes(var_19)
    var_21 = module_0.base64_encode(var_20)
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'aGVsbG8='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'hello'
    var_25 = 'aGVsbG8=='
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'hello'



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
    var_13 = b'\xfb\xff\xff\xff\xff'
    var_14 = module_0.base64_encode(var_13)
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'x'
    var_17 = 1000
    var_18 = var_16 * var_17
    var_19 = module_0.base64_encode(var_18)
    var_20 = module_0.base64_decode(var_19)



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
    var_5 = 'aGVsbG8gd29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'hello world'
    var_7 = 'YQ=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'a'
    var_9 = 'YWI='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'ab'
    var_11 = 'YWJj'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'abc'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'Pj4-Pg=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'>>>'
    var_17 = '!!!invalid!!!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'not valid base64'
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = module_0.base64_encode(var_0)
    var_2 = module_0.base64_decode(var_1)
    var_3 = b'aGVsbG8gd29ybGQ='
    var_4 = module_0.base64_decode(var_3)
    assert var_4 == b'hello world'
    var_5 = b''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'aGVsbG8gd29ybGQ='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello world'
    var_11 = b'test'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'test'
    var_14 = 'test'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'test'
    var_17 = b'dGVzdA'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'test'
    var_19 = b'!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'aGVsbG8gd29ybGQ'
    var_22 = module_0.base64_decode(var_21)
    var_23 = b'\xff\xfe\xfd'
    var_24 = module_0.base64_decode(var_23)



# Parsed testcases at query #31
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
    var_6 = b'dGVzdA=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'test'
    var_8 = 'Pj4-Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'>>>'
    var_10 = '!!!invalid!!!'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'Hello World'
    var_13 = module_0.base64_decode(var_12)



# Parsed testcases at query #32
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
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
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
    var_22 = b'!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'aGVs$bG8='
    var_25 = module_0.base64_decode(var_24)
    var_26 = b'aGVsbG8\xff'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'hello'



# Parsed testcases at query #33
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
    var_6 = 'aGVsbG8_d29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello?world'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'aGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = 'YQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'a'
    var_14 = '!!!invalid!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8$'
    var_17 = module_0.base64_decode(var_16)
    var_18 = module_0.base64_decode(var_8)
    assert var_18 == b'Hello'



# Parsed testcases at query #34
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
    var_6 = 'YQ=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'a'
    var_8 = b'dGVzdA=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'test'
    var_10 = 'aGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello'
    var_12 = '!!!invalid!!!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'dGVzdA==\n'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'test'
    var_16 = b'MTIz'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'123'
    var_18 = b'QWJj'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Abc'



# Parsed testcases at query #35
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
    var_11 = b'\x00\x01\x02'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\x00\x01\x02'
    var_14 = b'aGVsbG8\xff'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'hello'
    var_16 = b'!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'\xff\xff\xff\xff'
    var_19 = module_0.base64_decode(var_18)



