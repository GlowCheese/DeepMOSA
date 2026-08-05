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
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello?'
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello/'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!@#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'Invalid@@'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #2
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
    var_6 = 'SGVsbG8-'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello+'
    var_8 = 'SGVsbG8_'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello/'
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = 'SGVsbG8=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = 'SGVsbG8!@#'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8=!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8==!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8=!@#'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8=!@#$%^&*()'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8=!@#$%^&*()_+-=[]{};:\'",./<>?|`~'
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'SGVsbG8=!@#$%^&*()_+-=[]{};:\'",./<>?|`~1234567890'
    var_30 = module_0.base64_decode(var_29)
    var_31 = 'SGVsbG8=!@#$%^&*()_+-=[]{};:\'",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    var_32 = module_0.base64_decode(var_31)
    var_33 = 'SGVsbG8=!@#$%^&*()_+-=[]{};:\'",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{};:\'",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    var_34 = module_0.base64_decode(var_33)
    var_35 = 'SGVsbG8=!@#$%^&*()_+-=[]{};:\'",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{};:\'",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{};:\'",./<>?|`~1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    var_36 = module_0.base64_decode(var_35)



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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = module_0.base64_decode(var_2)
    var_10 = module_0.base64_decode(var_2)
    var_11 = module_0.base64_decode(var_2)
    var_12 = module_0.base64_decode(var_2)
    var_13 = module_0.base64_decode(var_2)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8=='
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8==='
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8=!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=ÿ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = 'SGVsbG8=ÿÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'



# Parsed testcases at query #4
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = 'SGVsbG8=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = module_0.base64_decode(var_0)
    assert var_15 == b'Hello'
    var_16 = module_0.base64_decode(var_2)
    assert var_16 == b'Hello'
    var_17 = 'SGVsbG8-'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = 'SGVsbG8_'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = 'SGVsbG8!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8@'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8#'
    var_26 = module_0.base64_decode(var_25)
    var_27 = '8J+YgA=='
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'\x1f\x8b\x08'
    var_29 = '8J+YgA'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'\x1f\x8b\x08'
    var_31 = 'SGVsbG8====='
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'Hello'



# Parsed testcases at query #5
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8!'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = b'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = 'Invalid@'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'Invalid@'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8é'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'
    var_25 = b'SGVsbG8\xff'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_0.base64_decode(var_0)
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #7
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
    var_6 = 'PGJpZ2Zvb3Q+'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'<bigfoot>'
    var_8 = 'PGJpZ2Zvb3Q'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<bigfoot>'
    var_10 = 'PGJpZ2Zvb3Q='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigfoot>'
    var_12 = 'PGJpZ2Zvb3Q=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #8
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
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8gV29ybGQh'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello World!'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'SGVsbG8!@#'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = b'SGVsbG8!@#'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = module_0.base64_decode(var_0)
    assert var_18 == b'Hello'
    var_19 = 'SGVsbG8=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = 'SGVsbG8ÿ'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = b'SGVsbG8\xff'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'
    var_25 = 'SGVsbG8!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8=!'
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'SGVsbG8==!'
    var_30 = module_0.base64_decode(var_29)



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
    var_4 = 'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = 'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = b'SGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = 'PGJpZ25hbWU+'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<big name>'
    var_16 = 'PGJpZ25hbWU'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'<big name>'
    var_18 = '-_'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'\xfb\xff'
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=ÿ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = 'SGVsbG8ÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'



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
    var_4 = 'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = 'SGVsbG8_'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = b'SGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = b'SGVsbG8'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8@'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8#'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8ÿ'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'
    var_25 = 'SGVsbG8\x00'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'
    var_27 = module_0.base64_decode(var_21)
    assert var_27 == b'Hello'
    var_28 = 'SGVsbG8=='
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'



# Parsed testcases at query #11
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = 'SGVsbG8_'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = 'Invalid!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = b'Invalid!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8ÿ'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = b'SGVsbG8\xff'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'



# Parsed testcases at query #12
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJyPg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<br>'
    var_10 = 'PGJyPg'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<br>'
    var_12 = module_0.base64_decode(var_8)
    assert var_12 == b'<br>'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8='
    var_20 = 1000
    var_21 = var_19 * var_20
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8='
    var_24 = 1000
    var_25 = var_23 * var_24
    var_26 = '!'
    var_27 = var_25 + var_26
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'SGVsbG8=ÿ'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'
    var_31 = 'SGVsbG8=ÿÿ'
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'Hello'



# Parsed testcases at query #13
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
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8==='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = 'SGVsbG8!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8@'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8#'
    var_19 = module_0.base64_decode(var_18)
    var_20 = module_0.base64_decode(var_18)
    assert var_20 == b'Hello'
    var_21 = module_0.base64_decode(var_2)
    assert var_21 == b'Hello'
    var_22 = module_0.base64_decode(var_18)
    assert var_22 == b'Hello'
    var_23 = module_0.base64_decode(var_2)
    assert var_23 == b'Hello'
    var_24 = module_0.base64_decode(var_18)
    assert var_24 == b'Hello'
    var_25 = module_0.base64_decode(var_2)
    assert var_25 == b'Hello'



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
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_2)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_2)
    assert var_9 == b'Hello'
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = module_0.base64_decode(var_2)
    assert var_11 == b'Hello'
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = module_0.base64_decode(var_2)
    assert var_13 == b'Hello'
    var_14 = module_0.base64_decode(var_0)
    assert var_14 == b'Hello'
    var_15 = module_0.base64_decode(var_2)
    assert var_15 == b'Hello'
    var_16 = module_0.base64_decode(var_0)
    assert var_16 == b'Hello'
    var_17 = module_0.base64_decode(var_2)
    assert var_17 == b'Hello'
    var_18 = module_0.base64_decode(var_0)
    assert var_18 == b'Hello'
    var_19 = module_0.base64_decode(var_2)
    assert var_19 == b'Hello'
    var_20 = module_0.base64_decode(var_0)
    assert var_20 == b'Hello'
    var_21 = module_0.base64_decode(var_2)
    assert var_21 == b'Hello'
    var_22 = module_0.base64_decode(var_0)
    assert var_22 == b'Hello'
    var_23 = module_0.base64_decode(var_2)
    assert var_23 == b'Hello'
    var_24 = module_0.base64_decode(var_0)
    assert var_24 == b'Hello'
    var_25 = module_0.base64_decode(var_2)
    assert var_25 == b'Hello'
    var_26 = module_0.base64_decode(var_0)
    assert var_26 == b'Hello'
    var_27 = module_0.base64_decode(var_2)
    assert var_27 == b'Hello'
    var_28 = module_0.base64_decode(var_0)
    assert var_28 == b'Hello'
    var_29 = module_0.base64_decode(var_2)
    assert var_29 == b'Hello'
    var_30 = module_0.base64_decode(var_0)
    assert var_30 == b'Hello'
    var_31 = module_0.base64_decode(var_2)
    assert var_31 == b'Hello'
    var_32 = module_0.base64_decode(var_0)
    assert var_32 == b'Hello'
    var_33 = module_0.base64_decode(var_2)
    assert var_33 == b'Hello'
    var_34 = module_0.base64_decode(var_0)
    assert var_34 == b'Hello'
    var_35 = module_0.base64_decode(var_2)
    assert var_35 == b'Hello'
    var_36 = module_0.base64_decode(var_0)
    assert var_36 == b'Hello'
    var_37 = module_0.base64_decode(var_2)
    assert var_37 == b'Hello'
    var_38 = module_0.base64_decode(var_0)
    assert var_38 == b'Hello'
    var_39 = module_0.base64_decode(var_2)
    assert var_39 == b'Hello'
    var_40 = module_0.base64_decode(var_0)
    assert var_40 == b'Hello'
    var_41 = module_0.base64_decode(var_2)
    assert var_41 == b'Hello'
    var_42 = module_0.base64_decode(var_0)
    assert var_42 == b'Hello'
    var_43 = module_0.base64_decode(var_2)
    assert var_43 == b'Hello'
    var_44 = module_0.base64_decode(var_0)
    assert var_44 == b'Hello'
    var_45 = module_0.base64_decode(var_2)
    assert var_45 == b'Hello'
    var_46 = module_0.base64_decode(var_0)
    assert var_46 == b'Hello'
    var_47 = module_0.base64_decode(var_2)
    assert var_47 == b'Hello'
    var_48 = module_0.base64_decode(var_0)
    assert var_48 == b'Hello'
    var_49 = module_0.base64_decode(var_2)
    assert var_49 == b'Hello'
    var_50 = module_0.base64_decode(var_0)
    assert var_50 == b'Hello'
    var_51 = module_0.base64_decode(var_2)
    assert var_51 == b'Hello'
    var_52 = module_0.base64_decode(var_0)
    assert var_52 == b'Hello'
    var_53 = module_0.base64_decode(var_2)
    assert var_53 == b'Hello'
    var_54 = module_0.base64_decode(var_0)
    assert var_54 == b'Hello'
    var_55 = module_0.base64_decode(var_2)
    assert var_55 == b'Hello'
    var_56 = module_0.base64_decode(var_0)
    assert var_56 == b'Hello'
    var_57 = module_0.base64_decode(var_2)
    assert var_57 == b'Hello'
    var_58 = module_0.base64_decode(var_0)
    assert var_58 == b'Hello'
    var_59 = module_0.base64_decode(var_2)
    assert var_59 == b'Hello'
    var_60 = module_0.base64_decode(var_0)
    assert var_60 == b'Hello'
    var_61 = module_0.base64_decode(var_2)
    assert var_61 == b'Hello'
    var_62 = module_0.base64_decode(var_0)
    assert var_62 == b'Hello'
    var_63 = module_0.base64_decode(var_2)
    assert var_63 == b'Hello'
    var_64 = module_0.base64_decode(var_0)
    assert var_64 == b'Hello'
    var_65 = module_0.base64_decode(var_2)
    assert var_65 == b'Hello'
    var_66 = module_0.base64_decode(var_0)
    assert var_66 == b'Hello'
    var_67 = module_0.base64_decode(var_2)
    assert var_67 == b'Hello'
    var_68 = module_0.base64_decode(var_0)
    assert var_68 == b'Hello'
    var_69 = module_0.base64_decode(var_2)
    assert var_69 == b'Hello'
    var_70 = module_0.base64_decode(var_0)
    assert var_70 == b'Hello'
    var_71 = module_0.base64_decode(var_2)
    assert var_71 == b'Hello'
    var_72 = module_0.base64_decode(var_0)
    assert var_72 == b'Hello'
    var_73 = module_0.base64_decode(var_2)
    assert var_73 == b'Hello'
    var_74 = module_0.base64_decode(var_0)
    assert var_74 == b'Hello'
    var_75 = module_0.base64_decode(var_2)
    assert var_75 == b'Hello'
    var_76 = module_0.base64_decode(var_0)
    assert var_76 == b'Hello'
    var_77 = module_0.base64_decode(var_2)
    assert var_77 == b'Hello'
    var_78 = module_0.base64_decode(var_0)
    assert var_78 == b'Hello'
    var_79 = module_0.base64_decode(var_2)
    assert var_79 == b'Hello'
    var_80 = module_0.base64_decode(var_0)
    assert var_80 == b'Hello'
    var_81 = module_0.base64_decode(var_2)
    assert var_81 == b'Hello'
    var_82 = module_0.base64_decode(var_0)
    assert var_82 == b'Hello'
    var_83 = module_0.base64_decode(var_2)
    assert var_83 == b'Hello'
    var_84 = module_0.base64_decode(var_0)
    assert var_84 == b'Hello'
    var_85 = module_0.base64_decode(var_2)
    assert var_85 == b'Hello'
    var_86 = module_0.base64_decode(var_0)
    assert var_86 == b'Hello'
    var_87 = module_0.base64_decode(var_2)
    assert var_87 == b'Hello'
    var_88 = module_0.base64_decode(var_0)
    assert var_88 == b'Hello'
    var_89 = module_0.base64_decode(var_2)
    assert var_89 == b'Hello'
    var_90 = module_0.base64_decode(var_0)
    assert var_90 == b'Hello'
    var_91 = module_0.base64_decode(var_2)
    assert var_91 == b'Hello'
    var_92 = module_0.base64_decode(var_0)
    assert var_92 == b'Hello'
    var_93 = module_0.base64_decode(var_2)
    assert var_93 == b'Hello'
    var_94 = module_0.base64_decode(var_0)
    assert var_94 == b'Hello'
    var_95 = module_0.base64_decode(var_2)
    assert var_95 == b'Hello'
    var_96 = module_0.base64_decode(var_0)
    assert var_96 == b'Hello'
    var_97 = module_0.base64_decode(var_2)
    assert var_97 == b'Hello'
    var_98 = module_0.base64_decode(var_0)
    assert var_98 == b'Hello'
    var_99 = module_0.base64_decode(var_2)
    assert var_99 == b'Hello'
    var_100 = module_0.base64_decode(var_0)
    assert var_100 == b'Hello'
    var_101 = module_0.base64_decode(var_2)
    assert var_101 == b'Hello'
    var_102 = module_0.base64_decode(var_0)
    assert var_102 == b'Hello'
    var_103 = module_0.base64_decode(var_2)
    assert var_103 == b'Hello'
    var_104 = module_0.base64_decode(var_0)
    assert var_104 == b'Hello'
    var_105 = module_0.base64_decode(var_2)
    assert var_105 == b'Hello'
    var_106 = module_0.base64_decode(var_0)
    assert var_106 == b'Hello'
    var_107 = module_0.base64_decode(var_2)
    assert var_107 == b'Hello'
    var_108 = module_0.base64_decode(var_0)
    assert var_108 == b'Hello'
    var_109 = module_0.base64_decode(var_2)
    assert var_109 == b'Hello'
    var_110 = module_0.base64_decode(var_0)
    assert var_110 == b'Hello'



# Parsed testcases at query #15
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
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!\n'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = b'SGVsbG8!\n'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'Invalid@@'
    var_23 = module_0.base64_decode(var_22)
    var_24 = b'Invalid@@'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8é'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = b'SGVsbG8\xc3\xa9'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'



# Parsed testcases at query #16
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
    var_6 = 'SGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZ25hbWU+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<big name>'
    var_12 = 'PGJpZ25hbWU'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<big name>'
    var_14 = 'YWJjZGVmZ2g='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abcdefgh'
    var_16 = 'YWJjZGVmZ2g'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abcdefgh'
    var_18 = 'YWJj'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'ab'
    var_20 = 'YWJjZGU='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'abcde'
    var_22 = 'Hello!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = '12345*'
    var_25 = module_0.base64_decode(var_24)
    var_26 = '!@#$%^'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8=☺'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'
    var_30 = 'SGVsbG8=😊'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'
    var_32 = ' SGVsbG8= '
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'Hello'
    var_34 = 'SG Vs bG 8='
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'Hello'
    var_36 = 'SGVs\nbG8='
    var_37 = module_0.base64_decode(var_36)
    assert var_37 == b'Hello'
    var_38 = 'SGVs\r\nbG8='
    var_39 = module_0.base64_decode(var_38)
    assert var_39 == b'Hello'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8-V29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello-World'
    var_14 = 'SGVsbG8_V29ybGQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello_World'
    var_16 = module_0.base64_decode(var_0)
    assert var_16 == b'Hello'
    var_17 = 'SGVsbG8=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = 'SGVsbG8==='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = ''
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b''
    var_23 = 'SGVsbG8!@#$'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'
    var_25 = 'SGVsbG8!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8=!'
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'SGVsbG8ÿ'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'



# Parsed testcases at query #18
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'aGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'hello'
    var_18 = 'aGVsbG8'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'hello'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8#'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8$'
    var_25 = module_0.base64_decode(var_24)
    var_26 = module_0.base64_decode(var_2)
    assert var_26 == b'Hello'
    var_27 = module_0.base64_decode(var_24)
    assert var_27 == b'Hello'
    var_28 = module_0.base64_decode(var_10)
    assert var_28 == b'Hello'
    var_29 = 'SGVsbG8ÿ'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'
    var_31 = 'SGVsbG8\x00'
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'Hello'



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
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_2)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_2)
    assert var_9 == b'Hello'
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = module_0.base64_decode(var_2)
    assert var_11 == b'Hello'
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = module_0.base64_decode(var_2)
    assert var_13 == b'Hello'
    var_14 = module_0.base64_decode(var_0)
    assert var_14 == b'Hello'
    var_15 = module_0.base64_decode(var_2)
    assert var_15 == b'Hello'
    var_16 = module_0.base64_decode(var_0)
    assert var_16 == b'Hello'
    var_17 = module_0.base64_decode(var_2)
    assert var_17 == b'Hello'
    var_18 = module_0.base64_decode(var_0)
    assert var_18 == b'Hello'
    var_19 = module_0.base64_decode(var_2)
    assert var_19 == b'Hello'
    var_20 = module_0.base64_decode(var_0)
    assert var_20 == b'Hello'
    var_21 = module_0.base64_decode(var_2)
    assert var_21 == b'Hello'
    var_22 = module_0.base64_decode(var_0)
    assert var_22 == b'Hello'
    var_23 = module_0.base64_decode(var_2)
    assert var_23 == b'Hello'
    var_24 = module_0.base64_decode(var_0)
    assert var_24 == b'Hello'
    var_25 = module_0.base64_decode(var_2)
    assert var_25 == b'Hello'
    var_26 = module_0.base64_decode(var_0)
    assert var_26 == b'Hello'
    var_27 = module_0.base64_decode(var_2)
    assert var_27 == b'Hello'
    var_28 = module_0.base64_decode(var_0)
    assert var_28 == b'Hello'
    var_29 = module_0.base64_decode(var_2)
    assert var_29 == b'Hello'
    var_30 = module_0.base64_decode(var_0)
    assert var_30 == b'Hello'
    var_31 = module_0.base64_decode(var_2)
    assert var_31 == b'Hello'
    var_32 = module_0.base64_decode(var_0)
    assert var_32 == b'Hello'
    var_33 = module_0.base64_decode(var_2)
    assert var_33 == b'Hello'
    var_34 = module_0.base64_decode(var_0)
    assert var_34 == b'Hello'
    var_35 = module_0.base64_decode(var_2)
    assert var_35 == b'Hello'
    var_36 = module_0.base64_decode(var_0)
    assert var_36 == b'Hello'
    var_37 = module_0.base64_decode(var_2)
    assert var_37 == b'Hello'
    var_38 = module_0.base64_decode(var_0)
    assert var_38 == b'Hello'
    var_39 = module_0.base64_decode(var_2)
    assert var_39 == b'Hello'
    var_40 = module_0.base64_decode(var_0)
    assert var_40 == b'Hello'
    var_41 = module_0.base64_decode(var_2)
    assert var_41 == b'Hello'
    var_42 = module_0.base64_decode(var_0)
    assert var_42 == b'Hello'
    var_43 = module_0.base64_decode(var_2)
    assert var_43 == b'Hello'
    var_44 = module_0.base64_decode(var_0)
    assert var_44 == b'Hello'
    var_45 = module_0.base64_decode(var_2)
    assert var_45 == b'Hello'
    var_46 = module_0.base64_decode(var_0)
    assert var_46 == b'Hello'
    var_47 = module_0.base64_decode(var_2)
    assert var_47 == b'Hello'
    var_48 = module_0.base64_decode(var_0)
    assert var_48 == b'Hello'
    var_49 = module_0.base64_decode(var_2)
    assert var_49 == b'Hello'
    var_50 = module_0.base64_decode(var_0)
    assert var_50 == b'Hello'
    var_51 = module_0.base64_decode(var_2)
    assert var_51 == b'Hello'
    var_52 = module_0.base64_decode(var_0)
    assert var_52 == b'Hello'
    var_53 = module_0.base64_decode(var_2)
    assert var_53 == b'Hello'
    var_54 = module_0.base64_decode(var_0)
    assert var_54 == b'Hello'
    var_55 = module_0.base64_decode(var_2)
    assert var_55 == b'Hello'
    var_56 = module_0.base64_decode(var_0)
    assert var_56 == b'Hello'
    var_57 = module_0.base64_decode(var_2)
    assert var_57 == b'Hello'
    var_58 = module_0.base64_decode(var_0)
    assert var_58 == b'Hello'
    var_59 = module_0.base64_decode(var_2)
    assert var_59 == b'Hello'
    var_60 = module_0.base64_decode(var_0)
    assert var_60 == b'Hello'
    var_61 = module_0.base64_decode(var_2)
    assert var_61 == b'Hello'
    var_62 = module_0.base64_decode(var_0)
    assert var_62 == b'Hello'
    var_63 = module_0.base64_decode(var_2)
    assert var_63 == b'Hello'
    var_64 = module_0.base64_decode(var_0)
    assert var_64 == b'Hello'
    var_65 = module_0.base64_decode(var_2)
    assert var_65 == b'Hello'
    var_66 = module_0.base64_decode(var_0)
    assert var_66 == b'Hello'
    var_67 = module_0.base64_decode(var_2)
    assert var_67 == b'Hello'
    var_68 = module_0.base64_decode(var_0)
    assert var_68 == b'Hello'
    var_69 = module_0.base64_decode(var_2)
    assert var_69 == b'Hello'
    var_70 = module_0.base64_decode(var_0)
    assert var_70 == b'Hello'
    var_71 = module_0.base64_decode(var_2)
    assert var_71 == b'Hello'
    var_72 = module_0.base64_decode(var_0)
    assert var_72 == b'Hello'
    var_73 = module_0.base64_decode(var_2)
    assert var_73 == b'Hello'
    var_74 = module_0.base64_decode(var_0)
    assert var_74 == b'Hello'
    var_75 = module_0.base64_decode(var_2)
    assert var_75 == b'Hello'
    var_76 = module_0.base64_decode(var_0)
    assert var_76 == b'Hello'
    var_77 = module_0.base64_decode(var_2)
    assert var_77 == b'Hello'
    var_78 = module_0.base64_decode(var_0)
    assert var_78 == b'Hello'
    var_79 = module_0.base64_decode(var_2)
    assert var_79 == b'Hello'
    var_80 = module_0.base64_decode(var_0)
    assert var_80 == b'Hello'
    var_81 = module_0.base64_decode(var_2)
    assert var_81 == b'Hello'
    var_82 = module_0.base64_decode(var_0)
    assert var_82 == b'Hello'
    var_83 = module_0.base64_decode(var_2)
    assert var_83 == b'Hello'
    var_84 = module_0.base64_decode(var_0)
    assert var_84 == b'Hello'
    var_85 = module_0.base64_decode(var_2)
    assert var_85 == b'Hello'
    var_86 = module_0.base64_decode(var_0)
    assert var_86 == b'Hello'
    var_87 = module_0.base64_decode(var_2)
    assert var_87 == b'Hello'



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = b'SGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'Hello'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = 'Zg=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'f'
    var_11 = 'Zg'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'f'
    var_13 = 'SGVsbG8gV29ybGQ='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello World'
    var_15 = 'PGJyPg=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'<br>'
    var_17 = 'PGJyPg'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'<br>'
    var_19 = '-_='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'\xff'
    var_21 = '-_'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'\xff'
    var_23 = 'Invalid!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8=SGVsbG8='
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'SGVsbG8=\x00\x01\x02'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'
    var_31 = 'SGVsbG8=ÿþý'
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'Hello'
    var_33 = 'SGVs'
    var_34 = module_0.base64_decode(var_33)
    assert var_34 == b'H'
    var_35 = 'SGV'
    var_36 = module_0.base64_decode(var_35)
    assert var_36 == b'H'
    var_37 = 'SG'
    var_38 = module_0.base64_decode(var_37)
    assert var_38 == b'H'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJhcmV2YWx1ZT4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<barevalue>'
    var_10 = 'PGJhcmV2YWx1ZT4'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<barevalue>'
    var_12 = module_0.base64_decode(var_8)
    assert var_12 == b'<barevalue>'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8!='
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8=!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8==='
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8===='
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8=ÿ'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'
    var_29 = 'SGVsbG8=ÿþ'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'



# Parsed testcases at query #22
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
    var_8 = 'Zg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'f'
    var_10 = 'Zm8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'fo'
    var_12 = 'Zm9v'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'foo'
    var_14 = 'Zm9vYg=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'foob'
    var_16 = 'Zm9vYmE='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'fooba'
    var_18 = 'Zm9vYmFy'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'foobar'
    var_20 = 'PGJhcj4='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'<bar>'
    var_22 = 'PGJhcj4'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'<bar>'
    var_24 = module_0.base64_decode(var_20)
    assert var_24 == b'<bar>'
    var_25 = module_0.base64_decode(var_22)
    assert var_25 == b'<bar>'
    var_26 = module_0.base64_decode(var_20)
    assert var_26 == b'<bar>'
    var_27 = module_0.base64_decode(var_22)
    assert var_27 == b'<bar>'
    var_28 = 'SGVsbG8!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8?'
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG8@'
    var_33 = module_0.base64_decode(var_32)
    var_34 = 'SGVsbG8#'
    var_35 = module_0.base64_decode(var_34)
    var_36 = 'SGVsbG8$'
    var_37 = module_0.base64_decode(var_36)
    var_38 = 'SGVsbG8%'
    var_39 = module_0.base64_decode(var_38)
    var_40 = 'SGVsbG8^'
    var_41 = module_0.base64_decode(var_40)
    var_42 = 'SGVsbG8&'
    var_43 = module_0.base64_decode(var_42)
    var_44 = 'SGVsbG8*'
    var_45 = module_0.base64_decode(var_44)
    var_46 = 'SGVsbG8('
    var_47 = module_0.base64_decode(var_46)
    var_48 = 'SGVsbG8)'
    var_49 = module_0.base64_decode(var_48)
    var_50 = 'SGVsbG8_'
    var_51 = module_0.base64_decode(var_50)
    var_52 = 'SGVsbG8+'
    var_53 = module_0.base64_decode(var_52)
    var_54 = 'SGVsbG8 '
    var_55 = module_0.base64_decode(var_54)
    var_56 = 'SGVsbG8\t'
    var_57 = module_0.base64_decode(var_56)
    var_58 = 'SGVsbG8\n'
    var_59 = module_0.base64_decode(var_58)
    var_60 = 'SGVsbG8\r'
    var_61 = module_0.base64_decode(var_60)
    var_62 = 'SGVsbG8\x00'
    var_63 = module_0.base64_decode(var_62)
    var_64 = 'SGVsbG8\x01'
    var_65 = module_0.base64_decode(var_64)
    var_66 = 'SGVsbG8\x1f'
    var_67 = module_0.base64_decode(var_66)
    var_68 = 'SGVsbG8\x7f'
    var_69 = module_0.base64_decode(var_68)
    var_70 = 'SGVsbG8ÿ'
    var_71 = module_0.base64_decode(var_70)



# Parsed testcases at query #23
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8!@#'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = b'SGVsbG8!@#'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = 'SGVsbG8é'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = b'SGVsbG8\xc3\xa9'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = 'SGVsbG8!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'SGVsbG8!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8==='
    var_28 = module_0.base64_decode(var_27)
    var_29 = b'SGVsbG8==='
    var_30 = module_0.base64_decode(var_29)



# Parsed testcases at query #24
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJvZHk+'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<body>'
    var_10 = 'PGJvZHk'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<body>'
    var_12 = 'PGJvZHk='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<body>'
    var_14 = module_0.base64_decode(var_8)
    assert var_14 == b'<body>'
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8@'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8#'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8ÿ'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'
    var_25 = 'SGVsbG8\x00'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_2)
    assert var_9 == b'Hello'
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = module_0.base64_decode(var_2)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'Invalid!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8=ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #26
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
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!@#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8ÿ'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'



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
    var_4 = 'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'PGJpZ2Zvb3Q+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = 'PGJpZ2Zvb3Q'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigfoot>'
    var_16 = 'YWJjZGVmZ2g='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abcdefgh'
    var_18 = 'YWJjZGVmZ2g'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abcdefgh'
    var_20 = '8J+YgA=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\x10\xff\x00'
    var_22 = '8J+YgA'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'\x10\xff\x00'
    var_24 = '!!!invalid!!!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8=!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8=😊'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'
    var_32 = 'SGVsbG8=äöü'
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'Hello'



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b'SGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b''
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
    var_20 = 'PGJhc2U2ND4='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'<base64>'
    var_22 = 'PGJhc2U2ND4'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'<base64>'
    var_24 = module_0.base64_decode(var_20)
    assert var_24 == b'<base64>'
    var_25 = module_0.base64_decode(var_22)
    assert var_25 == b'<base64>'
    var_26 = 'SGVsbG8ÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = b'SGVsbG8\xff'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'
    var_30 = 'SGVsbG8!'
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG8@'
    var_33 = module_0.base64_decode(var_32)
    var_34 = 'SGVsbG8#'
    var_35 = module_0.base64_decode(var_34)
    var_36 = 'SGVsbG8$'
    var_37 = module_0.base64_decode(var_36)
    var_38 = 'SGVsbG8%'
    var_39 = module_0.base64_decode(var_38)
    var_40 = 'SGVsbG8^'
    var_41 = module_0.base64_decode(var_40)
    var_42 = 'SGVsbG8&'
    var_43 = module_0.base64_decode(var_42)
    var_44 = 'SGVsbG8*'
    var_45 = module_0.base64_decode(var_44)
    var_46 = 'SGVsbG8('
    var_47 = module_0.base64_decode(var_46)
    var_48 = 'SGVsbG8)'
    var_49 = module_0.base64_decode(var_48)
    var_50 = 'SGVsbG8+'
    var_51 = module_0.base64_decode(var_50)
    var_52 = 'SGVsbG8/'
    var_53 = module_0.base64_decode(var_52)
    var_54 = 'SGVsbG8<'
    var_55 = module_0.base64_decode(var_54)
    var_56 = 'SGVsbG8>'
    var_57 = module_0.base64_decode(var_56)
    var_58 = 'SGVsbG8['
    var_59 = module_0.base64_decode(var_58)
    var_60 = 'SGVsbG8]'
    var_61 = module_0.base64_decode(var_60)
    var_62 = 'SGVsbG8{'
    var_63 = module_0.base64_decode(var_62)
    var_64 = 'SGVsbG8}'
    var_65 = module_0.base64_decode(var_64)
    var_66 = 'SGVsbG8|'
    var_67 = module_0.base64_decode(var_66)
    var_68 = 'SGVsbG8\\'
    var_69 = module_0.base64_decode(var_68)
    var_70 = 'SGVsbG8"'
    var_71 = module_0.base64_decode(var_70)
    var_72 = "SGVsbG8'"
    var_73 = module_0.base64_decode(var_72)
    var_74 = 'SGVsbG8`'
    var_75 = module_0.base64_decode(var_74)
    var_76 = 'SGVsbG8~'
    var_77 = module_0.base64_decode(var_76)
    var_78 = 'SGVsbG8 '
    var_79 = module_0.base64_decode(var_78)
    var_80 = 'SGVsbG8\t'
    var_81 = module_0.base64_decode(var_80)
    var_82 = 'SGVsbG8\n'
    var_83 = module_0.base64_decode(var_82)
    var_84 = 'SGVsbG8\r'
    var_85 = module_0.base64_decode(var_84)
    var_86 = 'SGVsbG8\x0b'
    var_87 = module_0.base64_decode(var_86)
    var_88 = 'SGVsbG8\x0c'
    var_89 = module_0.base64_decode(var_88)
    var_90 = 'SGVsbG8\r'
    var_91 = module_0.base64_decode(var_90)
    var_92 = 'SGVsbG8\x0e'
    var_93 = module_0.base64_decode(var_92)
    var_94 = 'SGVsbG8\x0f'
    var_95 = module_0.base64_decode(var_94)
    var_96 = 'SGVsbG8\x10'
    var_97 = module_0.base64_decode(var_96)
    var_98 = 'SGVsbG8\x11'
    var_99 = module_0.base64_decode(var_98)
    var_100 = 'SGVsbG8\x12'
    var_101 = module_0.base64_decode(var_100)
    var_102 = 'SGVsbG8\x13'
    var_103 = module_0.base64_decode(var_102)
    var_104 = 'SGVsbG8\x14'
    var_105 = module_0.base64_decode(var_104)
    var_106 = 'SGVsbG8\x15'
    var_107 = module_0.base64_decode(var_106)
    var_108 = 'SGVsbG8\x16'
    var_109 = module_0.base64_decode(var_108)
    var_110 = 'SGVsbG8\x17'
    var_111 = module_0.base64_decode(var_110)
    var_112 = 'SGVsbG8\x18'
    var_113 = module_0.base64_decode(var_112)
    var_114 = 'SGVsbG8\x19'
    var_115 = module_0.base64_decode(var_114)
    var_116 = 'SGVsbG8\x1a'
    var_117 = module_0.base64_decode(var_116)
    var_118 = 'SGVsbG8\x1b'
    var_119 = module_0.base64_decode(var_118)
    var_120 = 'SGVsbG8\x1c'
    var_121 = module_0.base64_decode(var_120)
    var_122 = 'SGVsbG8\x1d'
    var_123 = module_0.base64_decode(var_122)
    var_124 = 'SGVsbG8\x1e'
    var_125 = module_0.base64_decode(var_124)
    var_126 = 'SGVsbG8\x1f'
    var_127 = module_0.base64_decode(var_126)
    var_128 = 'SGVsbG8\x7f'
    var_129 = module_0.base64_decode(var_128)
    var_130 = 'SGVsbG8\x80'
    var_131 = module_0.base64_decode(var_130)
    var_132 = 'SGVsbG8\x81'
    var_133 = module_0.base64_decode(var_132)
    var_134 = 'SGVsbG8\x82'
    var_135 = module_0.base64_decode(var_134)
    var_136 = 'SGVsbG8\x83'
    var_137 = module_0.base64_decode(var_136)
    var_138 = 'SGVsbG8\x84'
    var_139 = module_0.base64_decode(var_138)
    var_140 = 'SGVsbG8\x85'
    var_141 = module_0.base64_decode(var_140)
    var_142 = 'SGVsbG8\x86'
    var_143 = module_0.base64_decode(var_142)
    var_144 = 'SGVsbG8\x87'
    var_145 = module_0.base64_decode(var_144)
    var_146 = 'SGVsbG8\x88'
    var_147 = module_0.base64_decode(var_146)
    var_148 = 'SGVsbG8\x89'
    var_149 = module_0.base64_decode(var_148)
    var_150 = 'SGVsbG8\x8a'
    var_151 = module_0.base64_decode(var_150)
    var_152 = 'SGVsbG8\x8b'
    var_153 = module_0.base64_decode(var_152)
    var_154 = 'SGVsbG8\x8c'
    var_155 = module_0.base64_decode(var_154)
    var_156 = 'SGVsbG8\x8d'
    var_157 = module_0.base64_decode(var_156)
    var_158 = 'SGVsbG8\x8e'
    var_159 = module_0.base64_decode(var_158)
    var_160 = 'SGVsbG8\x8f'
    var_161 = module_0.base64_decode(var_160)
    var_162 = 'SGVsbG8\x90'
    var_163 = module_0.base64_decode(var_162)
    var_164 = 'SGVsbG8\x91'
    var_165 = module_0.base64_decode(var_164)



# Parsed testcases at query #29
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJyb250Zz4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<bront>'
    var_10 = 'PGJyb250Zz4'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bront>'
    var_12 = module_0.base64_decode(var_8)
    assert var_12 == b'<bront>'
    var_13 = module_0.base64_decode(var_10)
    assert var_13 == b'<bront>'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8='
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8=ÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = b'SGVsbG8=\xff'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = 'SGVsbG8=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = module_0.base64_decode(var_2)
    assert var_15 == b'Hello'
    var_16 = module_0.base64_decode(var_2)
    assert var_16 == b'Hello'
    var_17 = module_0.base64_decode(var_0)
    assert var_17 == b'Hello'
    var_18 = 'SGVsbG8-'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8_'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8@'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8#'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8=äöü'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = 'SGVsbG8==='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = 'SGVsbG8-V29ybGQ='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello-World'
    var_15 = 'SGVsbG8_V29ybGQ='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello_World'
    var_17 = b'SGVsbG8='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = b'SGVsbG8'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = ''
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b''
    var_23 = 'SGVsbG8!@#$'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'
    var_25 = 'InvalidBase64!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8=InvalidPadding='
    var_28 = module_0.base64_decode(var_27)



# Parsed testcases at query #32
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
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'SGVsbG8!#'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8ÿÿ'
    var_25 = module_0.base64_decode(var_24)



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
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = 'SGVsbG8gV29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'Hello World'
    var_7 = 'SGVsbG8gV29ybGQ'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello World'
    var_9 = ''
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b''
    var_11 = 'SGVsbG8-V29ybGQ'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello-World'
    var_13 = b'SGVsbG8='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = b'SGVsbG8'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = 'SGVsbG8!\n'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = 'Invalid@@'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8=='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = 'SGVsbG8=😊'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'PGJyb2FkY2FzdD4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<broadcast>'
    var_10 = 'PGJyb2FkY2FzdD4'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<broadcast>'
    var_12 = module_0.base64_decode(var_8)
    assert var_12 == b'<broadcast>'
    var_13 = 'aGVsbG8gd29ybGQ='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'hello world'
    var_15 = 'aGVsbG8gd29ybGQ'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'hello world'
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8==='
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8ÿ'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'



# Parsed testcases at query #35
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = module_0.base64_decode(var_2)
    assert var_12 == b'Hello'
    var_13 = module_0.base64_decode(var_0)
    assert var_13 == b'Hello'
    var_14 = 'SGVsbG8=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8@'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8#'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8=äöü'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = 'SGVsbG8=😊'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'



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
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_2)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_2)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_2)
    assert var_9 == b'Hello'
    var_10 = module_0.base64_decode(var_2)
    assert var_10 == b'Hello'
    var_11 = module_0.base64_decode(var_2)
    assert var_11 == b'Hello'
    var_12 = module_0.base64_decode(var_2)
    assert var_12 == b'Hello'
    var_13 = module_0.base64_decode(var_2)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = module_0.base64_decode(var_2)
    assert var_16 == b'Hello'
    var_17 = module_0.base64_decode(var_2)
    assert var_17 == b'Hello'
    var_18 = module_0.base64_decode(var_2)
    assert var_18 == b'Hello'
    var_19 = b'SGVsbG8='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = b'SGVsbG8'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = 'SGVsbG8!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8@'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8#'
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'SGVsbG8ÿ'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'
    var_31 = 'SGVsbG8\x00'
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'Hello'



# Parsed testcases at query #37
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
    var_6 = 'SGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZ25lcnM+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigners>'
    var_12 = 'PGJpZ25lcnM-'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigners>'
    var_14 = 'YQ=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = 'YWI='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'ab'
    var_18 = 'YWJj'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abc'
    var_20 = 'Invalid!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=\x00\x01'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = 'SGVsbG8=\x7f'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'



# Parsed testcases at query #38
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = module_0.base64_decode(var_2)
    var_10 = module_0.base64_decode(var_2)
    var_11 = module_0.base64_decode(var_2)
    var_12 = module_0.base64_decode(var_2)
    var_13 = module_0.base64_decode(var_2)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'YQ=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'a'
    var_20 = 'YQ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'a'
    var_22 = 'YWE='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'aa'
    var_24 = 'YWE'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'aa'
    var_26 = 'YWFh'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'aaa'
    var_28 = module_0.base64_decode(var_26)
    assert var_28 == b'aaa'
    var_29 = 'SGVsbG8!'
    var_30 = module_0.base64_decode(var_29)
    var_31 = 'SGVsbG8='
    var_32 = module_0.base64_decode(var_31)
    var_33 = 'SGVsbG8'
    var_34 = module_0.base64_decode(var_33)
    var_35 = 'SGVsbG8'
    var_36 = module_0.base64_decode(var_35)



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'Zg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'f'
    var_10 = 'Zm8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'fo'
    var_12 = 'Zm9v'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'foo'
    var_14 = 'Zm9vYg=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'foob'
    var_16 = 'Zm9vYmE='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'fooba'
    var_18 = 'Zm9vYmFy'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'foobar'
    var_20 = 'PGJhcj4='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'<bar>'
    var_22 = 'PGJhcj4'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'<bar>'
    var_24 = module_0.base64_decode(var_20)
    assert var_24 == b'<bar>'
    var_25 = module_0.base64_decode(var_22)
    assert var_25 == b'<bar>'
    var_26 = 'SGVsbG8!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8@'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8#'
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG8$'
    var_33 = module_0.base64_decode(var_32)
    var_34 = 'SGVsbG8=😊'
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'Hello'
    var_36 = 'SGVsbG8====='
    var_37 = module_0.base64_decode(var_36)
    assert var_37 == b'Hello'



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'PGJpZ2Zvb3Q+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = 'PGJpZ2Zvb3Q'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigfoot>'
    var_16 = 'YQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = 'YWE='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'aa'
    var_20 = 'YWFh'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'aaa'
    var_22 = ''
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b''
    var_24 = 'SGVsbG8!@#$'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = 'Invalid@@@'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8=!'
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG8ÿ'
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'Hello'



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
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = 'SGVsbG8gd29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'Hello world'
    var_7 = 'SGVsbG8gd29ybGQ'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello world'
    var_9 = module_0.base64_decode(var_5)
    assert var_9 == b'Hello world'
    var_10 = module_0.base64_decode(var_7)
    assert var_10 == b'Hello world'
    var_11 = module_0.base64_decode(var_5)
    assert var_11 == b'Hello world'
    var_12 = module_0.base64_decode(var_7)
    assert var_12 == b'Hello world'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b'SGVsbG8='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = b'SGVsbG8'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = 'Invalid!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8='
    var_24 = 1000
    var_25 = var_23 * var_24
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8=ÿ'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'
    var_29 = 'SGVsbG8ÿ'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'



# Parsed testcases at query #42
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'a'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xab'
    var_14 = 'ab'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xab'
    var_16 = 'abc'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xab'
    var_18 = 'PDw_Pz8-Pg'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'<<???>'
    var_20 = 'SGVsbG8!@#'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8='
    var_25 = 100
    var_26 = var_24 * var_25
    var_27 = '!'
    var_28 = var_26 + var_27
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8=äöü'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_0.base64_decode(var_2)
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello world'
    var_8 = 'SGVsbG8gd29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello world'
    var_10 = 'SGVsbG8gd29ybGQh'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello world!'
    var_12 = 'SGVsbG8gd29ybGQhIQ=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello world!!'
    var_14 = 'SGVsbG8gd29ybGQhIQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello world!!'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = b''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8='
    var_23 = 100
    var_24 = var_22 * var_23
    var_25 = '!'
    var_26 = var_24 + var_25
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8=ÿ'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'
    var_30 = 'SGVsbG8=ÿþ'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'



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
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8-V29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello-World'
    var_10 = 'SGVsbG8_V29ybGQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello_World'
    var_12 = b'SGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = b'SGVsbG8'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8='
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8==='
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #46
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
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'SGVsbG8gV29ybGQ-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello World+'
    var_12 = 'SGVsbG8gV29ybGQ_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello World/'
    var_14 = 'SGVsbG8ÿ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8@'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8#'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 32
    var_23 = b'='



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8_V29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello?World'
    var_10 = 'SGVsbG8-V29ybGQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello-World'
    var_12 = b'SGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = b'SGVsbG8'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!\n'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'Invalid@Base64'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8=='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'



# Parsed testcases at query #48
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
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello+'
    var_11 = 'SGVsbG8_'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello/'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = 'SGVsbG8!@#'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = b'SGVsbG8!@#'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = 'SGVsbG8!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8='
    var_24 = 1000
    var_25 = var_23 * var_24
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8é'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'
    var_29 = 'SGVsbG8é='
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8===='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8@'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8#'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8ÿ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = 'SGVsbG8\x00'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = 'VGhpcyBpcyBhIGxvbmdlciBzdHJpbmcgdGhhdCBzaG91bGQgYmUgdGVzdGVk'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'This is a longer string that should be tested'



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = b'SGVsbG8='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'Hello'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = module_0.base64_decode(var_2)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = 'SGVsbG8!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8@'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8$'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = 'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = 'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = b'SGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = 'PGJpZ25hbWU+'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<big name>'
    var_16 = module_0.base64_decode(var_14)
    assert var_16 == b'<big name>'
    var_17 = ''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''
    var_19 = '!!!invalid!!!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8='
    var_24 = 100
    var_25 = var_23 * var_24
    var_26 = '!'
    var_27 = var_25 + var_26
    var_28 = module_0.base64_decode(var_27)
    var_29 = module_0.base64_decode(var_25)
    assert var_29 == b'Hello'
    var_30 = 'SGVsbG8=='
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'
    var_32 = 'SGVsbG8=ÿ'
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'Hello'
    var_34 = 'SGVsbG8=ÿÿ'
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'Hello'



# Parsed testcases at query #52
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'Invalid!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = b'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!@#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = b'SGVsbG8!@#'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



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
    var_4 = 'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'PGJpZ25hbWU+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<big name>'
    var_14 = 'PGJpZ25hbWU'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<big name>'
    var_16 = '-_='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'\xfb\xff\xff'
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8#'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8$'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=ÿ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = 'SGVsbG8ÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'



# Parsed testcases at query #54
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
    var_6 = module_0.base64_decode(var_4)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_0)
    assert var_7 == b'Hello'
    var_8 = 'SGVsbG8=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'Invalid!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8ÿ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'



# Parsed testcases at query #55
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
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!\n'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'Invalid!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'This is a longer string to test base64 decoding.'
    var_23 = module_1.encode()
    var_24 = module_1.urlsafe_b64encode(var_23)
    var_25 = b'='
    var_26 = module_1.encode()



# Parsed testcases at query #56
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
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello\xff'
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello\xfb'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!\x00'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = b'SGVsbG8!\x00'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8\x00'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8='
    var_27 = 100
    var_28 = var_26 * var_27
    var_29 = module_0.base64_decode(var_28)



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
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = 'SGVsbG8_'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'Invalid!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8ÿ'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



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
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8-V29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello-World'
    var_10 = 'SGVsbG8_V29ybGQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello_World'
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = 'SGVsbG8=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = 'SGVsbG8==='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = b'SGVsbG8='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = b'SGVsbG8'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = 'Invalid!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8=😊'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'
    var_27 = 'SGVsbG8😊'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'
    var_29 = ''
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b''
    var_31 = b''
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b''



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
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8==='
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8=ÿ'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8===='
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8=😊'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello world'
    var_8 = 'SGVsbG8gd29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello world'
    var_10 = module_0.base64_decode(var_6)
    assert var_10 == b'Hello world'
    var_11 = module_0.base64_decode(var_8)
    assert var_11 == b'Hello world'
    var_12 = module_0.base64_decode(var_6)
    assert var_12 == b'Hello world'
    var_13 = module_0.base64_decode(var_8)
    assert var_13 == b'Hello world'
    var_14 = module_0.base64_decode(var_0)
    assert var_14 == b'Hello'
    var_15 = module_0.base64_decode(var_2)
    assert var_15 == b'Hello'
    var_16 = module_0.base64_decode(var_6)
    assert var_16 == b'Hello world'
    var_17 = module_0.base64_decode(var_8)
    assert var_17 == b'Hello world'
    var_18 = 'SGVsbG8gd29ybGQ-'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello world+'
    var_20 = 'SGVsbG8gd29ybGQ_'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello world/'
    var_22 = ''
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b''
    var_24 = 'Invalid!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8='
    var_29 = 100
    var_30 = var_28 * var_29
    var_31 = '!'
    var_32 = var_30 + var_31
    var_33 = module_0.base64_decode(var_32)
    var_34 = 'SGVsbG8gd29ybGQ=ÿ'
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'Hello world'



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
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8gV29ybGQh'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello World!'
    var_10 = 'SGVsbG8gV29ybGQh-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello World!-'
    var_12 = b'SGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = b'SGVsbG8'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!\x00'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'Invalid!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8='
    var_25 = 100
    var_26 = var_24 * var_25
    var_27 = '!'
    var_28 = var_26 + var_27
    var_29 = module_0.base64_decode(var_28)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = 'SGVsbG8==='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8-V29ybGQ='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello-World'
    var_17 = 'SGVsbG8_V29ybGQ='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello_World'
    var_19 = b'SGVsbG8='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = b'SGVsbG8'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = 'Invalid@Base64'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8=ÿ'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'



# Parsed testcases at query #8
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = 'aQ=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'i'
    var_19 = 'aWE='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'ia'
    var_21 = 'aWFi'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'iab'
    var_23 = 'SGVsbG8!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8@'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8#'
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'SGVsbG8ÿ'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'
    var_31 = 'SGVsbG8\x00'
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'PGJpZ2Zvb3Q+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = 'PGJpZ2Zvb3Q'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigfoot>'
    var_16 = 'SGVsbG8!@#'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = 'SGVsbG'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hell'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8='
    var_23 = 1000
    var_24 = var_22 * var_23
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #10
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
    var_6 = 'SGVsbG8gV29ybGQh'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World!'
    var_8 = 'PGJyPg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<br>'
    var_10 = 'PGJyPg'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<br>'
    var_12 = module_0.base64_decode(var_8)
    assert var_12 == b'<br>'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'Invalid!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8=😊'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8gV29ybGQh'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello World!'
    var_14 = 'SGVsbG8gV29ybGQh-'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello World!'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = module_0.base64_decode(var_12)
    assert var_18 == b'Hello World!'
    var_19 = 'SGVsbG8gV29ybGQh=='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello World!'
    var_21 = 'SGVsbG8gV29ybGQh='
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello World!'
    var_23 = 'Invalid!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8gV29ybGQ!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8gV29ybGQ!\x00'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello World!'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8gd29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello world'
    var_6 = b'SGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJyIC8+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<br />'
    var_12 = 'PGJyL14='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<br/~'
    var_14 = 'SGVsbG8!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8=😊'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



# Parsed testcases at query #13
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8___'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'Invalid!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 123
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8ÿ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = 'SGVsbG8\x00\x01'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_2)
    assert var_9 == b'Hello'
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = 'Invalid!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'SGVsbG8!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8ÿ'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'



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
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = module_0.base64_decode(var_0)
    var_9 = module_0.base64_decode(var_2)
    var_10 = 'SGVsbG8-V29ybGQ='
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8-V29ybGQ'
    var_13 = module_0.base64_decode(var_12)
    var_14 = b'SGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = b'SGVsbG8'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = ''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = 'Invalid!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=ÿ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8-'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'SGVsbG8_'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'SGVsbG8!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8==='
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8!'
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG8!'
    var_33 = module_0.base64_decode(var_32)
    var_34 = 'SGVsbG8!'
    var_35 = module_0.base64_decode(var_34)



# Parsed testcases at query #17
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'SGVsbG8!@#'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = b'SGVsbG8!@#'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = b'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hell'
    var_22 = b'SGVsbG'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hell'
    var_24 = module_0.base64_decode(var_18)
    var_25 = module_0.base64_decode(var_18)
    var_26 = module_0.base64_decode(var_2)
    var_27 = module_0.base64_decode(var_2)
    var_28 = 'SGVsbG8=ñ'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'
    var_30 = b'SGVsbG8=\xc3\xb1'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = 'aGVsbG8='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = 'aGVsbG8'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello'
    var_10 = 'SGVsbG8h'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello!'
    var_12 = module_0.base64_decode(var_10)
    assert var_12 == b'Hello!'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'Zg=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'f'
    var_17 = 'Zg'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'f'
    var_19 = module_0.base64_decode(var_0)
    assert var_19 == b'Hello'
    var_20 = module_0.base64_decode(var_2)
    assert var_20 == b'Hello'
    var_21 = 'SGVsbG8-'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = 'SGVsbG8_'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'
    var_25 = b'SGVsbG8='
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'
    var_27 = b'SGVsbG8'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'
    var_29 = b'SGVsbG8-'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'
    var_31 = b'SGVsbG8_'
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'Hello'
    var_33 = 'SGVsbG8!'
    var_34 = module_0.base64_decode(var_33)
    var_35 = 'SGVsbG8#'
    var_36 = module_0.base64_decode(var_35)
    var_37 = 'SGVsbG8$'
    var_38 = module_0.base64_decode(var_37)
    var_39 = 'SGVsbG8%'
    var_40 = module_0.base64_decode(var_39)
    var_41 = 'SGVsbG8&'
    var_42 = module_0.base64_decode(var_41)
    var_43 = 'SGVsbG8*'
    var_44 = module_0.base64_decode(var_43)
    var_45 = 'SGVsbG8+'
    var_46 = module_0.base64_decode(var_45)
    var_47 = 'SGVsbG8/'
    var_48 = module_0.base64_decode(var_47)
    var_49 = 'SGVsbG8:'
    var_50 = module_0.base64_decode(var_49)



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8h'
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello!'
    var_2 = b'SGVsbG8h'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello!'
    var_4 = 'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = 'aGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = module_0.base64_decode(var_0)
    var_10 = module_0.base64_decode(var_0)
    var_11 = module_0.base64_decode(var_0)
    var_12 = '-__'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\xff\xff'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8h=='
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8h='
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8h==='
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8hÿ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello!'
    var_26 = 'SGVsbG8h\x00'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello!'



# Parsed testcases at query #20
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'PGJpZ2Zvb3Q+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = 'PGJpZ2Zvb3Q'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigfoot>'
    var_16 = 'SGVsbG8h'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello!'
    var_18 = 'SGVsbG8_'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello_'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8@'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8#'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8ÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = 'SGVsbG8\x00'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'
    var_30 = 'SGVsbG8===='
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'
    var_32 = 'SGVsbG8====='
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = module_0.base64_decode(var_2)
    assert var_11 == b'Hello'
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = module_0.base64_decode(var_2)
    assert var_13 == b'Hello'
    var_14 = 'Invalid@Base64'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8=😊'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8😊'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #22
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
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello-'
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello_'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!@#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8\x00\x01'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8ÿ'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8é'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = 'SGVsbG8_'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'Invalid!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8ÿ'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = 'SGVsbG8gV29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'Hello World'
    var_7 = 'SGVsbG8gV29ybGQ'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello World'
    var_9 = b'SGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = b'SGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = 'PGJyPg=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'<br>'
    var_15 = 'PGJyPg'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'<br>'
    var_17 = module_0.base64_decode(var_13)
    assert var_17 == b'<br>'
    var_18 = module_0.base64_decode(var_15)
    assert var_18 == b'<br>'
    var_19 = ''
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b''
    var_21 = 'Invalid!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8=ÿ'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'



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
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = ''
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b''
    var_7 = 'Zg=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'f'
    var_9 = 'Zm8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'fo'
    var_11 = 'Zm9v'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'foo'
    var_13 = 'Zm9vYg=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'foob'
    var_15 = 'Zm9vYmE='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'fooba'
    var_17 = 'Zm9vYmFy'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'foobar'
    var_19 = b'SGVsbG8='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = b'SGVsbG8'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = 'PGJyPg=='
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'<br>'
    var_25 = 'PGJyPg'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'<br>'
    var_27 = module_0.base64_decode(var_23)
    assert var_27 == b'<br>'
    var_28 = 'Invalid!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8!'
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG8=!'
    var_33 = module_0.base64_decode(var_32)
    var_34 = 'SGVsbG8=äöü'
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'Hello'
    var_36 = 'SGVsbG8=😊'
    var_37 = module_0.base64_decode(var_36)
    assert var_37 == b'Hello'



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
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_2)
    assert var_7 == b'Hello'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
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
    var_20 = module_0.base64_decode(var_0)
    assert var_20 == b'Hello'
    var_21 = module_0.base64_decode(var_2)
    assert var_21 == b'Hello'
    var_22 = module_0.base64_decode(var_0)
    assert var_22 == b'Hello'
    var_23 = module_0.base64_decode(var_2)
    assert var_23 == b'Hello'
    var_24 = 'SGVsbG8!@#$'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = module_0.base64_decode(var_24)
    assert var_26 == b'Hello'
    var_27 = 'SGVsbG8é'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'
    var_29 = module_0.base64_decode(var_27)
    assert var_29 == b'Hello'
    var_30 = 'SGVsbG8!'
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG8!'
    var_33 = module_0.base64_decode(var_32)
    var_34 = 'SGVsbG8!'
    var_35 = module_0.base64_decode(var_34)
    var_36 = 'SGVsbG8!'
    var_37 = module_0.base64_decode(var_36)
    var_38 = 'SGVsbG8!'
    var_39 = module_0.base64_decode(var_38)
    var_40 = 'SGVsbG8!'
    var_41 = module_0.base64_decode(var_40)
    var_42 = 'SGVsbG8!'
    var_43 = module_0.base64_decode(var_42)
    var_44 = 'SGVsbG8!'
    var_45 = module_0.base64_decode(var_44)
    var_46 = 'SGVsbG8!'
    var_47 = module_0.base64_decode(var_46)
    var_48 = 'SGVsbG8!'
    var_49 = module_0.base64_decode(var_48)



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_2)
    assert var_9 == b'Hello'
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = module_0.base64_decode(var_2)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8!@#'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = b'SGVsbG8!@#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8='
    var_25 = module_0.base64_decode(var_24)
    var_26 = b'SGVsbG8='
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_2)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = 'SGVsbG8==='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = module_0.base64_decode(var_2)
    assert var_15 == b'Hello'
    var_16 = module_0.base64_decode(var_2)
    assert var_16 == b'Hello'
    var_17 = module_0.base64_decode(var_2)
    assert var_17 == b'Hello'
    var_18 = 'SGVsbG8!@#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = b'SGVsbG8='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = b'SGVsbG8'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = 'SGVsbG8!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8!@#$'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8!@#$%^&*()'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_33 = module_0.base64_decode(var_32)
    var_34 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_35 = module_0.base64_decode(var_34)
    var_36 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_37 = module_0.base64_decode(var_36)
    var_38 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_39 = module_0.base64_decode(var_38)
    var_40 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_41 = module_0.base64_decode(var_40)
    var_42 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_43 = module_0.base64_decode(var_42)
    var_44 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_45 = module_0.base64_decode(var_44)
    var_46 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_47 = module_0.base64_decode(var_46)
    var_48 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_49 = module_0.base64_decode(var_48)
    var_50 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_51 = module_0.base64_decode(var_50)
    var_52 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_53 = module_0.base64_decode(var_52)
    var_54 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_55 = module_0.base64_decode(var_54)
    var_56 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_57 = module_0.base64_decode(var_56)
    var_58 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_59 = module_0.base64_decode(var_58)
    var_60 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_61 = module_0.base64_decode(var_60)
    var_62 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_63 = module_0.base64_decode(var_62)
    var_64 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_65 = module_0.base64_decode(var_64)
    var_66 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_67 = module_0.base64_decode(var_66)
    var_68 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_69 = module_0.base64_decode(var_68)
    var_70 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_71 = module_0.base64_decode(var_70)
    var_72 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_73 = module_0.base64_decode(var_72)
    var_74 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_75 = module_0.base64_decode(var_74)
    var_76 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_77 = module_0.base64_decode(var_76)
    var_78 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_79 = module_0.base64_decode(var_78)
    var_80 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_81 = module_0.base64_decode(var_80)
    var_82 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_83 = module_0.base64_decode(var_82)
    var_84 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_85 = module_0.base64_decode(var_84)
    var_86 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_87 = module_0.base64_decode(var_86)
    var_88 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_89 = module_0.base64_decode(var_88)
    var_90 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_91 = module_0.base64_decode(var_90)
    var_92 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_93 = module_0.base64_decode(var_92)
    var_94 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_95 = module_0.base64_decode(var_94)
    var_96 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_97 = module_0.base64_decode(var_96)
    var_98 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_99 = module_0.base64_decode(var_98)
    var_100 = 'SGVsbG8!@#$%^&*()_+-=[]{};:\'",./<>?`~'
    var_101 = module_0.base64_decode(var_100)



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
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8gV29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello World'
    var_10 = 'SGVsbG8gV29ybGQh'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello World!'
    var_12 = 'SGVsbG8-V29ybGQh'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello+World!'
    var_14 = 'SGVsbG8_V29ybGQh'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello/World!'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = b''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = 'YQ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'a'
    var_22 = 'YWE'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'aa'
    var_24 = 'YWFh'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'aaa'
    var_26 = 'SGVsbG8!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8='
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8==='
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG8=ÿ'
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'Hello'



# Parsed testcases at query #30
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
    var_8 = 'SGVsbG8_'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello?'
    var_10 = 'SGVsbG8=😊'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'Invalid!'
    var_13 = module_0.base64_decode(var_12)
    var_14 = 'SGVsbG8==='
    var_15 = module_0.base64_decode(var_14)



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'SGVsbG8!'
    var_9 = module_0.base64_decode(var_8)
    var_10 = 'SGVsbG8==='
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8=ÿ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ' SGVsbG8= '
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = 'SGVs\nbG8='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'aGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZ25hbWU+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<big name>'
    var_12 = 'PGJpZ25hbWU'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<big name>'
    var_14 = 'YWJjZGVmZ2g='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abcdefgh'
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8#'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 123
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8=ÿ'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = 'SGVsbG8=ÿþ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJhcj4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<bar>'
    var_10 = 'PGJhcj4'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bar>'
    var_12 = module_0.base64_decode(var_8)
    assert var_12 == b'<bar>'
    var_13 = module_0.base64_decode(var_10)
    assert var_13 == b'<bar>'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8@'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8#'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=ÿ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = 'SGVsbG8ÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'aGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZ25hbWU+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<big name>'
    var_12 = 'PGJpZ25hbWU'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<big name>'
    var_14 = 'YWJjZGVmZw=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abcdefg'
    var_16 = 'YQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = 'YWE='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'aa'
    var_20 = 'YWFh'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'aaa'
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8==='
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8=ÿ'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'
    var_32 = 'SGVsbG8=ÿþ'
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'Hello'



# Parsed testcases at query #35
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8!@#'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = b'SGVsbG8!@#'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = 'SGVsbG'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = b'SGVsbG'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = 'SGVsbG8!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'SGVsbG8!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8é'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'
    var_29 = b'SGVsbG8\xc3\xa9'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = b'SGVsbG8gd29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = 'SGVsbG8gd29ybGQ'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello world'
    var_6 = 'SGVsbG8gd29ybGQ=='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello world'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello world'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello world'
    var_10 = module_0.base64_decode(var_6)
    assert var_10 == b'Hello world'
    var_11 = 'SGVsbG8gd29ybGQ-'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello world'
    var_13 = 'SGVsbG8gd29ybGQ_'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello world'
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = 'SGVsbG8gd29ybGQ!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8gd29ybGQ=ÿ'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello world'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'a'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'\x80'
    var_12 = 'aa'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\x80\x80'
    var_14 = 'aaa'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\x80\x80\x80'
    var_16 = 'PGJpZ2Zvb3Q+'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'<bigfoot>'
    var_18 = module_0.base64_decode(var_16)
    assert var_18 == b'<bigfoot>'
    var_19 = 'SGVsbG8!@#$'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = 'SGVsbG8!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8='
    var_24 = 1000
    var_25 = var_23 * var_24
    var_26 = '!'
    var_27 = var_25 + var_26
    var_28 = module_0.base64_decode(var_27)
    var_29 = 'SGVsbG8ÿ'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'



# Parsed testcases at query #38
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = module_0.base64_decode(var_2)
    assert var_13 == b'Hello'
    var_14 = 'SGVsbG8-'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello\x00'
    var_16 = 'SGVsbG8_'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello\x00'
    var_18 = 'SGVsbG8!@#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8=!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8==!'
    var_25 = module_0.base64_decode(var_24)



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
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8-V29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello-World'
    var_10 = 'SGVsbG8_V29ybGQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello_World'
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = 'SGVsbG8=='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = 'SGVsbG8==='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = b'SGVsbG8='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = b'SGVsbG8'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = ''
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b''
    var_23 = 'Invalid@Base64'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8=ÿ'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'



# Parsed testcases at query #40
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
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJvZHk+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<body>'
    var_12 = 'PGJvZHk-'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<body>'
    var_14 = 'PGJvZHk_'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<body>'
    var_16 = 'Invalid!@#$'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = module_0.base64_decode(var_20)
    assert var_22 == b'Hello'
    var_23 = module_0.base64_decode(var_4)
    assert var_23 == b'Hello'
    var_24 = module_0.base64_decode(var_18)
    assert var_24 == b'Hello'
    var_25 = 'SGVsbG8=='
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'PGJpZ2Zvb3Q+'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<bigfoot>'
    var_10 = 'PGJpZ2Zvb3Q'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigfoot>'
    var_12 = 'YWJjX2RlZg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abc_def'
    var_14 = 'YWJjX2RlZg'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc_def'
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8='
    var_19 = module_0.base64_decode(var_18)
    var_20 = 123
    var_21 = module_0.base64_decode(var_20)



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = 'SGVsbG8=='
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = 'SGVsbG8-V29ybGQ='
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello-World'
    var_15 = 'SGVsbG8_V29ybGQ='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello_World'
    var_17 = b'SGVsbG8='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = b'SGVsbG8'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = 'Invalid@Base64'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8=ÿ'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'
    var_27 = 'SGVsbG8=ÿþ'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJvZHk+'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<body>'
    var_10 = 'PGJvZHk'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<body>'
    var_12 = 'PGJvZHk-Pg=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<body>'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'YQ=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'a'
    var_20 = 'YWE='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'aa'
    var_22 = 'YWFh'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'aaa'
    var_24 = 'YWFhYQ=='
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'aaaa'
    var_26 = 'SGVsbG8!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8!='
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8===='
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG'
    var_33 = module_0.base64_decode(var_32)



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8= '
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ' SGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8==!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8= ='
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8= = ='
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8= = = ='
    var_29 = module_0.base64_decode(var_28)



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
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = module_0.base64_decode(var_6)
    assert var_12 == b'Hello World'
    var_13 = 'SGVsbG8-V29ybGQ'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello World'
    var_15 = 'SGVsbG8_V29ybGQ'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello World'
    var_17 = ''
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b''
    var_19 = module_0.base64_decode(var_0)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8=='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'Invalid!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8=!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8gV29ybGQÿ'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello World'



# Parsed testcases at query #46
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8!@#'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = b'SGVsbG8!@#'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = 'SGVsbG8!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = b'SGVsbG8!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8==='
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'SGVsbG8==='
    var_26 = module_0.base64_decode(var_25)



# Parsed testcases at query #47
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJpZ2Zvb3Q+'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<bigfoot>'
    var_10 = 'PGJpZ2Zvb3Q'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigfoot>'
    var_12 = 'PGJpZ2Zvb3Q='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = 'PGJpZ2Zvb3Q=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigfoot>'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = b''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8@'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8#'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8ÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = 'SGVsbG8\x00'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'SGVsbG8gd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = 'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8====='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b'SGVsbG8gd29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello world'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'SGVsbG8gd29ybGQ!@#'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello world'
    var_14 = 'Invalid@@@'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8gd29ybGQ===='
    var_17 = module_0.base64_decode(var_16)



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
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_2)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_2)
    assert var_9 == b'Hello'
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = module_0.base64_decode(var_2)
    assert var_11 == b'Hello'
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = module_0.base64_decode(var_2)
    assert var_13 == b'Hello'
    var_14 = module_0.base64_decode(var_0)
    assert var_14 == b'Hello'
    var_15 = module_0.base64_decode(var_2)
    assert var_15 == b'Hello'
    var_16 = module_0.base64_decode(var_0)
    assert var_16 == b'Hello'
    var_17 = module_0.base64_decode(var_2)
    assert var_17 == b'Hello'
    var_18 = module_0.base64_decode(var_0)
    assert var_18 == b'Hello'
    var_19 = module_0.base64_decode(var_2)
    assert var_19 == b'Hello'
    var_20 = module_0.base64_decode(var_0)
    assert var_20 == b'Hello'
    var_21 = module_0.base64_decode(var_2)
    assert var_21 == b'Hello'
    var_22 = module_0.base64_decode(var_0)
    assert var_22 == b'Hello'
    var_23 = module_0.base64_decode(var_2)
    assert var_23 == b'Hello'
    var_24 = module_0.base64_decode(var_0)
    assert var_24 == b'Hello'
    var_25 = module_0.base64_decode(var_2)
    assert var_25 == b'Hello'
    var_26 = module_0.base64_decode(var_0)
    assert var_26 == b'Hello'
    var_27 = module_0.base64_decode(var_2)
    assert var_27 == b'Hello'
    var_28 = module_0.base64_decode(var_0)
    assert var_28 == b'Hello'
    var_29 = module_0.base64_decode(var_2)
    assert var_29 == b'Hello'
    var_30 = module_0.base64_decode(var_0)
    assert var_30 == b'Hello'
    var_31 = module_0.base64_decode(var_2)
    assert var_31 == b'Hello'
    var_32 = module_0.base64_decode(var_0)
    assert var_32 == b'Hello'
    var_33 = module_0.base64_decode(var_2)
    assert var_33 == b'Hello'
    var_34 = module_0.base64_decode(var_0)
    assert var_34 == b'Hello'
    var_35 = module_0.base64_decode(var_2)
    assert var_35 == b'Hello'
    var_36 = module_0.base64_decode(var_0)
    assert var_36 == b'Hello'
    var_37 = module_0.base64_decode(var_2)
    assert var_37 == b'Hello'
    var_38 = module_0.base64_decode(var_0)
    assert var_38 == b'Hello'
    var_39 = module_0.base64_decode(var_2)
    assert var_39 == b'Hello'
    var_40 = module_0.base64_decode(var_0)
    assert var_40 == b'Hello'
    var_41 = module_0.base64_decode(var_2)
    assert var_41 == b'Hello'
    var_42 = module_0.base64_decode(var_0)
    assert var_42 == b'Hello'
    var_43 = module_0.base64_decode(var_2)
    assert var_43 == b'Hello'
    var_44 = module_0.base64_decode(var_0)
    assert var_44 == b'Hello'
    var_45 = module_0.base64_decode(var_2)
    assert var_45 == b'Hello'
    var_46 = module_0.base64_decode(var_0)
    assert var_46 == b'Hello'
    var_47 = module_0.base64_decode(var_2)
    assert var_47 == b'Hello'
    var_48 = module_0.base64_decode(var_0)
    assert var_48 == b'Hello'
    var_49 = module_0.base64_decode(var_2)
    assert var_49 == b'Hello'
    var_50 = module_0.base64_decode(var_0)
    assert var_50 == b'Hello'
    var_51 = module_0.base64_decode(var_2)
    assert var_51 == b'Hello'
    var_52 = module_0.base64_decode(var_0)
    assert var_52 == b'Hello'
    var_53 = module_0.base64_decode(var_2)
    assert var_53 == b'Hello'
    var_54 = module_0.base64_decode(var_0)
    assert var_54 == b'Hello'
    var_55 = module_0.base64_decode(var_2)
    assert var_55 == b'Hello'
    var_56 = module_0.base64_decode(var_0)
    assert var_56 == b'Hello'
    var_57 = module_0.base64_decode(var_2)
    assert var_57 == b'Hello'
    var_58 = module_0.base64_decode(var_0)
    assert var_58 == b'Hello'
    var_59 = module_0.base64_decode(var_2)
    assert var_59 == b'Hello'
    var_60 = module_0.base64_decode(var_0)
    assert var_60 == b'Hello'
    var_61 = module_0.base64_decode(var_2)
    assert var_61 == b'Hello'
    var_62 = module_0.base64_decode(var_0)
    assert var_62 == b'Hello'
    var_63 = module_0.base64_decode(var_2)
    assert var_63 == b'Hello'
    var_64 = module_0.base64_decode(var_0)
    assert var_64 == b'Hello'
    var_65 = module_0.base64_decode(var_2)
    assert var_65 == b'Hello'
    var_66 = module_0.base64_decode(var_0)
    assert var_66 == b'Hello'
    var_67 = module_0.base64_decode(var_2)
    assert var_67 == b'Hello'
    var_68 = module_0.base64_decode(var_0)
    assert var_68 == b'Hello'
    var_69 = module_0.base64_decode(var_2)
    assert var_69 == b'Hello'
    var_70 = module_0.base64_decode(var_0)
    assert var_70 == b'Hello'
    var_71 = module_0.base64_decode(var_2)
    assert var_71 == b'Hello'
    var_72 = module_0.base64_decode(var_0)
    assert var_72 == b'Hello'
    var_73 = module_0.base64_decode(var_2)
    assert var_73 == b'Hello'
    var_74 = module_0.base64_decode(var_0)
    assert var_74 == b'Hello'
    var_75 = module_0.base64_decode(var_2)
    assert var_75 == b'Hello'
    var_76 = module_0.base64_decode(var_0)
    assert var_76 == b'Hello'
    var_77 = module_0.base64_decode(var_2)
    assert var_77 == b'Hello'
    var_78 = module_0.base64_decode(var_0)
    assert var_78 == b'Hello'
    var_79 = module_0.base64_decode(var_2)
    assert var_79 == b'Hello'
    var_80 = module_0.base64_decode(var_0)
    assert var_80 == b'Hello'
    var_81 = module_0.base64_decode(var_2)
    assert var_81 == b'Hello'
    var_82 = module_0.base64_decode(var_0)
    assert var_82 == b'Hello'
    var_83 = module_0.base64_decode(var_2)
    assert var_83 == b'Hello'
    var_84 = module_0.base64_decode(var_0)
    assert var_84 == b'Hello'
    var_85 = module_0.base64_decode(var_2)
    assert var_85 == b'Hello'
    var_86 = module_0.base64_decode(var_0)
    assert var_86 == b'Hello'
    var_87 = module_0.base64_decode(var_2)
    assert var_87 == b'Hello'



# Parsed testcases at query #50
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'PGJpZ2Zvb3Q+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = 'PGJpZ2Zvb3Q'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigfoot>'
    var_16 = 'SGVsbG8!@#'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = 'SGVsbG'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hell'
    var_20 = 'Invalid!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'Invalid!'
    var_23 = module_0.base64_decode(var_22)



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = 'SGVsbG8gV29ybGQ='
    var_6 = module_0.base64_decode(var_5)
    assert var_6 == b'Hello World'
    var_7 = 'SGVsbG8gV29ybGQh'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello World!'
    var_9 = module_0.base64_decode(var_7)
    assert var_9 == b'Hello World!'
    var_10 = b'SGVsbG8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = b'SGVsbG8'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8@'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = 'SGVsbG8#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'Invalid!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = module_0.base64_decode(var_2)
    assert var_24 == b'Hello'
    var_25 = module_0.base64_decode(var_22)
    assert var_25 == b'Hello'
    var_26 = 'SGVsbG8=='
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = 'SGVsbG8ÿ'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'



# Parsed testcases at query #52
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJhcmZvbz4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<barfoo>'
    var_10 = 'PGJhcmZvbz4'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<barfoo>'
    var_12 = 'PGJhcmZvb-'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<barfoo>'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8#'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8$'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8ÿ'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = 'SGVsbG8\x00'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'



# Parsed testcases at query #53
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
    var_6 = module_0.base64_decode(var_4)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_0)
    assert var_7 == b'Hello'
    var_8 = 'SGVsbG8=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'SGVsbG8!'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = '!!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8===='
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #54
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8gV29ybGQ='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello World'
    var_4 = 'SGVsbG8gV29ybGQh'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World!'
    var_6 = 'SGVsbG8-V29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello-World'
    var_8 = 'SGVsbG8_V29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello_World'
    var_10 = 'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8gV29ybGQ'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello World'
    var_14 = b'SGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = b'SGVsbG8gV29ybGQ='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello World'
    var_18 = ''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = 'SGVsbG8!\n\r='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'Invalid@@Base64'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8===='
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #55
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_2)
    assert var_7 == b'Hello'
    var_8 = 'aGVsbG8gd29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'hello world'
    var_10 = 'aGVsbG8gd29ybGQ'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'hello world'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b'SGVsbG8='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = b'SGVsbG8'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = 'PGJyPg=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'<br>'
    var_20 = 'PGJyPg'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'<br>'
    var_22 = module_0.base64_decode(var_18)
    assert var_22 == b'<br>'
    var_23 = module_0.base64_decode(var_20)
    assert var_23 == b'<br>'
    var_24 = 'Invalid@Base64'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8='
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8=😊'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'
    var_32 = 'SGVsbG8😊'
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'Hello'



# Parsed testcases at query #56
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8gV29ybGQ='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello World'
    var_6 = 'SGVsbG8gV29ybGQ'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'PGJyb2tlbiBieWU9ImZvbyI+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<broken bye="foo">'
    var_14 = module_0.base64_decode(var_12)
    assert var_14 == b'<broken bye="foo">'
    var_15 = ''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = 'Invalid!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8=ÿ'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'



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
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJpZ2Zvb3Q+'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<bigfoot>'
    var_10 = 'PGJpZ2Zvb3Q'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigfoot>'
    var_12 = 'PGJpZ2Zvb3Q='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = 'PGJpZ2Zvb3Q=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigfoot>'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = b''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = 'SGVsbG8=😊'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'SGVsbG8=äöü'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = 'SGVsbG8!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8=!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8==!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8===='
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG8=1'
    var_33 = module_0.base64_decode(var_32)



# Parsed testcases at query #58
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
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = b''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8!@#'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = b'SGVsbG8!@#'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = 'SGVsbG8é'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = b'SGVsbG8\xc3\xa9'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = 'SGVsbG8!'
    var_24 = module_0.base64_decode(var_23)
    var_25 = b'SGVsbG8!'
    var_26 = module_0.base64_decode(var_25)
    var_27 = 'SGVsbG8===='
    var_28 = module_0.base64_decode(var_27)
    var_29 = b'SGVsbG8===='
    var_30 = module_0.base64_decode(var_29)



# Parsed testcases at query #59
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
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8gV29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello World'
    var_10 = 'SGVsbG8gV29ybGQh'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello World!'
    var_12 = module_0.base64_decode(var_10)
    assert var_12 == b'Hello World!'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8gV29ybGQ!\n'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello World!'
    var_17 = 'Invalid!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8gV29ybGQ!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8gV29ybGQ==='
    var_22 = module_0.base64_decode(var_21)



