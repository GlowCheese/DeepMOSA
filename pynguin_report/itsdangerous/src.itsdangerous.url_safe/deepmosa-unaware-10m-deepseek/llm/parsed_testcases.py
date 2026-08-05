####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJ0ZXN0IjogImRhdGEifQ=='
    var_2 = var_0.load_payload(var_1)
    var_3 = b'.eJxLy8lPUtCvTi0qLsnMz1MoSczJzUtUqMkvSkxPVSgqzcxLBQBqZg5y'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = 100
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = var_0.load_payload(var_3)
    var_11 = b'invalid-base64!!!'
    var_12 = var_0.load_payload(var_11)
    var_13 = b'.dGVzdA=='
    var_14 = var_0.load_payload(var_13)
    var_15 = b''
    var_16 = var_0.load_payload(var_15)
    var_17 = b'.'
    var_18 = var_0.load_payload(var_17)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = module_1.base64_decode(var_4)
    assert var_6 == b'{"key":"value"}'
    var_7 = 'data'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = var_0.dump_payload(var_11)
    var_13 = 1
    var_14 = var_12[var_13:]
    var_15 = module_1.base64_decode(var_14)
    var_16 = {}
    var_17 = var_0.dump_payload(var_16)
    var_18 = 'number'
    var_19 = 'float'
    var_20 = 42
    var_21 = 3.14
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = var_0.dump_payload(var_22)
    var_24 = module_1.base64_decode(var_23)
    var_25 = 2
    var_26 = 3
    var_27 = 4
    var_28 = 5
    var_29 = [var_13, var_25, var_26, var_27, var_28]
    var_30 = var_0.dump_payload(var_29)
    var_31 = module_1.base64_decode(var_30)
    assert var_31 == b'[1,2,3,4,5]'
    var_32 = 20
    var_33 = var_8 * var_32
    var_34 = {var_7: var_33}
    var_35 = var_0.dump_payload(var_34)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = {}
    var_12 = None
    var_13 = {var_3: var_12}
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16, var_0]



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = var_0.dump_payload(var_7)
    var_9 = b'.'
    var_10 = 'short'
    var_11 = var_0.dump_payload(var_10)
    var_12 = len(var_8)
    var_13 = len(var_11)
    var_14 = 1
    var_15 = var_13 + var_14
    var_16 = {}
    var_17 = var_0.dump_payload(var_16)
    var_18 = 'key'
    var_19 = 'number'
    var_20 = 'value'
    var_21 = 42
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = 'a'
    var_24 = 500
    var_25 = var_23 * var_24



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_1: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = 1
    var_12 = var_10[var_11:]
    var_13 = module_1.base64_decode(var_12)
    var_14 = {}
    var_15 = var_0.dump_payload(var_14)
    var_16 = 'nested'
    var_17 = 'list'
    var_18 = 2
    var_19 = 3
    var_20 = [var_11, var_18, var_19]
    var_21 = 'test'
    var_22 = {var_17: var_20, var_2: var_21}
    var_23 = {var_16: var_22}
    var_24 = var_0.dump_payload(var_23)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'short data'
    var_1 = b'a'
    var_2 = 'x'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = b'.'
    var_6 = 1
    var_7 = 'abc123'
    var_8 = 10
    var_9 = var_7 * var_8
    var_10 = ''



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key": "value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = 'data'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = module_1._CompactJSON()
    var_9 = b'.'
    var_10 = b'invalid-base64!!!'
    var_11 = b'not-compressed'
    var_12 = module_0.base64_encode(var_11)
    var_13 = var_9 + var_12



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = 'x'
    var_4 = 1000
    var_5 = var_3 * var_4
    var_6 = {var_1: var_5}
    var_7 = b'.'
    var_8 = 1
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = {}
    var_13 = 2
    var_14 = 3
    var_15 = [var_8, var_13, var_14]



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'.'
    var_12 = var_0.load_payload(var_11)
    var_13 = module_2._CompactJSON()
    var_14 = b'{"custom":"data"}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15, serializer=var_13)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'a'
    var_2 = 1000
    var_3 = var_1 * var_2
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'hello'
    var_7 = var_0.dump_payload(var_6)
    var_8 = 'TestSerializer'
    var_9 = {}
    var_10 = 'test'
    var_11 = 100
    var_12 = var_10 * var_11
    var_13 = ''
    var_14 = var_0.dump_payload(var_13)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = 1
    var_13 = var_11[var_12:]
    var_14 = module_1.base64_decode(var_13)
    var_15 = len(var_14)
    var_16 = str(var_10)
    var_17 = {}
    var_18 = var_0.dump_payload(var_17)
    var_19 = len(var_18)
    var_20 = 'number'
    var_21 = 42
    var_22 = {var_20: var_21}
    var_23 = var_0.dump_payload(var_22)
    var_24 = b''
    var_25 = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = 1
    var_13 = var_11[var_12:]
    var_14 = {}
    var_15 = var_0.dump_payload(var_14)
    var_16 = 42
    var_17 = var_0.dump_payload(var_16)
    var_18 = 2
    var_19 = 3
    var_20 = [var_12, var_18, var_19]
    var_21 = var_0.dump_payload(var_20)
    var_22 = 'a'
    var_23 = 10000
    var_24 = var_22 * var_23
    var_25 = {var_6: var_24}
    var_26 = var_0.dump_payload(var_25)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'{"key":"value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = b'.'
    var_5 = b'invalid-base64!!!'
    var_6 = b'not-actually-compressed'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b''
    var_10 = module_1.base64_encode(var_9)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = {}
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = 'test'
    var_14 = [var_10, var_11, var_12, var_13]
    var_15 = None
    var_16 = {var_13: var_4}
    var_17 = 'ascii'



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = 1
    var_7 = var_5[var_6:]
    var_8 = var_0.load_payload(var_7)
    var_9 = 'data'
    var_10 = 'x'
    var_11 = 1000
    var_12 = var_10 * var_11
    var_13 = {var_9: var_12}
    var_14 = b'.'
    var_15 = var_0.dump_payload(var_13)
    var_16 = b'!!!invalid_base64!!!'
    var_17 = var_0.load_payload(var_16)
    var_18 = b'not_compressed_data'
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_14 + var_19
    var_21 = var_0.load_payload(var_20)



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key": "value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid-base64!!!'
    var_5 = b'not-valid-zlib-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''
    var_9 = b'.'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'a'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_1: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = b'.'
    var_12 = var_0.load_payload(var_10)
    var_13 = b'invalid_base64!!'
    var_14 = var_0.load_payload(var_13)
    var_15 = b'.invalid_base64'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'not_compressed_data'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_11 + var_18
    var_20 = var_0.load_payload(var_19)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"compressed":true}'
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'{"special":"!@#$%^&*()"}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'e30'
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = 1
    var_12 = {}
    var_13 = 'a'
    var_14 = 'c'
    var_15 = 'b'
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_13: var_15, var_14: var_19}
    var_21 = 'ascii'
    var_22 = 1
    var_23 = 'test'
    var_24 = 'number'
    var_25 = 42
    var_26 = {var_23: var_6, var_24: var_25}
    var_27 = 'large'
    var_28 = 'nested'
    var_29 = 500
    var_30 = var_7 * var_29
    var_31 = 100
    var_32 = var_2 * var_31
    var_33 = {var_22: var_32}
    var_34 = {var_27: var_30, var_28: var_33}
    var_35 = 'small'
    var_36 = {var_35: var_6}



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = module_1.base64_decode(var_4)
    var_7 = 'data'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = var_0.dump_payload(var_11)
    var_13 = 1
    var_14 = var_12[var_13:]
    var_15 = module_1.base64_decode(var_14)
    var_16 = {}
    var_17 = var_0.dump_payload(var_16)
    var_18 = 'a'
    var_19 = 'b'
    var_20 = 2
    var_21 = 3
    var_22 = [var_13, var_20, var_21]
    var_23 = 'c'
    var_24 = 'test'
    var_25 = {var_23: var_24}
    var_26 = {var_18: var_22, var_19: var_25}
    var_27 = var_0.dump_payload(var_26)



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.load_payload(var_11)
    var_13 = var_0.load_payload(var_4)
    var_14 = {}
    var_15 = var_0.dump_payload(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = 42
    var_18 = var_0.dump_payload(var_17)
    var_19 = var_0.load_payload(var_18)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = 'test'
    var_24 = [var_20, var_21, var_22, var_23]
    var_25 = var_0.dump_payload(var_24)
    var_26 = var_0.load_payload(var_25)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'a'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_1: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = var_0.load_payload(var_10)
    var_12 = {}
    var_13 = var_0.dump_payload(var_12)
    var_14 = 10
    var_15 = var_6 * var_14
    var_16 = {var_1: var_15}
    var_17 = var_0.dump_payload(var_16)



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'{"key":"value"}'
    var_4 = module_0.base64_encode(var_3)
    var_5 = var_2 + var_4
    var_6 = b'invalid!@#$'
    var_7 = b'not compressed data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_2 + var_8
    var_10 = b'{}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = 'c'
    var_19 = 'd'
    var_20 = {var_18: var_19}
    var_21 = {var_12: var_17, var_13: var_20}



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = b'{"a": 1}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'{"b": 2}'
    var_4 = b'.'
    var_5 = b'{"c": 3}'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_4 + var_6
    var_8 = b'invalid_base64!!!'
    var_9 = b'not_compressed_data'
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_4 + var_10
    var_12 = b''
    var_13 = b'.'
    var_14 = 'data'
    var_15 = 'x'
    var_16 = 1000
    var_17 = var_15 * var_16
    var_18 = {var_14: var_17}
    var_19 = module_1._CompactJSON()
    var_20 = b'{"nested": {"key": "value"}}'
    var_21 = module_0.base64_encode(var_20)
    var_22 = b'!!!invalid_base64!!!'



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'Test that dump_payload properly compresses, base64 encodes, and adds \n    compression marker when compression reduces size.'
    var_1 = module_0.URLSafeSerializer()
    var_2 = 'a'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = b'.'
    var_6 = "Compressed payload should start with '.'"
    var_7 = 1
    var_8 = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.'
    var_9 = 'Should be valid base64 URL-safe characters'
    var_10 = 'hello'



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'a'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.load_payload(var_11)
    var_13 = var_0.load_payload(var_4)
    var_14 = {}
    var_15 = var_0.dump_payload(var_14)
    var_16 = 'b'
    var_17 = {var_7: var_16}
    var_18 = var_0.dump_payload(var_17)
    var_19 = 'ascii'
    var_20 = '^[A-Za-z0-9_\\-\\.]+$'



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'a'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = b'.'
    var_8 = 'short'
    var_9 = b'invalid-base64!!!'
    var_10 = b'not-actually-compressed'
    var_11 = module_0.base64_encode(var_10)
    var_12 = var_7 + var_11
    var_13 = ''
    var_14 = 'list'
    var_15 = 'nested'
    var_16 = 'bool'
    var_17 = 'null'
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = [var_18, var_19, var_20]
    var_22 = {var_4: var_18}
    var_23 = True
    var_24 = None
    var_25 = {var_14: var_21, var_15: var_22, var_16: var_23, var_17: var_24}



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = module_1.base64_decode(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = b'.'
    var_13 = 1
    var_14 = var_11[var_13:]
    var_15 = module_1.base64_decode(var_14)
    var_16 = 'small'
    var_17 = {var_16: var_6}
    var_18 = var_0.dump_payload(var_17)
    var_19 = var_0.load_payload(var_4)
    var_20 = var_0.load_payload(var_11)



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'{"count": 42}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"items": [1, 2, 3]}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = b'{"nested": {"a": 1}}'
    var_14 = module_0.base64_encode(var_13)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = var_0.load_payload(var_4)
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = {}
    var_12 = var_0.dump_payload(var_11)
    var_13 = 'x'
    var_14 = 1000
    var_15 = var_13 * var_14
    var_16 = {var_1: var_15}
    var_17 = var_0.dump_payload(var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = 'test'
    var_22 = [var_18, var_19, var_20, var_21]
    var_23 = var_0.dump_payload(var_22)
    var_24 = var_0.load_payload(var_23)
    var_25 = None
    var_26 = var_0.dump_payload(var_25)
    var_27 = var_0.load_payload(var_26)
    assert var_27 is None
    var_28 = 42
    var_29 = var_0.dump_payload(var_28)
    var_30 = var_0.load_payload(var_29)
    assert var_30 == 42
    var_31 = 'test_string'
    var_32 = var_0.dump_payload(var_31)
    var_33 = var_0.load_payload(var_32)
    assert var_33 == 'test_string'



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_2: var_7}
    var_9 = {}
    var_10 = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.'



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'data'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = b'invalid-base64!!!'
    var_10 = b'.'
    var_11 = b'invalid-compressed-data'
    var_12 = module_0.base64_encode(var_11)
    var_13 = var_10 + var_12
    var_14 = b''
    var_15 = b'.'



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'large_key'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.dump_payload(var_3)
    var_13 = module_1.base64_encode(var_12)
    var_14 = 0
    var_15 = 1
    var_16 = var_11[var_14:var_15]
    var_17 = var_16 == var_5
    var_18 = var_11[var_15:]
    var_19 = var_18 if var_17 else var_11
    var_20 = module_1.base64_decode(var_19)
    var_21 = 'test'
    var_22 = 'number'
    var_23 = 'data'
    var_24 = 42
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = var_0.dump_payload(var_25)
    var_27 = var_0.load_payload(var_26)
    var_28 = {}
    var_29 = var_0.dump_payload(var_28)
    var_30 = var_0.load_payload(var_29)
    var_31 = 'two'
    var_32 = 3.0
    var_33 = [var_15, var_31, var_32]
    var_34 = var_0.dump_payload(var_33)
    var_35 = var_0.load_payload(var_34)
    var_36 = 'a'
    var_37 = 'b'
    var_38 = {var_36: var_37}
    var_39 = var_0.dump_payload(var_38)
    var_40 = 500
    var_41 = var_36 * var_40
    var_42 = {var_23: var_41}
    var_43 = var_0.dump_payload(var_42)
    var_44 = var_0.load_payload(var_43)



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = ''
    var_7 = range(var_2)
    var_8 = b'_-.'
    var_9 = 'test data for roundtrip'
    var_10 = var_0.dump_payload(var_9)
    var_11 = var_0.load_payload(var_10)
    var_12 = 'x'
    var_13 = 1000
    var_14 = var_12 * var_13
    var_15 = var_0.dump_payload(var_14)
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_2: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = 'key'
    var_12 = 'number'
    var_13 = 'value'
    var_14 = 42
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = var_0.dump_payload(var_15)
    var_17 = var_0.load_payload(var_16)
    var_18 = 'y'
    var_19 = 500
    var_20 = var_18 * var_19
    var_21 = {var_2: var_20}
    var_22 = var_0.dump_payload(var_21)
    var_23 = var_0.load_payload(var_22)
    var_24 = 'simple'
    var_25 = {var_24: var_1}
    var_26 = var_0.dump_payload(var_25)



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'{}'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_0.load_payload(var_6)
    var_8 = b'{"special":"!@#$%^&*()"}'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_0.load_payload(var_9)
    var_11 = b'invalid_base64!!!'
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'a'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_2: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = var_0.load_payload(var_10)
    var_12 = var_0.load_payload(var_4)
    var_13 = 'x'
    var_14 = 'y'
    var_15 = {var_13: var_14}
    var_16 = var_0.dump_payload(var_15)
    var_17 = module_1.base64_decode(var_16)



# Parsed testcases at query #38
#--------------------------




# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = 'data'
    var_4 = 'x'
    var_5 = 100
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'{"data":"'
    var_9 = b'x'
    var_10 = var_9 * var_5
    var_11 = var_8 + var_10
    var_12 = b'"}'
    var_13 = var_11 + var_12
    var_14 = b'.'
    var_15 = b'invalid-base64!!!'
    var_16 = b'not-really-compressed'
    var_17 = module_0.base64_encode(var_16)
    var_18 = var_14 + var_17
    var_19 = b'{}'
    var_20 = module_0.base64_encode(var_19)
    var_21 = 'test'
    var_22 = 'hello_world-foo.bar'
    var_23 = {var_21: var_22}
    var_24 = b'{"test":"hello_world-foo.bar"}'
    var_25 = module_0.base64_encode(var_24)
    var_26 = 'nested'
    var_27 = 'list'
    var_28 = 'bool'
    var_29 = 1
    var_30 = 2
    var_31 = 3
    var_32 = [var_29, var_30, var_31]
    var_33 = True
    var_34 = {var_27: var_32, var_28: var_33}
    var_35 = {var_26: var_34}
    var_36 = b'{"nested":{"list":[1,2,3],"bool":true}}'
    var_37 = module_0.base64_encode(var_36)



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = 'test'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = 'utf-8'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_3: var_8}
    var_10 = b'.'
    var_11 = 'small'
    var_12 = {var_3: var_11}



# Parsed testcases at query #41
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.load_payload(var_11)
    var_13 = {}
    var_14 = var_0.dump_payload(var_13)
    var_15 = 'a'
    var_16 = 100
    var_17 = var_15 * var_16
    var_18 = {var_6: var_17}
    var_19 = var_0.dump_payload(var_18)
    var_20 = 'test'
    var_21 = True
    var_22 = {var_20: var_21}
    var_23 = var_0.dump_payload(var_22)
    var_24 = var_23[var_21:]
    var_25 = b'-_'
    var_26 = 10000
    var_27 = var_7 * var_26
    var_28 = {var_6: var_27}
    var_29 = var_0.dump_payload(var_28)
    var_30 = {var_6: var_7}
    var_31 = var_0.dump_payload(var_30)
    var_32 = len(var_29)
    var_33 = len(var_31)
    var_34 = 0.5
    var_35 = var_33 * var_34
    var_36 = 'nested'
    var_37 = 123
    var_38 = 2
    var_39 = 3
    var_40 = [var_21, var_38, var_39]
    var_41 = {var_15: var_40}
    var_42 = {var_20: var_37, var_36: var_41}
    var_43 = var_0.dump_payload(var_42)
    var_44 = var_0.load_payload(var_43)



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = var_0.dump_payload(var_8)
    var_10 = 'small'
    var_11 = var_0.dump_payload(var_10)
    var_12 = len(var_9)
    var_13 = len(var_11)
    var_14 = 'test'
    var_15 = 'nested'
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_21}
    var_23 = {var_14: var_19, var_15: var_22}
    var_24 = var_0.dump_payload(var_23)
    var_25 = var_0.load_payload(var_24)
    var_26 = {}
    var_27 = var_0.dump_payload(var_26)



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = 'test'
    var_13 = 123
    var_14 = {var_12: var_13}
    var_15 = var_0.dump_payload(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = 'y'
    var_18 = 500
    var_19 = var_17 * var_18
    var_20 = {var_6: var_19}
    var_21 = var_0.dump_payload(var_20)
    var_22 = var_0.load_payload(var_21)
    var_23 = {}
    var_24 = var_0.dump_payload(var_23)
    var_25 = 'outer'
    var_26 = 'inner'
    var_27 = 'flag'
    var_28 = 1
    var_29 = 2
    var_30 = 3
    var_31 = [var_28, var_29, var_30]
    var_32 = True
    var_33 = {var_26: var_31, var_27: var_32}
    var_34 = {var_25: var_33}
    var_35 = var_0.dump_payload(var_34)
    var_36 = var_0.load_payload(var_35)
    var_37 = 'a'
    var_38 = {var_37: var_32}
    var_39 = var_0.dump_payload(var_38)
    var_40 = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-'



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = var_0.dump_payload
    var_2 = 'a'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = 'test_data'
    var_6 = b'.'
    var_7 = 1
    var_8 = 'ab'
    var_9 = b'test_payload'
    var_10 = b'Hello World! Hello World! Hello World! Hello World! Hello World! Hello World! Hello World! Hello World! Hello World!'
    var_11 = 1
    var_12 = module_1.base64_decode(var_3)



# Parsed testcases at query #45
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 100
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = b'invalid_base64!!!'
    var_11 = var_0.load_payload(var_10)
    var_12 = b'not_compressed_data'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_9 + var_13
    var_15 = var_0.load_payload(var_14)
    var_16 = b''
    var_17 = var_0.load_payload(var_16)
    var_18 = b'.'
    var_19 = var_0.load_payload(var_18)



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'test_payload'
    var_2 = 'small_data'
    var_3 = var_0.dump_payload(var_2)
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = var_0.dump_payload(var_6)
    var_8 = b'.'
    var_9 = ''
    var_10 = var_0.dump_payload(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = var_0.dump_payload(var_13)



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = 'data'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = {}
    var_11 = 'string'
    var_12 = 'number'
    var_13 = 'list'
    var_14 = 'nested'
    var_15 = 'hello'
    var_16 = 42
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = 'key'
    var_22 = {var_21: var_2}
    var_23 = {var_11: var_15, var_12: var_16, var_13: var_20, var_14: var_22}



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'{"complex":"data"}'
    var_4 = b'.'
    var_5 = b'invalid-base64!!!'
    var_6 = b'not-compressed-data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b''
    var_10 = b'.'
    var_11 = b'{"dotty":"value"}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = 'long'
    var_14 = 'data'
    var_15 = 100
    var_16 = var_14 * var_15
    var_17 = {var_13: var_16}
    var_18 = b'{"long":"datadatadatadatadatadata..."}'



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'{"key":"value"}'
    var_5 = module_1.base64_encode(var_4)
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = 'a'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = 'data'
    var_12 = {var_11: var_10}
    var_13 = b'{"data":"'
    var_14 = b'"}'
    var_15 = b'invalid-base64!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'not-compressed-data'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_7 + var_18
    var_20 = var_0.load_payload(var_19)
    var_21 = b''
    var_22 = var_0.load_payload(var_21)
    var_23 = b'.'
    var_24 = var_0.load_payload(var_23)



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'ey'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = b'.'
    var_13 = var_0.load_payload(var_11)
    var_14 = {}
    var_15 = var_0.dump_payload(var_14)
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = var_0.dump_payload(var_19)
    var_21 = var_0.load_payload(var_20)
    var_22 = 'test string'
    var_23 = var_0.dump_payload(var_22)
    var_24 = var_0.load_payload(var_23)
    var_25 = None
    var_26 = var_0.dump_payload(var_25)
    var_27 = var_0.load_payload(var_26)
    assert var_27 is None



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'data'
    var_5 = 'a'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = 1
    var_10 = module_0.base64_decode(var_1)
    var_11 = 'test'
    var_12 = 'nested'
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 'b'
    var_18 = {var_5: var_17}
    var_19 = {var_11: var_16, var_12: var_18}
    var_20 = {}
    var_21 = 'message'
    var_22 = 'hello world'
    var_23 = {var_21: var_22}
    var_24 = 'ascii'



# Parsed testcases at query #52
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = {}
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = 'test'
    var_14 = [var_10, var_11, var_12, var_13]
    var_15 = 'level1'
    var_16 = 'level2'
    var_17 = 'level3'
    var_18 = 'deep'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = {var_15: var_20}
    var_22 = 1
    var_23 = module_0.base64_decode(var_1)



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'ey'
    var_4 = b'.'
    var_5 = 'data'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = 1
    var_11 = {}
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_13}
    var_15 = {var_12: var_10}



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 'Test that dump_payload correctly serializes, optionally compresses, and base64 encodes.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = 'data'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = 'test'
    var_11 = 'number'
    var_12 = 42
    var_13 = {var_10: var_5, var_11: var_12}
    var_14 = {}
    var_15 = 'a'
    var_16 = 'b'
    var_17 = {var_15: var_16}
    var_18 = 'special'
    var_19 = 'data with spaces and symbols!@#$%^&*()'
    var_20 = {var_18: var_19}
    var_21 = 32
    var_22 = 126
    var_23 = 95
    var_24 = 45
    var_25 = 46
    var_26 = (var_23, var_24, var_25)
    var_27 = 'y'
    var_28 = 50
    var_29 = var_27 * var_28
    var_30 = {var_6: var_29}



# Parsed testcases at query #55
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = 1
    var_6 = var_4[var_5:]
    var_7 = var_0.load_payload(var_6)
    var_8 = 'data'
    var_9 = 'x'
    var_10 = 1000
    var_11 = var_9 * var_10
    var_12 = {var_8: var_11}
    var_13 = var_0.dump_payload(var_12)
    var_14 = b'.'
    var_15 = var_0.load_payload(var_13)
    var_16 = 'small'
    var_17 = {var_16: var_8}
    var_18 = var_0.dump_payload(var_17)
    var_19 = 1
    var_20 = var_18[var_19:]
    var_21 = var_0.load_payload(var_20)
    var_22 = b'invalid_base64!!!'
    var_23 = var_0.load_payload(var_22)
    var_24 = b'not_compressed_data'
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_14 + var_25
    var_27 = var_0.load_payload(var_26)
    var_28 = b''
    var_29 = var_0.load_payload(var_28)
    var_30 = b'.'
    var_31 = var_0.load_payload(var_30)



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'test_secret_key'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = 'data'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = b'.'
    var_11 = 'test'
    var_12 = 'number'
    var_13 = 42
    var_14 = {var_11: var_5, var_12: var_13}
    var_15 = 1
    var_16 = b'=='
    var_17 = 'ascii'
    var_18 = '^[A-Za-z0-9._-]+$'



# Parsed testcases at query #57
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = 1
    var_12 = 'ab'
    var_13 = {var_6: var_12}
    var_14 = {}
    var_15 = 2
    var_16 = 3
    var_17 = [var_11, var_15, var_16]
    var_18 = 'test'
    var_19 = 'number'
    var_20 = 'roundtrip'
    var_21 = 42
    var_22 = {var_18: var_20, var_19: var_21}



# Parsed testcases at query #58
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_2: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = module_0.URLSafeSerializerMixin()
    var_12 = 'hello'
    var_13 = 'number'
    var_14 = 'world'
    var_15 = 42
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = var_11.dump_payload(var_16)
    var_18 = var_11.load_payload(var_17)
    var_19 = 'a'
    var_20 = 500
    var_21 = var_19 * var_20
    var_22 = {var_2: var_21}
    var_23 = var_11.dump_payload(var_22)
    var_24 = var_11.load_payload(var_23)



# Parsed testcases at query #59
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0.URLSafeSerializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = b'.'
    var_7 = b'ey'
    var_8 = 'data'
    var_9 = 'x'
    var_10 = 1000
    var_11 = var_9 * var_10
    var_12 = {var_8: var_11}
    var_13 = var_9 * var_10
    var_14 = {var_8: var_13}
    var_15 = 10
    var_16 = var_9 * var_15
    var_17 = {var_8: var_16}
    var_18 = 'y'
    var_19 = 50
    var_20 = var_18 * var_19
    var_21 = {var_8: var_20}
    var_22 = 1
    var_23 = b'=='
    var_24 = {}
    var_25 = 'test-key'
    var_26 = None
    var_27 = lambda : var_26
    var_28 = module_0.URLSafeSerializer(var_25, var_1, var_27)



# Parsed testcases at query #60
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = module_1.base64_decode(var_4)
    var_7 = var_0.dump_payload(var_3)
    var_8 = 'data'
    var_9 = 'x'
    var_10 = 1000
    var_11 = var_9 * var_10
    var_12 = {var_8: var_11}
    var_13 = var_0.dump_payload(var_12)
    var_14 = 'abc123'
    var_15 = {var_8: var_14}
    var_16 = var_0.dump_payload(var_15)



# Parsed testcases at query #61
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJhIjogMX0='
    var_2 = var_0.load_payload(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = 100
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = b'invalid!!'
    var_10 = var_0.load_payload(var_9)
    var_11 = b'.'
    var_12 = b'corrupted_data'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_11 + var_13
    var_15 = var_0.load_payload(var_14)
    var_16 = b'{}'
    var_17 = module_1.base64_encode(var_16)
    var_18 = var_0.load_payload(var_17)
    var_19 = 'test'
    var_20 = 'hello_world-test.test'
    var_21 = {var_19: var_20}



# Parsed testcases at query #62
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = 'hello'
    var_3 = 'world'
    var_4 = {var_2: var_3}
    var_5 = 'data'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = b'.'
    var_11 = {}
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = 'test'
    var_16 = [var_12, var_13, var_14, var_15]
    var_17 = 'ascii'
    var_18 = '_-.'



# Parsed testcases at query #63
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'a'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = len(var_11)
    var_13 = 'test'
    var_14 = {var_13: var_6}
    var_15 = var_0.dump_payload(var_14)
    var_16 = 1
    var_17 = var_15[var_16:]
    var_18 = var_15
    var_19 = {}
    var_20 = var_0.dump_payload(var_19)
    var_21 = 'string'
    var_22 = 'number'
    var_23 = 'list'
    var_24 = 'nested'
    var_25 = 'hello'
    var_26 = 42
    var_27 = 1
    var_28 = 2
    var_29 = 3
    var_30 = [var_27, var_28, var_29]
    var_31 = 'inner'
    var_32 = {var_31: var_2}
    var_33 = {var_21: var_25, var_22: var_26, var_23: var_30, var_24: var_32}
    var_34 = var_0.dump_payload(var_33)
    var_35 = 'message'
    var_36 = 'test data'
    var_37 = {var_35: var_36}
    var_38 = var_0.dump_payload(var_37)
    var_39 = var_0.load_payload(var_38)
    var_40 = 'repeated'
    var_41 = 'x'
    var_42 = 500
    var_43 = var_41 * var_42
    var_44 = {var_40: var_43}
    var_45 = var_0.dump_payload(var_44)
    var_46 = var_0.load_payload(var_45)
    var_47 = 50
    var_48 = var_7 * var_47
    var_49 = {var_6: var_48}
    var_50 = var_0.dump_payload(var_49)



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'a'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = 1
    var_8 = ''



# Parsed testcases at query #65
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_2: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = var_0.dump_payload(var_9)
    var_12 = len(var_11)
    var_13 = 1
    var_14 = 'a'
    var_15 = {var_14: var_13}
    var_16 = var_0.dump_payload(var_15)
    var_17 = {}
    var_18 = var_0.dump_payload(var_17)



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = {}
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = 'test'



# Parsed testcases at query #67
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = 'test'
    var_13 = {var_12: var_6}
    var_14 = var_0.dump_payload(var_13)
    var_15 = 'message'
    var_16 = 'number'
    var_17 = 'hello world'
    var_18 = 42
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = var_0.dump_payload(var_19)
    var_21 = var_0.load_payload(var_20)



# Parsed testcases at query #68
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'{"key":"value with spaces & symbols!"}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"outer":{"inner":"value"}}'
    var_12 = module_0.base64_encode(var_11)



# Parsed testcases at query #69
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_1: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = var_0.dump_payload(var_3)
    var_12 = len(var_10)
    var_13 = len(var_11)
    var_14 = var_12 < var_13
    var_15 = len(var_10)
    var_16 = len(var_11)
    var_17 = 1
    var_18 = var_16 + var_17
    var_19 = var_15 == var_18
    var_20 = 'test'
    var_21 = 'number'
    var_22 = 'data'
    var_23 = 42
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = var_0.dump_payload(var_24)
    var_26 = var_0.load_payload(var_25)
    var_27 = {}
    var_28 = var_0.dump_payload(var_27)
    var_29 = 'outer'
    var_30 = 'inner'
    var_31 = 2
    var_32 = 3
    var_33 = [var_17, var_31, var_32]
    var_34 = {var_30: var_33}
    var_35 = {var_29: var_34}
    var_36 = var_0.dump_payload(var_35)
    var_37 = var_0.load_payload(var_36)
    var_38 = var_0.dump_payload(var_3)
    var_39 = 'ascii'
    var_40 = '_-.'



# Parsed testcases at query #70
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed":"data"}'
    var_3 = b'.'
    var_4 = b'invalid_base64!!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'.'
    var_11 = module_1._CompactJSON()
    var_12 = b'{"custom":true}'
    var_13 = module_0.base64_encode(var_12)



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = 'small'
    var_10 = {var_9: var_4}
    var_11 = {}



# Parsed testcases at query #72
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'a'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_1: var_6}
    var_8 = 'hello'
    var_9 = 'number'
    var_10 = 'world'
    var_11 = 42
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'list'
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = {var_13: var_17}
    var_19 = 'nested'
    var_20 = 'key'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = {var_19: var_22}
    var_24 = 'boolean'
    var_25 = 'null'
    var_26 = True
    var_27 = None
    var_28 = {var_24: var_26, var_25: var_27}
    var_29 = 'unicode'
    var_30 = 'héllo wörld'
    var_31 = {var_29: var_30}
    var_32 = [var_18, var_23, var_28, var_31]
    var_33 = b'^[A-Za-z0-9_.-]+$'



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = 'Large payload should be compressed'
    var_11 = b'!invalid_base64!'
    var_12 = b'not_compressed_data'



# Parsed testcases at query #74
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.load_payload(var_11)
    var_13 = {}
    var_14 = var_0.dump_payload(var_13)
    var_15 = None
    var_16 = {var_2: var_15}
    var_17 = var_0.dump_payload(var_16)
    var_18 = 'nested'
    var_19 = 'list'
    var_20 = 'bool'
    var_21 = 1
    var_22 = 2
    var_23 = 3
    var_24 = [var_21, var_22, var_23]
    var_25 = True
    var_26 = {var_19: var_24, var_20: var_25}
    var_27 = {var_18: var_26}
    var_28 = var_0.dump_payload(var_27)
    var_29 = var_0.load_payload(var_28)



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = 'Large compressible payload should be compressed'
    var_10 = 'test'
    var_11 = 'nested'
    var_12 = 123
    var_13 = 'a'
    var_14 = 1
    var_15 = {var_13: var_14}
    var_16 = {var_10: var_12, var_11: var_15}
    var_17 = 'numbers'
    var_18 = 'y'
    var_19 = 500
    var_20 = var_18 * var_19
    var_21 = 100
    var_22 = range(var_21)
    var_23 = list(var_22)
    var_24 = {var_4: var_20, var_17: var_23}
    var_25 = {}
    var_26 = 'two'
    var_27 = 3.0
    var_28 = [var_14, var_26, var_27]
    var_29 = None



# Parsed testcases at query #76
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = 1
    var_6 = var_4[var_5:]
    var_7 = var_0.load_payload(var_6)
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_5: var_10}
    var_12 = var_0.dump_payload(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'.'
    var_15 = b'{"test": "data"}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_14 + var_16
    var_18 = var_0.load_payload(var_17)
    var_19 = b'invalid_base64!!!'
    var_20 = var_0.load_payload(var_19)
    var_21 = b'not_compressed_data'
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_14 + var_22
    var_24 = var_0.load_payload(var_23)
    var_25 = b''
    var_26 = var_0.load_payload(var_25)
    var_27 = module_2._CompactJSON()
    var_28 = b'{"custom": "data"}'
    var_29 = module_1.base64_encode(var_28)
    var_30 = var_0.load_payload(var_29, serializer=var_27)



# Parsed testcases at query #77
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = 'compressed'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = b'not_compressed_data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_5 + var_8
    var_10 = b'{}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = 'nested'
    var_13 = 'list'
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_13: var_16}
    var_18 = {var_12: var_17}



# Parsed testcases at query #78
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous._json as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1._CompactJSON()
    var_7 = module_1._CompactJSON()
    var_8 = b'.'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = 'test'
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = module_1._CompactJSON()
    var_15 = 'nested'
    var_16 = 'inner'
    var_17 = 'a'
    var_18 = 'b'
    var_19 = [var_17, var_18]
    var_20 = {var_16: var_19}
    var_21 = 123
    var_22 = {var_15: var_20, var_3: var_21}
    var_23 = module_1._CompactJSON()
    var_24 = b'invalid_base64!!!'
    var_25 = var_0.load_payload(var_24)
    var_26 = b'not_compressed_data'
    var_27 = module_2.base64_encode(var_26)
    var_28 = var_8 + var_27
    var_29 = var_0.load_payload(var_28)
    var_30 = {}
    var_31 = module_1._CompactJSON()
    var_32 = 'special'
    var_33 = '!@#$%^&*()_+-=[]{}|;\':",./<>?`~'
    var_34 = {var_32: var_33}
    var_35 = module_1._CompactJSON()



# Parsed testcases at query #79
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid-base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not-compressed-data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = 'nested'
    var_15 = 'list'
    var_16 = 'bool'
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = True
    var_22 = {var_15: var_20, var_16: var_21}
    var_23 = {var_14: var_22}
    var_24 = b'{"nested":{"list":[1,2,3],"bool":true}}'
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_0.load_payload(var_25)



# Parsed testcases at query #80
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = 'long'
    var_10 = {var_9: var_8}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.load_payload(var_11)
    var_13 = 'short'
    var_14 = 'abc'
    var_15 = {var_13: var_14}
    var_16 = var_0.dump_payload(var_15)
    var_17 = var_0.load_payload(var_16)
    var_18 = {}
    var_19 = var_0.dump_payload(var_18)
    var_20 = var_0.load_payload(var_19)
    var_21 = 'number'
    var_22 = 'float'
    var_23 = 'list'
    var_24 = 'nested'
    var_25 = 42
    var_26 = 3.14
    var_27 = 1
    var_28 = 2
    var_29 = 3
    var_30 = [var_27, var_28, var_29]
    var_31 = 'inner'
    var_32 = {var_31: var_2}
    var_33 = {var_21: var_25, var_22: var_26, var_23: var_30, var_24: var_32}
    var_34 = var_0.dump_payload(var_33)
    var_35 = var_0.load_payload(var_34)
    var_36 = b'invalid_base64!!'
    var_37 = var_0.load_payload(var_36)
    var_38 = b'{"test": "data"}'
    var_39 = b'.'
    var_40 = 10
    var_41 = b'corrupted'
    var_42 = 'compress'
    var_43 = 'me'
    var_44 = 500
    var_45 = var_43 * var_44
    var_46 = {var_42: var_45}
    var_47 = var_0.dump_payload(var_46)
    var_48 = var_0.load_payload(var_47)
    var_49 = 'data'
    var_50 = {var_13: var_49}
    var_51 = var_0.dump_payload(var_50)
    var_52 = var_0.load_payload(var_51)
    var_53 = {var_2: var_14}
    var_54 = var_0.dump_payload(var_53)
    var_55 = var_0.load_payload(var_54)



# Parsed testcases at query #81
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.load_payload(var_11)
    var_13 = b'.'
    var_14 = b'{"test": true}'
    var_15 = b'invalid_base64!!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'.'
    var_18 = b'not_compressed_data'
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_17 + var_19
    var_21 = var_0.load_payload(var_20)
    var_22 = b''
    var_23 = var_0.load_payload(var_22)



# Parsed testcases at query #82
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = 'Large payload should be compressed'
    var_11 = 'small'
    var_12 = {var_4: var_11}
    var_13 = b'invalid-base64!!!'
    var_14 = b'not-compressed-data'
    var_15 = b''
    var_16 = b'.'
    var_17 = 'a'
    var_18 = 1
    var_19 = {var_17: var_18}



# Parsed testcases at query #83
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'{"a":1}'
    var_4 = module_0.base64_encode(var_3)
    var_5 = b'invalid_base64!!'
    var_6 = b'corrupted_data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_2 + var_7
    var_9 = b'{}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"nested":{"key":"value"}}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = b'[1,2,3]'
    var_14 = module_0.base64_encode(var_13)
    var_15 = b'{"test":"data"}'



# Parsed testcases at query #84
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'utf-8'
    var_5 = b'.'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = b'!!!invalid_base64!!!'
    var_10 = var_0.load_payload(var_9)
    var_11 = b'invalid_compressed_data'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_5 + var_12
    var_14 = var_0.load_payload(var_13)
    var_15 = {}
    var_16 = 'msg'
    var_17 = 'hello world! @#$%'
    var_18 = {var_16: var_17}



# Parsed testcases at query #85
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = b'.'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_4 + var_6



# Parsed testcases at query #86
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'{"compressed": true}'
    var_6 = b'invalid-base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not-compressed-data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'[1, 2, 3]'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)
    var_18 = b'{"outer": {"inner": "test"}}'



# Parsed testcases at query #87
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'[1,2,3]'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)
    var_18 = b'{"nested":{"a":1}}'
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_0.load_payload(var_19)



# Parsed testcases at query #88
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"compressed": true}'
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = var_0.load_payload(var_12)



# Parsed testcases at query #89
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'null'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    assert var_13 is None



# Parsed testcases at query #90
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'{"key":"value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = b'.'
    var_5 = b'x'
    var_6 = 100
    var_7 = var_5 * var_6
    var_8 = b'invalid-base64!!!'
    var_9 = b'not-compressed-data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_4 + var_10
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = b'{"nested":{"key":"value"}}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = b'{"string":"hello","number":42,"boolean":true,"array":[1,2,3]}'
    var_17 = module_1.base64_encode(var_16)



# Parsed testcases at query #91
#--------------------------




# Parsed testcases at query #92
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'data'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = b'invalid_base64!!!'
    var_10 = b'not_compressed_data'
    var_11 = module_0.base64_encode(var_10)
    var_12 = var_8 + var_11
    var_13 = b''
    var_14 = 42
    var_15 = 'string'
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = 'nested'
    var_21 = True
    var_22 = {var_3: var_21}
    var_23 = {var_20: var_22}
    var_24 = None
    var_25 = 3.14
    var_26 = [var_14, var_15, var_19, var_23, var_24, var_25]



# Parsed testcases at query #93
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'x'
    var_4 = 1000
    var_5 = var_3 * var_4
    var_6 = 'short'
    var_7 = b'.'
    var_8 = b'invalid_base64!!!'
    var_9 = b'not_compressed_data'
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_7 + var_10
    var_12 = b''
    var_13 = b'.'
    var_14 = 'list'
    var_15 = 'nested'
    var_16 = 'number'
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_22}
    var_24 = 42
    var_25 = {var_14: var_20, var_15: var_23, var_16: var_24}



# Parsed testcases at query #94
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key": "value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid-base64!!!'
    var_5 = b'not-actually-compressed'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''
    var_9 = b'not valid json'



# Parsed testcases at query #95
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = module_2._CompactJSON()
    var_6 = module_1.base64_encode(var_1)
    var_7 = var_0.load_payload(var_6, serializer=var_5)
    var_8 = b'invalid_base64'
    var_9 = var_0.load_payload(var_8)
    var_10 = b'not_compressed'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_9 + var_11
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = b'{"data":"test with spaces"}'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_0.load_payload(var_18)



# Parsed testcases at query #96
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = var_0.load_payload(var_5)
    var_7 = module_0.URLSafeSerializerMixin()
    var_8 = 100
    var_9 = var_2 * var_8
    var_10 = {var_1: var_9}
    var_11 = var_7.dump_payload(var_10)
    var_12 = b'.'
    var_13 = var_7.load_payload(var_11)
    var_14 = module_0.URLSafeSerializerMixin()
    var_15 = 'short'
    var_16 = {var_1: var_15}
    var_17 = var_14.dump_payload(var_16)
    var_18 = var_14.load_payload(var_17)
    var_19 = module_0.URLSafeSerializerMixin()
    var_20 = b'invalid_base64@@@'
    var_21 = var_19.load_payload(var_20)
    var_22 = module_0.URLSafeSerializerMixin()
    var_23 = b'.invalid_base64@@@'
    var_24 = var_22.load_payload(var_23)
    var_25 = module_0.URLSafeSerializerMixin()
    var_26 = b'not_compressed_data'
    var_27 = module_1.base64_encode(var_26)
    var_28 = b'.'
    var_29 = var_28 + var_27
    var_30 = var_25.load_payload(var_29)
    var_31 = module_0.URLSafeSerializerMixin()
    var_32 = b''
    var_33 = var_31.load_payload(var_32)



# Parsed testcases at query #97
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.load_payload(var_11)
    var_13 = b'!@#$%^&*()'
    var_14 = var_0.load_payload(var_13)
    var_15 = b'.'
    var_16 = b'not_compressed_data'
    var_17 = module_1.base64_encode(var_16)
    var_18 = var_15 + var_17
    var_19 = var_0.load_payload(var_18)
    var_20 = b'{"test": "data"}'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_15 + var_21
    var_23 = var_0.load_payload(var_22)
    var_24 = b'{}'
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_0.load_payload(var_25)
    var_27 = 'string'
    var_28 = 'number'
    var_29 = 'list'
    var_30 = 'nested'
    var_31 = 'test'
    var_32 = 42
    var_33 = 1
    var_34 = 2
    var_35 = 3
    var_36 = [var_33, var_34, var_35]
    var_37 = 'a'
    var_38 = {var_37: var_33}
    var_39 = {var_27: var_31, var_28: var_32, var_29: var_36, var_30: var_38}
    var_40 = var_0.dump_payload(var_39)
    var_41 = var_0.load_payload(var_40)
    var_42 = var_0.dump_payload(var_3)
    var_43 = module_2._CompactJSON()
    var_44 = var_0.load_payload(var_42, serializer=var_43)



# Parsed testcases at query #98
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = var_0.dump_payload(var_8)
    var_10 = b'.'
    var_11 = var_0.load_payload(var_9)
    var_12 = module_1._CompactJSON()
    var_13 = var_0.load_payload(var_9, serializer=var_12)
    var_14 = b'not_base64_encoded_data'
    var_15 = var_0.load_payload(var_14)
    var_16 = b'invalid_compressed_data'
    var_17 = var_10 + var_16
    var_18 = var_0.load_payload(var_17)



# Parsed testcases at query #99
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJmb28iOiAiYmFyIn0'
    var_2 = var_0.load_payload(var_1)
    var_3 = b'.'
    var_4 = b'eJxTqo6NBQAAQwEBgQ=='
    var_5 = var_3 + var_4
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_base64!!!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'dGVzdA=='
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = var_0.load_payload(var_12)
    var_14 = b'.'
    var_15 = var_0.load_payload(var_14)
    var_16 = 'extra_arg'
    var_17 = 'value'



# Parsed testcases at query #100
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'{"custom":"data"}'
    var_4 = module_0.base64_encode(var_3)
    var_5 = module_1._CompactJSON()
    var_6 = b'invalid!!!'
    var_7 = b'invalid_compressed_data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_2 + var_8
    var_10 = b'{}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = b'{"special":"test@123!#$%"}'
    var_13 = module_0.base64_encode(var_12)
    var_14 = 'key'
    var_15 = 'x'
    var_16 = 1000
    var_17 = var_15 * var_16
    var_18 = {var_14: var_17}
    var_19 = module_1._CompactJSON()



# Parsed testcases at query #101
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{"special":"!@#$%^&*()"}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = b'{"nested":{"key":"value","list":[1,2,3]}}'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_0.load_payload(var_18)



# Parsed testcases at query #102
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'utf-8'
    var_5 = b'.'
    var_6 = module_0.URLSafeSerializerMixin()
    var_7 = 'data'
    var_8 = 'a'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = b'%%%invalid%%%'
    var_13 = var_0.load_payload(var_12)
    var_14 = b'not_compressed_data'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_5 + var_15
    var_17 = var_0.load_payload(var_16)
    var_18 = b''
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_0.load_payload(var_19)
    var_21 = "Compressed payload should start with '.'"



# Parsed testcases at query #103
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'corrupted_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'{"special":"test/with/slashes"}'
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #104
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key": "value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid-base64!!!'
    var_5 = b'corrupted-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'{"special": "test@123!"}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = 'data'
    var_13 = 'x'
    var_14 = 1000
    var_15 = var_13 * var_14
    var_16 = {var_12: var_15}
    var_17 = b'{"data": "'
    var_18 = b'x'
    var_19 = var_18 * var_14
    var_20 = var_17 + var_19
    var_21 = b'"}'
    var_22 = var_20 + var_21
    var_23 = module_0.base64_encode(var_22)
    var_24 = var_18 * var_14
    var_25 = var_17 + var_24
    var_26 = var_25 + var_21



# Parsed testcases at query #105
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = b'!!!invalid!!!'
    var_11 = b'corrupted_data'
    var_12 = module_0.base64_encode(var_11)
    var_13 = var_9 + var_12
    var_14 = b'{}'
    var_15 = module_0.base64_encode(var_14)
    var_16 = module_0.base64_encode(var_14)
    var_17 = var_9 + var_16



# Parsed testcases at query #106
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"compressed":true}'
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #107
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'{"compressed":true}'
    var_4 = b'.'
    var_5 = b'not-valid-base64!!'
    var_6 = b'not-compressed-data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b'{}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"num":42,"list":[1,2,3],"nested":{"a":"b"}}'
    var_12 = module_0.base64_encode(var_11)



# Parsed testcases at query #108
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = b'.'
    var_7 = b'invalid-base64!!!'
    var_8 = b'not-compressed-data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_6 + var_9
    var_11 = b''
    var_12 = b'.'
    var_13 = 'data'
    var_14 = 'x'
    var_15 = 1000
    var_16 = var_14 * var_15
    var_17 = {var_13: var_16}
    var_18 = 'small'
    var_19 = {var_13: var_18}
    var_20 = module_2._CompactJSON()
    var_21 = 'custom'
    var_22 = {var_21: var_13}



# Parsed testcases at query #109
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"name": "test", "data": "x" * 100}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!!!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'not_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)
    var_13 = b''
    var_14 = var_0.load_payload(var_13)
    var_15 = b''
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_14 + var_16
    var_18 = var_0.load_payload(var_17)



# Parsed testcases at query #110
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'{"number":42,"list":[1,2,3],"nested":{"a":"b"}}'
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #111
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = module_0.URLSafeSerializerMixin()
    var_10 = var_9.dump_payload(var_8)
    var_11 = b'.'
    var_12 = var_9.load_payload(var_10)
    var_13 = 'short'
    var_14 = module_0.URLSafeSerializerMixin()
    var_15 = var_14.dump_payload(var_13)
    var_16 = var_14.load_payload(var_15)
    var_17 = module_0.URLSafeSerializerMixin()
    var_18 = b'invalid-base64!!'
    var_19 = var_17.load_payload(var_18)
    var_20 = module_0.URLSafeSerializerMixin()
    var_21 = b'not-valid-base64'
    var_22 = var_11 + var_21
    var_23 = var_20.load_payload(var_22)
    var_24 = module_0.URLSafeSerializerMixin()
    var_25 = b''
    var_26 = var_24.load_payload(var_25)



# Parsed testcases at query #112
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test load_payload method with various scenarios.'
    var_1 = module_0.URLSafeSerializerMixin()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = module_1.base64_encode(var_5)
    var_7 = 1
    var_8 = var_6[var_7:]
    var_9 = var_1.load_payload(var_8)
    var_10 = 'data'
    var_11 = 'x'
    var_12 = 1000
    var_13 = var_11 * var_12
    var_14 = {var_10: var_13}
    var_15 = var_1.dump_payload(var_14)
    var_16 = b'.'
    var_17 = "Expected compressed payload to start with '.'"
    var_18 = var_1.load_payload(var_15)
    var_19 = b'!!!invalid_base64!!!'
    var_20 = var_1.load_payload(var_19)
    var_21 = b'not_compressed_data'
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_16 + var_22
    var_24 = var_1.load_payload(var_23)
    var_25 = b'{}'
    var_26 = module_1.base64_encode(var_25)
    var_27 = var_1.load_payload(var_26)
    var_28 = 'special'
    var_29 = '!@#$%^&*()_+-=[]{}|;\':",./<>?'
    var_30 = {var_28: var_29}
    var_31 = var_1.dump_payload(var_30)
    var_32 = 1
    var_33 = var_31[var_32:]
    var_34 = var_1.load_payload(var_33)
    var_35 = 'small'
    var_36 = 'test'
    var_37 = {var_35: var_36}
    var_38 = var_1.dump_payload(var_37)
    var_39 = var_1.load_payload(var_38)
    var_40 = 'a'
    var_41 = 'b'
    var_42 = {var_40: var_41}
    var_43 = var_1.dump_payload(var_42)
    var_44 = var_1.load_payload(var_43)
    var_45 = 'level1'
    var_46 = 'level2'
    var_47 = 'level3'
    var_48 = 1
    var_49 = 2
    var_50 = 3
    var_51 = [var_48, var_49, var_50]
    var_52 = {var_40: var_41}
    var_53 = {var_46: var_51, var_47: var_52}
    var_54 = {var_45: var_53}
    var_55 = var_1.dump_payload(var_54)
    var_56 = 1
    var_57 = var_55[var_56:]
    var_58 = var_1.load_payload(var_57)
    var_59 = 'c'
    var_60 = [var_48, var_49, var_50, var_40, var_41, var_59]
    var_61 = var_1.dump_payload(var_60)
    var_62 = 1
    var_63 = var_61[var_62:]
    var_64 = var_1.load_payload(var_63)



# Parsed testcases at query #113
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'not-valid-base64!!'
    var_4 = b'not-compressed-data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'null'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'[1,2,3]'
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #114
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_1: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = b'.'
    var_12 = var_0.load_payload(var_10)
    var_13 = b'{"a":1}'
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_11 + var_14
    var_16 = var_0.load_payload(var_15)
    var_17 = b'invalid_base64!!!'
    var_18 = var_0.load_payload(var_17)
    var_19 = b'.'
    var_20 = b'not_compressed_data'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_19 + var_21
    var_23 = var_0.load_payload(var_22)
    var_24 = module_0.URLSafeSerializerMixin()
    var_25 = 'test'
    var_26 = var_24.dump_payload(var_25)
    var_27 = b''
    var_28 = var_0.load_payload(var_27)
    var_29 = b'.'
    var_30 = var_0.load_payload(var_29)



# Parsed testcases at query #115
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'eyJmb28iOiAiYmFyIn0'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 100
    var_5 = var_3 * var_4
    var_6 = {var_2: var_5}
    var_7 = b'.'
    var_8 = b'invalid-base64!!!'
    var_9 = b'corrupted-data'
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_7 + var_10
    var_12 = b''



# Parsed testcases at query #116
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'.'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_4 + var_6
    var_8 = b''
    var_9 = b'.'



# Parsed testcases at query #117
#--------------------------


def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = b'.'
    var_12 = "Compressed payload should start with '.'"
    var_13 = b'not-valid-base64!!!'
    var_14 = b'.invalidbase64'
    var_15 = b''
    var_16 = b'.'



# Parsed testcases at query #118
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'corrupted_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b''
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'.'
    var_10 = 'key'
    var_11 = 'x'
    var_12 = 1000
    var_13 = var_11 * var_12
    var_14 = {var_10: var_13}



# Parsed testcases at query #119
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b''
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #120
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'{"key":"value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = b'.'
    var_5 = b'!!!invalid_base64!!!'
    var_6 = b'not_compressed_data'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b''
    var_10 = module_1.base64_encode(var_9)
    var_11 = module_1.base64_encode(var_9)
    var_12 = var_4 + var_11
    var_13 = b'{"test":123}'
    var_14 = module_1.base64_encode(var_13)
    var_15 = 'large'
    var_16 = 'data'
    var_17 = 100
    var_18 = var_16 * var_17
    var_19 = {var_15: var_18}
    var_20 = b'{"large":"data'
    var_21 = b'data'
    var_22 = 99
    var_23 = var_21 * var_22
    var_24 = var_20 + var_23
    var_25 = b'"}'
    var_26 = var_24 + var_25



# Parsed testcases at query #121
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b''
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    assert var_13 is None
    var_14 = b'.'
    var_15 = var_0.load_payload(var_14)



# Parsed testcases at query #122
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'eyJmb28iOiAiYmFyIn0='
    var_1 = b'{"hello": "world"}'
    var_2 = b'.'
    var_3 = b'invalid-base64!!!'
    var_4 = b'not-compressed-data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_3 + var_5
    var_7 = b''
    var_8 = b'.'



# Parsed testcases at query #123
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid-base64!!'
    var_5 = b'corrupted-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)



# Parsed testcases at query #124
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = module_1._CompactJSON()
    var_4 = b'{"custom":"data"}'
    var_5 = module_0.base64_encode(var_4)
    var_6 = b'invalid_base64!!!'
    var_7 = b'not_compressed_data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_2 + var_8
    var_10 = b'null'
    var_11 = module_0.base64_encode(var_10)
    var_12 = b'{"special":"test with spaces"}'
    var_13 = module_0.base64_encode(var_12)
    var_14 = b'12345'
    var_15 = module_0.base64_encode(var_14)
    var_16 = b'true'
    var_17 = module_0.base64_encode(var_16)
    var_18 = b'[1,2,3]'
    var_19 = module_0.base64_encode(var_18)



# Parsed testcases at query #125
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = b'.'
    var_13 = 'Compression should have been used for large data'
    var_14 = var_0.load_payload(var_11)
    var_15 = 'small'
    var_16 = {var_15: var_6}
    var_17 = var_0.dump_payload(var_16)
    var_18 = var_0.load_payload(var_17)
    var_19 = b'invalid_base64!!'
    var_20 = var_0.load_payload(var_19)
    var_21 = b'.'
    var_22 = b'not_compressed_data'
    var_23 = module_1.base64_encode(var_22)
    var_24 = var_21 + var_23
    var_25 = var_0.load_payload(var_24)
    var_26 = b''
    var_27 = var_0.load_payload(var_26)
    var_28 = 'special'
    var_29 = '!@#$%^&*()_+-=[]{}|;\':",./<>?'
    var_30 = {var_28: var_29}
    var_31 = var_0.dump_payload(var_30)
    var_32 = var_0.load_payload(var_31)
    var_33 = 'outer'
    var_34 = 'numbers'
    var_35 = 'inner'
    var_36 = 'text'
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = 'hello'
    var_42 = {var_35: var_40, var_36: var_41}
    var_43 = 4
    var_44 = 5
    var_45 = 6
    var_46 = [var_43, var_44, var_45]
    var_47 = {var_33: var_42, var_34: var_46}
    var_48 = var_0.dump_payload(var_47)
    var_49 = var_0.load_payload(var_48)
    var_50 = None
    var_51 = {var_26: var_50}
    var_52 = var_0.dump_payload(var_51)
    var_53 = var_0.load_payload(var_52)
    var_54 = 'true_val'
    var_55 = 'false_val'
    var_56 = True
    var_57 = False
    var_58 = {var_54: var_56, var_55: var_57}
    var_59 = var_0.dump_payload(var_58)
    var_60 = var_0.load_payload(var_59)



# Parsed testcases at query #126
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed": "data"}'
    var_3 = b'.'
    var_4 = b'invalid_base64!!!'
    var_5 = b'.'
    var_6 = b'not_compressed_properly'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_5 + var_7
    var_9 = b''
    var_10 = b'.'
    var_11 = b'{"test": "data"}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = 'arg1'
    var_14 = 'value1'



# Parsed testcases at query #127
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b''
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #128
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key": "value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'{"compressed": true}'
    var_4 = b'.'
    var_5 = b'not-valid-base64!!'
    var_6 = b'not-compressed-data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b'{}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"nested": {"list": [1, 2, 3]}}'
    var_12 = module_0.base64_encode(var_11)



# Parsed testcases at query #129
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Test load_payload method of URLSafeSerializerMixin.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = 'data'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = b'.'
    var_11 = 'Large payload should be compressed'
    var_12 = b'not-valid-base64!!'
    var_13 = b'not compressed data'
    var_14 = module_0.base64_encode(var_13)
    var_15 = var_10 + var_14
    var_16 = b'{}'
    var_17 = module_0.base64_encode(var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = [var_18, var_19, var_20]
    var_22 = 1



# Parsed testcases at query #130
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{"data":"test with spaces and symbols: !@#"}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #131
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = module_2._CompactJSON()
    var_12 = b'{"test":123}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13, serializer=var_11)



# Parsed testcases at query #132
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous._json as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.URLSafeSerializer(var_0)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = b'.'
    var_12 = module_0.URLSafeSerializer(var_0)
    var_13 = 'test'
    var_14 = {var_13: var_3}
    var_15 = module_1._CompactJSON()
    var_16 = module_0.URLSafeSerializer(var_0)
    var_17 = b'!!!invalid_base64!!!'
    var_18 = module_0.URLSafeSerializer(var_0)
    var_19 = b'not_compressed_data'
    var_20 = module_2.base64_encode(var_19)
    var_21 = var_11 + var_20
    var_22 = module_0.URLSafeSerializer(var_0)
    var_23 = 'simple'
    var_24 = {var_23: var_13}
    var_25 = module_1._CompactJSON()



# Parsed testcases at query #133
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = b'.'
    var_13 = 'Large payload should be compressed'
    var_14 = var_0.load_payload(var_11)
    var_15 = b'invalid_base64!!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'.'
    var_18 = b'not_compressed_data'
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_17 + var_19
    var_21 = var_0.load_payload(var_20)
    var_22 = b''
    var_23 = var_0.load_payload(var_22)
    var_24 = b'.'
    var_25 = var_0.load_payload(var_24)



# Parsed testcases at query #134
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{"nested":{"list":[1,2,3],"bool":true,"null":null}}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = b'{"data":"'
    var_18 = b'a'
    var_19 = 100
    var_20 = var_18 * var_19
    var_21 = var_17 + var_20
    var_22 = b'"}'
    var_23 = var_21 + var_22
    var_24 = b'{"not_compressed":true}'
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_0.load_payload(var_25)



# Parsed testcases at query #135
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = ','
    var_5 = ':'
    var_6 = (var_4, var_5)
    var_7 = 'utf-8'
    var_8 = b'.'
    var_9 = b'invalid-base64!!!'
    var_10 = b'not-compressed-data'
    var_11 = module_0.base64_encode(var_10)
    var_12 = var_8 + var_11
    var_13 = b''
    var_14 = b'not-json'
    var_15 = module_0.base64_encode(var_14)



# Parsed testcases at query #136
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'{"special":"!@#$%^&*()"}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"nested":{"a":[1,2,3]}}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = 'data'
    var_14 = 'x'
    var_15 = 1000
    var_16 = var_14 * var_15
    var_17 = {var_13: var_16}
    var_18 = b'{"test":true}'
    var_19 = module_0.base64_encode(var_18)
    var_20 = var_2 + var_19
    var_21 = 'short'
    var_22 = {var_21: var_13}
    var_23 = 'long'
    var_24 = 100
    var_25 = var_14 * var_24
    var_26 = {var_23: var_25}



# Parsed testcases at query #137
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test load_payload method of URLSafeSerializerMixin'
    var_1 = module_0.URLSafeSerializerMixin()
    var_2 = '{"key": "value"}'
    var_3 = b'.'
    var_4 = b'invalid_base64!!'
    var_5 = var_1.load_payload(var_4)
    var_6 = b'not_compressed_data'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_3 + var_7
    var_9 = var_1.load_payload(var_8)
    var_10 = b'{}'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_1.load_payload(var_11)
    var_13 = 'nested'
    var_14 = 'list'
    var_15 = 'bool'
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = True
    var_21 = {var_14: var_19, var_15: var_20}
    var_22 = {var_13: var_21}
    var_23 = str(var_22)



# Parsed testcases at query #138
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = 1
    var_6 = var_4[var_5:]
    var_7 = var_0.load_payload(var_6)
    var_8 = 'data'
    var_9 = 'x'
    var_10 = 1000
    var_11 = var_9 * var_10
    var_12 = {var_8: var_11}
    var_13 = var_0.dump_payload(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'!!!invalid_base64!!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'.'
    var_18 = b'invalid_compressed_data'
    var_19 = var_17 + var_18
    var_20 = var_0.load_payload(var_19)



# Parsed testcases at query #139
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous._json as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.load_payload(var_11)
    var_13 = module_1._CompactJSON()
    var_14 = var_0.dump_payload(var_3)
    var_15 = var_0.load_payload(var_14, serializer=var_13)
    var_16 = b'invalid_base64!!!'
    var_17 = var_0.load_payload(var_16)
    var_18 = b'.'
    var_19 = b'not_compressed_data'
    var_20 = module_2.base64_encode(var_19)
    var_21 = var_18 + var_20
    var_22 = var_0.load_payload(var_21)
    var_23 = b'test_data'
    var_24 = module_2.base64_encode(var_23)
    var_25 = var_18 + var_24
    var_26 = var_0.load_payload(var_25)
    var_27 = b''
    var_28 = var_0.load_payload(var_27)
    var_29 = b'.'
    var_30 = var_0.load_payload(var_29)



# Parsed testcases at query #140
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJmb28iOiAiYmFyIn0='
    var_2 = var_0.load_payload(var_1)
    var_3 = 'very_long_key'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = b'invalid-base64!!'
    var_10 = var_0.load_payload(var_9)
    var_11 = b'not-compressed-data'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_8 + var_12
    var_14 = var_0.load_payload(var_13)
    var_15 = b'{}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)
    var_18 = 'key'
    var_19 = 'value with spaces & special chars!'
    var_20 = {var_18: var_19}
    var_21 = b'123'
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_0.load_payload(var_22)
    assert var_23 == 123
    var_24 = b'["a", "b", "c"]'
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_0.load_payload(var_25)



# Parsed testcases at query #141
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = b'!invalid_base64!'
    var_12 = b'.'
    var_13 = b'invalid_compressed_data'
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_12 + var_14
    var_16 = b''
    var_17 = 'special'
    var_18 = 'test_with_underscores_and-hyphens'
    var_19 = {var_17: var_18}
    var_20 = 'small'
    var_21 = 'test'
    var_22 = {var_20: var_21}
    var_23 = 'large'
    var_24 = 500
    var_25 = var_7 * var_24
    var_26 = {var_23: var_25}



# Parsed testcases at query #142
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = b'eyJ0ZXN0IjogImRhdGEifQ=='
    var_2 = 'test'
    var_3 = 'data'
    var_4 = 100
    var_5 = var_3 * var_4
    var_6 = {var_2: var_5}
    var_7 = b'.'
    var_8 = b'not-valid-base64!!!'
    var_9 = b'corrupted-data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_7 + var_10
    var_12 = module_0.URLSafeTimedSerializer()
    var_13 = b'eyJ0ZXN0IjogImRhdGEifQ=='



# Parsed testcases at query #143
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_1: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = var_0.load_payload(var_10)
    var_12 = b'invalid_base64!!!'
    var_13 = var_0.load_payload(var_12)
    var_14 = b'.'
    var_15 = b'not_compressed_data'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_14 + var_16
    var_18 = var_0.load_payload(var_17)
    var_19 = b''
    var_20 = var_0.load_payload(var_19)
    var_21 = module_2._CompactJSON()
    var_22 = 'test'
    var_23 = 123
    var_24 = {var_22: var_23}
    var_25 = var_0.dump_payload(var_24)
    var_26 = var_0.load_payload(var_25, serializer=var_21)
    var_27 = 'a'
    var_28 = 50
    var_29 = var_27 * var_28
    var_30 = {var_19: var_29}
    var_31 = var_0.dump_payload(var_30)
    var_32 = var_0.load_payload(var_31)



# Parsed testcases at query #144
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = '{"test": true}'
    var_11 = b'invalid-base64!!!'
    var_12 = b'not-actually-compressed'
    var_13 = b'{}'



# Parsed testcases at query #145
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = b'.'
    var_13 = "Compressed payload should start with b'.'"
    var_14 = var_0.load_payload(var_11)
    var_15 = 'small'
    var_16 = {var_15: var_6}
    var_17 = var_0.dump_payload(var_16)
    var_18 = var_0.load_payload(var_17)
    var_19 = b'!!!invalid_base64!!!'
    var_20 = var_0.load_payload(var_19)
    var_21 = b'not_compressed_data'
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_12 + var_22
    var_24 = var_0.load_payload(var_23)
    var_25 = {}
    var_26 = var_0.dump_payload(var_25)
    var_27 = var_0.load_payload(var_26)
    var_28 = 'string'
    var_29 = 'number'
    var_30 = 'list'
    var_31 = 'nested'
    var_32 = 'test'
    var_33 = 42
    var_34 = 1
    var_35 = 2
    var_36 = 3
    var_37 = [var_34, var_35, var_36]
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_39}
    var_41 = {var_28: var_32, var_29: var_33, var_30: var_37, var_31: var_40}
    var_42 = var_0.dump_payload(var_41)
    var_43 = var_0.load_payload(var_42)



# Parsed testcases at query #146
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"test":"data"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed":true}'
    var_3 = b'.'
    var_4 = b'invalid!!!'
    var_5 = b'not_compressed'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'{"special":"test/with+chars"}'
    var_11 = module_0.base64_encode(var_10)



# Parsed testcases at query #147
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b''
    var_12 = var_0.load_payload(var_11)
    var_13 = b'.'
    var_14 = var_0.load_payload(var_13)
    var_15 = module_2._CompactJSON()
    var_16 = var_0.load_payload(var_2, serializer=var_15)



# Parsed testcases at query #148
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{"nested":{"list":[1,2,3],"bool":true}}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = b'{"key":"value"}'



# Parsed testcases at query #149
#--------------------------


import src.itsdangerous._json as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'hello world'
    var_9 = module_0._CompactJSON()
    var_10 = b'invalid_base64!!!'
    var_11 = b'not_valid_zlib_data'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_3 + var_12



# Parsed testcases at query #150
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'data'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = {var_3: var_4}
    var_10 = b'invalid_base64!!!'
    var_11 = b'not_compressed_data'
    var_12 = module_0.base64_encode(var_11)
    var_13 = var_8 + var_12
    var_14 = b''
    var_15 = module_1._CompactJSON()
    var_16 = 'test'
    var_17 = 123
    var_18 = {var_16: var_17}
    var_19 = 'a'
    var_20 = 500
    var_21 = var_19 * var_20
    var_22 = {var_3: var_21}



# Parsed testcases at query #151
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Test load_payload method of URLSafeSerializerMixin'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'{"a":1}'
    var_5 = module_0.base64_encode(var_4)
    var_6 = b'invalid_base64!!!'
    var_7 = b'not_compressed'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_3 + var_8
    var_10 = b'{}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = b''



# Parsed testcases at query #152
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not compressed data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = 'nested'
    var_16 = 'value'
    var_17 = 'key'
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = [var_18, var_19, var_20]
    var_22 = {var_17: var_21}
    var_23 = 'test'
    var_24 = {var_15: var_22, var_16: var_23}
    var_25 = b'{"nested": {"key": [1, 2, 3]}, "value": "test"}'
    var_26 = module_1.base64_encode(var_25)
    var_27 = var_0.load_payload(var_26)



# Parsed testcases at query #153
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = b'.'
    var_13 = var_0.load_payload(var_11)
    var_14 = 'small'
    var_15 = {var_14: var_6}
    var_16 = var_0.dump_payload(var_15)
    var_17 = var_0.load_payload(var_16)
    var_18 = b'invalid-base64!!!'
    var_19 = var_0.load_payload(var_18)
    var_20 = b'not-compressed-data'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_12 + var_21
    var_23 = var_0.load_payload(var_22)
    var_24 = {}
    var_25 = var_0.dump_payload(var_24)
    var_26 = var_0.load_payload(var_25)
    var_27 = 'string'
    var_28 = 'number'
    var_29 = 'list'
    var_30 = 'nested'
    var_31 = 'hello'
    var_32 = 42
    var_33 = 1
    var_34 = 2
    var_35 = 3
    var_36 = [var_33, var_34, var_35]
    var_37 = 'inner'
    var_38 = {var_37: var_19}
    var_39 = {var_27: var_31, var_28: var_32, var_29: var_36, var_30: var_38}
    var_40 = var_0.dump_payload(var_39)
    var_41 = var_0.load_payload(var_40)



# Parsed testcases at query #154
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'eyJmb28iOiAiYmFyIn0='
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = b'.'
    var_7 = b'!!!invalid base64!!!'
    var_8 = b'not compressed data'
    var_9 = module_0.base64_encode(var_8)
    var_10 = var_6 + var_9
    var_11 = b'{}'
    var_12 = module_0.base64_encode(var_11)



# Parsed testcases at query #155
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = '{"key": "value"}'
    var_2 = b'.'
    var_3 = b'invalid_base64!!'
    var_4 = var_0.load_payload(var_3)
    var_5 = b'corrupted_data'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_2 + var_6
    var_8 = var_0.load_payload(var_7)
    var_9 = b'{}'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_0.load_payload(var_10)
    var_12 = '{"test": 123}'



# Parsed testcases at query #156
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = 'Large payload should be compressed'
    var_11 = b'!!!invalid base64!!!'
    var_12 = b'not json'
    var_13 = module_0.base64_encode(var_12)
    var_14 = b'not actually compressed'
    var_15 = module_0.base64_encode(var_14)
    var_16 = var_9 + var_15
    var_17 = b''



# Parsed testcases at query #157
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJmb28iOiAiYmFyIn0='
    var_2 = var_0.load_payload(var_1)
    var_3 = b'.eJzTyCkw5AIAAksDIg=='
    var_4 = var_0.load_payload(var_3)
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = str(var_5)
    var_8 = b'.eJzTyCkw5AIAAksDIg'
    var_9 = var_0.load_payload(var_8)
    var_10 = b''
    var_11 = var_0.load_payload(var_10)
    var_12 = b'.'
    var_13 = var_0.load_payload(var_12)
    var_14 = b'not json'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = module_2._CompactJSON()
    var_18 = b'eyJmb28iOiAiYmFyIn0='
    var_19 = var_0.load_payload(var_18, serializer=var_17)



# Parsed testcases at query #158
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'null'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    assert var_14 is None



# Parsed testcases at query #159
#--------------------------


def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'invalid-base64!!'
    var_10 = b'.invalid-base64'
    var_11 = b''
    var_12 = b'.'
    var_13 = 'test'
    var_14 = {var_13: var_4}
    var_15 = b'.'
    var_16 = {var_13: var_4}
    var_17 = 5



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"test": "compressed"}'
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = 'key1'
    var_16 = 'key2'
    var_17 = 'key3'
    var_18 = 'value1'
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = 'nested'
    var_24 = True
    var_25 = {var_23: var_24}
    var_26 = {var_15: var_18, var_16: var_22, var_17: var_25}
    var_27 = var_0.load_payload(var_13)
    var_28 = 'x'
    var_29 = 100
    var_30 = var_28 * var_29



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test load_payload with various scenarios.'
    var_1 = module_0.URLSafeSerializerMixin()
    var_2 = b'{"key":"value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1.load_payload(var_3)
    var_5 = b'.'
    var_6 = b'{}'
    var_7 = b'invalid-base64!!!'
    var_8 = var_1.load_payload(var_7)
    var_9 = b'.'
    var_10 = b'not-compressed-data'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_9 + var_11
    var_13 = var_1.load_payload(var_12)
    var_14 = b'.'
    var_15 = var_1.load_payload(var_14)
    var_16 = b'.invalid-base64'
    var_17 = var_1.load_payload(var_16)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Test load_payload method of URLSafeSerializerMixin.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = b'{"compressed":true}'
    var_11 = b'invalid-base64!!!'
    var_12 = b'not-compressed-data'
    var_13 = module_0.base64_encode(var_12)
    var_14 = var_9 + var_13
    var_15 = {}
    var_16 = 'string'
    var_17 = 'number'
    var_18 = 'list'
    var_19 = 'nested'
    var_20 = 'hello'
    var_21 = 42
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = 'a'
    var_27 = 'b'
    var_28 = {var_26: var_22, var_27: var_23}
    var_29 = {var_16: var_20, var_17: var_21, var_18: var_25, var_19: var_28}



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = module_1.base64_decode(var_4)
    var_7 = 'data'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = var_0.dump_payload(var_11)
    var_13 = 1
    var_14 = var_12[var_13:]
    var_15 = module_1.base64_decode(var_14)
    var_16 = {}
    var_17 = var_0.dump_payload(var_16)
    var_18 = module_1.base64_decode(var_17)
    var_19 = 'special'
    var_20 = 'test/data?query=1&param=2'
    var_21 = {var_19: var_20}
    var_22 = var_0.dump_payload(var_21)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.dump_payload(var_3)
    var_13 = var_0.load_payload(var_12)
    var_14 = var_0.dump_payload(var_10)
    var_15 = var_0.load_payload(var_14)
    var_16 = {}
    var_17 = var_0.dump_payload(var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = [var_18, var_19, var_20]
    var_22 = var_0.dump_payload(var_21)
    var_23 = var_0.load_payload(var_22)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'a'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_1: var_6}
    var_8 = '.'
    var_9 = '{"key":"value"}'
    var_10 = b'.'
    var_11 = b'invalid-base64!!!'
    var_12 = b'not-compressed-data'
    var_13 = module_0.base64_encode(var_12)
    var_14 = var_10 + var_13



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'eyJmb28iOiAiYmFyIn0='
    var_3 = 'data'
    var_4 = 'x'
    var_5 = 100
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'{"data": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}'
    var_9 = b'.'
    var_10 = b'!!!invalid_base64!!!'
    var_11 = b'corrupted_data'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_9 + var_12
    var_14 = b''
    var_15 = b'.'
    var_16 = b'eyJrZXkiOiAidmFsdWUifQ=='
    var_17 = b'eyJhIjogMX0='
    var_18 = None



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = 10000
    var_10 = var_5 * var_9
    var_11 = {var_4: var_10}
    var_12 = b'.'
    var_13 = 'test'
    var_14 = True
    var_15 = {var_13: var_14}



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = module_1.base64_decode(var_4)
    var_7 = var_0.dump_payload(var_3)
    var_8 = 'data'
    var_9 = 'x'
    var_10 = 1000
    var_11 = var_9 * var_10
    var_12 = {var_8: var_11}
    var_13 = var_0.dump_payload(var_12)
    var_14 = 1
    var_15 = var_13[var_14:]
    var_16 = module_1.base64_decode(var_15)
    var_17 = var_0.dump_payload(var_12)
    var_18 = {}
    var_19 = var_0.dump_payload(var_18)
    var_20 = 42
    var_21 = var_0.dump_payload(var_20)
    var_22 = 2
    var_23 = 3
    var_24 = [var_14, var_22, var_23]
    var_25 = var_0.dump_payload(var_24)
    var_26 = 'test'
    var_27 = 'number'
    var_28 = 123
    var_29 = {var_26: var_8, var_27: var_28}
    var_30 = var_0.dump_payload(var_29)
    var_31 = var_0.load_payload(var_30)
    var_32 = 'a'
    var_33 = 500
    var_34 = var_32 * var_33
    var_35 = {var_8: var_34}
    var_36 = var_0.dump_payload(var_35)
    var_37 = var_0.load_payload(var_36)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = module_1.base64_decode(var_4)
    var_7 = 'data'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = var_0.dump_payload(var_11)
    var_13 = 1
    var_14 = var_12[var_13:]
    var_15 = module_1.base64_decode(var_14)
    var_16 = 'a'
    var_17 = 50
    var_18 = var_16 * var_17
    var_19 = {var_7: var_18}
    var_20 = var_0.dump_payload(var_19)
    var_21 = {}
    var_22 = var_0.dump_payload(var_21)
    var_23 = len(var_22)
    var_24 = 'level1'
    var_25 = 'level2'
    var_26 = 2
    var_27 = 3
    var_28 = [var_13, var_26, var_27]
    var_29 = {var_25: var_28}
    var_30 = {var_24: var_29}
    var_31 = var_0.dump_payload(var_30)
    var_32 = var_0.load_payload(var_31)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'
    var_8 = 'a'
    var_9 = var_0.dump_payload(var_8)
    var_10 = var_0.load_payload(var_6)
    var_11 = var_0.load_payload(var_9)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'a'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"test":"data"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid!@#$'
    var_5 = b'not-compressed'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'{"key":"value with spaces & symbols"}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = b'{"count":123}'
    var_13 = module_0.base64_encode(var_12)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = b'{"key":"value"}'
    var_5 = module_0.base64_encode(var_4)
    var_6 = b'.'
    var_7 = b'invalid_base64!!!'
    var_8 = b'not_compressed_data'
    var_9 = module_0.base64_encode(var_8)
    var_10 = var_6 + var_9
    var_11 = b''
    var_12 = module_0.base64_encode(var_11)
    var_13 = b'{"test":123}'
    var_14 = module_0.base64_encode(var_13)
    var_15 = var_6 + var_14



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = 'test'
    var_10 = 'number'
    var_11 = 42
    var_12 = {var_9: var_4, var_10: var_11}
    var_13 = 'long_string'
    var_14 = 'a'
    var_15 = 500
    var_16 = var_14 * var_15
    var_17 = {var_13: var_16}
    var_18 = 'small'
    var_19 = {var_18: var_4}
    var_20 = {}



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = 'test'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = module_1.base64_decode(var_2)
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_3: var_9}
    var_11 = b'.'
    var_12 = 1
    var_13 = {}
    var_14 = 2
    var_15 = 3
    var_16 = [var_12, var_14, var_15, var_2]
    var_17 = 'utf-8'
    var_18 = '+'
    var_19 = '/'
    var_20 = '='
    var_21 = [var_18, var_19, var_20]
    var_22 = module_2._CompactJSON()



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'data'
    var_5 = 'a'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = b'not-valid-base64!!'
    var_11 = b'not-valid-json'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_9 + var_12
    var_14 = {}
    var_15 = {}



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = module_1.base64_decode(var_4)
    var_7 = var_0.load_payload(var_4)
    var_8 = module_0.URLSafeSerializerMixin()
    var_9 = 'data'
    var_10 = 'a'
    var_11 = 1000
    var_12 = var_10 * var_11
    var_13 = {var_9: var_12}
    var_14 = var_8.dump_payload(var_13)
    var_15 = 1
    var_16 = var_14[var_15:]
    var_17 = module_1.base64_decode(var_16)
    var_18 = var_8.load_payload(var_14)
    var_19 = module_0.URLSafeSerializerMixin()
    var_20 = 'short'
    var_21 = {var_15: var_20}
    var_22 = var_19.dump_payload(var_21)
    var_23 = module_0.URLSafeSerializerMixin()
    var_24 = {}
    var_25 = var_23.dump_payload(var_24)
    var_26 = var_23.load_payload(var_25)



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = module_0.URLSafeSerializer()
    var_7 = 'a'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {}
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = b'^[A-Za-z0-9_-]+$'
    var_17 = 12345
    var_18 = 'a'
    var_19 = 'b'
    var_20 = 'c'
    var_21 = 'd'
    var_22 = {var_20: var_21}
    var_23 = [var_12, var_13, var_22]
    var_24 = {var_19: var_23}
    var_25 = {var_18: var_24}
    var_26 = True
    var_27 = None
    var_28 = 20
    var_29 = var_8 * var_28



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous._json as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = var_0.dump_payload(var_8)
    var_10 = b'.'
    var_11 = var_0.load_payload(var_9)
    var_12 = module_1._CompactJSON()
    var_13 = var_0.dump_payload(var_3)
    var_14 = var_0.load_payload(var_13, serializer=var_12)
    var_15 = b'invalid_base64!!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'not_compressed_data'
    var_18 = module_2.base64_encode(var_17)
    var_19 = var_10 + var_18
    var_20 = var_0.load_payload(var_19)
    var_21 = b''
    var_22 = var_0.load_payload(var_21)
    var_23 = b''
    var_24 = module_2.base64_encode(var_23)
    var_25 = var_10 + var_24
    var_26 = var_0.load_payload(var_25)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'Test dump_payload method of URLSafeSerializerMixin.'
    var_1 = module_0.URLSafeSerializer()
    var_2 = 'a'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = b'.'
    var_6 = 'hello'
    var_7 = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.'
    var_8 = ''



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = b'.'
    var_5 = module_0.URLSafeSerializer()
    var_6 = 'abc123'
    var_7 = module_0.URLSafeSerializer()
    var_8 = ''
    var_9 = module_0.URLSafeSerializer()
    var_10 = None



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Test load_payload method of URLSafeSerializerMixin.'
    var_1 = b'eyJmb28iOiAiYmFyIn0'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 100
    var_5 = var_3 * var_4
    var_6 = {var_2: var_5}
    var_7 = b'.'
    var_8 = b'invalid_base64!!!'
    var_9 = b'not_compressed_data'
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_7 + var_10
    var_12 = b''
    var_13 = b'.'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'ey'
    var_4 = b'.'
    var_5 = 'data'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = {}
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = b'.'
    var_5 = b'invalid-base64!!!'
    var_6 = b'not-compressed-data'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b'{}'
    var_10 = module_1.base64_encode(var_9)
    var_11 = b'{"string": "test", "number": 42, "list": [1,2,3]}'
    var_12 = module_1.base64_encode(var_11)



# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = 1
    var_6 = var_4[var_5:]
    var_7 = var_0.load_payload(var_6)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'x'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'
    var_8 = 'Large payload should be compressed'
    var_9 = var_0.load_payload(var_6)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'!!!invalid_base64!!!'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'not_compressed_data'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = module_1._CompactJSON()
    var_6 = var_0.load_payload(var_4, serializer=var_5)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b''
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_2: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = 1
    var_12 = var_10[var_11:]
    var_13 = module_1.base64_decode(var_12)
    var_14 = 'simple'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = var_0.dump_payload(var_16)
    var_18 = module_1.base64_decode(var_17)
    var_19 = {}
    var_20 = var_0.dump_payload(var_19)
    var_21 = 'number'
    var_22 = 'float'
    var_23 = 42
    var_24 = 3.14
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = var_0.dump_payload(var_25)
    var_27 = 'items'
    var_28 = 2
    var_29 = 3
    var_30 = 4
    var_31 = 5
    var_32 = [var_11, var_28, var_29, var_30, var_31]
    var_33 = {var_27: var_32}
    var_34 = var_0.dump_payload(var_33)
    var_35 = 'a'
    var_36 = 'b'
    var_37 = {var_35: var_36}
    var_38 = var_0.dump_payload(var_37)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'message'
    var_3 = 'value'
    var_4 = 'hello'
    var_5 = 42
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_1.dump_payload(var_6)
    var_8 = b'.'
    var_9 = module_1.base64_decode(var_7)
    assert var_9 == b'{"message":"hello","value":42}'
    var_10 = 'a'
    var_11 = 100
    var_12 = var_10 * var_11
    var_13 = 'data'
    var_14 = {var_13: var_12}
    var_15 = var_1.dump_payload(var_14)
    var_16 = 1
    var_17 = var_15[var_16:]
    var_18 = module_1.base64_decode(var_17)
    var_19 = 'x'
    var_20 = {var_19: var_16}
    var_21 = var_1.dump_payload(var_20)
    var_22 = module_1.base64_decode(var_21)
    assert var_22 == b'{"x":1}'
    var_23 = {}
    var_24 = var_1.dump_payload(var_23)
    var_25 = module_1.base64_decode(var_24)
    assert var_25 == b'{}'
    var_26 = 'test'
    var_27 = 'nested'
    var_28 = 'key'
    var_29 = {var_28: var_3}
    var_30 = {var_26: var_13, var_27: var_29}
    var_31 = var_1.dump_payload(var_30)
    var_32 = 1
    var_33 = var_31[var_32:]
    var_34 = True
    var_35 = var_31
    var_36 = False
    var_37 = module_1.base64_decode(var_35)
    assert var_37 == b'{"test":"data","nested":{"key":"value"}}'



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = b'{"compressed": true}'
    var_5 = b'.'
    var_6 = b'not-valid-base64!!!'
    var_7 = b'corrupted-data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_5 + var_8
    var_10 = b'{}'
    var_11 = module_1.base64_encode(var_10)
    var_12 = b'{"dot": "prefix"}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_5 + var_13
    var_15 = 'test'
    var_16 = 'number'
    var_17 = 'data'
    var_18 = 42
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'test-key'
    var_21 = b'{"custom": true}'
    var_22 = module_1.base64_encode(var_21)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'.'
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = b'invalid_base64!!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = 'url'
    var_11 = 'https://example.com/path?query=value&more=test'
    var_12 = {var_10: var_11}



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'!!!invalid_base64!!!'
    var_4 = b'not_compressed'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'null'
    var_8 = module_0.base64_encode(var_7)
    var_9 = module_1._CompactJSON()
    var_10 = b'{"custom":"data"}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = 'c'
    var_19 = 'd'
    var_20 = {var_18: var_19}
    var_21 = {var_12: var_17, var_13: var_20}
    var_22 = b'{"a":[1,2,3],"b":{"c":"d"}}'
    var_23 = module_0.base64_encode(var_22)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = b'.'
    var_13 = 'Large payload should be compressed'
    var_14 = var_0.load_payload(var_11)
    var_15 = 'small'
    var_16 = {var_15: var_6}
    var_17 = var_0.dump_payload(var_16)
    var_18 = var_0.load_payload(var_17)
    var_19 = b'invalid_base64!!!'
    var_20 = var_0.load_payload(var_19)
    var_21 = b'not compressed data'
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_12 + var_22
    var_24 = var_0.load_payload(var_23)
    var_25 = b''
    var_26 = var_0.load_payload(var_25)
    var_27 = b'.'
    var_28 = var_0.load_payload(var_27)
    var_29 = module_2._CompactJSON()
    var_30 = 'test'
    var_31 = {var_30: var_6}
    var_32 = var_0.dump_payload(var_31)
    var_33 = var_0.load_payload(var_32, serializer=var_29)
    var_34 = 'round'
    var_35 = 'data'
    var_36 = 'test'
    var_37 = 50
    var_38 = var_36 * var_37
    var_39 = var_0.dump_payload(var_31)
    var_40 = var_0.load_payload(var_39)



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed":true}'
    var_3 = b'.'
    var_4 = b'{"short":1}'
    var_5 = b'.'
    var_6 = var_5 + var_2
    var_7 = module_0.base64_encode(var_4)
    var_8 = b'invalid_base64!!!'
    var_9 = b'.'
    var_10 = b'not_compressed_data'
    var_11 = module_0.base64_encode(var_10)
    var_12 = var_9 + var_11
    var_13 = b''
    var_14 = b'{"test":true}'
    var_15 = module_0.base64_encode(var_14)
    var_16 = var_11 + var_15



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = module_1.base64_encode(var_1)
    var_7 = var_0.load_payload(var_6)
    var_8 = b'invalid_base64!!!'
    var_9 = var_0.load_payload(var_8)
    var_10 = b'not_compressed_data'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_9 + var_11
    var_13 = var_0.load_payload(var_12)
    var_14 = b'null'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    assert var_16 is None
    var_17 = b'123'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_0.load_payload(var_18)
    assert var_19 == 123
    var_20 = b'[1, 2, 3]'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_0.load_payload(var_21)
    var_23 = 'key'
    var_24 = 'x'
    var_25 = 1000
    var_26 = var_24 * var_25
    var_27 = {var_23: var_26}
    var_28 = module_2._CompactJSON()
    var_29 = var_0.load_payload(var_21)
    var_30 = b'{"test": "data"}'
    var_31 = module_1.base64_encode(var_30)
    var_32 = var_9 + var_31
    var_33 = var_0.load_payload(var_32)



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = var_0.dump_payload(var_8)
    var_10 = b'.'
    var_11 = var_0.load_payload(var_9)
    var_12 = b'invalid_base64!!!'
    var_13 = var_0.load_payload(var_12)
    var_14 = b'.'
    var_15 = b'not_compressed_data'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_14 + var_16
    var_18 = var_0.load_payload(var_17)
    var_19 = module_0.URLSafeSerializerMixin()
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = var_19.dump_payload(var_23)
    var_25 = var_19.load_payload(var_24)
    var_26 = 'short'
    var_27 = var_0.dump_payload(var_26)
    var_28 = var_0.load_payload(var_27)



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'{"key":"value"}'
    var_5 = module_1.base64_encode(var_4)
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'invalid_base64!!!'
    var_9 = var_0.load_payload(var_8)
    var_10 = b'not_compressed_data'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_7 + var_11
    var_13 = var_0.load_payload(var_12)
    var_14 = b''
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = b'.'
    var_18 = var_0.load_payload(var_17)
    var_19 = module_2._CompactJSON()
    var_20 = b'{"test":123}'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_0.load_payload(var_21, serializer=var_19)



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_actually_compressed'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{"key":"value with spaces & symbols!"}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #41
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'TestSerializer'
    var_1 = {}
    var_2 = b'{"key":"value"}'
    var_3 = module_0.base64_encode(var_2)
    var_4 = b'{"compressed":true}'
    var_5 = b'.'
    var_6 = b'{"not_compressed":true}'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_5 + var_7
    var_9 = b'invalid_base64!!!'
    var_10 = b''
    var_11 = b'.'



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous._json as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = 'Large payload should be compressed'
    var_11 = module_1._CompactJSON()
    var_12 = 'custom'
    var_13 = True
    var_14 = {var_12: var_13}
    var_15 = b'!@#$%^'
    var_16 = b'not compressed data'
    var_17 = module_2.base64_encode(var_16)
    var_18 = var_9 + var_17



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'{"special":"!@#$%^&*()"}'
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"compressed": true}'
    var_5 = b'.'
    var_6 = b'{"not": "compressed"}'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_5 + var_7
    var_9 = var_0.load_payload(var_8)
    var_10 = b'not-valid-base64!!!'
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = 'CustomSerializer'
    var_16 = ()
    var_17 = 'loads'
    var_18 = 'custom'
    var_19 = lambda x: {var_18: x.decode()}
    var_20 = staticmethod(var_19)
    var_21 = {var_17: var_20}
    var_22 = type(var_15, var_16, var_21)
    var_23 = b'custom data'
    var_24 = module_1.base64_encode(var_23)
    var_25 = b'not-zlib-compressed'
    var_26 = module_1.base64_encode(var_25)
    var_27 = var_5 + var_26
    var_28 = var_0.load_payload(var_27)



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = b'eyJmb28iOiAiYmFyIn0='
    var_1 = b'.eJwljjkOAyEMAP.CpEFiOXqP5IqWYqNYIqX47'
    var_2 = b'invalid!base64@@'
    var_3 = b'.aW52YWxpZCB6bGliIGRhdGE='
    var_4 = b'.eyJmb28iOiAiYmFyIn0='
    var_5 = b''
    var_6 = b'eyJuZXN0ZWQiOiB7ImtleSI6ICJ2YWx1ZSJ9fQ=='
    var_7 = b'WzEsIDIsIDNd'



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid-base64!'
    var_5 = b'not-compressed-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'{"test":"data"}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = var_3 + var_11



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '{"key":"value"}'
    var_5 = b'.'
    var_6 = b'invalid_base64!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'.'
    var_9 = b'corrupted_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)
    var_13 = b''
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_0.load_payload(var_14)
    var_16 = 'special'
    var_17 = '!@#$%^&*()'
    var_18 = {var_16: var_17}
    var_19 = '{"special":"!@#$%^&*()"}'
    var_20 = 42
    var_21 = '42'
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = '[1,2,3]'
    var_27 = 'a'
    var_28 = 'b'
    var_29 = 'c'
    var_30 = {var_28: var_29}
    var_31 = {var_27: var_30}
    var_32 = '{"a":{"b":"c"}}'
    var_33 = 'data'
    var_34 = 'x'
    var_35 = 100
    var_36 = var_34 * var_35
    var_37 = {var_33: var_36}
    var_38 = '{"data":"'
    var_39 = var_34 * var_35
    var_40 = var_38 + var_39
    var_41 = '"}'
    var_42 = var_40 + var_41



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'invalid_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b''
    var_8 = b'.'



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b''
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = 'nested'
    var_15 = 'list'
    var_16 = 'bool'
    var_17 = 'null'
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = [var_18, var_19, var_20]
    var_22 = True
    var_23 = None
    var_24 = {var_15: var_21, var_16: var_22, var_17: var_23}
    var_25 = {var_14: var_24}
    var_26 = str(var_25)
    var_27 = var_0.load_payload(var_2)
    var_28 = b'simple test'
    var_29 = module_1.base64_encode(var_28)
    var_30 = var_0.load_payload(var_29)
    assert var_30 == 'simple test'



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'{"compressed":true}'
    var_4 = b'.'
    var_5 = b'invalid-base64!!!'
    var_6 = b'not-zlib-compressed'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b'{}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"special":"!@#$%^&*()"}'
    var_12 = module_0.base64_encode(var_11)



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'{}'
    var_4 = module_0.base64_encode(var_3)
    var_5 = b'[1,2,3]'
    var_6 = module_0.base64_encode(var_5)
    var_7 = b'invalid_base64!!!'
    var_8 = b'not_compressed'
    var_9 = module_0.base64_encode(var_8)
    var_10 = var_2 + var_9
    var_11 = b'{"a":1,"b":{"c":2}}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = b''
    var_14 = module_0.base64_encode(var_13)
    var_15 = b'{"text":"hello world"}'
    var_16 = module_0.base64_encode(var_15)
    var_17 = b'{"first":1}'
    var_18 = b'{"second":2}'
    var_19 = module_0.base64_encode(var_18)



# Parsed testcases at query #52
#--------------------------


import src.itsdangerous._json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'x'
    var_4 = 1000
    var_5 = var_3 * var_4
    var_6 = {var_0: var_5}
    var_7 = b'.'
    var_8 = "Compressed payload should start with b'.'"
    var_9 = b'invalid_base64!!!'
    var_10 = b'.'
    var_11 = b'aGVsbG8='
    var_12 = var_10 + var_11
    var_13 = module_0._CompactJSON()
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]



# Parsed testcases at query #53
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid-base64!!!'
    var_4 = b'not-compressed-data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = 'nested'
    var_10 = 'list'
    var_11 = 'bool'
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = True
    var_17 = {var_10: var_15, var_11: var_16}
    var_18 = {var_9: var_17}
    var_19 = b'{"nested":{"list":[1,2,3],"bool":true}}'
    var_20 = module_0.base64_encode(var_19)
    var_21 = b'{"test":"data"}'
    var_22 = module_0.base64_encode(var_21)
    var_23 = var_2 + var_22



# Parsed testcases at query #54
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = 'Test load_payload method of URLSafeSerializerMixin.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = '{"test": "data"}'
    var_10 = b'.'
    var_11 = b'!!!invalid_base64!!!'
    var_12 = b'not_compressed_data'
    var_13 = module_0.base64_encode(var_12)
    var_14 = var_10 + var_13
    var_15 = b''
    var_16 = b'.'
    var_17 = 'small'
    var_18 = {var_17: var_4}
    var_19 = module_1._CompactJSON()
    var_20 = 'custom'
    var_21 = {var_20: var_4}



# Parsed testcases at query #55
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'.'



# Parsed testcases at query #56
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"compressed": true}'
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'{"nested": {"key": [1, 2, 3]}}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)



# Parsed testcases at query #57
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = var_0.load_payload(var_6)
    var_8 = 'data'
    var_9 = 'x'
    var_10 = 1000
    var_11 = var_9 * var_10
    var_12 = {var_8: var_11}
    var_13 = var_0.dump_payload(var_12)
    var_14 = b'.'
    var_15 = "Compressed payload should start with '.'"
    var_16 = var_0.load_payload(var_13)
    var_17 = b'invalid_base64!'
    var_18 = var_0.load_payload(var_17)
    var_19 = b'.'
    var_20 = b'not_compressed_data'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_19 + var_21
    var_23 = var_0.load_payload(var_22)
    var_24 = b''
    var_25 = var_0.load_payload(var_24)
    var_26 = b'.'
    var_27 = var_0.load_payload(var_26)
    var_28 = 'small'
    var_29 = {var_28: var_8}
    var_30 = var_0.dump_payload(var_29)
    var_31 = var_0.load_payload(var_30)



# Parsed testcases at query #58
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'eyJmb28iOiAiYmFyIn0'
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = b'.'
    var_7 = b'invalid-base64!!!'
    var_8 = b'not-compressed-data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_6 + var_9



# Parsed testcases at query #59
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = var_0.dump_payload(var_8)
    var_10 = var_0.load_payload(var_9)
    var_11 = b'.'
    var_12 = b'{"test":"data"}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'invalid_base64!!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'not_compressed_data'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_11 + var_18
    var_20 = var_0.load_payload(var_19)
    var_21 = b'{}'
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_0.load_payload(var_22)



# Parsed testcases at query #60
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Test load_payload with various scenarios including compressed and uncompressed payloads.'
    var_1 = 'test-secret'
    var_2 = b'eyJmb28iOiAiYmFyIn0'
    var_3 = 'test'
    var_4 = 'data'
    var_5 = 100
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = b'invalid-base64!!!'
    var_10 = b'not-compressed-data'
    var_11 = module_0.base64_encode(var_10)
    var_12 = var_8 + var_11
    var_13 = b''



# Parsed testcases at query #61
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous._json as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_1._CompactJSON()
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = module_1._CompactJSON()
    var_9 = b'.'
    var_10 = b'invalid_base64!!!'
    var_11 = var_0.load_payload(var_10)
    var_12 = b'not_compressed_data'
    var_13 = module_2.base64_encode(var_12)
    var_14 = var_9 + var_13
    var_15 = var_0.load_payload(var_14)
    var_16 = module_1._CompactJSON()
    var_17 = None
    var_18 = var_0.load_payload(var_14)
    assert var_18 is None
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = 'nested'
    var_24 = {var_15: var_11}
    var_25 = {var_23: var_24}
    var_26 = 'simple string'
    var_27 = 42
    var_28 = True
    var_29 = False
    var_30 = [var_22, var_25, var_26, var_27, var_28, var_29]
    var_31 = module_1._CompactJSON()
    var_32 = module_2.base64_encode(var_4)
    var_33 = var_0.load_payload(var_32)



# Parsed testcases at query #62
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = b'.'
    var_5 = b'{"compressed": true}'
    var_6 = b'invalid_base64!!!'
    var_7 = b'not_compressed_data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = b'{}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = 'list'
    var_13 = 'nested'
    var_14 = 'bool'
    var_15 = 'null'
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = 'a'
    var_21 = {var_20: var_16}
    var_22 = True
    var_23 = None
    var_24 = {var_12: var_19, var_13: var_21, var_14: var_22, var_15: var_23}
    var_25 = 1
    var_26 = module_1._CompactJSON()
    var_27 = 'custom'
    var_28 = True
    var_29 = {var_27: var_28}



# Parsed testcases at query #63
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'not-valid-base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not-compressed-data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b''
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)



# Parsed testcases at query #64
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid-base64!!!'
    var_4 = b'not-zlib-compressed'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b''
    var_8 = b'.'
    var_9 = module_1._CompactJSON()
    var_10 = b'{"custom":"data"}'
    var_11 = module_0.base64_encode(var_10)



# Parsed testcases at query #65
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = module_0.URLSafeSerializerMixin()
    var_7 = 'data'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = var_6.dump_payload(var_11)
    var_13 = b'.'
    var_14 = var_6.load_payload(var_12)
    var_15 = module_0.URLSafeSerializerMixin()
    var_16 = 'small'
    var_17 = {var_7: var_16}
    var_18 = var_15.dump_payload(var_17)
    var_19 = var_15.load_payload(var_18)
    var_20 = module_0.URLSafeSerializerMixin()
    var_21 = b'!!!invalid_base64!!!'
    var_22 = var_20.load_payload(var_21)
    var_23 = module_0.URLSafeSerializerMixin()
    var_24 = b'invalid_compressed_data'
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_13 + var_25
    var_27 = var_23.load_payload(var_26)
    var_28 = module_0.URLSafeSerializerMixin()
    var_29 = b''
    var_30 = var_28.load_payload(var_29)
    var_31 = module_0.URLSafeSerializerMixin()
    var_32 = b'.'
    var_33 = var_31.load_payload(var_32)



# Parsed testcases at query #66
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'{}'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_0.load_payload(var_6)
    var_8 = b'{"count": 42}'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{"items": [1, 2, 3]}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'invalid_base64!!'
    var_15 = var_0.load_payload(var_14)
    var_16 = module_2._CompactJSON()
    var_17 = b'{"custom": true}'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_0.load_payload(var_18, serializer=var_16)



# Parsed testcases at query #67
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test":"data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"test":"compressed"}'
    var_5 = b'.'
    var_6 = b'{}'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_0.load_payload(var_7)
    var_9 = b'invalid_base64!!'
    var_10 = var_0.load_payload(var_9)
    var_11 = b'not_compressed'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_5 + var_12
    var_14 = var_0.load_payload(var_13)



# Parsed testcases at query #68
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = 'Test load_payload method with various scenarios.'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid_base64!!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = 'nested'
    var_11 = 'array'
    var_12 = 'bool'
    var_13 = 'null'
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = True
    var_19 = None
    var_20 = {var_11: var_17, var_12: var_18, var_13: var_19}
    var_21 = {var_10: var_20}
    var_22 = module_1._CompactJSON()
    var_23 = b'{"custom":"data"}'
    var_24 = module_0.base64_encode(var_23)
    var_25 = 'data'
    var_26 = 'x'
    var_27 = 1000
    var_28 = var_26 * var_27
    var_29 = {var_25: var_28}
    var_30 = 'special'
    var_31 = '!@#$%^&*()_+-=[]{}|;\':",./<>?`~'
    var_32 = {var_30: var_31}
    var_33 = b'{"test":true}'
    var_34 = module_0.base64_encode(var_33)



# Parsed testcases at query #69
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"compressed":true}'
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'{"nested":{"key":[1,2,3]}}'
    var_16 = b'{"special":"!@#$%^&*()"}'
    var_17 = module_1.base64_encode(var_16)
    var_18 = var_0.load_payload(var_17)



# Parsed testcases at query #70
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'{"key": "value2"}'
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_4 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = module_2._CompactJSON()
    var_16 = 'test'
    var_17 = 'data'
    var_18 = {var_16: var_17}



# Parsed testcases at query #71
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = 'test'
    var_5 = 'data'
    var_6 = 100
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = b'invalid_base64!!!'
    var_11 = var_0.load_payload(var_10)
    var_12 = b'not_compressed_data'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_9 + var_13
    var_15 = var_0.load_payload(var_14)
    var_16 = b'{}'
    var_17 = module_1.base64_encode(var_16)
    var_18 = var_0.load_payload(var_17)
    var_19 = 'special'
    var_20 = '!@#$%^&*()'
    var_21 = {var_19: var_20}
    var_22 = 'level1'
    var_23 = 'level2'
    var_24 = 'level2_2'
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_23: var_28, var_24: var_11}
    var_30 = {var_22: var_29}
    var_31 = module_2._CompactJSON()
    var_32 = 'custom'
    var_33 = 'serializer'
    var_34 = {var_32: var_33}



# Parsed testcases at query #72
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJmb28iOiAiYmFyIn0='
    var_2 = var_0.load_payload(var_1)
    var_3 = b'.eJw1yjsKwCAMANC9pC4OQvoH0KWDQ6GtYqF4d_Xd3vAmO0vNIUQEYJ4FJtu1ImfPB1pNFd8='
    var_4 = var_0.load_payload(var_3)
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.aW52YWxpZGNvbXByZXNzZWQ='
    var_8 = var_0.load_payload(var_7)
    var_9 = b''
    var_10 = var_0.load_payload(var_9)
    var_11 = b'.'
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #73
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJmb28iOiAiYmFyIn0'
    var_2 = var_0.load_payload(var_1)
    var_3 = 'test'
    var_4 = 'data'
    var_5 = 100
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = var_0.load_payload(var_1)
    var_10 = b'invalid_base64!!!'
    var_11 = var_0.load_payload(var_10)
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = -5
    var_16 = b'XXXXX'
    var_17 = var_0.load_payload(var_1)
    var_18 = b''
    var_19 = var_0.load_payload(var_18)
    var_20 = b'.'
    var_21 = var_0.load_payload(var_20)



# Parsed testcases at query #74
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = 'test'
    var_5 = 'data'
    var_6 = 100
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = b'invalid_base64!!!'
    var_11 = var_0.load_payload(var_10)
    var_12 = b'not_compressed_data'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_9 + var_13
    var_15 = var_0.load_payload(var_14)
    var_16 = b'{}'
    var_17 = module_1.base64_encode(var_16)
    var_18 = var_0.load_payload(var_17)
    var_19 = 'special'
    var_20 = '!@#$%^&*()'
    var_21 = {var_19: var_20}
    var_22 = 'level1'
    var_23 = 'level2'
    var_24 = 'key'
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = 'value'
    var_30 = {var_23: var_28, var_24: var_29}
    var_31 = {var_22: var_30}
    var_32 = 'two'
    var_33 = 'nested'
    var_34 = True
    var_35 = {var_33: var_34}
    var_36 = [var_25, var_32, var_27, var_35]
    var_37 = b'.'
    var_38 = var_0.load_payload(var_37)
    var_39 = b'..'
    var_40 = module_1.base64_encode(var_37)
    var_41 = var_39 + var_40
    var_42 = var_0.load_payload(var_41)



# Parsed testcases at query #75
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'not-valid-base64!!!'
    var_5 = b'corrupted-compressed-data'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'null'
    var_9 = module_1.base64_encode(var_8)
    var_10 = 'nested'
    var_11 = 'list'
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = [var_15, var_16, var_17]
    var_19 = {var_10: var_14, var_11: var_18}



# Parsed testcases at query #76
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'test_payload'
    var_3 = b'x'
    var_4 = 1000
    var_5 = var_3 * var_4
    var_6 = b'.'
    var_7 = "Expected compressed payload to start with '.'"
    var_8 = b'invalid_base64'
    var_9 = b'not_zlib_compressed'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_6 + var_10
    var_12 = b''
    var_13 = b'short'



# Parsed testcases at query #77
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test load_payload method with various scenarios.'
    var_1 = module_0.URLSafeSerializerMixin()
    var_2 = b'{"key": "value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1.load_payload(var_3)
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = var_1.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = var_1.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_1.load_payload(var_13)
    var_15 = b'{"nested": {"list": [1, 2, 3]}}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_1.load_payload(var_16)



# Parsed testcases at query #78
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'not-valid-base64!!'
    var_4 = b'not-compressed-data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'{"special":"test&value"}'
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #79
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'{"nested":{"key":123,"list":[1,2,3]}}'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_0.load_payload(var_6)
    var_8 = b'{}'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_0.load_payload(var_9)
    var_11 = b'invalid_base64!!!'
    var_12 = var_0.load_payload(var_11)
    var_13 = b'not_compressed_data'
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_12 + var_14
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #80
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-key'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = b'.'
    var_5 = b'invalid-base64!!!'
    var_6 = b'not-compressed-data'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b'null'
    var_10 = module_1.base64_encode(var_9)
    var_11 = b'[1, 2, 3]'
    var_12 = module_1.base64_encode(var_11)
    var_13 = b'{"a": {"b": "c"}}'
    var_14 = module_1.base64_encode(var_13)



# Parsed testcases at query #81
#--------------------------


def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = b'.'
    var_12 = 'Large payload should be compressed'
    var_13 = b'!!!invalid-base64!!!'
    var_14 = b'.invalid-base64'
    var_15 = b''
    var_16 = b'.'
    var_17 = 'small'
    var_18 = {var_17: var_6}



# Parsed testcases at query #82
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = b'invalid-base64!!!'
    var_10 = b'.'
    var_11 = b'not-actually-compressed'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_10 + var_12
    var_14 = b''
    var_15 = b'.'



# Parsed testcases at query #83
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = module_0.base64_encode(var_0)
    var_4 = var_2 + var_3
    var_5 = b'invalid_base64!!!'
    var_6 = b'not_compressed_data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_2 + var_7
    var_9 = b'.'
    var_10 = b'test_data'
    var_11 = module_0.base64_encode(var_10)



# Parsed testcases at query #84
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_actually'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_3 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'{"url": "https://example.com/path?param=value&other=1"}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"a": 1}'
    var_12 = b'[1, 2, 3]'
    var_13 = module_0.base64_encode(var_12)



# Parsed testcases at query #85
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = b'eyJrZXkiOiAidmFsdWUifQ=='
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = 'utf-8'
    var_7 = b'.'
    var_8 = b'not-valid-base64!!'
    var_9 = b'not-compressed-data'
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_7 + var_10
    var_12 = b'{}'
    var_13 = module_0.base64_encode(var_12)
    var_14 = 'special'
    var_15 = 'test/with+slashes_and_dashes'
    var_16 = {var_14: var_15}
    var_17 = 'small'
    var_18 = 'data'
    var_19 = {var_17: var_18}
    var_20 = b'[1, 2, 3]'
    var_21 = module_0.base64_encode(var_20)
    var_22 = b'.'
    var_23 = module_1.Serializer()
    var_24 = b'{"custom": true}'
    var_25 = module_0.base64_encode(var_24)



# Parsed testcases at query #86
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'{"key":"value"}'
    var_9 = b'invalid_base64!!!'
    var_10 = var_0.load_payload(var_9)
    var_11 = b'not_compressed_data'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_7 + var_12
    var_14 = var_0.load_payload(var_13)



# Parsed testcases at query #87
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test load_payload method with various scenarios.'
    var_1 = module_0.URLSafeSerializer()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'data'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = b'.'
    var_11 = "Compressed payload should start with '.'"
    var_12 = b'invalid-base64!!!'
    var_13 = b'corrupted_compressed_data'
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_10 + var_14
    var_16 = b''
    var_17 = b'.'
    var_18 = 'test'
    var_19 = {var_18: var_5}
    var_20 = 1



# Parsed testcases at query #88
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = b'eyJ0ZXN0IjogImRhdGEifQ=='
    var_2 = b'{"test": "data"}'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = b'.'
    var_6 = b'invalid-base64!!!'
    var_7 = b'not-zlib-compressed'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_5 + var_8
    var_10 = b''
    var_11 = b'.'



# Parsed testcases at query #89
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Test URLSafeSerializerMixin.load_payload with various scenarios.'
    var_1 = 'test-secret'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = b'.'
    var_6 = 1
    var_7 = 'data'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = b'{"test": "data"}'
    var_13 = module_0.base64_encode(var_12)
    var_14 = b'invalid-base64!!!'
    var_15 = b'corrupted-data'
    var_16 = module_0.base64_encode(var_15)
    var_17 = var_5 + var_16
    var_18 = b'null'
    var_19 = module_0.base64_encode(var_18)
    var_20 = b'["a", "b", "c"]'
    var_21 = module_0.base64_encode(var_20)
    var_22 = b'{"key": "value"}'
    var_23 = module_0.base64_encode(var_22)
    var_24 = 'extra_arg'
    var_25 = 'test'



# Parsed testcases at query #90
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"compressed": true}'
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_valid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)



# Parsed testcases at query #91
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"compressed":true}'
    var_5 = b'.'
    var_6 = b'not-valid-base64!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'corrupted-data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    assert var_14 is None
    var_15 = b'{"simple":true}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_5 + var_16
    var_18 = var_0.load_payload(var_17)
    var_19 = b'42'
    var_20 = module_1.base64_encode(var_19)
    var_21 = var_0.load_payload(var_20)
    assert var_21 == 42
    var_22 = b'[1,2,3]'
    var_23 = module_1.base64_encode(var_22)
    var_24 = var_0.load_payload(var_23)



# Parsed testcases at query #92
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = b'.'
    var_3 = b'invalid!!'
    var_4 = var_0.load_payload(var_3)
    var_5 = b'not_compressed_data'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_2 + var_6
    var_8 = var_0.load_payload(var_7)
    var_9 = b'.'
    var_10 = var_0.load_payload(var_9)
    var_11 = module_2._CompactJSON()
    var_12 = b'[1, 2, 3]'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13, serializer=var_11)



# Parsed testcases at query #93
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_1: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = b'.'
    var_12 = "Expected compressed payload to start with '.'"
    var_13 = var_0.load_payload(var_10)
    var_14 = b'invalid!base64!'
    var_15 = var_0.load_payload(var_14)
    var_16 = b'not json'



# Parsed testcases at query #94
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'x'
    var_4 = 1000
    var_5 = var_3 * var_4
    var_6 = b'invalid_base64!!!'
    var_7 = b'.'
    var_8 = b'not_compressed_data'
    var_9 = module_0.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = b''
    var_12 = b'.'



# Parsed testcases at query #95
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'{"another": "test"}'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_4 + var_6
    var_8 = var_0.load_payload(var_7)
    var_9 = b'invalid_base64!!!'
    var_10 = var_0.load_payload(var_9)
    var_11 = b'not_actually_compressed'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_9 + var_12
    var_14 = var_0.load_payload(var_13)



# Parsed testcases at query #96
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '{"test": "data"}'
    var_5 = b'.'
    var_6 = b'invalid-base64!!!'
    var_7 = b'corrupted-data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_5 + var_8
    var_10 = '{"normal": "payload"}'
    var_11 = b''



# Parsed testcases at query #97
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'data'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = "Compressed payload should start with b'.'"
    var_10 = b'!@#$%^&*()'
    var_11 = b'not_compressed_data'
    var_12 = module_0.base64_encode(var_11)
    var_13 = var_8 + var_12
    var_14 = {}
    var_15 = 'text'
    var_16 = 'hello world! @#$%'
    var_17 = {var_15: var_16}
    var_18 = 'short'
    var_19 = {var_18: var_3}



# Parsed testcases at query #98
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'{}'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_0.load_payload(var_7)
    var_9 = b'[1, 2, 3]'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{"nested": {"a": 1}}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'invalid-base64!!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'.'
    var_18 = b'not-compressed-data'
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_17 + var_19
    var_21 = var_0.load_payload(var_20)
    var_22 = b''
    var_23 = module_1.base64_encode(var_22)
    var_24 = var_0.load_payload(var_23)
    assert var_24 is None
    var_25 = b'{"count": 42}'
    var_26 = module_1.base64_encode(var_25)
    var_27 = var_0.load_payload(var_26)
    var_28 = b'{"active": true, "completed": false}'
    var_29 = module_1.base64_encode(var_28)
    var_30 = var_0.load_payload(var_29)
    var_31 = b'{"data": null}'
    var_32 = module_1.base64_encode(var_31)
    var_33 = var_0.load_payload(var_32)



# Parsed testcases at query #99
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b''
    var_8 = module_0.base64_encode(var_7)
    var_9 = None



# Parsed testcases at query #100
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5



# Parsed testcases at query #101
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{"data":"test with spaces & symbols!"}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #102
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'{"key":"value with spaces & symbols"}'
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #103
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = 'data'
    var_6 = 'a'
    var_7 = 100
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = b'.'
    var_11 = b'invalid_base64!!!'
    var_12 = b'not_compressed_data'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_10 + var_13
    var_15 = b''
    var_16 = b'.'



# Parsed testcases at query #104
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'eyJhIjogMX0='
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'!!!invalid_base64!!!'
    var_6 = b'not_compressed_data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b'{}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = 'key'
    var_12 = 'value with spaces & symbols'
    var_13 = {var_11: var_12}



# Parsed testcases at query #105
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'{"key": "value"}'
    var_5 = module_1.base64_encode(var_4)
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'invalid!@#$'
    var_9 = var_0.load_payload(var_8)
    var_10 = b'not_compressed_data'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_7 + var_11
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #106
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'null'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_0.load_payload(var_6)
    assert var_7 is None
    var_8 = b'invalid_base64!!!'
    var_9 = var_0.load_payload(var_8)
    var_10 = b'not_compressed_data'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_9 + var_11
    var_13 = var_0.load_payload(var_12)
    var_14 = module_2._CompactJSON()
    var_15 = b'{"custom":"data"}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16, serializer=var_14)
    var_18 = b'42'
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_0.load_payload(var_19)
    assert var_20 == 42
    var_21 = b'["a","b","c"]'
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_0.load_payload(var_22)



# Parsed testcases at query #107
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = b'.'
    var_13 = var_0.load_payload(var_11)
    var_14 = 'small'
    var_15 = {var_6: var_14}
    var_16 = var_0.dump_payload(var_15)
    var_17 = var_0.load_payload(var_16)
    var_18 = b'invalid_base64!!!'
    var_19 = var_0.load_payload(var_18)
    var_20 = b'corrupted_data'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_12 + var_21
    var_23 = var_0.load_payload(var_22)
    var_24 = b''
    var_25 = var_0.load_payload(var_24)



# Parsed testcases at query #108
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'{"compressed":true}'
    var_4 = b'invalid_base64!!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_2 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b''
    var_11 = module_0.base64_encode(var_10)
    var_12 = var_2 + var_11



# Parsed testcases at query #109
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'null'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    assert var_14 is None
    var_15 = b'{"number": 42, "list": [1, 2, 3], "bool": true}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)



# Parsed testcases at query #110
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'data'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = b'invalid_base64!!!'
    var_10 = b'not_compressed_data'
    var_11 = module_0.base64_encode(var_10)
    var_12 = var_8 + var_11



# Parsed testcases at query #111
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'data'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = "Expected compressed payload to start with '.'"
    var_10 = 'short'
    var_11 = {var_3: var_10}
    var_12 = {}
    var_13 = 'special'
    var_14 = '!@#$%^&*()_+-=[]{}|;\':",./<>?`~'
    var_15 = {var_13: var_14}
    var_16 = b'!invalid_base64!'
    var_17 = b'invalid_compressed_data'
    var_18 = module_0.base64_encode(var_17)
    var_19 = var_8 + var_18
    var_20 = module_1._CompactJSON()
    var_21 = 'test'
    var_22 = {var_21: var_1}



# Parsed testcases at query #112
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"key": "value", "another": "long data"}'
    var_3 = b'.'
    var_4 = b'invalid_base64!!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_4 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'{"message": "hello world!"}'
    var_11 = module_0.base64_encode(var_10)



# Parsed testcases at query #113
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = b'.'
    var_13 = "Compressed payload should start with '.'"
    var_14 = var_0.load_payload(var_11)
    var_15 = b'!!!invalid_base64!!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'invalid_zlib_data'
    var_18 = {}
    var_19 = var_0.dump_payload(var_18)
    var_20 = var_0.load_payload(var_19)
    var_21 = 'test@#$%^&*()'
    var_22 = {var_6: var_21}
    var_23 = var_0.dump_payload(var_22)
    var_24 = var_0.load_payload(var_23)



# Parsed testcases at query #114
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'{"key":"value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = b'.'
    var_5 = b'invalid!@#$%'
    var_6 = b'not_compressed_data'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b'{"empty": true}'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_4 + var_10
    var_12 = b'{"nested": {"a": 1, "b": [2, 3]}}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = b'{"x":1}'
    var_15 = module_1.base64_encode(var_14)



# Parsed testcases at query #115
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJrZXkiOiAidmFsdWUifQ=='
    var_2 = var_0.load_payload(var_1)
    var_3 = 'test'
    var_4 = 'data'
    var_5 = 100
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = b'invalid base64!!!'
    var_10 = var_0.load_payload(var_9)
    var_11 = b'not compressed data'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_8 + var_12
    var_14 = var_0.load_payload(var_13)
    var_15 = b''
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #116
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJmb28iOiAiYmFyIn0='
    var_2 = var_0.load_payload(var_1)
    var_3 = 'test'
    var_4 = 'data'
    var_5 = 100
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'.'
    var_9 = b'invalid_base64!!!'
    var_10 = var_0.load_payload(var_9)
    var_11 = b'not_valid_compressed_data'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_8 + var_12
    var_14 = var_0.load_payload(var_13)
    var_15 = b''
    var_16 = var_0.load_payload(var_15)
    var_17 = b'.'
    var_18 = var_0.load_payload(var_17)



# Parsed testcases at query #117
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'{"key":"value"}'
    var_4 = module_0.base64_encode(var_3)
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = b'not_compressed_data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_5 + var_8
    var_10 = b''
    var_11 = b'.'



# Parsed testcases at query #118
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed":true}'
    var_3 = b'.'
    var_4 = b'invalid-base64!!!'
    var_5 = b'not-compressed-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''
    var_9 = b''
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_3 + var_10



# Parsed testcases at query #119
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = module_2._CompactJSON()
    var_15 = b'{"test":123}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16, serializer=var_14)



# Parsed testcases at query #120
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJhIjoiYiJ9'
    var_2 = var_0.load_payload(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = b'.'
    var_7 = var_0.load_payload(var_1)
    var_8 = b'invalid'
    var_9 = var_0.load_payload(var_8)
    var_10 = b'!!!invalid_base64!!!'
    var_11 = var_0.load_payload(var_10)
    var_12 = b'.'
    var_13 = b'not_compressed_data'
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_12 + var_14
    var_16 = var_0.load_payload(var_15)
    var_17 = 'test'
    var_18 = 'data'
    var_19 = {var_17: var_18}
    var_20 = var_0.load_payload(var_15)



# Parsed testcases at query #121
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{"special": "test_value_123"}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = b'{"nested": {"key": "value"}, "list": [1, 2, 3]}'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_0.load_payload(var_18)



# Parsed testcases at query #122
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = module_2._CompactJSON()
    var_16 = b'{"custom":"value"}'
    var_17 = module_1.base64_encode(var_16)
    var_18 = var_0.load_payload(var_17, serializer=var_15)



# Parsed testcases at query #123
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed": true}'
    var_3 = b'.'
    var_4 = b'{"a": 1}'
    var_5 = module_0.base64_encode(var_4)
    var_6 = b'invalid_base64!!!'
    var_7 = b'not_compressed_data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_3 + var_8
    var_10 = b'{}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = b'{"nested": {"array": [1, 2, 3], "value": "test"}}'
    var_13 = module_0.base64_encode(var_12)



# Parsed testcases at query #124
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test load_payload with various scenarios.'
    var_1 = module_0.URLSafeSerializerMixin()
    var_2 = b'{"key": "value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1.load_payload(var_3)
    var_5 = b'.'
    var_6 = b'{}'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_1.load_payload(var_7)
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_13, var_14]
    var_16 = 'd'
    var_17 = 'e'
    var_18 = {var_16: var_17}
    var_19 = {var_9: var_12, var_10: var_15, var_11: var_18}
    var_20 = 'dumps'
    var_21 = str(var_19)
    var_22 = 'utf-8'
    var_23 = bytes(var_21, var_22)
    var_24 = b'!!!invalid_base64!!!'
    var_25 = var_1.load_payload(var_24)
    var_26 = b'not_compressed_data'
    var_27 = module_1.base64_encode(var_26)
    var_28 = var_5 + var_27
    var_29 = var_1.load_payload(var_28)
    var_30 = b''
    var_31 = var_1.load_payload(var_30)



# Parsed testcases at query #125
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJ0ZXN0IjogImRhdGEifQ=='
    var_2 = var_0.load_payload(var_1)
    var_3 = b'.'
    var_4 = b'{"test": "compressed_data"}'
    var_5 = b'not-valid-base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not-compressed-data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = module_2._CompactJSON()
    var_12 = b'{"key": "value"}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13, serializer=var_11)
    var_15 = b''
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)



# Parsed testcases at query #126
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed": True}'
    var_3 = b'.'
    var_4 = b'invalid_base64!!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'.'
    var_11 = 'data'
    var_12 = 'x'
    var_13 = 1000
    var_14 = var_12 * var_13
    var_15 = {var_11: var_14}
    var_16 = module_1._CompactJSON()
    var_17 = module_1._CompactJSON()
    var_18 = 'custom'
    var_19 = True
    var_20 = {var_18: var_19}



# Parsed testcases at query #127
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = b'=='
    var_4 = var_2 + var_3
    var_5 = var_0.load_payload(var_4)
    var_6 = b'.'
    var_7 = var_0.load_payload(var_2)
    var_8 = module_2._CompactJSON()
    var_9 = module_0.URLSafeSerializerMixin(serializer=var_8)
    var_10 = module_1.base64_encode(var_1)
    var_11 = var_10 + var_3
    var_12 = var_9.load_payload(var_11)
    var_13 = module_0.URLSafeSerializerMixin()
    var_14 = module_1.base64_encode(var_1)
    var_15 = var_14 + var_3
    var_16 = 'arg1'
    var_17 = 'kwarg'
    var_18 = module_0.URLSafeSerializerMixin()
    var_19 = b'invalid!@#$'
    var_20 = var_18.load_payload(var_19)
    var_21 = module_0.URLSafeSerializerMixin()
    var_22 = b'.'
    var_23 = b'not_compressed_data'
    var_24 = module_1.base64_encode(var_23)
    var_25 = var_22 + var_24
    var_26 = b'=='
    var_27 = var_25 + var_26
    var_28 = var_21.load_payload(var_27)
    var_29 = module_0.URLSafeSerializerMixin()
    var_30 = b'{}'
    var_31 = module_1.base64_encode(var_30)
    var_32 = var_31 + var_23
    var_33 = var_29.load_payload(var_32)



# Parsed testcases at query #128
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = 'Large payload should be compressed'
    var_11 = 'CustomSerializer'
    var_12 = ()
    var_13 = 'loads'
    var_14 = 'custom'
    var_15 = lambda self, x: {var_14: x}
    var_16 = {var_13: var_15}
    var_17 = type(var_11, var_12, var_16)



# Parsed testcases at query #129
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid-base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not-compressed-data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{"special":"test_123"}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = b'{"nested":{"array":[1,2,3]}}'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_0.load_payload(var_18)
    var_20 = b'{"key":null}'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_0.load_payload(var_21)



# Parsed testcases at query #130
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = b'{"key":"value"}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = b'{"compressed":true}'
    var_5 = b'.'
    var_6 = b'{"small":true}'
    var_7 = b'invalid-base64!!!'
    var_8 = b'not-compressed-data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = b'{"number":42,"list":[1,2,3],"nested":{"a":"b"}}'
    var_14 = module_1.base64_encode(var_13)
    var_15 = b'{"test":true}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = b'{"test":"compressed"}'



# Parsed testcases at query #131
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'{"key": "value" * 100}'
    var_6 = b'!!!invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_4 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'{"special": "!@#$%^&*()"}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)



# Parsed testcases at query #132
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = b'.'
    var_13 = "Compressed payload should start with '.'"
    var_14 = var_0.load_payload(var_11)
    var_15 = var_0.dump_payload(var_3)
    var_16 = var_12 + var_15
    var_17 = var_0.load_payload(var_16)
    var_18 = b'invalid-base64!!!'
    var_19 = var_0.load_payload(var_18)
    var_20 = b'not-compressed'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_12 + var_21
    var_23 = var_0.load_payload(var_22)
    var_24 = {}
    var_25 = var_0.dump_payload(var_24)
    var_26 = var_0.load_payload(var_25)
    var_27 = 'special'
    var_28 = '!@#$%^&*()_+-=[]{}|;\':",./<>?`~'
    var_29 = {var_27: var_28}
    var_30 = var_0.dump_payload(var_29)
    var_31 = var_0.load_payload(var_30)
    var_32 = 'small'
    var_33 = {var_32: var_6}
    var_34 = var_0.dump_payload(var_33)



# Parsed testcases at query #133
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'eyJmb28iOiAiYmFyIn0'
    var_2 = b'{"test": "data"}'
    var_3 = b'.'
    var_4 = b'invalid!!!'
    var_5 = b'not-valid-zlib-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''



# Parsed testcases at query #134
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'not-valid-base64!!'
    var_5 = b'not-compressed-data'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_1.base64_encode(var_8)
    var_10 = 'list'
    var_11 = 'nested'
    var_12 = 'bool'
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 'a'
    var_18 = {var_17: var_13}
    var_19 = True
    var_20 = {var_10: var_16, var_11: var_18, var_12: var_19}
    var_21 = b'{"list":[1,2,3],"nested":{"a":1},"bool":true}'
    var_22 = module_1.base64_encode(var_21)
    var_23 = 'x'
    var_24 = 1000
    var_25 = var_23 * var_24
    var_26 = 'data'
    var_27 = {var_26: var_25}
    var_28 = b'{"data":"'
    var_29 = b'"}'
    var_30 = b'.'
    var_31 = var_30 + var_3



# Parsed testcases at query #135
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b''
    var_8 = b'.'



# Parsed testcases at query #136
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'!!!invalid_base64!!!'
    var_10 = b'.'
    var_11 = b'{"key":"value"}'
    var_12 = -5
    var_13 = b'{}'
    var_14 = module_1.base64_encode(var_13)
    var_15 = 'special'
    var_16 = '!@#$%^&*()_+-=[]{}|;\':",./<>?'
    var_17 = {var_15: var_16}



# Parsed testcases at query #137
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = b'.'
    var_6 = b'{"compressed": true}'
    var_7 = b'{"normal": "data"}'
    var_8 = module_1.base64_encode(var_7)
    var_9 = b'invalid_base64!!!'
    var_10 = b'not_zlib_compressed_data'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_5 + var_11
    var_13 = b''
    var_14 = b''
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_5 + var_15
    var_17 = 'data'
    var_18 = 'x'
    var_19 = 1000
    var_20 = var_18 * var_19
    var_21 = {var_17: var_20}
    var_22 = 'small'
    var_23 = {var_17: var_22}
    var_24 = module_2._CompactJSON()
    var_25 = b'{"custom": "test"}'
    var_26 = module_1.base64_encode(var_25)



# Parsed testcases at query #138
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = b'{"long_key": "long_value" * 50}'
    var_3 = b'.'
    var_4 = b'not_valid_base64!!!'
    var_5 = var_0.load_payload(var_4)
    var_6 = b'not_compressed_data'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_3 + var_7
    var_9 = var_0.load_payload(var_8)
    var_10 = b'null'
    var_11 = module_2._CompactJSON()
    var_12 = b'[1,2,3]'



# Parsed testcases at query #139
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'{"compressed":true}'
    var_6 = b'{"small":true}'
    var_7 = b'invalid_base64!!!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'not_compressed_but_marked'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)
    var_13 = b'{}'
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_0.load_payload(var_14)
    var_16 = b'{"nested":{"array":[1,2,3],"string":"test"}}'
    var_17 = module_1.base64_encode(var_16)
    var_18 = var_0.load_payload(var_17)



# Parsed testcases at query #140
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key": "value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid-base64!!!'
    var_5 = b'not-compressed-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'null'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'{"int": 42, "list": [1, 2, 3], "bool": true}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = 'data'
    var_13 = 'x'
    var_14 = 1000
    var_15 = var_13 * var_14
    var_16 = {var_12: var_15}
    var_17 = module_1._CompactJSON()
    var_18 = b'{"short": "data"}'
    var_19 = module_0.base64_encode(var_18)



# Parsed testcases at query #141
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"key":"value","nested":{"a":1}}'
    var_3 = b'.'
    var_4 = b'not-valid-base64!!!'
    var_5 = b'not-compressed-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_1._CompactJSON()
    var_11 = b'[1,2,3]'
    var_12 = module_0.base64_encode(var_11)
    var_13 = b'{"test":"data"}'
    var_14 = module_0.base64_encode(var_13)
    var_15 = b'fake-compressed'
    var_16 = module_0.base64_encode(var_15)
    var_17 = var_3 + var_16



# Parsed testcases at query #142
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = b'.'
    var_2 = b'invalid_base64!!!'
    var_3 = b'not_actually_compressed'
    var_4 = module_0.base64_encode(var_3)
    var_5 = var_1 + var_4
    var_6 = b'{}'
    var_7 = module_0.base64_encode(var_6)
    var_8 = '{"string": "hello", "number": 42, "list": [1,2,3], "null": null}'



# Parsed testcases at query #143
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"data":"test"}'
    var_3 = b'.'
    var_4 = b'invalid_base64!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)



# Parsed testcases at query #144
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = var_0.load_payload
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = b'.'
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    assert var_14 is None
    var_15 = b'null'



# Parsed testcases at query #145
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous._json as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_1: var_6}
    var_8 = module_1._CompactJSON()
    var_9 = module_0.URLSafeSerializer(serializer=var_8)
    var_10 = 'test'
    var_11 = 123
    var_12 = {var_10: var_11}
    var_13 = b'invalid_base64!!'
    var_14 = 'data'
    var_15 = {var_10: var_14}
    var_16 = 5
    var_17 = b'corrupted'
    var_18 = b''
    var_19 = b'.'
    var_20 = b'not_compressed'
    var_21 = module_2.base64_encode(var_20)
    var_22 = var_19 + var_21



# Parsed testcases at query #146
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'{"a":1}'
    var_6 = b'invalid_base64!!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)



# Parsed testcases at query #147
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = 'small'
    var_11 = {var_10: var_4}
    var_12 = b'invalid-base64-data!'
    var_13 = b'not-real-compressed-data'
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_9 + var_14
    var_16 = {}
    var_17 = 'url_chars'
    var_18 = 'abc123_-.'
    var_19 = {var_17: var_18}



# Parsed testcases at query #148
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = 'data'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = b'.'
    var_11 = "Expected compressed payload to start with '.'"
    var_12 = b'invalid_base64!!!'
    var_13 = b'.'
    var_14 = b'invalid_compressed_data'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_13 + var_15
    var_17 = b''
    var_18 = b'.'



# Parsed testcases at query #149
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = b'!!!invalid_base64!!!'
    var_11 = b'invalid_compressed_data'
    var_12 = module_0.base64_encode(var_11)
    var_13 = var_9 + var_12
    var_14 = b'{}'
    var_15 = module_0.base64_encode(var_14)
    var_16 = 'custom'
    var_17 = 'test'
    var_18 = {var_16: var_17}



# Parsed testcases at query #150
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'{"nested": {"array": [1, 2, 3]}}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)
    var_18 = b'{"message": "hello world!"}'
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_0.load_payload(var_19)



# Parsed testcases at query #151
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = 'key'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'{"key":"'
    var_10 = b'x'
    var_11 = var_10 * var_6
    var_12 = var_9 + var_11
    var_13 = b'"}'
    var_14 = var_12 + var_13
    var_15 = b'invalid-base64!!!'
    var_16 = b'.'
    var_17 = b'not-compressed-data'
    var_18 = module_0.base64_encode(var_17)
    var_19 = var_16 + var_18
    var_20 = b''
    var_21 = b'.'



# Parsed testcases at query #152
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid!@#$'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = module_2._CompactJSON()
    var_12 = b'{"test":123}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13, serializer=var_11)
    var_15 = b'{}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)



# Parsed testcases at query #153
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'eyJ0ZXN0IjogImRhdGEifQ=='
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 100
    var_5 = var_3 * var_4
    var_6 = {var_2: var_5}
    var_7 = b'.'
    var_8 = b'!!!invalid_base64!!!'
    var_9 = b'corrupted_data'
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_7 + var_10



