####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
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
    var_9 = 'Large payload should be compressed'
    var_10 = b'!!!invalid-base64!!!'
    var_11 = b'not-compressed-data'
    var_12 = module_0.base64_encode(var_11)
    var_13 = var_8 + var_12
    var_14 = b''
    var_15 = module_1._CompactJSON()
    var_16 = 'string'
    var_17 = 'hello'
    var_18 = {var_16: var_17}
    var_19 = 'number'
    var_20 = 42
    var_21 = {var_19: var_20}
    var_22 = 'list'
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = {var_22: var_26}
    var_28 = 'nested'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = 'c'
    var_32 = {var_30: var_31}
    var_33 = {var_29: var_32}
    var_34 = {var_28: var_33}
    var_35 = 'boolean'
    var_36 = True
    var_37 = {var_35: var_36}
    var_38 = 'null'
    var_39 = None
    var_40 = {var_38: var_39}
    var_41 = 'mixed'
    var_42 = 'two'
    var_43 = 'three'
    var_44 = {var_43: var_25}
    var_45 = [var_36, var_42, var_44]
    var_46 = {var_41: var_45}
    var_47 = [var_18, var_21, var_27, var_34, var_37, var_40, var_46]



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous._json as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0._CompactJSON()
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = b'.'
    var_7 = 'data'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = 1
    var_13 = {}
    var_14 = 2
    var_15 = 3
    var_16 = [var_12, var_14, var_15]
    var_17 = 'ascii'
    var_18 = '_-'



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value" * 100}'
    var_5 = b'.'
    var_6 = b'invalid_base64!!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b'{}'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    var_15 = b'{"data": "test with spaces and symbols!@#"}'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)
    var_18 = b'{"count": 42, "price": 19.99}'
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_0.load_payload(var_19)
    var_21 = b'{"items": [1, 2, 3], "nested": {"key": "value"}}'
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_0.load_payload(var_22)



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous._json as module_2

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid!base64'
    var_5 = b'not_compressed_data'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'null'
    var_9 = module_1.base64_encode(var_8)
    var_10 = module_2._CompactJSON()
    var_11 = b'{"custom":"data"}'
    var_12 = module_1.base64_encode(var_11)



# Parsed testcases at query #6
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
    var_9 = b'{"nested":{"list":[1,2,3]}}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"special":"value with spaces & symbols!"}'
    var_12 = module_0.base64_encode(var_11)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'e30'
    var_6 = b'.'
    var_7 = 'data'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = var_0.dump_payload(var_11)
    var_13 = 'test'
    var_14 = 'number'
    var_15 = 42
    var_16 = {var_13: var_7, var_14: var_15}
    var_17 = var_0.dump_payload(var_16)
    var_18 = 'msg'
    var_19 = 'hello'
    var_20 = {var_18: var_19}
    var_21 = var_0.dump_payload(var_20)
    var_22 = 'ascii'
    var_23 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.'
    var_24 = {}
    var_25 = var_0.dump_payload(var_24)
    var_26 = 'level1'
    var_27 = 'level2'
    var_28 = 1
    var_29 = 2
    var_30 = 3
    var_31 = [var_28, var_29, var_30]
    var_32 = {var_27: var_31}
    var_33 = {var_26: var_32}
    var_34 = var_0.dump_payload(var_33)



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = b'.'
    var_2 = b'!!!invalid base64!!!'
    var_3 = b'.'
    var_4 = b'not compressed data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_3 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = 'nested'
    var_10 = 'list'
    var_11 = 'bool'
    var_12 = 'null'
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = True
    var_18 = None
    var_19 = {var_10: var_16, var_11: var_17, var_12: var_18}
    var_20 = {var_9: var_19}
    var_21 = module_1._CompactJSON()
    var_22 = b'.'



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
    var_5 = var_0.load_payload(var_4)
    var_6 = b'.'
    var_7 = b'{"compressed": true}'
    var_8 = b'invalid_base64!!!'
    var_9 = var_0.load_payload(var_8)
    var_10 = b'not_compressed_data'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_6 + var_11
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = 'data'
    var_18 = 'test with special chars: @#$%^&*()'
    var_19 = {var_17: var_18}
    var_20 = var_0.dump_payload(var_19)
    var_21 = var_0.load_payload(var_20)
    var_22 = 'x'
    var_23 = 1000
    var_24 = var_22 * var_23
    var_25 = {var_17: var_24}
    var_26 = var_0.dump_payload(var_25)
    var_27 = var_0.load_payload(var_26)
    var_28 = 'int'
    var_29 = 'float'
    var_30 = 'list'
    var_31 = 42
    var_32 = 3.14
    var_33 = 1
    var_34 = 2
    var_35 = 3
    var_36 = [var_33, var_34, var_35]
    var_37 = {var_28: var_31, var_29: var_32, var_30: var_36}
    var_38 = var_0.dump_payload(var_37)
    var_39 = var_0.load_payload(var_38)
    var_40 = 'level1'
    var_41 = 'level2'
    var_42 = 'level3'
    var_43 = 'deep'
    var_44 = {var_42: var_43}
    var_45 = {var_41: var_44}
    var_46 = {var_40: var_45}
    var_47 = var_0.dump_payload(var_46)
    var_48 = var_0.load_payload(var_47)
    var_49 = 'null_value'
    var_50 = None
    var_51 = {var_49: var_50}
    var_52 = var_0.dump_payload(var_51)
    var_53 = var_0.load_payload(var_52)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'null'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'[1,2,3]'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"a":{"b":"c"}}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_1._CompactJSON()
    var_14 = b'{"test":123}'
    var_15 = module_0.base64_encode(var_14)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.'
    var_6 = 'a'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = var_0.dump_payload(var_8)
    var_10 = 'a'
    var_11 = 10
    var_12 = var_10 * var_11
    var_13 = var_0.dump_payload(var_12)
    var_14 = len(var_9)
    var_15 = len(var_13)
    var_16 = var_0.load_payload(var_4)
    var_17 = 'short'
    var_18 = var_0.dump_payload(var_17)
    var_19 = b'.'
    var_20 = 500
    var_21 = var_6 * var_20
    var_22 = var_0.dump_payload(var_21)
    var_23 = var_0.load_payload(var_22)



# Parsed testcases at query #12
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
    var_9 = 'a'
    var_10 = 1
    var_11 = {var_9: var_10}
    var_12 = 'test'
    var_13 = 'number'
    var_14 = 42
    var_15 = {var_12: var_4, var_13: var_14}
    var_16 = 'special'
    var_17 = 'characters/?:&='
    var_18 = {var_16: var_17}
    var_19 = 'ascii'
    var_20 = '._-'
    var_21 = 'y'
    var_22 = 100
    var_23 = var_21 * var_22
    var_24 = {var_5: var_23}
    var_25 = 1
    var_26 = module_0.base64_decode(var_1)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'eyJmb28iOiAiYmFyIn0='
    var_2 = 'test'
    var_3 = 'value'
    var_4 = 100
    var_5 = var_3 * var_4
    var_6 = {var_2: var_5}
    var_7 = b'.'
    var_8 = b'not-valid-base64!!!'
    var_9 = b'not-compressed-data'
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_7 + var_10
    var_12 = b'{}'
    var_13 = module_0.base64_encode(var_12)



# Parsed testcases at query #14
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
    var_6 = module_1.base64_decode(var_4)
    var_7 = 'large'
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



# Parsed testcases at query #15
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
    var_7 = b''



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11, var_0]



# Parsed testcases at query #17
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
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.load_payload(var_11)
    var_13 = 'small'
    var_14 = {var_13: var_6}
    var_15 = var_0.dump_payload(var_14)
    var_16 = b'.'
    var_17 = var_0.load_payload(var_15)
    var_18 = b'invalid-base64!!!'
    var_19 = var_0.load_payload(var_18)
    var_20 = b'.corrupted-data'
    var_21 = var_0.load_payload(var_20)
    var_22 = b'.not-compressed'
    var_23 = var_0.load_payload(var_22)
    var_24 = module_1._CompactJSON()
    var_25 = 'custom'
    var_26 = True
    var_27 = {var_25: var_26}
    var_28 = var_0.dump_payload(var_27)
    var_29 = var_0.load_payload(var_28, serializer=var_24)
    var_30 = b''
    var_31 = var_0.load_payload(var_30)
    var_32 = b'.'
    var_33 = var_0.load_payload(var_32)



# Parsed testcases at query #18
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
    var_9 = 1
    var_10 = b'=='
    var_11 = {}
    var_12 = None
    var_13 = 2
    var_14 = 3
    var_15 = [var_9, var_13, var_14]



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.'
    var_4 = 'data'
    var_5 = 'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = {var_4: var_7}
    var_9 = b'.'
    var_10 = 'small'
    var_11 = {var_4: var_10}



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed": true}'
    var_3 = b'.'
    var_4 = b'not_valid_base64!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = 'nested'
    var_11 = 'null'
    var_12 = 'list'
    var_13 = 'bool'
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = True
    var_19 = {var_12: var_17, var_13: var_18}
    var_20 = None
    var_21 = {var_10: var_19, var_11: var_20}
    var_22 = b'{"nested": {"list": [1, 2, 3], "bool": true}, "null": null}'
    var_23 = module_0.base64_encode(var_22)



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'test payload'
    var_2 = var_0.dump_payload(var_1)
    var_3 = b'.'
    var_4 = module_1.base64_decode(var_2)
    var_5 = b'x'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = var_0.dump_payload(var_7)
    var_9 = 1
    var_10 = var_8[var_9:]
    var_11 = module_1.base64_decode(var_10)
    var_12 = b'ab'
    var_13 = var_0.dump_payload(var_12)
    var_14 = module_1.base64_decode(var_13)
    var_15 = b''
    var_16 = var_0.dump_payload(var_15)
    var_17 = module_1.base64_decode(var_16)
    var_18 = b'y'
    var_19 = 100
    var_20 = var_18 * var_19
    var_21 = var_0.dump_payload(var_20)
    var_22 = module_1.base64_decode(var_21)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key": "value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = '{"key": "'
    var_5 = 'x'
    var_6 = 100
    var_7 = var_5 * var_6
    var_8 = var_4 + var_7
    var_9 = '"}'
    var_10 = var_8 + var_9
    var_11 = b'!!!invalid_base64!!!'
    var_12 = b'not_compressed_data'
    var_13 = module_0.base64_encode(var_12)
    var_14 = var_3 + var_13
    var_15 = b'{}'
    var_16 = module_0.base64_encode(var_15)
    var_17 = b'{"test": "data"}'
    var_18 = module_0.base64_encode(var_17)
    var_19 = var_3 + var_18



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.URLSafeSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = b'.'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_2: var_8}
    var_10 = 1
    var_11 = {}
    var_12 = 2
    var_13 = 3
    var_14 = 4
    var_15 = 5
    var_16 = [var_10, var_12, var_13, var_14, var_15]
    var_17 = 'hello world'



# Parsed testcases at query #24
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



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed": true}'
    var_3 = b'.'
    var_4 = b'{"dot": true}'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_3 + var_5
    var_7 = b'invalid_base64!!!'
    var_8 = b'not_actually_compressed'
    var_9 = module_0.base64_encode(var_8)
    var_10 = var_3 + var_9
    var_11 = b'{}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = 'nested'
    var_14 = 'null_val'
    var_15 = 'list'
    var_16 = 'bool'
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = True
    var_22 = {var_15: var_20, var_16: var_21}
    var_23 = None
    var_24 = {var_13: var_22, var_14: var_23}
    var_25 = b'test_data'
    var_26 = module_0.base64_encode(var_25)
    var_27 = 'data'
    var_28 = 'x'
    var_29 = 1000
    var_30 = var_28 * var_29
    var_31 = {var_27: var_30}



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'hello'
    var_1 = b'.'
    var_2 = 'a'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = 'test'
    var_6 = ''
    var_7 = 'Hello World! '
    var_8 = 50
    var_9 = var_7 * var_8
    var_10 = 'Long repetitive payload should be compressed'
    var_11 = 12345
    var_12 = 1
    var_13 = 'x'



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test_data'
    var_2 = var_0.dump_payload(var_1)
    var_3 = b'.'
    var_4 = 'a'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = var_0.dump_payload(var_6)
    var_8 = 1
    var_9 = var_7[var_8:]
    var_10 = module_1.base64_decode(var_9)
    var_11 = 'short'
    var_12 = var_0.dump_payload(var_11)
    var_13 = module_1.base64_decode(var_12)
    var_14 = ''
    var_15 = var_0.dump_payload(var_14)
    var_16 = module_1.base64_decode(var_15)
    assert var_16 == b'""'
    var_17 = 123
    var_18 = var_0.dump_payload(var_17)



# Parsed testcases at query #28
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
    var_9 = "Compressed payload should start with '.'"
    var_10 = b'invalid_base64!!!'
    var_11 = b'not_actually_compressed'
    var_12 = module_0.base64_encode(var_11)
    var_13 = var_8 + var_12
    var_14 = b''
    var_15 = b'.'
    var_16 = 'small'
    var_17 = True
    var_18 = {var_16: var_17}



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed":true}'
    var_3 = b'.'
    var_4 = b'invalid_base64!!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = module_1._CompactJSON()
    var_11 = b'{"custom":true}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = b'{"a":1}'
    var_14 = module_0.base64_encode(var_13)



# Parsed testcases at query #30
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
    var_9 = 1
    var_10 = 'a'
    var_11 = {var_10: var_9}
    var_12 = 'test'
    var_13 = 'data'
    var_14 = {var_12: var_13}
    var_15 = b'.'
    var_16 = 'test'
    var_17 = {var_16: var_13}
    var_18 = 'message'
    var_19 = 'count'
    var_20 = 'hello world'
    var_21 = 42
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = 1
    var_24 = module_0.base64_decode(var_13)



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"compressed": true}'
    var_3 = b'.'
    var_4 = b'invalid_base64!!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'{"special": "test with spaces & symbols!"}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = b'{"number": 42}'
    var_13 = module_0.base64_encode(var_12)
    var_14 = b'["item1", "item2"]'
    var_15 = module_0.base64_encode(var_14)
    var_16 = b'{"nested": {"inner": "value"}}'
    var_17 = module_0.base64_encode(var_16)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'a'
    var_8 = 500
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = var_0.dump_payload(var_10)
    var_12 = var_0.load_payload(var_4)
    var_13 = var_0.load_payload(var_11)
    var_14 = {}
    var_15 = var_0.dump_payload(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = 'list'
    var_18 = 'nested'
    var_19 = 'number'
    var_20 = 'bool'
    var_21 = 1
    var_22 = 2
    var_23 = 3
    var_24 = [var_21, var_22, var_23]
    var_25 = 'key'
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = 42
    var_29 = True
    var_30 = {var_17: var_24, var_18: var_27, var_19: var_28, var_20: var_29}
    var_31 = var_0.dump_payload(var_30)
    var_32 = var_0.load_payload(var_31)
    var_33 = 'x'
    var_34 = 1000
    var_35 = var_33 * var_34
    var_36 = {var_6: var_35}
    var_37 = var_0.dump_payload(var_36)
    var_38 = 'short'
    var_39 = {var_38: var_6}
    var_40 = var_0.dump_payload(var_39)



# Parsed testcases at query #33
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
    var_12 = b'.'
    var_13 = 1
    var_14 = var_4[var_13:]
    var_15 = var_14 if var_2 else var_4
    var_16 = var_0.load_payload(var_4)
    var_17 = var_0.load_payload(var_11)



# Parsed testcases at query #34
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
    var_12 = b'_-.'
    var_13 = 48
    var_14 = 58
    var_15 = range(var_13, var_14)
    var_16 = 'small'
    var_17 = {var_16: var_6}
    var_18 = var_0.dump_payload(var_17)
    var_19 = var_0.dump_payload(var_10)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = 100
    var_3 = var_0 * var_2
    var_4 = b'.'
    var_5 = b'invalid-base64!!!'
    var_6 = b'not-compressed-data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b'{}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'{"special":"!@#$%^&*()"}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = '{"unicode":"héllo wörld"}'
    var_14 = b'{"number":42,"float":3.14,"boolean":true}'
    var_15 = module_0.base64_encode(var_14)
    var_16 = b'{"nested":{"inner":"value"},"array":[1,2,3]}'
    var_17 = module_0.base64_encode(var_16)
    var_18 = b'{"a":1}'
    var_19 = module_0.base64_encode(var_18)



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'eyJmb28iOiAiYmFyIn0='
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = b'.'
    var_7 = b'!!!invalid_base64!!!'
    var_8 = b'not_compressed_data'
    var_9 = module_0.base64_encode(var_8)
    var_10 = var_6 + var_9
    var_11 = b'{}'
    var_12 = module_0.base64_encode(var_11)
    var_13 = module_1._CompactJSON()
    var_14 = b'eyJhIjogMX0='



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 1
    var_7 = var_4[var_6:]
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = var_0.dump_payload(var_10)
    var_12 = 'small'
    var_13 = var_0.dump_payload(var_12)
    var_14 = ''
    var_15 = var_0.dump_payload(var_14)
    var_16 = 'key'
    var_17 = 'number'
    var_18 = 'value'
    var_19 = 42
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = var_0.dump_payload(var_20)
    var_22 = var_21[var_6:]
    var_23 = None
    var_24 = var_0.dump_payload(var_23)
    var_25 = 2
    var_26 = 3
    var_27 = [var_6, var_25, var_26]
    var_28 = var_0.dump_payload(var_27)



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'Test dump_payload method with various scenarios.'
    var_1 = module_0.URLSafeSerializerMixin()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = b'.'
    var_7 = 'data'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = var_1.dump_payload(var_11)
    var_13 = 'ascii'
    var_14 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-./'
    var_15 = set(var_14)
    var_16 = 'test'
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_16: var_20}
    var_22 = var_1.dump_payload(var_21)
    var_23 = var_1.load_payload(var_22)
    var_24 = {}
    var_25 = var_1.dump_payload(var_24)
    var_26 = 'outer'
    var_27 = 'list'
    var_28 = 'inner'
    var_29 = {var_28: var_3}
    var_30 = 'nested'
    var_31 = True
    var_32 = {var_30: var_31}
    var_33 = [var_17, var_18, var_32]
    var_34 = {var_26: var_29, var_27: var_33}
    var_35 = var_1.dump_payload(var_34)
    var_36 = var_1.load_payload(var_35)
    var_37 = 'just a string'
    var_38 = var_1.dump_payload(var_37)
    var_39 = var_1.load_payload(var_38)
    var_40 = 42
    var_41 = var_1.dump_payload(var_40)
    var_42 = var_1.load_payload(var_41)
    var_43 = True
    var_44 = var_1.dump_payload(var_43)
    var_45 = var_1.load_payload(var_44)
    var_46 = None
    var_47 = var_1.dump_payload(var_46)
    var_48 = var_1.load_payload(var_47)
    assert var_48 is None
    var_49 = 'two'
    var_50 = 'four'
    var_51 = 4
    var_52 = {var_50: var_51}
    var_53 = [var_31, var_49, var_19, var_52]
    var_54 = var_1.dump_payload(var_53)
    var_55 = var_1.load_payload(var_54)
    var_56 = 'small'
    var_57 = {var_56: var_7}
    var_58 = var_1.dump_payload(var_57)
    var_59 = 'large'
    var_60 = 5000
    var_61 = var_8 * var_60
    var_62 = {var_59: var_61}
    var_63 = var_1.dump_payload(var_62)



# Parsed testcases at query #39
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
    var_12 = var_0.load_payload(var_4)
    var_13 = var_0.load_payload(var_11)
    var_14 = {}
    var_15 = var_0.dump_payload(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = 'int'
    var_18 = 'float'
    var_19 = 42
    var_20 = 3.14
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = var_0.dump_payload(var_21)
    var_23 = var_0.load_payload(var_22)
    var_24 = 'items'
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_24: var_28}
    var_30 = var_0.dump_payload(var_29)
    var_31 = var_0.load_payload(var_30)
    var_32 = 'outer'
    var_33 = 'inner'
    var_34 = {var_33: var_2}
    var_35 = {var_32: var_34}
    var_36 = var_0.dump_payload(var_35)
    var_37 = var_0.load_payload(var_36)
    var_38 = 'x'
    var_39 = 'y'
    var_40 = {var_38: var_39}
    var_41 = var_0.dump_payload(var_40)
    var_42 = b'^[A-Za-z0-9_\\-\\.]+$'



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'ey'
    var_6 = {var_1: var_2}
    var_7 = var_0.dump_payload(var_6)
    var_8 = b'.'
    var_9 = {var_1: var_2}
    var_10 = var_0.dump_payload(var_9)
    var_11 = module_0.URLSafeSerializer()
    var_12 = 'test'
    var_13 = 'data'
    var_14 = {var_12: var_13}
    var_15 = len(var_10)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'Test dump_payload method with various scenarios.'
    var_1 = module_0.URLSafeSerializer()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = b'.'
    var_6 = 'data'
    var_7 = 'x'
    var_8 = 1000
    var_9 = var_7 * var_8
    var_10 = {var_6: var_9}
    var_11 = {var_6: var_7}
    var_12 = 1
    var_13 = {}
    var_14 = 'special'
    var_15 = 'test_value_with_underscores_and-dashes'
    var_16 = {var_14: var_15}
    var_17 = b'+'
    var_18 = b'/'
    var_19 = b'='
    var_20 = [var_17, var_18, var_19]
    var_21 = 'level1'
    var_22 = 'level2'
    var_23 = 2
    var_24 = 3
    var_25 = [var_12, var_23, var_24]
    var_26 = {var_22: var_25}
    var_27 = {var_21: var_26}
    var_28 = 'a'
    var_29 = 'b'
    var_30 = {var_28: var_29}
    var_31 = 'y'
    var_32 = {var_7: var_31}



# Parsed testcases at query #2
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
    var_13 = 'nested'
    var_14 = 123
    var_15 = 'a'
    var_16 = 1
    var_17 = {var_15: var_16}
    var_18 = {var_12: var_14, var_13: var_17}
    var_19 = var_0.dump_payload(var_18)
    var_20 = var_0.load_payload(var_19)
    var_21 = 10000
    var_22 = var_15 * var_21
    var_23 = {var_6: var_22}
    var_24 = var_0.dump_payload(var_23)
    var_25 = {}
    var_26 = var_0.dump_payload(var_25)
    var_27 = 'value with spaces and spéciäl chärs'
    var_28 = {var_1: var_27}
    var_29 = var_0.dump_payload(var_28)
    var_30 = var_0.load_payload(var_29)
    var_31 = {var_15: var_16}
    var_32 = var_0.dump_payload(var_31)
    var_33 = 'ascii'
    var_34 = '_-.'



# Parsed testcases at query #3
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
    var_12 = len(var_11)
    var_13 = 'short'
    var_14 = {var_6: var_13}
    var_15 = var_0.dump_payload(var_14)
    var_16 = len(var_15)
    var_17 = module_0.URLSafeSerializerMixin()
    var_18 = var_0.dump_payload(var_3)
    var_19 = var_17.load_payload(var_18)
    var_20 = var_0.dump_payload(var_10)
    var_21 = var_17.load_payload(var_20)
    var_22 = {}
    var_23 = var_0.dump_payload(var_22)
    var_24 = var_17.load_payload(var_23)
    var_25 = 'int'
    var_26 = 'float'
    var_27 = 'list'
    var_28 = 'bool'
    var_29 = 'none'
    var_30 = 42
    var_31 = 3.14
    var_32 = 1
    var_33 = 2
    var_34 = 3
    var_35 = [var_32, var_33, var_34]
    var_36 = True
    var_37 = None
    var_38 = {var_25: var_30, var_26: var_31, var_27: var_35, var_28: var_36, var_29: var_37}
    var_39 = var_0.dump_payload(var_38)
    var_40 = var_17.load_payload(var_39)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)
    var_6 = var_0.load_payload(var_4)
    var_7 = 'data'
    var_8 = 'x'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = var_0.dump_payload(var_11)
    var_13 = b'.'
    var_14 = var_0.load_payload(var_12)
    var_15 = b'invalid_base64!!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = -5
    var_18 = b'test'
    var_19 = b''
    var_20 = var_0.load_payload(var_19)
    var_21 = b'.'
    var_22 = var_0.load_payload(var_21)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'not-valid-base64!!!'
    var_4 = b'not-zlib-data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b''
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



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'not-valid-base64!!!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'corrupted-data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = 'custom'
    var_15 = 'test'
    var_16 = {var_14: var_15}
    var_17 = b'{"custom":"test"}'
    var_18 = module_1.base64_encode(var_17)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid@@base64@@'
    var_4 = b'not_actually_compressed'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b''
    var_8 = module_1._CompactJSON()
    var_9 = b'{"custom":"data"}'
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
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
    var_14 = b'{"special": "test_with_underscores_and-dashes"}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #9
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
    var_6 = b'test'
    var_7 = b'corrupted'
    var_8 = b''
    var_9 = module_1.base64_encode(var_8)
    var_10 = b'{"count": 42}'
    var_11 = module_1.base64_encode(var_10)
    var_12 = b'{"data": [1, 2, {"nested": "value"}]}'
    var_13 = module_1.base64_encode(var_12)



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
    assert var_6 == b'{"key":"value"}'
    var_7 = 'data'
    var_8 = 'a'
    var_9 = 1000
    var_10 = var_8 * var_9
    var_11 = {var_7: var_10}
    var_12 = var_0.dump_payload(var_11)
    var_13 = 1
    var_14 = var_12[var_13:]
    var_15 = module_1.base64_decode(var_14)
    var_16 = 'test'
    var_17 = 'nested'
    var_18 = 2
    var_19 = 3
    var_20 = [var_13, var_18, var_19]
    var_21 = 'b'
    var_22 = {var_8: var_21}
    var_23 = {var_16: var_20, var_17: var_22}
    var_24 = var_0.dump_payload(var_23)
    var_25 = var_0.load_payload(var_24)
    var_26 = {}
    var_27 = var_0.dump_payload(var_26)
    var_28 = 'x'
    var_29 = {var_28: var_13}
    var_30 = var_0.dump_payload(var_29)
    var_31 = module_1.base64_decode(var_30)
    assert var_31 == b'{"x":1}'



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = b'eyJrZXkiOiAidmFsdWUifQ=='
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 100
    var_5 = var_3 * var_4
    var_6 = {var_2: var_5}
    var_7 = b'.'
    var_8 = b'invalid_base64!!!'
    var_9 = b'invalid_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_7 + var_10
    var_12 = b''
    var_13 = b'not valid json'
    var_14 = module_1.base64_encode(var_13)



# Parsed testcases at query #12
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
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'{"key": null}'
    var_10 = module_0.base64_encode(var_9)
    var_11 = b'[1, 2, 3]'
    var_12 = module_0.base64_encode(var_11)
    var_13 = b'{"nested": {"inner": "value"}}'
    var_14 = module_0.base64_encode(var_13)



# Parsed testcases at query #13
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
    var_12 = var_0.load_payload(var_11)
    var_13 = b'.'
    var_14 = b'invalid_base64!!!'
    var_15 = var_0.load_payload(var_14)
    var_16 = -5
    var_17 = b'{"corrupted": true}'
    var_18 = b''
    var_19 = var_0.load_payload(var_18)
    var_20 = b'.'
    var_21 = var_0.load_payload(var_20)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = b'.'
    var_2 = b'invalid_base64!!!'
    var_3 = b'not_compressed'
    var_4 = module_0.base64_encode(var_3)
    var_5 = var_1 + var_4
    var_6 = b'{}'
    var_7 = module_0.base64_encode(var_6)



# Parsed testcases at query #15
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
    var_8 = b'this is not valid base64!!!'
    var_9 = var_0.load_payload(var_8)
    var_10 = b'not valid zlib data'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_7 + var_11
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = 'special'
    var_18 = '!@#$%^&*()'
    var_19 = {var_17: var_18}
    var_20 = b'{"special": "!@#$%^&*()"}'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_0.load_payload(var_21)
    var_23 = 42
    var_24 = {var_2: var_23}
    var_25 = b'{"value": 42}'
    var_26 = module_1.base64_encode(var_25)
    var_27 = var_0.load_payload(var_26)
    var_28 = 'outer'
    var_29 = 'inner'
    var_30 = 1
    var_31 = 2
    var_32 = 3
    var_33 = [var_30, var_31, var_32]
    var_34 = {var_29: var_33}
    var_35 = {var_28: var_34}
    var_36 = b'{"outer": {"inner": [1, 2, 3]}}'
    var_37 = module_1.base64_encode(var_36)
    var_38 = var_0.load_payload(var_37)



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJrZXkiOiAidmFsdWUifQ=='
    var_2 = var_0.load_payload(var_1)
    var_3 = 'key'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = b'{"key": "'
    var_9 = b'x'
    var_10 = var_9 * var_5
    var_11 = var_8 + var_10
    var_12 = b'"}'
    var_13 = var_11 + var_12
    var_14 = b'.'
    var_15 = b'invalid_base64!!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'not_compressed_data'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_14 + var_18
    var_20 = var_0.load_payload(var_19)
    var_21 = b''
    var_22 = var_0.load_payload(var_21)
    var_23 = b'.'
    var_24 = var_0.load_payload(var_23)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key": "value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid-base64!!!'
    var_5 = b'not-compressed-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'{"key": "value with spaces & symbols!"}'
    var_11 = module_0.base64_encode(var_10)



# Parsed testcases at query #18
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
    var_10 = b'invalid_base64!!!'
    var_11 = b'not_compressed_data'
    var_12 = module_0.base64_encode(var_11)
    var_13 = var_9 + var_12
    var_14 = b''
    var_15 = b'.'



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

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
    var_21 = module_1._CompactJSON()
    var_22 = b'{"test":"custom"}'
    var_23 = module_0.base64_encode(var_22)



# Parsed testcases at query #20
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
    var_10 = b'invalid-base64!!!'
    var_11 = b'.'
    var_12 = b'not-compressed-data'
    var_13 = module_0.base64_encode(var_12)
    var_14 = var_11 + var_13
    var_15 = b''
    var_16 = b'.'
    var_17 = 'test'
    var_18 = {var_17: var_13}
    var_19 = 1



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializer()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = b'{"compressed":true}'
    var_4 = b'.'
    var_5 = b'{"special":"!@#$%^&*()"}'
    var_6 = module_1.base64_encode(var_5)
    var_7 = b'{}'
    var_8 = module_1.base64_encode(var_7)
    var_9 = b'{"nested":{"key":"value"}}'
    var_10 = module_1.base64_encode(var_9)
    var_11 = b'[1,2,3]'
    var_12 = module_1.base64_encode(var_11)
    var_13 = b'invalid_base64!!!'
    var_14 = b'not_compressed_data'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_4 + var_15
    var_17 = b''
    var_18 = b'{"key":null}'
    var_19 = module_1.base64_encode(var_18)
    var_20 = b'{"flag":true,"other":false}'
    var_21 = module_1.base64_encode(var_20)
    var_22 = b'{"int":42,"float":3.14}'
    var_23 = module_1.base64_encode(var_22)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'invalid_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = b'.'



# Parsed testcases at query #23
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
    var_15 = b'not-valid-base64!!!'
    var_16 = var_0.load_payload(var_15)
    var_17 = b'corrupt-compressed-data'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_12 + var_18
    var_20 = var_0.load_payload(var_19)
    var_21 = {}
    var_22 = var_0.dump_payload(var_21)
    var_23 = var_0.load_payload(var_22)
    var_24 = 'special'
    var_25 = '!@#$%^&*()_+-=[]{}|;\':",./<>?`~'
    var_26 = {var_24: var_25}
    var_27 = var_0.dump_payload(var_26)
    var_28 = var_0.load_payload(var_27)



# Parsed testcases at query #24
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



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid-base64!!!'
    var_5 = b'not-compressed-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'.'



# Parsed testcases at query #26
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
    var_7 = b'corrupted-compressed-data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b'{}'
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{"a":{"b":"c"}}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = b'[1,2,3]'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_0.load_payload(var_18)
    var_20 = b'{"test":"data"}'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_6 + var_21
    var_23 = var_0.load_payload(var_22)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 100
    var_5 = var_3 * var_4
    var_6 = {var_2: var_5}
    var_7 = b'.'
    var_8 = b'invalid!!!'
    var_9 = b'not compressed data'
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_7 + var_10



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.encoding as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'invalid_base64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'{}'
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'.'
    var_10 = module_1._CompactJSON()
    var_11 = b'{"custom": true}'
    var_12 = module_0.base64_encode(var_11)



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = b'invalid-base64!!!'
    var_5 = b'not-compressed-data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = b'{"special":"!@#$%^&*()"}'
    var_11 = module_0.base64_encode(var_10)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Test load_payload method with various scenarios.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'utf-8'
    var_5 = 'test'
    var_6 = 'data'
    var_7 = 100
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = b'.'
    var_11 = 'small'
    var_12 = {var_11: var_6}
    var_13 = b'!!!invalid!!!'
    var_14 = b'not_compressed_data'
    var_15 = module_0.base64_encode(var_14)
    var_16 = var_10 + var_15
    var_17 = None
    var_18 = 'string'
    var_19 = 'number'
    var_20 = 'list'
    var_21 = 'nested'
    var_22 = 'hello'
    var_23 = 42
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = 'a'
    var_29 = 'b'
    var_30 = {var_28: var_29}
    var_31 = {var_18: var_22, var_19: var_23, var_20: var_27, var_21: var_30}
    var_32 = '{"custom": "serializer"}'
    var_33 = staticmethod(lambda s: json_module.loads(s))



# Parsed testcases at query #31
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
    var_9 = var_5 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b''
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    assert var_13 is None
    var_14 = b'.'
    var_15 = var_0.load_payload(var_14)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'eyJhIjogMX0='
    var_1 = 'b'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'invalid_base64!!!'
    var_6 = b'not_compressed_data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b''



# Parsed testcases at query #33
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
    var_12 = b'.'
    var_13 = var_0.load_payload(var_11)
    var_14 = 'small'
    var_15 = {var_14: var_6}
    var_16 = var_0.dump_payload(var_15)
    var_17 = var_0.load_payload(var_16)
    var_18 = module_1._CompactJSON()
    var_19 = 'test'
    var_20 = 123
    var_21 = {var_19: var_20}
    var_22 = var_0.dump_payload(var_21)
    var_23 = var_0.load_payload(var_22, serializer=var_18)
    var_24 = b'invalid_base64!!!'
    var_25 = var_0.load_payload(var_24)
    var_26 = b'.'
    var_27 = b'not_compressed_data'
    var_28 = module_2.base64_encode(var_27)
    var_29 = var_26 + var_28
    var_30 = var_0.load_payload(var_29)
    var_31 = b''
    var_32 = var_0.load_payload(var_31)



# Parsed testcases at query #34
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
    var_18 = b'!!!invalid_base64!!!'
    var_19 = var_0.load_payload(var_18)
    var_20 = b'invalid_base64'
    var_21 = var_12 + var_20
    var_22 = var_0.load_payload(var_21)
    var_23 = b'invalid_json'
    var_24 = b''
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_0.load_payload(var_25)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'data'
    var_6 = 'x'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = {var_5: var_8}
    var_10 = b'.'
    var_11 = "Compressed payload should start with '.'"
    var_12 = 'short'
    var_13 = {var_12: var_5}
    var_14 = b'not-valid-base64!!!'
    var_15 = b'not-compressed-data'
    var_16 = module_0.base64_encode(var_15)
    var_17 = var_10 + var_16
    var_18 = {}
    var_19 = 42
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = 'test'
    var_24 = [var_20, var_21, var_22, var_23]



