####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'this is a longer string that should trigger compression in most zlib scenarios'
    var_3 = b'.'
    var_4 = 'utf-8'
    var_5 = b'!!!NotBase64!!!'
    var_6 = b'not_compressed_data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_3 + var_7
    var_9 = b''



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'long_string_to_ensure_compression_benefit_long_string_to_ensure_compression_benefit'
    var_3 = b'.'
    var_4 = 'utf-8'
    var_5 = b'!!!'
    var_6 = b'not_zlib_data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_3 + var_7
    var_9 = b'args'
    var_10 = module_0.base64_encode(var_9)
    var_11 = 'val'



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"data": "'
    var_3 = b'a'
    var_4 = 100
    var_5 = var_3 * var_4
    var_6 = var_2 + var_5
    var_7 = b'"}'
    var_8 = var_6 + var_7
    var_9 = b'.'
    var_10 = 'utf-8'
    var_11 = b'!!!'
    var_12 = b'not_compressed_data'
    var_13 = module_0.base64_encode(var_12)
    var_14 = var_9 + var_13
    var_15 = b'\xff\xfe\xfd'
    var_16 = module_0.base64_encode(var_15)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'This is a longer string that should trigger compression logic'
    var_3 = b'.'
    var_4 = 'utf-8'
    var_5 = b'!!!'
    var_6 = b'not compressed data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_3 + var_7
    var_9 = b''
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"a":1}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"large_key_to_ensure_compression_is_beneficial": "some_value"}'
    var_3 = 10
    var_4 = var_2 * var_3
    var_5 = b'.'
    var_6 = 'utf-8'
    var_7 = b'!!!'
    var_8 = b'not_compressed_data'
    var_9 = module_0.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = b''



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'!!!NotBase64!!!'
    var_4 = b'not compressed data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = 'extra_arg'
    var_8 = 'test_val'



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"a": "long_string_to_ensure_compression_is_beneficial"}'
    var_3 = b'.'
    var_4 = b'!!!NotBase64!!!'
    var_5 = b'not_actually_zlib_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = b'{"key": "value", "number": 123}'
    var_6 = module_0.base64_encode(var_5)
    var_7 = b'.'
    var_8 = b'!!!NotBase64!!!'
    var_9 = b'not compressed data'
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_7 + var_10
    var_12 = b'{"broken": '
    var_13 = module_0.base64_encode(var_12)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.load_payload
    var_5 = b'{"a":1}'
    var_6 = module_1.base64_encode(var_5)
    var_7 = b'.'
    var_8 = b'!!!'
    var_9 = b'not_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_7 + var_10
    var_12 = b'\x00\x01\x02'
    var_13 = module_1.base64_encode(var_12)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'this is a much longer string that should definitely be compressed by zlib'
    var_3 = b'.'
    var_4 = 'utf-8'
    var_5 = b'!!!not_base64!!!'
    var_6 = b'not_compressed_data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_3 + var_7
    var_9 = b''
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'this is a much longer string that should benefit from compression'
    var_3 = var_2
    var_4 = b'.'
    var_5 = b'!!!'
    var_6 = b'not compressed data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_4 + var_7
    var_9 = b'just some bytes'
    var_10 = module_0.base64_encode(var_9)
    var_11 = var_4 + var_10
    var_12 = b''



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'"hello"'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = 50
    var_4 = var_2 * var_3
    var_5 = b'.'
    var_6 = 'utf-8'
    var_7 = b'!!!not_base64!!!'
    var_8 = b'not_compressed_data'
    var_9 = module_0.base64_encode(var_8)
    var_10 = var_5 + var_9
    var_11 = b''
    var_12 = 'success'
    var_13 = b'data'
    var_14 = module_0.base64_encode(var_13)
    var_15 = 'val'
    var_16 = 'test'



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"a":1}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"large_key": "large_value_to_ensure_compression_is_beneficial"}'
    var_3 = b'.'
    var_4 = b'!!!not_base64!!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''
    var_9 = module_0.base64_encode(var_8)



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'hello world long string'
    var_3 = b'.'
    var_4 = b'!!!NotBase64!!!'
    var_5 = b'not compressed data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'this is a string that should be compressed by zlib'
    var_3 = b'.'
    var_4 = 'utf-8'
    var_5 = b'!!!not_base64!!!'
    var_6 = b'not_zlib_data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_3 + var_7
    var_9 = b'a'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"a":1}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"key": "value" * 50}'
    var_3 = b'.'
    var_4 = b'!!!not_base64!!!'
    var_5 = b'not_compressed_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"a":1}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"data": "this is a very long string that should definitely trigger zlib compression for testing purposes"}'
    var_3 = b'.'
    var_4 = 'utf-8'
    var_5 = b'!!!not_base64!!!'
    var_6 = b'not_compressed_data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_3 + var_7
    var_9 = b''



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"long_key": "some long value to ensure compression works"}'
    var_3 = b'.'
    var_4 = b'!!!NotBase64!!!'
    var_5 = b'not compressed data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b'{"test": 1}'
    var_9 = module_0.base64_encode(var_8)
    var_10 = 'extra_arg'
    var_11 = 'test_context'



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'!!!not_base64!!!'
    var_4 = b'not_actually_zlib_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b'.!!!'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'utf-8'
    var_2 = b'.'
    var_3 = b'!!!NotBase64!!!'
    var_4 = b'not_compressed_data'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = 'decoded'
    var_8 = 'extra_arg'
    var_9 = 'extra_kwarg'



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'"hello"'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = 50
    var_4 = var_2 * var_3
    var_5 = b'.'
    var_6 = b'!!!not_base64!!!'
    var_7 = b'not_compressed_data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_5 + var_8
    var_10 = b'{"data": 1}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = 'extra_arg'
    var_13 = 'extra_kwarg'



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"a": 1}'
    var_3 = b'.'
    var_4 = b'!!!NotBase64!!!'
    var_5 = b'not compressed data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''
    var_9 = module_0.base64_encode(var_8)



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'"test"'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'"large_payload_data_to_ensure_compression_logic_triggers"'
    var_3 = b'.'
    var_4 = b'!!!not_base64!!!'
    var_5 = b'not_actually_zlib_data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = 'success'
    var_9 = b'"data"'
    var_10 = module_0.base64_encode(var_9)
    var_11 = 'extra_arg'
    var_12 = 'extra_kwarg'



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"a":1}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"large_key":"'
    var_3 = b'x'
    var_4 = 100
    var_5 = var_3 * var_4
    var_6 = var_2 + var_5
    var_7 = b'"}'
    var_8 = var_6 + var_7
    var_9 = b'.'
    var_10 = module_0.base64_encode(var_8)
    var_11 = b'!!!'
    var_12 = b'.'
    var_13 = b'not_actually_zlib_data'
    var_14 = module_0.base64_encode(var_13)
    var_15 = var_12 + var_14
    var_16 = b'{"key":"val"}'
    var_17 = module_0.base64_encode(var_16)
    var_18 = 'arg1'
    var_19 = 'extra_val'



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'a'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = b'.'
    var_6 = b'!!!not_base64!!!'
    var_7 = b'not_compressed_data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_5 + var_8
    var_10 = b'{"data": 1}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = 'extra_arg'
    var_13 = 'extra_kwarg'



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'a'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = b'.'
    var_6 = b'!!!'
    var_7 = b'not_compressed_data'
    var_8 = module_0.base64_encode(var_7)
    var_9 = var_5 + var_8
    var_10 = b'{"key": "val"}'
    var_11 = module_0.base64_encode(var_10)
    var_12 = 'extra_arg'
    var_13 = 'extra_kwarg'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'abc'
    var_1 = b'.'
    var_2 = 'A'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = 1
    var_6 = b'abcde'
    var_7 = b'A'
    var_8 = 100
    var_9 = var_7 * var_8



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'long_string_to_ensure_compression_is_likely'
    var_3 = b'.'
    var_4 = 'utf-8'
    var_5 = b'!!!not_base64!!!'
    var_6 = b'not_compressed_data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_3 + var_7
    var_9 = b''
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'this is a much longer string to ensure compression logic is triggered'
    var_3 = b'.'
    var_4 = b'!!!NotBase64!!!'
    var_5 = b'not compressed data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_3 + var_6
    var_8 = b''
    var_9 = module_0.base64_encode(var_8)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'not-base64-!!!'
    var_4 = b'not-compressed'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = b''
    var_8 = module_0.base64_encode(var_7)



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'short'
    var_2 = 'key'
    var_3 = 'val'
    var_4 = {var_2: var_3}
    var_5 = var_0.dump_payload(var_4)
    var_6 = {var_2: var_3}
    var_7 = module_1.base64_encode(var_1)
    var_8 = b'.'
    var_9 = b'a'
    var_10 = 1000
    var_11 = var_9 * var_10
    var_12 = 'key'
    var_13 = 'large_data'
    var_14 = {var_12: var_13}
    var_15 = var_0.dump_payload(var_14)
    var_16 = b'.'
    var_17 = var_16 + var_8
    var_18 = b'abcde'
    var_19 = 'boundary'
    var_20 = var_0.dump_payload(var_19)
    var_21 = b'.'
    var_22 = var_21 + var_13
    var_23 = module_1.base64_encode(var_18)
    var_24 = 'data'
    var_25 = 'error'
    var_26 = {var_24: var_25}
    var_27 = var_0.dump_payload(var_26)
    var_28 = str(var_24)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = b'abc'
    var_1 = b'.'
    var_2 = b'a'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = 1
    var_6 = b'some random data that might or might not compress'
    var_7 = 1



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = b'abc'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'a'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = 'a'
    var_9 = 'long_string_to_trigger_compression'
    var_10 = {var_8: var_9}
    var_11 = b'.'
    var_12 = 1
    var_13 = b'some_data'
    var_14 = b'.'



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'abc'
    var_1 = 'abc'
    var_2 = b'abc'
    var_3 = module_0.base64_encode(var_2)
    var_4 = 'a'
    var_5 = 100
    var_6 = var_4 * var_5
    var_7 = 'utf-8'
    var_8 = b'.'
    var_9 = 'test'
    var_10 = b'test'
    var_11 = module_0.base64_encode(var_10)
    var_12 = ''
    var_13 = b''
    var_14 = module_0.base64_encode(var_13)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"a":1}'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = b'{"x":1}'
    var_6 = b'a'
    var_7 = 1000
    var_8 = var_6 * var_7
    var_9 = 'data'
    var_10 = 'a'
    var_11 = 1000
    var_12 = var_10 * var_11
    var_13 = {var_9: var_12}
    var_14 = b'.'
    var_15 = 'a'
    var_16 = 1
    var_17 = {var_15: var_16}
    var_18 = str(var_15)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = b'abc'
    var_1 = 'key'
    var_2 = 'val'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'a'
    var_6 = 1000
    var_7 = var_5 * var_6
    var_8 = 'key'
    var_9 = 'large'
    var_10 = {var_8: var_9}
    var_11 = b'.'
    var_12 = 1
    var_13 = b'xyz'
    var_14 = 'key'
    var_15 = 'uncompressible'
    var_16 = {var_14: var_15}
    var_17 = b'.'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Tests the dump_payload method of URLSafeSerializerMixin for both \n    compressed and uncompressed scenarios.\n    '
    var_1 = b'short'
    var_2 = 'key'
    var_3 = 'val'
    var_4 = {var_2: var_3}
    var_5 = b'.'
    var_6 = b'a'
    var_7 = 100
    var_8 = var_6 * var_7
    var_9 = 'key'
    var_10 = 'large_repetition_data'
    var_11 = {var_9: var_10}
    var_12 = b'.'
    var_13 = 1
    var_14 = {}

def test_case_0():
    var_0 = 'Parametrized version to strictly verify the compression logic branch.'
    var_1 = None



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'short'
    var_2 = var_0.dump_payload(var_1)
    var_3 = b'.'
    var_4 = b'a'
    var_5 = 100
    var_6 = var_4 * var_5
    var_7 = 'large'
    var_8 = var_0.dump_payload(var_7)
    var_9 = 1
    var_10 = var_8[var_9:]
    var_11 = module_1.base64_decode(var_10)
    var_12 = module_0.URLSafeSerializerMixin()
    var_13 = 'edge'
    var_14 = var_12.dump_payload(var_13)
    var_15 = b'.'

def test_case_0():
    var_0 = 'Verify that the logic specifically checks if len(compressed) < (len(json) - 1)'
    var_1 = b'123'
    var_2 = 'small'
    var_3 = b'.'
    var_4 = b'x'
    var_5 = 100
    var_6 = var_4 * var_5
    var_7 = 'large'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = b'abc'
    var_1 = 'abc'
    var_2 = b'.'
    var_3 = 'a'
    var_4 = 1000
    var_5 = var_3 * var_4
    var_6 = 'key'
    var_7 = 'list'
    var_8 = 'value'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = 50
    var_14 = var_12 * var_13
    var_15 = {var_6: var_8, var_7: var_14}
    var_16 = 'repeat'
    var_17 = 100
    var_18 = var_16 * var_17
    var_19 = 'utf-8'

def test_case_0():
    var_0 = 'Specific check for the compression flag logic.'
    var_1 = b'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = b'.'
    var_5 = b'a'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'small'
    var_1 = b'.'
    var_2 = 'large_string_to_force_compression'
    var_3 = 10
    var_4 = var_2 * var_3
    var_5 = b'"test_data"'
    var_6 = 'test_data'
    var_7 = b'"a"'



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'"hello"'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = b'!!!'
    var_4 = b'not compressed'
    var_5 = module_0.base64_encode(var_4)
    var_6 = var_2 + var_5
    var_7 = True



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'abc'
    var_1 = 'some_obj'
    var_2 = b'a'
    var_3 = module_0.base64_encode(var_2)
    var_4 = module_0.base64_decode(var_3)
    var_5 = b'.'
    var_6 = 100
    var_7 = var_2 * var_6
    var_8 = 1

def test_case_0():
    var_0 = None



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'a'
    var_1 = 'key'
    var_2 = 'val'
    var_3 = {var_1: var_2}
    var_4 = module_0.base64_encode(var_0)
    var_5 = b'.'
    var_6 = b'a'
    var_7 = 100
    var_8 = var_6 * var_7
    var_9 = 'key'
    var_10 = 'val'
    var_11 = {var_9: var_10}
    var_12 = b'.'
    var_13 = b'abcde'
    var_14 = module_0.base64_encode(var_13)
    var_15 = b'.'
    var_16 = var_15 + var_10
    var_17 = 'key'
    var_18 = 'val'
    var_19 = {var_17: var_18}



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'{"key": "value", "long_string": "some repetitive data" * 10}'
    var_3 = b'.'
    var_4 = 'utf-8'
    var_5 = b'!!!NotBase64!!!'
    var_6 = b'not compressed data'
    var_7 = module_0.base64_encode(var_6)
    var_8 = var_3 + var_7
    var_9 = b''
    var_10 = module_0.base64_encode(var_9)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'uncompressible'
    var_1 = b'.'
    var_2 = b'a'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = b'1234567890'
    var_6 = b'small'
    var_7 = var_2 * var_3
    var_8 = 1



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'small'
    var_1 = b'.'
    var_2 = 'large_string_to_force_compression'
    var_3 = 10
    var_4 = var_2 * var_3
    var_5 = 1
    var_6 = 'test'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'data'
    var_1 = 'small'
    var_2 = {var_0: var_1}
    var_3 = b'.'
    var_4 = b'a'
    var_5 = 1000
    var_6 = var_4 * var_5



