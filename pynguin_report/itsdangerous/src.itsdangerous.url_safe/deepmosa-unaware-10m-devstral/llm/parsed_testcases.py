####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'
    var_8 = 'short'
    var_9 = var_0.dump_payload(var_8)
    var_10 = 'test'
    var_11 = 'number'
    var_12 = 'data'
    var_13 = 42
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = var_0.dump_payload(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = {}
    var_18 = var_0.dump_payload(var_17)
    var_19 = var_0.load_payload(var_18)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."'
    var_5 = b'"'
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'.'
    var_9 = b'not_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)
    var_13 = b''
    var_14 = var_0.load_payload(var_13)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value"}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'x\x9c'
    var_10 = b'invalid'
    var_11 = var_9 + var_10
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_8 + var_12
    var_14 = var_0.load_payload(var_13)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."test"'
    var_5 = var_0.load_payload(var_4)
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'{"test": "data"}'
    var_9 = b'.'
    var_10 = var_0.load_payload(var_4)
    var_11 = b''
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    assert var_13 == ''



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJrZXkiOiJ2YWx1ZSJ9'
    var_2 = var_0.load_payload(var_1)
    var_3 = b'.eJxLtDK2MjI0MrdSMbIyNlAyNjU1NlAyXQMAAA=='
    var_4 = var_0.load_payload(var_3)
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.invalid_base64'
    var_8 = var_0.load_payload(var_7)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."test compressed data"'
    var_5 = var_0.load_payload(var_4)
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'.'
    var_9 = b'invalid_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value", "data": "x" * 1000}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'not_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_5 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value", "data": "x" * 1000}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'.'
    var_10 = b'not_compressed_data'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_9 + var_11
    var_13 = var_0.load_payload(var_12)
    var_14 = b'{}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'
    var_8 = {var_1: var_2}
    var_9 = var_0.dump_payload(var_8)
    var_10 = 'test'
    var_11 = 'number'
    var_12 = 'data'
    var_13 = 42
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = var_0.dump_payload(var_14)
    var_16 = var_0.load_payload(var_15)
    var_17 = {}
    var_18 = var_0.dump_payload(var_17)
    var_19 = var_0.load_payload(var_18)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."'
    var_5 = b'"'
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_zlib_compressed'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_10 + var_5
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'
    var_8 = {var_1: var_2}
    var_9 = var_0.dump_payload(var_8)
    var_10 = {}
    var_11 = var_0.dump_payload(var_10)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJ0ZXN0IjogInZhbHVlIn0='
    var_2 = var_0.load_payload(var_1)
    var_3 = '{"test": "value"}'
    var_4 = b'.'
    var_5 = var_0.load_payload(var_1)
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'.invalid_compressed_data'
    var_9 = var_0.load_payload(var_8)
    var_10 = b''
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'
    var_8 = {var_1: var_2}
    var_9 = var_0.dump_payload(var_8)
    var_10 = {}
    var_11 = var_0.dump_payload(var_10)



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJrZXkiOiJ2YWx1ZSJ9'
    var_2 = var_0.load_payload(var_1)
    var_3 = b'{"key": "value", "another_key": "another_value"}'
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #16
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
    var_11 = var_0.load_payload(var_4)
    var_12 = var_0.load_payload(var_10)
    var_13 = {}
    var_14 = var_0.dump_payload(var_13)
    var_15 = var_0.load_payload(var_14)



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
    var_5 = b'.'
    var_6 = module_1.base64_decode(var_4)
    assert var_6 == b'{"key":"value"}'
    var_7 = 1000
    var_8 = var_2 * var_7
    var_9 = {var_1: var_8}
    var_10 = var_0.dump_payload(var_9)
    var_11 = 1
    var_12 = var_10[var_11:]
    var_13 = module_1.base64_decode(var_12)
    var_14 = {}
    var_15 = var_0.dump_payload(var_14)
    var_16 = module_1.base64_decode(var_15)
    assert var_16 == b'{}'



# Parsed testcases at query #18
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
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'not_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value", "data": "x" * 1000}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'not_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'
    var_8 = {var_1: var_2}
    var_9 = var_0.dump_payload(var_8)
    var_10 = {}
    var_11 = var_0.dump_payload(var_10)
    var_12 = 'test'
    var_13 = 'number'
    var_14 = 'data'
    var_15 = 42
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = var_0.dump_payload(var_16)
    var_18 = var_0.load_payload(var_17)



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value", "data": "x" * 1000}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'invalid_zlib_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_5 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'invalid_zlib'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'invalid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'.invalid_base64!'
    var_9 = var_0.load_payload(var_8)
    var_10 = b''
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_0.load_payload(var_11)
    assert var_12 == ''



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



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
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."test compressed data"'
    var_5 = var_0.load_payload(var_4)
    var_6 = b'invalid base64'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'.invalid compressed data'
    var_9 = var_0.load_payload(var_8)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'invalid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = var_0.load_payload(var_12)



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.invalid_zlib_data'
    var_8 = var_0.load_payload(var_7)
    var_9 = b''
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'invalid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = var_0.load_payload(var_12)



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJrZXkiOiJ2YWx1ZSJ9'
    var_2 = var_0.load_payload(var_1)
    var_3 = b'{"key": "value", "another_key": "another_value"}'
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."test"'
    var_5 = module_0.URLSafeSerializerMixin()
    var_6 = b'{"test": "compressed"}'
    var_7 = b'.'
    var_8 = var_5.load_payload(var_2)
    var_9 = module_0.URLSafeSerializerMixin()
    var_10 = b'invalid_base64'
    var_11 = var_9.load_payload(var_10)
    var_12 = module_0.URLSafeSerializerMixin()
    var_13 = b'invalid_compressed_data'
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_11 + var_14
    var_16 = var_12.load_payload(var_15)



# Parsed testcases at query #41
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'
    var_8 = 1
    var_9 = var_6[var_8:]
    var_10 = module_1.base64_decode(var_9)
    var_11 = 'short'
    var_12 = var_0.dump_payload(var_11)
    var_13 = module_1.base64_decode(var_12)
    var_14 = ''
    var_15 = var_0.dump_payload(var_14)
    var_16 = module_1.base64_decode(var_15)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'
    var_8 = 1
    var_9 = var_6[var_8:]
    var_10 = module_1.base64_decode(var_9)
    var_11 = var_0.load_payload(var_6)
    var_12 = {var_1: var_2}
    var_13 = var_0.dump_payload(var_12)
    var_14 = module_1.base64_decode(var_13)
    var_15 = var_0.load_payload(var_13)
    var_16 = 1
    var_17 = var_13[var_16:]
    var_18 = module_1.base64_decode(var_17)
    var_19 = var_0.load_payload(var_13)
    var_20 = {}
    var_21 = var_0.dump_payload(var_20)
    var_22 = var_0.load_payload(var_21)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'
    var_8 = {var_1: var_2}
    var_9 = var_0.dump_payload(var_8)
    var_10 = {}
    var_11 = var_0.dump_payload(var_10)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'invalid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value"}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'not_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid_base64'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'.'
    var_9 = b'invalid_zlib_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)
    var_13 = b''
    var_14 = var_0.load_payload(var_13)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"test": "data"}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'.'
    var_10 = b'invalid_zlib_data'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_9 + var_11
    var_13 = var_0.load_payload(var_12)
    var_14 = b''
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)
    assert var_16 == ''
    var_17 = b'{"special": "chars: @#$%^&*()"}'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_0.load_payload(var_18)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value", "data": "x" * 1000}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'invalid_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'invalid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = var_0.load_payload(var_12)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = var_0.load_payload(var_12)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value"}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'.invalid_zlib'
    var_10 = var_0.load_payload(var_9)
    var_11 = b''
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    assert var_13 == ''
    var_14 = b'{"key": "value with spaces & symbols"}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid_base64'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_compressed'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJ0ZXN0IjogInZhbHVlIn0='
    var_2 = var_0.load_payload(var_1)
    var_3 = b'.eJwLyjJTqgwAAB+LCg=='
    var_4 = var_0.load_payload(var_3)
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.invalid_compressed!'
    var_8 = var_0.load_payload(var_7)



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'invalid_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"test": "data"}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'invalid_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.invalid_compressed_data'
    var_8 = var_0.load_payload(var_7)
    var_9 = b''
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."test compressed data"'
    var_5 = module_0.URLSafeSerializerMixin()
    var_6 = var_5.load_payload(var_4)
    var_7 = b'invalid base64'
    var_8 = var_5.load_payload(var_7)
    var_9 = b''
    var_10 = var_5.load_payload(var_9)



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid_base64'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'invalid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value"}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'not_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJ0ZXN0IjogInZhbHVlIn0='
    var_2 = var_0.load_payload(var_1)
    var_3 = b'.'
    var_4 = b'{"test": "value"}'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_5 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."test_compressed_data"'
    var_5 = var_0.load_payload(var_4)
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'.invalid_compressed_data'
    var_9 = var_0.load_payload(var_8)



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."'
    var_5 = b'"'
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'not_zlib_compressed'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_10 + var_5
    var_12 = var_0.load_payload(var_11)



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
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)
    var_11 = b''
    var_12 = module_1.base64_encode(var_11)
    var_13 = var_0.load_payload(var_12)
    assert var_13 == ''
    var_14 = b'{"key": "value with spaces & symbols"}'
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_0.load_payload(var_15)



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJ0ZXN0IjogInZhbHVlIn0='
    var_2 = 'test'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.load_payload(var_1)
    var_6 = b'{"test": "value"}'
    var_7 = b'.'
    var_8 = b'invalid_base64!'
    var_9 = var_0.load_payload(var_8)
    var_10 = b'.invalid_base64!'
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"test": "data"}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'not_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value", "data": "x" * 1000}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'invalid_zlib_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value", "data": "x" * 1000}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'invalid_zlib_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_5 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid_base64'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'.'
    var_9 = b'invalid_zlib_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)
    var_13 = module_0.URLSafeSerializerMixin()
    var_14 = b'{"custom": "serializer"}'
    var_15 = module_1.base64_encode(var_14)



# Parsed testcases at query #41
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'invalid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    assert var_14 == ''
    var_15 = b'{"special": "chars: @#$%^&*()}")'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_0.load_payload(var_16)



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #45
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.'
    var_8 = b'invalid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'{"key": "value", "another_key": "another_value"}'
    var_5 = b'.'
    var_6 = var_0.load_payload(var_2)
    var_7 = b'invalid_base64!'
    var_8 = var_0.load_payload(var_7)
    var_9 = b'not_compressed_data'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_8 + var_10
    var_12 = var_0.load_payload(var_11)



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'not_zlib_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'invalid_compressed_data'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_4 + var_8
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."'
    var_5 = b'invalid_base64'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'.invalid_compressed_data'
    var_8 = var_0.load_payload(var_7)
    var_9 = b''
    var_10 = var_0.load_payload(var_9)



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'."'
    var_5 = b'invalid_base64!'
    var_6 = var_0.load_payload(var_5)
    var_7 = b'."'
    var_8 = b'invalid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)
    var_12 = b''
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_0.load_payload(var_13)
    assert var_14 == ''



# Parsed testcases at query #52
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
    var_6 = b'invalid_base64'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'invalid_zlib_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)



# Parsed testcases at query #53
#--------------------------


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    var_4 = b'.'
    var_5 = var_0.load_payload(var_2)
    var_6 = b'invalid_base64!'
    var_7 = var_0.load_payload(var_6)
    var_8 = b'invalid_compressed_data'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = var_0.load_payload(var_10)



