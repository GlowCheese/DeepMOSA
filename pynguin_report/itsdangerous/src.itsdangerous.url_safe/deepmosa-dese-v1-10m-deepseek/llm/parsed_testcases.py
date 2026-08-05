####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dump_payload_without_compression. Retrieved 6/8 statements.
# Partially parsed test_dump_payload_with_compression. Retrieved 8/10 statements.
# Partially parsed test_dump_payload_compression_threshold. Retrieved 6/8 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'x'
    var_2 = 1000
    var_3 = var_1 * var_2
    var_4 = 'data'
    var_5 = {var_4: var_3}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'x'
    var_2 = 'data'
    var_3 = {var_2: var_1}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_payload_compressed_valid. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid_base64'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'not_zlib_compressed'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b''
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    assert var_3 is None

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'null'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    assert var_3 is None



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'a'
    var_2 = 1000
    var_3 = var_1 * var_2
    var_4 = var_0.dump_payload(var_3)



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



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJhIjogMX0'
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_payload_compressed_base64. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'{"key": "value"}'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid!!!'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'not compressed'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b''
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_payload_no_compression. Retrieved 6/7 statements.
# Partially parsed test_dump_payload_with_compression. Retrieved 8/9 statements.
# Partially parsed test_dump_payload_compressed_length_check. Retrieved 6/7 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'ey'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'x'
    var_2 = 1000
    var_3 = var_1 * var_2
    var_4 = 'data'
    var_5 = {var_4: var_3}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'short'
    var_2 = 'data'
    var_3 = {var_2: var_1}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 11/15 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 10/18 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 8/13 statements.
# Partially parsed test_load_payload_bad_compression. Retrieved 11/16 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = 'load_payload'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = lambda self, payload, *args, **kwargs: var_4
    var_6 = {var_1: var_5}
    var_7 = module_0.URLSafeSerializerMixin()
    var_8 = b'{"key": "value"}'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7.load_payload(var_9)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = 'load_payload'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = lambda self, payload, *args, **kwargs: var_4
    var_6 = {var_1: var_5}
    var_7 = module_0.URLSafeSerializerMixin()
    var_8 = b'{"key": "value"}'
    var_9 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = 'load_payload'
    var_2 = {}
    var_3 = lambda self, payload, *args, **kwargs: var_2
    var_4 = {var_1: var_3}
    var_5 = module_0.URLSafeSerializerMixin()
    var_6 = b'invalid!'
    var_7 = var_5.load_payload(var_6)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = 'load_payload'
    var_2 = {}
    var_3 = lambda self, payload, *args, **kwargs: var_2
    var_4 = {var_1: var_3}
    var_5 = module_0.URLSafeSerializerMixin()
    var_6 = b'.'
    var_7 = b'not compressed'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_6 + var_8
    var_10 = var_5.load_payload(var_9)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dump_payload_compression_triggered. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = 1000
    var_3 = var_1 * var_2
    var_4 = b'.'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dump_payload_compression_triggered. Retrieved 7/8 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'a'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = var_1.dump_payload(var_4)
    var_6 = b'.'



# Parsed testcases at query #11
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJhIjoiYiJ9'
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_payload_with_valid_compressed_payload_does_not_raise. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}



# Parsed testcases at query #13
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'.aW52YWxpZA=='
    var_3 = var_1.load_payload(var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_payload_compressed_with_invalid_zlib_data_raises_badpayload. Retrieved 5/15 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'.'
    var_2 = b'invalid-zlib-data'
    var_3 = module_0.base64_encode(var_2)
    var_4 = var_1 + var_3



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"a":1}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_payload_compressed_with_dot_prefix. Retrieved 3/7 statements.
# Partially parsed test_load_payload_corrupted_compressed_data_raises_badpayload. Retrieved 4/10 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"a":1}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"a":1}'
    var_2 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid!!!'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"a":1}'
    var_2 = -1
    var_3 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b''
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = 1
    var_7 = var_5[var_6:]
    var_8 = var_1.load_payload(var_7)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_payload_no_decompression_on_valid_compressed_payload. Retrieved 10/11 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = b'.'
    var_7 = 1
    var_8 = var_5[var_7:]
    var_9 = var_1.load_payload(var_8)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_payload_with_decompress_valid_base64. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid_base64'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'invalid_compressed_data'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #20
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_payload_compressed. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid!!!'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'not compressed'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_payload_with_compression. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid!'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'not compressed data'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b''
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_payload_no_compression. Retrieved 3/8 statements.
# Partially parsed test_dump_payload_with_compression. Retrieved 9/13 statements.
# Partially parsed test_dump_payload_compression_not_applied_when_not_beneficial. Retrieved 4/8 statements.
# Partially parsed test_dump_payload_compression_boundary. Retrieved 5/12 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test'
    var_2 = var_0.dump_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'x'
    var_2 = 1000
    var_3 = var_1 * var_2
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 1
    var_7 = var_4[var_6:]
    var_8 = module_1.base64_decode(var_7)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'short'
    var_2 = var_0.dump_payload(var_1)
    var_3 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test'
    var_2 = 1
    var_3 = var_0.dump_payload(var_1)
    var_4 = b'.'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_payload_without_compression. Retrieved 10/13 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 10/18 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = ()
    var_2 = 'load_payload'
    var_3 = 'decoded'
    var_4 = lambda self, payload, *args, **kwargs: var_3
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)
    var_7 = module_0.URLSafeSerializerMixin()
    var_8 = b'eyJhIjoiYiJ9'
    var_9 = var_7.load_payload(var_8)
    assert var_9 == 'decoded'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = ()
    var_2 = 'load_payload'
    var_3 = 'decoded'
    var_4 = lambda self, payload, *args, **kwargs: var_3
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)
    var_7 = module_0.URLSafeSerializerMixin()
    var_8 = b'{"a":"b"}'
    var_9 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid!!!'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.invalidbase64'
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_payload_with_compression. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'{"key":"value"}'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid!base64'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'not_valid_zlib'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b''
    var_2 = var_0.load_payload(var_1)
    assert var_2 is None



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_payload_with_compressed_data_triggers_decompress. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_with_compression. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid!!!'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'not_compressed'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b''
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_payload_with_compressed_flag_but_invalid_data_does_not_raise_zlib_error. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = b'{"key":"value"}'
    var_2 = b'eJxLys9PUbC1AwAADgACAQ=='
    var_3 = b'.'
    var_4 = var_3 + var_2



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'{"a":1}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_payload_compressed_payload. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid!!!'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'not compressed data'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #14
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'invalid_zlib_data'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_payload_with_compression. Retrieved 11/12 statements.


import src.itsdangerous.serializer as module_0
import src.itsdangerous.url_safe as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Serializer(var_0)
    var_2 = module_1.URLSafeSerializerMixin(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dump_payload(var_5)
    var_7 = var_2.load_payload(var_6)

import src.itsdangerous.serializer as module_0
import src.itsdangerous.url_safe as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Serializer(var_0)
    var_2 = module_1.URLSafeSerializerMixin(var_1)
    var_3 = 'key'
    var_4 = 'x'
    var_5 = 1000
    var_6 = var_4 * var_5
    var_7 = {var_3: var_6}
    var_8 = var_2.dump_payload(var_7)
    var_9 = b'.'
    var_10 = var_2.load_payload(var_8)

import src.itsdangerous.serializer as module_0
import src.itsdangerous.url_safe as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Serializer(var_0)
    var_2 = module_1.URLSafeSerializerMixin(var_1)
    var_3 = b'invalid_base64'
    var_4 = var_2.load_payload(var_3)

import src.itsdangerous.serializer as module_0
import src.itsdangerous.url_safe as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Serializer(var_0)
    var_2 = module_1.URLSafeSerializerMixin(var_1)
    var_3 = b'.invalid_base64'
    var_4 = var_2.load_payload(var_3)



# Parsed testcases at query #16
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'.'
    var_3 = b'not valid zlib compressed data'
    var_4 = module_1.base64_encode(var_3)
    var_5 = var_2 + var_4
    var_6 = var_1.load_payload(var_5)



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = var_1.load_payload(var_5)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_payload_with_dot_prefix_but_valid_base64_and_regular_compressed_data_does_not_raise_bad_payload_at_line_25. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'not_zlib_compressed'
    var_5 = b'='
    var_6 = b'.'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_payload_base64_decode_succeeds. Retrieved 6/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'load_payload'
    var_2 = {}
    var_3 = b'{"key": "value"}'
    var_4 = module_1.base64_encode(var_3)
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #20
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key": "value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = b'.'
    var_4 = var_3 + var_2
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #21
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJmb28iOiAiYmFyIn0'
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #22
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'{"a":1}'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1.load_payload(var_3)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_payload_with_compression. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"key":"value"}'
    var_2 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid_base64!'
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
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'null'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    assert var_3 is None



# Parsed testcases at query #24
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'!!!invalid base64!!!'
    var_2 = var_0.load_payload(var_1)



