####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_payload_basic. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupted_compression. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty. Retrieved 1/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'.'
    var_1 = b'{"key":"value"}'

def test_case_0():
    var_0 = b'invalid!!!'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'not compressed'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = b''
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_payload_decompress_true. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = {var_0: var_3}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_payload_normal_base64. Retrieved 2/4 statements.
# Partially parsed test_load_payload_compressed. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_invalid_compressed. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'!!!invalid!!!'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'not compressed data'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_payload_without_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupted_compressed_data. Retrieved 4/7 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid_base64'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'corrupted_data'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 8/13 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupted_compression. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty_payload. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"'
    var_1 = b'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = b'"}'
    var_6 = var_4 + var_5
    var_7 = b'.'

def test_case_0():
    var_0 = b'invalid!!!'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'not_compressed'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{}'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_payload_with_compressed_data_triggers_decompress. Retrieved 2/7 statements.


def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = b'.'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_payload_starts_with_dot. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'.eJw9yjEOgCAMBdC7vAEFCkU4i4uLYXAwxnt4e3VxetMX_1I7Vg'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_normal. Retrieved 2/4 statements.
# Partially parsed test_load_payload_compressed. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupted_compressed. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid!!!'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'not-zlib-data'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line22_evaluates_to_true. Retrieved 3/11 statements.


def test_case_0():
    var_0 = b'.'
    var_1 = b'{"test": "value"}'
    var_2 = b'='



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_payload_base64_decode_success. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJhIjoiYiJ9'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_payload_with_valid_base64_no_exception. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_payload_with_compressed_payload_triggers_decompress. Retrieved 2/7 statements.


def test_case_0():
    var_0 = b'.'
    var_1 = b'{"key":"value"}'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 8/13 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupted_compression. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty_payload. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"'
    var_1 = b'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = var_0 + var_3
    var_5 = b'"}'
    var_6 = var_4 + var_5
    var_7 = b'.'

def test_case_0():
    var_0 = b'invalid!!!'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'not_compressed_data'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_payload_base64_decode_success. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJhIjogMX0'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_payload_decompress_true. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'{"key": "value"}'
    var_4 = b'.'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_decompress_flag_true. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '.'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupted_compression. Retrieved 4/7 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid!'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'not compressed data'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 6/13 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 6/16 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 5/13 statements.
# Partially parsed test_load_payload_corrupted_compression. Retrieved 8/16 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = 'load_payload'
    var_2 = lambda self, json, *args, **kwargs: json
    var_3 = {var_1: var_2}
    var_4 = b'{"key":"value"}'
    var_5 = module_0.base64_encode(var_4)

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = 'load_payload'
    var_2 = lambda self, json, *args, **kwargs: json
    var_3 = {var_1: var_2}
    var_4 = b'{"key":"value"}'
    var_5 = b'.'

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = 'load_payload'
    var_2 = lambda self, json, *args, **kwargs: json
    var_3 = {var_1: var_2}
    var_4 = b'invalid!'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'MockSerializer'
    var_1 = 'load_payload'
    var_2 = lambda self, json, *args, **kwargs: json
    var_3 = {var_1: var_2}
    var_4 = b'.'
    var_5 = b'not compressed data'
    var_6 = module_0.base64_encode(var_5)
    var_7 = var_4 + var_6
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_invalid_compressed. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty_payload. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid_base64'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'invalid_compressed_data'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'null'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_payload_starts_with_dot_sets_decompress_true. Retrieved 4/8 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'.'
    var_3 = b'{"a":1}'



# Parsed testcases at query #21
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = []
    var_7 = {}
    var_8 = var_1.load_payload(var_5, *var_6, **var_7)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_payload_without_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_compressed_invalid_zlib. Retrieved 4/7 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid_base64'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'not_zlib_compressed'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_payload_with_compressed_flag_but_invalid_compressed_data_does_not_raise_bad_payload. Retrieved 5/13 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key":"value"}'
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'.'
    var_4 = var_3 + var_2



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_payload_compressed_and_invalid_base64_does_not_raise_zlib_error. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'.'
    var_2 = b'invalid_base64'
    var_3 = var_1 + var_2



# Parsed testcases at query #25
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'.!!'
    var_3 = []
    var_4 = {}
    var_5 = var_1.load_payload(var_2, *var_3, **var_4)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_payload_with_compressed_flag_and_non_compressed_data_does_not_raise_zlib_error. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = b'test data'
    var_2 = b'.'
    var_3 = 'utf-8'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_payload_uncompressed_valid. Retrieved 2/4 statements.
# Partially parsed test_load_payload_compressed_valid. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_compressed_corrupt. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_serializer. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid@@@'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'corrupt data'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_encode(var_0)

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'[1,2,3]'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #28
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = []
    var_7 = {}
    var_8 = var_1.load_payload(var_5, *var_6, **var_7)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_payload_base64_decode_success. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"test":"data"}'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_payload_after_base64_decode_does_not_raise_zlib_error. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = 'ascii'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_payload_without_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_invalid_compression. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty_payload. Retrieved 1/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid!!!'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'invalid compressed data'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = b''
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_payload_no_compression. Retrieved 4/9 statements.
# Partially parsed test_dump_payload_with_compression. Retrieved 7/13 statements.
# Partially parsed test_dump_payload_compression_threshold. Retrieved 4/7 statements.
# Partially parsed test_dump_payload_empty_object. Retrieved 2/6 statements.
# Partially parsed test_dump_payload_list. Retrieved 4/7 statements.
# Partially parsed test_dump_payload_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'

def test_case_0():
    var_0 = 'key'
    var_1 = 'x'
    var_2 = 1000
    var_3 = var_1 * var_2
    var_4 = {var_0: var_3}
    var_5 = b'.'
    var_6 = 1

def test_case_0():
    var_0 = 'key'
    var_1 = 'short'
    var_2 = {var_0: var_1}
    var_3 = b'.'

def test_case_0():
    var_0 = {}
    var_1 = b'.'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupt_compression. Retrieved 4/7 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid!'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'not_compressed_data'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_payload_no_compression. Retrieved 4/10 statements.
# Partially parsed test_dump_payload_with_compression. Retrieved 6/10 statements.
# Partially parsed test_dump_payload_compression_not_beneficial. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}

def test_case_0():
    var_0 = 'data'
    var_1 = 'x'
    var_2 = 1000
    var_3 = var_1 * var_2
    var_4 = {var_0: var_3}
    var_5 = b'.'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = b'.'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_payload_decompression_true. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'utf-8'
    var_5 = b'.'
    var_6 = b'{"key":"value"}'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_payload_compression_triggered. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1000
    var_2 = var_0 * var_1
    var_3 = b'.'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_payload_with_compressed_data_starts_with_dot. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'test_secret'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = b'.'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_without_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupt_compression. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty_string. Retrieved 2/4 statements.
# Partially parsed test_load_payload_none. Retrieved 1/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid_base64'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'corrupt_data'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = var_2 + var_1
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dump_payload_no_compression. Retrieved 4/8 statements.
# Partially parsed test_dump_payload_with_compression. Retrieved 6/10 statements.
# Partially parsed test_dump_payload_compressed_shorter. Retrieved 7/20 statements.
# Partially parsed test_dump_payload_no_compression_when_compressed_larger. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'.'

def test_case_0():
    var_0 = 'key'
    var_1 = 'x'
    var_2 = 1000
    var_3 = var_1 * var_2
    var_4 = {var_0: var_3}
    var_5 = b'.'

def test_case_0():
    var_0 = 'key'
    var_1 = 'a'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = {var_0: var_3}
    var_5 = b'.'
    var_6 = bool(var_1)
    assert var_6 is True
    var_7 = b'.'
    var_8 = bool(not var_1)
    assert var_8 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 'a'
    var_2 = {var_0: var_1}
    var_3 = b'.'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dump_payload_triggers_compression. Retrieved 7/8 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'a'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = var_1.dump_payload(var_4)
    var_6 = b'.'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_payload_with_dot_prefix. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = b'.'
    var_7 = var_6 + var_5
    var_8 = []
    var_9 = {}
    var_10 = var_1.load_payload(var_7, *var_8, **var_9)
    var_11 = bool(var_10 == var_4)
    assert var_11 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_payload_normal_payload. Retrieved 2/5 statements.
# Partially parsed test_load_payload_compressed_payload. Retrieved 2/8 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/5 statements.
# Partially parsed test_load_payload_corrupted_compressed. Retrieved 4/8 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = b'{"a":1}'
    var_3 = module_0.base64_encode(var_2)

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = b'{"a":1}'
    var_3 = b'.'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = b'invalid!!!'
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = b'not valid zlib data'
    var_3 = module_0.base64_encode(var_2)
    var_4 = b'.'
    var_5 = var_4 + var_3
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = []
    var_7 = {}
    var_8 = var_1.load_payload(var_5, *var_6, **var_7)



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.dump_payload(var_4)
    var_6 = []
    var_7 = {}
    var_8 = var_1.load_payload(var_5, *var_6, **var_7)



# Parsed testcases at query #16
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'!!!'
    var_3 = []
    var_4 = {}
    var_5 = var_1.load_payload(var_2, *var_3, **var_4)
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_payload_base64_decode_success. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJhIjoyfQ=='



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_payload_with_dot_prefix_and_valid_base64_but_invalid_zlib_does_not_raise_bad_payload. Retrieved 5/14 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'.'
    var_2 = b'not compressed data'
    var_3 = module_0.base64_encode(var_2)
    var_4 = var_1 + var_3



# Parsed testcases at query #19
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'.invalidbase64'
    var_3 = []
    var_4 = {}
    var_5 = var_1.load_payload(var_2, *var_3, **var_4)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_payload_base64_decode_success. Retrieved 1/4 statements.


def test_case_0():
    var_0 = b'eyJhIjogMX0'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_payload_without_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_with_zlib_error. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty_compressed. Retrieved 2/7 statements.
# Partially parsed test_load_payload_with_serializer_override. Retrieved 3/5 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid@@@'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'not compressed data'
    var_1 = module_0.base64_encode(var_0)
    var_2 = b'.'
    var_3 = var_2 + var_1
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = b'.'
    var_1 = b'{}'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)
    var_2 = None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_payload_with_compressed_flag_and_invalid_compressed_data_does_not_raise_zlib_error. Retrieved 6/12 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'.'
    var_3 = -1
    var_4 = b'{"valid": "json"}'
    var_5 = b'x'



# Parsed testcases at query #23
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'.invalid-base64!!'
    var_3 = []
    var_4 = {}
    var_5 = var_1.load_payload(var_2, *var_3, **var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_payload_no_exception_on_valid_base64. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_payload_no_decompress_after_decode. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_payload_uncompressed. Retrieved 2/4 statements.
# Partially parsed test_load_payload_compressed. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupt_compressed. Retrieved 4/7 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid!'

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'not_zlib_compressed'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2



