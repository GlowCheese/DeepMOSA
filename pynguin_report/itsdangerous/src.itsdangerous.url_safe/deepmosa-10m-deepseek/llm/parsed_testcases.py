####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 1/3 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_invalid_compression. Retrieved 1/4 statements.


def test_case_0():
    var_0 = b'eyJhIjogMX0='

def test_case_0():
    var_0 = b'.'
    var_1 = b'{"a": 1}'

def test_case_0():
    var_0 = b'invalid!'

def test_case_0():
    var_0 = b'.eyJhIjogMX0='



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 3/6 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 3/6 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_bad_compression. Retrieved 1/4 statements.
# Partially parsed test_load_payload_empty_payload. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'x'
    var_1 = 1000
    var_2 = var_0 * var_1

def test_case_0():
    var_0 = b'!!!invalid!!!'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = b'.AAAA'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = b''
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_payload_with_compressed_payload_enters_decompress_branch. Retrieved 2/7 statements.


def test_case_0():
    var_0 = b'.'
    var_1 = b'{"a":1}'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_payload_does_not_raise_base64_exception. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_payload_no_compression_no_prefix. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression_prefix. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_invalid_json. Retrieved 2/5 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid_base64!!!'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'not json'
    var_1 = module_0.base64_encode(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_payload_base64_decode_no_exception_with_valid_payload. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJhIjogMX0'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 1/3 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 1/3 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_empty. Retrieved 1/4 statements.
# Partially parsed test_load_payload_with_serializer_arg. Retrieved 1/3 statements.
# Partially parsed test_load_payload_with_additional_args. Retrieved 2/4 statements.


def test_case_0():
    var_0 = b'eyJhIjogMX0'

def test_case_0():
    var_0 = b'.eJxT0lEoSczJUdJRAABlKwR3'

def test_case_0():
    var_0 = b'invalid!!!'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = b''
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = b'WzFd'

def test_case_0():
    var_0 = b'ImhlbGxvIg'
    var_1 = 'extra'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_with_decompress_exception. Retrieved 10/24 statements.


def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test'
    var_2 = 'data'
    var_3 = 1000
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = 'ascii'
    var_7 = b'.'
    var_8 = b'corrupted_base64'
    var_9 = var_7 + var_8
    var_10 = 'Could not zlib decompress'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupt_zlib. Retrieved 4/7 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'.'
    var_1 = b'{"key":"value"}'

def test_case_0():
    var_0 = b'invalid'
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



# Parsed testcases at query #10
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'.'
    var_3 = b'not zlib compressed data'
    var_4 = module_1.base64_encode(var_3)
    var_5 = var_2 + var_4
    var_6 = []
    var_7 = {}
    var_8 = var_1.load_payload(var_5, *var_6, **var_7)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_payload_with_compressed_flag_but_valid_base64_and_valid_zlib. Retrieved 2/7 statements.


def test_case_0():
    var_0 = b'.'
    var_1 = b'{"a":1}'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_payload_does_not_raise_on_non_compressed_valid_payload. Retrieved 5/7 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_payload_decompress_false_does_not_raise. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_payload_no_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_invalid_compression. Retrieved 4/7 statements.


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
    var_1 = b'not compressed data'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_payload_does_not_raise_badpayload_for_valid_base64. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJmb28iOiAiYmFyIn0'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_payload_normal_base64_decoded. Retrieved 2/4 statements.
# Partially parsed test_load_payload_compressed. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64_raises_badpayload. Retrieved 1/4 statements.
# Partially parsed test_load_payload_compressed_corrupt_zlib_raises_badpayload. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty_string_byte. Retrieved 2/4 statements.


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
    var_1 = b'not valid zlib'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_payload_normal_payload. Retrieved 2/4 statements.
# Partially parsed test_load_payload_compressed_payload. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_corrupted_compressed. Retrieved 4/7 statements.
# Partially parsed test_load_payload_empty_payload. Retrieved 1/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid!!'
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

def test_case_0():
    var_0 = b''
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_payload_base64_decode_success. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJmb28iOiJiYXIifQ'



# Parsed testcases at query #19
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'.'
    var_3 = 1
    var_4 = b'{"a":1}'
    var_5 = zlib.compress(var_4)[:var_3]
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_2 + var_6
    var_8 = []
    var_9 = {}
    var_10 = var_1.load_payload(var_7, *var_8, **var_9)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_payload_no_decompress_does_not_raise_on_invalid_zlib. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_payload_base64_decode_success. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJhIjoiYiJ9'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_payload_without_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_with_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_with_corrupted_compressed. Retrieved 4/7 statements.
# Partially parsed test_load_payload_with_empty_payload. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid!base64'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'notvalidzlib'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_payload_no_compression. Retrieved 4/7 statements.
# Partially parsed test_dump_payload_with_compression. Retrieved 6/11 statements.


def test_case_0():
    var_0 = b'{"a":1}'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = b'a'
    var_1 = 100
    var_2 = var_0 * var_1
    var_3 = 'test'
    var_4 = b'.'
    var_5 = b'='



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_payload_no_exception_on_valid_base64. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_payload_with_compressed_data_triggers_decompress_path. Retrieved 2/7 statements.


def test_case_0():
    var_0 = b'.'
    var_1 = b'{"key":"value"}'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_payload_base64_decodes_payload. Retrieved 2/4 statements.
# Partially parsed test_load_payload_decompresses_when_prefixed_with_dot. Retrieved 2/7 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_base64_decode_error. Retrieved 1/4 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_decompress_error. Retrieved 4/7 statements.
# Partially parsed test_load_payload_handles_empty_payload. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid!base64'
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
    var_0 = b'null'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_payload_without_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_decompression_error. Retrieved 4/7 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid-base64!!!'
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_payload_no_decompress_no_exception. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_empty_payload. Retrieved 1/4 statements.
# Partially parsed test_load_payload_with_dot. Retrieved 1/4 statements.
# Partially parsed test_load_payload_without_dot. Retrieved 1/4 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_compressed. Retrieved 1/4 statements.


def test_case_0():
    var_0 = b''

def test_case_0():
    var_0 = b'.eJw9kE0OAiEMhe8y6wYosDDxLgY3Jh7AoTPh5u6F0bj5vveT9vV7z3s_AQ'

def test_case_0():
    var_0 = b'eyJhIjogMX0'

def test_case_0():
    var_0 = b'!!!'

def test_case_0():
    var_0 = b'.eJw9kE0OAiEMhe8y6wYosDDxLgY3Jh7AoTPh5u6F0bj5vveT9vV7z3s_AQ'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_payload_base64_decode_success. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJrZXkiOiAidmFsdWUifQ'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_payload_base64_decode_succeeds. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJrZXkiOiAidmFsdWUifQ'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_payload_no_decompress_no_exception. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJhIjoxfQ'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_payload_normal_base64_decoded. Retrieved 2/4 statements.
# Partially parsed test_load_payload_compressed. Retrieved 2/7 statements.
# Partially parsed test_load_payload_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_bad_compressed_data. Retrieved 4/7 statements.


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
    var_1 = b'not compressed'
    var_2 = module_0.base64_encode(var_1)
    var_3 = var_0 + var_2
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_payload_without_compression. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_compression. Retrieved 2/7 statements.
# Partially parsed test_load_payload_with_compression_prefix. Retrieved 2/7 statements.
# Partially parsed test_load_payload_raises_bad_payload_for_invalid_base64. Retrieved 1/4 statements.
# Partially parsed test_load_payload_raises_bad_payload_for_corrupted_compressed_data. Retrieved 4/7 statements.
# Partially parsed test_load_payload_with_empty_payload. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_none_payload. Retrieved 2/4 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = module_0.base64_encode(var_0)

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'{"key":"value"}'
    var_1 = b'.'

def test_case_0():
    var_0 = b'invalid_base64!!!'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'.'
    var_1 = b'not_valid_zlib_data'
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
    var_0 = b'null'
    var_1 = module_0.base64_encode(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_payload_with_dot_prefix_and_invalid_compressed_data_raises_bad_payload. Retrieved 5/11 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'x\x9c\xcbH\xcd\xc9\xc9\x07\x00\x06,\x02\x15'
    var_3 = b'='
    var_4 = b'.'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'.invalid_base64'
    var_3 = []
    var_4 = {}
    var_5 = var_1.load_payload(var_2, *var_3, **var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_payload_no_base64_exception. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'SGVsbG8='



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_payload_no_decompress_does_not_raise_decompress_exception. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJrZXkiOiAidmFsdWUifQ'



# Parsed testcases at query #18
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'eyJhIjogMX0'
    var_3 = []
    var_4 = {}
    var_5 = var_1.load_payload(var_2, *var_3, **var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.URLSafeSerializerMixin(var_0)
    var_2 = b'eyJrZXkiOiAidmFsdWUifQ'
    var_3 = []
    var_4 = {}
    var_5 = var_1.load_payload(var_2, *var_3, **var_4)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_payload_base64_decode_does_not_raise. Retrieved 1/3 statements.


def test_case_0():
    var_0 = b'eyJhIjoiYiJ9'



