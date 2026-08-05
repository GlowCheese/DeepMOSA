####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dump_payload_with_compression. Retrieved 6/8 statements.
# Partially parsed test_dump_payload_without_compression. Retrieved 4/6 statements.


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
    var_1 = 'short'
    var_2 = var_0.dump_payload(var_1)
    var_3 = b'.'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_compression_when_shorter. Retrieved 6/7 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eNpLzEosS0lORVGyUk7MT1WyUjJSM7My1FEoLkktKsxLzUnNTS3WAMJD8gk='
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eyJrZXkiOiAidmFsdWUifQ=='
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid_base64_data'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'eNpLzEosS0lORVGyUk7MT1WyUjJSM7My1FEoLkktKsxLzUnNTS3WAMJD8gk='
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b''
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.eNrzSM3JyVcozy/KSdEoLkktKk1R0lFKzy/K1dUqzi9KL8lMyU9JLEpNlQwAAP//AwA='
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_payload_compression_predicate. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = b'.'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_payload_compression_when_compressed_is_smaller. Retrieved 8/9 statements.


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



# Parsed testcases at query #8
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'test_data'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    assert var_3 == b'test_data'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dump_payload_with_compression. Retrieved 6/8 statements.
# Partially parsed test_dump_payload_without_compression. Retrieved 4/6 statements.


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
    var_1 = 'short'
    var_2 = var_0.dump_payload(var_1)
    var_3 = b'.'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_payload_with_compressed_data. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'{"test": "data"}'

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

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
    var_2 = b'invalid_zlib_data'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dump_payload_compression_predicate. Retrieved 8/9 statements.


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



# Parsed testcases at query #13
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #14
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'x\x9c\xabH\xcd\xc9\xc9W(\xcf/\xcaI\x01\x00\x18\xab\x04T'
    var_2 = module_1.base64_encode(var_1)
    var_3 = b'.'
    var_4 = var_3 + var_2
    var_5 = var_0.load_payload(var_4)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid_base64'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'x\x00\x00\x00'
    var_2 = module_1.base64_encode(var_1)
    var_3 = b'.'
    var_4 = var_3 + var_2
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dump_payload_compression_when_compressed_is_smaller. Retrieved 8/9 statements.


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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_payload_with_compressed_data. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'{"test": "data"}'

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

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
    var_2 = b'not_zlib_compressed'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dump_payload_with_compression. Retrieved 6/8 statements.
# Partially parsed test_dump_payload_without_compression. Retrieved 6/8 statements.


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
    var_1 = 'key'
    var_2 = 'a'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dump_payload_compression_predicate. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'x'
    var_1 = 1000
    var_2 = var_0 * var_1
    var_3 = len(var_2)
    var_4 = 1
    var_5 = var_3 - var_4



# Parsed testcases at query #20
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #21
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dump_payload_compression_predicate. Retrieved 8/9 statements.


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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_payload_with_compressed_data. Retrieved 4/9 statements.
# Partially parsed test_load_payload_with_uncompressed_data. Retrieved 3/6 statements.
# Partially parsed test_load_payload_with_invalid_zlib_data. Retrieved 4/9 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = b'='
    var_3 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = b'='

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid_base64'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid_zlib_data'
    var_2 = b'='
    var_3 = b'.'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_payload_with_compressed_data. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'{"test": "data"}'

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

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
    var_2 = b'corrupted_data'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dump_payload_compression_predicate. Retrieved 8/9 statements.


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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dump_payload_compresses_when_shorter. Retrieved 8/9 statements.


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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dump_payload_compression_predicate. Retrieved 6/11 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dump_payload_compresses_when_shorter. Retrieved 8/9 statements.


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



# Parsed testcases at query #29
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #30
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dump_payload_compresses_when_shorter. Retrieved 8/9 statements.


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



# Parsed testcases at query #32
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'valid_base64_payload'
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dump_payload_compression_flag_set. Retrieved 8/9 statements.


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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_payload_without_compression_flag. Retrieved 4/5 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'valid_base64_payload'
    var_2 = b'.'
    var_3 = var_0.load_payload(var_1)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dump_payload_with_compression. Retrieved 6/8 statements.
# Partially parsed test_dump_payload_without_compression. Retrieved 4/6 statements.


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
    var_1 = 'short'
    var_2 = var_0.dump_payload(var_1)
    var_3 = b'.'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_payload_compresses_when_shorter. Retrieved 8/9 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'data'
    var_2 = 'a'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = {var_1: var_4}
    var_6 = var_0.dump_payload(var_5)
    var_7 = b'.'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_payload_compression_predicate. Retrieved 6/7 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'a'
    var_2 = 1000
    var_3 = var_1 * var_2
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_payload_with_compressed_data. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'{"test": "data"}'

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

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
    var_2 = b'invalid_zlib_data'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dump_payload_with_compression. Retrieved 9/12 statements.
# Partially parsed test_dump_payload_without_compression. Retrieved 5/7 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.dump_payload(var_3)
    var_5 = b'.'
    var_6 = 1
    var_7 = var_4[var_6:]
    var_8 = module_1.base64_decode(var_7)

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = 'a'
    var_2 = var_0.dump_payload(var_1)
    var_3 = b'.'
    var_4 = module_1.base64_decode(var_2)



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.valid_base64_encoded_data'
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.eNrzSM3JyVcozy/KSdEoLkktUivOz0vMy0xJzUnNS87P1FEozi9KLdEpzy9KLQEAAAD__w=='
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_payload_with_compressed_data. Retrieved 3/7 statements.
# Partially parsed test_load_payload_with_uncompressed_data. Retrieved 2/4 statements.
# Partially parsed test_load_payload_with_invalid_zlib_data. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = b'.'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid_base64'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid_zlib_data'
    var_2 = b'.'



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.valid_base64_payload'
    var_2 = var_0.load_payload(var_1)



# Parsed testcases at query #10
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()



# Parsed testcases at query #11
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_payload_with_compressed_data. Retrieved 3/7 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'.'
    var_2 = b'{"test": "data"}'

import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)

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
    var_2 = b'invalid_zlib_data'
    var_3 = module_1.base64_encode(var_2)
    var_4 = var_1 + var_3
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #14
#--------------------------




import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'x\x9c\xabH\xcd\xc9\xc9W(\xcf/\xcaI\x01\x00\x18\xab\x04T'
    var_2 = b'eJxrSFzL2VkpAQAYoNJN'
    var_3 = b'.'
    var_4 = var_3 + var_2
    var_5 = var_0.load_payload(var_4)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = b'eyJ0ZXN0IjogImRhdGEifQ'
    var_3 = var_2
    var_4 = var_0.load_payload(var_3)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'invalid$$$'
    var_2 = var_0.load_payload(var_1)

import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'x\x9c\xabH\xcd\xc9\xc9W(\xcf/\xcaI\x01\x00\x18\xab\x04'
    var_2 = b'eJxrSFzL2VkpAQAYoNJ'
    var_3 = b'.'
    var_4 = var_3 + var_2
    var_5 = var_0.load_payload(var_4)



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'test'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)
    assert var_3 == b'test'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_payload_with_compressed_data_that_fails_decompression. Retrieved 4/12 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'test data'
    var_2 = 2
    var_3 = b'.'



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_payload_with_valid_compressed_data. Retrieved 4/9 statements.


import src.itsdangerous.url_safe as module_0

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = b'='
    var_3 = b'.'



# Parsed testcases at query #19
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #20
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #21
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #22
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_payload_without_compression_flag. Retrieved 4/5 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'test_data'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_payload_without_compression. Retrieved 4/5 statements.


import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'test_data'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #25
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



# Parsed testcases at query #26
#--------------------------




import src.itsdangerous.url_safe as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = module_0.URLSafeSerializerMixin()
    var_1 = b'{"test": "data"}'
    var_2 = module_1.base64_encode(var_1)
    var_3 = var_0.load_payload(var_2)



