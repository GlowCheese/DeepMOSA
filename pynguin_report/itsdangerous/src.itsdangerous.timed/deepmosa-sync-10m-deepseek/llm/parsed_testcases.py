####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loads_with_str_returns_payload. Retrieved 3/5 statements.
# Partially parsed test_loads_with_bytes_returns_payload. Retrieved 3/6 statements.
# Partially parsed test_loads_with_return_timestamp_true_returns_tuple. Retrieved 4/7 statements.
# Partially parsed test_loads_with_max_age_valid_returns_payload. Retrieved 4/6 statements.
# Partially parsed test_loads_with_max_age_expired_raises_signature_expired. Retrieved 5/10 statements.
# Partially parsed test_loads_with_custom_salt_returns_payload. Retrieved 4/6 statements.
# Partially parsed test_loads_with_wrong_salt_raises_bad_signature. Retrieved 5/8 statements.
# Partially parsed test_loads_raises_bad_signature_on_first_signer_and_second_signer_fails. Retrieved 5/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 3600

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 0.1
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom1'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'test'
    var_4 = 'custom2'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'BadSignature'
    var_3 = [var_2]
    var_4 = {}
    var_5 = [var_2]
    var_6 = {}
    var_7 = b'test'
    var_8 = var_1.loads(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 6/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_returns_tuple_when_return_timestamp_true. Retrieved 7/9 statements.
# Partially parsed test_unsign_raises_bad_signature_on_invalid_signature. Retrieved 4/9 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_missing. Retrieved 6/15 statements.
# Partially parsed test_unsign_raises_signature_expired_when_age_negative. Retrieved 5/20 statements.
# Partially parsed test_unsign_returns_value_and_timestamp_when_return_timestamp_true. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    assert var_5 == b'value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = var_5[0]
    assert var_6 == b'value'
    var_7 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = b'fake'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = -1
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000
    var_3 = b'value'
    var_4 = 3600
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'data'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'bytes'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'bytes'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'string'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'string'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_missing_timestamp. Retrieved 6/11 statements.
# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 5/15 statements.
# Partially parsed test_unsign_with_invalid_signature_and_valid_timestamp. Retrieved 7/14 statements.
# Partially parsed test_unsign_with_unicode_string_input. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 86400
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'invalid'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'not_a_valid_timestamp'
    var_4 = module_1.base64_encode(var_3)
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'corrupted'
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b''
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_age_less_than_zero_raises_signature_expired. Retrieved 11/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp
    var_5 = var_4()
    var_6 = 10
    var_7 = var_5 + var_6
    var_8 = var_1.sign(var_2)
    var_9 = 0
    var_10 = var_1.unsign(var_8, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'age'
    var_13 = '< 0'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unsign_returns_bytes. Retrieved 5/6 statements.
# Partially parsed test_unsign_with_return_timestamp_returns_tuple. Retrieved 10/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = None
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_sep_not_in_result_and_sig_error_is_none. Retrieved 15/19 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.get_timestamp()
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = var_1.sep
    var_7 = module_1.want_bytes(var_6)
    var_8 = var_2 + var_7
    var_9 = var_8 + var_5
    var_10 = var_9 + var_7
    var_11 = var_2 + var_7
    var_12 = var_11 + var_5
    var_13 = var_1.sep
    var_14 = module_1.want_bytes(var_13)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unsign_valid_with_max_age_and_return_timestamp. Retrieved 6/8 statements.
# Partially parsed test_unsign_valid_no_max_age_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_expired_signature_future. Retrieved 8/11 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 9/14 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 2
    var_5 = 1
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 100
    var_3 = 'test'
    var_4 = var_1.sign(var_3)
    var_5 = 200
    var_6 = 50
    var_7 = var_1.unsign(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'age'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = b'.'
    var_6 = -1
    var_7 = var_1.unsign(var_3)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'not-a-timestamp'
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = True
    var_9 = var_1.unsign(var_7, return_timestamp=var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unsign_signature_ok_but_timestamp_is_none. Retrieved 7/18 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'\xff\xff'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Malformed timestamp'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loads_without_return_timestamp_returns_payload_only. Retrieved 5/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 5/10 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 6/12 statements.
# Partially parsed test_unsign_with_sig_error_and_malformed_timestamp. Retrieved 6/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'X'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'not-base64'
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'tampered'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_sig_error_not_none_and_ts_int_is_not_none_evaluates_to_false. Retrieved 9/18 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'bad'
    var_7 = b'dGVzdA'
    var_8 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true_returns_payload_and_timestamp. Retrieved 6/16 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_loads_returns_payload_when_return_timestamp_is_false. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = False



# Parsed testcases at query #16
#--------------------------




import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.get_timestamp()
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = var_1.sep
    var_7 = module_1.want_bytes(var_6)
    var_8 = var_2 + var_7
    var_9 = var_8 + var_5
    var_10 = var_9 + var_7
    var_11 = b'corrupted'
    var_12 = var_10 + var_11
    var_13 = var_1.unsign(var_12)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_timestamp_signer_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #18
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_exception_at_line_43_occurs_when_base64_decode_fails. Retrieved 7/17 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'!!!invalid-base64!!!'
    var_6 = 0



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret-key'
    var_3 = var_1.salt
    assert var_3 == b'itsdangerous.Signer'
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.key_derivation
    assert var_5 == 'django-concat'
    var_6 = var_1.digest_method
    var_7 = bool(var_1.digest_method is not None)
    assert var_7 is True
    var_8 = var_1.algorithm
    var_9 = bool(var_1.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm is var_0)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_32_evaluates_to_false. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_loads_returns_tuple_when_return_timestamp_is_true. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.salt
    assert var_3 == b'itsdangerous.Signer'
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.key_derivation
    assert var_5 == 'django-concat'
    var_6 = var_1.digest_method
    var_7 = bool(var_1.digest_method is not None)
    assert var_7 is True
    var_8 = var_1.algorithm
    var_9 = bool(var_1.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes_secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'bytes_secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'none'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'none'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_with_algorithm. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_separator_not_in_base64_alphabet. Retrieved 1/5 statements.
# Partially parsed test_timestamp_signer_separator_not_in_base64_alphabet_bytes. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'!'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'!'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '!'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'!'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new'



# Parsed testcases at query #25
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = -1
    var_7 = var_3[var_6:]
    var_8 = b'x'
    var_9 = var_7 != var_8
    var_10 = b'y'
    var_11 = var_8 if var_9 else var_10
    var_12 = var_5 + var_11
    var_13 = var_1.unsign(var_12)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_loads_returns_payload_when_signature_valid_and_no_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 6/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_2 + var_4
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.sep.badtimestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret1'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'secret2'
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = 'test'
    var_5 = var_1.sign(var_4)
    var_6 = var_3.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_loads_returns_payload_when_no_max_age_and_no_return_timestamp. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_true. Retrieved 4/7 statements.
# Partially parsed test_loads_raises_signature_expired_when_max_age_exceeded. Retrieved 5/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 0.1
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_unsign_valid_signature_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/16 statements.
# Partially parsed test_unsign_with_negative_age. Retrieved 5/20 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test_value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test_value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid_data'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = b'malformed_timestamp'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'no_separator_here'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 10000
    var_3 = b'test_value'
    var_4 = 3600
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_timestamp_signer_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_with_custom_algorithm. Retrieved 1/6 statements.
# Partially parsed test_timestamp_signer_with_sep_in_base64_alphabet. Retrieved 1/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method.__name__
    assert var_7 == 'sha1'
    var_8 = var_1.algorithm
    var_9 = bool(var_1.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True

def test_case_0():
    var_0 = 'secret-key'
    var_1 = bool(False)
    assert var_1 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-key'])
    assert var_3 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_loads_without_max_age_and_without_return_timestamp. Retrieved 3/5 statements.
# Partially parsed test_loads_with_max_age_and_return_timestamp. Retrieved 5/8 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_max_age_expired. Retrieved 5/11 statements.
# Partially parsed test_loads_with_string_input. Retrieved 3/6 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/5 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 3600
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 0.01
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'testsalt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'test'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.salt
    assert var_3 == b'itsdangerous.Signer'
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.key_derivation
    assert var_5 == 'django-concat'
    var_6 = var_1.digest_method
    var_7 = bool(var_1.digest_method is not None)
    assert var_7 is True
    var_8 = var_1.algorithm
    var_9 = bool(var_1.algorithm is not None)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old_key'
    var_1 = b'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 7/9 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 15/19 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = var_5[0]
    assert var_6 == b'value'
    var_7 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_2 + var_4
    var_6 = b'invalidsig'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_2 + var_4
    var_6 = b'badts'
    var_7 = var_5 + var_6
    var_8 = var_1.sep
    var_9 = module_1.want_bytes(var_8)
    var_10 = var_7 + var_9
    var_11 = var_1.sep
    var_12 = module_1.want_bytes(var_11)
    var_13 = var_2 + var_12
    var_14 = var_13 + var_6
    var_15 = bool(False)
    assert var_15 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_missing_timestamp. Retrieved 3/9 statements.
# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 4/16 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'bad'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'malformed'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_exception_handling_at_line_52. Retrieved 8/18 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = module_1.int_to_bytes(var_4)
    var_6 = module_1.base64_encode(var_5)
    var_7 = b'test'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_unsign_predicate_at_line_43_evaluates_to_false. Retrieved 6/20 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'!!invalid_base64!!'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 7/18 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'!!!'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_unsign_valid_return_timestamp. Retrieved 6/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = var_5[0]
    assert var_6 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0.1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Signature age'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'invalid'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = var_3.rsplit(var_5, var_6)[var_4]
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'timestamp missing'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_loads_return_timestamp_false. Retrieved 6/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_unsign_valid_signature_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature_raises. Retrieved 7/12 statements.
# Partially parsed test_unsign_negative_age_raises. Retrieved 7/12 statements.
# Partially parsed test_unsign_malformed_timestamp_raises. Retrieved 8/13 statements.
# Partially parsed test_unsign_bad_signature_with_timestamp_raises. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'hello'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'hello'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = 10
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = 10
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = b'.invalid'
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b'hello'
    var_5 = b'world'
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'bad_signature'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b''
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_line_52_false. Retrieved 6/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret-key'
    var_3 = var_1.secret_keys
    var_4 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_4 is True
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_1.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_1.digest_method
    var_9 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret-key'
    var_3 = var_1.secret_keys
    var_4 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm is var_0)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #44
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.secret_keys
    var_4 = bool(var_1.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.sep
    assert var_6 == b'.'
    var_7 = var_1.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_1.digest_method
    var_9 = var_1.algorithm
    var_10 = bool(var_1.algorithm is not None)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 5/11 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 5/15 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'bad'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'notbase64'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 6/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_unsign_returns_value_and_timestamp. Retrieved 8/9 statements.
# Partially parsed test_unsign_with_missing_timestamp_raises_bad_time_signature. Retrieved 4/9 statements.
# Partially parsed test_unsign_with_malformed_timestamp_raises_bad_time_signature. Retrieved 4/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5[0]
    assert var_7 == b'test'
    var_8 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'invalid'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'x'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'badts'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 86400
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp_true_returns_tuple. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 11/15 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_2 + var_4
    var_6 = b'invalid'
    var_7 = var_5 + var_6
    var_8 = var_1.sep
    var_9 = module_1.want_bytes(var_8)
    var_10 = var_7 + var_9
    var_11 = bool(False)
    assert var_11 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = 1
    var_8 = var_3.rsplit(var_6, var_7)[var_4]
    var_9 = var_1.unsign(var_8)
    var_10 = bool(False)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = -1
    var_7 = var_3[var_6:]
    var_8 = b'x'
    var_9 = var_7 != var_8
    var_10 = b'y'
    var_11 = var_8 if var_9 else var_10
    var_12 = var_5 + var_11
    var_13 = var_1.unsign(var_12)
    var_14 = bool(False)
    assert var_14 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_predicate_at_line_52_evaluates_to_true. Retrieved 6/19 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 999999999999999999999999999999
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_timestamp_missing_raises_bad_time_signature. Retrieved 5/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = b'invalid_timestamp'
    var_4 = b'fakesignature'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Malformed timestamp'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_line52_evaluates_to_false. Retrieved 6/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_max_age_exceeded. Retrieved 7/10 statements.
# Partially parsed test_unsign_with_negative_age. Retrieved 8/15 statements.
# Partially parsed test_unsign_with_missing_timestamp. Retrieved 3/9 statements.
# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 3/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    assert var_5 == b'test_value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test_value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 2
    var_5 = 1
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp
    var_3 = 100
    var_4 = 'test_value'
    var_5 = var_1.sign(var_4)
    var_6 = 3600
    var_7 = var_1.unsign(var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid.data'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value.sep.badtimestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value.sep.invalidtimestamp'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_loads_with_string_input. Retrieved 3/5 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_with_max_age_and_valid. Retrieved 4/6 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_max_age_and_return_timestamp. Retrieved 5/8 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_expired_signature. Retrieved 5/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'hello'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'hello'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'hello'
    var_3 = 3600

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'hello'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'hello'
    var_3 = 3600
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'hello'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'hello'
    var_3 = 0.1
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid.data'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 6/12 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 7/16 statements.
# Partially parsed test_unsign_negative_age. Retrieved 10/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = 'Expected SignatureExpired'
    var_7 = AssertionError(var_6)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = 'Expected BadSignature'
    var_5 = AssertionError(var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'fakesignature'
    var_4 = 'Expected BadTimeSignature'
    var_5 = AssertionError(var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'invalidtimestamp'
    var_4 = b'fakesignature'
    var_5 = 'Expected BadTimeSignature'
    var_6 = AssertionError(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 0
    var_3 = 'test'
    var_4 = var_1.sign(var_3)
    var_5 = 100
    var_6 = 50
    var_7 = var_1.unsign(var_4, var_6)
    var_8 = 'Expected SignatureExpired'
    var_9 = AssertionError(var_8)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_unsign_ts_int_is_none_raises_bad_time_signature. Retrieved 15/21 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.get_timestamp()
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = var_1.sep
    var_7 = module_1.want_bytes(var_6)
    var_8 = var_2 + var_7
    var_9 = var_8 + var_5
    var_10 = var_9 + var_7
    var_11 = var_2 + var_7
    var_12 = var_11 + var_5
    var_13 = -1
    var_14 = b'X'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_timestamp_signer_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_with_custom_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm
    var_10 = bool(var_1.algorithm is not None)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'The given separator cannot be used'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'concat'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new-key'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp_true_returns_tuple. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp_raises_bad_time_signature. Retrieved 5/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -5
    var_5 = var_3[:var_4]
    var_6 = b'invalid'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalidtimestamp.invalidsig'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_sep_in_result_with_sig_error_false. Retrieved 6/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = b'|'
    var_5 = 1
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_predicate_at_line_52_evaluates_to_true. Retrieved 18/35 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = module_1.want_bytes(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = module_1.int_to_bytes(var_4)
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.sep
    var_8 = module_1.want_bytes(var_7)
    var_9 = var_3 + var_8
    var_10 = var_9 + var_6
    var_11 = var_10 + var_8
    var_12 = var_3 + var_8
    var_13 = var_12 + var_6
    var_14 = b'123456789'
    var_15 = b'='
    var_16 = var_3 + var_8
    var_17 = var_3 + var_8
    var_18 = 'Malformed timestamp'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_loads_without_max_age_and_return_timestamp. Retrieved 3/5 statements.
# Partially parsed test_loads_with_max_age_valid. Retrieved 4/6 statements.
# Partially parsed test_loads_with_max_age_expired. Retrieved 5/11 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_max_age_and_return_timestamp. Retrieved 5/8 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/8 statements.
# Partially parsed test_loads_with_custom_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_with_wrong_salt. Retrieved 5/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 3600

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 0.01
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 3600
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'custom'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'custom'
    var_4 = 'wrong'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_unsign_with_valid_signed_value_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_missing_timestamp. Retrieved 5/9 statements.
# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 4/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 'test'
    var_5 = var_1.sign(var_4)
    var_6 = 3600
    var_7 = var_1.unsign(var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 'test'
    var_5 = var_1.sign(var_4)
    var_6 = 3600
    var_7 = var_1.unsign(var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'invalid'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true_returns_tuple. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #65
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_loads_returns_payload_when_valid. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_true. Retrieved 4/8 statements.
# Partially parsed test_loads_raises_signature_expired_when_max_age_exceeded. Retrieved 5/10 statements.
# Partially parsed test_loads_raises_bad_signature_when_salt_mismatch. Retrieved 5/8 statements.
# Partially parsed test_loads_accepts_bytes_input. Retrieved 3/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 0.01
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'salt1'
    var_4 = 'salt2'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_unsign_returns_tuple_with_timestamp. Retrieved 8/10 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_missing. Retrieved 5/9 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_malformed. Retrieved 7/15 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5[0]
    assert var_7 == b'test'
    var_8 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'zzz'
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    assert var_5 == b'test'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 4/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'!!!invalid!!!'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_loads_no_return_timestamp_with_valid_data. Retrieved 6/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = False



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '-'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret-key'
    var_3 = var_1.salt
    assert var_3 == b'itsdangerous.Signer'
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.key_derivation
    assert var_5 == 'django-concat'
    var_6 = var_1.digest_method
    var_7 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_false. Retrieved 8/11 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = 1



# Parsed testcases at query #73
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_loads_return_timestamp_true_returns_tuple. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_predicate_at_line_52_evaluates_to_false. Retrieved 5/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_unsign_timestamp_to_datetime_raises_value_error. Retrieved 10/25 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 2
    var_5 = 63
    var_6 = var_4 ** var_5
    var_7 = '>Q'
    var_8 = b'='
    var_9 = b'test'
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Malformed timestamp'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_ts_int_is_none_after_bad_timestamp. Retrieved 23/29 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_1.get_timestamp()
    var_6 = module_1.int_to_bytes(var_5)
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2 + var_4
    var_9 = var_8 + var_7
    var_10 = var_9 + var_4
    var_11 = var_2 + var_4
    var_12 = var_11 + var_7
    var_13 = b'!!invalid!!'
    var_14 = b'x'
    var_15 = 9
    var_16 = var_14 * var_15
    var_17 = module_1.base64_encode(var_16)
    var_18 = var_2 + var_4
    var_19 = var_18 + var_17
    var_20 = var_19 + var_4
    var_21 = var_2 + var_4
    var_22 = var_21 + var_17
    var_23 = 'Malformed timestamp'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_unsign_with_bad_signature_and_malformed_timestamp_does_not_raise_bad_time_signature_from_exception. Retrieved 8/20 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.get_timestamp()
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = -1
    var_7 = b'X'
    var_8 = 'Malformed timestamp'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_loads_returns_tuple_when_return_timestamp_true. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_loads_returns_payload_when_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_true. Retrieved 4/8 statements.
# Partially parsed test_loads_raises_signature_expired_when_max_age_exceeded. Retrieved 5/10 statements.
# Partially parsed test_loads_with_salt_uses_correct_signer. Retrieved 4/6 statements.
# Partially parsed test_loads_raises_bad_signature_when_wrong_salt. Retrieved 5/8 statements.
# Partially parsed test_loads_with_string_input_works. Retrieved 3/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid.data'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 0.1
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt1'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'test'
    var_4 = 'wrong_salt'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loads_basic_payload. Retrieved 3/5 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_max_age_valid. Retrieved 4/6 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 3600

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = -1
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads_unsafe(var_2)
    var_4 = var_3[0]
    assert var_4 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '-'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'-'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_with_return_timestamp_true_returns_tuple. Retrieved 8/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.==invalid=='
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5[0]
    var_8 = bool(var_5[0] == var_2)
    assert var_8 is True
    var_9 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.badtimestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'bad'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.sep.malformed'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b''
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm
    var_10 = bool(var_1.algorithm is not None)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 4/9 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 5/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'!!!'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = 'test_value'
    var_4 = var_1.sign(var_3)
    var_5 = b'.'
    var_6 = module_1.Signer(var_0)
    var_7 = b'test_value'
    var_8 = var_6.sign(var_7)
    var_9 = var_1.unsign(var_8)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_line49_evaluates_to_false. Retrieved 5/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 7/9 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 6/10 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 13/17 statements.
# Partially parsed test_unsign_return_timestamp_false_by_default. Retrieved 5/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = var_5[0]
    assert var_6 == b'test'
    var_7 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = module_1.want_bytes(var_2)
    var_4 = var_1.sep
    var_5 = module_1.want_bytes(var_4)
    var_6 = b'abc'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_3 + var_5
    var_9 = var_8 + var_7
    var_10 = var_9 + var_5
    var_11 = var_3 + var_5
    var_12 = var_11 + var_7
    var_13 = bool(False)
    assert var_13 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1000
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_loads_signature_expired_raised. Retrieved 5/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 0.1
    var_4 = 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_ts_int_not_none_after_exception. Retrieved 4/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'!'
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_timestamp_signer_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_with_custom_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.secret_keys
    var_4 = bool(var_1.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_1.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_1.digest_method
    var_9 = bool(var_1.digest_method is not None)
    assert var_9 is True
    var_10 = var_1.algorithm
    var_11 = bool(var_1.algorithm is not None)
    assert var_11 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'django-concat'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 4/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = b'invalid_timestamp'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 4/18 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'!!!invalid_base64!!!'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unsign_predicate_true. Retrieved 6/16 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'invalid_base64!!!'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_age_negative_raises_signature_expired. Retrieved 11/29 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 10
    var_5 = 5
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = 5
    var_8 = var_1.unsign(var_3, var_7)
    var_9 = 100
    var_10 = var_1.unsign(var_3, var_9)
    assert var_10 == b'test'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 6/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'='
    var_5 = b'invalid_base64!'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unsign_valid_signature_with_max_age_expired. Retrieved 7/10 statements.
# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp_raises_bad_time_signature. Retrieved 3/9 statements.
# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 4/14 statements.
# Partially parsed test_unsign_valid_signature_with_negative_age. Retrieved 7/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 2
    var_5 = 1
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'not-a-timestamp'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Malformed timestamp'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 3600
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'age'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unsign_with_sig_error_and_none_ts_int. Retrieved 4/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'invalid!base64'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unsign_with_sig_error_and_ts_int_none. Retrieved 5/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = b'invalid_base64'
    var_4 = b'badsig'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_false. Retrieved 7/17 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = b'!!invalid!!'



# Parsed testcases at query #22
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp_return. Retrieved 7/9 statements.
# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 10/19 statements.
# Partially parsed test_unsign_missing_timestamp_raises_bad_time_signature. Retrieved 9/16 statements.
# Partially parsed test_unsign_invalid_signature_raises_bad_time_signature. Retrieved 10/18 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = var_5[0]
    assert var_6 == b'test'
    var_7 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = 1
    var_8 = 0
    var_9 = b'invalid_timestamp'
    var_10 = bool(False)
    assert var_10 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = 1
    var_8 = 0
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = 1
    var_8 = 0
    var_9 = b'tampered'
    var_10 = bool(False)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = None
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 12/16 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = b'not_a_valid_timestamp'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_2 + var_4
    var_8 = var_7 + var_6
    var_9 = var_8 + var_4
    var_10 = var_2 + var_4
    var_11 = var_10 + var_6



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_custom_algorithm. Retrieved 1/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'



# Parsed testcases at query #27
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_unsign_with_bad_signature_and_timestamp_that_raises_overflow_error. Retrieved 9/16 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 1
    var_4 = 62
    var_5 = var_3 << var_4
    var_6 = module_1.int_to_bytes(var_5)
    var_7 = module_1.base64_encode(var_6)
    var_8 = b'invalid'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm
    var_10 = bool(var_1.algorithm is not None)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_bad_time_signature_on_overflow_error. Retrieved 11/24 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 2
    var_5 = 63
    var_6 = var_4 ** var_5
    var_7 = 1
    var_8 = var_6 - var_7
    var_9 = module_1.int_to_bytes(var_8)
    var_10 = module_1.base64_encode(var_9)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_line_52_evaluates_to_false. Retrieved 7/16 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_timestamp()
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_constructor_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_constructor_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'A'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 8/14 statements.
# Partially parsed test_unsign_returns_bytes. Retrieved 5/6 statements.
# Partially parsed test_unsign_returns_tuple_when_return_timestamp_true. Retrieved 10/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = var_3.rsplit(var_5, var_6)[var_4]
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = b'invalid'
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = var_5[var_4]



# Parsed testcases at query #35
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp_raises_bad_time_signature. Retrieved 3/6 statements.
# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 7/14 statements.
# Partially parsed test_unsign_with_max_age_and_return_timestamp. Retrieved 6/8 statements.
# Partially parsed test_unsign_unicode_value. Retrieved 5/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1000000
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'invalid'
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b''
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b''

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'héllo'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = 'test'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_timestamp_signer_constructor_default. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'newer'
    var_2 = 'newest'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.TimestampSigner(var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'old', b'newer', b'newest'])
    assert var_6 is True
    var_7 = var_4.secret_key
    assert var_7 == b'newest'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '+'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_unsign_ts_int_is_none_raises_bad_time_signature. Retrieved 7/22 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'test|invalid_timestamp'
    var_5 = b'test|invalid_timestamp'
    var_6 = var_1.unsign(var_3)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_with_future_timestamp. Retrieved 7/12 statements.
# Partially parsed test_unsign_missing_separator. Retrieved 6/11 statements.
# Partially parsed test_unsign_non_timestamp_data. Retrieved 6/11 statements.
# Partially parsed test_unsign_with_string_input. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'X'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 3600
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'test'
    var_5 = b'bad'
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -10
    var_5 = var_3[:var_4]
    var_6 = b'bad'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'test'
    var_5 = b'data'
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 7/9 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 6/10 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 7/20 statements.
# Partially parsed test_unsign_valid_with_max_age_and_timestamp. Retrieved 8/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'hello'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = var_5[0]
    assert var_6 == b'hello'
    var_7 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'X'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'not-a-timestamp'
    var_6 = 0
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'hello'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = True
    var_6 = var_1.unsign(var_3, var_4, var_5)
    var_7 = var_6[0]
    assert var_7 == b'hello'
    var_8 = var_6[var_5]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret-key'
    var_3 = var_1.salt
    assert var_3 == b'itsdangerous.Signer'
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.key_derivation
    assert var_5 == 'django-concat'
    var_6 = var_1.digest_method
    var_7 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_age_less_than_zero_evaluates_true. Retrieved 7/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = 'age -1'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 6/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.sep
    var_3 = b'test'
    var_4 = b'!!!invalid!!!'
    var_5 = b'fakesig'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_age_less_than_zero_raises_signature_expired. Retrieved 6/29 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp
    var_3 = b'dummy'
    var_4 = 100
    var_5 = 50
    var_6 = 'Signature age'
    var_7 = '0 seconds'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_base64_decode_raises_exception_on_invalid_input. Retrieved 4/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = b'!!!invalid_base64!!!'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.salt
    assert var_3 == b'itsdangerous.Signer'
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.key_derivation
    assert var_5 == 'django-concat'
    var_6 = var_1.digest_method
    var_7 = bool(var_1.digest_method is var_1.default_digest_method)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 8/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test-value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = True
    var_6 = var_2.unsign(var_4, return_timestamp=var_5)
    var_7 = var_6[0]
    assert var_7 == b'test-value'
    var_8 = var_6[var_5]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = -1
    var_6 = var_2.unsign(var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = 0
    var_6 = var_2.unsign(var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'invalid-value'
    var_4 = var_2.unsign(var_3)
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = b'test-value.'
    var_4 = b'invalid'
    var_5 = module_1.base64_encode(var_4)
    var_6 = var_3 + var_5
    var_7 = var_2.unsign(var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = -1
    var_6 = var_4[:var_5]
    var_7 = b'X'
    var_8 = var_6 + var_7
    var_9 = var_2.unsign(var_8)
    var_10 = bool(False)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = b'invalid-value'
    var_4 = var_2.unsign(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new_secret'
    var_5 = var_3.secret_keys
    var_6 = bool(var_3.secret_keys == [b'old_secret', b'new_secret'])
    assert var_6 is True



# Parsed testcases at query #49
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_sep_in_base64_alphabet_raises. Retrieved 1/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.Signer'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm
    var_10 = bool(var_1.algorithm is not None)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

def test_case_0():
    var_0 = 'secret-key'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_at_line_77_evaluates_to_true. Retrieved 10/35 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 1000
    var_6 = 0
    var_7 = 'different_secret'
    var_8 = module_0.TimestampSigner(var_7)
    var_9 = 10



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_ts_int_is_none_after_base64_decode_failure. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = b'!!invalid_base64!!'
    var_6 = 'Malformed timestamp'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_unsign_with_valid_timestamp_after_exception_handling. Retrieved 7/19 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'not_base64!!'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_timestamp_signer_with_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.secret_keys
    var_4 = bool(var_1.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.salt
    assert var_6 == b'itsdangerous.Signer'
    var_7 = var_1.key_derivation
    assert var_7 == 'django-concat'
    var_8 = var_1.digest_method
    var_9 = bool(var_1.digest_method is not None)
    assert var_9 is True
    var_10 = var_1.algorithm
    var_11 = bool(var_1.algorithm is not None)
    assert var_11 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm is var_0)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'key2'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_timestamp_signer_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_with_algorithm. Retrieved 1/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'
    var_3 = var_1.salt
    assert var_3 == b'itsdangerous.Signer'
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.key_derivation
    assert var_5 == 'django-concat'
    var_6 = var_1.digest_method

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_key
    assert var_2 == b'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new_key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'+'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_unsign_valid_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 6/11 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'x'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = b'invalidsig'
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'not_base64'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_unsign_valid_signature_with_max_age. Retrieved 8/10 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/11 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 3/10 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/17 statements.
# Partially parsed test_unsign_bad_signature_with_timestamp. Retrieved 5/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5[0]
    assert var_7 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0.01
    var_5 = 3600
    var_6 = True
    var_7 = var_1.unsign(var_3, var_5, var_6)
    var_8 = var_7[0]
    assert var_8 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0.01
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'invalid'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'MTIzNDU2Nzg5'
    var_4 = b'badsig'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = ''
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = var_5[0]
    assert var_6 == b''



# Parsed testcases at query #58
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 3/9 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/16 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'x'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'invalid'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_timestamp_signer_with_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_with_algorithm. Retrieved 1/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.Signer'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm
    var_10 = bool(var_1.algorithm is not None)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'bytes-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'bytes-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



