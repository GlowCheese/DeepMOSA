####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret_key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unsign_valid_signature_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/14 statements.
# Partially parsed test_unsign_negative_age. Retrieved 8/13 statements.


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
    var_4 = b'invalid'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)

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
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = signed.rsplit(var_5, var_6)[var_4]
    var_8 = var_1.unsign(var_7)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'not-a-timestamp'

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp
    var_5 = 100
    var_6 = 50
    var_7 = var_1.unsign(var_3, var_6)

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_negative_age_raises_signature_expired. Retrieved 7/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 10
    var_6 = var_1.unsign(var_3, var_5)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_all_params. Retrieved 4/10 statements.
# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_custom_algorithm. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = 'mykey'
    var_1 = b'mysalt'
    var_2 = b':'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'newer'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'-'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'+'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.NoneAlgorithm()
    var_2 = module_1.TimestampSigner(var_0, algorithm=var_1)
    var_3 = var_2.algorithm



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_timestamp_signer_with_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'none'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_all_params. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = '|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp_raises_bad_time_signature. Retrieved 4/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = b'!!!invalid_base64!!!'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.digest_method
    var_3 = callable(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_secret'
    var_1 = 'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_line49_false_when_ts_int_is_none_and_sig_error_not_none. Retrieved 6/22 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp
    var_3 = b'test'
    var_4 = 'ascii'
    var_5 = b'not-valid-base64'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_age_less_than_zero_raises_signature_expired. Retrieved 7/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = 10
    var_6 = var_1.unsign(var_3, var_5)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_age_negative_raises_signature_expired. Retrieved 9/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = 10
    var_6 = var_4 - var_5
    var_7 = 5
    var_8 = var_1.unsign(var_3, var_7)



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unsign_signature_ok_timestamp_malformed_ts_int_none. Retrieved 7/25 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 1234567890
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = str(var_0)
    assert var_6 == 'Malformed timestamp'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_future_timestamp. Retrieved 15/19 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 12/16 statements.


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
    var_2 = b'invalid|data|signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'nodata'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = module_1.want_bytes(var_2)
    var_4 = 9999999999
    var_5 = module_1.int_to_bytes(var_4)
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.sep
    var_8 = module_1.want_bytes(var_7)
    var_9 = var_3 + var_8
    var_10 = var_9 + var_6
    var_11 = var_10 + var_8
    var_12 = var_3 + var_8
    var_13 = var_12 + var_6
    var_14 = 100

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = module_1.want_bytes(var_2)
    var_4 = b'not-a-timestamp'
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = var_3 + var_6
    var_8 = var_7 + var_4
    var_9 = var_8 + var_6
    var_10 = var_3 + var_6
    var_11 = var_10 + var_4



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unsign_exception_in_timestamp_decoding_leaves_ts_int_none. Retrieved 7/19 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'!!!invalid!!!'
    var_6 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loads_returns_payload_when_no_return_timestamp. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_true. Retrieved 4/8 statements.
# Partially parsed test_loads_raises_signature_expired_when_max_age_exceeded. Retrieved 5/10 statements.
# Partially parsed test_loads_uses_salt_properly. Retrieved 4/6 statements.
# Partially parsed test_loads_raises_bad_signature_with_wrong_salt. Retrieved 5/8 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'mysalt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'salt1'
    var_4 = 'salt2'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_value_error_on_invalid_timestamp. Retrieved 7/15 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'invalid_base64'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 7/10 statements.
# Partially parsed test_unsign_negative_age. Retrieved 8/13 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

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
    var_4 = b'bad'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = b'malformed'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = module_0.TimestampSigner(var_0)
    var_5 = 100
    var_6 = 10
    var_7 = var_4.unsign(var_3, var_6)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_loads_returns_tuple_when_return_timestamp_is_true. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_age_less_than_zero_raises_signature_expired. Retrieved 9/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = 100
    var_6 = var_4 + var_5
    var_7 = 50
    var_8 = var_1.unsign(var_3, var_7)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_line32_false_when_sep_in_result_and_sig_error_none. Retrieved 4/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'MTIzNDU2Nzg5'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_unsign_predicate_line49_false_when_ts_int_none_and_sig_error_not_none. Retrieved 7/17 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'invalid_timestamp'
    var_4 = b'value'
    var_5 = b'!!'
    var_6 = b'badsig'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_max_age_expired. Retrieved 7/12 statements.
# Partially parsed test_unsign_negative_age. Retrieved 7/12 statements.
# Partially parsed test_unsign_return_timestamp_with_max_age. Retrieved 6/8 statements.


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
    var_4 = 7200
    var_5 = 3600
    var_6 = var_1.unsign(var_3, var_5)

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
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_2 + var_4
    var_6 = b'bad_timestamp'
    var_7 = var_5 + var_6
    var_8 = var_1.sep
    var_9 = module_1.want_bytes(var_8)
    var_10 = var_7 + var_9
    var_11 = b'signature'
    var_12 = var_10 + var_11
    var_13 = var_1.unsign(var_12)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 7200
    var_5 = 3600
    var_6 = var_1.unsign(var_3, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_algorithm. Retrieved 2/6 statements.
# Partially parsed test_unsign_with_digest_method. Retrieved 2/5 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 1000000
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = signed.split(var_5)[var_4]
    var_7 = var_6 + var_5
    var_8 = b'invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)

def test_case_0():
    var_0 = 'secret'
    var_1 = 'value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = 'value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = 'value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'value'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'value'



# Parsed testcases at query #26
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_all_parameters. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = b'custom_salt'
    var_4 = b'|'
    var_5 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'string_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_unsign_valid_signature_return_timestamp. Retrieved 8/9 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 5/10 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 6/12 statements.


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
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'value'

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 1

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'abc'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_unsign_with_bad_signature_and_malformed_timestamp_does_not_raise_bad_time_signature_from_timestamp_conversion. Retrieved 8/15 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.get_timestamp()
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = b'wrongsignature'
    var_7 = False



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_custom_algorithm. Retrieved 1/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp_true_returns_tuple. Retrieved 7/9 statements.
# Partially parsed test_unsign_with_malformed_timestamp_raises_bad_time_signature. Retrieved 8/14 statements.
# Partially parsed test_unsign_with_unicode_string_returns_bytes. Retrieved 4/6 statements.


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
    var_6 = var_5[var_4]

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
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
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
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = b'invalid'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1000
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_timestamp_signer_with_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'my-secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'my-secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_valid_timestamp_should_not_set_ts_int_to_none. Retrieved 14/24 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_timestamp()
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = module_1.want_bytes(var_5)
    var_7 = var_1.sep
    var_8 = module_1.want_bytes(var_7)
    var_9 = var_2 + var_8
    var_10 = var_9 + var_6
    var_11 = var_10 + var_8
    var_12 = var_2 + var_8
    var_13 = var_12 + var_6



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_52_evaluates_to_true. Retrieved 9/23 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 0
    var_3 = module_1.int_to_bytes(var_2)
    var_4 = module_1.base64_encode(var_3)
    var_5 = b'test_value'
    var_6 = ()
    var_7 = 'test'
    var_8 = ValueError(var_7)



# Parsed testcases at query #35
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 5/9 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 7/15 statements.


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
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'!!!'



# Parsed testcases at query #37
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_for_malformed_timestamp_when_sig_error_is_not_none. Retrieved 6/16 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'test'
    var_5 = b'!!!invalid-base64!!!'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_unsign_returns_tuple_with_timestamp_when_return_timestamp_true. Retrieved 6/7 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_missing. Retrieved 6/11 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_malformed. Retrieved 4/14 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_decode_fails_with_signature_error. Retrieved 5/12 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b'x'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 1000000
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = b'zzzz'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = b'!!!!'
    var_4 = b'invalidsig'



# Parsed testcases at query #40
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = signed_value.split(var_5)[var_4]
    var_7 = b'extra'
    var_8 = var_6 + var_7
    var_9 = var_1.unsign(var_8)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_unsign_signature_ok_timestamp_malformed_ts_int_none. Retrieved 4/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'!!invalid!!'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_max_age_exceeded. Retrieved 7/10 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 4/10 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 5/13 statements.
# Partially parsed test_unsign_string_input. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test-value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test-value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = 0.1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = b'not-base64'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = ''
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b''



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_age_less_than_zero_raises_signature_expired. Retrieved 14/17 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp
    var_5 = -1
    var_6 = b'.'
    var_7 = signed_value.split(var_6)[var_5]
    var_8 = module_1.base64_decode(var_7)
    var_9 = module_1.bytes_to_int(var_8)
    var_10 = 1
    var_11 = var_9 - var_10
    var_12 = 3600
    var_13 = var_1.unsign(var_3, var_12)



# Parsed testcases at query #45
#--------------------------




import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = module_1.want_bytes(var_2)
    var_4 = 2
    var_5 = 63
    var_6 = var_4 ** var_5
    var_7 = module_1.int_to_bytes(var_6)
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.sep
    var_10 = module_1.want_bytes(var_9)
    var_11 = var_3 + var_10
    var_12 = var_11 + var_8
    var_13 = var_12 + var_10
    var_14 = b'badsignature'
    var_15 = var_13 + var_14
    var_16 = var_1.unsign(var_15)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)



# Parsed testcases at query #47
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'!'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'none'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_unsign_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_negative_age. Retrieved 9/11 statements.
# Partially parsed test_unsign_return_timestamp_type. Retrieved 5/7 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = 100
    var_6 = var_4 - var_5
    var_7 = 50
    var_8 = var_1.unsign(var_3, var_7)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.sep.invalid'
    var_3 = var_1.unsign(var_2)

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
    var_2 = b'test.sep.invalid'
    var_3 = var_1.unsign(var_2)

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
    var_2 = b'test.sep.data'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp_raises_bad_time_signature. Retrieved 8/15 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = -4



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_at_line_52_evaluates_to_false. Retrieved 10/20 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.sep
    var_5 = module_1.want_bytes(var_4)
    var_6 = 1
    var_7 = 0
    var_8 = module_1.int_to_bytes(var_7)
    var_9 = module_1.base64_encode(var_8)



# Parsed testcases at query #52
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes_secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'str_secret'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_unsign_returns_tuple_with_timestamp_when_return_timestamp_true. Retrieved 5/7 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_missing. Retrieved 7/12 statements.
# Partially parsed test_unsign_raises_signature_expired_when_age_negative. Retrieved 7/9 statements.
# Partially parsed test_unsign_raises_bad_time_signature_on_malformed_timestamp. Retrieved 10/14 statements.
# Partially parsed test_unsign_returns_value_when_signature_valid_with_string_input. Retrieved 4/6 statements.


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
    var_2 = b'test.invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = b'invalid'

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
    var_4 = 0
    var_5 = 3600
    var_6 = var_1.unsign(var_3, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'.'
    var_4 = b'!!invalid!!'
    var_5 = var_2 + var_3
    var_6 = var_5 + var_4
    var_7 = var_6 + var_3
    var_8 = var_2 + var_3
    var_9 = var_8 + var_4

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b''
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_timestamp_signer_unsign_age_negative. Retrieved 8/16 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp
    var_5 = 1
    var_6 = 100
    var_7 = var_1.unsign(var_3, var_6)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 4/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'invalid_timestamp'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes-secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = 'Expected ValueError for separator in base64 alphabet'
    var_4 = AssertionError(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)



# Parsed testcases at query #59
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_unsign_predicate_line43_false. Retrieved 7/20 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'not-a-number'
    var_6 = module_1.base64_encode(var_5)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_ts_int_is_none_when_base64_decode_raises_exception. Retrieved 4/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'!!!invalid_base64!!!'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_unsign_valid_value_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp_raises_bad_time_signature. Retrieved 5/9 statements.
# Partially parsed test_unsign_invalid_timestamp_raises_bad_time_signature. Retrieved 7/15 statements.
# Partially parsed test_unsign_bad_signature_without_timestamp_raises_bad_signature. Retrieved 7/15 statements.
# Partially parsed test_unsign_negative_age_raises_signature_expired. Retrieved 7/9 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'invalid'

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'bad'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 100
    var_6 = var_1.unsign(var_3, var_5)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '1'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '-'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_unsign_with_valid_signed_value_and_no_max_age_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_invalid_signature_and_missing_timestamp_raises_bad_signature. Retrieved 4/9 statements.
# Partially parsed test_unsign_with_invalid_signature_and_valid_timestamp_raises_bad_time_signature. Retrieved 3/12 statements.
# Partially parsed test_unsign_with_negative_age_raises_signature_expired. Retrieved 4/17 statements.
# Partially parsed test_unsign_with_malformed_timestamp_raises_bad_time_signature. Retrieved 4/12 statements.


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
    var_2 = b'test'
    var_3 = b'invalid'

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
    var_3 = var_1.unsign(var_2)

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
    var_2 = 100
    var_3 = b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'notbase64'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = None
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_all_params. Retrieved 6/9 statements.
# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_default_algorithm. Retrieved 3/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = b'custom-salt'
    var_4 = b'|'
    var_5 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'my-secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'my-secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'none'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired. Retrieved 7/10 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 7/12 statements.


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
    var_4 = 0.1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b'x'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = b'invaliddata'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b''
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'my-secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp_true. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_string_input. Retrieved 5/7 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalidsig'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.sep.malformedtimestamp.sep.sig'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 86400
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
    var_4 = 1000000
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_32_evaluates_to_false. Retrieved 5/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unsign_valid_signature_return_timestamp. Retrieved 5/7 statements.


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
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)

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
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = signed.rsplit(var_5, var_6)[var_4]
    var_8 = b'.signature'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = signed.rsplit(var_5, var_6)[var_4]
    var_8 = b'.bad-timestamp'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 9/20 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'\x00'
    var_6 = 8
    var_7 = var_5 * var_6
    var_8 = module_1.base64_encode(var_7)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loads_returns_payload_without_timestamp. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp. Retrieved 4/8 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 5/8 statements.
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
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 0
    var_4 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'custom'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unsign_valid_return_timestamp_true. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_separator_raises_bad_time_signature. Retrieved 6/9 statements.
# Partially parsed test_unsign_expired_signature_raises_signature_expired. Retrieved 7/10 statements.


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
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = b''

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.secret'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 0.1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unsign_returns_value_and_timestamp. Retrieved 7/9 statements.
# Partially parsed test_unsign_raises_bad_time_signature_on_missing_timestamp. Retrieved 9/14 statements.


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
    var_6 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 0
    var_3 = 'test'
    var_4 = var_1.sign(var_3)
    var_5 = 1
    var_6 = b'dummy'
    var_7 = var_2 + var_6
    var_8 = var_1.unsign(var_7)

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
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true_returns_payload_and_timestamp. Retrieved 6/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_52_evaluates_to_true. Retrieved 7/18 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = -1
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = b'fakesignature'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 11/15 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = b'!!invalid!!'
    var_6 = var_2 + var_4
    var_7 = var_6 + var_5
    var_8 = var_7 + var_4
    var_9 = var_2 + var_4
    var_10 = var_9 + var_5



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_missing_timestamp_raises_bad_time_signature. Retrieved 19/25 statements.


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
    var_13 = b'!!!'
    var_14 = var_2 + var_7
    var_15 = var_14 + var_13
    var_16 = var_15 + var_7
    var_17 = var_2 + var_7
    var_18 = var_17 + var_13



# Parsed testcases at query #16
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test-value'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '-'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '_'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '='
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_age_less_than_zero_raises_signature_expired. Retrieved 6/26 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'test'
    var_2 = 3000
    var_3 = module_0.int_to_bytes(var_2)
    var_4 = module_0.base64_encode(var_3)
    var_5 = 100



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unsign_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/9 statements.
# Partially parsed test_unsign_future_timestamp. Retrieved 7/9 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 5/9 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 7/15 statements.
# Partially parsed test_unsign_string_input. Retrieved 4/6 statements.


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
    var_4 = 0
    var_5 = 1
    var_6 = var_1.unsign(var_3, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = 1
    var_6 = var_1.unsign(var_3, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'invalid'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test_value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_age_less_than_zero_evaluates_true. Retrieved 13/24 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 100
    var_3 = b'test'
    var_4 = b'.'
    var_5 = b'abc'
    var_6 = var_3 + var_4
    var_7 = var_6 + var_5
    var_8 = var_7 + var_4
    var_9 = var_3 + var_4
    var_10 = var_9 + var_5
    var_11 = 50
    var_12 = 10



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_loads_basic_payload. Retrieved 3/5 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/8 statements.
# Partially parsed test_loads_with_max_age_valid. Retrieved 4/6 statements.
# Partially parsed test_loads_with_max_age_expired. Retrieved 5/10 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_with_wrong_salt. Retrieved 5/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = 3600

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = 2
    var_4 = 1

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid_data'
    var_3 = var_1.loads(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom_salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'test_data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt1'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'test_data'
    var_4 = 'wrong_salt'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unsign_valid_with_return_timestamp_returns_tuple. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_bad_signature_and_timestamp_raises_bad_time_signature. Retrieved 8/16 statements.
# Partially parsed test_unsign_with_missing_separator_raises_bad_time_signature. Retrieved 6/9 statements.
# Partially parsed test_unsign_with_malformed_timestamp_raises_bad_time_signature. Retrieved 8/14 statements.


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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = b'.bad'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = b''

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = b'notabasetimestamp'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true_returns_payload_and_timestamp. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_loads_basic_returns_payload. Retrieved 3/5 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_max_age_valid. Retrieved 4/6 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_expired_signature_raises. Retrieved 5/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 3600

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 0.1
    var_4 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'bad-data'
    var_3 = var_1.loads(var_2)



# Parsed testcases at query #25
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    assert var_5 == b'test-value'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unsign_with_bad_signature_and_invalid_timestamp_raises_malformed_timestamp. Retrieved 11/22 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = b'invalid_base64'
    var_6 = var_2 + var_4
    var_7 = var_6 + var_5
    var_8 = var_7 + var_4
    var_9 = var_2 + var_4
    var_10 = var_9 + var_5



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_custom_algorithm. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_separator_in_base64_alphabet. Retrieved 1/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'bytes-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes-key'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp_and_signature_error_does_not_set_ts_int. Retrieved 8/19 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'invalid_base64!'
    var_6 = 0
    var_7 = b'tampered'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.TimestampSigner(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp_true_returns_tuple. Retrieved 7/9 statements.
# Partially parsed test_unsign_signature_without_timestamp_raises_bad_time_signature. Retrieved 3/9 statements.
# Partially parsed test_unsign_future_timestamp_raises_signature_expired. Retrieved 5/20 statements.
# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 4/14 statements.


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
    var_6 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'

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
    var_2 = 10000
    var_3 = b'test'
    var_4 = 3600

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'!!!'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_unsign_with_sig_error_and_malformed_timestamp_raises_bad_time_signature. Retrieved 15/33 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 63
    var_7 = var_5 ** var_6
    var_8 = module_1.int_to_bytes(var_7)
    var_9 = module_1.base64_encode(var_8)
    var_10 = 0
    var_11 = -1
    var_12 = -1
    var_13 = b'x'
    var_14 = b'y'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_loads_returns_payload_timestamp_tuple_when_return_timestamp_true. Retrieved 6/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_loads_returns_payload_when_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_tuple_with_timestamp_when_return_timestamp_true. Retrieved 4/7 statements.
# Partially parsed test_loads_raises_signature_expired_when_max_age_exceeded. Retrieved 4/7 statements.
# Partially parsed test_loads_with_salt_parameter. Retrieved 4/6 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_returns_none_when_payload_is_none. Retrieved 3/5 statements.


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
    var_3 = -1

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)

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
    var_2 = None



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_unsign_with_sep_in_result_and_sig_error_none. Retrieved 5/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_line32_evaluates_to_false. Retrieved 10/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = b'test'
    var_6 = var_4 == var_5
    var_7 = 0
    var_8 = var_4[var_7]
    var_9 = var_8 == var_5



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp_raises_bad_time_signature. Retrieved 7/11 statements.
# Partially parsed test_unsign_expired_signature_raises_signature_expired. Retrieved 10/22 statements.
# Partially parsed test_unsign_future_signature_raises_signature_expired. Retrieved 10/22 statements.
# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 12/16 statements.


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
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = module_1.want_bytes(var_2)
    var_4 = var_1.sep
    var_5 = module_1.want_bytes(var_4)
    var_6 = var_3 + var_5

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000
    var_3 = 'test'
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = var_4 + var_6
    var_8 = var_4 + var_6
    var_9 = 500

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000
    var_3 = 'test'
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = var_4 + var_6
    var_8 = var_4 + var_6
    var_9 = 500

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = module_1.want_bytes(var_2)
    var_4 = var_1.sep
    var_5 = module_1.want_bytes(var_4)
    var_6 = b'not-a-timestamp'
    var_7 = var_3 + var_5
    var_8 = var_7 + var_6
    var_9 = var_8 + var_5
    var_10 = var_3 + var_5
    var_11 = var_10 + var_6



# Parsed testcases at query #40
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_unsign_signature_okay_timestamp_malformed_ts_int_is_none. Retrieved 10/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'.'
    var_4 = b'!!invalid!!'
    var_5 = var_2 + var_3
    var_6 = var_5 + var_4
    var_7 = var_6 + var_3
    var_8 = var_2 + var_3
    var_9 = var_8 + var_4



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_loads_returns_tuple_when_return_timestamp_is_true. Retrieved 6/12 statements.


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

# Partially parsed test_unsign_with_return_timestamp_returns_tuple. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_expired_signature_raises_signature_expired. Retrieved 7/13 statements.
# Partially parsed test_unsign_with_negative_age_raises_signature_expired. Retrieved 7/13 statements.


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
    var_4 = 100
    var_5 = 10
    var_6 = var_1.unsign(var_3, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = 3600
    var_6 = var_1.unsign(var_3, var_5)

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
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -10
    var_5 = var_3[:var_4]
    var_6 = b'invalid'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid.no.timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -10
    var_5 = var_3[:var_4]
    var_6 = b'invalid'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true_returns_payload_and_timestamp. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 11/15 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = b'not-a-timestamp'
    var_6 = var_2 + var_4
    var_7 = var_6 + var_5
    var_8 = var_7 + var_4
    var_9 = var_2 + var_4
    var_10 = var_9 + var_5

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_unsign_without_max_age_returns_bytes. Retrieved 5/6 statements.
# Partially parsed test_unsign_with_max_age_exceeded_raises_signature_expired. Retrieved 8/10 statements.
# Partially parsed test_unsign_with_return_timestamp_true_returns_tuple. Retrieved 8/10 statements.
# Partially parsed test_unsign_with_invalid_signature_raises_bad_time_signature. Retrieved 5/6 statements.
# Partially parsed test_unsign_with_missing_timestamp_raises_bad_time_signature. Retrieved 5/6 statements.
# Partially parsed test_unsign_with_malformed_timestamp_raises_bad_time_signature. Retrieved 5/6 statements.
# Partially parsed test_unsign_with_negative_timestamp_age_raises_signature_expired. Retrieved 8/10 statements.


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
    var_4 = var_1.unsign(var_3)

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
    var_4 = 100
    var_5 = None
    var_6 = 0
    var_7 = var_1.unsign(var_3, var_6)

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
    var_7 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalidsignature'
    var_3 = None
    var_4 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = None
    var_4 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.sep.malformed'
    var_3 = None
    var_4 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = None
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_all_params. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

def test_case_0():
    var_0 = 'my-secret'
    var_1 = 'custom-salt'
    var_2 = '|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes-secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'bytes-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, digest_method=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, algorithm=var_1)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_unsign_valid_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 6/10 statements.


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
    var_4 = 0.1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
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
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'xx'
    var_5 = 1



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_predicate_line_52_evaluates_to_false. Retrieved 16/26 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.sep
    var_5 = module_1.want_bytes(var_4)
    var_6 = 1
    var_7 = 0
    var_8 = module_1.int_to_bytes(var_7)
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_1.sep
    var_11 = module_1.want_bytes(var_10)
    var_12 = var_1.sep
    var_13 = module_1.want_bytes(var_12)
    var_14 = var_1.sep
    var_15 = module_1.want_bytes(var_14)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true_returns_tuple. Retrieved 6/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #52
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_unsign_predicate_line52_false. Retrieved 6/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_loads_returns_payload_when_return_timestamp_is_false. Retrieved 6/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = False



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 8/14 statements.


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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'valuewithoutseparator'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = b'!!!'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_custom_algorithm. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_invalid_separator. Retrieved 1/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-bytes'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'!'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_unsign_valid_signature_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_separator. Retrieved 5/9 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/13 statements.


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
    var_4 = 3600
    var_5 = False
    var_6 = var_1.unsign(var_3, var_4, var_5)
    assert var_6 == b'test'

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b''

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
    var_2 = b'test'
    var_3 = b'invalid'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_unsign_predicate_line43_false. Retrieved 6/15 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.get_timestamp()
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'invalid!!'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_sep_in_result_when_signed_with_timestamp. Retrieved 5/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true_and_signature_expired_exception. Retrieved 9/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = 3600
    var_4 = b'payload'
    var_5 = var_2.sign(var_4)
    var_6 = 1
    var_7 = True
    var_8 = var_1.loads(var_5, var_6, var_7)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '+'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_value_error. Retrieved 8/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 999999999999999999999
    var_3 = 'test'
    var_4 = var_1.sign(var_3)
    var_5 = 0
    var_6 = True
    var_7 = var_1.unsign(var_4, return_timestamp=var_6)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_age_negative_raises_signature_expired. Retrieved 9/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 3600
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = 'Expected SignatureExpired'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_loads_returns_payload_without_timestamp. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp. Retrieved 4/8 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 5/10 statements.
# Partially parsed test_loads_with_custom_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_payload'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_payload'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_payload'
    var_3 = 0.01
    var_4 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid_data'
    var_3 = var_1.loads(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_payload'
    var_3 = 'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_payload'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_unsign_ts_int_is_none_raises_bad_time_signature. Retrieved 14/23 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = 1234567890
    var_6 = module_1.int_to_bytes(var_5)
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2 + var_4
    var_9 = var_8 + var_7
    var_10 = var_9 + var_4
    var_11 = var_2 + var_4
    var_12 = var_11 + var_7
    var_13 = b'invalid_base64'



# Parsed testcases at query #68
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_unsign_without_timestamp_raises_bad_time_signature. Retrieved 5/9 statements.
# Partially parsed test_unsign_with_invalid_timestamp_raises_bad_time_signature. Retrieved 4/14 statements.
# Partially parsed test_unsign_with_signature_error_raises_bad_time_signature. Retrieved 7/14 statements.
# Partially parsed test_unsign_with_expired_signature_raises_signature_expired. Retrieved 5/21 statements.
# Partially parsed test_unsign_with_future_timestamp_raises_signature_expired. Retrieved 5/21 statements.
# Partially parsed test_unsign_successful_without_return_timestamp. Retrieved 3/17 statements.
# Partially parsed test_unsign_successful_with_return_timestamp. Retrieved 4/19 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'.'
    var_4 = var_2 + var_3

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'invalid'

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 1234567890
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = b'bad'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 100
    var_4 = 50

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 100
    var_4 = 50

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
    var_3 = True



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'+'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '+'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret-key'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #71
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = 3600
    var_8 = var_1.unsign(var_3, var_7)
    var_9 = var_1.unsign(var_3, var_7, var_5)
    var_10 = -1
    var_11 = var_1.unsign(var_3, var_10)
    var_12 = -1
    var_13 = var_1.unsign(var_3, var_12, var_5)



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_loads_returns_payload_when_no_max_age_or_return_timestamp. Retrieved 3/5 statements.
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



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_unsign_returns_value_and_timestamp_when_return_timestamp_true. Retrieved 7/9 statements.


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
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = var_5[var_4]

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
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = module_1.want_bytes(var_2)
    var_4 = var_1.sep
    var_5 = module_1.want_bytes(var_4)
    var_6 = var_3 + var_5
    var_7 = b'invalidsignature'
    var_8 = var_6 + var_7
    var_9 = var_1.unsign(var_8)



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 7/9 statements.
# Partially parsed test_unsign_future_timestamp_raises_signature_expired. Retrieved 8/10 statements.


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
    var_6 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'tampered'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)

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
    var_4 = b'test'
    var_5 = b'future'
    var_6 = 100
    var_7 = var_1.unsign(var_3, var_6)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.malformed'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_false. Retrieved 7/18 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'!!invalid!!'
    var_6 = 0



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_loads_returns_timestamp_when_return_timestamp_is_true. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_unsign_valid_signature_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_bad_signature_no_timestamp. Retrieved 4/9 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/9 statements.
# Partially parsed test_unsign_string_input. Retrieved 5/7 statements.


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
    var_4 = 3600
    var_5 = False
    var_6 = var_1.unsign(var_3, var_4, var_5)
    assert var_6 == b'test'

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
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'invalid'

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
    var_2 = b'test'
    var_3 = b'invalid'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 8/14 statements.
# Partially parsed test_unsign_return_timestamp_type. Retrieved 5/7 statements.


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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'valuewithoutseparator'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = b'notbase64'

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
    var_2 = b''
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b''



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_timestamp_signer_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_with_custom_algorithm. Retrieved 1/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

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



