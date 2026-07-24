####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_timestamp_signer_with_custom_digest_method. Retrieved 1/4 statements.


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
    var_0 = 'secret'
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

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

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm == var_0)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unsign_returns_value_and_timestamp_when_return_timestamp_true. Retrieved 5/7 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_malformed_timestamp. Retrieved 7/14 statements.
# Partially parsed test_unsign_raises_signature_expired_when_age_exceeds_max_age. Retrieved 10/19 statements.
# Partially parsed test_unsign_raises_signature_expired_when_age_negative. Retrieved 8/22 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_signature_invalid_but_timestamp_present. Retrieved 9/11 statements.


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
    var_2 = b'no_separator'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'invalid_base64'
    var_5 = b'test'
    var_6 = b'signature'
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 0
    var_3 = module_1.int_to_bytes(var_2)
    var_4 = module_1.base64_encode(var_3)
    var_5 = var_1.sep
    var_6 = 'test'
    var_7 = var_6 + var_5
    var_8 = var_6 + var_5
    var_9 = 10
    var_10 = bool(False)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 100
    var_3 = var_1.sep
    var_4 = 'test'
    var_5 = var_4 + var_3
    var_6 = var_4 + var_3
    var_7 = 10
    var_8 = bool(False)
    assert var_8 is True

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
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_valid_with_return_timestamp. Retrieved 5/7 statements.


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
    var_5 = var_1.unsign(var_3, var_4)
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
    var_6 = b'invalid'
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
    var_2 = b''
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unsign_returns_tuple_when_return_timestamp_is_true. Retrieved 10/13 statements.


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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unsign_valid_signature_with_max_age_expired. Retrieved 7/9 statements.
# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 6/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = False
    var_6 = var_1.unsign(var_3, var_4, var_5)
    var_7 = bool(var_6 == var_2)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = 2020

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'invalid_base64'
    var_6 = var_4 + var_5
    var_7 = var_1.unsign(var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'YWJjZGVmZ2g='
    var_6 = var_4 + var_5
    var_7 = var_1.unsign(var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b''
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 8/14 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 7/11 statements.
# Partially parsed test_unsign_with_negative_age. Retrieved 7/10 statements.
# Partially parsed test_unsign_string_input. Retrieved 4/6 statements.


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
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test value'

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
    var_4 = 1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = b'invalid'
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = b'x'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 3600
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = ''
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b''

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 3/9 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/16 statements.
# Partially parsed test_unsign_with_bad_signature_and_malformed_timestamp. Retrieved 5/13 statements.
# Partially parsed test_unsign_with_bad_signature_and_valid_timestamp. Retrieved 7/23 statements.


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
    var_2 = b'value'
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'badts'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'badts'
    var_4 = b'badsig'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 8
    var_3 = 'big'
    var_4 = b'\x00'
    var_5 = b'value'
    var_6 = b'badsig'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.


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
    var_4 = 0.01
    var_5 = 0
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
    var_2 = b'value.sep'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_all_parameters. Retrieved 4/10 statements.


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
    var_7 = var_1.digest_method.__name__
    assert var_7 == 'sha1'
    var_8 = var_1.algorithm
    var_9 = bool(var_1.algorithm is not None)
    assert var_9 is True

def test_case_0():
    var_0 = 'mysecret'
    var_1 = b'mysalt'
    var_2 = b'|'
    var_3 = 'hmac'

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
    assert var_3 is True

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
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.default_digest_method
    var_3 = bool(var_1.default_digest_method is not None)
    assert var_3 is True
    var_4 = var_1.default_key_derivation
    assert var_4 == 'django-concat'



# Parsed testcases at query #11
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unsign_with_max_age_and_age_negative. Retrieved 7/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 100
    var_6 = var_1.unsign(var_3, var_5)



# Parsed testcases at query #13
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
    var_0 = 'secret-key'
    var_1 = b'-'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'-'

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
    var_0 = b'bytes-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-secret'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_timestamp_to_datetime_does_not_raise_for_valid_timestamp. Retrieved 6/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unsign_with_bad_signature_but_valid_timestamp. Retrieved 6/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'invalid'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loads_with_string_input. Retrieved 3/5 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_with_max_age. Retrieved 4/6 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 5/10 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 3600

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
    var_3 = 'custom'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 0.1
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_bad_time_signature_on_value_error. Retrieved 9/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = ()
    var_5 = 'invalid'
    var_6 = ValueError(var_5)
    var_7 = False
    var_8 = var_1.unsign(var_3, return_timestamp=var_7)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_sep_in_result_after_unsign_with_signature_error. Retrieved 7/16 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'invalid_signature'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unsign_ts_bytes_decode_raises_exception_and_ts_int_remains_none. Retrieved 7/23 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 1
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = b'!!!invalid_base64!!!'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 7/18 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_loads_returns_payload_when_no_max_age_and_no_return_timestamp. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_true. Retrieved 4/7 statements.
# Partially parsed test_loads_raises_signature_expired_when_max_age_exceeded. Retrieved 5/10 statements.
# Partially parsed test_loads_with_salt_uses_correct_signer. Retrieved 4/6 statements.
# Partially parsed test_loads_raises_bad_signature_with_wrong_salt. Retrieved 5/8 statements.
# Partially parsed test_loads_returns_payload_when_max_age_not_exceeded. Retrieved 4/6 statements.
# Partially parsed test_loads_returns_payload_with_return_timestamp_and_max_age. Retrieved 5/8 statements.


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
    var_2 = b'invalid'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_4 = 'salt2'
    var_5 = bool(False)
    assert var_5 is True

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
    var_3 = 3600
    var_4 = True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_loads_raises_signature_expired_when_expired. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = -1



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_unsign_without_timestamp_raises_bad_time_signature. Retrieved 4/9 statements.
# Partially parsed test_unsign_with_malformed_timestamp_raises_bad_time_signature. Retrieved 7/15 statements.
# Partially parsed test_unsign_with_expired_signature_raises_signature_expired. Retrieved 8/27 statements.
# Partially parsed test_unsign_with_negative_age_raises_signature_expired. Retrieved 8/27 statements.
# Partially parsed test_unsign_return_timestamp_returns_tuple. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'invalidsignature'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'notb64'
    var_7 = bool(False)
    assert var_7 is True

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

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 1000
    var_5 = 1
    var_6 = 0
    var_7 = 10
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 1000
    var_5 = 1
    var_6 = 0
    var_7 = 10
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test_value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test_value'



# Parsed testcases at query #24
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
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True

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
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_loads_returns_payload_when_return_timestamp_is_false. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unsign_returns_tuple_when_return_timestamp_true. Retrieved 7/9 statements.
# Partially parsed test_unsign_returns_bytes_on_validation_failure_with_timestamp. Retrieved 7/9 statements.


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
    var_2 = b'value.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

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
    var_2 = b'value.sep.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

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



# Parsed testcases at query #27
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
    var_8 = var_1.algorithm
    var_9 = bool(var_1.algorithm is not None)
    assert var_9 is True
    var_10 = var_1.digest_method
    var_11 = bool(var_1.digest_method is not None)
    assert var_11 is True

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
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'

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
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_all_parameters. Retrieved 4/10 statements.
# Partially parsed test_timestamp_signer_constructor_with_key_derivation_none. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = 'my-secret'
    var_1 = 'custom-salt'
    var_2 = '|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'none'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'none'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '_'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'_'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'bytes-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'bytes-salt'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.
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
    var_7 = var_1.digest_method.__name__
    assert var_7 == 'sha1'
    var_8 = var_1.algorithm.algorithm_type
    assert var_8 == 'hmac'

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
    var_0 = 'secret'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_unsign_age_negative. Retrieved 7/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = 10
    var_6 = var_1.unsign(var_3, var_5)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 6/11 statements.


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
    var_4 = 1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = b'badsig'
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b''
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/3 statements.
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
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

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
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes_secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes_secret'])
    assert var_3 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_on_value_error_from_timestamp_to_datetime. Retrieved 10/23 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 10
    var_6 = 100
    var_7 = var_5 ** var_6
    var_8 = module_1.int_to_bytes(var_7)
    var_9 = module_1.base64_encode(var_8)
    var_10 = 'Malformed timestamp'



# Parsed testcases at query #34
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = var_1.unsign(var_5)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_line52_evaluates_to_false. Retrieved 6/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_at_line_52_evaluates_to_true. Retrieved 8/18 statements.


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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_unsign_successful_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_raises_bad_time_signature_on_malformed_timestamp. Retrieved 12/16 statements.
# Partially parsed test_unsign_raises_signature_expired_on_negative_age. Retrieved 10/22 statements.
# Partially parsed test_unsign_returns_payload_on_bad_signature_with_valid_timestamp. Retrieved 8/17 statements.


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
    var_2 = b'justdata'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_1.Signer(var_0)
    var_3 = b'test'
    var_4 = var_2.sign(var_3)
    var_5 = var_1.unsign(var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = module_1.want_bytes(var_2)
    var_4 = var_1.sep
    var_5 = module_1.want_bytes(var_4)
    var_6 = b'not-valid-base64!!!'
    var_7 = var_3 + var_5
    var_8 = var_7 + var_6
    var_9 = var_8 + var_5
    var_10 = var_3 + var_5
    var_11 = var_10 + var_6
    var_12 = bool(False)
    assert var_12 is True

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
    var_2 = 1000
    var_3 = 'test'
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = var_4 + var_6
    var_8 = var_4 + var_6
    var_9 = 10
    var_10 = bool(False)
    assert var_10 is True

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
    var_7 = b'wrongsignature'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_unsign_returns_tuple_when_return_timestamp_true. Retrieved 7/9 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_missing. Retrieved 3/9 statements.
# Partially parsed test_unsign_returns_tuple_with_valid_max_age_and_return_timestamp. Retrieved 8/10 statements.


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
    var_6 = var_5[0]
    assert var_6 == b'test'
    var_7 = var_5[var_4]

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
    var_3 = bool(False)
    assert var_3 is True

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
    var_6 = var_1.unsign(var_3, var_4, var_5)
    var_7 = var_6[0]
    assert var_7 == b'test'
    var_8 = var_6[var_5]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_unsign_success_returns_tuple_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_preserves_payload_from_bad_signature. Retrieved 4/9 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_missing. Retrieved 4/9 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_malformed. Retrieved 5/13 statements.


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
    var_4 = 1000000
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = b'timestamp'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'sig'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'invalidsig'
    var_4 = b'badts'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'value'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 4/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'!!!'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 2/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.


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
    var_7 = var_1.algorithm
    var_8 = bool(var_1.algorithm is not None)
    assert var_8 is True

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
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True

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
    var_1 = ':'
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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_loads_with_str_input. Retrieved 3/5 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/9 statements.
# Partially parsed test_loads_with_max_age_not_expired. Retrieved 4/6 statements.
# Partially parsed test_loads_with_max_age_expired. Retrieved 5/10 statements.
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
    var_2 = b'invalid_data'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'custom_salt'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/6 statements.


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
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_unsign_predicate_line_43_false. Retrieved 7/23 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_2 + var_4
    var_6 = var_2 + var_4



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_unsign_returns_tuple_when_return_timestamp_true. Retrieved 7/9 statements.
# Partially parsed test_unsign_raises_bad_time_signature_for_missing_timestamp. Retrieved 5/9 statements.
# Partially parsed test_unsign_raises_signature_expired_when_max_age_exceeded. Retrieved 7/10 statements.


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
    var_2 = b'value.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
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
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_exception_on_invalid_timestamp. Retrieved 7/18 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'\xff\xff\xff\xff'
    var_6 = module_1.base64_encode(var_5)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.


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
    var_8 = bool(var_1.digest_method is not None)
    assert var_8 is True
    var_9 = var_1.algorithm
    var_10 = bool(var_1.algorithm is not None)
    assert var_10 is True

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
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'cannot be used'

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
    var_0 = b'bytes_secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes_secret'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #48
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'invalid'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 5/7 statements.
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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_4 = b'wrong'
    var_5 = 4
    var_6 = var_3[var_5:]
    var_7 = var_4 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 8/14 statements.
# Partially parsed test_unsign_returns_bytes. Retrieved 5/6 statements.
# Partially parsed test_unsign_returns_tuple_with_timestamp. Retrieved 10/13 statements.


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
    var_4 = 0
    var_5 = b'.'
    var_6 = var_3.split(var_5)[var_4]
    var_7 = var_1.unsign(var_6)
    var_8 = bool(False)
    assert var_8 is True

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



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_loads_returns_payload_when_valid. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp. Retrieved 4/9 statements.
# Partially parsed test_loads_with_max_age_raises_signature_expired. Retrieved 5/10 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/5 statements.
# Partially parsed test_loads_with_string_input. Retrieved 3/6 statements.


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



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


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
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-key'])
    assert var_3 is True

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
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #54
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



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_predicate_at_line_52_evaluates_to_false. Retrieved 9/18 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = module_1.int_to_bytes(var_4)
    var_6 = module_1.base64_encode(var_5)
    var_7 = b'test'
    var_8 = False



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_unsign_valid_signature_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 8/14 statements.


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
    var_6 = b'invalid'
    var_7 = 0
    var_8 = bool(False)
    assert var_8 is True

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



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_age_less_than_zero_raises_signature_expired. Retrieved 7/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 100
    var_6 = var_1.unsign(var_3, var_5)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_with_missing_timestamp. Retrieved 6/9 statements.
# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 7/12 statements.


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
    var_4 = b'bad'
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
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = b'notbase64'
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



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/6 statements.


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
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

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
    var_0 = 'key'
    var_1 = b'-'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'-'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'my-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'my-secret'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'binary-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'binary-key'])
    assert var_3 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.


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
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
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

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0.HMACAlgorithm()
    var_1 = 'secret'
    var_2 = module_1.TimestampSigner(var_1, algorithm=var_0)
    var_3 = var_2.algorithm
    var_4 = bool(var_2.algorithm is var_0)
    assert var_4 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_unsign_valid_return_timestamp_true. Retrieved 7/9 statements.
# Partially parsed test_unsign_valid_with_max_age_and_return_timestamp. Retrieved 8/10 statements.
# Partially parsed test_unsign_missing_timestamp_raises_bad_time_signature. Retrieved 4/9 statements.
# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 5/13 statements.


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
    var_7 = var_5[var_4]

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = True
    var_6 = var_1.unsign(var_3, var_4, var_5)
    var_7 = var_6[0]
    assert var_7 == b'test'
    var_8 = var_6[var_5]

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
    var_2 = b'test'
    var_3 = b'badsig'
    var_4 = bool(False)
    assert var_4 is True

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
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'!'
    var_4 = b'signature'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_age_negative_raises_signature_expired. Retrieved 8/17 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 1000
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = 500
    var_7 = 1000



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_loads_returns_payload_when_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 4/7 statements.
# Partially parsed test_loads_raises_bad_signature_when_signed_with_different_secret. Retrieved 5/8 statements.
# Partially parsed test_loads_raises_signature_expired_when_max_age_exceeded. Retrieved 5/10 statements.
# Partially parsed test_loads_accepts_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_with_salt_parameter. Retrieved 4/6 statements.


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
    var_2 = 'wrong-key'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = 'test_payload'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_payload'
    var_3 = 0.01
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

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
    var_3 = 'custom_salt'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 6/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 3/9 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/16 statements.


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
    var_7 = var_5[1].tzinfo
    var_8 = bool(var_5[1].tzinfo is not None)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0.1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
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
    var_2 = b'value'
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'invalid'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_unsign_ts_int_is_none_causes_bad_time_signature. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = 0
    var_6 = b'!!!'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.


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
    var_1 = ':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

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
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_unsign_returns_tuple_with_timestamp. Retrieved 7/9 statements.
# Partially parsed test_unsign_with_max_age_expired. Retrieved 8/11 statements.
# Partially parsed test_unsign_with_max_age_negative. Retrieved 8/11 statements.
# Partially parsed test_unsign_raises_bad_time_signature_on_missing_timestamp. Retrieved 7/10 statements.
# Partially parsed test_unsign_raises_bad_time_signature_on_malformed_timestamp. Retrieved 9/18 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
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
    var_2 = 1000
    var_3 = 'test'
    var_4 = var_1.sign(var_3)
    var_5 = 2000
    var_6 = 500
    var_7 = var_1.unsign(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 2000
    var_3 = 'test'
    var_4 = var_1.sign(var_3)
    var_5 = 1000
    var_6 = 3600
    var_7 = var_1.unsign(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.sign
    var_3 = b'.'
    var_4 = 'test'
    var_5 = var_1.sign(var_4)
    var_6 = var_1.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.sep
    var_5 = module_1.want_bytes(var_4)
    var_6 = 1
    var_7 = 0
    var_8 = b'invalid_timestamp'
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 3/9 statements.
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
    var_6 = b'notbase64'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #69
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



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
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
    var_1 = b'-'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'-'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loads_returns_payload_without_timestamp. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_true. Retrieved 4/7 statements.
# Partially parsed test_loads_raises_signature_expired_when_max_age_exceeded. Retrieved 5/10 statements.
# Partially parsed test_loads_raises_bad_signature_when_signature_invalid. Retrieved 7/15 statements.
# Partially parsed test_loads_raises_bad_signature_when_data_tampered. Retrieved 4/8 statements.
# Partially parsed test_loads_uses_salt_parameter. Retrieved 4/6 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.


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
    var_2 = 'test'
    var_3 = -1
    var_4 = -1
    var_5 = '0'
    var_6 = '1'
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'hack'
    var_4 = bool(False)
    assert var_4 is True

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_valid_with_return_timestamp. Retrieved 5/6 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 3/9 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/14 statements.
# Partially parsed test_unsign_string_input. Retrieved 4/6 statements.


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
    var_2 = b'test.invalidtimestamp.invalidsignature'
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
    var_4 = 10
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'!!invalid!!'
    var_4 = bool(False)
    assert var_4 is True

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.


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
    var_0 = 'secret'
    var_1 = 'custom_salt'
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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_missing_timestamp. Retrieved 3/9 statements.
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
    var_2 = b'not-a-timestamp'
    var_3 = b'test'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 100000
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 10/14 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 6/11 statements.


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
    var_2 = b'test'
    var_3 = b'.'
    var_4 = b'notbase64'
    var_5 = var_2 + var_3
    var_6 = var_5 + var_4
    var_7 = var_6 + var_3
    var_8 = var_2 + var_3
    var_9 = var_8 + var_4
    var_10 = bool(False)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 2
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loads_raises_signature_expired_when_signature_expired. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = -1
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unsign_valid_signature_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_unicode_value. Retrieved 5/6 statements.
# Partially parsed test_unsign_negative_age. Retrieved 8/13 statements.


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
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

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
    var_4 = b'x'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

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
    var_4 = b'x'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b''
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'héllo'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = -2
    var_6 = b'.'
    var_7 = var_3.split(var_6)[var_5]
    var_8 = module_1.base64_decode(var_7)
    var_9 = module_1.bytes_to_int(var_8)
    var_10 = var_4 - var_9
    var_11 = var_1.unsign(var_3, var_10)
    assert var_11 == b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = 1000
    var_4 = 'test'
    var_5 = var_2.sign(var_4)
    var_6 = 3600
    var_7 = var_1.unsign(var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #10
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 8/17 statements.


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
    var_4 = -1
    var_5 = var_3[:var_4]
    var_6 = b'x'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = var_3.split(var_5)[var_4]
    var_7 = var_6 + var_5
    var_8 = -1
    var_9 = var_3.split(var_5)[var_8]
    var_10 = var_7 + var_9
    var_11 = var_1.unsign(var_10)
    var_12 = bool(False)
    assert var_12 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 0
    var_6 = b'not-a-timestamp'
    var_7 = 2
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/6 statements.


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
    var_0 = 'secret-key'
    var_1 = ':'
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
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_all_parameters. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 'mykey'
    var_1 = 'mysalt'
    var_2 = '|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'oldkey'
    var_1 = 'newkey'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'oldkey', b'newkey'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'newkey'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 7/9 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 5/9 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/9 statements.
# Partially parsed test_unsign_future_signature. Retrieved 7/9 statements.


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
    var_2 = b'value'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
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
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 50
    var_5 = 100
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unsign_age_negative. Retrieved 12/24 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = 100
    var_6 = var_4 + var_5
    var_7 = module_1.int_to_bytes(var_6)
    var_8 = module_1.base64_encode(var_7)
    var_9 = 0
    var_10 = 2
    var_11 = 10



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_loads_returns_payload_when_not_return_timestamp. Retrieved 6/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = False



# Parsed testcases at query #19
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_loads_returns_payload_without_timestamp. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp. Retrieved 4/8 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 4/7 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_returns_payload_with_max_age. Retrieved 4/6 statements.


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
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'hello'
    var_3 = -1
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'hello'
    var_3 = 'custom'

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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_32_evaluates_to_false. Retrieved 9/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = 0
    var_6 = var_4[var_5]
    var_7 = b'test'
    var_8 = var_6 == var_7



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_future_timestamp. Retrieved 10/22 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 8/12 statements.
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
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = module_1.want_bytes(var_2)
    var_4 = var_1.sep
    var_5 = module_1.want_bytes(var_4)
    var_6 = 1000
    var_7 = var_3 + var_5
    var_8 = var_3 + var_5
    var_9 = 3600
    var_10 = bool(False)
    assert var_10 is True

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
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = module_1.want_bytes(var_2)
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = var_4 + var_6
    var_8 = bool(False)
    assert var_8 is True

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
    var_12 = bool(False)
    assert var_12 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = module_1.want_bytes(var_2)
    var_5 = var_1.sep
    var_6 = module_1.want_bytes(var_5)
    var_7 = var_4 + var_6
    var_8 = b'badsig'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)
    var_11 = bool(False)
    assert var_11 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b''
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_32_evaluates_to_false. Retrieved 5/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #24
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 4/8 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 6/10 statements.
# Partially parsed test_unsign_returns_bytes. Retrieved 5/6 statements.


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
    var_4 = 1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.'
    var_3 = b'hello'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.'
    var_3 = b'!!!'
    var_4 = var_2 + var_3
    var_5 = b'hello.!!!'
    var_6 = bool(False)
    assert var_6 is True

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
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unsign_handles_timestamp_to_datetime_error_returns_none. Retrieved 13/26 statements.


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
    var_7 = var_1.timestamp_to_datetime
    var_8 = ()
    var_9 = 'bad ts'
    var_10 = ValueError(var_9)
    var_11 = False
    var_12 = var_1.unsign(var_3, return_timestamp=var_11)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_loads_returns_tuple_when_return_timestamp_is_true. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_unsign_success_without_return_timestamp. Retrieved 1/3 statements.
# Partially parsed test_unsign_success_with_return_timestamp. Retrieved 2/5 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 2/5 statements.
# Partially parsed test_unsign_future_signature. Retrieved 2/5 statements.
# Partially parsed test_unsign_invalid_signature. Retrieved 2/6 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 2/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 'test_value'
    var_1 = True

def test_case_0():
    var_0 = 'test_value'
    var_1 = -1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0

def test_case_0():
    var_0 = 'test_value'
    var_1 = b'tampered'

def test_case_0():
    var_0 = b'test_value'
    var_1 = b'invalid'

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1
    var_2 = 0
    var_3 = b'notbase64'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_loads_returns_payload_when_not_return_timestamp_and_no_expiry. Retrieved 6/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = False



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_false. Retrieved 5/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_32_evaluates_to_false. Retrieved 7/17 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_2 + var_4
    var_6 = var_2 + var_4



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_loads_returns_payload_when_no_return_timestamp. Retrieved 3/5 statements.
# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_true. Retrieved 4/7 statements.
# Partially parsed test_loads_with_max_age_valid. Retrieved 4/6 statements.
# Partially parsed test_loads_with_max_age_and_return_timestamp. Retrieved 5/8 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_raises_signature_expired_on_old_timestamp. Retrieved 5/10 statements.


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
    var_3 = True

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
    var_3 = 3600
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'my_salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'hello'

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
    var_2 = 'hello'
    var_3 = 0.1
    var_4 = 0



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_unsign_returns_bytes_when_no_timestamp. Retrieved 5/6 statements.
# Partially parsed test_unsign_returns_bytes_and_datetime_when_return_timestamp_true. Retrieved 7/9 statements.
# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_missing. Retrieved 5/9 statements.


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
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = var_5[0]
    assert var_6 == b'test-value'
    var_7 = var_5[var_4]

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
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test-value'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_exception_at_line_52_raises_bad_time_signature_with_malformed_timestamp. Retrieved 6/17 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'!invalid!'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_loads_returns_payload_when_return_timestamp_is_false. Retrieved 6/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_loads_with_return_timestamp_false_returns_payload. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = False



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


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
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes-key'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = "Expected ValueError for '.' separator"
    var_4 = AssertionError(var_3)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_custom_algorithm. Retrieved 1/6 statements.


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
    var_0 = 'secret-key'
    var_1 = ':'
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

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_with_all_parameters. Retrieved 4/10 statements.


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

def test_case_0():
    var_0 = b'mykey'
    var_1 = b'mysalt'
    var_2 = b'|'
    var_3 = 'hmac'

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
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_loads_without_return_timestamp_does_not_enter_if_branch. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = False



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_on_invalid_timestamp. Retrieved 20/33 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 2
    var_3 = 63
    var_4 = var_2 ** var_3
    var_5 = 1
    var_6 = var_4 + var_5
    var_7 = b'\xff'
    var_8 = 8
    var_9 = var_7 * var_8
    var_10 = b'test'
    var_11 = b'.'
    var_12 = b'secret'
    var_13 = b'secret-salt'
    var_14 = 10
    var_15 = 18
    var_16 = var_14 ** var_15
    var_17 = '>Q'
    var_18 = module_0.TimestampSigner(var_0)
    var_19 = var_10 + var_11
    var_20 = bool(False)
    assert var_20 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_algorithm. Retrieved 1/6 statements.


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
    var_0 = 'secret'
    var_1 = b':'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b':'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom'
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
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_unsign_successful_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 5/9 statements.


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



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_invalid_base64_timestamp. Retrieved 8/13 statements.


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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'no_separator'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = 0
    var_7 = b'.invalid_base64'
    var_8 = bool(False)
    assert var_8 is True

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
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 1000000
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_unsign_valid_with_return_timestamp. Retrieved 7/9 statements.
# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 4/13 statements.
# Partially parsed test_unsign_expired_max_age_raises_signature_expired. Retrieved 7/15 statements.
# Partially parsed test_unsign_negative_age_raises_signature_expired. Retrieved 5/18 statements.
# Partially parsed test_unsign_with_bad_signature_and_valid_timestamp_raises_bad_time_signature. Retrieved 8/16 statements.
# Partially parsed test_unsign_with_bad_signature_and_malformed_timestamp_raises_bad_time_signature. Retrieved 4/12 statements.


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
    var_2 = b'nodata'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = b'invalidbase64'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000000
    var_3 = b'hello'
    var_4 = module_1.int_to_bytes(var_2)
    var_5 = module_1.base64_encode(var_4)
    var_6 = 1
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000
    var_3 = b'hello'
    var_4 = 100
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.get_timestamp()
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = 'wrong_secret'
    var_7 = module_0.TimestampSigner(var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = b'badbase64'
    var_4 = bool(False)
    assert var_4 is True



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
    var_6 = b'X'
    var_7 = var_5 + var_6
    var_8 = False
    var_9 = var_1.unsign(var_7, return_timestamp=var_8)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp_return. Retrieved 7/9 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/13 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 5/11 statements.
# Partially parsed test_unsign_bad_signature_with_timestamp. Retrieved 6/14 statements.
# Partially parsed test_unsign_bad_signature_without_timestamp. Retrieved 7/14 statements.


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
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

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
    var_2 = b'test.'
    var_3 = b'invalid'
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'test'
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'modified'
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'bad'
    var_5 = b'invalidsig'
    var_6 = b'timestamp'
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
    var_2 = None
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_loads_with_str_input. Retrieved 3/5 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/9 statements.
# Partially parsed test_loads_with_max_age_valid. Retrieved 4/6 statements.
# Partially parsed test_loads_with_max_age_expired. Retrieved 5/10 statements.
# Partially parsed test_loads_with_invalid_signature. Retrieved 7/16 statements.
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
    var_4 = 0.01
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = -1
    var_4 = b'x'
    var_5 = -1
    var_6 = 'x'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'custom'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 11/19 statements.


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



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_timestamp_signer_constructor_custom_digest_method. Retrieved 1/4 statements.


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
    var_4 = bool(var_2.algorithm == var_0)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_loads_returns_payload_without_timestamp. Retrieved 5/7 statements.
# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_true. Retrieved 6/9 statements.
# Partially parsed test_loads_raises_signature_expired_when_max_age_exceeded. Retrieved 7/12 statements.
# Partially parsed test_loads_with_salt_uses_correct_signer. Retrieved 6/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True

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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = 0
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


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
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret'])
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
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #54
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'just_value'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_unsign_exception_at_line_43_does_not_affect_result. Retrieved 7/22 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'!!!invalid!!!'
    var_6 = 0



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp_raises. Retrieved 4/9 statements.


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
    var_4 = 1000000
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'no_separator'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'bad_timestamp'
    var_4 = bool(False)
    assert var_4 is True

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'no_timestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_unsign_valid_with_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 5/10 statements.
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
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = b'not_a_valid_timestamp'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.


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
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'bytes_key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes_key'])
    assert var_3 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_loads_returns_payload_when_return_timestamp_is_false. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = False



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_loads_with_str_input. Retrieved 3/5 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_with_max_age. Retrieved 4/6 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
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
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'test'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_unsign_success_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 10/14 statements.


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
    var_2 = b'invalid|data|signature'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'nodata'
    var_3 = var_1.unsign(var_2)
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
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.'
    var_3 = b'notanint'
    var_4 = module_1.base64_encode(var_3)
    var_5 = var_2 + var_4
    var_6 = b'.'
    var_7 = var_5 + var_6
    var_8 = module_1.base64_encode(var_3)
    var_9 = var_2 + var_8
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #62
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_unsign_valid_return_timestamp. Retrieved 6/8 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 6/10 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 8/11 statements.
# Partially parsed test_unsign_future_signature. Retrieved 8/11 statements.
# Partially parsed test_unsign_with_bad_signature_and_valid_timestamp. Retrieved 9/17 statements.
# Partially parsed test_unsign_with_bad_signature_and_invalid_timestamp. Retrieved 4/8 statements.


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
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.invalid'
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
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000
    var_3 = 'hello'
    var_4 = var_1.sign(var_3)
    var_5 = 2000
    var_6 = 500
    var_7 = var_1.unsign(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 2000
    var_3 = 'hello'
    var_4 = var_1.sign(var_3)
    var_5 = 1000
    var_6 = 500
    var_7 = var_1.unsign(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.'
    var_3 = b'invalid_base64'
    var_4 = var_2 + var_3
    var_5 = var_1.unsign(var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1
    var_6 = b'wrong'
    var_7 = var_6 + var_4
    var_8 = b'wrong.'
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.invalid_timestamp.'
    var_3 = b'hello.invalid_timestamp'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_timestamp_signer_custom_digest_method. Retrieved 1/4 statements.


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
    var_0 = 'secret-key'
    var_1 = b':'
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
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_key
    assert var_4 == b'new_key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'A'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_ts_int_is_none_at_line_63. Retrieved 4/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'!!!invalid!!!'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #66
#--------------------------




import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 9999999999
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = var_1.sep
    var_7 = module_1.want_bytes(var_6)
    var_8 = var_2 + var_7
    var_9 = var_8 + var_5
    var_10 = var_9 + var_7
    var_11 = b'badsignature'
    var_12 = var_10 + var_11
    var_13 = var_1.unsign(var_12)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/8 statements.


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
    var_1 = '|'
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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'key1', b'key2'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'key2'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_ts_int_is_none_causes_bad_time_signature. Retrieved 9/19 statements.


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
    var_7 = b'notanint'
    var_8 = module_1.base64_encode(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_line_52_true. Retrieved 9/21 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = ()
    var_6 = OverflowError(var_2)
    var_7 = 0
    var_8 = var_1.unsign(var_3, var_7)



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_unsign_sep_not_in_result_and_sig_error_false. Retrieved 6/13 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = module_1.Signer(var_0)



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_unsign_predicate_line52_false. Retrieved 7/16 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.get_timestamp()
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = 3600



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_loads_signature_expired_raises_properly. Retrieved 13/22 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test payload'
    var_3 = -1
    var_4 = -1
    var_5 = b'a'
    var_6 = b'b'
    var_7 = 0
    var_8 = module_0.TimestampSigner(var_7)
    var_9 = b'test payload'
    var_10 = var_8.sign(var_9)
    var_11 = -1
    var_12 = var_1.loads(var_10, var_11)



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.TimestampSigner(var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'key1', b'key2', b'key3'])
    assert var_6 is True

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
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_predicate_age_less_than_zero. Retrieved 5/20 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000
    var_3 = b'test'
    var_4 = 0



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_unsign_valid_signature_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 6/11 statements.
# Partially parsed test_unsign_bad_timestamp_encoding. Retrieved 4/9 statements.


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
    var_4 = 0.1
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
    var_4 = b'test'
    var_5 = b'invalidsignature'
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
    var_2 = b'test'
    var_3 = b'notb64!signature'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #76
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #77
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_unsign_valid_signature_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 8/15 statements.


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
    var_4 = 0
    var_5 = b'test.'
    var_6 = var_3.split(var_5)[var_4]
    var_7 = b'.signature'
    var_8 = var_6 + var_7
    var_9 = var_1.unsign(var_8)
    var_10 = bool(False)
    assert var_10 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 2
    var_6 = 0
    var_7 = b'.invalid_timestamp.'
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
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



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_digest_method. Retrieved 1/4 statements.


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
    var_0 = 'secret'
    var_1 = 'custom_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom_salt'

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
    var_0 = 'old_key'
    var_1 = 'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'bytes_salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'bytes_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_malformed_timestamp_raises_bad_time_signature. Retrieved 19/25 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 1234567890
    var_4 = module_1.int_to_bytes(var_3)
    var_5 = module_1.base64_encode(var_4)
    var_6 = b'.'
    var_7 = var_2 + var_6
    var_8 = var_7 + var_5
    var_9 = var_8 + var_6
    var_10 = var_2 + var_6
    var_11 = var_10 + var_5
    var_12 = b'corrupt'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_2 + var_6
    var_15 = var_14 + var_13
    var_16 = var_15 + var_6
    var_17 = var_2 + var_6
    var_18 = var_17 + var_13
    var_19 = bool(False)
    assert var_19 is True
    var_20 = 'Malformed timestamp'



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp_raises_bad_time_signature. Retrieved 4/9 statements.
# Partially parsed test_unsign_with_bad_signature_and_invalid_timestamp_raises_bad_time_signature. Retrieved 4/9 statements.
# Partially parsed test_unsign_with_bad_signature_and_valid_timestamp_raises_bad_time_signature. Retrieved 6/11 statements.
# Partially parsed test_unsign_with_return_timestamp_true_returns_value_and_datetime. Retrieved 7/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value_without_sep'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'invalid_base64'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'AAAA'

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000000
    var_3 = module_1.int_to_bytes(var_2)
    var_4 = module_1.base64_encode(var_3)
    var_5 = b'value'

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
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = True
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = var_5[0]
    assert var_6 == b'test'
    var_7 = var_5[var_4]



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_timestamp_signer_default_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_with_digest_method. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_with_custom_algorithm. Retrieved 1/4 statements.


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
    var_0 = b'bytes_key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'bytes_key'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'string_key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'string_key'])
    assert var_3 is True



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_unsign_sep_not_in_result_sig_error_is_none. Retrieved 27/55 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = b'invalid-timestamp'
    var_4 = b'value'
    var_5 = b'MTIzNDU2Nzg5'
    var_6 = b'|'
    var_7 = '|'
    var_8 = module_0.TimestampSigner(var_0, sep=var_7)
    var_9 = b'value'
    var_10 = b'MTIzNDU2Nzg5'
    var_11 = var_9 + var_6
    var_12 = var_11 + var_10
    var_13 = var_12 + var_6
    var_14 = var_9 + var_6
    var_15 = var_14 + var_10
    var_16 = module_0.TimestampSigner(var_0)
    var_17 = b'test-value'
    var_18 = b'MTIzNDU2Nzg5'
    var_19 = b'tampered-signature'
    var_20 = b'value-without-sep'
    var_21 = 'test-value'
    var_22 = var_16.sign(var_21)
    var_23 = module_0.TimestampSigner(var_0)
    var_24 = b'test-value'
    var_25 = var_23.sign(var_24)
    var_26 = var_23.unsign(var_25)
    var_27 = bool(var_26 == var_24)
    assert var_27 is True



