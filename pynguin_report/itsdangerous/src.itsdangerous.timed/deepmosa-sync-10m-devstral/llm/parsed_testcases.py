####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_string_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_unsafe_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_unsafe_with_invalid_signature. Retrieved 3/4 statements.
# Partially parsed test_loads_unsafe_with_expired_signature. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid_data'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = -1
    var_4 = bool(False)
    assert var_4 is True

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
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = 'custom_salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'invalid_data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = -1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


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
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
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
    var_2 = b'test'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.sep.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_on_malformed_timestamp. Retrieved 7/16 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid_timestamp'
    var_7 = 'Malformed timestamp'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loads_with_return_timestamp. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unsign_with_sig_error_and_invalid_timestamp. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid_timestamp'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Malformed timestamp'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
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
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_when_age_exceeds_max_age. Retrieved 7/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 100
    var_4 = var_1.sign(var_2)
    var_5 = var_1.unsign
    var_6 = 100



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.sep.malformed'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Malformed timestamp'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_4 = 1000
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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid_timestamp'
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
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid_timestamp'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid_signature'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = -1
    var_4 = bool(False)
    assert var_4 is True

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
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = 'custom_salt'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_is_malformed. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid_timestamp'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_params. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous.TimestampSigner'
    var_3 = var_1.sep
    assert var_3 == b'.'
    var_4 = var_1.key_derivation
    assert var_4 == 'django-concat'
    var_5 = var_1.digest_method
    var_6 = var_1.algorithm
    var_7 = var_1.secret_keys
    var_8 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_8 is True

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_signature_expired_raises_immediately. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unsign_with_return_timestamp_true. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_return_timestamp_false. Retrieved 6/7 statements.


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
    var_4 = 1000
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
    var_2 = b'invalid'
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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

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
    var_4 = -1000
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #17
#--------------------------




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
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret-key'
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
    var_7 = module_1._lazy_sha1()
    var_8 = var_1.digest_method
    var_9 = bool(var_1.digest_method == var_7)
    assert var_9 is True
    var_10 = var_1.algorithm

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 7/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid_timestamp'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_loads_with_return_timestamp. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid_timestamp'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
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

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_for_negative_age. Retrieved 14/20 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = 100
    var_6 = var_4 + var_5
    var_7 = 0
    var_8 = 1
    var_9 = var_1.sep
    var_10 = module_1.int_to_bytes(var_6)
    var_11 = module_1.base64_encode(var_10)
    var_12 = 50
    var_13 = var_1.timestamp_to_datetime(var_6)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_for_negative_age. Retrieved 12/20 statements.


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
    var_7 = 0
    var_8 = 1
    var_9 = var_1.sep
    var_10 = module_1.int_to_bytes(var_6)
    var_11 = module_1.base64_encode(var_10)
    var_12 = 'Signature age -100 < 0 seconds'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_TimestampSigner_constructor_with_defaults. Retrieved 4/5 statements.
# Partially parsed test_TimestampSigner_constructor_with_custom_digest_method. Retrieved 1/5 statements.
# Partially parsed test_TimestampSigner_constructor_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret-key'
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
    var_7 = module_1._lazy_sha1()
    var_8 = var_1.digest_method
    var_9 = bool(var_1.digest_method == var_7)
    assert var_9 is True
    var_10 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = b'secret-key'

def test_case_0():
    var_0 = b'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = '='
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unsign_malformed_timestamp. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid_timestamp'



# Parsed testcases at query #27
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'expired_signature'
    var_3 = 0
    var_4 = var_1.loads(var_2, var_3)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_custom_parameters. Retrieved 4/9 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.sep
    assert var_4 == b'.'
    var_5 = var_1.salt
    assert var_5 == b'itsdangerous.TimestampSigner'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = module_1._lazy_sha1()
    var_8 = var_1.digest_method
    var_9 = bool(var_1.digest_method == var_7)
    assert var_9 is True
    var_10 = var_1.algorithm

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_loads_without_return_timestamp. Retrieved 3/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_loads_without_return_timestamp. Retrieved 3/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
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
    var_2 = b'hello'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.sep.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Malformed timestamp'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = 'Signature age'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = 'Signature age'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/5 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.TimestampSigner'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.sep
    assert var_3 == b'|'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.key_derivation
    assert var_3 == 'hmac'

def test_case_0():
    var_0 = b'secret-key'

def test_case_0():
    var_0 = b'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.


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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_oserror. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = True
    var_4 = var_1.unsign(var_2, return_timestamp=var_3)
    var_5 = str(var_3)
    var_6 = 'Malformed timestamp'
    var_7 = bool('Malformed timestamp' in var_5)
    assert var_7 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_oserror. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'invalid.sig'
    var_3 = True
    var_4 = var_1.unsign(var_2, return_timestamp=var_3)
    var_5 = str(var_2)
    var_6 = 'Malformed timestamp'
    var_7 = bool('Malformed timestamp' in var_5)
    assert var_7 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_unsign_with_negative_age. Retrieved 9/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = 'test_value'
    var_4 = var_1.sign(var_3)
    var_5 = 10
    var_6 = var_2 - var_5
    var_7 = 5
    var_8 = var_1.unsign(var_4, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_custom_salt. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid_signature'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = 0
    var_4 = bool(False)
    assert var_4 is True

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
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = 'custom_salt'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
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
    var_9 = var_1.algorithm.digest_method

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old_key'
    var_1 = b'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old_key', b'new_key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new_key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = b'test.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_2 = b'test.malformed'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
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
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_loads_with_return_timestamp. Retrieved 7/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_unsign_without_separator_and_no_signature_error. Retrieved 5/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value_without_separator'
    var_3 = var_1.sep
    var_4 = bool(var_1.sep not in var_2)
    assert var_4 is True
    var_5 = 'sig_error'
    var_6 = hasattr(var_1, var_5)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 7/10 statements.


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
    var_2 = b'test.invalid_timestamp.invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = b'X'
    var_6 = 1

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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_oserror. Retrieved 5/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = str(var_2)
    var_5 = 'Malformed timestamp'
    var_6 = bool('Malformed timestamp' in var_4)
    assert var_6 is True



# Parsed testcases at query #45
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value_without_separator'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #46
#--------------------------




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
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_loads_with_return_timestamp. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #48
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'expired_signature'
    var_3 = 0
    var_4 = var_1.loads(var_2, var_3)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_timestamp. Retrieved 5/7 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1
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
    var_2 = b'test.sep.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #50
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = var_3.rsplit(var_5, var_6)[var_4]
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_timestamp. Retrieved 5/7 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test:invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Malformed timestamp'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
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
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous.Signer'
    var_3 = var_1.sep
    assert var_3 == b'.'
    var_4 = var_1.key_derivation
    assert var_4 == 'django-concat'
    var_5 = var_1.digest_method
    var_6 = var_1.algorithm
    var_7 = var_1.secret_keys
    var_8 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_8 is True

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test:invalid_timestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.sep.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Malformed timestamp'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_unsign_without_timestamp_raises_bad_time_signature. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = 'timestamp missing'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.


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



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_for_negative_age. Retrieved 7/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 10
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = str(var_4)
    var_7 = 'Signature age -'
    var_8 = bool('Signature age -' in var_6)
    assert var_8 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid_data'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = -1
    var_4 = bool(False)
    assert var_4 is True

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
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = 'custom_salt'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous.Signer'
    var_3 = var_1.sep
    assert var_3 == b'.'
    var_4 = var_1.key_derivation
    assert var_4 == 'django-concat'
    var_5 = var_1.digest_method
    var_6 = var_1.algorithm
    var_7 = var_1.secret_keys
    var_8 = bool(var_1.secret_keys == [b'secret'])
    assert var_8 is True

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-string'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-string'])
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
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.malformed_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value_without_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.timestamp.invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unsign_with_invalid_timestamp. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




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
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unsign_with_missing_timestamp. Retrieved 3/7 statements.
# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 4/9 statements.
# Partially parsed test_unsign_with_return_timestamp_true. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_return_timestamp_false. Retrieved 6/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 1000
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
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
    var_2 = b'value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'invalid'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'malformed'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = var_3.rsplit(var_5, var_6)[var_4]
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unsign_with_return_timestamp_true. Retrieved 5/7 statements.


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
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

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
    var_2 = b'invalid'
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
    var_2 = b'test.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
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
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid_timestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unsign_with_max_age_expired. Retrieved 6/7 statements.
# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_invalid_signature. Retrieved 4/5 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 4/5 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 8/12 statements.


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
    var_4 = var_1.unsign
    var_5 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.unsign
    var_3 = b'invalid'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.unsign
    var_3 = b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid'
    var_7 = var_1.unsign



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unsign_valid_signature_with_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid_signature'
    var_3 = var_1.unsign(var_2)

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
    var_2 = 'value.sep.malformed_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #14
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'expired_signature'
    var_3 = 0
    var_4 = var_1.loads(var_2, var_3)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = True

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
    var_2 = b'value.invalid_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value_without_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
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

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

def test_case_0():
    var_0 = b'secret-key'

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
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #17
#--------------------------




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
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_TimestampSigner_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_TimestampSigner_constructor_custom_values. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret-key'
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
    var_7 = module_1._lazy_sha1()
    var_8 = var_1.digest_method
    var_9 = bool(var_1.digest_method == var_7)
    assert var_9 is True
    var_10 = var_1.algorithm

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_loads_with_signature_expired_raises_immediately. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_loads_with_valid_signature_and_no_max_age. Retrieved 3/5 statements.
# Partially parsed test_loads_with_valid_signature_and_max_age_not_exceeded. Retrieved 4/6 statements.
# Partially parsed test_loads_with_valid_signature_and_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = 60

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid signed data'
    var_3 = var_1.loads(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = 'custom-salt'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #24
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value_without_timestamp'
    var_3 = var_1.sep
    var_4 = bool(var_1.sep not in var_2)
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.TimestampSigner'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

def test_case_0():
    var_0 = b'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
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

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = b'value.sep.invalid_signature'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.signature'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.malformed_timestamp.sep.signature'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Malformed timestamp'

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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_unsign_without_sep_in_result_and_no_sig_error. Retrieved 7/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'.'
    var_4 = bool(var_3 not in var_2)
    assert var_4 is True
    var_5 = 'sig_error'
    var_6 = hasattr(var_1, var_5)
    var_7 = var_1.unsign(var_2)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_loads_with_return_timestamp. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_timestamp. Retrieved 5/7 statements.


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
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test:malformed'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Malformed timestamp'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
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
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_params. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.salt
    assert var_2 == b'itsdangerous.TimestampSigner'
    var_3 = var_1.sep
    assert var_3 == b'.'
    var_4 = var_1.key_derivation
    assert var_4 == 'django-concat'
    var_5 = var_1.digest_method
    var_6 = var_1.algorithm
    var_7 = var_1.secret_keys
    var_8 = bool(var_1.secret_keys == [b'secret'])
    assert var_8 is True

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret_string'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret_string'])
    assert var_3 is True

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
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = b'test.invalid_timestamp'
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
    var_2 = b'test.wrong_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #34
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value_without_separator'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'secret-key'])
    assert var_3 is True
    var_4 = var_1.salt
    assert var_4 == b'itsdangerous.TimestampSigner'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'A'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #36
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = 10
    var_4 = var_2 + var_3
    var_5 = b'value'
    var_6 = var_1.sign(var_5)
    var_7 = 0
    var_8 = var_1.unsign(var_6, var_7)
    var_9 = type(var_8)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 'secret-key'
    var_2 = module_0.TimestampSigner(var_1, digest_method=var_0)
    var_3 = var_2.digest_method
    var_4 = bool(var_2.digest_method == var_0)
    assert var_4 is True

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0.HMACAlgorithm(var_0)
    var_2 = 'secret-key'
    var_3 = module_1.TimestampSigner(var_2, algorithm=var_1)
    var_4 = var_3.algorithm
    var_5 = bool(var_3.algorithm == var_1)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_unsign_with_return_timestamp_true. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_return_timestamp_false. Retrieved 6/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 60
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
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
    var_2 = b'hello'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_4 = 1000
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

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
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid_timestamp.invalid_sig'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.invalid_sig'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/11 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
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
    var_7 = module_1._lazy_sha1()
    var_8 = var_1.digest_method
    var_9 = bool(var_1.digest_method == var_7)
    assert var_9 is True
    var_10 = var_1.algorithm

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.sep.invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Malformed timestamp'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_TimestampSigner_constructor. Retrieved 4/5 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

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
    var_7 = module_1._lazy_sha1()
    var_8 = var_1.digest_method
    var_9 = bool(var_1.digest_method == var_7)
    assert var_9 is True
    var_10 = var_1.algorithm



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_future_timestamp. Retrieved 7/21 statements.


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
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

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
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'malformed'
    var_6 = var_4 + var_5
    var_7 = var_6 + var_3
    var_8 = b'sig'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'sig'
    var_6 = var_4 + var_5
    var_7 = var_1.unsign(var_6)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = 100
    var_4 = b'.'
    var_5 = var_2 + var_4
    var_6 = var_2 + var_4



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_valid_signature_and_max_age. Retrieved 4/6 statements.
# Partially parsed test_loads_with_valid_signature_and_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.


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
    var_3 = 3600

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
    var_3 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid_signature'
    var_3 = var_1.loads(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_data'
    var_3 = 'custom_salt'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 7/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid_timestamp'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid signature'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = -1
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = 'custom-salt'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 11/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = var_3.rsplit(var_5, var_6)[var_4]
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
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

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

def test_case_0():
    var_0 = b'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-key', b'new-key'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'invalid-timestamp'
    var_6 = var_4 + var_5
    var_7 = var_6 + var_3
    var_8 = b'signature'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'signature'
    var_6 = var_4 + var_5
    var_7 = var_1.unsign(var_6)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
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
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



