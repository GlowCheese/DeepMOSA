####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_missing_timestamp. Retrieved 5/9 statements.
# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 6/14 statements.
# Partially parsed test_unsign_with_future_timestamp. Retrieved 7/24 statements.


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
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'invalid'
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
    var_6 = 'Signature age'
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 1000
    var_5 = 0
    var_6 = -1
    var_7 = 'Signature age'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_when_age_exceeds_max_age. Retrieved 9/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.get_timestamp()
    var_4 = 100
    var_5 = var_3 - var_4
    var_6 = var_1.sign(var_2)
    var_7 = var_1.unsign
    var_8 = 100



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 4/8 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'test_salt'
    var_2 = b'|'
    var_3 = 'hmac'



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value_without_separator'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_49_evaluates_to_false. Retrieved 11/14 statements.


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
    var_8 = var_1.get_timestamp
    var_9 = 0
    var_10 = var_1.unsign(var_7)



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_oserror. Retrieved 5/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = str(var_2)
    var_5 = 'Malformed timestamp'
    var_6 = bool('Malformed timestamp' in var_4)
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 5/7 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 6/9 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 6/9 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 6/9 statements.
# Partially parsed test_loads_with_salt. Retrieved 6/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid-signed-data'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = -1
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'custom-salt'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/3 statements.


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
    var_1 = staticmethod(var_0)
    var_2 = 'secret-key'
    var_3 = module_0.TimestampSigner(var_2, digest_method=var_1)
    var_4 = var_3.digest_method
    var_5 = bool(var_3.digest_method == var_1)
    assert var_5 is True

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



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_signature_expired_exception_is_raised_immediately. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loads_with_valid_signature_and_no_max_age. Retrieved 3/5 statements.
# Partially parsed test_loads_with_valid_signature_and_max_age_not_exceeded. Retrieved 4/6 statements.
# Partially parsed test_loads_with_valid_signature_and_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_string_input. Retrieved 3/5 statements.
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
    var_2 = 'invalid_data'
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
    var_3 = 'utf-8'

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
    var_3 = 'custom_salt'



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'expired_signature'
    var_3 = 0
    var_4 = var_1.loads(var_2, var_3)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
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
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_signature_expired_raises_immediately. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = 1



# Parsed testcases at query #21
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
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #22
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp.sep.signature'
    var_3 = True
    var_4 = var_1.unsign(var_2, return_timestamp=var_3)



# Parsed testcases at query #23
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
    var_2 = b'test.sep.malformed'
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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #24
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
    var_2 = b'test.invalid_timestamp.invalid_signature'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'timestamp missing'

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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.malformed_timestamp.signature'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Malformed timestamp'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_for_negative_age. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign
    var_5 = 100



# Parsed testcases at query #26
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



# Parsed testcases at query #27
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
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #28
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



# Parsed testcases at query #29
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
    var_2 = b'test.sep.invalid_timestamp'
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
    var_2 = b'test.sep.invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_future_timestamp. Retrieved 15/19 statements.


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
    var_2 = b'test:invalid_timestamp:signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test:signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.get_timestamp()
    var_4 = 3600
    var_5 = var_3 + var_4
    var_6 = module_1.int_to_bytes(var_5)
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_1.sep
    var_9 = module_1.want_bytes(var_8)
    var_10 = var_2 + var_9
    var_11 = var_10 + var_7
    var_12 = var_11 + var_9
    var_13 = var_2 + var_9
    var_14 = var_13 + var_7



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 6/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = b'='
    var_5 = b''



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = -1

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = 'custom-salt'



# Parsed testcases at query #34
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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_unsign_with_invalid_timestamp_raises_bad_time_signature. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp.sep.signature'
    var_3 = var_1.unsign(var_2)
    var_4 = 'Malformed timestamp'



# Parsed testcases at query #36
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #37
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
    var_2 = b'test'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = var_2 + var_3
    var_5 = b'invalid'
    var_6 = var_4 + var_5
    var_7 = var_1.unsign(var_6)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_params. Retrieved 6/12 statements.


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
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'custom-salt'
    var_4 = '|'
    var_5 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #39
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #40
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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.


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



# Parsed testcases at query #42
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.12345.invalid_signature'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_oserror. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = True
    var_4 = var_1.unsign(var_2, return_timestamp=var_3)
    var_5 = str(var_2)
    var_6 = 'Malformed timestamp'
    var_7 = bool('Malformed timestamp' in var_5)
    assert var_7 is True



# Parsed testcases at query #44
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'value'
    var_4 = None
    var_5 = b'.'
    var_6 = bool(var_5 not in var_3)
    assert var_6 is True
    var_7 = bool(not var_4)
    assert var_7 is True



# Parsed testcases at query #45
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
    var_2 = b'test.sep.malformed'
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
    var_2 = b'test.sep.12345678.invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #46
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'expired_signature'
    var_3 = 0
    var_4 = var_1.loads(var_2, var_3)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #48
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.validate(var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #50
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = -1
    var_3 = 'test'
    var_4 = var_1.sign(var_3)[:var_2]
    var_5 = var_1.unsign(var_4)
    var_6 = 'timestamp missing'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = b'secret-key'
    var_2 = module_0.TimestampSigner(var_1, digest_method=var_0)
    var_3 = var_2.digest_method
    var_4 = bool(var_2.digest_method == var_0)
    assert var_4 is True

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
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #53
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
    assert var_4 == b'itsdangerous.TimestampSigner'
    var_5 = var_1.sep
    assert var_5 == b'.'
    var_6 = var_1.key_derivation
    assert var_6 == 'django-concat'
    var_7 = var_1.digest_method
    var_8 = var_1.algorithm



# Parsed testcases at query #54
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_unsign_with_invalid_timestamp_raises_bad_time_signature. Retrieved 7/16 statements.


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



# Parsed testcases at query #56
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
    var_2 = b'invalid'
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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 4/5 statements.


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
    var_1 = b'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'|'
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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = b'secret-key'
    var_2 = module_0.TimestampSigner(var_1, digest_method=var_0)
    var_3 = var_2.digest_method
    var_4 = bool(var_2.digest_method == var_0)
    assert var_4 is True

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = module_0._lazy_sha1()
    var_1 = module_0.HMACAlgorithm(var_0)
    var_2 = b'secret-key'
    var_3 = module_1.TimestampSigner(var_2, algorithm=var_1)
    var_4 = var_3.algorithm
    var_5 = bool(var_3.algorithm == var_1)
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
    var_6 = var_3.secret_key
    assert var_6 == b'new-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #58
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'invalid_base64'
    var_1 = module_0.bytes_to_int(var_0)
    assert var_1 == 0



# Parsed testcases at query #59
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_str_input. Retrieved 4/6 statements.


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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)



# Parsed testcases at query #62
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
    var_2 = b'test:invalid_timestamp:signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test:signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test:timestamp:invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #63
#--------------------------




import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = 10
    var_4 = var_2 + var_3
    var_5 = -10
    var_6 = b'test'
    var_7 = var_1.sign(var_6)[:var_5]
    var_8 = module_1.int_to_bytes(var_4)
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = 5
    var_12 = var_1.unsign(var_10, var_11)



# Parsed testcases at query #64
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



# Parsed testcases at query #65
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



# Parsed testcases at query #66
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



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_unsign_with_return_timestamp_true. Retrieved 5/7 statements.


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
    var_4 = 100
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
    var_3 = var_1.unsign(var_2)
    var_4 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value:malformed'
    var_3 = var_1.unsign(var_2)
    var_4 = 'Malformed timestamp'

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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #68
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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_is_malformed. Retrieved 7/16 statements.


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



# Parsed testcases at query #70
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
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_unsign_with_invalid_timestamp. Retrieved 4/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp.sep.signature'
    var_3 = var_1.unsign



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


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
    var_0 = b'custom_secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'

def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_params. Retrieved 6/12 statements.


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
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = b'custom-salt'
    var_4 = b'|'
    var_5 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_missing_timestamp. Retrieved 3/7 statements.
# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 4/9 statements.


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
    var_2 = b'invalid'
    var_3 = var_1.unsign(var_2)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_negative_age_raises_signature_expired. Retrieved 12/20 statements.


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
    var_12 = 'Signature age -100 < 0 seconds'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/7 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/3 statements.


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



# Parsed testcases at query #2
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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = 100
    var_4 = var_2 + var_3
    var_5 = 'value'
    var_6 = var_1.sign(var_5)
    var_7 = 0
    var_8 = var_1.unsign(var_6, var_7)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp.sep.signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.timestamp.sep.invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'value'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_for_negative_age. Retrieved 7/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign
    var_6 = 10



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_on_malformed_timestamp. Retrieved 5/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.1234567890'
    var_3 = var_1.unsign(var_2)
    var_4 = str(var_3)
    var_5 = 'Malformed timestamp'
    var_6 = bool('Malformed timestamp' in var_4)
    assert var_6 is True



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_loads_raises_signature_expired. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp_raises_bad_time_signature. Retrieved 7/16 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid'
    var_7 = 'Malformed timestamp'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_when_age_is_negative. Retrieved 17/21 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = 1000
    var_6 = var_4 + var_5
    var_7 = b'value.'
    var_8 = module_1.int_to_bytes(var_6)
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_7 + var_9
    var_11 = b'.'
    var_12 = var_10 + var_11
    var_13 = module_1.int_to_bytes(var_6)
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_7 + var_14
    var_16 = var_1.timestamp_to_datetime(var_6)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_is_malformed. Retrieved 7/14 statements.


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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_ts_int_is_none_raises_bad_time_signature. Retrieved 11/14 statements.


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
    var_11 = 'Malformed timestamp'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_custom_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_with_wrong_salt. Retrieved 5/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid-data'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = -1
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = 'custom-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = 'custom-salt'
    var_4 = 'wrong-salt'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #12
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
    var_6 = b'invalid'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #14
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value_without_timestamp_and_signature'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_string_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/5 statements.
# Partially parsed test_loads_with_custom_salt. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid-signed-data'
    var_3 = var_1.loads(var_2)
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = -1
    var_4 = bool(False)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test-data'
    var_3 = 'custom-salt'



# Parsed testcases at query #17
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'expired_signature'
    var_3 = 0
    var_4 = var_1.loads(var_2, var_3)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 4/7 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'test_salt'
    var_2 = b'|'
    var_3 = 'hmac'



# Parsed testcases at query #19
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
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


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

def test_case_0():
    var_0 = 'custom-secret'
    var_1 = 'custom-salt'
    var_2 = '|'
    var_3 = 'hmac'

def test_case_0():
    var_0 = 'secret-key'

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
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unsign_with_invalid_timestamp_raises_bad_time_signature. Retrieved 8/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = var_1.sep
    var_7 = b'AAAAAAAAAAA='



# Parsed testcases at query #23
#--------------------------




import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value_without_separator'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = bool(var_4 not in var_2)
    assert var_5 is True
    var_6 = var_1.unsign(var_2)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unsign_with_missing_timestamp_and_no_signature_error. Retrieved 7/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'value'
    var_4 = b'.'
    var_5 = bool(var_4 not in var_3)
    assert var_5 is True
    var_6 = 'sig_error'
    var_7 = hasattr(var_1, var_6)



# Parsed testcases at query #26
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.unsign(var_2)
    assert var_3 is None



# Parsed testcases at query #27
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/7 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/3 statements.


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



# Parsed testcases at query #29
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
    var_7 = 'Signature age'

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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


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

def test_case_0():
    var_0 = b'custom-secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-secret'
    var_1 = b'new-secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-secret', b'new-secret'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #31
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


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
    var_6 = var_3.secret_key
    assert var_6 == b'key2'

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



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_parameters. Retrieved 4/10 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.salt
    assert var_3 == b'itsdangerous.Signer'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_is_malformed. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid'
    var_7 = 'Malformed timestamp'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_loads_raises_signature_expired_immediately. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1



# Parsed testcases at query #36
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
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test:malformed'
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



# Parsed testcases at query #37
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid_data_without_separator'
    var_3 = var_1.unsign(var_2)
    var_4 = 'sep'
    var_5 = bool('sep' not in var_3)
    assert var_5 is True



# Parsed testcases at query #38
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
    var_5 = 'timestamp missing'

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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Signature age'

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = module_1.want_bytes(var_3)
    var_5 = var_2 + var_4
    var_6 = b'invalid_timestamp'
    var_7 = var_5 + var_6
    var_8 = var_7 + var_4
    var_9 = b'signature'
    var_10 = var_8 + var_9
    var_11 = var_1.unsign(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'Malformed timestamp'



# Parsed testcases at query #39
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



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


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
    var_0 = b'secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-secret'
    var_1 = b'new-secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old-secret', b'new-secret'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new-secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_for_future_timestamp. Retrieved 11/25 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = 100
    var_4 = var_2 + var_3
    var_5 = b'value'
    var_6 = var_1.sign(var_5)
    var_7 = 0
    var_8 = module_1.int_to_bytes(var_4)
    var_9 = module_1.base64_encode(var_8)
    var_10 = -1
    var_11 = 'Signature age -100 < 0 seconds'



# Parsed testcases at query #42
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1234567890
    var_3 = var_1.timestamp_to_datetime(var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #43
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
    var_2 = 'test_data'
    var_3 = -1

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid_data'
    var_3 = var_1.loads(var_2)

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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_custom_parameters. Retrieved 4/8 statements.


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



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/11 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_age_negative_raises_signature_expired. Retrieved 7/10 statements.


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



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 7/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid_timestamp'



# Parsed testcases at query #48
#--------------------------




import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'invalid'
    var_1 = module_0.bytes_to_int(var_0)
    assert var_1 is None



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_4 = True

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
    var_5 = b'invalid'
    var_6 = var_4 + var_5
    var_7 = var_1.unsign(var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_loads_with_signature_expired_raises_immediately. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1



# Parsed testcases at query #51
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
    var_0 = 'string-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.secret_keys
    var_3 = bool(var_1.secret_keys == [b'string-secret'])
    assert var_3 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'cannot be used'



# Parsed testcases at query #52
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



# Parsed testcases at query #53
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp.sep.signature'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_unsign_with_negative_age_raises_signature_expired. Retrieved 7/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = str(var_5)
    var_7 = 'Signature age -50 < 0 seconds'
    var_8 = bool('Signature age -50 < 0 seconds' in var_6)
    assert var_8 is True



# Parsed testcases at query #55
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = bool(not b'.' in b'result_without_separator')
    assert var_2 is True



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



# Parsed testcases at query #58
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #59
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



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_loads_raises_signature_expired_when_signature_is_expired. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = 'test-salt'
    var_4 = -1
    var_5 = 'test-salt'



# Parsed testcases at query #61
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #62
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



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_negative_age_raises_signature_expired. Retrieved 9/20 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = 0
    var_6 = 1
    var_7 = var_1.sep
    var_8 = 50
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #64
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
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_timestamp_to_datetime_with_invalid_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp.sep.signature'
    var_3 = var_1.unsign(var_2)
    var_4 = str(var_3)
    var_5 = 'Malformed timestamp'
    var_6 = bool('Malformed timestamp' in var_4)
    assert var_6 is True



