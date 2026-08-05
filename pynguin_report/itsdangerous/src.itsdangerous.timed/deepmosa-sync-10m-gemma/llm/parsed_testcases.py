####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loads_returns_payload_when_valid. Retrieved 6/15 statements.
# Partially parsed test_loads_returns_payload_and_timestamp_when_requested. Retrieved 5/13 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 4/13 statements.
# Partially parsed test_loads_raises_bad_signature_after_trying_all_signers. Retrieved 5/16 statements.
# Partially parsed test_loads_handles_string_input_by_converting_to_bytes. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'data'
    var_1 = b'base64encoded'
    var_2 = 123456789
    var_3 = b'signature'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = 'data'
    var_1 = b'base64encoded'
    var_2 = 123456789
    var_3 = b'signature'
    var_4 = True

def test_case_0():
    var_0 = b'signature'
    var_1 = 10
    var_2 = 'SignatureExpired was not raised'
    var_3 = AssertionError(var_2)

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'error 2'
    var_2 = b'signature'
    var_3 = 'BadSignature was not raised'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = 'data'
    var_1 = b'base64'
    var_2 = 123
    var_3 = 'string_input'
    var_4 = b'string_input'
    var_5 = None
    var_6 = True



# Parsed testcases at query #2
#--------------------------




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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'|'
    var_3 = 'hmac'
    var_4 = None
    var_5 = module_0.TimestampSigner(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5.secret_keys
    var_7 = bool(var_5.secret_keys == [b'secret'])
    assert var_7 is True
    var_8 = var_5.salt
    assert var_8 == b'salt'
    var_9 = var_5.sep
    assert var_9 == b'|'
    var_10 = var_5.key_derivation
    assert var_10 == 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'A'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret_string'
    var_1 = 'salt_string'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret_string'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'salt_string'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_returns_tuple_with_datetime_when_return_timestamp_is_true. Retrieved 5/7 statements.
# Partially parsed test_unsign_raises_SignatureExpired_when_max_age_is_exceeded. Retrieved 9/13 statements.
# Partially parsed test_unsign_handles_malformed_timestamp_in_bad_signature. Retrieved 7/14 statements.


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
    var_7 = b'hello'
    var_8 = 'SignatureExpired not raised'
    var_9 = AssertionError(var_8)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -5
    var_5 = var_3[:var_4]
    var_6 = b'wrong'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'data.not_a_timestamp'
    var_3 = b'data.invalid'
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'\x00\x00\x00\x00\x00\x00\x00\x01'
    var_3 = b'.'
    var_4 = b'data'
    var_5 = var_4 + var_3
    var_6 = b'wrongsignature'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'nosplit'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unsign_with_max_age_not_none. Retrieved 7/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = 'utf-8'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_loads_success_without_timestamp. Retrieved 7/15 statements.
# Partially parsed test_loads_success_with_timestamp. Retrieved 6/12 statements.
# Partially parsed test_loads_with_max_age. Retrieved 7/14 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 4/11 statements.
# Partially parsed test_loads_raises_bad_signature_on_failure. Retrieved 7/16 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'data'
    var_1 = 'value'
    var_2 = b'base64data'
    var_3 = 123456789
    var_4 = b'signed_data'
    var_5 = None
    var_6 = True

def test_case_0():
    var_0 = 'data'
    var_1 = 'value'
    var_2 = b'base64data'
    var_3 = 123456789
    var_4 = b'signed_data'
    var_5 = True

def test_case_0():
    var_0 = 'data'
    var_1 = 'value'
    var_2 = b'base64data'
    var_3 = 123456789
    var_4 = b'signed_data'
    var_5 = 100
    var_6 = True

def test_case_0():
    var_0 = 'expired'
    var_1 = b'signed_data'
    var_2 = 'SignatureExpired was not raised'
    var_3 = AssertionError(var_2)

import src.itsdangerous.exc as module_0

def test_case_0():
    var_0 = 'bad 1'
    var_1 = module_0.BadSignature(var_0)
    var_2 = 'bad 2'
    var_3 = module_0.BadSignature(var_2)
    var_4 = b'signed_data'
    var_5 = 'BadSignature was not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = b'base64'
    var_1 = 123
    var_2 = b'data'
    var_3 = 'mysalt'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_loads_returns_payload_when_return_timestamp_is_false. Retrieved 6/15 statements.


def test_case_0():
    var_0 = b'payload_base64'
    var_1 = 123456789
    var_2 = 'data'
    var_3 = 'value'
    var_4 = b'some_signature'
    var_5 = False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_invalid_signature. Retrieved 6/9 statements.
# Partially parsed test_unsign_validates_correctly. Retrieved 4/9 statements.


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
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b's'
    var_5 = b'x'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'data'
    var_3 = b'.'
    var_4 = b'!!!'
    var_5 = var_2 + var_3
    var_6 = var_5 + var_4
    var_7 = var_6 + var_3
    var_8 = b'invalid_sig'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'just_a_payload_no_sep'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'hello'
    var_2 = b'h'
    var_3 = b'z'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loads_success_returns_payload. Retrieved 7/14 statements.
# Partially parsed test_loads_success_returns_payload_and_timestamp. Retrieved 6/12 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 2/8 statements.
# Partially parsed test_loads_raises_bad_signature_on_all_signers. Retrieved 5/15 statements.
# Partially parsed test_loads_stops_at_first_valid_signer. Retrieved 4/12 statements.


def test_case_0():
    var_0 = b'payload_base64'
    var_1 = 123456789
    var_2 = 'data'
    var_3 = 'test'
    var_4 = b'signature'
    var_5 = 100
    var_6 = True

def test_case_0():
    var_0 = b'payload_base64'
    var_1 = 123456789
    var_2 = 'data'
    var_3 = 'test'
    var_4 = b'signature'
    var_5 = True

def test_case_0():
    var_0 = b'signature'
    var_1 = 10

import src.itsdangerous.exc as module_0

def test_case_0():
    var_0 = 'bad 1'
    var_1 = module_0.BadSignature(var_0)
    var_2 = 'bad 2'
    var_3 = module_0.BadSignature(var_2)
    var_4 = b'signature'

def test_case_0():
    var_0 = 'bad'
    var_1 = b'valid_base64'
    var_2 = 123
    var_3 = b'signature'



# Parsed testcases at query #9
#--------------------------




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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'|'
    var_3 = 'hmac'
    var_4 = module_0.TimestampSigner(var_0, var_1, var_2, var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'salt'
    var_8 = var_4.sep
    assert var_8 == b'|'
    var_9 = var_4.key_derivation
    assert var_9 == 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'A'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = 'The given separator cannot be used'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.secret_keys
    var_4 = bool(var_2.secret_keys == [b'secret'])
    assert var_4 is True
    var_5 = var_2.salt
    assert var_5 == b'salt'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unsign_sig_error_and_valid_timestamp_fails_predicate. Retrieved 25/46 statements.


import src.itsdangerous.encoding as module_0
import src.itsdangerous.exc as module_1
import src.itsdangerous.timed as module_2

def test_case_0():
    var_0 = 1000
    var_1 = module_0.int_to_bytes(var_0)
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'data.some_timestamp_part'
    var_4 = module_0.int_to_bytes(var_0)
    var_5 = module_0.base64_encode(var_4)
    var_6 = b'value'
    var_7 = b'.'
    var_8 = var_6 + var_7
    var_9 = var_8 + var_5
    var_10 = 'Bad signature'
    var_11 = module_1.BadSignature(var_10)
    var_12 = 12345
    var_13 = module_0.int_to_bytes(var_12)
    var_14 = module_0.base64_encode(var_13)
    var_15 = b'some_data'
    var_16 = var_15 + var_7
    var_17 = var_16 + var_14
    var_18 = 'Invalid signature'
    var_19 = module_1.BadSignature(var_18)
    var_20 = 'secret'
    var_21 = module_2.TimestampSigner(var_20)
    var_22 = b'hello'
    var_23 = var_21.sign(var_22)
    var_24 = var_21.unsign(var_23)
    assert var_24 == b'hello'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_is_malformed. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'value'
    var_2 = b'.'
    var_3 = var_1 + var_2
    var_4 = b'invalid_base64_!'
    var_5 = var_3 + var_4
    var_6 = b'some_signed_value'
    var_7 = 'BadTimeSignature was not raised for malformed timestamp'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unsign_age_less_than_zero. Retrieved 16/47 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'sha1'
    var_2 = 100
    var_3 = b'.'
    var_4 = b'data'
    var_5 = var_4 + var_3
    var_6 = b'dummy_signature'
    var_7 = b'data'
    var_8 = b'.'
    var_9 = 0
    var_10 = b'.signature'
    var_11 = 1000
    var_12 = 500
    var_13 = 1000
    var_14 = '-500'
    var_15 = 'SignatureExpired with negative age was not raised'
    var_16 = AssertionError(var_15)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_when_age_is_negative. Retrieved 16/87 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 1100
    var_1 = int(var_0)
    var_2 = 8
    var_3 = 'big'
    var_4 = b'payload.'
    var_5 = 'secret'
    var_6 = module_0.TimestampSigner(var_5)
    var_7 = 1100
    var_8 = int(var_7)
    var_9 = module_0.TimestampSigner(var_5)
    var_10 = 100
    var_11 = module_0.TimestampSigner(var_5)
    var_12 = module_0.TimestampSigner(var_5)
    var_13 = 200
    var_14 = int(var_13)
    var_15 = module_0.TimestampSigner(var_5)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'payload.'
    var_2 = b'!!!'
    var_3 = var_1 + var_2
    var_4 = str(var_0)
    var_5 = 'Malformed timestamp'
    var_6 = bool('Malformed timestamp' in var_4)
    assert var_6 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loads_success_with_timestamp. Retrieved 4/10 statements.
# Partially parsed test_loads_success_without_timestamp. Retrieved 3/8 statements.
# Partially parsed test_loads_expired_signature_raises_error. Retrieved 4/10 statements.
# Partially parsed test_loads_bad_signature_raises_error. Retrieved 4/11 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/9 statements.
# Partially parsed test_loads_with_wrong_salt_raises_error. Retrieved 5/11 statements.
# Partially parsed test_loads_handles_bytes_input. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = True

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'tampered'

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'my-salt'

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'correct-salt'
    var_4 = 'wrong-salt'

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'utf-8'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true. Retrieved 8/25 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'payload_bytes'
    var_1 = 1600000000
    var_2 = 'secret'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = 'decoded_payload'
    var_5 = 'some_data'
    var_6 = True
    var_7 = var_3.loads(var_5, return_timestamp=var_6)
    var_8 = bool(var_7 == ('decoded_payload', 1600000000))
    assert var_8 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loads_success_no_timestamp. Retrieved 4/12 statements.
# Partially parsed test_loads_success_with_timestamp. Retrieved 5/13 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 4/13 statements.
# Partially parsed test_loads_raises_bad_signature_on_all_signers. Retrieved 5/16 statements.
# Partially parsed test_loads_handles_multiple_signers_fallback. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'data'
    var_1 = b'base64'
    var_2 = 12345
    var_3 = b'signature'

def test_case_0():
    var_0 = 'data'
    var_1 = b'base64'
    var_2 = 12345
    var_3 = b'signature'
    var_4 = True

def test_case_0():
    var_0 = b'signature'
    var_1 = 10
    var_2 = 'SignatureExpired not raised'
    var_3 = AssertionError(var_2)

def test_case_0():
    var_0 = 'bad1'
    var_1 = 'bad2'
    var_2 = b'signature'
    var_3 = 'BadSignature not raised'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = 'recovered'
    var_1 = 'bad'
    var_2 = b'valid_base64'
    var_3 = 12345
    var_4 = b'signature'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loads_return_timestamp_false. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = b'base64data'
    var_3 = 123456789
    var_4 = b'data'
    var_5 = False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_loads_returns_payload_without_timestamp. Retrieved 6/15 statements.


def test_case_0():
    var_0 = b'payload_base64'
    var_1 = 123456789
    var_2 = 'data'
    var_3 = 'test'
    var_4 = b'some_signature'
    var_5 = False



# Parsed testcases at query #6
#--------------------------




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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'mysalt'
    var_2 = b':'
    var_3 = 'hmac'
    var_4 = module_0.TimestampSigner(var_0, var_1, var_2, var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'mysalt'
    var_8 = var_4.sep
    assert var_8 == b':'
    var_9 = var_4.key_derivation
    assert var_9 == 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'A'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 4/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 11/14 statements.


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
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b'tampered'
    var_5 = var_3 + var_4
    var_6 = var_1.unsign(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'payload.sig'
    var_3 = var_1.unsign(var_2)
    var_4 = 'timestamp missing'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = b'.'
    var_4 = b'not-base64-data!!!'
    var_5 = var_2 + var_3
    var_6 = var_5 + var_4
    var_7 = var_6 + var_3
    var_8 = b'signature'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)
    var_11 = 'Malformed timestamp'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.validate(var_3)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid.signature'
    var_3 = var_1.validate(var_2)
    assert var_3 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loads_returns_payload_immediately_on_success. Retrieved 6/12 statements.


def test_case_0():
    var_0 = b'base64_payload'
    var_1 = 123456789
    var_2 = 'data'
    var_3 = 'success'
    var_4 = 'some_signature'
    var_5 = False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_malformed_timestamp_raises_error. Retrieved 11/18 statements.


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
    var_2 = b'wrong_signature_here'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(True)
    assert var_4 is True
    var_5 = 'Should have raised BadSignature'
    var_6 = AssertionError(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = b'hello'
    var_4 = var_2.sign(var_3)
    var_5 = b'.'
    var_6 = 1
    var_7 = 0
    var_8 = b'not_base64_!!!'
    var_9 = bool(True)
    assert var_9 is True
    var_10 = 'Should have raised error due to malformed timestamp'
    var_11 = AssertionError(var_10)

import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_1.Signer(var_0)
    var_3 = b'hello'
    var_4 = var_2.sign(var_3)
    var_5 = var_1.unsign(var_4)
    var_6 = 'timestamp missing'
    var_7 = bool('timestamp missing' in str(e).lower())
    assert var_7 is True
    var_8 = 'Should have raised BadTimeSignature for missing timestamp'
    var_9 = AssertionError(var_8)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.validate(var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = var_1.validate(var_5)
    assert var_6 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.


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
    var_4 = 1.1
    var_5 = 1
    var_6 = var_1.unsign(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -5
    var_5 = var_3[:var_4]
    var_6 = b'error'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)
    var_9 = bool(False)
    assert var_9 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'.'
    var_3 = b'data'
    var_4 = var_3 + var_2
    var_5 = b'invalid_base64_!!!'
    var_6 = var_4 + var_5
    var_7 = var_1.unsign(var_6)
    var_8 = bool(False)
    assert var_8 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'nodatahere'
    var_3 = var_1.unsign(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------




import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = b'data'
    var_4 = 123456789
    var_5 = module_1.int_to_bytes(var_4)
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_2.sign(var_3)
    var_8 = var_2.unsign(var_7)
    var_9 = bool(var_8 == var_3)
    assert var_9 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unsign_age_less_than_zero_raises_signature_expired. Retrieved 8/23 statements.
# Partially parsed test_trigger_age_less_than_zero. Retrieved 9/23 statements.
# Partially parsed test_unsign_negative_age_logic. Retrieved 9/19 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 2000000000
    var_1 = module_0.int_to_bytes(var_0)
    var_2 = module_0.base64_encode(var_1)
    var_3 = b'data'
    var_4 = b'.'
    var_5 = var_3 + var_4
    var_6 = b'some_signed_value'
    var_7 = 3600

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000
    var_3 = b'payload'
    var_4 = var_1.sign(var_3)
    var_5 = 3600
    var_6 = var_1.unsign(var_4, var_5)
    var_7 = 'Signature age'
    var_8 = 'SignatureExpired not raised for negative age'
    var_9 = AssertionError(var_8)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 5000
    var_3 = b'data'
    var_4 = var_1.sign(var_3)
    var_5 = 10000
    var_6 = var_1.unsign(var_4, var_5)
    var_7 = 'age'
    var_8 = bool('age' in str(e).lower())
    assert var_8 is True
    var_9 = 'Failed to trigger age < 0 exception'
    var_10 = AssertionError(var_9)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_loads_returns_timestamp_when_requested. Retrieved 6/12 statements.


def test_case_0():
    var_0 = b'payload_base64'
    var_1 = 123456789
    var_2 = 'data'
    var_3 = 'test'
    var_4 = b'some_signature'
    var_5 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_loads_returns_payload_when_valid. Retrieved 6/14 statements.
# Partially parsed test_loads_returns_tuple_when_return_timestamp_is_true. Retrieved 5/12 statements.
# Partially parsed test_loads_raises_signature_expired_immediately. Retrieved 3/10 statements.
# Partially parsed test_loads_tries_next_signer_on_bad_signature. Retrieved 5/15 statements.
# Partially parsed test_loads_raises_last_bad_signature_if_all_signers_fail. Retrieved 3/15 statements.


def test_case_0():
    var_0 = b'payload_base64'
    var_1 = 123456789
    var_2 = 'decoded_payload'
    var_3 = b'signature_data'
    var_4 = 100
    var_5 = True

def test_case_0():
    var_0 = b'payload_base64'
    var_1 = 123456789
    var_2 = 'decoded_payload'
    var_3 = b'signature_data'
    var_4 = True

def test_case_0():
    var_0 = 'expired'
    var_1 = b'signature_data'
    var_2 = 10

def test_case_0():
    var_0 = 'bad'
    var_1 = b'payload_base64'
    var_2 = 123456789
    var_3 = 'decoded_payload'
    var_4 = b'signature_data'

def test_case_0():
    var_0 = 'err1'
    var_1 = 'err2'
    var_2 = b'signature_data'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_when_age_is_negative. Retrieved 14/22 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = b'data'
    var_4 = 2000000000
    var_5 = 1000000000
    var_6 = module_1.int_to_bytes(var_4)
    var_7 = module_1.base64_encode(var_6)
    var_8 = b'.'
    var_9 = var_3 + var_8
    var_10 = var_9 + var_7
    var_11 = 10
    var_12 = var_2.unsign(var_10, var_11)
    var_13 = str(var_9)
    var_14 = 'Signature age'
    var_15 = bool('Signature age' in var_13)
    assert var_15 is True



# Parsed testcases at query #16
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b':'
    var_3 = 'hmac'
    var_4 = module_0.TimestampSigner(var_0, var_1, var_2, var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.salt
    assert var_7 == b'salt'
    var_8 = var_4.sep
    assert var_8 == b':'
    var_9 = var_4.key_derivation
    assert var_9 == 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'A'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = ':'
    var_3 = module_0.TimestampSigner(var_0, var_1, var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'secret'])
    assert var_5 is True
    var_6 = var_3.salt
    assert var_6 == b'salt'
    var_7 = var_3.sep
    assert var_7 == b':'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unsign_valid_timestamp_decode. Retrieved 8/21 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 8
    var_3 = 'big'
    var_4 = b'='
    var_5 = b'test_payload'
    var_6 = b'.'
    var_7 = var_5 + var_6



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 3/23 statements.


def test_case_0():
    var_0 = 'bad sig'
    var_1 = b'value.invalid'
    var_2 = b'value.!!!'



# Parsed testcases at query #19
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b':'
    var_3 = 'hmac'
    var_4 = module_0.TimestampSigner(var_0, var_1, var_2, var_3)
    var_5 = var_4.secret_keys
    var_6 = bool(var_4.secret_keys == [b'secret'])
    assert var_6 is True
    var_7 = var_4.sep
    assert var_7 == b':'
    var_8 = var_4.salt
    assert var_8 == b'salt'
    var_9 = var_4.key_derivation
    assert var_9 == 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.secret_keys
    var_5 = bool(var_3.secret_keys == [b'old', b'new'])
    assert var_5 is True
    var_6 = var_3.secret_key
    assert var_6 == b'new'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'A'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_is_malformed. Retrieved 15/42 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = b'value.invalid_base64_!'
    var_3 = b'not-a-number'
    var_4 = 'ascii'
    var_5 = b'value'
    var_6 = b'.'
    var_7 = b'abc'
    var_8 = 0
    var_9 = b'.signature_placeholder'
    var_10 = 0
    var_11 = b'.'
    var_12 = var_1 + var_11
    var_13 = b'not_base64_int'
    var_14 = var_12 + var_13



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 7/37 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Bad Signature'
    var_1 = 'secret'
    var_2 = 'sha1'
    var_3 = module_0.TimestampSigner(var_1, digest_method=var_2)
    var_4 = b'value.YQ=='
    var_5 = var_3.unsign(var_4)
    var_6 = str(var_4)
    var_7 = 'Malformed timestamp'
    var_8 = bool('Malformed timestamp' in var_6)
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unsign_valid_signature. Retrieved 2/5 statements.
# Partially parsed test_unsign_with_timestamp_return. Retrieved 3/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 3/8 statements.
# Partially parsed test_unsign_invalid_signature_raises_bad_timesignature. Retrieved 4/11 statements.
# Partially parsed test_unsign_missing_timestamp_raises_bad_timesignature. Retrieved 3/7 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 3/7 statements.
# Partially parsed test_unsign_future_signature_raises_expired. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hello'
    var_2 = True

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hello'
    var_2 = -1
    var_3 = b'hello'
    var_4 = 'Signature age'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hello'
    var_2 = -5
    var_3 = b'error'

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = b'value.'

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = b'payload.!!!!'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'hello'
    var_2 = -1
    var_3 = b'hello'



# Parsed testcases at query #23
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = b'mysalt'
    var_4 = b'|'
    var_5 = 'hmac'
    var_6 = module_0.TimestampSigner(var_2, var_3, var_4, var_5)
    var_7 = var_6.secret_keys
    var_8 = bool(var_6.secret_keys == [b'old', b'new'])
    assert var_8 is True
    var_9 = var_6.secret_key
    assert var_9 == b'new'
    var_10 = var_6.salt
    assert var_10 == b'mysalt'
    var_11 = var_6.sep
    assert var_11 == b'|'
    var_12 = var_6.key_derivation
    assert var_12 == 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'A'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_loads_return_timestamp_false_skips_tuple_return. Retrieved 6/26 statements.


def test_case_0():
    var_0 = b'payload'
    var_1 = 123456789
    var_2 = 'secret'
    var_3 = 'data'
    var_4 = b'some_signature'
    var_5 = False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.


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
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 1.1
    var_5 = 1
    var_6 = var_1.unsign(var_3, var_5)

def test_case_0():
    pass

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'payload.wrongsignature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = b'.'
    var_4 = b'!!!'
    var_5 = var_2 + var_3
    var_6 = var_5 + var_4
    var_7 = var_6 + var_3
    var_8 = b'signature'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)

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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.validate(var_3)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid'
    var_3 = var_1.validate(var_2)
    assert var_3 is False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_loads_success_payload_only. Retrieved 6/13 statements.
# Partially parsed test_loads_success_with_timestamp. Retrieved 5/11 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 4/11 statements.
# Partially parsed test_loads_raises_bad_signature_on_failure. Retrieved 4/11 statements.
# Partially parsed test_loads_iterates_through_multiple_unsigners. Retrieved 5/14 statements.
# Partially parsed test_loads_passes_max_age_to_signer. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'data'
    var_1 = b'base64_payload'
    var_2 = 123456789
    var_3 = b'signed_data'
    var_4 = None
    var_5 = True

def test_case_0():
    var_0 = 'data'
    var_1 = b'base64_payload'
    var_2 = 123456789
    var_3 = b'signed_data'
    var_4 = True

def test_case_0():
    var_0 = b'signed_data'
    var_1 = 10
    var_2 = 'SignatureExpired should have been raised'
    var_3 = AssertionError(var_2)

def test_case_0():
    var_0 = 'invalid'
    var_1 = b'signed_data'
    var_2 = 'BadSignature should have been raised'
    var_3 = AssertionError(var_2)

def test_case_0():
    var_0 = 'data'
    var_1 = 'bad'
    var_2 = b'base64_payload'
    var_3 = 123456789
    var_4 = b'signed_data'

def test_case_0():
    var_0 = 'data'
    var_1 = b'base64_payload'
    var_2 = 123456789
    var_3 = b'signed_data'
    var_4 = 100
    var_5 = True



