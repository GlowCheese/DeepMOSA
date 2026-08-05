####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = b'mysalt'
    var_4 = b':'
    var_5 = 'hmac'
    var_6 = module_0.TimestampSigner(var_2, var_3, var_4, var_5)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loads_success. Retrieved 4/8 statements.
# Partially parsed test_loads_with_timestamp. Retrieved 5/10 statements.
# Partially parsed test_loads_expired_raises_error. Retrieved 7/14 statements.
# Partially parsed test_loads_bad_signature_raises_error. Retrieved 8/16 statements.
# Partially parsed test_loads_with_salt. Retrieved 5/9 statements.
# Partially parsed test_loads_bytes_input. Retrieved 4/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = -1
    var_5 = 'SignatureExpired not raised'
    var_6 = AssertionError(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = -5
    var_5 = b'xxxxx'
    var_6 = 'BadSignature not raised'
    var_7 = AssertionError(var_6)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'my-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_unsign_future_signature_error. Retrieved 9/26 statements.
# Partially parsed test_unsign_malformed_timestamp. Retrieved 10/14 statements.
# Partially parsed test_unsign_missing_timestamp. Retrieved 5/9 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000
    var_3 = b'.'
    var_4 = b'hello'
    var_5 = var_4 + var_3
    var_6 = 100
    var_7 = 'age -1000'
    var_8 = 'age < 0'

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
    var_2 = b'.'
    var_3 = b'hello'
    var_4 = b'not-base64!!!'
    var_5 = var_3 + var_2
    var_6 = var_5 + var_4
    var_7 = var_3 + var_2
    var_8 = var_7 + var_4
    var_9 = var_8 + var_2

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = b'.'
    var_4 = var_2 + var_3

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.validate(var_3)
    assert var_4 is True

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
    var_8 = var_1.validate(var_7)
    assert var_8 is False



# Parsed testcases at query #4
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'test-payload'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.sign(var_2)
    var_5 = 30
    var_6 = var_1.unsign(var_4, var_5)
    var_7 = 'SignatureExpired was not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_invalid_signature. Retrieved 2/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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

def test_case_0():
    var_0 = 'secret'
    var_1 = b'payload.incorrect_signature'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'payload.invalid_ts_!!!'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'just_payload_no_sep'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'valid'
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



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = b'hello'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loads_success. Retrieved 5/17 statements.
# Partially parsed test_loads_return_timestamp. Retrieved 4/15 statements.
# Partially parsed test_loads_signature_expired. Retrieved 5/15 statements.
# Partially parsed test_loads_bad_signature_raises_exception. Retrieved 5/16 statements.
# Partially parsed test_loads_with_salt. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'data'
    var_1 = 'payload'
    var_2 = 0
    var_3 = b'base64encodedpayload'
    var_4 = b'some_signature'

def test_case_0():
    var_0 = 0
    var_1 = b'payload'
    var_2 = b'sig'
    var_3 = True

def test_case_0():
    var_0 = 0
    var_1 = 'expired'
    var_2 = b'sig'
    var_3 = 'SignatureExpired should have been raised'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = 'bad 1'
    var_1 = 'bad 2'
    var_2 = b'sig'
    var_3 = 'BadSignature should have been raised'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = 0
    var_1 = b'p'
    var_2 = 123
    var_3 = b'sig'
    var_4 = 'my_salt'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 7/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 1.1
    var_5 = 0
    var_6 = var_1.unsign(var_3, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'not-a-valid-signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'.'
    var_3 = b'data'
    var_4 = b'not-base64-at-all!!!'
    var_5 = var_3 + var_2
    var_6 = var_5 + var_4
    var_7 = var_6 + var_2
    var_8 = b'signature'
    var_9 = var_7 + var_8

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.validate(var_3)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'invalid-signature'
    var_3 = var_1.validate(var_2)
    assert var_3 is False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unsign_raises_bad_timesignature_on_malformed_timestamp. Retrieved 16/89 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'Invalid timestamp'
    var_1 = ValueError(var_0)
    var_2 = 'secret'
    var_3 = 'salt'
    var_4 = 's'
    var_5 = b'data.'
    var_6 = b'\x01'
    var_7 = module_0.int_to_bytes(var_6)
    var_8 = module_0.base64_encode(var_7)
    var_9 = 's'
    var_10 = b'\x01'
    var_11 = module_0.int_to_bytes(var_10)
    var_12 = module_0.base64_encode(var_11)
    var_13 = b'content.'
    var_14 = var_13 + var_12
    var_15 = b'some_value'



# Parsed testcases at query #10
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = module_0.TimestampSigner(var_1)
    var_3 = 1000
    var_4 = 2000
    var_5 = module_0.TimestampSigner()
    var_6 = 2000
    var_7 = b'test'
    var_8 = b'.'
    var_9 = b'test'
    var_10 = var_5.sign(var_9)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unsign_malformed_timestamp_raises_bad_time_signature. Retrieved 13/26 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = b'payload'
    var_4 = b'.'
    var_5 = var_3 + var_4
    var_6 = b'not-a-number'
    var_7 = 'secret'
    var_8 = module_0.TimestampSigner(var_7)
    var_9 = b'payload'
    var_10 = var_8.sign(var_9)
    var_11 = var_8.unsign(var_10)
    var_12 = str(var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unsign_with_valid_timestamp. Retrieved 8/15 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'payload'
    var_3 = 8
    var_4 = 'big'
    var_5 = b'.'
    var_6 = var_1.sign(var_2)
    var_7 = var_1.unsign(var_6)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unsign_with_valid_timestamp_bytes. Retrieved 5/17 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'payload'
    var_3 = b'.'
    var_4 = var_2 + var_3



# Parsed testcases at query #14
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'|'
    var_3 = 'hmac'
    var_4 = None
    var_5 = module_0.TimestampSigner(var_0, var_1, var_2, var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b':'
    var_3 = 'hmac'
    var_4 = module_0.TimestampSigner(var_0, var_1, var_2, var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)



# Parsed testcases at query #16
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = b'mysalt'
    var_4 = b':'
    var_5 = 'hmac'
    var_6 = None
    var_7 = module_0.TimestampSigner(var_2, var_3, var_4, var_5, var_6)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'A'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_loads_returns_payload_when_return_timestamp_is_false. Retrieved 6/29 statements.


def test_case_0():
    var_0 = b'payload_data'
    var_1 = 123456789
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 'some_data'
    var_5 = False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_loads_returns_payload_when_return_timestamp_is_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = b'payload_base64'
    var_1 = 123456789
    var_2 = 'data'
    var_3 = 'test'
    var_4 = 'encoded_string'
    var_5 = False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/8 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 8/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = 1.1
    var_6 = 0
    var_7 = var_1.unsign(var_3, var_6)

def test_case_0():
    pass

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -5
    var_5 = var_3[:var_4]
    var_6 = b'xxxxx'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)

import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = '.'
    var_3 = module_1.Signer(sep=var_2)
    var_4 = b'data'
    var_5 = var_3.sign(var_4)
    var_6 = var_1.unsign(var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'.'
    var_3 = b'data'
    var_4 = b'!!!'
    var_5 = b'signature'
    var_6 = var_3 + var_2
    var_7 = var_6 + var_4
    var_8 = var_7 + var_2
    var_9 = var_8 + var_5
    var_10 = var_1.unsign(var_9)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.validate(var_3)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = b'extra'
    var_5 = var_3 + var_4
    var_6 = var_1.validate(var_5)
    assert var_6 is False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unsign_timestamp_decoding_success. Retrieved 15/35 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_payload'
    var_3 = 8
    var_4 = 'big'
    var_5 = True
    var_6 = b'='
    var_7 = b'.'
    var_8 = module_1.Signer(var_0)
    var_9 = b''
    var_10 = b'.'
    var_11 = var_2 + var_10
    var_12 = -1
    var_13 = var_15.split(var_10)[var_12]
    var_14 = var_2 + var_10



# Parsed testcases at query #21
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = b'mysalt'
    var_4 = b'|'
    var_5 = 'hmac'
    var_6 = module_0.TimestampSigner(var_2, var_3, var_4, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret_string'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_loads_returns_payload_directly_when_return_timestamp_is_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = b'payload_bytes'
    var_1 = 123456789
    var_2 = 'some_signature'
    var_3 = False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_timestamp_signer_constructor_invalid_separator. Retrieved 3/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = b'mysalt'
    var_4 = b':'
    var_5 = 'hmac'
    var_6 = 'sha256'
    var_7 = module_0.TimestampSigner(var_2, var_3, var_4, var_5, var_6)

def test_case_0():
    var_0 = b'a'
    var_1 = 'ascii'
    var_2 = b'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret_string'
    var_1 = 'salt_string'
    var_2 = module_0.TimestampSigner(var_0, var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/8 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 10/24 statements.


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
    var_4 = 100
    var_5 = b'.'
    var_6 = b'hello'
    var_7 = var_6 + var_5
    var_8 = b'invalid_signature'
    var_9 = 10

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.notbase64!.signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.signature'
    var_3 = var_1.unsign(var_2)

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loads_success. Retrieved 8/17 statements.
# Partially parsed test_loads_without_timestamp. Retrieved 7/15 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 5/14 statements.
# Partially parsed test_loads_raises_bad_signature_on_all_signers. Retrieved 5/16 statements.
# Partially parsed test_loads_with_salt. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'data'
    var_1 = 'payload'
    var_2 = {var_0: var_1}
    var_3 = b'base64_payload'
    var_4 = 123456789
    var_5 = b'encoded_string'
    var_6 = True
    var_7 = None

def test_case_0():
    var_0 = 'data'
    var_1 = 'payload'
    var_2 = {var_0: var_1}
    var_3 = b'base64_payload'
    var_4 = 123456789
    var_5 = b'encoded_string'
    var_6 = False

def test_case_0():
    var_0 = 'expired'
    var_1 = b'encoded_string'
    var_2 = 10
    var_3 = 'SignatureExpired should have been raised'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = 'bad 1'
    var_1 = 'bad 2'
    var_2 = b'encoded_string'
    var_3 = 'BadSignature should have been raised'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = 'payload'
    var_1 = b'base64'
    var_2 = 123
    var_3 = b'encoded_string'
    var_4 = 'my_salt'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unsign_with_valid_timestamp. Retrieved 8/16 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello'
    var_3 = b'='
    var_4 = b'.'
    var_5 = b'fake_sig'
    var_6 = var_1.sign(var_2)
    var_7 = var_1.unsign(var_6)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unsign_age_less_than_zero_raises_signature_expired. Retrieved 15/36 statements.
# Partially parsed test_unsign_future_timestamp_logic. Retrieved 11/25 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = 100
    var_4 = b'data'
    var_5 = b'.'
    var_6 = b'data'
    var_7 = b'.'
    var_8 = var_6 + var_7
    var_9 = b'.signature'
    var_10 = b'data.future_ts.signature'
    var_11 = 500
    var_12 = var_2.unsign(var_10, var_11)
    var_13 = 'SignatureExpired was not raised for future timestamp'
    var_14 = AssertionError(var_13)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1000
    var_3 = b'data'
    var_4 = b'.'
    var_5 = var_3 + var_4
    var_6 = b'dummy_signed_value'
    var_7 = 100
    var_8 = var_1.unsign(var_6, var_7)
    var_9 = 'Did not trigger line 77'
    var_10 = AssertionError(var_9)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_future_signature_raises_error. Retrieved 5/13 statements.


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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.invalid_sig'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.notbase64!!!'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'no_separator_here'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 100
    var_3 = b'hello'
    var_4 = b'.'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.validate(var_3)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.badsignature'
    var_3 = var_1.validate(var_2)
    assert var_3 is False



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = b'mysalt'
    var_4 = b'|'
    var_5 = 'hmac'
    var_6 = module_0.TimestampSigner(var_2, var_3, var_4, var_5)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-string'
    var_1 = 'salt-string'
    var_2 = module_0.TimestampSigner(var_0, var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_timestamp_signer_constructor_invalid_separator. Retrieved 2/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'mysalt'
    var_2 = b':'
    var_3 = 'hmac'
    var_4 = module_0.TimestampSigner(var_0, var_1, var_2, var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

def test_case_0():
    var_0 = b'abc'
    var_1 = b'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'string_key'
    var_1 = 'salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unsign_sep_in_result_evaluates_predicate_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'data'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loads_success_with_timestamp. Retrieved 8/17 statements.
# Partially parsed test_loads_success_without_timestamp. Retrieved 6/14 statements.
# Partially parsed test_loads_raises_signature_expired. Retrieved 5/14 statements.
# Partially parsed test_loads_raises_bad_signature_on_all_signers. Retrieved 5/16 statements.
# Partially parsed test_loads_with_max_age_parameter. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 123456789
    var_4 = b'base64payload'
    var_5 = b'signature'
    var_6 = True
    var_7 = None

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = b'base64payload'
    var_4 = 12345
    var_5 = b'signature'

def test_case_0():
    var_0 = 'expired'
    var_1 = b'signature'
    var_2 = 10
    var_3 = 'SignatureExpired was not raised'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = 'bad 1'
    var_1 = 'bad 2'
    var_2 = b'signature'
    var_3 = 'BadSignature was not raised'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = b'payload'
    var_1 = 123
    var_2 = 'data'
    var_3 = b'signature'
    var_4 = 60
    var_5 = True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/7 statements.
# Partially parsed test_unsign_expired_signature. Retrieved 6/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'hello'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = 1.1
    var_5 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'hello.invalid_sig'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = b'payload.notbase64!!!'
    var_4 = var_2.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = b'noparameterhere'
    var_4 = var_2.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = 'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.validate(var_3)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'not_signed'
    var_3 = var_1.validate(var_2)
    assert var_3 is False



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b':'
    var_3 = 'hmac'
    var_4 = module_0.TimestampSigner(var_0, var_1, var_2, var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_loads_returns_payload_when_return_timestamp_is_false. Retrieved 6/15 statements.


def test_case_0():
    var_0 = b'payload_base64'
    var_1 = 123456789
    var_2 = 'data'
    var_3 = 'test'
    var_4 = 'some_signature'
    var_5 = False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_loads_returns_payload_when_return_timestamp_is_false. Retrieved 10/21 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'base64_data'
    var_1 = 123456789
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 'input_string'
    var_5 = 100
    var_6 = 'test_salt'
    var_7 = False
    var_8 = module_0.want_bytes(var_4)
    var_9 = True



# Parsed testcases at query #15
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_2 = b'data'
    var_3 = var_1.sign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'data.invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = b'payload.!!!'
    var_4 = var_2.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = b'no_separator_here'
    var_4 = b'just_payload_no_sep'
    var_5 = var_2.unsign(var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'valid'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.validate(var_3)
    assert var_4 is True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'invalid_signature'
    var_3 = var_1.validate(var_2)
    assert var_3 is False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unsign_avoids_exception_on_valid_timestamp_conversion. Retrieved 14/46 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1600000000
    var_3 = 8
    var_4 = 'big'
    var_5 = b'='
    var_6 = b'data'
    var_7 = b'.'
    var_8 = var_6 + var_7
    var_9 = 'bad sig'
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = 1000
    var_12 = b'value'
    var_13 = var_12 + var_7



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unsign_with_timestamp_return. Retrieved 5/7 statements.


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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'data.invalid_sig'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'payload.notbase64!!!'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'payloadwithoutsep'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unsign_handles_bad_signature_with_valid_timestamp_without_exception. Retrieved 16/55 statements.


import src.itsdangerous.exc as module_0

def test_case_0():
    var_0 = 1234567890
    var_1 = 8
    var_2 = 'big'
    var_3 = b'some_value'
    var_4 = b'.'
    var_5 = var_3 + var_4
    var_6 = 'Invalid signature'
    var_7 = module_0.BadSignature(var_6)
    var_8 = 1600000000
    var_9 = b'data'
    var_10 = var_9 + var_4
    var_11 = 'Bad Signature'
    var_12 = module_0.BadSignature(var_11)
    var_13 = 'secret'
    var_14 = None
    var_15 = b'some_signed_value'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_when_age_is_negative. Retrieved 12/32 statements.


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 1000
    var_1 = 2000
    var_2 = 8
    var_3 = 'big'
    var_4 = b'data. '
    var_5 = b'='
    var_6 = b''
    var_7 = module_0.int_to_bytes(var_1)
    var_8 = module_0.base64_encode(var_7)
    var_9 = b'my_value.'
    var_10 = var_9 + var_8
    var_11 = 5000



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_is_malformed. Retrieved 10/17 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner()
    var_2 = b'value'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'invalid_base64_!!!'
    var_6 = var_4 + var_5
    var_7 = b'some_signed_value'
    var_8 = var_1.unsign(var_7)
    var_9 = str(var_7)



