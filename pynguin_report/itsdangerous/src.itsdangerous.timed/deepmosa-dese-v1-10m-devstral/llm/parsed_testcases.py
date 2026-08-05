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
    var_2 = var_1.algorithm

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #2
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
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = 'custom-salt'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unsign_with_return_timestamp_true. Retrieved 5/7 statements.


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
    var_2 = 'invalid_signature'
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
    var_2 = 'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)

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
    var_4 = False
    var_5 = var_1.unsign(var_3, return_timestamp=var_4)
    assert var_5 == b'value'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loads_raises_signature_expired_immediately. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1



# Parsed testcases at query #5
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
    var_6 = b'invalid'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #7
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = signed_value.rsplit(var_5, var_6)[var_4]
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unsign_with_negative_age. Retrieved 15/19 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = 10
    var_4 = var_2 + var_3
    var_5 = b'test_value'
    var_6 = module_1.int_to_bytes(var_4)
    var_7 = module_1.base64_encode(var_6)
    var_8 = b'.'
    var_9 = var_5 + var_8
    var_10 = var_9 + var_7
    var_11 = var_10 + var_8
    var_12 = var_5 + var_8
    var_13 = var_12 + var_7
    var_14 = 5



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)



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
    var_6 = b'invalid'



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)

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
    var_2 = b'hello.invalid_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_parameters. Retrieved 6/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = 'custom-salt'
    var_4 = '|'
    var_5 = 'hmac'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_4 = 3600
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
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_missing_timestamp. Retrieved 6/9 statements.
# Partially parsed test_unsign_with_malformed_timestamp. Retrieved 7/14 statements.


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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'invalid'
    var_6 = var_1.unsign(var_3)

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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_value_error. Retrieved 6/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value.sep.invalid_timestamp'
    var_3 = True
    var_4 = var_1.unsign(var_2, return_timestamp=var_3)
    var_5 = str(var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_loads_raises_signature_expired. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old_key'
    var_1 = b'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #22
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value_without_separator'
    var_3 = 'payload'
    var_4 = hasattr(var_2, var_3)



# Parsed testcases at query #23
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp.sep.signature'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #24
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)
    assert var_3 == b'value'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid'
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
    var_2 = b'value.sep.invalid'
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
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

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
    var_4 = True

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
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'



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

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/11 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm
    var_3 = var_1.get_timestamp()

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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_salt. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_sep. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_key_derivation. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/3 statements.
# Partially parsed test_timestamp_signer_constructor_with_key_list. Retrieved 5/6 statements.


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
    var_3 = var_2.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = var_2.algorithm

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = var_2.algorithm

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = var_3.algorithm



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_params. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = b'secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old_secret'
    var_1 = b'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.TimestampSigner(var_0, var_1)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_params. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old_key'
    var_1 = b'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_unsign_without_timestamp_raises_bad_time_signature. Retrieved 4/5 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.unsign
    var_3 = 'value'



# Parsed testcases at query #33
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
    var_4 = True



# Parsed testcases at query #34
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



# Parsed testcases at query #35
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



# Parsed testcases at query #36
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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)

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
    var_4 = -1
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
    var_2 = b'hello:invalid'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 4/5 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_1._lazy_sha1()
    var_3 = var_1.algorithm

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 'secret-key'
    var_2 = module_0.TimestampSigner(var_1, digest_method=var_0)

import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = lambda x: x
    var_1 = module_0.HMACAlgorithm(var_0)
    var_2 = 'secret-key'
    var_3 = module_1.TimestampSigner(var_2, algorithm=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old_key'
    var_1 = b'new_key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_salt. Retrieved 5/6 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_sep. Retrieved 5/6 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_key_derivation. Retrieved 5/6 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/7 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 2/5 statements.
# Partially parsed test_timestamp_signer_constructor_with_key_rotation. Retrieved 6/7 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_1._lazy_sha1()
    var_3 = var_1.algorithm

import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'custom-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = module_1._lazy_sha1()
    var_4 = var_2.algorithm

import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '|'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = module_1._lazy_sha1()
    var_4 = var_2.algorithm

import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'concat'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)
    var_3 = module_1._lazy_sha1()
    var_4 = var_2.algorithm

def test_case_0():
    var_0 = 'secret-key'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0._lazy_sha1()

import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'old-key'
    var_1 = 'new-key'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)
    var_4 = module_1._lazy_sha1()
    var_5 = var_3.algorithm



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_timestamp_decode_failure. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_oserror. Retrieved 5/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = str(var_2)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_negative_age. Retrieved 2/9 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'test'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)

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
    var_4 = var_1.unsign(var_3)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_unsign_malformed_timestamp. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = b'secret'
    var_1 = b'custom-salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-string'
    var_1 = module_0.TimestampSigner(var_0)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old-secret'
    var_1 = b'new-secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #46
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
    var_4 = True

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
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_timestamp_signer_unsign_with_negative_age. Retrieved 10/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = 100
    var_4 = var_2 + var_3
    var_5 = 'data'
    var_6 = 'time'
    var_7 = 'test'
    var_8 = {var_5: var_7, var_6: var_4}
    var_9 = 0



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/3 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_1._lazy_sha1()
    var_3 = var_1.algorithm

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_timestamp_signer_constructor_with_defaults. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_values. Retrieved 4/9 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_1._lazy_sha1()
    var_3 = var_1.algorithm

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_unsign_malformed_timestamp. Retrieved 7/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'.invalid'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid'
    var_3 = var_1.unsign(var_2)
    assert var_3 == b''

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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)

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
    var_2 = b'value'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.invalid_timestamp'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm



# Parsed testcases at query #54
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
    var_2 = b'test'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = var_1.unsign(var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'invalid_timestamp'
    var_6 = var_4 + var_5
    var_7 = var_1.unsign(var_6)

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
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test'



# Parsed testcases at query #55
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
    var_7 = signed_value.rsplit(var_5, var_6)[var_4]
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

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



# Parsed testcases at query #57
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = -1
    var_3 = 'test'
    var_4 = signer.sign(var_3)[:var_2]
    var_5 = var_1.unsign(var_4)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_custom_params. Retrieved 4/8 statements.
# Partially parsed test_timestamp_signer_constructor_with_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_1._lazy_sha1()
    var_3 = var_1.algorithm

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'

def test_case_0():
    var_0 = b'secret'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old_secret'
    var_1 = b'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 3600
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
    var_2 = b'test'
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



# Parsed testcases at query #3
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'invalid_signature_without_separator'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_with_max_age_not_expired. Retrieved 4/6 statements.
# Partially parsed test_loads_with_max_age_expired. Retrieved 4/7 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 4/7 statements.
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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = 10

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
    var_2 = 'test data'
    var_3 = True

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
    var_3 = 'custom-salt'



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'expired_signature'
    var_3 = 0
    var_4 = var_1.loads(var_2, var_3)



# Parsed testcases at query #6
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)
    var_6 = 100
    var_7 = var_1.timestamp_to_datetime(var_6)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loads_with_valid_signature_and_no_max_age. Retrieved 3/5 statements.
# Partially parsed test_loads_with_valid_signature_and_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 3/6 statements.
# Partially parsed test_loads_with_custom_salt. Retrieved 4/6 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid'
    var_3 = var_1.loads(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = 'custom'



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
    var_2 = b'test.sep.12345.invalid'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

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
    var_4 = 1000
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loads_with_return_timestamp_true. Retrieved 5/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = True
    var_4 = 2



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_timestamp_missing_raises_bad_time_signature. Retrieved 7/13 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.sep
    var_5 = module_1.want_bytes(var_4)
    var_6 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_timestamp. Retrieved 5/7 statements.


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
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.invalid_timestamp'
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
    var_2 = b'hello.1234567890.invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #14
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
    var_2 = b'value'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'invalid_timestamp'
    var_6 = var_4 + var_5
    var_7 = var_6 + var_3
    var_8 = b'signature'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'signature'
    var_6 = var_4 + var_5
    var_7 = var_1.unsign(var_6)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value'
    var_3 = b'.'
    var_4 = var_2 + var_3
    var_5 = b'timestamp'
    var_6 = var_4 + var_5
    var_7 = var_6 + var_3
    var_8 = b'invalid_signature'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_params. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = b'custom-salt'
    var_4 = b'|'
    var_5 = 'hmac'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 6/8 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 7/10 statements.
# Partially parsed test_loads_with_return_timestamp. Retrieved 7/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test_salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test_salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'invalid_data'
    var_4 = var_2.loads(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test_salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = -1

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test_salt'
    var_2 = module_0.TimedSerializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_2 = b'test.sep.invalid_timestamp.sep.signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.sep.signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #18
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
    var_6 = b'invalid'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unsign_malformed_timestamp. Retrieved 11/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = signed_value.rsplit(var_5, var_6)[var_4]
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_loads_with_valid_signature_and_no_max_age. Retrieved 3/5 statements.
# Partially parsed test_loads_with_valid_signature_and_max_age_not_exceeded. Retrieved 4/6 statements.
# Partially parsed test_loads_with_valid_signature_and_return_timestamp. Retrieved 4/7 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/7 statements.
# Partially parsed test_loads_with_bytes_input. Retrieved 4/7 statements.
# Partially parsed test_loads_with_salt. Retrieved 4/6 statements.
# Partially parsed test_loads_with_wrong_salt. Retrieved 5/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = 3600

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'invalid_signature'
    var_3 = var_1.loads(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = 'salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = 'salt'
    var_4 = 'wrong_salt'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unsign_raises_signature_expired_for_negative_age. Retrieved 9/13 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = 100
    var_6 = var_4 + var_5
    var_7 = 50
    var_8 = var_1.unsign(var_3, var_7)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


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
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.invalid_timestamp'
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
    var_2 = b'hello.1234567890.invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old_secret'
    var_1 = b'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_signature_expired_exception_is_raised_immediately. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 1000
    var_5 = var_1.unsign(var_3, var_4)

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
    var_3 = var_1.sep
    var_4 = var_2 + var_3
    var_5 = b'invalid_timestamp'
    var_6 = var_4 + var_5
    var_7 = var_1.sep
    var_8 = var_6 + var_7
    var_9 = b'signature'
    var_10 = var_8 + var_9
    var_11 = var_1.unsign(var_10)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sep
    var_4 = var_2 + var_3
    var_5 = b'signature'
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



# Parsed testcases at query #26
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



# Parsed testcases at query #27
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



# Parsed testcases at query #28
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



# Parsed testcases at query #29
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
    var_7 = signed_value.rsplit(var_5, var_6)[var_4]
    var_8 = b'.invalid_timestamp'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign(var_9)



# Parsed testcases at query #30
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
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)
    assert var_5 == b'test_value'

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
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value:invalid_timestamp:signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value:signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value:timestamp:invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/2 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/3 statements.


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
    var_1 = '|'
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = b'custom_secret'
    var_1 = b'custom_salt'
    var_2 = b'|'
    var_3 = 'hmac'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'old_secret'
    var_1 = b'new_secret'
    var_2 = [var_0, var_1]
    var_3 = module_0.TimestampSigner(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_params. Retrieved 6/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

def test_case_0():
    var_0 = b'old-key'
    var_1 = b'new-key'
    var_2 = [var_0, var_1]
    var_3 = b'custom-salt'
    var_4 = b'|'
    var_5 = 'hmac'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = -1
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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

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
    var_2 = b'test:invalid_timestamp'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #38
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'expired_signature'
    var_3 = 0
    var_4 = var_1.loads(var_2, var_3)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_timestamp_signer_constructor_defaults. Retrieved 3/4 statements.
# Partially parsed test_timestamp_signer_constructor_custom_values. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 4/5 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_digest_method. Retrieved 1/6 statements.
# Partially parsed test_timestamp_signer_constructor_with_custom_algorithm. Retrieved 1/4 statements.


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_1._lazy_sha1()
    var_3 = var_1.algorithm

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
    var_1 = 'a'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_2 = b'test.invalid_timestamp.signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test.timestamp.invalid_signature'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_when_timestamp_is_malformed. Retrieved 11/12 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = b'.'
    var_6 = 1
    var_7 = signed_value.rsplit(var_5, var_6)[var_4]
    var_8 = b'.invalid'
    var_9 = var_7 + var_8
    var_10 = var_1.unsign



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_loads_with_valid_signature. Retrieved 3/5 statements.
# Partially parsed test_loads_with_expired_signature. Retrieved 4/9 statements.
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
    var_2 = 'test data'
    var_3 = 0

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



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_unsign_with_negative_age. Retrieved 9/20 statements.


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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_loads_with_return_timestamp. Retrieved 4/10 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_TimestampSigner_constructor. Retrieved 3/4 statements.
# Partially parsed test_TimestampSigner_constructor_with_custom_digest_method. Retrieved 1/3 statements.
# Partially parsed test_TimestampSigner_constructor_with_custom_algorithm. Retrieved 1/4 statements.


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
    var_1 = 'hmac'
    var_2 = module_0.TimestampSigner(var_0, key_derivation=var_1)

def test_case_0():
    var_0 = 'secret-key'

def test_case_0():
    var_0 = 'secret-key'

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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_unsign_with_return_timestamp_true. Retrieved 5/7 statements.


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
    var_4 = 3600
    var_5 = var_1.unsign(var_3, var_4)

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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
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
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #49
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



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_timestamp_to_datetime_raises_os_error. Retrieved 5/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'invalid.timestamp'
    var_3 = var_1.unsign(var_2)
    var_4 = str(var_2)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_loads_with_return_timestamp. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #53
#--------------------------




import src.itsdangerous.signer as module_0
import src.itsdangerous.timed as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'test_salt'
    var_2 = b'|'
    var_3 = 'hmac'
    var_4 = 'hashlib'
    var_5 = __import__(var_4)
    var_6 = lambda x: var_1.sha256(x)
    var_7 = module_0.HMACAlgorithm(var_6)
    var_8 = module_1.TimestampSigner(var_0, var_1, var_2, var_3, var_6, var_7)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_unsign_with_return_timestamp_true. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 100
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



# Parsed testcases at query #55
#--------------------------




import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'value.sep.invalid_timestamp'
    var_3 = var_1.unsign(var_2)



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

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 100
    var_5 = var_1.unsign(var_3, var_4)

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
    var_2 = b'test.invalid_timestamp'
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

# Partially parsed test_returns_payload_and_timestamp_when_return_timestamp_is_true. Retrieved 4/8 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_unsign_with_valid_signature_and_return_timestamp. Retrieved 5/7 statements.
# Partially parsed test_unsign_with_future_timestamp. Retrieved 7/11 statements.


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
    var_2 = b'hello.sep.invalid_timestamp.sep.signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.sep.signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello.sep.timestamp.sep.invalid_signature'
    var_3 = var_1.unsign(var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = 1000
    var_4 = var_1.sign(var_2)
    var_5 = 0
    var_6 = var_1.unsign(var_4, var_5)



# Parsed testcases at query #60
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
    var_3 = 'utf-8'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test data'
    var_3 = 'custom-salt'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)

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
    var_2 = b'test'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

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
    var_4 = True

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
    var_2 = b'test.malformed'
    var_3 = var_1.unsign(var_2)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_unsign_with_return_timestamp. Retrieved 5/7 statements.


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
    var_2 = 'invalid'
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
    var_2 = b'hello.sep.invalidts'
    var_3 = var_1.unsign(var_2)

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
    var_4 = 0
    var_5 = var_1.unsign(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello'
    var_3 = var_1.sign(var_2)
    var_4 = True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_loads_with_signature_expired_raises_immediately. Retrieved 4/7 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'data'
    var_3 = -1



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_unsign_raises_bad_time_signature_for_malformed_timestamp. Retrieved 7/14 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = 0
    var_5 = 1
    var_6 = b'invalid'



# Parsed testcases at query #65
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
    var_2 = 'invalid_signature'
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
    var_5 = var_3[:var_4]
    var_6 = b'X'
    var_7 = var_5 + var_6
    var_8 = var_1.unsign(var_7)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'value'
    var_3 = var_1.sign(var_2)
    var_4 = -1
    var_5 = var_1.unsign(var_3, var_4)



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_timestamp_signer_constructor. Retrieved 3/4 statements.


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.algorithm



