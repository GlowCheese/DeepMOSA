####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the loads method of TimedSerializer covering:\n    1. Successful loading (returning payload).\n    2. Successful loading with return_timestamp=True.\n    3. SignatureExpired exception handling.\n    4. BadSignature exception handling when multiple signers are present.\n    5. Integration with payload loading logic.\n    '
    var_1 = b'hello-world'
    var_2 = 1600000000
    var_3 = module_0.TimedSerializer()
    var_4 = b'decoded-data'
    var_5 = b'some-signed-blob'
    var_6 = var_3.loads(var_5)
    assert var_6 == b'decoded-data'
    var_7 = None
    var_8 = True
    var_9 = var_3.loads(var_5, return_timestamp=var_8)
    var_10 = 'Expired'
    var_11 = b'old'
    var_12 = var_3.loads(var_5)
    var_13 = b'second-payload'
    var_14 = 'Bad signature'
    var_15 = b'bad'
    var_16 = var_3.loads(var_5)
    assert var_16 == b'second-decoded'
    var_17 = 'Bad 1'
    var_18 = b'1'
    var_19 = 'Bad 2'
    var_20 = b'2'
    var_21 = var_3.loads(var_5)
    var_22 = 60
    var_23 = var_3.loads(var_5, var_22)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = 'signer.Signer'
    var_3 = module_0.TimestampSigner(sep=var_1)
    var_4 = var_3.get_timestamp
    var_5 = callable(var_4)
    var_6 = var_3.timestamp_to_datetime
    var_7 = callable(var_6)
    var_8 = var_3.sign
    var_9 = callable(var_8)
    var_10 = var_3.unsign
    var_11 = callable(var_10)
    var_12 = var_3.validate
    var_13 = callable(var_12)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = "\n    Tests the initialization and basic structure of TimedSerializer.\n    Since the constructor doesn't take specific arguments for logic \n    in this implementation, we verify it instantiates correctly \n    and inherits the expected default signer.\n    "



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Signer'
    var_1 = 'secret-key'
    var_2 = '.'
    var_3 = module_0.TimestampSigner(sep=var_2)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the unsign method of TimestampSigner covering various scenarios:\n    valid signature, expired signature, malformed timestamp, and bad signature.\n    '
    var_1 = b'.'
    var_2 = b'hello_world'
    var_3 = True
    var_4 = 1000
    var_5 = 100
    var_6 = 50
    var_7 = 1000
    var_8 = 10
    var_9 = 50
    var_10 = b'not-base64-encoded-properly!@#$'
    var_11 = 0
    var_12 = 2
    var_13 = b'just_a_string_no_separators'
    var_14 = b'tampered'
    var_15 = var_14 + var_1



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the instantiation and basic properties of TimedSerializer.\n    Since the constructor of TimedSerializer is inherited from Serializer \n    and does not take specific arguments in its definition, we verify \n    it initializes correctly with expected default attributes.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = 'loads'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'dumps'
    var_5 = hasattr(var_1, var_4)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the initialization of TimedSerializer by verifying it \n    correctly inherits and sets up its default signer.\n    '
    var_1 = b'secret-key'
    var_2 = 'signer'
    var_3 = '_signer'



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'.'
    var_3 = b'hello-world'
    var_4 = var_1.sign(var_3)
    var_5 = module_0.TimestampSigner(var_0)
    var_6 = var_5.sign(var_3)
    var_7 = True
    var_8 = 1000
    var_9 = var_1.unsign(var_6)
    var_10 = 60
    var_11 = var_1.unsign(var_6, var_10)
    var_12 = 60
    var_13 = var_1.unsign(var_6, var_12)
    var_14 = str(var_12)
    var_15 = 60
    var_16 = var_1.unsign(var_6, var_15)
    var_17 = str(var_15)
    var_18 = b'wrong-signature-data'
    var_19 = var_1.unsign(var_18)
    var_20 = var_3 + var_2
    var_21 = b'not-base64-or-not-int!!'
    var_22 = var_20 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = str(var_23)
    var_25 = b'just-payload-no-separator'
    var_26 = var_1.unsign(var_25)
    var_27 = str(var_26)
    var_28 = var_1.validate(var_6)
    assert var_28 is True
    var_29 = 10
    var_30 = var_1.validate(var_6, var_29)
    assert var_30 is False
    var_31 = b'invalid'
    var_32 = var_1.validate(var_31)



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello-world'
    var_3 = 1700000000
    var_4 = var_1.sign(var_2)
    var_5 = var_1.unsign(var_4)
    var_6 = True
    var_7 = 100
    var_8 = var_1.unsign(var_4, var_7)
    var_9 = 10
    var_10 = var_1.unsign(var_4, var_9)
    var_11 = var_3 + var_7
    var_12 = module_1.int_to_bytes(var_11)
    var_13 = module_1.base64_encode(var_12)
    var_14 = 10
    var_15 = 'negative'
    var_16 = ' < 0'
    var_17 = b'tampered'
    var_18 = -5
    var_19 = var_4[:var_18]
    var_20 = b'wrong'
    var_21 = var_19 + var_20
    var_22 = var_1.unsign(var_21)
    var_23 = b'not-base64-!!'
    var_24 = b'no-separator-here'
    var_25 = var_2 + var_24
    var_26 = var_1.unsign(var_25)
    var_27 = b'invalid-signature-bits'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'valid'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.validate(var_3)
    assert var_4 is True
    var_5 = 10000
    var_6 = var_1.validate(var_3, var_5)
    assert var_6 is True
    var_7 = 1
    var_8 = var_1.validate(var_3, var_7)
    assert var_8 is False
    var_9 = b'invalid-data'
    var_10 = var_1.validate(var_9)
    assert var_10 is False



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = '\n    Tests the loads method of TimedSerializer covering:\n    1. Successful loading (returning payload).\n    2. Successful loading with return_timestamp=True.\n    3. Signature expiration (SignatureExpired).\n    4. Bad signature (BadSignature).\n    5. Multiple signers handling.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = b'original_payload'
    var_3 = 1600000000
    var_4 = 'decoded_data'
    var_5 = b'valid_signed_string'
    var_6 = var_1.loads(var_5)
    assert var_6 == 'decoded_data'
    var_7 = b'valid_signed_string'
    var_8 = True
    var_9 = 'Expired'
    var_10 = b'expired_payload'
    var_11 = b'expired_string'
    var_12 = 10
    var_13 = var_1.loads(var_11, var_12)
    var_14 = b'second_payload'
    var_15 = 'second_decoded'
    var_16 = 'Bad sig'
    var_17 = b'bad_payload'
    var_18 = b'multi_signer_string'
    var_19 = var_1.loads(var_18)
    assert var_19 == 'second_decoded'
    var_20 = 'Final failure'
    var_21 = b'failed'
    var_22 = module_1.BadSignature(var_20, var_21)
    var_23 = b'all_fail_string'
    var_24 = var_1.loads(var_23)
    var_25 = str(var_23)
    var_26 = b'salt_test'
    var_27 = 'my_salt'
    var_28 = var_1.loads(var_26, salt=var_27)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = b'hello-world'
    var_4 = var_2.sign(var_3)
    var_5 = var_4
    var_6 = var_2.unsign(var_5)
    var_7 = True
    var_8 = 1000000
    var_9 = var_2.sign(var_3)
    var_10 = 50
    var_11 = var_2.unsign(var_9, var_10)
    var_12 = str(var_10)
    var_13 = 200
    var_14 = var_2.unsign(var_9, var_13)
    var_15 = b'hello'
    var_16 = b'hallo'
    var_17 = b'payload.notbase64!!!'
    var_18 = var_2.unsign(var_17)
    var_19 = b'payload.notbase64'
    var_20 = b'payload.'
    var_21 = b'notbase64_encoded_garbage'
    var_22 = var_20 + var_21
    var_23 = var_2.unsign(var_22)
    var_24 = str(var_20)
    var_25 = b'no_separator_at_all'
    var_26 = var_2.unsign(var_25)
    var_27 = var_2.validate(var_4)
    assert var_27 is True



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the initialization and core properties of TimedSerializer.\n    Since the constructor is inherited from Serializer, we verify \n    it correctly sets up the default signer as TimestampSigner.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = 'iter_unsigners'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the initialization and basic properties of TimestampSigner.\n    Since TimestampSigner inherits from Signer, we verify it can be \n    instantiated with a secret and maintains expected attributes.\n    '
    var_1 = b'secret-key'
    var_2 = b'.'
    var_3 = module_0.TimestampSigner(sep=var_2)
    var_4 = var_3.get_timestamp()

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Tests that the constructor handles string separators if passed.'
    var_1 = b'key'
    var_2 = '.'
    var_3 = module_0.TimestampSigner(sep=var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Tests that timestamp_to_datetime produces UTC aware datetimes.'
    var_1 = b'key'
    var_2 = module_0.TimestampSigner()
    var_3 = 1609459200
    var_4 = var_2.timestamp_to_datetime(var_3)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Test the construction and basic properties of TimedSerializer.\n    Since the constructor is inherited from Serializer, we verify \n    that it initializes with expected attributes and that its \n    default signer is correctly set to TimestampSigner.\n    '
    var_1 = b'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'loads'
    var_4 = hasattr(var_2, var_3)
    var_5 = 'dumps'
    var_6 = hasattr(var_2, var_5)



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = b'data'
    var_2 = 2023
    var_3 = 1
    var_4 = None
    var_5 = True
    var_6 = True
    var_7 = 60
    var_8 = True
    var_9 = 'expired'
    var_10 = 'bad'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = b'data'
    var_2 = 2020
    var_3 = 1
    var_4 = 'data'
    var_5 = 'my_salt'
    var_6 = b'some_signed_value'
    var_7 = var_0.loads(var_6, salt=var_5)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = '\n    Test the construction and basic initialization of TimestampSigner.\n    Since TimestampSigner inherits from Signer, we verify it can be \n    instantiated and maintains expected signer attributes.\n    '
    var_1 = b'super-secret-key'
    var_2 = b'.'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the loads method of TimedSerializer.\n    Covers successful decryption/unsigning and handles return_timestamp logic.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = b'encoded_data'
    var_3 = 1600000000
    var_4 = b'signed_value'
    var_5 = None
    var_6 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Tests that SignatureExpired is raised and not swallowed by the loop.'
    var_1 = module_0.TimedSerializer()
    var_2 = 'Expired'
    var_3 = b'old'
    var_4 = b'some_value'
    var_5 = var_1.loads(var_4)

import src.itsdangerous.exc as module_0

def test_case_0():
    var_0 = 'Tests that BadSignature is raised if all signers fail.'
    var_1 = 'fail 1'
    var_2 = b'p1'
    var_3 = module_0.BadSignature(var_1, var_2)
    var_4 = 'fail 2'
    var_5 = b'p2'
    var_6 = module_0.BadSignature(var_4, var_5)
    var_7 = b'invalid_value'



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = b'hello-world'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = True
    var_7 = var_2.sign(var_3)
    var_8 = 500
    var_9 = var_2.unsign(var_7, var_8)
    var_10 = str(var_8)
    var_11 = var_2.sign(var_3)
    var_12 = 500
    var_13 = var_2.unsign(var_11, var_12)
    var_14 = str(var_12)
    var_15 = b'tampered'
    var_16 = len(var_15)
    var_17 = var_4[var_16:]
    var_18 = var_15 + var_17
    var_19 = var_2.unsign(var_18)
    var_20 = b'only-payload'
    var_21 = var_20 + var_1
    var_22 = b'no-timestamp-here'
    var_23 = var_2.unsign(var_22)
    var_24 = var_3 + var_1
    var_25 = b'not-base64-and-not-int!!!'
    var_26 = var_24 + var_25
    var_27 = var_2.sign(var_3)
    var_28 = var_3 + var_1
    var_29 = b'invalid_b64_data'
    var_30 = var_28 + var_29
    var_31 = var_30 + var_1
    var_32 = var_3 + var_1
    var_33 = var_32 + var_29
    var_34 = var_2.validate(var_4)
    assert var_34 is True
    var_35 = var_2.validate(var_18)
    assert var_35 is False
    var_36 = var_2.sign(var_3)
    var_37 = 10
    var_38 = var_2.validate(var_36, var_37)
    assert var_38 is False



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'.'
    var_3 = b'hello-world'
    var_4 = 1000000
    var_5 = True
    var_6 = 100
    var_7 = var_4 - var_6
    var_8 = 50
    var_9 = str(var_6)
    var_10 = var_4 + var_6
    var_11 = str(var_6)
    var_12 = str(var_6)
    var_13 = var_3 + var_2
    var_14 = b'not-base64-int'
    var_15 = var_13 + var_14
    var_16 = var_15 + var_2
    var_17 = b'signature'
    var_18 = var_16 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = str(var_19)
    var_21 = b'something-extra'
    assert var_21 is True
    assert var_21 is False
    var_22 = var_3 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = str(var_21)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = b'super-secret-key'
    var_1 = b'.'
    var_2 = b'|'
    var_3 = 'get_timestamp'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = "\n    Tests the initialization and basic properties of a TimestampSigner instance.\n    Since TimestampSigner inherits from Signer, we verify it can be instantiated\n    and maintains standard Signer attributes like 'sep'.\n    "
    var_1 = b'super-secret-key'
    var_2 = b'.'

def test_case_0():
    var_0 = 'Verifies that TimestampSigner correctly inherits from Signer.'
    var_1 = b'key'



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = var_2.get_timestamp()
    var_4 = var_2.timestamp_to_datetime(var_3)



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello-world'
    var_3 = b'.'
    var_4 = 1700000000
    var_5 = var_1.sign(var_2)
    var_6 = var_1.unsign(var_5)
    var_7 = True
    var_8 = 100
    var_9 = var_1.unsign(var_5, var_8)
    var_10 = 100
    var_11 = var_1.unsign(var_5, var_10)
    var_12 = str(var_10)
    var_13 = 10
    var_14 = var_1.unsign(var_5, var_13)
    var_15 = b'tampered'
    var_16 = len(var_2)
    var_17 = var_5[var_16:]
    var_18 = var_15 + var_17
    var_19 = -5
    var_20 = var_5[:var_19]
    var_21 = b'wrong'
    var_22 = var_20 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = 0
    var_25 = b'not-base64-!!!'
    var_26 = 2
    var_27 = b'no_separator_here'
    var_28 = var_1.unsign(var_27)
    var_29 = var_1.sign(var_2)
    var_30 = var_1.unsign(var_29)
    var_31 = var_1.validate(var_5)
    assert var_31 is True
    var_32 = 10
    var_33 = var_1.validate(var_5, var_32)
    assert var_33 is False



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = '\n    Tests the loads method of TimedSerializer covering:\n    1. Successful loading of payload.\n    2. Returning timestamp when return_timestamp=True.\n    3. Raising SignatureExpired when max_age is exceeded.\n    4. Raising BadSignature when signature is invalid.\n    5. Iterating through multiple unsigners (salts).\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = b'encoded_payload'
    var_6 = b'some_signed_data'
    var_7 = var_1.loads(var_6)
    var_8 = None
    var_9 = True
    var_10 = b'encoded_payload'
    var_11 = b'some_signed_data'
    var_12 = True
    var_13 = 'Expired'
    var_14 = b'data'
    var_15 = b'some_signed_data'
    var_16 = 10
    var_17 = var_1.loads(var_15, var_16)
    var_18 = 'Invalid'
    var_19 = b'bad'
    var_20 = b'good_payload'
    var_21 = 'success'
    var_22 = True
    var_23 = b'some_signed_data'
    var_24 = var_1.loads(var_23)
    var_25 = 'Final failure'
    var_26 = b'none'
    var_27 = module_1.BadSignature(var_25, var_26)
    var_28 = b'some_signed_data'
    var_29 = var_1.loads(var_28)
    var_30 = str(var_20)
    assert var_30 == 'Final failure'
    var_31 = b'p'
    var_32 = b'data'
    var_33 = b'my_salt'
    var_34 = var_1.loads(var_32, salt=var_33)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = b'hello-world'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = True
    var_7 = 1000000
    var_8 = var_2.sign(var_3)
    var_9 = 50
    var_10 = var_2.unsign(var_8, var_9)
    var_11 = var_2.sign(var_3)
    var_12 = 50
    var_13 = var_2.unsign(var_11, var_12)
    var_14 = b'hello'
    var_15 = b'bad'
    var_16 = var_3 + var_1
    var_17 = b'not-base64-valid-timestamp!!!'
    var_18 = var_16 + var_17
    var_19 = var_2.unsign(var_18)
    var_20 = b'no-separator-here'
    var_21 = var_2.unsign(var_20)
    var_22 = module_1.int_to_bytes(var_7)
    var_23 = module_1.base64_encode(var_22)
    var_24 = var_3 + var_1
    var_25 = var_24 + var_23
    var_26 = var_25 + var_1
    var_27 = b'invalid-signature'
    var_28 = var_26 + var_27
    var_29 = var_2.unsign(var_28)
    var_30 = var_2.validate(var_4)
    assert var_30 is True
    var_31 = 50
    var_32 = var_2.validate(var_8, var_31)
    assert var_32 is False



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the loads method of TimedSerializer covering:\n    - Successful loading (returning payload)\n    - Successful loading with return_timestamp=True\n    - Handling of SignatureExpired (should raise error and not try next signer)\n    - Handling of BadSignature (should try next signer)\n    - Final failure when all signers fail\n    '
    var_1 = b'secret_data'
    var_2 = b'ZW5jb2RlZF9kYXRh'
    var_3 = 1600000000
    var_4 = b'.'
    var_5 = module_0.TimedSerializer()
    var_6 = 'Invalid'
    var_7 = b'corrupt'
    var_8 = 'Expired'
    var_9 = b'payload.timestamp.signature'
    var_10 = var_5.loads(var_9)
    var_11 = None
    var_12 = True
    var_13 = var_5.loads(var_9, return_timestamp=var_12)
    var_14 = var_5.loads(var_9)
    var_15 = 'Bad 1'
    var_16 = 'Bad 2'
    var_17 = var_5.loads(var_9)
    var_18 = 3600
    var_19 = var_5.loads(var_9, var_18)
    var_20 = 'my_salt'
    var_21 = var_5.loads(var_9, salt=var_20)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the initialization and basic properties of TimedSerializer.\n    Since the constructor is inherited from Serializer, we verify \n    the default signer assignment and class structure.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = 'loads'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'dumps'
    var_5 = hasattr(var_1, var_4)
    var_6 = 'iter_unsigners'
    var_7 = hasattr(var_1, var_6)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Verify that TimedSerializer correctly identifies its default signer \n    as TimestampSigner.\n    '
    var_1 = module_0.TimedSerializer()



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the initialization and properties of TimedSerializer.\n    Since the constructor is inherited from Serializer, we verify \n    that it correctly sets up the default signer as TimestampSigner.\n    '
    var_1 = module_0.TimedSerializer()

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Verifies properties of the TimedSerializer class itself.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = 'iter_unsigners'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = var_1.timestamp_to_datetime(var_2)
    var_4 = 'string-key'
    var_5 = module_0.TimestampSigner(var_4)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = b'test_payload'
    var_4 = 1700000000
    var_5 = var_2.sign(var_3)
    var_6 = var_2.unsign(var_5)
    var_7 = True
    var_8 = 100
    var_9 = var_2.unsign(var_5, var_8)
    var_10 = 100
    var_11 = var_2.unsign(var_5, var_10)
    var_12 = str(var_10)
    var_13 = 100
    var_14 = var_2.unsign(var_5, var_13)
    var_15 = str(var_13)
    var_16 = b'tampered'
    var_17 = len(var_3)
    var_18 = var_5[var_17:]
    var_19 = var_16 + var_18
    var_20 = var_2.unsign(var_19)
    var_21 = b'.'
    var_22 = var_3 + var_21
    var_23 = b'not-base64-at-all!!!'
    var_24 = var_22 + var_23
    var_25 = var_2.unsign(var_24)
    var_26 = b'.something'
    var_27 = var_3 + var_26
    var_28 = var_2.unsign(var_3)
    var_29 = module_1.int_to_bytes(var_4)
    var_30 = module_1.base64_encode(var_29)
    var_31 = var_3 + var_21
    var_32 = var_31 + var_30
    var_33 = var_32 + var_21
    var_34 = b'wrong-signature'
    var_35 = var_33 + var_34
    var_36 = var_2.unsign(var_35)
    var_37 = var_2.validate(var_5)
    assert var_37 is True
    var_38 = 10
    var_39 = var_2.validate(var_5, var_38)
    assert var_39 is False
    var_40 = b'invalid-data'
    var_41 = var_2.validate(var_40)
    assert var_41 is False



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the construction and basic characteristics of TimedSerializer.\n    Since the constructor is inherited from Serializer, we verify \n    it initializes correctly with expected default attributes.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = 'dumps'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'loads'
    var_5 = hasattr(var_1, var_4)
    var_6 = 'loads_unsafe'
    var_7 = hasattr(var_1, var_6)
    var_8 = b'test_salt'
    var_9 = var_1.iter_unsigners(var_8)
    var_10 = list(var_9)

def test_case_0():
    var_0 = 'Verifies the class hierarchy for TimedSerializer.'



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = b'hello-world'
    var_4 = 1700000000
    var_5 = var_2.sign(var_3)
    var_6 = var_2.unsign(var_5)
    var_7 = True
    var_8 = 100
    var_9 = var_2.validate(var_5, var_8)
    assert var_9 is True
    var_10 = var_2.unsign(var_5, var_8)
    var_11 = 10
    var_12 = var_2.unsign(var_5, var_11)
    var_13 = 10
    var_14 = var_2.unsign(var_5, var_13)
    var_15 = b'tampered'
    var_16 = b'hello-world'
    var_17 = len(var_16)
    var_18 = var_5[var_17:]
    var_19 = var_15 + var_18
    var_20 = var_2.unsign(var_19)
    var_21 = var_3 + var_1
    var_22 = b'not-base64-!!!'
    var_23 = var_21 + var_22
    var_24 = b'no-separator'
    var_25 = var_3 + var_24
    var_26 = var_2.unsign(var_25)
    var_27 = var_2.sign(var_3)
    var_28 = var_3 + var_1
    var_29 = b'timestamp_encoded_here'
    var_30 = var_28 + var_29
    var_31 = var_30 + var_1
    var_32 = b'badsignature'
    var_33 = var_31 + var_32
    var_34 = b'payload'
    var_35 = var_34 + var_1
    var_36 = b'invalid_base64_content'
    var_37 = var_35 + var_36
    var_38 = var_2.unsign(var_37)
    var_39 = b'invalid-signature-entirely'
    var_40 = var_2.unsign(var_39)
    var_41 = b'invalid-signature-entirely'
    var_42 = var_2.validate(var_41)
    assert var_42 is False



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the loads method of TimedSerializer covering successful loads,\n    expired signatures, and invalid signatures.\n    '
    var_1 = 'secret'
    var_2 = module_0.TimestampSigner()
    var_3 = 'hello'
    var_4 = var_2.sign(var_3)
    var_5 = True
    var_6 = var_2.sign(var_3)
    var_7 = 20
    var_8 = var_2.sign(var_3)
    var_9 = 10
    var_10 = b'wrong_signature_format'
    var_11 = 'wrong'
    var_12 = module_0.TimestampSigner()
    var_13 = 'secret'
    var_14 = module_0.TimestampSigner()
    var_15 = var_14.sign(var_3)
    var_16 = var_14.sign(var_3)
    var_17 = 1



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the constructor and basic initialization behavior \n    of the TimedSerializer class.\n    '
    var_1 = module_0.TimedSerializer()



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'hello world'
    var_1 = True
    var_2 = 10
    var_3 = 1
    var_4 = 100
    var_5 = 50
    var_6 = 1
    var_7 = 10
    var_8 = 50
    var_9 = b'tampered'
    var_10 = b'not-base64!!!'
    var_11 = b'signature'
    var_12 = b'just-some-data'
    var_13 = b'123456789'
    var_14 = 123456789
    var_15 = module_0.int_to_bytes(var_14)
    var_16 = module_0.base64_encode(var_15)
    var_17 = var_7 + var_16
    var_18 = var_17 + var_9
    var_19 = b'sig'
    var_20 = var_18 + var_19
    var_21 = 'invalid sig'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = "\n    Tests the initialization and basic structure of TimedSerializer.\n    Since the provided code doesn't define a custom __init__, \n    we verify it inherits correctly from Serializer and maintains \n    the expected class attributes.\n    "
    var_1 = 'test_secret'
    var_2 = 'TimorestSerializer'
    var_3 = globals()
    var_4 = var_2 in var_3
    var_5 = module_0.TimedSerializer()

def test_case_0():
    var_0 = 'Verify the class hierarchy.'



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the initialization and basic properties of TimestampSigner.\n    Since TimestampSigner inherits from Signer, we verify it correctly \n    inherits/uses its configuration and maintains expected attributes.\n    '
    var_1 = b'secret-key'
    var_2 = b'.'
    var_3 = module_0.TimestampSigner(var_1, sep=var_2)
    var_4 = var_3.get_timestamp()
    var_5 = var_3.timestamp_to_datetime(var_4)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the loads method of TimedSerializer covering:\n    1. Successful decryption/unsigning with timestamp return.\n    2. Successful decryption without timestamp return.\n    3. Raising SignatureExpired when max_age is exceeded.\n    4. Raising BadSignature when signature is invalid.\n    5. Handling multiple unsigners (iterating through them).\n    '
    var_1 = b'decoded_payload'



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Tests the loads method of TimedSerializer with various scenarios.'
    var_1 = 'secret'
    var_2 = module_0.TimestampSigner()
    var_3 = [var_2]
    var_4 = True
    var_5 = 10
    var_6 = 50
    var_7 = b'h'
    var_8 = b'z'
    var_9 = b'payload'
    var_10 = 'some_blob'
    var_11 = 'my_salt'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the initialization and basic properties of TimedSerializer.\n    Since the provided code shows TimedSerializer inherits from Serializer,\n    we verify it correctly identifies its default signer class.\n    '
    var_1 = '__main__.Serializer'
    var_2 = module_0.TimedSerializer()
    var_3 = 'loads'
    var_4 = hasattr(var_2, var_3)
    var_5 = 'loads_unsafe'
    var_6 = hasattr(var_2, var_5)
    var_7 = 'iter_unsigners'
    var_8 = hasattr(var_2, var_7)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the instantiation and basic properties of TimedSerializer.\n    Since TimedSerializer inherits from Serializer, we verify its \n    specific type-related attributes and its default signer.\n    '
    var_1 = module_0.TimedSerializer()

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests that iter_unsigners returns an iterator of TimestampSigner instances.\n    '
    var_1 = module_0.TimedSerializer()

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Ensures the TimedSerializer maintains its identity as a Serializer subclass.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = 'loads'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'loads_unsafe'
    var_5 = hasattr(var_1, var_4)



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the initialization and basic property configuration \n    of the TimedSerializer class.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = b'test_salt'
    var_3 = var_1.iter_unsigners(var_2)
    var_4 = list(var_3)



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = 'get_timestamp'
    var_4 = hasattr(var_2, var_3)
    var_5 = 'timestamp_to_datetime'
    var_6 = hasattr(var_2, var_5)
    var_7 = 'sign'
    var_8 = hasattr(var_2, var_7)
    var_9 = 'unsign'
    var_10 = hasattr(var_2, var_9)
    var_11 = 'validate'
    var_12 = hasattr(var_2, var_11)
    var_13 = 1600000000
    var_14 = var_2.timestamp_to_datetime(var_13)
    var_15 = None
    var_16 = 0
    var_17 = b'hello'
    var_18 = var_2.sign(var_17)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'secret'
    var_1 = 'utf-8'
    var_2 = True
    var_3 = 50
    var_4 = 'wrong'
    var_5 = 'secret1'
    var_6 = 'secret2'
    var_7 = 'Fail'
    var_8 = b'corrupted'
    var_9 = b'payload_data'
    var_10 = b'some_blob'
    var_11 = 'string_payload'



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = var_1.timestamp_to_datetime(var_2)



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = "\n    Tests the initialization and basic properties of TimedSerializer.\n    Since the constructor is inherited from Serializer, we verify \n    the class-specific attribute 'default_signer'.\n    "



