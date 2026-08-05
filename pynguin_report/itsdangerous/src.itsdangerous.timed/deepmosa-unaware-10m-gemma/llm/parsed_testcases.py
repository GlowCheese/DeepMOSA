####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the constructor and basic initialization of TimedSerializer.\n    Since TimedSerializer inherits from Serializer, we verify it \n    correctly sets up its default signer and maintains expected attributes.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = module_0.TimedSerializer()



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.exc as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = b'secret'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = 1700000000
    var_4 = b'hello'
    var_5 = var_2.sign(var_4)
    var_6 = var_2.unsign(var_5)
    var_7 = True
    var_8 = 100
    var_9 = var_2.unsign(var_5, var_8)
    var_10 = 10
    var_11 = var_2.unsign(var_5, var_10)
    var_12 = var_4 + var_1
    var_13 = b'notbase64!!!'
    var_14 = var_12 + var_13
    var_15 = var_14 + var_1
    var_16 = b'sig'
    var_17 = var_15 + var_16
    var_18 = var_2.unsign(var_17)
    var_19 = str(var_18)
    var_20 = b'no_separators_here'
    var_21 = var_2.unsign(var_20)
    var_22 = str(var_21)
    var_23 = 'signature failed'
    var_24 = module_1.BadSignature(var_23, var_4)
    var_25 = var_4 + var_1
    var_26 = module_2.int_to_bytes(var_3)
    var_27 = module_2.base64_encode(var_26)
    var_28 = var_25 + var_27
    var_29 = var_28 + var_1
    var_30 = b'badsig'
    var_31 = var_29 + var_30
    var_32 = var_2.unsign(var_31)
    var_33 = 1000
    var_34 = var_3 + var_33
    var_35 = 100
    var_36 = str(var_35)
    var_37 = ' '
    var_38 = ''



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = b'secret'
    var_1 = b'hello'
    var_2 = b'.'
    var_3 = var_1 + var_2
    var_4 = b'validsig'
    var_5 = True
    var_6 = 100
    var_7 = var_1 + var_2
    var_8 = 50
    var_9 = var_1 + var_2
    var_10 = 50
    var_11 = False
    var_12 = var_1 + var_2
    var_13 = b'wrongsig'
    var_14 = var_1 + var_2
    var_15 = b'notbase64!!!'
    var_16 = var_14 + var_15
    var_17 = var_16 + var_2
    var_18 = var_17 + var_4
    var_19 = b'just-a-string-without-separators'
    var_20 = var_1 + var_2
    var_21 = 50



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the initialization and basic properties of TimedSerializer.\n    Since the constructor is inherited from Serializer, we verify \n    the class-specific defaults and type attributes.\n    '
    var_1 = module_0.TimedSerializer()



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = b'hello'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = True
    var_7 = 1000
    var_8 = 500
    var_9 = var_2.unsign(var_4, var_8)
    var_10 = str(var_8)
    var_11 = 500
    var_12 = var_2.unsign(var_4, var_11)
    var_13 = str(var_11)
    var_14 = b'tampered'
    var_15 = len(var_14)
    var_16 = var_4[var_15:]
    var_17 = var_14 + var_16
    var_18 = var_2.unsign(var_17)
    var_19 = b'data'
    var_20 = var_19 + var_1
    var_21 = b'notbase64!!!'
    var_22 = var_20 + var_21
    var_23 = var_22 + var_1
    var_24 = b'signature'
    var_25 = var_23 + var_24
    var_26 = var_2.unsign(var_25)
    var_27 = b'only_one_part_no_timestamp'
    var_28 = var_2.unsign(var_27)
    var_29 = b'original'
    var_30 = var_2.sign(var_29)
    var_31 = b'corrupted'
    var_32 = var_31 + var_1
    var_33 = 1
    var_34 = var_32 + var_20
    var_35 = var_34 + var_1
    var_36 = 2
    var_37 = var_35 + var_24
    var_38 = var_2.unsign(var_37)
    var_39 = 'BadSignature'
    var_40 = 'signature'
    var_41 = b'valid'
    var_42 = var_2.sign(var_41)
    var_43 = var_2.validate(var_42)
    assert var_43 is True
    var_44 = var_2.validate(var_17)
    assert var_44 is False
    var_45 = 500
    var_46 = var_2.validate(var_42, var_45)
    assert var_46 is False
    var_47 = 1500
    var_48 = var_2.validate(var_42, var_47)
    assert var_48 is True



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Tests the loads method of TimedSerializer for various scenarios.'
    var_1 = module_0.TimedSerializer()
    var_2 = 'payload_data'
    var_3 = b'base64_payload'
    var_4 = 2023
    var_5 = 1
    var_6 = b'signed_data'
    var_7 = var_1.loads(var_6)
    assert var_7 == 'payload_data'
    var_8 = None
    var_9 = True
    var_10 = True
    var_11 = 100
    var_12 = var_1.loads(var_6, var_11)
    var_13 = True
    var_14 = 'Expired'
    var_15 = b'data'
    var_16 = b'signed_data'
    var_17 = var_1.loads(var_16)
    var_18 = b'base64_payload_2'
    var_19 = 'Bad signature'
    var_20 = b'bad'
    var_21 = var_1.loads(var_6)
    assert var_21 == 'payload_data_2'
    var_22 = 'Final failure'
    var_23 = b'final'
    var_24 = b'signed_data'
    var_25 = var_1.loads(var_24)
    var_26 = 'my_salt'
    var_27 = var_1.loads(var_6, salt=var_26)
    var_28 = True



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the construction and basic state of TimedSerializer.\n    Since the constructor is inherited from Serializer, we verify \n    it initializes with expected default attributes and class properties.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = 'default_signer'
    var_3 = hasattr(var_1, var_2)
    var_4 = module_0.TimedSerializer()

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Verifies that TimedSerializer properly inherits and overrides \n    the expected signer class.\n    '
    var_1 = module_0.TimedSerializer()



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

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
    var_9 = var_2.unsign(var_5, var_8)
    var_10 = 10
    var_11 = var_2.unsign(var_5, var_10)
    var_12 = var_4 + var_8
    var_13 = module_1.int_to_bytes(var_12)
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_3 + var_1
    var_16 = var_15 + var_14
    var_17 = var_16 + var_1
    var_18 = b'fake-sig'
    var_19 = var_17 + var_18
    var_20 = 10
    var_21 = var_2.unsign(var_19, var_20)
    var_22 = str(var_20)
    var_23 = b'tampered'
    var_24 = var_23 + var_1
    var_25 = -1
    var_26 = signed_val.split(var_1)[var_25:]
    var_27 = var_24 + var_26
    var_28 = var_3 + var_1
    var_29 = module_1.int_to_bytes(var_4)
    var_30 = module_1.base64_encode(var_29)
    var_31 = var_28 + var_30
    var_32 = var_31 + var_1
    var_33 = b'wrong-sig'
    var_34 = var_32 + var_33
    var_35 = var_2.unsign(var_34)
    var_36 = var_3 + var_1
    var_37 = b'not-base64!!!'
    var_38 = var_36 + var_37
    var_39 = var_38 + var_1
    var_40 = b'sig'
    var_41 = var_39 + var_40
    var_42 = var_2.unsign(var_41)
    var_43 = var_2.unsign(var_3)
    var_44 = var_2.validate(var_5)
    assert var_44 is True
    var_45 = 1
    var_46 = var_2.validate(var_5, var_45)
    assert var_46 is False



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'.'
    var_3 = b'hello-world'
    var_4 = var_1.sign(var_3)
    var_5 = var_1.unsign(var_4)
    var_6 = True
    var_7 = 100
    var_8 = var_1.validate(var_4, var_7)
    assert var_8 is True
    var_9 = 10
    var_10 = var_1.unsign(var_4, var_9)
    var_11 = str(var_9)
    var_12 = 10
    var_13 = var_1.unsign(var_4, var_12)
    var_14 = str(var_12)
    var_15 = b'tampered'
    var_16 = var_3 + var_2
    var_17 = b'not-base64-valid-data!!'
    var_18 = var_16 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = var_1.unsign(var_3)
    var_21 = var_3 + var_2
    var_22 = b'wrong-signature'



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'super-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = var_1.timestamp_to_datetime(var_2)
    var_4 = b'test-payload'
    var_5 = var_1.sign(var_4)
    var_6 = var_1.unsign(var_5)
    var_7 = True
    var_8 = var_1.validate(var_5)
    assert var_8 is True
    var_9 = b'invalid-data'
    var_10 = var_1.validate(var_9)
    assert var_10 is False
    var_11 = 'time.time'
    var_12 = 2000000000
    var_13 = lambda : var_12
    var_14 = var_1.sign(var_4)
    var_15 = 'get_timestamp'
    var_16 = 2000000100
    var_17 = lambda : var_16
    var_18 = 10
    var_19 = var_1.unsign(var_14, var_18)
    var_20 = b'not-a-timestamped-value'
    var_21 = var_1.unsign(var_20)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = "\n    Tests the initialization and basic functionality of TimestampSigner.\n    Since it inherits from Signer, we verify that it can be instantiated \n    and maintains expected attributes like 'secret' and 'sep'.\n    "
    var_1 = b'super-secret-key'
    var_2 = b'.'
    var_3 = module_0.TimestampSigner(sep=var_2)
    var_4 = 'get_timestamp'
    var_5 = hasattr(var_3, var_4)
    var_6 = 'timestamp_to_datetime'
    var_7 = hasattr(var_3, var_6)
    var_8 = 'sign'
    var_9 = hasattr(var_3, var_8)
    var_10 = 'unsign'
    var_11 = hasattr(var_3, var_10)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Tests constructor with a non-standard separator.'
    var_1 = b'secret'
    var_2 = b'|'
    var_3 = module_0.TimestampSigner(sep=var_2)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Tests that TimestampSigner behaves like a Signer in terms of interface.'
    var_1 = b'secret'
    var_2 = module_0.TimestampSigner()
    var_3 = b'hello'
    var_4 = var_2.sign(var_3)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello-world'
    var_3 = b'.'
    var_4 = var_1.sign(var_2)
    var_5 = True
    var_6 = var_1.unsign(var_4)
    var_7 = var_1.sign(var_2)
    var_8 = 'get_timestamp'
    var_9 = 200
    var_10 = 50
    var_11 = var_1.unsign(var_7, var_10)
    var_12 = str(var_10)
    var_13 = var_1.sign(var_2)
    var_14 = 100
    var_15 = var_1.unsign(var_13, var_14)
    var_16 = str(var_14)
    var_17 = b'-tampered'
    var_18 = var_2 + var_17
    var_19 = 0
    var_20 = b'wrongsignature'
    var_21 = var_2 + var_3
    var_22 = b'not-base64-garbage!!!'
    var_23 = var_21 + var_22
    var_24 = var_23 + var_3
    var_25 = b'signature'
    var_26 = var_24 + var_25
    var_27 = var_1.unsign(var_26)
    var_28 = var_1.unsign(var_2)
    var_29 = var_1.validate(var_4)
    assert var_29 is True
    var_30 = 10
    var_31 = var_1.validate(var_7, var_30)
    assert var_31 is False
    var_32 = 10
    var_33 = var_1.validate(var_7, var_32)
    assert var_33 is False



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'.'
    var_3 = b'hello-world'
    var_4 = var_1.sign(var_3)
    var_5 = var_1.unsign(var_4)
    var_6 = True
    var_7 = 1000000
    var_8 = var_1.sign(var_3)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = b'secret_key'
    var_1 = b'hello_world'
    var_2 = True
    var_3 = 10
    var_4 = 10
    var_5 = str(var_4)
    var_6 = 100
    var_7 = b'.'
    var_8 = var_1 + var_7
    var_9 = 10
    var_10 = False
    var_11 = b'not_an_int'
    var_12 = module_0.base64_encode(var_11)
    var_13 = var_1 + var_7
    var_14 = var_13 + var_12
    var_15 = var_14 + var_7
    var_16 = b'sig'
    var_17 = var_15 + var_16
    var_18 = b'.'
    var_19 = var_1 + var_18
    var_20 = b'garbage'
    var_21 = module_0.base64_encode(var_20)
    var_22 = var_19 + var_21
    var_23 = var_22 + var_18
    var_24 = b'fake_sig'
    var_25 = var_23 + var_24
    var_26 = b'.not_enough_parts'
    var_27 = var_1 + var_26
    var_28 = b'invalid_base64_!@#'
    var_29 = var_1 + var_7
    var_30 = var_29 + var_28
    var_31 = var_30 + var_0
    var_32 = var_30 + var_7



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = b'test_secret'
    var_1 = b':'
    var_2 = 'sign'
    var_3 = 'unsign'
    var_4 = 'get_timestamp'



# Parsed testcases at query #16
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
    var_7 = 10
    var_8 = var_2.unsign(var_4, var_7)
    var_9 = 1000000
    var_10 = b'old-data'
    var_11 = var_2.sign(var_10)
    var_12 = 20
    var_13 = 5
    var_14 = var_2.unsign(var_11, var_13)
    var_15 = b'future-data'
    var_16 = var_2.sign(var_15)
    var_17 = 5
    var_18 = var_2.unsign(var_16, var_17)
    var_19 = 'less than 0'
    var_20 = var_19 in var_8
    var_21 = 'age -1'
    var_22 = -5
    var_23 = var_4[:var_22]
    var_24 = b'xxxxx'
    var_25 = var_23 + var_24
    var_26 = var_2.unsign(var_25)
    var_27 = var_3 + var_1
    var_28 = b'not-base64-valid-enough!!!'
    var_29 = var_27 + var_28
    var_30 = var_2.unsign(var_29)
    var_31 = b'no_separator_here'
    var_32 = var_2.unsign(var_31)
    var_33 = var_2.validate(var_4)
    assert var_33 is True
    var_34 = var_2.validate(var_25)
    assert var_34 is False
    var_35 = b'test'
    var_36 = var_2.sign(var_35)
    var_37 = 1
    var_38 = var_2.validate(var_36, var_37)
    assert var_38 is False
    var_39 = b'any-value'
    var_40 = var_2.unsign(var_39)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'.'
    var_3 = b'hello-world'
    var_4 = var_1.sign(var_3)
    var_5 = var_1.unsign(var_4)
    var_6 = True
    var_7 = 100
    var_8 = var_1.unsign(var_4, var_7)
    var_9 = -1
    var_10 = var_1.unsign(var_4, var_9)
    var_11 = b'-tampered'
    var_12 = var_3 + var_11
    var_13 = var_12 + var_2
    var_14 = b'invalid-timestamp-part'
    var_15 = var_13 + var_14
    var_16 = var_15 + var_2
    var_17 = b'wrong-signature'
    var_18 = var_16 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = var_3 + var_2
    var_21 = b'not-base64-at-all!!!'
    var_22 = var_20 + var_21
    var_23 = b'.broken'
    var_24 = var_3 + var_23
    var_25 = var_1.unsign(var_24)
    var_26 = int(var_23)
    var_27 = module_1.int_to_bytes(var_26)
    var_28 = module_1.base64_encode(var_27)
    var_29 = 'Invalid signature'
    var_30 = var_3 + var_2
    var_31 = var_30 + var_28
    var_32 = b'some-signed-data'
    var_33 = var_1.unsign(var_32)
    var_34 = var_1.sign(var_3)
    var_35 = 10
    var_36 = var_1.unsign(var_34, var_35)
    var_37 = str(var_35)
    var_38 = ' '
    var_39 = ''



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0, sep=var_1)
    var_3 = b'other'
    var_4 = b':'
    var_5 = module_0.TimestampSigner(var_3, sep=var_4)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the loads method of TimedSerializer covering:\n    1. Successful loading of payload and timestamp.\n    2. Successful loading of only payload.\n    3. Handling of expired signatures (SignatureExpired).\n    4. Handling of invalid signatures (BadSignature).\n    5. Iteration through multiple signers with salt.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = b'encoded_payload'
    var_6 = 2023
    var_7 = 1
    var_8 = 12
    var_9 = 0
    var_10 = b'some_signed_data'
    var_11 = True
    var_12 = var_1.loads(var_10, return_timestamp=var_11)
    var_13 = b'some_signed_data'
    var_14 = False
    var_15 = var_1.loads(var_13, return_timestamp=var_14)
    var_16 = 'Expired'
    var_17 = b'data'
    var_18 = b'some_signed_data'
    var_19 = var_1.loads(var_18)
    var_20 = 'Bad'
    var_21 = b'fail'
    var_22 = b'some_signed_data'
    var_23 = var_1.loads(var_22)
    var_24 = 'Bad 1'
    var_25 = 'Bad 2'
    var_26 = b'some_signed_data'
    var_27 = var_1.loads(var_26)
    var_28 = str(var_26)
    var_29 = b'data'
    var_30 = 'my_salt'
    var_31 = var_1.loads(var_29, salt=var_30)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'.'
    var_3 = b'my-payload'
    var_4 = var_1.sign(var_3)
    var_5 = var_1.unsign(var_4)
    var_6 = True
    var_7 = 1000.0
    var_8 = var_1.sign(var_3)
    var_9 = 100
    var_10 = 50
    var_11 = var_1.unsign(var_8, var_10)
    var_12 = b'Signature age 100 > 50 seconds'
    var_13 = '100'
    var_14 = 2000.0
    var_15 = var_1.sign(var_3)
    var_16 = 50
    var_17 = 100
    var_18 = var_1.unsign(var_15, var_17)
    var_19 = str(var_18)
    var_20 = b'my'
    var_21 = b'no'
    var_22 = var_3 + var_2
    var_23 = b'not-b64!!!'
    var_24 = var_22 + var_23
    var_25 = var_3 + var_2
    var_26 = var_25 + var_23
    var_27 = var_26 + var_2
    var_28 = b'.some-signature'
    var_29 = var_3 + var_28
    var_30 = var_1.unsign(var_29)
    var_31 = var_1.validate(var_4)
    assert var_31 is True
    var_32 = 1
    var_33 = var_1.validate(var_4, var_32)
    assert var_33 is False
    var_34 = 5000.0
    var_35 = var_1.sign(var_3)
    var_36 = 'wrong-key'
    var_37 = module_0.TimestampSigner(var_36)
    var_38 = -10
    var_39 = var_35[:var_38]
    var_40 = b'badsig'
    var_41 = var_39 + var_40
    var_42 = var_37.unsign(var_41)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = var_1.timestamp_to_datetime(var_2)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'string-key'
    var_4 = '.'
    var_5 = module_0.TimestampSigner(sep=var_4)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.exc as module_1

def test_case_0():
    var_0 = '\n    Unit tests for the loads method of TimedSerializer.\n    Tests various scenarios: success (with and without timestamp), \n    expired signature, bad signature, and multiple signers.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = b'encoded_payload'
    var_6 = b'some_signed_string'
    var_7 = var_1.loads(var_6)
    var_8 = None
    var_9 = True
    var_10 = 2023
    var_11 = 12
    var_12 = 0
    var_13 = 'expired'
    var_14 = b'old'
    var_15 = b'expired_string'
    var_16 = 10
    var_17 = var_1.loads(var_15, var_16)
    var_18 = 'bad signature'
    var_19 = b'payload_data'
    var_20 = module_1.BadSignature(var_18, var_19)
    var_21 = b'multi_signer_string'
    var_22 = var_1.loads(var_21)
    var_23 = 'error1'
    var_24 = 'error2'
    var_25 = b'last_payload'
    var_26 = b'all_fail_string'
    var_27 = var_1.loads(var_26)
    var_28 = b'test'
    var_29 = 100
    var_30 = var_1.loads(var_28, var_29)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 2023
    var_2 = 1
    var_3 = 'expired'
    var_4 = b'data'
    var_5 = 'bad sig'
    var_6 = b'tampered'
    var_7 = b'signed_blob'
    var_8 = True
    var_9 = b'signed_blob'
    var_10 = b'signed_blob'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'fail'
    var_2 = b'bad'
    var_3 = b'good'
    var_4 = b'some_blob'
    var_5 = var_0.loads(var_4)
    assert var_5 == 'success'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'expired'
    var_2 = b'data'
    var_3 = b'some_blob'
    var_4 = var_0.loads(var_3)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 1000000
    var_1 = 'Expired'
    var_2 = b'data'
    var_3 = 'Bad Sig'
    var_4 = b'data'
    var_5 = b'data'
    var_6 = b'signed_blob'
    var_7 = b'signed_blob'
    var_8 = 'data'
    var_9 = 1

def test_case_0():
    var_0 = 'Test that loads iterates through multiple signers until one works.'
    var_1 = 'First failed'
    var_2 = b'partially_valid'
    var_3 = b'success'
    var_4 = b'some_blob'



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the initialization and basic properties of TimedSerializer.\n    Since the constructor for TimedSerializer (inherited from Serializer) \n    is standard, we verify it correctly identifies its default signer \n    and maintains the expected class structure.\n    '
    var_1 = module_0.TimedSerializer()
    var_2 = b'test_salt'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = module_0.TimedSerializer()



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'super-secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.exc as module_0

def test_case_0():
    var_0 = '\n    Test the loads method of TimedSerializer covering:\n    1. Successful loading of payload.\n    2. Successful loading with timestamp return.\n    3. Failure due to expired signature (SignatureExpired).\n    4. Failure due to invalid signature (BadSignature).\n    5. Handling multiple signers where one is valid and another is not.\n    '
    var_1 = 'test_salt'
    var_2 = b'hello_world'
    var_3 = 'hello_world'
    var_4 = 2023
    var_5 = 1
    var_6 = 12
    var_7 = 0
    var_8 = b'encoded_payload.timestamp.signature'
    var_9 = b'encoded_payload.timestamp.signature'
    var_10 = None
    var_11 = True
    var_12 = True
    var_13 = 'Expired'
    var_14 = b'old'
    var_15 = 10
    var_16 = 'Invalid signature'
    var_17 = b'corrupted'
    var_18 = module_0.BadSignature(var_16, var_17)
    var_19 = b'second_payload'
    var_20 = 'Bad'
    var_21 = b'payload1'
    var_22 = b'second_payload'
    var_23 = b'some_data'
    var_24 = 100
    var_25 = True



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello-world'
    var_3 = b'.'
    var_4 = var_1.sign(var_2)
    var_5 = var_1.unsign(var_4)
    var_6 = var_1.sign(var_2)
    var_7 = True
    var_8 = 1000
    var_9 = 500
    var_10 = var_1.unsign(var_6, var_9)
    var_11 = str(var_9)
    var_12 = 500
    var_13 = var_1.unsign(var_6, var_12)
    var_14 = str(var_12)
    var_15 = b'tampered-data'
    var_16 = var_1.sign(var_15)
    var_17 = var_2 + var_3
    var_18 = b'invalid_b64'
    var_19 = var_17 + var_18
    var_20 = var_2 + var_3
    var_21 = var_20 + var_18
    var_22 = var_21 + var_3
    var_23 = str(var_22)
    var_24 = var_2 + var_3
    var_25 = b'signature_only'
    var_26 = var_24 + var_25
    var_27 = var_1.unsign(var_26)
    var_28 = str(var_24)
    var_29 = var_1.sign(var_2)
    var_30 = var_1.validate(var_29)
    assert var_30 is True
    var_31 = 10
    var_32 = var_1.validate(var_29, var_31)
    assert var_32 is False
    var_33 = 100
    var_34 = var_1.unsign(var_6, var_33)



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 1000000000
    var_2 = b'encoded_payload'
    var_3 = b'dummy_data'
    var_4 = 3600
    var_5 = True
    var_6 = var_3
    var_7 = 3600
    var_8 = False

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 10
    var_2 = b'some_payload'
    var_3 = b'signed_data'
    var_4 = 3600
    var_5 = var_0.loads(var_3, var_4)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'Invalid signature'
    var_2 = b'corrupted'
    var_3 = b'bad_data'
    var_4 = var_0.loads(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = b'payload'
    var_2 = b'data'
    var_3 = b'new_salt'
    var_4 = var_0.loads(var_2, salt=var_3)
    assert var_4 == 'success'



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = 1700000000
    var_4 = b'hello-world'
    var_5 = var_2.sign(var_4)
    var_6 = var_2.unsign(var_5)
    var_7 = True
    var_8 = 10
    var_9 = var_2.unsign(var_5, var_8)
    var_10 = -1
    var_11 = 100
    var_12 = var_3 - var_11
    var_13 = var_2.sign(var_4)
    var_14 = 50
    var_15 = var_2.unsign(var_13, var_14)
    var_16 = str(var_14)
    var_17 = b'hello'
    var_18 = b'bad'
    var_19 = b'just-payload-no-timestamp'
    var_20 = var_2.unsign(var_19)
    var_21 = str(var_20)
    var_22 = 0
    var_23 = var_8 + var_1
    var_24 = b'not-base64-!!!'
    var_25 = var_23 + var_24
    var_26 = var_25 + var_1
    var_27 = 2
    var_28 = module_1.int_to_bytes(var_3)
    var_29 = module_1.base64_encode(var_28)
    var_30 = var_4 + var_1
    var_31 = var_30 + var_29
    var_32 = var_31 + var_1
    var_33 = b'wrong-signature'
    var_34 = var_32 + var_33
    var_35 = var_2.unsign(var_34)
    var_36 = 'BadSignature'
    var_37 = var_2.validate(var_5)
    assert var_37 is True
    var_38 = 1
    var_39 = var_2.validate(var_5, var_38)
    assert var_39 is False

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 1600000000
    var_3 = var_1.timestamp_to_datetime(var_2)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '\n    Test the loads method of TimedSerializer covering:\n    1. Successful decoding with timestamp return.\n    2. Successful decoding without timestamp return.\n    3. Handling of SignatureExpired (should raise and not try next signer).\n    4. Handling of BadSignature (should try next signer).\n    5. Handling of multiple signers.\n    '
    var_1 = b'hello_world'
    var_2 = 'hello_world'
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = b'hello_payload'
    var_8 = b'some_signed_data'
    var_9 = True
    var_10 = 'expired'
    var_11 = b'expired_data'
    var_12 = b'some_signed_data'
    var_13 = 'bad'
    var_14 = b'bad_payload'
    var_15 = b'good_payload'
    var_16 = 'all_bad'
    var_17 = b'none'
    var_18 = b'some_signed_data'
    var_19 = b'payload'
    var_20 = b'data'
    var_21 = 60
    var_22 = True



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'serializer.Serializer.iter_unsigners'
    var_1 = module_0.TimedSerializer()
    var_2 = var_1.iter_unsigners()



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = '\n    Tests the loads method of TimedSerializer covering successful loading,\n    timestamp retrieval, expiration, and signature failure.\n    '
    var_1 = 'secret'
    var_2 = module_0.TimestampSigner()
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = b'{"key": "value"}'
    var_7 = var_2.sign(var_6)
    var_8 = True
    var_9 = var_2.sign(var_6)
    var_10 = 10
    var_11 = str(var_10)
    var_12 = -5
    var_13 = var_7[:var_12]
    var_14 = b'abcde'
    var_15 = var_13 + var_14
    var_16 = b'.'
    var_17 = var_2.sign(var_6)
    var_18 = 2
    var_19 = var_10.rsplit(var_16, var_18)[var_8]
    var_20 = var_6 + var_16
    var_21 = b'not_base64_junk!'
    var_22 = var_20 + var_21
    var_23 = var_22 + var_16
    var_24 = b'signature'
    var_25 = var_23 + var_24
    var_26 = 'Bad Sig'
    var_27 = 'wrong'
    var_28 = module_0.TimestampSigner()
    var_29 = module_0.TimestampSigner()
    var_30 = 'Expired'



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'.'
    var_2 = module_0.TimestampSigner(sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'string-key'
    var_1 = '.'
    var_2 = module_0.TimestampSigner(sep=var_1)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = b'key'
    var_1 = module_0.TimestampSigner()
    var_2 = 'get_timestamp'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'timestamp_to_datetime'
    var_5 = hasattr(var_1, var_4)
    var_6 = 'sign'
    var_7 = hasattr(var_1, var_6)
    var_8 = 'unsign'
    var_9 = hasattr(var_1, var_8)



