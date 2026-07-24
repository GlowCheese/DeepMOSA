####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 'get_timestamp'
    var_8 = 100
    var_9 = 10
    var_10 = b'invalid-data'
    var_11 = var_1.loads(var_10)
    var_12 = 'different-salt'
    var_13 = 'another-secret-key'
    var_14 = module_0.TimedSerializer(var_13)
    var_15 = [var_14]
    var_16 = module_0.TimedSerializer(var_10, fallback_signers=var_15)
    var_17 = b'invalid-data'
    var_18 = var_16.loads(var_17)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test constructor of TimestampSigner.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimestampSigner(var_1)
    var_3 = 'custom-key'
    var_4 = 'custom-salt'
    var_5 = '-'
    var_6 = 'none'
    var_7 = 'hs256'
    var_8 = b'bytes-key'
    var_9 = module_0.TimestampSigner(var_8)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'my-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'my-salt'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'key'
    var_6 = 'salt'
    var_7 = module_0.TimedSerializer(var_5, var_6)
    var_8 = 'test'
    var_9 = 'data'
    var_10 = {var_8: var_9}
    var_11 = b'test bytes'



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = ':'
    var_5 = 'none'
    var_6 = b'test-value'
    var_7 = True
    var_8 = b'invalid'
    var_9 = 3600
    var_10 = -1
    var_11 = b'invalid.signature'
    var_12 = 'string-value'



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test-value'
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    assert var_9 == b'test-value'
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = var_10.get_timestamp
    var_12 = 100
    var_13 = var_10.sign(var_2)
    var_14 = 50
    var_15 = var_10.unsign(var_13, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = var_16.sign(var_15)
    var_18 = 50
    var_19 = 3600
    var_20 = var_16.unsign(var_17, var_19)
    var_21 = b'invalid-signature'
    var_22 = var_1.unsign(var_21)
    var_23 = var_1.sign(var_22)
    var_24 = b'test-value'
    var_25 = b'tampered-value'
    var_26 = b'no-timestamp'
    var_27 = var_1.unsign(var_26)
    var_28 = module_0.TimestampSigner(var_26)
    var_29 = module_1.want_bytes(var_27)
    var_30 = var_28.sep
    var_31 = module_1.want_bytes(var_30)
    var_32 = b'not-a-number'
    var_33 = module_1.base64_encode(var_32)
    var_34 = var_29 + var_31
    var_35 = var_34 + var_33
    var_36 = var_35 + var_31
    var_37 = var_29 + var_31
    var_38 = var_37 + var_33
    var_39 = module_0.TimestampSigner(var_26)
    var_40 = module_1.want_bytes(var_27)
    var_41 = var_39.sep
    var_42 = module_1.want_bytes(var_41)
    var_43 = var_39.get_timestamp()
    var_44 = module_1.int_to_bytes(var_43)
    var_45 = module_1.base64_encode(var_44)
    var_46 = b'invalid-signature'
    var_47 = var_40 + var_42
    var_48 = var_47 + var_45
    var_49 = var_48 + var_42
    var_50 = var_49 + var_46
    var_51 = var_39.unsign(var_50)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'custom-key'
    var_4 = 'custom-salt'
    var_5 = 'json'
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}
    var_9 = module_0.TimedSerializer(var_3, var_4, var_5, signer_kwargs=var_8)
    var_10 = var_0.default_signer
    var_11 = var_0.iter_unsigners()
    var_12 = list(var_11)
    var_13 = len(var_12)
    var_14 = module_0.TimedSerializer()
    var_15 = 'message'
    var_16 = 'value'
    var_17 = 'hello'
    var_18 = 42
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 3600
    var_21 = True



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'user'
    var_4 = 'role'
    var_5 = 'test'
    var_6 = 'admin'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = True
    var_9 = 3600
    var_10 = module_0.TimedSerializer(var_1)
    var_11 = 7200
    var_12 = 1
    var_13 = 'invalid-data'
    var_14 = var_2.loads(var_13)
    var_15 = ''
    var_16 = var_2.loads(var_15)
    var_17 = 'custom-salt'
    var_18 = 'wrong-salt'
    var_19 = 'test-secret-key-2'
    var_20 = module_0.TimedSerializer(var_19)
    var_21 = 'list'
    var_22 = 'nested'
    var_23 = 'numbers'
    var_24 = 'boolean'
    var_25 = 'none_value'
    var_26 = 2
    var_27 = 3
    var_28 = [var_8, var_26, var_27]
    var_29 = 'key'
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = 1.5
    var_33 = 2.5
    var_34 = [var_32, var_33]
    var_35 = None
    var_36 = {var_21: var_28, var_22: var_31, var_23: var_34, var_24: var_8, var_25: var_35}



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = module_0.TimestampSigner(var_0)
    var_4 = var_2.get_timestamp()
    var_5 = var_2.timestamp_to_datetime(var_4)
    var_6 = 'test-value'
    var_7 = var_2.sign(var_6)
    var_8 = var_2.unsign(var_7)
    assert var_8 == b'test-value'
    var_9 = True
    var_10 = var_2.validate(var_7)
    assert var_10 is True
    var_11 = b'invalid'
    var_12 = var_2.validate(var_11)
    assert var_12 is False
    var_13 = -1
    var_14 = var_2.unsign(var_7, var_13)
    var_15 = 100
    var_16 = module_0.TimestampSigner(var_13, var_14)
    var_17 = var_16.sign(var_6)
    var_18 = 50
    var_19 = var_2.unsign(var_17, var_18)



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test_value'
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    assert var_7 == b'test_value'
    assert var_7 == b'test_string'
    var_8 = var_1.get_timestamp
    var_9 = 7200
    var_10 = 3600
    var_11 = var_1.unsign(var_3, var_10)
    var_12 = var_1.get_timestamp
    var_13 = 'future_test'
    var_14 = var_1.sign(var_13)
    var_15 = 3600
    var_16 = var_1.unsign(var_14, var_15)
    var_17 = b'invalid_signature'
    var_18 = var_1.unsign(var_17)
    var_19 = 'test'
    var_20 = var_1.sign(var_19)
    var_21 = module_1.want_bytes(var_18)
    var_22 = b'not_base64'
    var_23 = 'test_string'
    var_24 = var_1.sign(var_23)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'my-salt'
    var_3 = module_0.TimestampSigner(var_0, var_2)
    var_4 = ':'
    var_5 = module_0.TimestampSigner(var_0, sep=var_4)
    var_6 = 'digest_method'
    var_7 = hasattr(var_1, var_6)
    var_8 = 'key_derivation'
    var_9 = hasattr(var_1, var_8)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test_value'
    var_5 = 'test_value'
    var_6 = var_1.sign(var_5)
    var_7 = True
    var_8 = var_1.sign(var_5)
    var_9 = 3600
    var_10 = var_1.unsign(var_8, var_9)
    assert var_10 == b'test_value'
    var_11 = module_0.TimestampSigner(var_0)
    var_12 = var_11.get_timestamp
    var_13 = 100
    var_14 = var_11.sign(var_5)
    var_15 = 200
    var_16 = 50
    var_17 = var_11.unsign(var_14, var_16)
    var_18 = module_0.TimestampSigner(var_16)
    var_19 = 1000
    var_20 = var_18.sign(var_17)
    var_21 = 500
    var_22 = 100
    var_23 = var_18.unsign(var_20, var_22)
    var_24 = b'test_value.sep.invalid_timestamp.sep.signature'
    var_25 = var_1.unsign(var_24)
    var_26 = module_1.Signer(var_25)
    var_27 = var_26.sign(var_23)
    var_28 = var_1.unsign(var_27)
    var_29 = 'different-secret'
    var_30 = module_0.TimestampSigner(var_29)
    var_31 = var_30.sign(var_23)
    var_32 = var_1.unsign(var_31)
    var_33 = b'bytes_value'
    var_34 = var_1.sign(var_33)
    var_35 = var_1.unsign(var_34)
    assert var_35 == b'bytes_value'
    assert var_35 == b'string_value'
    var_36 = 'string_value'
    var_37 = var_1.sign(var_36)
    var_38 = var_1.sign(var_23)
    var_39 = module_0.TimestampSigner(var_32)
    var_40 = var_39.get_timestamp()
    var_41 = var_40 + var_13
    var_42 = var_39.sign(var_23)
    var_43 = var_40 + var_13
    var_44 = var_39.unsign(var_42, var_13)
    assert var_44 == b'test_value'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic functionality.'
    var_1 = module_0.TimedSerializer()
    var_2 = 'test-secret'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = 'test-salt'
    var_5 = module_0.TimedSerializer(var_4)
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}
    var_9 = module_0.TimedSerializer(signer_kwargs=var_8)
    var_10 = 'serializer'
    var_11 = 'json'
    var_12 = {var_10: var_11}
    var_13 = module_0.TimedSerializer(serializer_kwargs=var_12)
    var_14 = 'test'
    var_15 = module_0.TimedSerializer(var_14)
    var_16 = 'key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = 3600
    var_20 = True



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test_value'
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    assert var_7 == b'test_value'
    var_8 = 0.1
    var_9 = 0
    var_10 = var_1.unsign(var_3, var_9)
    var_11 = var_1.get_timestamp
    var_12 = 100
    var_13 = 50
    var_14 = var_1.unsign(var_3, var_13)
    var_15 = b'tampered'
    var_16 = 8
    var_17 = var_3[var_16:]
    var_18 = var_15 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = b'no_timestamp'
    var_21 = var_1.unsign(var_20)
    var_22 = b'value'
    var_23 = b'invalid_timestamp'
    var_24 = b'signature'
    var_25 = 'test'
    var_26 = var_1.sign(var_25)
    var_27 = 0
    var_28 = b'corrupted'



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = 'test string'
    var_6 = var_1.sign(var_5)
    var_7 = var_1.unsign(var_6)
    assert var_7 == b'test string'
    var_8 = True
    var_9 = var_1.unsign(var_3, return_timestamp=var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = var_9[var_8]
    var_12 = 3600
    var_13 = var_1.unsign(var_3, var_12)
    var_14 = var_1.get_timestamp
    var_15 = int(var_0)
    var_16 = 100
    var_17 = var_15 - var_16
    var_18 = var_1.sign(var_2)
    var_19 = int(var_10)
    var_20 = 10
    var_21 = var_1.unsign(var_18, var_20)
    var_22 = int(var_20)
    var_23 = 100
    var_24 = var_22 + var_23
    var_25 = var_1.sign(var_2)
    var_26 = int(var_10)
    var_27 = 3600
    var_28 = var_1.unsign(var_25, var_27)
    var_29 = b'invalid'
    var_30 = var_1.unsign(var_29)
    var_31 = -1
    var_32 = var_3[:var_31]
    var_33 = b'x'
    var_34 = var_32 + var_33
    var_35 = var_1.unsign(var_34)
    var_36 = b'test'
    var_37 = b'invalid_ts'
    var_38 = '|'
    var_39 = module_0.TimestampSigner(var_35, sep=var_38)
    var_40 = var_39.sign(var_2)
    var_41 = var_39.unsign(var_40)
    var_42 = int(var_35)
    var_43 = 100
    var_44 = var_42 - var_43
    var_45 = var_1.sign(var_2)
    var_46 = int(var_10)
    var_47 = 10
    var_48 = True
    var_49 = var_1.unsign(var_45, var_47, var_48)
    var_50 = b''
    var_51 = var_1.sign(var_50)
    var_52 = var_1.unsign(var_51)
    assert var_52 == b''
    var_53 = b'value'
    var_54 = var_1.unsign(var_3)
    var_55 = module_1.Signer(var_47)
    var_56 = var_55.sign(var_2)
    var_57 = var_1.unsign(var_56)



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = '|'
    var_3 = 'custom-salt'
    var_4 = 'none'
    var_5 = var_1.get_timestamp()
    var_6 = var_1.timestamp_to_datetime(var_5)
    var_7 = 'test-value'
    var_8 = var_1.sign(var_7)
    var_9 = var_1.unsign(var_8)
    assert var_9 == b'test-value'
    var_10 = True
    var_11 = var_1.validate(var_8)
    assert var_11 is True
    var_12 = b'invalid-signature'
    var_13 = var_1.validate(var_12)
    assert var_13 is False
    var_14 = -1
    var_15 = var_1.unsign(var_8, var_14)
    var_16 = b'bytes-secret'
    var_17 = module_0.TimestampSigner(var_16)
    var_18 = b''
    var_19 = var_1.sign(var_18)
    var_20 = var_1.unsign(var_19)
    assert var_20 == b''



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-key'
    var_3 = 'custom-salt'
    var_4 = '|'
    var_5 = 'none'
    var_6 = 'sha256'
    var_7 = b'bytes-key'
    var_8 = module_0.TimestampSigner(var_7)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimestampSigner constructor and basic functionality.'
    var_1 = 'secret-key'
    var_2 = module_0.TimestampSigner(var_1)
    var_3 = 'custom-salt'
    var_4 = ':'
    var_5 = 'none'
    var_6 = var_2.get_timestamp()
    var_7 = var_2.timestamp_to_datetime(var_6)
    var_8 = 'my-secret'
    var_9 = module_0.TimestampSigner(var_8)
    var_10 = b'bytes-secret'
    var_11 = module_0.TimestampSigner(var_10)
    var_12 = 'test-value'
    var_13 = var_2.sign(var_12)
    var_14 = b'.'
    var_15 = 2
    var_16 = var_2.unsign(var_13)
    assert var_16 == b'test-value'
    var_17 = True
    var_18 = var_2.validate(var_13)
    assert var_18 is True
    var_19 = b'invalid-signature'
    var_20 = var_2.validate(var_19)
    assert var_20 is False
    var_21 = 3600
    var_22 = var_2.unsign(var_13, var_21)
    assert var_22 == b'test-value'
    var_23 = 7200
    var_24 = 'old-value'
    var_25 = var_2.sign(var_24)
    var_26 = 0
    var_27 = 3600
    var_28 = b'bytes-value'
    var_29 = var_2.sign(var_28)
    var_30 = var_2.unsign(var_29)
    assert var_30 == b'bytes-value'
    var_31 = ''
    var_32 = var_2.sign(var_31)
    var_33 = var_2.unsign(var_32)
    assert var_33 == b''



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test message'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = 'test message'
    var_6 = var_1.sign(var_5)
    var_7 = var_1.unsign(var_6)
    assert var_7 == b'test message'
    var_8 = var_1.sign(var_2)
    var_9 = True
    var_10 = var_1.sign(var_2)
    var_11 = 3600
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = module_0.TimestampSigner(var_0)
    var_14 = var_13.get_timestamp
    var_15 = 100
    var_16 = var_13.sign(var_2)
    var_17 = 10
    var_18 = var_13.unsign(var_16, var_17)
    var_19 = module_0.TimestampSigner(var_17)
    var_20 = var_19.sign(var_2)
    var_21 = 3600
    var_22 = var_1.unsign(var_20, var_21)
    var_23 = b'tampered'
    var_24 = 5
    var_25 = var_3[var_24:]
    var_26 = var_23 + var_25
    var_27 = var_1.unsign(var_26)
    var_28 = module_1.Signer(var_23)
    var_29 = var_28.sign(var_2)
    var_30 = var_1.unsign(var_29)
    var_31 = -5
    var_32 = var_3[:var_31]
    var_33 = b'!!!!!'
    var_34 = var_32 + var_33
    var_35 = -5
    var_36 = var_3[var_35:]
    var_37 = var_34 + var_36
    var_38 = var_1.unsign(var_37)
    var_39 = var_1.validate(var_3)
    assert var_39 is True
    var_40 = b'invalid'
    var_41 = var_1.validate(var_40)
    assert var_41 is False
    var_42 = var_1.validate(var_3, var_27)
    assert var_42 is True
    var_43 = 10
    var_44 = var_1.validate(var_16, var_43)
    assert var_44 is False
    var_45 = b''
    var_46 = var_1.sign(var_45)
    var_47 = var_1.unsign(var_46)
    assert var_47 == b''



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = var_7.signer.get_timestamp
    var_9 = 10000
    var_10 = 1
    var_11 = b'invalid-data'
    var_12 = var_1.loads(var_11)
    var_13 = 'different-salt'



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = module_0.TimestampSigner(var_0)
    var_8 = 1000
    var_9 = b'test'
    var_10 = var_7.sign(var_9)
    var_11 = 10
    var_12 = var_1.loads(var_10, var_11)
    var_13 = b'invalid-signature'
    var_14 = var_1.loads(var_13)
    var_15 = 'test-secret-2'
    var_16 = module_0.TimedSerializer(var_15)
    var_17 = 'different'
    var_18 = module_0.TimedSerializer(var_13, var_17)
    var_19 = 'custom-salt'
    var_20 = 'wrong-salt'
    var_21 = 'utf-8'
    var_22 = {}



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = '|'
    var_5 = 'none'
    var_6 = 'get_timestamp'
    var_7 = hasattr(var_1, var_6)
    var_8 = var_1.get_timestamp()
    var_9 = 'timestamp_to_datetime'
    var_10 = hasattr(var_1, var_9)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = -1
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = 3600
    var_11 = b'invalid-signature'
    var_12 = var_1.unsign(var_11)
    var_13 = -1
    var_14 = var_3[:var_13]
    var_15 = b'X'
    var_16 = var_14 + var_15
    var_17 = var_1.unsign(var_16)
    var_18 = b'.'
    var_19 = var_2 + var_18
    var_20 = var_2 + var_18
    var_21 = b'invalid-timestamp'
    var_22 = var_20 + var_21
    var_23 = var_22 + var_18
    var_24 = var_2 + var_18
    var_25 = var_24 + var_21
    var_26 = 'utf-8'
    var_27 = '|'
    var_28 = module_0.TimestampSigner(var_17, sep=var_27)
    var_29 = var_28.sign(var_2)
    var_30 = var_28.unsign(var_29)
    var_31 = b'test-bytes-value'
    var_32 = var_1.sign(var_31)
    var_33 = var_1.unsign(var_32)
    var_34 = 3600



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6[var_5]
    var_9 = var_6[var_5]
    var_10 = 3600
    var_11 = var_1.unsign(var_3, var_10)
    var_12 = 10
    var_13 = -1
    var_14 = var_3[:var_13]
    var_15 = b'x'
    var_16 = var_14 + var_15
    var_17 = var_1.unsign(var_16)
    var_18 = 0
    var_19 = b'invalid_timestamp'
    var_20 = module_1.Signer(var_17)
    var_21 = var_20.sign(var_2)
    var_22 = var_1.unsign(var_21)
    var_23 = 3600
    var_24 = var_1.unsign(var_3)



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test-value'
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    assert var_9 == b'test-value'
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = 100
    var_12 = var_10.sign(var_2)
    var_13 = module_0.TimestampSigner(var_0)
    var_14 = 50
    var_15 = var_13.unsign(var_12, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = var_16.sign(var_15)
    var_18 = 3600
    var_19 = var_13.unsign(var_17, var_18)
    var_20 = b'invalid-data'
    var_21 = var_1.unsign(var_20)
    var_22 = b'test-value'
    var_23 = b'invalid-timestamp'
    var_24 = 'wrong-secret'
    var_25 = module_0.TimestampSigner(var_24)
    var_26 = var_1.sign(var_21)
    var_27 = var_25.unsign(var_26)
    var_28 = 'signature mismatch'
    var_29 = 'bad signature'
    var_30 = var_1.sign(var_21)
    var_31 = b'test-bytes'
    var_32 = var_1.sign(var_31)
    var_33 = var_1.unsign(var_32)
    assert var_33 == b'test-bytes'



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test_value'
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    assert var_9 == b'test_value'
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = var_10.get_timestamp
    var_12 = 100
    var_13 = var_10.sign(var_2)
    var_14 = 50
    var_15 = var_10.unsign(var_13, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = var_16.sign(var_15)
    var_18 = 3600
    var_19 = var_16.unsign(var_17, var_18)
    var_20 = b'invalid_signature'
    var_21 = var_1.unsign(var_20)
    var_22 = b'test_value'
    var_23 = b'invalid_timestamp'
    var_24 = b'signature'
    var_25 = var_1.sign(var_21)
    var_26 = 0



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'secret-key'
    var_3 = 'custom-salt'
    var_4 = ':'
    var_5 = 'none'
    var_6 = 'hs512'
    var_7 = b'bytes-secret'
    var_8 = module_0.TimestampSigner(var_7)
    var_9 = var_8.get_timestamp()
    var_10 = var_8.timestamp_to_datetime(var_9)
    var_11 = 'test-value'
    var_12 = var_8.sign(var_11)
    var_13 = var_8.unsign(var_12)
    assert var_13 == b'test-value'
    var_14 = True
    var_15 = b'bytes-value'
    var_16 = var_8.sign(var_15)
    var_17 = var_8.unsign(var_16)
    assert var_17 == b'bytes-value'
    var_18 = var_8.validate(var_12)
    assert var_18 is True
    var_19 = b'invalid'
    var_20 = var_8.validate(var_19)
    assert var_20 is False
    var_21 = 'test'
    var_22 = var_8.sign(var_21)
    var_23 = 3600
    var_24 = var_8.unsign(var_22, var_23)
    assert var_24 == b'test'
    var_25 = 100
    var_26 = module_0.TimestampSigner(var_0)
    var_27 = 'old-value'
    var_28 = var_26.sign(var_27)
    var_29 = 50
    var_30 = var_8.unsign(var_28, var_29)
    var_31 = b'invalid-signature'
    var_32 = var_8.unsign(var_31)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test-value'
    var_6 = var_2.sign(var_3)
    var_7 = True
    var_8 = var_2.sign(var_3)
    var_9 = 3600
    var_10 = var_2.unsign(var_8, var_9)
    assert var_10 == b'test-value'
    var_11 = var_2.get_timestamp
    var_12 = 100
    var_13 = var_2.sign(var_3)
    var_14 = 10
    var_15 = var_2.unsign(var_13, var_14)
    var_16 = var_2.get_timestamp
    var_17 = var_2.sign(var_3)
    var_18 = 3600
    var_19 = var_2.unsign(var_17, var_18)
    var_20 = b'invalid-signature'
    var_21 = var_2.unsign(var_20)
    var_22 = b'not-a-timestamp'
    var_23 = module_1.base64_encode(var_22)
    var_24 = b'test-value'
    var_25 = b'signature'
    var_26 = var_2.sign(var_24)
    var_27 = var_2.sign(var_3)
    var_28 = 2
    var_29 = 1
    var_30 = var_2.unsign(var_27, var_29)



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = 'test_string'
    var_6 = var_1.sign(var_5)
    var_7 = var_1.unsign(var_6)
    assert var_7 == b'test_string'
    var_8 = True
    var_9 = var_1.unsign(var_3, return_timestamp=var_8)
    var_10 = 'Expected tuple when return_timestamp=True'
    var_11 = len(var_9)
    assert var_11 == 2
    var_12 = var_9[var_8]
    var_13 = 'Expected datetime object'
    var_14 = 3600
    var_15 = var_1.unsign(var_3, var_14)
    var_16 = module_0.TimestampSigner(var_0)
    var_17 = 100
    var_18 = var_16.sign(var_2)
    var_19 = 50
    var_20 = var_1.unsign(var_18, var_19)
    var_21 = -1
    var_22 = var_3[:var_21]
    var_23 = b'x'
    var_24 = var_22 + var_23
    var_25 = var_1.unsign(var_24)
    var_26 = b'fake_signature'
    var_27 = b'!!invalid!!'
    var_28 = b'sig'
    var_29 = module_0.TimestampSigner(var_25)
    var_30 = var_29.sign(var_2)
    var_31 = 3600
    var_32 = var_1.unsign(var_30, var_31)
    var_33 = 'wrong-key'
    var_34 = module_0.TimestampSigner(var_33)
    var_35 = var_34.sign(var_2)
    var_36 = var_1.unsign(var_35)
    var_37 = b''
    var_38 = var_1.sign(var_37)
    var_39 = var_1.unsign(var_38)
    var_40 = b'a'
    var_41 = 10000
    var_42 = var_40 * var_41
    var_43 = var_1.sign(var_42)
    var_44 = var_1.unsign(var_43)
    var_45 = var_1.sign(var_2)



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = '|'
    var_5 = 'none'
    var_6 = 'hmac-sha256'
    var_7 = 'test'
    var_8 = module_0.TimestampSigner(var_7)
    var_9 = b'bytes-secret'
    var_10 = module_0.TimestampSigner(var_9)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = var_0.get_timestamp
    var_2 = callable(var_1)
    var_3 = var_0.timestamp_to_datetime
    var_4 = callable(var_3)
    var_5 = var_0.sign
    var_6 = callable(var_5)
    var_7 = var_0.unsign
    var_8 = callable(var_7)
    var_9 = var_0.validate
    var_10 = callable(var_9)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = var_6[var_5]
    var_8 = 3600
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = var_10.get_timestamp
    var_12 = 100
    var_13 = var_10.sign(var_2)
    var_14 = 50
    var_15 = var_10.unsign(var_13, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = var_16.sign(var_2)
    var_18 = 3600
    var_19 = var_16.unsign(var_17, var_18)
    var_20 = b'test_value'
    var_21 = b'invalid_timestamp'
    var_22 = b'just_data'
    var_23 = b'MTIzNDU='
    var_24 = b'invalid'



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test the loads method of TimedSerializer.'
    var_1 = 'test-secret-key-12345'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 3600
    var_7 = True
    var_8 = 0.1
    var_9 = 0
    var_10 = b'invalid-data'
    var_11 = var_2.loads(var_10)
    var_12 = b''
    var_13 = var_2.loads(var_12)
    var_14 = 'different-salt'
    var_15 = module_0.TimedSerializer(var_13, var_14)
    var_16 = 'utf-8'
    var_17 = 'string'
    var_18 = 'number'
    var_19 = 'list'
    var_20 = 'nested'
    var_21 = 'hello'
    var_22 = 42
    var_23 = 2
    var_24 = 3
    var_25 = [var_7, var_23, var_24]
    var_26 = 'a'
    var_27 = 'b'
    var_28 = {var_26: var_7, var_27: var_23}
    var_29 = {var_17: var_21, var_18: var_22, var_19: var_25, var_20: var_28}
    var_30 = None
    var_31 = 12345



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic functionality.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = var_2.default_signer
    var_4 = b'test-secret-bytes'
    var_5 = module_0.TimedSerializer(var_4)
    var_6 = 'test'
    var_7 = 'custom-salt'
    var_8 = module_0.TimedSerializer(var_6, var_7)
    var_9 = 'skipkeys'
    var_10 = True
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(var_6, serializer_kwargs=var_11)
    var_13 = 'key_derivation'
    var_14 = 'hmac'
    var_15 = {var_13: var_14}
    var_16 = module_0.TimedSerializer(var_6, signer_kwargs=var_15)
    var_17 = module_0.TimedSerializer(var_6)
    var_18 = var_17.iter_unsigners()
    var_19 = list(var_18)
    var_20 = len(var_19)
    var_21 = module_0.TimedSerializer()
    var_22 = 'different-salt'



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test-salt'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(signer_kwargs=var_7)
    var_9 = 'serializer'
    var_10 = 'json'
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(serializer_kwargs=var_11)
    var_13 = 'fallback'
    var_14 = module_0.TimestampSigner(var_13)
    var_15 = [var_14]
    var_16 = module_0.TimedSerializer(fallback_signers=var_15)
    var_17 = 'test'
    var_18 = var_16.iter_unsigners(var_17)
    var_19 = list(var_18)
    var_20 = len(var_19)
    var_21 = 'key'
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = True
    var_25 = 0.1
    var_26 = 3600



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test the constructor of TimedSerializer class.'
    var_1 = module_0.TimedSerializer()
    var_2 = 'load_payload'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'dump_payload'
    var_5 = hasattr(var_1, var_4)
    var_6 = 'test-secret-key'
    var_7 = module_0.TimedSerializer(var_6)
    var_8 = 'test-salt'
    var_9 = module_0.TimedSerializer(var_8)
    var_10 = 'salt'
    var_11 = hasattr(var_9, var_10)
    var_12 = 'key_derivation'
    var_13 = 'hmac'
    var_14 = {var_12: var_13}
    var_15 = module_0.TimedSerializer(signer_kwargs=var_14)
    var_16 = 'signer_kwargs'
    var_17 = hasattr(var_15, var_16)
    var_18 = 'serializer'
    var_19 = 'json'
    var_20 = {var_18: var_19}
    var_21 = module_0.TimedSerializer(serializer_kwargs=var_20)
    var_22 = 'serializer_kwargs'
    var_23 = hasattr(var_21, var_22)
    var_24 = module_0.TimedSerializer()
    var_25 = 'test'
    var_26 = module_0.TimedSerializer(var_25)
    var_27 = 'test-secret'
    var_28 = module_0.TimedSerializer(var_27)
    var_29 = 'number'
    var_30 = 'data'
    var_31 = 42
    var_32 = {var_25: var_30, var_29: var_31}
    var_33 = 3600
    var_34 = True



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = 3600
    var_7 = {var_2: var_3}
    var_8 = True
    var_9 = {var_2: var_3}
    var_10 = {var_2: var_3}
    var_11 = 'custom-salt'
    var_12 = module_0.TimedSerializer(var_0)
    var_13 = {var_2: var_3}
    var_14 = 0
    var_15 = b'invalid-data'
    var_16 = var_1.loads(var_15)
    var_17 = 'secret1'
    var_18 = module_0.TimedSerializer(var_17)
    var_19 = 'secret2'
    var_20 = module_0.TimedSerializer(var_19)
    var_21 = {var_16: var_3}
    var_22 = 'list'
    var_23 = 'nested'
    var_24 = 'number'
    var_25 = 2
    var_26 = 3
    var_27 = [var_8, var_25, var_26]
    var_28 = 'key'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = 42
    var_32 = {var_22: var_27, var_23: var_30, var_24: var_31}
    var_33 = {var_16: var_3}



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test-salt'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = None
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads with various scenarios.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 3600
    var_7 = True
    var_8 = 'test-key'
    var_9 = 'digest_method'
    var_10 = 'sha1'
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(var_8, signer_kwargs=var_11)
    var_13 = 0
    var_14 = b'invalid-data'
    var_15 = var_2.loads(var_14)
    var_16 = 'custom-salt'
    var_17 = 'wrong-salt'
    var_18 = 'utf-8'
    var_19 = 'list'
    var_20 = 'nested'
    var_21 = 'bool'
    var_22 = 'none'
    var_23 = 2
    var_24 = 3
    var_25 = [var_7, var_23, var_24]
    var_26 = 'a'
    var_27 = {var_26: var_7}
    var_28 = None
    var_29 = {var_19: var_25, var_20: var_27, var_21: var_7, var_22: var_28}
    var_30 = {}
    var_31 = 'two'
    var_32 = [var_7, var_31, var_24]
    var_33 = 'test string'
    var_34 = 42
    var_35 = 3.14159
    var_36 = True
    var_37 = None
    var_38 = 'different-key'
    var_39 = module_0.TimedSerializer(var_38)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'message'
    var_3 = 'data'
    var_4 = 'hello'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = {var_2: var_4, var_3: var_8}
    var_10 = 3600
    var_11 = True
    var_12 = True
    var_13 = module_0.TimedSerializer(var_0)
    var_14 = 0.1
    var_15 = 0
    var_16 = b'invalid-signed-data'
    var_17 = var_1.loads(var_16)
    var_18 = 'utf-8'
    var_19 = 'custom-salt'
    var_20 = 'wrong-salt'
    var_21 = 3600
    var_22 = b'.'
    var_23 = 0
    var_24 = b'malformed-timestamp'
    var_25 = {}
    var_26 = 'list'
    var_27 = 'dict'
    var_28 = 'tuple'
    var_29 = [var_12, var_6, var_7]
    var_30 = 'a'
    var_31 = 'b'
    var_32 = {var_30: var_12, var_31: var_6}
    var_33 = (var_12, var_6)
    var_34 = {var_26: var_29, var_27: var_32, var_28: var_33}
    var_35 = True



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = b'test_value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = True
    var_7 = 3600
    var_8 = var_2.unsign(var_4, var_7)
    var_9 = 100
    var_10 = 3600
    var_11 = b'invalid_signature'
    var_12 = var_2.unsign(var_11)
    var_13 = module_1.Signer(var_11, var_12)
    var_14 = var_13.sign(var_3)
    var_15 = var_2.unsign(var_14)
    var_16 = 'utf-8'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test the TimedSerializer constructor and basic functionality.'
    var_1 = module_0.TimedSerializer()
    var_2 = 'my-secret-key'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = 'my-salt'
    var_5 = module_0.TimedSerializer(var_4)
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}
    var_9 = module_0.TimedSerializer(signer_kwargs=var_8)
    var_10 = module_0.TimedSerializer()
    var_11 = 'test'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = True
    var_15 = 3600
    var_16 = 0
    var_17 = b'invalid-data'
    var_18 = var_10.loads(var_17)
    var_19 = b'invalid-data'
    var_20 = var_10.loads_unsafe(var_19)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    assert var_9 == b'string_value'
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = var_10.get_timestamp
    var_12 = 100
    var_13 = var_10.sign(var_2)
    var_14 = 50
    var_15 = var_10.unsign(var_13, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = var_16.sign(var_2)
    var_18 = 3600
    var_19 = var_16.unsign(var_17, var_18)
    var_20 = b'value'
    var_21 = b'not_base64'
    var_22 = -1
    var_23 = var_7[:var_22]
    var_24 = -1
    var_25 = var_7[var_24:]
    var_26 = b'x'
    var_27 = var_25 != var_26
    var_28 = b'y'
    var_29 = var_26 if var_27 else var_28
    var_30 = var_23 + var_29
    var_31 = var_1.unsign(var_30)
    var_32 = 'string_value'
    var_33 = var_1.sign(var_32)
    var_34 = b''
    var_35 = var_1.sign(var_34)
    var_36 = var_1.unsign(var_35)
    assert var_36 == b''
    var_37 = module_0.TimestampSigner(var_31)
    var_38 = var_37.sign(var_2)
    var_39 = 50
    var_40 = True
    var_41 = var_37.unsign(var_38, var_39, var_40)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimedSerializer(var_0, var_2)
    var_4 = 'key_derivation'
    var_5 = 'hmac'
    var_6 = {var_4: var_5}
    var_7 = module_0.TimedSerializer(var_0, signer_kwargs=var_6)



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.signer
    var_2 = 'test-secret'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = var_3.signer
    var_5 = 'test-salt'
    var_6 = module_0.TimedSerializer(var_5)
    var_7 = var_6.signer
    var_8 = module_0.TimedSerializer(var_2, var_5)
    var_9 = var_8.signer
    var_10 = 'key_derivation'
    var_11 = 'hmac'
    var_12 = {var_10: var_11}
    var_13 = module_0.TimedSerializer(signer_kwargs=var_12)
    var_14 = var_13.signer
    var_15 = 'serializer'
    var_16 = 'json'
    var_17 = {var_15: var_16}
    var_18 = module_0.TimedSerializer(serializer_kwargs=var_17)
    var_19 = var_18.signer
    var_20 = module_0.TimedSerializer()
    var_21 = var_20.signer
    var_22 = module_0.TimedSerializer()
    var_23 = module_0.TimedSerializer(var_2, var_5)



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'mysecret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'mysalt'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(signer_kwargs=var_7)
    var_9 = 'test'
    var_10 = module_0.TimedSerializer(var_9)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 3600
    var_5 = True
    var_6 = module_0.TimedSerializer()
    var_7 = 0.1
    var_8 = 0
    var_9 = b'invalid'
    var_10 = var_0.loads(var_2)
    var_11 = b'invalid.data'
    var_12 = var_0.loads(var_11)
    var_13 = 'custom_salt'
    var_14 = 'wrong_salt'
    var_15 = b''
    var_16 = var_0.loads(var_15)
    var_17 = None
    var_18 = 0
    var_19 = True
    var_20 = 'different_key'
    var_21 = module_0.TimedSerializer(var_20)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test_value'
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    assert var_9 == b'test_value'
    var_10 = var_1.get_timestamp
    var_11 = 100
    var_12 = 'old_value'
    var_13 = var_1.sign(var_12)
    var_14 = 50
    var_15 = var_1.unsign(var_13, var_14)
    var_16 = 'future_value'
    var_17 = var_1.sign(var_16)
    var_18 = 1000
    var_19 = 0
    var_20 = -2
    var_21 = -2
    var_22 = var_1.sign(var_16)
    var_23 = 3600
    var_24 = var_1.unsign(var_22, var_23)
    var_25 = b'invalid_signature'
    var_26 = var_1.unsign(var_25)
    var_27 = b'test_value'
    var_28 = b'not_a_timestamp'
    var_29 = b'tampered'
    var_30 = len(var_27)
    var_31 = var_7[var_30:]
    var_32 = var_29 + var_31
    var_33 = var_1.unsign(var_32)
    var_34 = var_1.sign(var_26)
    var_35 = b''
    var_36 = var_1.sign(var_35)
    var_37 = var_1.unsign(var_36)
    assert var_37 == b''
    var_38 = b'bytes_value'
    var_39 = var_1.sign(var_38)
    var_40 = var_1.unsign(var_39)
    assert var_40 == b'bytes_value'
    var_41 = 'string_value'
    var_42 = var_1.sign(var_41)
    var_43 = var_1.unsign(var_42)
    assert var_43 == b'string_value'



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic initialization.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'another-secret'
    var_4 = 'custom-salt'
    var_5 = 'json'
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}
    var_9 = module_0.TimedSerializer(var_3, var_4, var_5, signer_kwargs=var_8)
    var_10 = 'secret'
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = True



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = -1
    var_8 = 'invalid-data'
    var_9 = var_1.loads(var_8)
    var_10 = -1
    var_11 = b'x'
    var_12 = 'custom-salt'
    var_13 = module_0.TimedSerializer(var_8, var_12)
    var_14 = {}
    var_15 = 'key_derivation'
    var_16 = 'none'
    var_17 = {var_15: var_16}
    var_18 = [var_14, var_17]
    var_19 = module_0.TimedSerializer(var_8, signer_kwargs=var_18)
    var_20 = ''
    var_21 = var_1.loads(var_20)
    var_22 = 0.1
    var_23 = 0



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test the constructor of TimedSerializer class.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = var_2.signer
    var_4 = 'custom-secret'
    var_5 = 'custom-salt'
    var_6 = module_0.TimedSerializer(var_4, var_5)
    var_7 = var_6.signer
    var_8 = 'key_derivation'
    var_9 = 'hmac'
    var_10 = {var_8: var_9}
    var_11 = module_0.TimedSerializer(var_1, serializer_kwargs=var_10)
    var_12 = var_11.signer
    var_13 = 'digest_method'
    var_14 = var_11.signer
    var_15 = var_11.signer
    var_16 = module_0.TimedSerializer(var_1)
    var_17 = b'bytes-secret'
    var_18 = module_0.TimedSerializer(var_17)
    var_19 = b'bytes-salt'
    var_20 = module_0.TimedSerializer(var_1, var_19)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'secret-key'
    var_1 = b'test_value'
    var_2 = True
    var_3 = 3600
    var_4 = 10
    var_5 = 9999999999
    var_6 = 10
    var_7 = b'invalid_signature'
    var_8 = -1
    var_9 = -1
    var_10 = 255
    var_11 = 0
    var_12 = b'invalid_base64'
    var_13 = b'no_separator'
    var_14 = 'test_string'



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    var_10 = var_1.get_timestamp
    var_11 = 100
    var_12 = var_1.sign(var_2)
    var_13 = 10
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = var_1.get_timestamp
    var_16 = var_1.sign(var_2)
    var_17 = 10
    var_18 = var_1.unsign(var_16, var_17)
    var_19 = b'malformed'
    var_20 = var_16 + var_19
    var_21 = var_1.unsign(var_20)
    var_22 = module_1.Signer(var_21)
    var_23 = var_22.sign(var_2)
    var_24 = var_1.unsign(var_23)
    var_25 = -1
    var_26 = var_16[:var_25]
    var_27 = b'x'
    var_28 = var_26 + var_27
    var_29 = var_1.unsign(var_28)
    var_30 = 'test_string'
    var_31 = var_1.sign(var_30)
    var_32 = var_1.unsign(var_31)
    assert var_32 == b'test_string'



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 7200
    var_8 = 3600
    var_9 = 'different-salt'
    var_10 = module_0.TimedSerializer(var_8, var_9)
    var_11 = module_0.TimedSerializer(var_8)
    var_12 = 'salt'
    var_13 = 'fallback'
    var_14 = b'invalid-data'
    var_15 = var_1.loads(var_14)
    var_16 = b'.'
    var_17 = b'not-a-timestamp'
    var_18 = module_1.base64_encode(var_17)
    var_19 = 0
    var_20 = 2



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = '|'
    var_4 = module_0.TimestampSigner(var_0, var_2, var_3)
    var_5 = 'hmac'
    var_6 = 'sha256'
    var_7 = module_0.TimestampSigner(var_0, key_derivation=var_5, digest_method=var_6)
    var_8 = module_0.TimestampSigner()



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic attributes.'
    var_1 = module_0.TimedSerializer()
    var_2 = 'secret-key'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = 'my-salt'
    var_5 = module_0.TimedSerializer(var_4)
    var_6 = 'key_derivation'
    var_7 = 'none'
    var_8 = {var_6: var_7}
    var_9 = module_0.TimedSerializer(signer_kwargs=var_8)
    var_10 = 'signer_kwargs'
    var_11 = 'hmac'
    var_12 = {var_6: var_11}
    var_13 = {var_10: var_12}
    var_14 = module_0.TimedSerializer(serializer_kwargs=var_13)
    var_15 = module_0.TimedSerializer(var_2)
    var_16 = module_0.TimedSerializer(var_2)



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = 'custom-salt'
    var_8 = ''
    var_9 = module_0.TimedSerializer(var_8)
    var_10 = 'test'
    var_11 = 'different-salt'
    var_12 = 'json'
    var_13 = 'key_derivation'
    var_14 = 'hmac'
    var_15 = {var_13: var_14}
    var_16 = module_0.TimedSerializer(var_10, var_11, var_12, signer_kwargs=var_15)



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test basic loads without max_age or return_timestamp.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test loads with max_age parameter.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test_data'
    var_4 = 3600
    var_5 = 0

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test loads with return_timestamp=True.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test_data'
    var_4 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test loads with both max_age and return_timestamp.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test_data'
    var_4 = 3600
    var_5 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test loads with salt parameter.'
    var_1 = 'secret-key'
    var_2 = 'my-salt'
    var_3 = module_0.TimedSerializer(var_1, var_2)
    var_4 = 'test_data'
    var_5 = 'wrong-salt'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test loads with invalid signature.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = b'invalid_data'
    var_4 = var_2.loads(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test loads with empty string.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = b''
    var_4 = var_2.loads(var_3)

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test loads with tampered data.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test_data'
    var_4 = -1
    var_5 = b'x'

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test loads with multiple signers.'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'key3'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.TimedSerializer(var_4)
    var_6 = 'test_data'

import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test loads with an older timestamp.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test_data'
    var_4 = module_0.TimestampSigner(var_1)
    var_5 = 100
    var_6 = module_1.int_to_bytes(var_5)
    var_7 = module_1.base64_encode(var_6)
    var_8 = 'test_data'
    var_9 = module_1.want_bytes(var_8)
    var_10 = var_4.sep
    var_11 = module_1.want_bytes(var_10)
    var_12 = var_9 + var_11
    var_13 = var_12 + var_7
    var_14 = var_13 + var_11
    var_15 = var_9 + var_11
    var_16 = var_15 + var_7
    var_17 = 50

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test return type of loads method.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test_data'
    var_4 = None
    var_5 = type(var_4)
    var_6 = True

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test loads with complex data types.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'string'
    var_4 = 'number'
    var_5 = 'list'
    var_6 = 'nested'
    var_7 = 'bool'
    var_8 = 'none'
    var_9 = 'hello'
    var_10 = 42
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = True
    var_19 = None
    var_20 = {var_3: var_9, var_4: var_10, var_5: var_14, var_6: var_17, var_7: var_18, var_8: var_19}

import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test loads with unicode data.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'héllo wörld 🌍'

def test_case_0():
    var_0 = 'Test loads with custom signer class.'
    var_1 = 'secret-key'
    var_2 = 'test_data'

def test_case_0():
    var_0 = 'Test loads with custom signer class.'
    var_1 = 'secret-key'
    var_2 = 'test_data'



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_data'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = 3600
    var_8 = var_1.unsign(var_3, var_7)
    assert var_8 == b'string_data'
    var_9 = 7200
    var_10 = var_1.sign(var_2)
    var_11 = 3600
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = var_1.sign(var_2)
    var_14 = 3600
    var_15 = var_1.unsign(var_13, var_14)
    var_16 = 'age -'
    var_17 = 'age 0'
    var_18 = -1
    var_19 = var_3[:var_18]
    var_20 = b'X'
    var_21 = var_19 + var_20
    var_22 = var_1.unsign(var_21)
    var_23 = module_1.Signer(var_22)
    var_24 = b'no_timestamp'
    var_25 = var_23.sign(var_24)
    var_26 = var_1.unsign(var_25)
    var_27 = b'.'
    var_28 = var_2 + var_27
    var_29 = b'inval1d_t1mestamp'
    var_30 = var_28 + var_29
    var_31 = var_1.unsign(var_30)
    var_32 = 'string_data'
    var_33 = var_1.sign(var_32)
    var_34 = module_0.TimestampSigner(var_31)
    var_35 = 100000
    var_36 = var_34.sign(var_2)
    var_37 = None
    var_38 = var_34.unsign(var_36, var_37)
    var_39 = var_1.unsign(var_3, var_7, var_15)
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = var_39[var_15]
    var_42 = '|'
    var_43 = module_0.TimestampSigner(var_31, sep=var_42)
    var_44 = var_43.sign(var_2)
    var_45 = var_43.unsign(var_44)
    var_46 = var_1.unsign(var_3, return_timestamp=var_15)
    var_47 = var_46[var_15]



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test value'
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    assert var_7 == b'test value'
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = var_8.get_timestamp
    var_10 = 10000
    var_11 = var_8.sign(var_2)
    var_12 = 100
    var_13 = var_8.unsign(var_11, var_12)
    var_14 = module_0.TimestampSigner(var_12)
    var_15 = var_14.sign(var_13)
    var_16 = 3600
    var_17 = var_14.unsign(var_15, var_16)
    var_18 = b'bad'
    var_19 = var_3 + var_18
    var_20 = var_1.unsign(var_19)
    var_21 = True
    var_22 = var_1.unsign(var_19, return_timestamp=var_21)
    var_23 = module_1.Signer(var_21)
    var_24 = var_23.sign(var_22)
    var_25 = var_1.unsign(var_24)
    var_26 = b'not-a-number'
    var_27 = module_2.base64_encode(var_26)
    var_28 = b'test value'
    var_29 = b'test bytes'
    var_30 = var_1.sign(var_29)
    var_31 = var_1.unsign(var_30)
    assert var_31 == b'test bytes'
    var_32 = 'test string'
    var_33 = var_1.sign(var_32)
    var_34 = var_1.unsign(var_33)
    assert var_34 == b'test string'



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimestampSigner(var_0, var_2)
    var_4 = 'none'
    var_5 = module_0.TimestampSigner(var_0, key_derivation=var_4)
    var_6 = var_5.get_timestamp()
    var_7 = var_5.timestamp_to_datetime(var_6)
    var_8 = 'test-value'
    var_9 = var_5.sign(var_8)
    var_10 = var_5.unsign(var_9)
    assert var_10 == b'test-value'
    var_11 = True
    var_12 = 3600
    var_13 = var_5.unsign(var_9, var_12)
    assert var_13 == b'test-value'
    var_14 = module_0.TimestampSigner(var_0)
    var_15 = 7200
    var_16 = var_14.sign(var_8)
    var_17 = 3600
    var_18 = var_5.unsign(var_16, var_17)
    var_19 = var_5.validate(var_9)
    assert var_19 is True
    var_20 = var_5.validate(var_9, var_12)
    assert var_20 is True
    var_21 = b'invalid-signature'
    var_22 = var_5.validate(var_21)
    assert var_22 is False
    var_23 = b'bytes-secret'
    var_24 = module_0.TimestampSigner(var_23)
    var_25 = var_24.sign(var_8)
    var_26 = var_24.unsign(var_25)
    assert var_26 == b'test-value'



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = 'other-secret'
    var_8 = module_0.TimedSerializer(var_7)
    var_9 = 'custom-salt'
    var_10 = 'test-key'
    var_11 = module_0.TimedSerializer(var_10)
    var_12 = 'old'
    var_13 = {var_12: var_3}
    var_14 = 0.1
    var_15 = 0
    var_16 = b'invalid-data'
    var_17 = var_1.loads(var_16)
    var_18 = b''



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'
    var_5 = 'test_string'
    var_6 = var_1.sign(var_5)
    var_7 = var_1.unsign(var_6)
    assert var_7 == b'test_string'
    var_8 = 'test'
    var_9 = var_1.sign(var_8)
    var_10 = True
    var_11 = 'fresh'
    var_12 = var_1.sign(var_11)
    var_13 = 3600
    var_14 = var_1.unsign(var_12, var_13)
    assert var_14 == b'fresh'
    var_15 = module_0.TimestampSigner(var_0)
    var_16 = var_15.get_timestamp
    var_17 = 100
    var_18 = 'old'
    var_19 = var_15.sign(var_18)
    var_20 = 10
    var_21 = var_15.unsign(var_19, var_20)
    var_22 = module_0.TimestampSigner(var_20)
    var_23 = 'future'
    var_24 = var_22.sign(var_23)
    var_25 = 3600
    var_26 = var_22.unsign(var_24, var_25)
    var_27 = b'invalid_signature'
    var_28 = var_1.unsign(var_27)
    var_29 = -1
    var_30 = var_3[:var_29]
    var_31 = -1
    var_32 = var_3[var_31:]
    var_33 = b'0'
    var_34 = var_32 == var_33
    var_35 = b'1'
    var_36 = var_35 if var_34 else var_33
    var_37 = var_30 + var_36
    var_38 = var_1.unsign(var_37)
    var_39 = module_1.Signer(var_38)
    var_40 = 'no_timestamp'
    var_41 = var_39.sign(var_40)
    var_42 = var_1.unsign(var_41)
    var_43 = 0
    var_44 = b'.'
    var_45 = signed.rsplit(var_44, var_10)[var_43]
    var_46 = b'.malformed'
    var_47 = var_45 + var_46
    var_48 = var_1.unsign(var_47)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = var_8.get_timestamp
    var_10 = 100
    var_11 = var_8.sign(var_2)
    var_12 = 50
    var_13 = var_8.unsign(var_11, var_12)
    var_14 = module_0.TimestampSigner(var_12)
    var_15 = var_14.sign(var_2)
    var_16 = 3600
    var_17 = var_14.unsign(var_15, var_16)
    var_18 = b'tampered'
    var_19 = var_3 + var_18
    var_20 = var_1.unsign(var_19)
    var_21 = b'.'
    var_22 = var_2 + var_21
    var_23 = var_2 + var_21
    var_24 = b'not_a_number'
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_23 + var_25
    var_27 = var_26 + var_21
    var_28 = var_2 + var_21
    var_29 = module_1.base64_encode(var_24)
    var_30 = var_28 + var_29



