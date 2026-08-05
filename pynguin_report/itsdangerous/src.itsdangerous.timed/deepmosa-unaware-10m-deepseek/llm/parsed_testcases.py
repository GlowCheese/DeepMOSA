####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic initialization.'
    var_1 = module_0.TimedSerializer()
    var_2 = 'test-secret'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = 'test-salt'
    var_5 = module_0.TimedSerializer(var_4)
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}
    var_9 = module_0.TimedSerializer(signer_kwargs=var_8)
    var_10 = 'complex-key'
    var_11 = 'complex-salt'
    var_12 = 'json'
    var_13 = 'digest_method'
    var_14 = 'sha256'
    var_15 = {var_13: var_14}



# Parsed testcases at query #2
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
    var_12 = 1000
    var_13 = var_10.sign(var_2)
    var_14 = 2000
    var_15 = 500
    var_16 = var_10.unsign(var_13, var_15)
    var_17 = module_0.TimestampSigner(var_15)
    var_18 = var_17.sign(var_16)
    var_19 = 3600
    var_20 = var_17.unsign(var_18, var_19)
    var_21 = -1
    var_22 = var_7[:var_21]
    var_23 = b'X'
    var_24 = var_22 + var_23
    var_25 = var_1.unsign(var_24)
    var_26 = b'test_value'
    var_27 = var_26 + var_20
    var_28 = b'invalidsignature'
    var_29 = var_27 + var_28
    var_30 = var_1.unsign(var_29)
    var_31 = -10
    var_32 = var_7[:var_31]
    var_33 = 10
    var_34 = var_23 * var_33
    var_35 = var_32 + var_34
    var_36 = var_1.unsign(var_35)
    var_37 = var_1.sign(var_20)
    var_38 = var_1.unsign(var_37, return_timestamp=var_27)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = 0
    var_41 = var_38[var_40]
    var_42 = var_38[var_27]



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test_value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test_value'
    var_6 = var_2.sign(var_3)
    var_7 = True
    var_8 = var_2.sign(var_3)
    var_9 = 3600
    var_10 = var_2.unsign(var_8, var_9)
    assert var_10 == b'test_value'
    var_11 = var_2.get_timestamp
    var_12 = 100
    var_13 = var_2.sign(var_3)
    var_14 = 50
    var_15 = var_2.unsign(var_13, var_14)
    var_16 = var_2.sign(var_3)
    var_17 = 3600
    var_18 = var_2.unsign(var_16, var_17)
    var_19 = var_2.sign(var_3)
    var_20 = -1
    var_21 = var_19[:var_20]
    var_22 = -1
    var_23 = var_19[var_22:]
    var_24 = b'\x00'
    var_25 = var_23 != var_24
    var_26 = b'\x01'
    var_27 = var_24 if var_25 else var_26
    var_28 = var_21 + var_27
    var_29 = var_2.unsign(var_28)
    var_30 = b'test_value'
    var_31 = b'bad_timestamp'
    var_32 = b'!!!invalid_base64!!!'
    var_33 = var_2.sign(var_30)
    var_34 = var_2.unsign(var_33)
    assert var_34 == b'test_value'
    var_35 = var_2.sign(var_3)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret_key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.signer



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = var_2.sign(var_3)
    var_7 = True
    var_8 = var_2.sign(var_3)
    var_9 = 3600
    var_10 = var_2.unsign(var_8, var_9)
    assert var_10 == b'test-string'
    var_11 = var_2.get_timestamp
    var_12 = 100
    var_13 = var_2.sign(var_3)
    var_14 = 1
    var_15 = var_2.unsign(var_13, var_14)
    var_16 = var_2.sign(var_3)
    var_17 = 3600
    var_18 = var_2.unsign(var_16, var_17)
    var_19 = b'test-value'
    var_20 = b'invalid-timestamp'
    var_21 = b'signature'
    var_22 = var_2.sign(var_3)
    var_23 = 0
    var_24 = b'badsig'
    var_25 = var_2.sign(var_3)
    var_26 = -1
    var_27 = var_25[:var_26]
    var_28 = -1
    var_29 = var_25[var_28:]
    var_30 = b'x'
    var_31 = var_29 != var_30
    var_32 = b'y'
    var_33 = var_30 if var_31 else var_32
    var_34 = var_27 + var_33
    var_35 = var_2.unsign(var_34)
    var_36 = 'test-string'
    var_37 = var_2.sign(var_36)
    var_38 = var_2.sign(var_3)
    var_39 = var_2.sign(var_3)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.signer
    var_3 = b'test-secret-key'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'test-salt'
    var_6 = module_0.TimedSerializer(var_0, var_5)
    var_7 = 'key_derivation'
    var_8 = 'none'
    var_9 = {var_7: var_8}
    var_10 = module_0.TimedSerializer(var_0, signer_kwargs=var_9)
    var_11 = var_1.signer



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimestampSigner constructor and basic functionality.'
    var_1 = 'secret-key'
    var_2 = module_0.TimestampSigner(var_1)



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'my-salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_1, signer_kwargs=var_7)
    var_9 = var_8.iter_unsigners()
    var_10 = list(var_9)
    var_11 = len(var_10)
    var_12 = 'test'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = True
    var_16 = 3600
    var_17 = -1



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.default_signer
    var_3 = 'custom-salt'
    var_4 = 'key_derivation'
    var_5 = 'hmac'
    var_6 = {var_4: var_5}
    var_7 = module_0.TimedSerializer(var_0, var_3, signer_kwargs=var_6)
    var_8 = 'serializer'
    var_9 = 'json'
    var_10 = {var_8: var_9}
    var_11 = module_0.TimedSerializer(var_0, serializer_kwargs=var_10)
    var_12 = var_1.iter_unsigners()
    var_13 = list(var_12)
    var_14 = len(var_13)



# Parsed testcases at query #10
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
    var_8 = 7200
    var_9 = 3600
    var_10 = 'invalid-data'
    var_11 = var_1.loads(var_2)
    var_12 = b''
    var_13 = var_1.loads(var_12)
    var_14 = 'custom-salt'
    var_15 = 'wrong-salt'
    var_16 = 'utf-8'
    var_17 = b'not-even-close-to-valid'
    var_18 = var_1.loads(var_17)



# Parsed testcases at query #11
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
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6[var_5]
    var_9 = 3600
    var_10 = var_1.unsign(var_3, var_9)
    assert var_10 == b'test_value'
    var_11 = 0
    var_12 = var_1.unsign(var_3, var_11)
    var_13 = b'invalid'
    var_14 = var_3 + var_13
    var_15 = var_1.unsign(var_14)
    var_16 = b'invalid_signature'
    var_17 = var_1.unsign(var_16)
    var_18 = 'different-key'
    var_19 = module_0.TimestampSigner(var_18)
    var_20 = var_19.unsign(var_3)
    var_21 = module_0.TimestampSigner(var_20)
    var_22 = var_21.get_timestamp
    var_23 = 100
    var_24 = var_21.sign(var_17)
    var_25 = 3600
    var_26 = var_21.unsign(var_24, var_25)



# Parsed testcases at query #12
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
    var_13 = 50
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = module_0.TimestampSigner(var_13)
    var_16 = var_15.sign(var_14)
    var_17 = 3600
    var_18 = var_1.unsign(var_16, var_17)
    var_19 = b'invalid-data'
    var_20 = var_1.unsign(var_19)
    var_21 = b'test-value'
    var_22 = b'bad-timestamp'
    var_23 = 'different-secret'
    var_24 = module_0.TimestampSigner(var_23)
    var_25 = var_24.sign(var_20)
    var_26 = var_1.unsign(var_25)
    var_27 = b'!!!'
    var_28 = b''
    var_29 = var_1.unsign(var_28)
    var_30 = var_1.unsign(var_28)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp
    var_3 = callable(var_2)
    var_4 = var_1.timestamp_to_datetime
    var_5 = callable(var_4)
    var_6 = var_1.sign
    var_7 = callable(var_6)
    var_8 = var_1.unsign
    var_9 = callable(var_8)
    var_10 = var_1.validate
    var_11 = callable(var_10)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = '|'
    var_5 = 'none'



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret_key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 3600
    var_8 = b'invalid_signature'
    var_9 = var_1.loads(var_8)
    var_10 = 'different_salt'
    var_11 = module_0.TimedSerializer(var_9, var_10)
    var_12 = 'key1'
    var_13 = 'key2'
    var_14 = [var_12, var_13]
    var_15 = module_0.TimedSerializer(var_14)



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test message'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    var_10 = var_1.sign(var_2)
    var_11 = -1
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = var_1.get_timestamp
    var_14 = 100
    var_15 = var_1.sign(var_2)
    var_16 = 50
    var_17 = var_1.unsign(var_15, var_16)
    var_18 = var_1.sign(var_2)
    var_19 = -1
    var_20 = var_18[:var_19]
    var_21 = b'x'
    var_22 = var_20 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = module_1.Signer(var_23)
    var_25 = var_24.sign(var_2)
    var_26 = var_1.unsign(var_25)
    var_27 = b'not-a-timestamp'
    var_28 = var_1.sign(var_2)
    var_29 = var_1.sign(var_2)



# Parsed testcases at query #17
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
    var_6 = var_1.sign(var_2)
    var_7 = 3600
    var_8 = var_1.unsign(var_6, var_7)
    assert var_8 == b'test_value'
    var_9 = module_0.TimestampSigner(var_0)
    var_10 = 100
    var_11 = var_9.sign(var_2)
    var_12 = 10
    var_13 = var_1.unsign(var_11, var_12)
    var_14 = module_0.TimestampSigner(var_12)
    var_15 = var_14.sign(var_13)
    var_16 = 3600
    var_17 = var_1.unsign(var_15, var_16)
    var_18 = b'test_value'
    var_19 = b'invalid_timestamp'
    var_20 = 0
    var_21 = var_1.sign(var_17)
    var_22 = var_1.sign(var_17)
    var_23 = b'tampered'
    var_24 = var_22 + var_23
    var_25 = var_1.unsign(var_24)
    var_26 = b'bytes_value'
    var_27 = var_1.sign(var_26)
    var_28 = var_1.unsign(var_27)
    assert var_28 == b'bytes_value'
    assert var_28 == b'string_value'
    var_29 = 'string_value'
    var_30 = var_1.sign(var_29)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 0.1
    var_8 = 0
    var_9 = b'invalid.signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'custom-salt'
    var_12 = 'wrong-salt'
    var_13 = ''
    var_14 = b'test_bytes'
    var_15 = 42
    var_16 = 2
    var_17 = 3
    var_18 = [var_5, var_16, var_17]



# Parsed testcases at query #19
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
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6[var_5]
    var_9 = 3600
    var_10 = var_1.unsign(var_3, var_9)
    assert var_10 == b'test_value'
    var_11 = 0
    var_12 = var_1.unsign(var_3, var_11)
    var_13 = -1
    var_14 = var_1.unsign(var_3, var_13)
    var_15 = b'invalid_signature'
    var_16 = var_1.unsign(var_15)
    var_17 = -1
    var_18 = var_3[:var_17]
    var_19 = -1
    var_20 = var_3[var_19:]
    var_21 = b'x'
    var_22 = var_20 != var_21
    var_23 = b'y'
    var_24 = var_21 if var_22 else var_23
    var_25 = var_18 + var_24
    var_26 = var_1.unsign(var_25)
    var_27 = 0
    var_28 = var_1.sign(var_16)
    var_29 = b'invalid_timestamp'
    var_30 = module_1.base64_encode(var_29)
    var_31 = b'test_value'
    var_32 = var_1.unsign(var_3, return_timestamp=var_5)
    var_33 = 'test'
    var_34 = var_1.sign(var_33)
    var_35 = b'test_bytes'
    var_36 = var_1.sign(var_35)
    var_37 = var_1.unsign(var_36)
    assert var_37 == b'test_bytes'
    var_38 = 'test_string'
    var_39 = var_1.sign(var_38)
    var_40 = var_1.unsign(var_39)
    assert var_40 == b'test_string'



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'load_payload'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'dump_payload'
    var_4 = hasattr(var_0, var_3)
    var_5 = 'test-secret'
    var_6 = module_0.TimedSerializer(var_5)
    var_7 = 'test-salt'
    var_8 = module_0.TimedSerializer(var_7)
    var_9 = 'key_derivation'
    var_10 = 'hmac'
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(signer_kwargs=var_11)
    var_13 = 'serializer_module'
    var_14 = 'json'
    var_15 = {var_13: var_14}
    var_16 = module_0.TimedSerializer(serializer_kwargs=var_15)



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = 100
    var_10 = var_8.sign(var_2)
    var_11 = 10
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = module_0.TimestampSigner(var_11)
    var_14 = var_13.sign(var_2)
    var_15 = 3600
    var_16 = var_1.unsign(var_14, var_15)
    var_17 = 'wrong-key'
    var_18 = module_0.TimestampSigner(var_17)
    var_19 = var_18.sign(var_2)
    var_20 = var_1.unsign(var_19)
    var_21 = b'malformed'
    var_22 = var_3 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = module_1.Signer(var_23)
    var_25 = var_24.sign(var_2)
    var_26 = var_1.unsign(var_25)
    var_27 = 'utf-8'



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = ':'
    var_5 = 'none'
    var_6 = b'bytes-secret'
    var_7 = module_0.TimestampSigner(var_6)
    var_8 = 'test'
    var_9 = 'fallback1'
    var_10 = 'fallback2'
    var_11 = [var_9, var_10]
    var_12 = module_0.TimestampSigner(var_8)
    var_13 = var_12.fallback_signers
    var_14 = len(var_13)
    assert var_14 == 2



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test-salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_1, signer_kwargs=var_7)
    var_9 = 'load_kwargs'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(var_1, serializer_kwargs=var_11)
    var_13 = module_0.TimedSerializer(var_1)
    var_14 = 'test'
    var_15 = 'data'
    var_16 = {var_14: var_15}



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'my-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'my-salt'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(signer_kwargs=var_7)
    var_9 = 'serializer'
    var_10 = 'json'
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(serializer_kwargs=var_11)
    var_13 = 'test-key'
    var_14 = 'test-salt'
    var_15 = {var_5: var_6}
    var_16 = {var_9: var_10}
    var_17 = module_0.TimedSerializer(var_13, var_14, serializer_kwargs=var_16, signer_kwargs=var_15)



# Parsed testcases at query #26
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
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6[var_5]
    var_9 = 3600
    var_10 = var_1.unsign(var_3, var_9)
    var_11 = var_1.get_timestamp
    var_12 = int(var_0)
    var_13 = 100
    var_14 = var_12 - var_13
    var_15 = var_1.sign(var_2)
    var_16 = int(var_8)
    var_17 = 200
    var_18 = var_16 + var_17
    var_19 = 50
    var_20 = var_1.unsign(var_15, var_19)
    var_21 = b'|invalid_timestamp'
    var_22 = var_3 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = b'invalid_signed_value'
    var_25 = var_1.unsign(var_24)
    var_26 = 'different-key'
    var_27 = module_0.TimestampSigner(var_26)
    var_28 = var_27.unsign(var_3)
    var_29 = b''
    var_30 = var_1.sign(var_29)
    var_31 = var_1.unsign(var_30)
    var_32 = b'line1\nline2\nline3'
    var_33 = var_1.sign(var_32)
    var_34 = var_1.unsign(var_33)
    var_35 = b'value_with_|_separator_and_special_chars!@#$%'
    var_36 = var_1.sign(var_35)
    var_37 = var_1.unsign(var_36)
    var_38 = var_1.get_timestamp
    var_39 = int(var_28)
    var_40 = 1000
    var_41 = var_39 + var_40
    var_42 = var_1.sign(var_2)
    var_43 = int(var_14)
    var_44 = 3600
    var_45 = var_1.unsign(var_42, var_44)
    var_46 = int(var_44)
    var_47 = 100
    var_48 = var_46 - var_47
    var_49 = var_1.sign(var_2)
    var_50 = int(var_43)
    var_51 = 200
    var_52 = var_50 + var_51
    var_53 = 50
    var_54 = True
    var_55 = var_1.unsign(var_49, var_53, var_54)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret_key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 0.1
    var_8 = 0
    var_9 = 'invalid_signature'
    var_10 = var_1.loads(var_9)
    var_11 = ''
    var_12 = var_1.loads(var_11)
    var_13 = 'different_salt'
    var_14 = module_0.TimedSerializer(var_11, var_13)
    var_15 = 'salt1'
    var_16 = module_0.TimedSerializer(var_11, var_15)



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = 0
    var_8 = b'invalid-data'
    var_9 = var_1.loads(var_8)
    var_10 = 'custom-salt'
    var_11 = module_0.TimedSerializer(var_8, var_10)
    var_12 = 'wrong-salt'
    var_13 = 'list'
    var_14 = 'nested'
    var_15 = 'number'
    var_16 = 'boolean'
    var_17 = 'none'
    var_18 = 2
    var_19 = 3
    var_20 = [var_6, var_18, var_19]
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_6, var_22: var_18}
    var_24 = 42
    var_25 = None
    var_26 = {var_13: var_20, var_14: var_23, var_15: var_24, var_16: var_6, var_17: var_25}
    var_27 = {}
    var_28 = 'test'
    var_29 = var_1.default_signer.get_timestamp
    var_30 = 3600



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 0.1
    var_7 = 0
    var_8 = 'custom-salt'
    var_9 = module_0.TimedSerializer(var_7, var_8)
    var_10 = 'wrong-salt'



# Parsed testcases at query #31
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
    var_10 = 100
    var_11 = module_0.TimestampSigner(var_0)
    var_12 = var_11.get_timestamp
    var_13 = var_11.sign(var_2)
    var_14 = 10
    var_15 = var_11.unsign(var_13, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = var_16.sign(var_15)
    var_18 = 3600
    var_19 = var_16.unsign(var_17, var_18)
    var_20 = var_1.sign(var_19)
    var_21 = -1
    var_22 = var_20[:var_21]
    var_23 = b'X'
    var_24 = var_22 + var_23
    var_25 = var_1.unsign(var_24)
    var_26 = module_1.want_bytes(var_19)
    var_27 = var_1.sep
    var_28 = module_1.want_bytes(var_27)
    var_29 = var_26 + var_28
    var_30 = var_26 + var_28
    var_31 = b'not-a-timestamp'
    var_32 = module_1.base64_encode(var_31)
    var_33 = var_30 + var_32
    var_34 = var_33 + var_28
    var_35 = var_26 + var_28
    var_36 = module_1.base64_encode(var_31)
    var_37 = var_35 + var_36
    var_38 = var_26 + var_28
    var_39 = b'!!!invalid'
    var_40 = var_38 + var_39
    var_41 = var_40 + var_28
    var_42 = var_26 + var_28
    var_43 = var_42 + var_39
    var_44 = ''
    var_45 = var_1.sign(var_44)
    var_46 = var_1.unsign(var_45)
    assert var_46 == b''
    var_47 = b'bytes-value'
    var_48 = var_1.sign(var_47)
    var_49 = var_1.unsign(var_48)
    assert var_49 == b'bytes-value'



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = 'test_string'
    var_6 = var_1.sign(var_5)
    var_7 = var_1.unsign(var_6)
    assert var_7 == b'test_string'
    assert var_7 == b'test_timestamp'
    var_8 = 'test_timestamp'
    var_9 = var_1.sign(var_8)
    var_10 = True
    var_11 = 'test_age'
    var_12 = var_1.sign(var_11)
    var_13 = 1000
    var_14 = var_1.unsign(var_12, var_13)
    assert var_14 == b'test_age'
    var_15 = module_0.TimestampSigner(var_0)
    var_16 = 100
    var_17 = 'test_expired'
    var_18 = var_15.sign(var_17)
    var_19 = 10
    var_20 = var_1.unsign(var_18, var_19)
    var_21 = b'value'
    var_22 = b'invalid_timestamp'
    var_23 = b'signature'
    var_24 = -1
    var_25 = var_3[:var_24]
    var_26 = b'x'
    var_27 = var_25 + var_26
    var_28 = var_1.unsign(var_27)



# Parsed testcases at query #33
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
    var_10 = 100
    var_11 = var_1.sign(var_2)
    var_12 = 50
    var_13 = var_1.unsign(var_11, var_12)
    var_14 = var_1.sign(var_2)
    var_15 = 0
    var_16 = b'invalid_timestamp'
    var_17 = var_1.sign(var_2)
    var_18 = var_1.sign(var_2)
    var_19 = bytearray(var_18)
    var_20 = -1
    var_21 = var_19[var_20]
    var_22 = 255
    var_23 = var_21 ^ var_22
    var_24 = bytes(var_19)
    var_25 = var_1.unsign(var_24)



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimestampSigner.unsign method with various scenarios.'
    var_1 = 'test-secret'
    var_2 = module_0.TimestampSigner(var_1)
    var_3 = b'test_value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = True
    var_7 = var_2.unsign(var_4, return_timestamp=var_6)
    var_8 = 'Expected tuple when return_timestamp=True'
    var_9 = len(var_7)
    assert var_9 == 2
    var_10 = var_7[var_6]
    var_11 = 'Expected datetime object'
    var_12 = var_2.sign(var_3)
    var_13 = 3600
    var_14 = var_2.unsign(var_12, var_13)
    var_15 = 3600
    var_16 = 3600
    var_17 = b'invalid_signature'
    var_18 = var_2.unsign(var_17)
    var_19 = var_2.sign(var_3)
    var_20 = -1
    var_21 = var_19[:var_20]
    var_22 = b'X'
    var_23 = var_21 + var_22
    var_24 = var_2.unsign(var_23)
    var_25 = b'value_without_separator'
    var_26 = var_2.unsign(var_25)
    var_27 = b'not_a_timestamp'
    var_28 = 'test_string'
    var_29 = var_2.sign(var_28)
    var_30 = var_2.unsign(var_29)
    var_31 = b'invalid'
    var_32 = var_2.unsign(var_31)
    var_33 = var_2.unsign(var_19, return_timestamp=var_6)
    var_34 = var_33[var_6]



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test-salt'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'json'
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}
    var_9 = 'ensure_ascii'
    var_10 = True
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(var_1, var_3, var_5, var_11, signer_kwargs=var_8)



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = '|'
    var_4 = 'none'
    var_5 = 'sha256'
    var_6 = module_0.TimestampSigner(var_0, var_2, var_3, var_4, var_5)
    var_7 = module_0.TimestampSigner()
    var_8 = var_7.secret_key



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test the constructor of TimedSerializer.'
    var_1 = 'test_secret_key'
    var_2 = module_0.TimedSerializer(var_1)



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret_key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.iter_unsigners()
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = 'custom_salt'
    var_6 = var_1.iter_unsigners(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)



# Parsed testcases at query #39
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
    var_8 = -1
    var_9 = b'invalid-signature'
    var_10 = var_2.loads(var_9)
    var_11 = b'custom-salt'
    var_12 = b'wrong-salt'
    var_13 = 'utf-8'
    var_14 = 2
    var_15 = 3
    var_16 = [var_7, var_14, var_15]
    var_17 = {}
    var_18 = None



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
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
    var_12 = var_10.sign(var_2)
    var_13 = 2000
    var_14 = 500
    var_15 = var_10.unsign(var_12, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = 1000
    var_18 = var_16.sign(var_15)
    var_19 = 500
    var_20 = 3600
    var_21 = var_16.unsign(var_18, var_20)
    var_22 = var_1.sign(var_21)
    var_23 = 0
    var_24 = b'invalid-timestamp'
    var_25 = var_1.sign(var_21)
    var_26 = 2
    var_27 = 0
    var_28 = 2
    var_29 = var_6 + var_8
    var_30 = var_1.unsign(var_29)
    var_31 = str(var_13)
    var_32 = var_1.sign(var_21)
    var_33 = -1
    var_34 = var_32[:var_33]
    var_35 = b'X'
    var_36 = var_34 + var_35
    var_37 = var_1.unsign(var_36)
    var_38 = ''
    var_39 = var_1.sign(var_38)
    var_40 = var_1.unsign(var_39)
    assert var_40 == b''
    var_41 = b'bytes-value'
    var_42 = var_1.sign(var_41)
    var_43 = var_1.unsign(var_42)
    assert var_43 == b'bytes-value'
    var_44 = 'value with spaces and !@#$%'
    var_45 = var_1.sign(var_44)
    var_46 = var_1.unsign(var_45)
    assert var_46 == b'value with spaces and !@#$%'



# Parsed testcases at query #41
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
    var_12 = var_10.sign(var_2)
    var_13 = 50
    var_14 = var_10.unsign(var_12, var_13)
    var_15 = var_10.sign(var_14)
    var_16 = 50
    var_17 = var_10.unsign(var_15, var_16)
    var_18 = b'invalid_signature'
    var_19 = var_1.unsign(var_18)
    var_20 = var_1.sign(var_19)
    var_21 = -1
    var_22 = var_20[:var_21]
    var_23 = -1
    var_24 = var_20[var_23:]
    var_25 = b'0'
    var_26 = var_24 == var_25
    var_27 = b'1'
    var_28 = var_27 if var_26 else var_25
    var_29 = var_22 + var_28
    var_30 = var_1.unsign(var_29)
    var_31 = b'test_value'
    var_32 = var_1.unsign(var_31)
    var_33 = b'not_a_timestamp'
    var_34 = b'test_value'
    var_35 = var_1.sign(var_32)
    var_36 = False
    var_37 = var_1.unsign(var_35, return_timestamp=var_36)
    var_38 = var_1.sign(var_32)
    var_39 = var_1.unsign(var_38, return_timestamp=var_6)
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = var_39[var_36]
    var_42 = var_39[var_6]



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'custom-key'
    var_2 = 'custom-salt'
    var_3 = 'json'
    var_4 = 'key_derivation'
    var_5 = 'hmac'
    var_6 = {var_4: var_5}
    var_7 = module_0.TimedSerializer(var_1, var_2, var_3, signer_kwargs=var_6)
    var_8 = var_7.iter_unsigners()
    var_9 = list(var_8)
    var_10 = len(var_9)
    var_11 = 'test'
    var_12 = 'data'
    var_13 = {var_11: var_12}
    var_14 = True



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 3600
    var_7 = True
    var_8 = b'invalid-data'
    var_9 = var_2.loads(var_8)
    var_10 = module_0.TimedSerializer(var_9)
    var_11 = {var_3: var_4}
    var_12 = 1.5
    var_13 = 1
    var_14 = 'custom-salt'
    var_15 = 'wrong-salt'
    var_16 = -1



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #45
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test-value'
    var_6 = True
    var_7 = 3600
    var_8 = var_2.unsign(var_4, var_7)
    assert var_8 == b'test-value'
    var_9 = var_2.get_timestamp
    var_10 = 10000
    var_11 = 1
    var_12 = var_2.unsign(var_4, var_11)
    var_13 = 'future-value'
    var_14 = var_2.sign(var_13)
    var_15 = -1
    var_16 = var_2.unsign(var_14, var_15)
    var_17 = b'invalid'
    var_18 = var_4 + var_17
    var_19 = var_2.unsign(var_18)
    var_20 = 0
    var_21 = var_2.sign(var_3)
    var_22 = -1
    var_23 = var_4[:var_22]
    var_24 = b'x'
    var_25 = var_23 + var_24
    var_26 = var_2.unsign(var_25)
    var_27 = b'bytes-value'
    var_28 = var_2.sign(var_27)
    var_29 = var_2.unsign(var_28)
    assert var_29 == b'bytes-value'
    assert var_29 == b'string-value'
    var_30 = 'string-value'
    var_31 = var_2.sign(var_30)



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test-value'
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 3600
    var_9 = var_1.unsign(var_3, var_8)
    assert var_9 == b'test-value'
    var_10 = 50
    var_11 = 200
    var_12 = b'test-value'
    var_13 = b'invalid-timestamp'
    var_14 = b'signature'
    var_15 = var_1.sign(var_2)
    var_16 = 0
    var_17 = -1
    var_18 = b'invalid-signature'
    var_19 = var_1.unsign(var_18)
    var_20 = 'different-key'
    var_21 = module_0.TimestampSigner(var_20)
    var_22 = var_21.sign(var_19)
    var_23 = var_1.unsign(var_22)
    var_24 = var_1.unsign(var_3, return_timestamp=var_5)



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret_key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = 'hello world'
    var_8 = b'bytes data'
    var_9 = 42
    var_10 = 2
    var_11 = 3
    var_12 = [var_6, var_10, var_11]
    var_13 = None
    var_14 = 'custom_salt'
    var_15 = module_0.TimedSerializer(var_0, var_14)
    var_16 = ''
    var_17 = 'data with special chars: !@#$%^&*()'
    var_18 = '数据 with unicode: 你好世界'
    var_19 = 'level1'
    var_20 = 'level2'
    var_21 = 'level3'
    var_22 = {var_21: var_3}
    var_23 = [var_6, var_10, var_22]
    var_24 = {var_20: var_23}
    var_25 = {var_19: var_24}
    var_26 = module_0.TimedSerializer(var_0)
    var_27 = module_0.TimedSerializer(var_0)
    var_28 = 10
    var_29 = b'invalid_data'
    var_30 = var_1.loads(var_29)
    var_31 = -1
    var_32 = var_1.loads(var_29)



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

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
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = var_8.get_timestamp
    var_10 = 0
    var_11 = 'old_value'
    var_12 = var_8.sign(var_11)
    var_13 = 1
    var_14 = var_8.unsign(var_12, var_13)
    var_15 = module_0.TimestampSigner(var_13)
    var_16 = 10000
    var_17 = 'future_value'
    var_18 = var_15.sign(var_17)
    var_19 = 3600
    var_20 = var_15.unsign(var_18, var_19)
    var_21 = b'invalid_signature'
    var_22 = var_1.unsign(var_21)
    var_23 = 10
    var_24 = var_3[:var_23]
    var_25 = b'X'
    var_26 = var_24 + var_25
    var_27 = 11
    var_28 = var_3[var_27:]
    var_29 = var_26 + var_28
    var_30 = var_1.unsign(var_29)
    var_31 = module_1.Signer(var_30)
    var_32 = 'no_timestamp'
    var_33 = var_31.sign(var_32)
    var_34 = var_1.unsign(var_33)
    var_35 = b'test_value'
    var_36 = b'invalid_base64'
    var_37 = b'bytes_value'
    var_38 = var_1.sign(var_37)
    var_39 = var_1.unsign(var_38)
    assert var_39 == b'bytes_value'
    assert var_39 == b'string_value'
    var_40 = 'string_value'
    var_41 = var_1.sign(var_40)



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = True
    var_7 = 3600
    var_8 = var_2.unsign(var_4, var_7)
    var_9 = 100
    var_10 = 1
    var_11 = var_2.unsign(var_4, var_10)
    var_12 = 3600
    var_13 = var_2.unsign(var_4, var_12)
    var_14 = b'x'
    var_15 = var_4 + var_14
    var_16 = var_2.unsign(var_15)
    var_17 = 0
    var_18 = var_2.sign(var_3)
    var_19 = -1
    var_20 = var_4[:var_19]
    var_21 = var_20 + var_14
    var_22 = var_2.unsign(var_21)



# Parsed testcases at query #50
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
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    assert var_9 == b'test_value'
    var_10 = var_1.get_timestamp
    var_11 = 100
    var_12 = var_1.sign(var_2)
    var_13 = 10
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = var_1.sign(var_14)
    var_16 = 50
    var_17 = 60
    var_18 = var_1.unsign(var_15, var_17)
    var_19 = b'invalid_signature'
    var_20 = var_1.unsign(var_19)
    var_21 = var_1.sign(var_20)
    var_22 = -1
    var_23 = var_21[:var_22]
    var_24 = b'X'
    var_25 = var_23 + var_24
    var_26 = var_1.unsign(var_25)
    var_27 = b'test_value'
    var_28 = var_27 + var_20
    var_29 = b'invalidsig'
    var_30 = var_28 + var_29
    var_31 = var_1.unsign(var_30)
    var_32 = b'not_a_timestamp'
    var_33 = module_1.base64_encode(var_32)
    var_34 = b'test_value'
    var_35 = b'signature'
    var_36 = b'bytes_value'
    var_37 = var_1.sign(var_36)
    var_38 = var_1.unsign(var_37)
    assert var_38 == b'bytes_value'
    assert var_38 == b'string_value'
    var_39 = 'string_value'
    var_40 = var_1.sign(var_39)



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)



# Parsed testcases at query #52
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 3600
    var_7 = True
    var_8 = 'custom-salt'
    var_9 = module_0.TimedSerializer(var_1)
    var_10 = module_0.TimestampSigner(var_1)
    var_11 = 10000
    var_12 = b'test'
    var_13 = var_10.sign(var_12)
    var_14 = 10
    var_15 = var_2.loads(var_13, var_14)
    var_16 = b'invalid-signature'
    var_17 = var_2.loads(var_16)
    var_18 = 'test'
    var_19 = {}
    var_20 = None
    var_21 = 2
    var_22 = 3
    var_23 = [var_7, var_21, var_22]



# Parsed testcases at query #53
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = module_0.TimestampSigner(var_0)
    var_8 = 100
    var_9 = str(var_4)
    var_10 = module_1.want_bytes(var_9)
    var_11 = var_7.sep
    var_12 = module_1.want_bytes(var_11)
    var_13 = var_10 + var_12
    var_14 = 10
    var_15 = b'invalid-data'
    var_16 = var_1.loads(var_15)
    var_17 = 'custom-salt'
    var_18 = 'wrong-salt'
    var_19 = 'utf-8'



# Parsed testcases at query #54
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
    var_9 = var_0.iter_unsigners()
    var_10 = list(var_9)
    var_11 = len(var_10)
    var_12 = 'test'
    var_13 = 'data'
    var_14 = {var_12: var_13}
    var_15 = True



# Parsed testcases at query #55
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
    var_10 = 100
    var_11 = 'old_value'
    var_12 = var_1.sign(var_11)
    var_13 = 10
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = b'invalid_data'
    var_16 = var_1.unsign(var_15)
    var_17 = b'value|invalid_timestamp|signature'
    var_18 = var_1.unsign(var_17)
    var_19 = b'value|signature'
    var_20 = var_1.unsign(var_19)
    var_21 = 'future_value'
    var_22 = var_1.sign(var_21)
    var_23 = 3600
    var_24 = var_1.unsign(var_22, var_23)



# Parsed testcases at query #56
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
    var_7 = 0.1
    var_8 = 0.001
    var_9 = 'custom-salt'
    var_10 = b'invalid-data'
    var_11 = var_1.loads(var_10)
    var_12 = 'salt1'
    var_13 = 'wrong-salt'
    var_14 = 0.001
    var_15 = module_0.TimedSerializer(var_14)
    var_16 = 'salt2'
    var_17 = 'wrong-salt'



# Parsed testcases at query #57
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()



# Parsed testcases at query #58
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test value'
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    assert var_9 == b'test value'
    assert var_9 == b'test string'
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = var_10.get_timestamp
    var_12 = var_10.sign(var_2)
    var_13 = 50
    var_14 = var_10.unsign(var_12, var_13)
    var_15 = var_10.sign(var_14)
    var_16 = 3600
    var_17 = var_10.unsign(var_15, var_16)
    var_18 = b'invalid_timestamp'
    var_19 = 0
    var_20 = var_1.sign(var_17)
    var_21 = 2
    var_22 = b'bad_signature'
    var_23 = 'test string'
    var_24 = var_1.sign(var_23)



# Parsed testcases at query #59
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = '|'
    var_4 = 'none'
    var_5 = 'sha256'



# Parsed testcases at query #60
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = 'hmac'
    var_4 = module_0.TimestampSigner(var_0, var_2, key_derivation=var_3)
    var_5 = 'sha256'
    var_6 = module_0.TimestampSigner(var_0, digest_method=var_5)
    var_7 = '.'
    var_8 = 'hmac-sha256'
    var_9 = module_0.TimestampSigner(var_0, var_2, var_7, var_3, var_5, var_8)



# Parsed testcases at query #61
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test TimedSerializer.loads with various scenarios.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 3600
    var_8 = 7200
    var_9 = var_0 - var_8
    var_10 = 3600
    var_11 = 'invalid-signature'
    var_12 = var_2.loads(var_11)
    var_13 = 'custom-salt'
    var_14 = 'wrong-salt'
    var_15 = 'utf-8'
    var_16 = ''
    var_17 = 'test-string'
    var_18 = 42
    var_19 = 2
    var_20 = 3
    var_21 = [var_6, var_19, var_20]
    var_22 = 'outer'
    var_23 = 'inner'
    var_24 = 'numbers'
    var_25 = [var_6, var_19]
    var_26 = {var_23: var_4, var_24: var_25}
    var_27 = {var_22: var_26}
    var_28 = -1
    var_29 = 0
    var_30 = b'\x00\x01\x02'
    var_31 = 'test-secret-key-2'
    var_32 = module_0.TimedSerializer(var_31)
    var_33 = module_0.TimedSerializer(var_12)
    var_34 = 'fallback_signers'
    var_35 = [var_32]
    var_36 = 7200
    var_37 = var_29 - var_36
    var_38 = 3600
    var_39 = True
    var_40 = 'test'
    var_41 = module_1.want_bytes(var_40)
    var_42 = var_2.signer.sep
    var_43 = module_1.want_bytes(var_42)
    var_44 = var_41 + var_43
    var_45 = b'invalid-timestamp'
    var_46 = var_44 + var_45
    var_47 = var_46 + var_43
    var_48 = b'fake-signature'
    var_49 = var_47 + var_48



# Parsed testcases at query #62
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret_key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = 50
    var_9 = b'invalid_data'
    var_10 = var_1.loads(var_9)
    var_11 = 'custom_salt'
    var_12 = 'wrong_salt'
    var_13 = {}
    var_14 = 2
    var_15 = 3
    var_16 = [var_5, var_14, var_15]
    var_17 = 'a'
    var_18 = 'b'
    var_19 = [var_5, var_14]
    var_20 = 'c'
    var_21 = 'test'
    var_22 = {var_20: var_21}
    var_23 = {var_17: var_19, var_18: var_22}



# Parsed testcases at query #63
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
    var_7 = 7200
    var_8 = var_0 - var_7
    var_9 = 3600
    var_10 = 'invalid-data'
    var_11 = var_1.loads(var_10)
    var_12 = 'different-salt'
    var_13 = 'wrong-salt'
    var_14 = ''
    var_15 = var_1.loads(var_14)
    var_16 = 'second-secret'
    var_17 = module_0.TimedSerializer(var_16)



# Parsed testcases at query #64
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
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    assert var_9 == b'test_value'
    var_10 = var_1.get_timestamp
    var_11 = 100
    var_12 = var_1.sign(var_2)
    var_13 = 10
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = var_1.sign(var_14)
    var_16 = 200
    var_17 = var_1.unsign(var_15, var_16)
    var_18 = 'different-secret'
    var_19 = module_0.TimestampSigner(var_18)
    var_20 = var_1.sign(var_17)
    var_21 = var_19.unsign(var_20)
    var_22 = module_0.TimestampSigner(var_21)
    var_23 = module_1.want_bytes(var_17)
    var_24 = var_22.sep
    var_25 = module_1.want_bytes(var_24)
    var_26 = var_22.get_timestamp()
    var_27 = module_1.int_to_bytes(var_26)
    var_28 = module_1.base64_encode(var_27)
    var_29 = var_23 + var_25
    var_30 = var_29 + var_28
    var_31 = var_30 + var_25
    var_32 = b'invalid!'
    var_33 = var_23 + var_25
    var_34 = var_33 + var_32
    var_35 = var_34 + var_25
    var_36 = var_1.sign(var_17)
    var_37 = 0
    var_38 = var_1.unsign(var_36)
    var_39 = b'bytes_value'
    var_40 = var_1.sign(var_39)
    var_41 = var_1.unsign(var_40)
    assert var_41 == b'bytes_value'
    assert var_41 == b'string_value'
    var_42 = 'string_value'
    var_43 = var_1.sign(var_42)



# Parsed testcases at query #65
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = module_0.TimestampSigner(var_0)
    var_3 = ':'
    var_4 = module_0.TimestampSigner(var_0, sep=var_3)
    var_5 = 'custom-salt'
    var_6 = module_0.TimestampSigner(var_0, var_5)
    var_7 = 'none'
    var_8 = module_0.TimestampSigner(var_0, key_derivation=var_7)



# Parsed testcases at query #66
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'custom-salt'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = ':'
    var_6 = module_0.TimedSerializer()
    var_7 = 'my-secret'
    var_8 = 'my-salt'
    var_9 = '|'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = module_0.TimedSerializer(var_7, var_8, serializer_kwargs=var_12)



# Parsed testcases at query #67
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor creates instance with correct defaults.'
    var_1 = module_0.TimedSerializer()



# Parsed testcases at query #68
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = '|'
    var_5 = 'none'
    var_6 = var_1.get_timestamp()
    var_7 = var_1.timestamp_to_datetime(var_6)
    var_8 = 'test-value'
    var_9 = var_1.sign(var_8)
    var_10 = var_1.unsign(var_9)
    assert var_10 == b'test-value'
    var_11 = True
    var_12 = var_1.validate(var_9)
    assert var_12 is True
    var_13 = b'invalid-signature'
    var_14 = var_1.validate(var_13)
    assert var_14 is False
    var_15 = b'invalid-signature'
    var_16 = var_1.unsign(var_15)
    var_17 = 0
    var_18 = var_1.unsign(var_9, var_17)
    var_19 = -1
    var_20 = var_1.unsign(var_9, var_19)



# Parsed testcases at query #69
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
    var_8 = 100
    var_9 = 10
    var_10 = b'invalid-data'
    var_11 = var_1.loads(var_10)
    var_12 = ''
    var_13 = var_1.loads(var_12)
    var_14 = 'salt1'
    var_15 = 'wrong-salt'



# Parsed testcases at query #70
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)



# Parsed testcases at query #71
#--------------------------


import src.itsdangerous.timed as module_0

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
    var_11 = var_2.sign(var_3)
    var_12 = -1
    var_13 = var_2.unsign(var_11, var_12)
    var_14 = b'invalid-data'
    var_15 = b'fake-signature'
    var_16 = b'test-value'
    var_17 = b'not-a-timestamp'
    var_18 = b'signature'
    var_19 = var_2.sign(var_3)
    var_20 = 0
    var_21 = var_2.unsign(var_19)
    var_22 = module_0.TimestampSigner(var_21, var_13)
    var_23 = var_22.get_timestamp
    var_24 = 10000
    var_25 = var_22.sign(var_3)
    var_26 = 3600
    var_27 = var_2.unsign(var_25, var_26)



# Parsed testcases at query #72
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test_value'
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6[var_5]
    var_9 = 3600
    var_10 = var_1.unsign(var_3, var_9)
    assert var_10 == b'test_value'
    var_11 = module_0.TimestampSigner(var_0)
    var_12 = 100
    var_13 = var_11.sign(var_2)
    var_14 = 50
    var_15 = var_1.unsign(var_13, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = var_16.sign(var_15)
    var_18 = 3600
    var_19 = var_1.unsign(var_17, var_18)
    var_20 = b'invalid_signature'
    var_21 = var_1.unsign(var_20)
    var_22 = 10
    var_23 = var_3[:var_22]
    var_24 = b'X'
    var_25 = var_23 + var_24
    var_26 = 11
    var_27 = var_3[var_26:]
    var_28 = var_25 + var_27
    var_29 = var_1.unsign(var_28)
    var_30 = b'.'
    var_31 = b''
    var_32 = b'test_value.'
    var_33 = b'not_a_timestamp'
    var_34 = var_32 + var_33
    var_35 = var_34 + var_30
    var_36 = 'different-salt'
    var_37 = module_0.TimestampSigner(var_29, var_36)
    var_38 = var_37.sign(var_21)
    var_39 = var_1.unsign(var_38)
    var_40 = b'bytes_value'
    var_41 = var_1.sign(var_40)
    var_42 = var_1.unsign(var_41)
    assert var_42 == b'bytes_value'
    var_43 = 'string_value'
    var_44 = var_1.sign(var_43)
    var_45 = var_1.unsign(var_44)
    assert var_45 == b'string_value'
    var_46 = var_1.unsign(var_3, return_timestamp=var_5)
    var_47 = b'complex_value_123'
    var_48 = var_1.sign(var_47)
    var_49 = var_1.unsign(var_48)
    var_50 = var_1.sign(var_31)
    var_51 = var_1.unsign(var_50)
    assert var_51 == b''



# Parsed testcases at query #73
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = '|'
    var_4 = 'custom-salt'
    var_5 = 'none'
    var_6 = b'bytes-secret'
    var_7 = module_0.TimestampSigner(var_6)
    var_8 = var_1.get_timestamp()
    var_9 = var_1.timestamp_to_datetime(var_8)
    var_10 = 'test-value'
    var_11 = var_1.sign(var_10)
    var_12 = var_1.unsign(var_11)
    var_13 = True
    var_14 = var_1.validate(var_11)
    assert var_14 is True
    var_15 = b'invalid'
    var_16 = var_1.validate(var_15)
    assert var_16 is False
    var_17 = var_1.sign(var_10)
    var_18 = 3600
    var_19 = var_1.validate(var_17, var_18)
    assert var_19 is True
    var_20 = 10000
    var_21 = b'.'
    var_22 = 3600
    var_23 = b'invalid-data'
    var_24 = var_1.unsign(var_23)
    var_25 = b'not-a-timestamp'



# Parsed testcases at query #74
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
    var_8 = var_1.get_timestamp
    var_9 = 7200
    var_10 = 3600
    var_11 = var_1.unsign(var_3, var_10)
    var_12 = 0
    var_13 = 3600
    var_14 = var_1.unsign(var_3, var_13)
    var_15 = -1
    var_16 = var_3[:var_15]
    var_17 = b'x'
    var_18 = var_16 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = var_1.sign(var_14)
    var_21 = var_1.sign(var_14)
    var_22 = b'invalid_timestamp'
    var_23 = var_1.unsign(var_21)



# Parsed testcases at query #75
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = var_2.get_timestamp
    var_4 = callable(var_3)
    var_5 = module_0.TimestampSigner(var_0)
    var_6 = var_2.get_timestamp()



# Parsed testcases at query #76
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = '|'
    var_4 = module_0.TimestampSigner(var_0, var_2, var_3)
    var_5 = b'secret-key'
    var_6 = module_0.TimestampSigner(var_5)



# Parsed testcases at query #77
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #78
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = 'custom-salt'
    var_8 = module_0.TimedSerializer(var_0, var_7)
    var_9 = 'test-key'
    var_10 = module_0.TimedSerializer(var_9)
    var_11 = 100
    var_12 = str(var_4)
    var_13 = module_1.want_bytes(var_12)
    var_14 = 50
    var_15 = b'tampered'
    var_16 = str(var_4)
    var_17 = module_1.want_bytes(var_16)
    var_18 = b'invalid-timestamp'
    var_19 = b''
    var_20 = var_1.loads(var_19)
    var_21 = None
    var_22 = var_1.loads(var_21)



# Parsed testcases at query #79
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
    var_7 = 'different-salt'
    var_8 = module_0.TimedSerializer(var_0, var_7)
    var_9 = module_0.TimestampSigner(var_0)
    var_10 = 100
    var_11 = 50
    var_12 = b'invalid-data'
    var_13 = var_1.loads(var_12)
    var_14 = 'utf-8'
    var_15 = b'data'
    var_16 = b'timestamp'
    var_17 = b'wrong-signature'
    var_18 = b'signature'
    var_19 = 'key1'
    var_20 = 'key2'
    var_21 = [var_19, var_20]
    var_22 = module_0.TimedSerializer(var_21)
    var_23 = 'test'



# Parsed testcases at query #80
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = 100
    var_9 = module_0.TimedSerializer(var_0)
    var_10 = 10
    var_11 = b'invalid|data|signature'
    var_12 = var_1.loads(var_11)
    var_13 = b''
    var_14 = var_1.loads(var_13)
    var_15 = 'custom-salt'
    var_16 = module_0.TimedSerializer(var_13, var_15)
    var_17 = 'wrong-salt'
    var_18 = 'utf-8'
    var_19 = -1



# Parsed testcases at query #81
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = module_0.TimestampSigner(var_0)
    var_8 = '{"key": "value"}'
    var_9 = module_1.want_bytes(var_8)
    var_10 = 100
    var_11 = var_7.sep
    var_12 = module_1.want_bytes(var_11)
    var_13 = var_9 + var_12
    var_14 = 10
    var_15 = 'invalid_data'
    var_16 = var_1.loads(var_15)
    var_17 = ''
    var_18 = var_1.loads(var_17)
    var_19 = 'custom_salt'
    var_20 = module_0.TimedSerializer(var_17, var_19)
    var_21 = 'wrong_salt'



# Parsed testcases at query #82
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #83
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
    var_7 = 0.1
    var_8 = 0
    var_9 = b'invalid-data'
    var_10 = var_1.loads(var_9)
    var_11 = 'custom-salt'
    var_12 = module_0.TimedSerializer(var_9, var_11)
    var_13 = 'wrong-salt'



# Parsed testcases at query #84
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic attributes.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'custom-salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)



# Parsed testcases at query #85
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test_value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test_value'
    var_6 = var_2.sign(var_3)
    var_7 = True
    var_8 = var_2.sign(var_3)
    var_9 = 3600
    var_10 = var_2.unsign(var_8, var_9)
    assert var_10 == b'test_value'
    var_11 = module_0.TimestampSigner(var_0, var_1)
    var_12 = var_11.get_timestamp
    var_13 = var_11.sign(var_3)
    var_14 = 1
    var_15 = var_11.unsign(var_13, var_14)
    var_16 = var_2.sign(var_3)
    var_17 = b'.'
    var_18 = 0
    var_19 = b'invalid'
    var_20 = module_1.base64_encode(var_19)
    var_21 = var_2.sign(var_3)
    var_22 = -1
    var_23 = var_21[:var_22]
    var_24 = b'x'
    var_25 = var_23 + var_24
    var_26 = var_2.unsign(var_25)
    var_27 = ''
    var_28 = var_2.sign(var_27)
    var_29 = var_2.unsign(var_28)
    assert var_29 == b''
    var_30 = b'test_value'
    var_31 = var_2.sign(var_30)
    var_32 = var_2.unsign(var_31)
    assert var_32 == b'test_value'
    var_33 = var_2.sign(var_3)
    var_34 = module_0.TimestampSigner(var_26, var_15)
    var_35 = var_34.get_timestamp
    var_36 = var_34.sign(var_3)
    var_37 = 3600
    var_38 = var_34.unsign(var_36, var_37)



# Parsed testcases at query #86
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test_value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test_value'
    var_6 = var_2.sign(var_3)
    var_7 = True
    var_8 = var_2.sign(var_3)
    var_9 = 3600
    var_10 = var_2.unsign(var_8, var_9)
    assert var_10 == b'test_value'
    var_11 = 50
    var_12 = 50
    var_13 = b'invalid_signature'
    var_14 = var_2.unsign(var_13)
    var_15 = var_2.sign(var_3)
    var_16 = 0
    var_17 = b'invalid_timestamp'
    var_18 = var_2.sign(var_3)
    var_19 = var_2.sign(var_3)
    var_20 = var_2.sign(var_3)



# Parsed testcases at query #87
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor creates instance with correct default signer.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)



# Parsed testcases at query #88
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
    var_11 = 100
    var_12 = var_10.sign(var_2)
    var_13 = 50
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = module_0.TimestampSigner(var_13)
    var_16 = var_15.sign(var_14)
    var_17 = 3600
    var_18 = var_1.unsign(var_16, var_17)
    var_19 = b'invalid_signature'
    var_20 = var_1.unsign(var_19)
    var_21 = var_1.sign(var_20)
    var_22 = -1
    var_23 = var_21[:var_22]
    var_24 = b'X'
    var_25 = var_23 + var_24
    var_26 = var_1.unsign(var_25)
    var_27 = b'test_value'
    var_28 = b'fake_signature'
    var_29 = b'invalid_timestamp'
    var_30 = b'signature'



# Parsed testcases at query #89
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.signer
    var_3 = 'key'
    var_4 = 'number'
    var_5 = 'value'
    var_6 = 42
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 3600
    var_9 = True
    var_10 = -1
    var_11 = 'invalid-data'
    var_12 = var_1.loads(var_11)
    var_13 = -1
    var_14 = None
    var_15 = 'custom-salt'
    var_16 = 'wrong-salt'



# Parsed testcases at query #90
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = ':'
    var_4 = 'none'
    var_5 = b'bytes-secret-key'
    var_6 = module_0.TimestampSigner(var_5)



# Parsed testcases at query #91
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
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = var_8.get_timestamp
    var_10 = 100
    var_11 = var_8.sign(var_2)
    var_12 = 50
    var_13 = var_8.unsign(var_11, var_12)
    var_14 = module_0.TimestampSigner(var_12)
    var_15 = var_14.sign(var_13)
    var_16 = 3600
    var_17 = var_14.unsign(var_15, var_16)
    var_18 = b'test_value'
    var_19 = b'invalid_timestamp'
    var_20 = 0
    var_21 = var_1.sign(var_17)
    var_22 = b'wrong_value'



# Parsed testcases at query #92
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
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = 'Should return tuple when return_timestamp=True'
    var_8 = len(var_6)
    assert var_8 == 2
    var_9 = var_6[var_5]
    var_10 = 'Second element should be datetime'
    var_11 = 3600
    var_12 = var_1.unsign(var_3, var_11)
    assert var_12 == b'test_string'
    var_13 = var_1.get_timestamp
    var_14 = 100
    var_15 = var_1.sign(var_2)
    var_16 = 50
    var_17 = var_1.unsign(var_15, var_16)
    var_18 = var_1.sign(var_2)
    var_19 = 3600
    var_20 = var_1.unsign(var_18, var_19)
    var_21 = b'malformed'
    var_22 = var_3 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = module_1.want_bytes(var_2)
    var_25 = var_1.sep
    var_26 = module_1.want_bytes(var_25)
    var_27 = var_24 + var_26
    var_28 = b'bad_value'
    var_29 = var_28 + var_26
    var_30 = module_1.int_to_bytes(var_14)
    var_31 = module_1.base64_encode(var_30)
    var_32 = var_29 + var_31
    var_33 = var_32 + var_26
    var_34 = b'bad_signature'
    var_35 = var_33 + var_34
    var_36 = var_1.unsign(var_35)
    var_37 = 'test_string'
    var_38 = var_1.sign(var_37)
    var_39 = var_1.unsign(var_3, var_11, var_20)
    var_40 = var_39[var_20]
    var_41 = '|'
    var_42 = module_0.TimestampSigner(var_36, sep=var_41)
    var_43 = b'custom_sep_value'
    var_44 = var_42.sign(var_43)
    var_45 = var_42.unsign(var_44)
    var_46 = var_1.unsign(var_3, return_timestamp=var_20)
    var_47 = b''
    var_48 = var_1.sign(var_47)
    var_49 = var_1.unsign(var_48)



# Parsed testcases at query #93
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

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
    var_8 = 50
    var_9 = 50
    var_10 = b'invalid_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = -1
    var_13 = var_3[:var_12]
    var_14 = -1
    var_15 = var_3[var_14:]
    var_16 = b'y'
    var_17 = var_15 == var_16
    var_18 = b'x'
    var_19 = var_18 if var_17 else var_16
    var_20 = var_13 + var_19
    var_21 = var_1.unsign(var_20)
    var_22 = module_1.Signer(var_21)
    var_23 = var_22.sign(var_11)
    var_24 = var_1.unsign(var_23)
    var_25 = b'test_value'
    var_26 = b'invalid_timestamp'
    var_27 = b'test bytes'
    var_28 = var_1.sign(var_27)
    var_29 = var_1.unsign(var_28)
    assert var_29 == b'test bytes'
    assert var_29 == b'test string'
    var_30 = 'test string'
    var_31 = var_1.sign(var_30)



# Parsed testcases at query #94
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.signer
    var_3 = 'custom-salt'
    var_4 = module_0.TimedSerializer(var_0, var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_0, signer_kwargs=var_7)



# Parsed testcases at query #95
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
    var_10 = var_1.sign(var_2)
    var_11 = -1
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = var_1.sign(var_12)
    var_14 = -1
    var_15 = var_13[:var_14]
    var_16 = -1
    var_17 = var_13[var_16:]
    var_18 = b'x'
    var_19 = var_17 != var_18
    var_20 = b'y'
    var_21 = var_18 if var_19 else var_20
    var_22 = var_15 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = b'no_separator_here'
    var_25 = var_1.unsign(var_24)
    var_26 = var_1.sign(var_25)
    var_27 = 0
    var_28 = b'invalid_base64!!'
    var_29 = var_1.get_timestamp
    var_30 = 1000
    var_31 = var_1.sign(var_25)
    var_32 = 3600
    var_33 = var_1.unsign(var_31, var_32)
    var_34 = 1234567890
    var_35 = var_1.timestamp_to_datetime(var_34)



# Parsed testcases at query #96
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'secret'
    var_3 = 'custom-salt'
    var_4 = '-'
    var_5 = 'none'
    var_6 = module_0.TimestampSigner(var_2, var_3, var_4, var_5)
    var_7 = var_6.get_timestamp()
    var_8 = var_6.timestamp_to_datetime(var_7)
    var_9 = 'test-value'
    var_10 = var_6.sign(var_9)
    var_11 = var_6.unsign(var_10)
    assert var_11 == b'test-value'
    var_12 = True
    var_13 = var_6.unsign(var_10, return_timestamp=var_12)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = var_13[var_12]
    var_16 = var_6.validate(var_10)
    assert var_16 is True
    var_17 = b'invalid-signature'
    var_18 = var_6.validate(var_17)
    assert var_18 is False
    var_19 = b'invalid-signature'
    var_20 = var_6.unsign(var_19)
    var_21 = 'test'
    var_22 = var_6.sign(var_21)
    var_23 = 3600
    var_24 = var_6.unsign(var_22, var_23)
    assert var_24 == b'test'
    var_25 = 'old-test'
    var_26 = var_6.sign(var_25)
    var_27 = 500
    var_28 = var_6.unsign(var_26, var_27)



# Parsed testcases at query #97
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic attributes.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'custom-salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_1, signer_kwargs=var_7)
    var_9 = var_2.iter_unsigners()
    var_10 = list(var_9)
    var_11 = len(var_10)



# Parsed testcases at query #98
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp
    var_3 = callable(var_2)
    var_4 = var_1.timestamp_to_datetime
    var_5 = callable(var_4)
    var_6 = var_1.sign
    var_7 = callable(var_6)
    var_8 = var_1.unsign
    var_9 = callable(var_8)
    var_10 = var_1.validate
    var_11 = callable(var_10)



# Parsed testcases at query #99
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = '|'
    var_5 = 'none'
    var_6 = 'sha512'
    var_7 = module_0.TimestampSigner(var_2, var_3, var_4, var_5, var_6)
    var_8 = var_1.get_timestamp()
    var_9 = var_1.timestamp_to_datetime(var_8)



# Parsed testcases at query #100
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

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
    var_11 = 100
    var_12 = var_10.sign(var_2)
    var_13 = 50
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = module_0.TimestampSigner(var_13)
    var_16 = var_15.sign(var_14)
    var_17 = 3600
    var_18 = var_1.unsign(var_16, var_17)
    var_19 = b'invalid_data'
    var_20 = var_1.unsign(var_19)
    var_21 = -1
    var_22 = var_7[:var_21]
    var_23 = b'x'
    var_24 = var_22 + var_23
    var_25 = var_1.unsign(var_24)
    var_26 = module_1.Signer(var_25)
    var_27 = var_26.sign(var_20)
    var_28 = var_1.unsign(var_27)
    var_29 = b'test_value'
    var_30 = b'invalid_timestamp'



# Parsed testcases at query #101
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'message'
    var_3 = 'hello'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = b'invalid_signature'
    var_8 = var_1.loads(var_7)
    var_9 = -1
    var_10 = 'custom-salt'
    var_11 = module_0.TimedSerializer(var_9, var_10)
    var_12 = 'wrong-salt'



# Parsed testcases at query #102
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test data'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 3600
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = 10000
    var_11 = module_0.TimestampSigner(var_0)
    var_12 = var_11.get_timestamp
    var_13 = var_11.sign(var_2)
    var_14 = 3600
    var_15 = var_11.unsign(var_13, var_14)
    var_16 = b'tampered'
    var_17 = var_3 + var_16
    var_18 = var_1.unsign(var_17)
    var_19 = b'.'
    var_20 = var_2 + var_19
    var_21 = var_2 + var_19
    var_22 = b'not-a-timestamp'
    var_23 = var_21 + var_22
    var_24 = var_23 + var_19
    var_25 = var_2 + var_19
    var_26 = var_25 + var_22
    var_27 = module_0.TimestampSigner(var_18)
    var_28 = var_27.sign(var_2)
    var_29 = 3600
    var_30 = var_27.unsign(var_28, var_29)
    var_31 = 'test string'
    var_32 = var_1.sign(var_31)
    var_33 = var_1.unsign(var_32)
    var_34 = b'test bytes'
    var_35 = var_1.sign(var_34)
    var_36 = var_1.unsign(var_35)
    var_37 = ':'
    var_38 = module_0.TimestampSigner(var_29, sep=var_37)
    var_39 = b'custom test'
    var_40 = var_38.sign(var_39)
    var_41 = var_38.unsign(var_40)
    var_42 = var_1.unsign(var_3, var_8, var_30)



# Parsed testcases at query #103
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = '|'
    var_4 = 'none'
    var_5 = b'bytes-secret'
    var_6 = module_0.TimestampSigner(var_5)
    var_7 = ''
    var_8 = module_0.TimestampSigner(var_7)



# Parsed testcases at query #104
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimestampSigner(var_0, var_2)
    var_4 = '|'
    var_5 = module_0.TimestampSigner(var_0, sep=var_4)



# Parsed testcases at query #105
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = 'custom-key'
    var_4 = 'custom-salt'
    var_5 = ':'
    var_6 = 'none'
    var_7 = module_0.TimestampSigner(var_3, var_4, var_5, var_6)



# Parsed testcases at query #106
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 0
    var_8 = 1000000
    var_9 = 3600
    var_10 = 'bad-signature'
    var_11 = var_1.loads(var_10)
    var_12 = 'custom-salt'
    var_13 = 'wrong-salt'
    var_14 = {}
    var_15 = 2
    var_16 = 3
    var_17 = [var_5, var_15, var_16]
    var_18 = None



# Parsed testcases at query #107
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = -1
    var_7 = True
    var_8 = b'invalid-data'
    var_9 = var_1.loads(var_8)
    var_10 = 'custom-salt'
    var_11 = 'wrong-salt'
    var_12 = {}
    var_13 = 'test-string'
    var_14 = 42
    var_15 = 2
    var_16 = 3
    var_17 = [var_7, var_15, var_16]



# Parsed testcases at query #108
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_data'
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
    var_15 = 50
    var_16 = var_1.sign(var_2)
    var_17 = 3600
    var_18 = var_1.unsign(var_16, var_17)
    var_19 = b'invalid_signature'
    var_20 = var_1.unsign(var_19)
    var_21 = var_1.sign(var_2)
    var_22 = -1
    var_23 = var_21[:var_22]
    var_24 = -1
    var_25 = var_21[var_24:]
    var_26 = b'x'
    var_27 = var_25 != var_26
    var_28 = b'y'
    var_29 = var_26 if var_27 else var_28
    var_30 = var_23 + var_29
    var_31 = var_1.unsign(var_30)
    var_32 = b'value_without_separator'
    var_33 = var_1.unsign(var_32)
    var_34 = 'test_string'
    var_35 = var_1.sign(var_34)
    var_36 = var_1.unsign(var_35)
    assert var_36 == b'test_string'
    var_37 = b''
    var_38 = var_1.sign(var_37)
    var_39 = var_1.unsign(var_38)
    assert var_39 == b''
    var_40 = var_1.sign(var_2)
    var_41 = -10
    var_42 = var_40[:var_41]
    var_43 = b'invalid_base64'
    var_44 = var_42 + var_43
    var_45 = -1
    var_46 = var_40[var_45:]
    var_47 = var_44 + var_46
    var_48 = var_1.unsign(var_47)
    var_49 = var_1.sign(var_2)



# Parsed testcases at query #109
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
    var_13 = 'salt'
    var_14 = 'fallback-salt'
    var_15 = {var_13: var_14}
    var_16 = [var_15]
    var_17 = module_0.TimedSerializer(fallback_signers=var_16)
    var_18 = var_17.fallback_signers
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = module_0.TimedSerializer()
    var_21 = 'test-data'
    var_22 = 'test-key'
    var_23 = module_0.TimedSerializer(var_22)
    var_24 = 'key'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = 3600
    var_28 = True
    var_29 = 'secret_key'
    var_30 = 'fallback-key'
    var_31 = {var_29: var_30}
    var_32 = [var_31]
    var_33 = module_0.TimedSerializer(var_22, fallback_signers=var_32)



# Parsed testcases at query #110
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
    var_8 = 3600
    var_9 = b'invalid-signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'custom-salt'
    var_12 = 'wrong-salt'



# Parsed testcases at query #111
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = b'.'
    var_5 = var_1.unsign(var_3)
    var_6 = True
    var_7 = var_1.unsign(var_3, return_timestamp=var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_7[var_6]
    var_10 = var_1.get_timestamp()
    var_11 = b'expired-test'
    var_12 = var_1.sign(var_11)
    var_13 = -1
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = b'invalid-data'
    var_16 = var_1.unsign(var_15)
    var_17 = var_1.validate(var_3)
    assert var_17 is True
    var_18 = b'invalid-data'
    var_19 = var_1.validate(var_18)
    assert var_19 is False



# Parsed testcases at query #112
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = ':'
    var_4 = module_0.TimestampSigner(var_0, var_2, var_3)
    var_5 = b'test-secret-bytes'
    var_6 = module_0.TimestampSigner(var_5)
    var_7 = 'hmac'
    var_8 = module_0.TimestampSigner(var_0, key_derivation=var_7)
    var_9 = 'sha256'
    var_10 = module_0.TimestampSigner(var_0, digest_method=var_9)



# Parsed testcases at query #113
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.TimedSerializer(var_0)
    var_6 = 'test'
    var_7 = 3600
    var_8 = module_0.TimedSerializer(var_0)
    var_9 = True
    var_10 = module_0.TimedSerializer(var_0)
    var_11 = module_0.TimedSerializer(var_0)
    var_12 = 0.1
    var_13 = 0
    var_14 = module_0.TimedSerializer(var_13)
    var_15 = 'invalid_signature'
    var_16 = var_14.loads(var_15)
    var_17 = 'custom_salt'
    var_18 = module_0.TimedSerializer(var_15, var_17)
    var_19 = module_0.TimedSerializer(var_15, var_17)
    var_20 = 'wrong_salt'
    var_21 = module_0.TimedSerializer(var_20)
    var_22 = module_0.TimedSerializer(var_20)
    var_23 = module_0.TimedSerializer(var_20)
    var_24 = 'list'
    var_25 = 'nested'
    var_26 = 'bool'
    var_27 = 'none'
    var_28 = 2
    var_29 = 3
    var_30 = [var_9, var_28, var_29]
    var_31 = 'a'
    var_32 = {var_31: var_9}
    var_33 = None
    var_34 = {var_24: var_30, var_25: var_32, var_26: var_9, var_27: var_33}



# Parsed testcases at query #114
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6[var_5]
    var_9 = 3600
    var_10 = var_1.unsign(var_3, var_9)
    var_11 = var_1.get_timestamp
    var_12 = var_1.sign(var_2)
    var_13 = 10
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = b'.'
    var_16 = var_2 + var_15
    var_17 = b'invalid-timestamp'
    var_18 = var_16 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = var_1.unsign(var_2)
    var_21 = b'tampered'
    var_22 = var_3 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = 'utf-8'
    var_25 = var_1.unsign(var_3, return_timestamp=var_14)
    var_26 = var_1.sign(var_2)
    var_27 = 100
    var_28 = var_1.unsign(var_26, var_27)
    var_29 = str(var_27)



# Parsed testcases at query #115
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method with various scenarios.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 3600
    var_8 = module_0.TimedSerializer(var_1)
    var_9 = var_8.default_signer.get_timestamp
    var_10 = 1
    var_11 = 'custom-salt'
    var_12 = 'wrong-salt'
    var_13 = b'invalid-data'
    var_14 = var_2.loads(var_13)
    var_15 = b'invalid|'
    var_16 = b'|'
    var_17 = b'invalid'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_16 + var_18
    var_20 = b'invalid-data'
    var_21 = var_2.loads_unsafe(var_20)
    var_22 = module_0.TimedSerializer(var_14)
    var_23 = 0
    var_24 = 'utf-8'



# Parsed testcases at query #116
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret_key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = 1.5
    var_9 = 1
    var_10 = 'invalid_signature_data'
    var_11 = var_1.loads(var_10)
    var_12 = 'custom_salt'
    var_13 = 'wrong_salt'



# Parsed testcases at query #117
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
    var_13 = {var_5: var_6}
    var_14 = {var_9: var_10}
    var_15 = module_0.TimedSerializer(var_1, var_3, serializer_kwargs=var_14, signer_kwargs=var_13)
    var_16 = module_0.TimedSerializer()
    var_17 = var_0.iter_unsigners()
    var_18 = list(var_17)
    var_19 = len(var_18)



# Parsed testcases at query #118
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
    var_9 = 'my-secret'
    var_10 = 'my-salt'
    var_11 = 'json'
    var_12 = 'none'
    var_13 = {var_5: var_12}
    var_14 = var_0.iter_unsigners()
    var_15 = list(var_14)
    var_16 = len(var_15)



# Parsed testcases at query #119
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and default attributes.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'custom-key'
    var_4 = 'custom-salt'
    var_5 = 'protocol'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = 'key_derivation'
    var_9 = 'hmac'
    var_10 = {var_8: var_9}
    var_11 = module_0.TimedSerializer(var_3, var_4, serializer_kwargs=var_7, signer_kwargs=var_10)
    var_12 = b'bytes-key'
    var_13 = module_0.TimedSerializer(var_12)
    var_14 = 'fallback-key'
    var_15 = 'key1'
    var_16 = 'key2'
    var_17 = [var_15, var_16]
    var_18 = module_0.TimedSerializer(var_14, fallback_signers=var_17)



# Parsed testcases at query #120
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test-value'
    var_6 = True
    var_7 = 3600
    var_8 = var_2.unsign(var_4, var_7)
    assert var_8 == b'test-value'
    var_9 = var_2.get_timestamp
    var_10 = 7200
    var_11 = 3600
    var_12 = var_2.unsign(var_4, var_11)
    var_13 = 'future-value'
    var_14 = var_2.sign(var_13)
    var_15 = var_2.get_timestamp
    var_16 = 100
    var_17 = 3600
    var_18 = var_2.unsign(var_14, var_17)
    var_19 = b'invalid-signature'
    var_20 = var_2.unsign(var_19)
    var_21 = -1
    var_22 = var_4[:var_21]
    var_23 = b'x'
    var_24 = var_22 + var_23
    var_25 = var_2.unsign(var_24)
    var_26 = b''
    var_27 = var_2.unsign(var_26)
    var_28 = 'other-key'
    var_29 = module_0.TimestampSigner(var_28)
    var_30 = 'other-value'
    var_31 = var_29.sign(var_30)
    var_32 = var_2.unsign(var_31)
    var_33 = b'test'
    var_34 = var_2.sep
    var_35 = var_2.sep
    var_36 = b'not-a-timestamp'
    var_37 = b'test'
    var_38 = var_2.sep
    var_39 = var_2.sep



# Parsed testcases at query #121
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
    var_8 = var_7.default_signer.get_timestamp
    var_9 = int(var_0)
    var_10 = 100
    var_11 = var_9 - var_10
    var_12 = int(var_6)
    var_13 = 10
    var_14 = b'invalid-data'
    var_15 = var_1.loads(var_14)
    var_16 = 'custom-salt'
    var_17 = 'wrong-salt'
    var_18 = 'test-secret-2'
    var_19 = module_0.TimedSerializer(var_18)



# Parsed testcases at query #122
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test'
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = var_6[var_5]
    var_8 = 'get_timestamp'
    var_9 = 1000
    var_10 = 'old'
    var_11 = var_1.sign(var_10)
    var_12 = 10
    var_13 = var_1.unsign(var_11, var_12)
    var_14 = 'future'
    var_15 = var_1.sign(var_14)
    var_16 = 10
    var_17 = var_1.unsign(var_15, var_16)
    var_18 = b'invalid'
    var_19 = var_1.unsign(var_18)
    var_20 = 10
    var_21 = 'test'
    var_22 = signer.sign(var_21)[:var_20]
    var_23 = var_1.unsign(var_22)
    var_24 = b'notanumber'
    var_25 = module_1.base64_encode(var_24)
    var_26 = b'test'



# Parsed testcases at query #123
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
    var_8 = var_7.default_signer.get_timestamp
    var_9 = 10
    var_10 = b'invalid-data'
    var_11 = var_1.loads(var_10)
    var_12 = 'custom-salt'
    var_13 = module_0.TimedSerializer(var_10, var_12)
    var_14 = 'wrong-salt'
    var_15 = 'utf-8'



# Parsed testcases at query #124
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = ':'
    var_4 = 'none'
    var_5 = 'sha256'
    var_6 = module_0.TimestampSigner(var_0, var_2, var_3, var_4, var_5)
    var_7 = None
    var_8 = module_0.TimestampSigner(var_0, var_7)



# Parsed testcases at query #125
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = b'test_value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = True
    var_7 = var_2.unsign(var_4, return_timestamp=var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_7[var_6]
    var_10 = 3600
    var_11 = var_2.unsign(var_4, var_10)
    var_12 = 3600
    var_13 = 3600
    var_14 = b'invalid_signature'
    var_15 = var_2.unsign(var_14)
    var_16 = -1
    var_17 = var_4[:var_16]
    var_18 = -1
    var_19 = var_4[var_18:]
    var_20 = b'\x00'
    var_21 = var_19 != var_20
    var_22 = b'\x01'
    var_23 = var_20 if var_21 else var_22
    var_24 = var_17 + var_23
    var_25 = var_2.unsign(var_24)
    var_26 = module_1.Signer(var_25, var_15)
    var_27 = var_26.sign(var_3)
    var_28 = var_2.unsign(var_27)
    var_29 = b'not_a_timestamp'
    var_30 = module_2.base64_encode(var_29)



# Parsed testcases at query #126
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'default-key'
    var_4 = module_0.TimestampSigner(var_3)
    var_5 = b'bytes-key'
    var_6 = module_0.TimestampSigner(var_5)



# Parsed testcases at query #127
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = 'my-secret-key'
    var_2 = module_0.TimestampSigner(var_1)
    var_3 = 'my-salt'
    var_4 = module_0.TimestampSigner(var_1, var_3)
    var_5 = 'hmac'
    var_6 = module_0.TimestampSigner(var_1, key_derivation=var_5)
    var_7 = 'sha256'
    var_8 = module_0.TimestampSigner(var_1, digest_method=var_7)
    var_9 = 'hmac-sha256'
    var_10 = module_0.TimestampSigner(var_1, algorithm=var_9)
    var_11 = '.'
    var_12 = module_0.TimestampSigner(var_1, sep=var_11)



# Parsed testcases at query #128
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test-value'
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    assert var_7 == b'test-value'
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = 3700
    var_10 = var_8.sign(var_2)
    var_11 = module_0.TimestampSigner(var_0)
    var_12 = 3600
    var_13 = var_11.unsign(var_10, var_12)
    var_14 = module_0.TimestampSigner(var_12)
    var_15 = 100
    var_16 = var_14.sign(var_13)
    var_17 = 3600
    var_18 = var_11.unsign(var_16, var_17)
    var_19 = b'invalid-signature'
    var_20 = var_1.unsign(var_19)
    var_21 = bytearray(var_3)
    var_22 = 0
    var_23 = var_21[var_22]
    var_24 = 'X'
    var_25 = ord(var_24)
    var_26 = var_23 != var_25
    var_27 = ord(var_24)
    var_28 = 'Y'
    var_29 = ord(var_28)
    var_30 = bytes(var_21)
    var_31 = var_1.unsign(var_30)
    var_32 = b'no-separator-here'
    var_33 = var_1.unsign(var_32)



# Parsed testcases at query #129
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = ':'
    var_4 = 'none'
    var_5 = 'get_timestamp'
    var_6 = hasattr(var_1, var_5)
    var_7 = 'timestamp_to_datetime'
    var_8 = hasattr(var_1, var_7)
    var_9 = 'sign'
    var_10 = hasattr(var_1, var_9)
    var_11 = 'unsign'
    var_12 = hasattr(var_1, var_11)
    var_13 = 'validate'
    var_14 = hasattr(var_1, var_13)



# Parsed testcases at query #130
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimestampSigner(var_0, var_2)
    var_4 = 'hmac'
    var_5 = module_0.TimestampSigner(var_0, key_derivation=var_4)
    var_6 = 'sha256'
    var_7 = module_0.TimestampSigner(var_0, digest_method=var_6)
    var_8 = 'hmac-sha1'
    var_9 = module_0.TimestampSigner(var_0, algorithm=var_8)



# Parsed testcases at query #131
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

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
    var_11 = var_1.unsign(var_8, return_timestamp=var_10)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = var_11[var_10]
    var_14 = b'test-bytes'
    var_15 = var_1.sign(var_14)
    var_16 = var_1.unsign(var_15)
    assert var_16 == b'test-bytes'
    var_17 = var_1.validate(var_8)
    assert var_17 is True
    var_18 = b'invalid-signature'
    var_19 = var_1.validate(var_18)
    assert var_19 is False
    var_20 = b'invalid-signature'
    var_21 = var_1.unsign(var_20)
    var_22 = 3600
    var_23 = var_1.unsign(var_8, var_22)
    assert var_23 == b'test-value'
    var_24 = 10000
    var_25 = b'old-value'
    var_26 = module_1.want_bytes(var_25)
    var_27 = var_1.sep
    var_28 = module_1.want_bytes(var_27)
    var_29 = var_26 + var_28
    var_30 = 100



# Parsed testcases at query #132
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimestampSigner(var_0, var_2)
    var_4 = ':'
    var_5 = module_0.TimestampSigner(var_0, sep=var_4)
    var_6 = 'hmac'
    var_7 = module_0.TimestampSigner(var_0, key_derivation=var_6)



# Parsed testcases at query #133
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test-salt'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'json'
    var_6 = module_0.TimedSerializer(serializer=var_5)
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = module_0.TimedSerializer(signer_kwargs=var_9)
    var_11 = 'sha256'
    var_12 = module_0.TimedSerializer()
    var_13 = {var_7: var_8}
    var_14 = module_0.TimedSerializer(var_1, var_3, var_5, signer_kwargs=var_13)



# Parsed testcases at query #134
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = 'none'
    var_4 = var_1.get_timestamp()
    var_5 = var_1.timestamp_to_datetime(var_4)
    var_6 = 'test-value'
    var_7 = var_1.sign(var_6)
    var_8 = var_1.unsign(var_7)
    assert var_8 == b'test-value'
    var_9 = True
    var_10 = var_1.validate(var_7)
    assert var_10 is True
    var_11 = b'invalid'
    var_12 = var_1.validate(var_11)
    assert var_12 is False



# Parsed testcases at query #135
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method with various scenarios.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 3600
    var_8 = 'signer_class'
    var_9 = 10
    var_10 = -1
    var_11 = b'X'
    var_12 = b''
    var_13 = var_2.loads(var_12)
    var_14 = b'invalid-data'
    var_15 = var_2.loads(var_14)
    var_16 = 'custom-salt'
    var_17 = module_0.TimedSerializer(var_15, var_16)
    var_18 = 'wrong-salt'



# Parsed testcases at query #136
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.signer



# Parsed testcases at query #137
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test_value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test_value'
    var_6 = var_2.sign(var_3)
    var_7 = True
    var_8 = var_2.sign(var_3)
    var_9 = 3600
    var_10 = var_2.unsign(var_8, var_9)
    assert var_10 == b'test_value'
    var_11 = var_2.get_timestamp
    var_12 = 100
    var_13 = var_2.sign(var_3)
    var_14 = 10
    var_15 = var_2.unsign(var_13, var_14)
    var_16 = var_2.sign(var_3)
    var_17 = 50
    var_18 = 3600
    var_19 = var_2.unsign(var_16, var_18)
    var_20 = b'test_value.sep.invalid_timestamp'
    var_21 = var_2.unsign(var_20)
    var_22 = var_2.sign(var_3)
    var_23 = b'.extra'
    var_24 = var_22 + var_23
    var_25 = var_2.unsign(var_24)
    var_26 = b'test_value.sep.timestamp.invalid_signature'
    var_27 = var_2.unsign(var_26)
    var_28 = b'test_bytes'
    var_29 = var_2.sign(var_28)
    var_30 = var_2.unsign(var_29)
    assert var_30 == b'test_bytes'
    assert var_30 == b'test_string'
    var_31 = 'test_string'
    var_32 = var_2.sign(var_31)
    var_33 = var_2.sign(var_3)
    var_34 = 10
    var_35 = True
    var_36 = var_2.unsign(var_33, var_34, var_35)
    var_37 = var_2.sign(var_36)
    var_38 = 10
    var_39 = var_2.unsign(var_37, var_38)



# Parsed testcases at query #138
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
    var_12 = var_10.sign(var_2)
    var_13 = 50
    var_14 = var_10.unsign(var_12, var_13)
    var_15 = module_0.TimestampSigner(var_13)
    var_16 = var_15.sign(var_14)
    var_17 = 3600
    var_18 = var_15.unsign(var_16, var_17)
    var_19 = b'invalid_signature'
    var_20 = var_1.unsign(var_19)
    var_21 = var_1.sign(var_20)
    var_22 = 0
    var_23 = b'invalid_timestamp'
    var_24 = b'value_without_timestamp'
    var_25 = var_1.unsign(var_24)
    var_26 = var_1.sign(var_25)
    var_27 = b'test_value_bytes'
    var_28 = var_1.sign(var_27)
    var_29 = var_1.unsign(var_28)
    assert var_29 == b'test_value_bytes'



# Parsed testcases at query #139
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = -1
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'invalid_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'garbage'
    var_13 = b'not_base64'
    var_14 = 0



# Parsed testcases at query #140
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
    var_7 = 0
    var_8 = b'invalid-signature'
    var_9 = var_1.loads(var_8)
    var_10 = 'custom-salt'
    var_11 = 'wrong-salt'
    var_12 = 0
    var_13 = True
    var_14 = {}
    var_15 = None



# Parsed testcases at query #141
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = '/'
    var_3 = module_0.TimestampSigner(var_0, sep=var_2)
    var_4 = 'custom-salt'
    var_5 = module_0.TimestampSigner(var_0, var_4)
    var_6 = 'hmac'
    var_7 = module_0.TimestampSigner(var_0, key_derivation=var_6)
    var_8 = 'sha256'
    var_9 = module_0.TimestampSigner(var_0, digest_method=var_8)
    var_10 = var_1.get_timestamp()
    var_11 = var_1.get_timestamp()
    var_12 = var_1.timestamp_to_datetime(var_11)
    var_13 = 'test-value'
    var_14 = var_1.sign(var_13)
    var_15 = b'test-value'
    var_16 = var_1.sign(var_15)
    var_17 = 'test-message'
    var_18 = var_1.sign(var_17)
    var_19 = var_1.unsign(var_18)
    var_20 = True
    var_21 = 'fresh-value'
    var_22 = var_1.sign(var_21)
    var_23 = 3600
    var_24 = var_1.unsign(var_22, var_23)
    assert var_24 == b'fresh-value'
    var_25 = var_1.validate(var_22)
    var_26 = b'invalid-signature'
    var_27 = var_1.validate(var_26)



# Parsed testcases at query #142
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp()
    var_3 = var_1.get_timestamp()



# Parsed testcases at query #143
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1
import src.itsdangerous.encoding as module_2

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
    var_14 = 10
    var_15 = var_10.unsign(var_13, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = var_16.get_timestamp
    var_18 = var_16.sign(var_15)
    var_19 = 10
    var_20 = var_16.unsign(var_18, var_19)
    var_21 = b'invalid-signature'
    var_22 = var_1.unsign(var_21)
    var_23 = module_1.Signer(var_21)
    var_24 = var_23.sign(var_22)
    var_25 = var_1.unsign(var_24)
    var_26 = b'test-value.'
    var_27 = b'malformed-timestamp'
    var_28 = var_26 + var_27
    var_29 = var_1.unsign(var_28)
    var_30 = module_2.want_bytes(var_22)
    var_31 = var_1.sep
    var_32 = module_2.want_bytes(var_31)
    var_33 = b'not-valid-base64'
    var_34 = var_30 + var_32
    var_35 = var_34 + var_33
    var_36 = var_35 + var_32



# Parsed testcases at query #144
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'my-salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)
    var_5 = 'skipkeys'
    var_6 = True
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_1, serializer_kwargs=var_7)
    var_9 = 'key_derivation'
    var_10 = 'hmac'
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(var_1, signer_kwargs=var_11)
    var_13 = 'sha256'
    var_14 = module_0.TimedSerializer(var_1)
    var_15 = module_0.TimedSerializer(var_1)
    var_16 = module_0.TimedSerializer(var_1)
    var_17 = 'key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = 0
    var_21 = 3600
    var_22 = 'secret1'
    var_23 = module_0.TimedSerializer(var_22)
    var_24 = 'secret2'
    var_25 = module_0.TimedSerializer(var_24)
    var_26 = 'test'
    var_27 = 'secret'
    var_28 = 'salt1'
    var_29 = module_0.TimedSerializer(var_27, var_28)
    var_30 = 'salt2'
    var_31 = module_0.TimedSerializer(var_27, var_30)
    var_32 = 'test'



# Parsed testcases at query #145
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

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
    var_12 = var_1.sign(var_2)
    var_13 = 50
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = var_1.sign(var_14)
    var_16 = 200
    var_17 = 3600
    var_18 = var_1.unsign(var_15, var_17)
    var_19 = b'invalid_signature'
    var_20 = var_1.unsign(var_19)
    var_21 = module_1.Signer(var_19)
    var_22 = var_21.sign(var_20)
    var_23 = var_1.unsign(var_22)
    var_24 = b'test_value.sep.invalid_timestamp.sep.signature'
    var_25 = var_1.unsign(var_24)



# Parsed testcases at query #146
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = '|'
    var_4 = 'custom-salt'
    var_5 = 'none'
    var_6 = module_0.TimestampSigner(var_2, var_4, var_3, var_5)
    var_7 = var_1.salt
    var_8 = var_1.salt
    var_9 = len(var_8)
    var_10 = b'bytes-secret'
    var_11 = module_0.TimestampSigner(var_10)
    var_12 = var_11.secret_key



# Parsed testcases at query #147
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
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6[var_5]
    var_9 = 3600
    var_10 = var_1.unsign(var_3, var_9)
    assert var_10 == b'test_value'
    var_11 = 0
    var_12 = var_1.unsign(var_3, var_11)
    var_13 = -1
    var_14 = var_1.unsign(var_3, var_13)
    var_15 = b'invalid_signature'
    var_16 = var_1.unsign(var_15)
    var_17 = b'test_value'
    var_18 = var_1.sep
    var_19 = module_1.want_bytes(var_18)
    var_20 = b'invalid'
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_17 + var_19
    var_23 = var_22 + var_21
    var_24 = var_23 + var_19
    var_25 = var_17 + var_19
    var_26 = var_25 + var_21
    var_27 = b'test_bytes'
    var_28 = var_1.sign(var_27)
    var_29 = var_1.unsign(var_28)
    assert var_29 == b'test_bytes'
    assert var_29 == b'test_string'
    var_30 = 'test_string'
    var_31 = var_1.sign(var_30)
    var_32 = var_1.unsign(var_3, return_timestamp=var_5)
    var_33 = var_1.unsign(var_3, var_9, var_5)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = var_33[var_5]



# Parsed testcases at query #148
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'user'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = 'TimestampSigner'
    var_8 = var_1.loads.__globals__[var_7]
    var_9 = var_8.get_timestamp
    var_10 = 10
    var_11 = b'tampered'
    var_12 = 'custom-salt'
    var_13 = 'salt1'
    var_14 = 'salt2'
    var_15 = 'another-secret'
    var_16 = module_0.TimedSerializer(var_15)
    var_17 = module_0.TimedSerializer(var_14)
    var_18 = b'binary data'



# Parsed testcases at query #149
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = ':'
    var_5 = 'none'
    var_6 = 'hs512'



# Parsed testcases at query #150
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = '-'
    var_5 = 'none'
    var_6 = b'bytes-secret'
    var_7 = module_0.TimestampSigner(var_6)
    var_8 = var_1.get_timestamp()
    var_9 = var_1.timestamp_to_datetime(var_8)
    var_10 = 'test-value'
    var_11 = var_1.sign(var_10)
    var_12 = var_1.unsign(var_11)
    assert var_12 == b'test-value'
    var_13 = True
    var_14 = 3600
    var_15 = var_1.unsign(var_11, var_14)
    assert var_15 == b'test-value'
    var_16 = var_1.validate(var_11)
    assert var_16 is True
    var_17 = var_1.validate(var_11, var_14)
    assert var_17 is True
    var_18 = b'invalid-signature'
    var_19 = var_1.validate(var_18)
    assert var_19 is False
    var_20 = b'invalid-value'
    var_21 = var_1.unsign(var_20)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 7200
    var_8 = var_0 + var_7
    var_9 = 1
    var_10 = 'invalid_data'
    var_11 = var_1.loads(var_10)
    var_12 = 'different-salt'
    var_13 = module_0.TimedSerializer(var_10, var_12)
    var_14 = 'wrong-salt'



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

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
    var_9 = 7200
    var_10 = 'old value'
    var_11 = var_8.sign(var_10)
    var_12 = 3600
    var_13 = var_1.unsign(var_11, var_12)
    var_14 = module_0.TimestampSigner(var_12)
    var_15 = 'future value'
    var_16 = var_14.sign(var_15)
    var_17 = 3600
    var_18 = var_1.unsign(var_16, var_17)
    var_19 = b'invalid_data'
    var_20 = var_1.unsign(var_19)
    var_21 = -1
    var_22 = var_3[:var_21]
    var_23 = -1
    var_24 = var_3[var_23:]
    var_25 = b'x'
    var_26 = var_24 != var_25
    var_27 = b'y'
    var_28 = var_25 if var_26 else var_27
    var_29 = var_22 + var_28
    var_30 = var_1.unsign(var_29)
    var_31 = module_1.Signer(var_30)
    var_32 = 'no timestamp'
    var_33 = var_31.sign(var_32)
    var_34 = var_1.unsign(var_33)
    var_35 = 0
    var_36 = b'invalid_timestamp'
    var_37 = b'bytes value'
    var_38 = var_1.sign(var_37)
    var_39 = var_1.unsign(var_38)
    assert var_39 == b'bytes value'
    assert var_39 == b'string value'
    assert var_39 == b'test value'
    var_40 = 'string value'
    var_41 = var_1.sign(var_40)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'default-secret'
    var_4 = module_0.TimestampSigner(var_3)
    var_5 = 'custom-secret'
    var_6 = ':'
    var_7 = module_0.TimestampSigner(var_5, sep=var_6)



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.default_signer



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 'test'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = 3600
    var_10 = 'expired'
    var_11 = {var_10: var_6}
    var_12 = 0
    var_13 = b'invalid.signature.data'
    var_14 = var_1.loads(var_13)
    var_15 = 'salt_test'
    var_16 = {var_15: var_5}
    var_17 = 'custom-salt'
    var_18 = 'wrong-salt'
    var_19 = 'bytes_input'
    var_20 = 123
    var_21 = {var_19: var_20}
    var_22 = 'timestamp_check'
    var_23 = {var_22: var_5}
    var_24 = 'list'
    var_25 = 'nested'
    var_26 = 'bool'
    var_27 = 'none'
    var_28 = 2
    var_29 = 3
    var_30 = [var_5, var_28, var_29]
    var_31 = 'a'
    var_32 = 'b'
    var_33 = {var_31: var_5, var_32: var_6}
    var_34 = None
    var_35 = {var_24: var_30, var_25: var_33, var_26: var_5, var_27: var_34}
    var_36 = {}
    var_37 = 42
    var_38 = 'test string'
    var_39 = {var_6: var_10}
    var_40 = -5
    var_41 = b'XXXXX'
    var_42 = 'large_max_age'
    var_43 = {var_42: var_5}
    var_44 = 999999999
    var_45 = module_0.TimedSerializer(var_18)
    var_46 = 'multi_signer'
    var_47 = {var_46: var_5}



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 3600
    var_7 = True
    var_8 = module_0.TimedSerializer(var_1)
    var_9 = 0.001
    var_10 = 0
    var_11 = b'invalid-data'
    var_12 = var_2.loads(var_11)
    var_13 = 'wrong-secret'
    var_14 = module_0.TimedSerializer(var_13)
    var_15 = 'custom-salt'
    var_16 = 'wrong-salt'



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret_key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = 0
    var_8 = b'invalid_data'
    var_9 = var_1.loads(var_8)
    var_10 = 'different_salt'
    var_11 = module_0.TimedSerializer(var_8, var_10)
    var_12 = b'test_bytes'
    var_13 = 'test_string'
    var_14 = ''
    var_15 = None
    var_16 = 123
    var_17 = 2
    var_18 = 3
    var_19 = [var_6, var_17, var_18]



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test-value'
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = var_6[var_5]
    var_8 = 3600
    var_9 = var_1.unsign(var_3, var_8)
    assert var_9 == b'test-value'
    var_10 = 100
    var_11 = var_1.sign(var_2)
    var_12 = 50
    var_13 = var_1.unsign(var_11, var_12)
    var_14 = var_1.sign(var_13)
    var_15 = 3600
    var_16 = var_1.unsign(var_14, var_15)
    var_17 = b'no-timestamp-here'
    var_18 = var_1.unsign(var_17)
    var_19 = b'value.invalid-timestamp.invalid-signature'
    var_20 = var_1.unsign(var_19)
    var_21 = 'different-secret'
    var_22 = module_0.TimestampSigner(var_21)
    var_23 = var_22.sign(var_20)
    var_24 = var_1.unsign(var_23)
    var_25 = var_1.sign(var_20)
    var_26 = var_1.unsign(var_25, var_8, var_5)
    var_27 = var_26[var_5]



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
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = var_10.get_timestamp
    var_12 = 100
    var_13 = var_10.sign(var_2)
    var_14 = 10
    var_15 = var_1.unsign(var_13, var_14)
    var_16 = var_1.sign(var_2)
    var_17 = -5
    var_18 = var_16[:var_17]
    var_19 = b'XXXXX'
    var_20 = var_18 + var_19
    var_21 = var_1.unsign(var_20)
    var_22 = module_1.Signer(var_21)
    var_23 = var_22.sign(var_2)
    var_24 = var_1.unsign(var_23)
    var_25 = b'invalid_data'
    var_26 = var_1.unsign(var_25)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and default attributes.'
    var_1 = 'test_secret_key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'custom_salt'
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_3, var_4, signer_kwargs=var_7)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test message'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = 100
    var_9 = 10
    var_10 = var_1.unsign(var_3, var_9)
    var_11 = 10
    var_12 = var_1.unsign(var_3, var_11)
    var_13 = b'tampered'
    var_14 = var_3 + var_13
    var_15 = var_1.unsign(var_14)
    var_16 = b'.'
    var_17 = var_2 + var_16
    var_18 = b'not_a_timestamp'
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_17 + var_19
    var_21 = var_20 + var_16
    var_22 = var_2 + var_16
    var_23 = module_1.base64_encode(var_18)
    var_24 = var_22 + var_23
    var_25 = var_2 + var_16



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = module_0.TimedSerializer(var_0)
    var_7 = 3600
    var_8 = True
    var_9 = b'invalid'
    var_10 = var_1.loads(var_9)
    var_11 = -1
    var_12 = -1
    var_13 = b'1'
    var_14 = b'0'
    var_15 = 'salt1'
    var_16 = 'wrong_salt'
    var_17 = ''



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'data'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = True
    var_9 = 'fresh'
    var_10 = {var_9: var_5}
    var_11 = 3600
    var_12 = module_0.TimedSerializer(var_0)
    var_13 = 'old'
    var_14 = {var_13: var_5}
    var_15 = 3600
    var_16 = 'invalid-data'
    var_17 = var_1.loads(var_16)
    var_18 = 'combined'
    var_19 = {var_18: var_6}
    var_20 = 'bytes'
    var_21 = {var_20: var_6}
    var_22 = 'salted'
    var_23 = {var_22: var_5}
    var_24 = 'custom-salt'
    var_25 = 'wrong-salt'
    var_26 = 'no-age'
    var_27 = 'limit'
    var_28 = {var_26: var_27}
    var_29 = None



# Parsed testcases at query #16
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
    var_7 = 0.1
    var_8 = 0
    var_9 = b'invalid-data'
    var_10 = var_1.loads(var_9)
    var_11 = b''
    var_12 = var_1.loads(var_11)
    var_13 = 'other-secret'
    var_14 = module_0.TimedSerializer(var_13)
    var_15 = b'test bytes'
    var_16 = None
    var_17 = 'list'
    var_18 = 'nested'
    var_19 = 'num'
    var_20 = 2
    var_21 = 3
    var_22 = [var_5, var_20, var_21]
    var_23 = 'a'
    var_24 = 'b'
    var_25 = {var_23: var_24}
    var_26 = 42
    var_27 = {var_17: var_22, var_18: var_25, var_19: var_26}
    var_28 = False



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimestampSigner constructor and basic signing/unsigning.'
    var_1 = 'test-secret'
    var_2 = module_0.TimestampSigner(var_1)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'custom-salt'
    var_6 = 'different-secret'
    var_7 = module_0.TimedSerializer(var_6)



# Parsed testcases at query #19
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
    var_11 = var_10.get_timestamp
    var_12 = var_10.sign(var_2)
    var_13 = 50
    var_14 = var_10.unsign(var_12, var_13)
    var_15 = module_0.TimestampSigner(var_13)
    var_16 = var_15.sign(var_14)
    var_17 = 50
    var_18 = var_15.unsign(var_16, var_17)
    var_19 = b'test-value'
    var_20 = b'malformed'
    var_21 = b'signature'
    var_22 = var_1.sign(var_18)
    var_23 = 0
    var_24 = -1



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'custom_salt'
    var_3 = module_0.TimedSerializer(var_0, var_2)
    var_4 = 'key_derivation'
    var_5 = 'hmac'
    var_6 = {var_4: var_5}
    var_7 = module_0.TimedSerializer(var_0, signer_kwargs=var_6)



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.signer
    var_3 = 'test-salt'
    var_4 = module_0.TimedSerializer(var_0, var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_0, signer_kwargs=var_7)
    var_9 = 'none'
    var_10 = {var_5: var_9}
    var_11 = module_0.TimedSerializer(var_0, serializer_kwargs=var_10)
    var_12 = 'sha256'
    var_13 = module_0.TimedSerializer(var_0)
    var_14 = module_0.TimedSerializer(var_0)
    var_15 = 'fallback1'
    var_16 = 'fallback2'
    var_17 = [var_15, var_16]
    var_18 = module_0.TimedSerializer(var_0, fallback_signers=var_17)
    var_19 = var_1.iter_unsigners()
    var_20 = list(var_19)
    var_21 = len(var_20)



# Parsed testcases at query #22
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
    var_11 = 100
    var_12 = var_10.sign(var_2)
    var_13 = 10
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = b'test_value'
    var_16 = b'invalid_timestamp'
    var_17 = var_1.sign(var_14)
    var_18 = -1
    var_19 = var_17[:var_18]
    var_20 = -1
    var_21 = var_17[var_20:]
    var_22 = b'0'
    var_23 = var_21 == var_22
    var_24 = b'1'
    var_25 = var_24 if var_23 else var_22
    var_26 = var_19 + var_25
    var_27 = True
    var_28 = var_1.unsign(var_26, return_timestamp=var_27)
    var_29 = module_0.TimestampSigner(var_27)
    var_30 = var_29.get_timestamp()
    var_31 = 1000
    var_32 = var_30 - var_31
    var_33 = var_29.sign(var_28)
    var_34 = 500
    var_35 = var_30 - var_34
    var_36 = 100
    var_37 = var_29.unsign(var_33, var_36)



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test TimestampSigner.unsign method.'
    var_1 = 'secret-key'
    var_2 = module_0.TimestampSigner(var_1)
    var_3 = b'test_value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = True
    var_7 = 3600
    var_8 = var_2.unsign(var_4, var_7)
    var_9 = 10
    var_10 = b'invalid_signed_value'
    var_11 = var_2.unsign(var_10)
    var_12 = -1
    var_13 = var_4[:var_12]
    var_14 = b'0'
    var_15 = var_13 + var_14
    var_16 = var_2.unsign(var_15)
    var_17 = module_1.want_bytes(var_3)
    var_18 = var_2.sep
    var_19 = module_1.want_bytes(var_18)
    var_20 = var_17 + var_19



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'user'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 0.1
    var_8 = 0
    var_9 = 'invalid_signed_value'
    var_10 = var_1.loads(var_9)
    var_11 = 'custom-salt'
    var_12 = 'wrong-salt'
    var_13 = 'utf-8'
    var_14 = {}
    var_15 = 'string'
    var_16 = 'number'
    var_17 = 'list'
    var_18 = 'dict'
    var_19 = 'bool'
    var_20 = 'none'
    var_21 = 42
    var_22 = 2
    var_23 = 3
    var_24 = [var_5, var_22, var_23]
    var_25 = 'nested'
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = None
    var_29 = {var_15: var_3, var_16: var_21, var_17: var_24, var_18: var_27, var_19: var_5, var_20: var_28}
    var_30 = -1
    var_31 = 999999
    var_32 = 'test-secret-2'
    var_33 = module_0.TimedSerializer(var_32)



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test the loads method of TimedSerializer.'
    var_1 = 'test_secret_key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 3600
    var_7 = True
    var_8 = module_0.TimedSerializer(var_1)
    var_9 = 3600
    var_10 = b'invalid_signature'
    var_11 = var_2.loads(var_10)
    var_12 = -1
    var_13 = b'0'



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test value'
    var_6 = True
    var_7 = var_2.sign(var_3)
    var_8 = 3600
    var_9 = var_2.unsign(var_7, var_8)
    assert var_9 == b'test value'
    var_10 = module_0.TimestampSigner(var_0, var_1)
    var_11 = 100
    var_12 = var_10.sign(var_3)
    var_13 = 10
    var_14 = var_2.unsign(var_12, var_13)
    var_15 = b'invalid'
    var_16 = 0
    var_17 = b'test'
    var_18 = var_2.sign(var_17)
    var_19 = -1
    var_20 = var_12[:var_19]
    var_21 = b'x'
    var_22 = var_20 + var_21
    var_23 = var_2.unsign(var_22)
    var_24 = b'bytes value'
    var_25 = var_2.sign(var_24)
    var_26 = var_2.unsign(var_25)
    assert var_26 == b'bytes value'
    var_27 = 'string value'
    var_28 = var_2.sign(var_27)
    var_29 = var_2.unsign(var_28)
    assert var_29 == b'string value'
    var_30 = module_0.TimestampSigner(var_23, var_14)
    var_31 = 1000
    var_32 = b'future value'
    var_33 = var_30.sign(var_32)
    var_34 = 3600
    var_35 = var_2.unsign(var_33, var_34)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = 100
    var_9 = 10
    var_10 = b'invalid-data'
    var_11 = var_1.loads(var_10)
    var_12 = 'utf-8'
    var_13 = 'different-salt'
    var_14 = module_0.TimedSerializer(var_10, var_13)
    var_15 = 'fallback-key'
    var_16 = [var_15]
    var_17 = module_0.TimedSerializer(var_10, fallback_signers=var_16)
    var_18 = b''
    var_19 = var_1.loads(var_18)



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic attributes.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = b'test-secret'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'custom-salt'
    var_6 = module_0.TimedSerializer(var_1, var_5)
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = module_0.TimedSerializer(var_1, signer_kwargs=var_9)



# Parsed testcases at query #29
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
    var_7 = b'{"key":"value"}'
    var_8 = 10
    var_9 = b'malformed-data'
    var_10 = var_1.loads(var_9)
    var_11 = b''
    var_12 = var_1.loads(var_11)
    var_13 = 'custom-salt'
    var_14 = 'wrong-salt'
    var_15 = 42
    var_16 = 2
    var_17 = 3
    var_18 = [var_5, var_16, var_17]
    var_19 = None



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test-value'
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = 3600
    var_8 = var_1.unsign(var_3, var_7)
    assert var_8 == b'test-value'
    var_9 = 100
    var_10 = 500
    var_11 = b'malformed'
    var_12 = var_3 + var_11
    var_13 = var_1.unsign(var_12)
    var_14 = b'test-value'
    var_15 = b'test-value'
    var_16 = var_15 + var_2
    var_17 = -1
    var_18 = var_3[:var_17]
    var_19 = b'x'
    var_20 = var_18 + var_19
    var_21 = var_1.unsign(var_20)
    var_22 = var_1.unsign(var_3)
    assert var_22 == b'test-value'



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = ':'
    var_5 = 'none'
    var_6 = 'hmac-sha256'
    var_7 = ''
    var_8 = module_0.TimestampSigner(var_7)
    var_9 = b'bytes-secret'
    var_10 = module_0.TimestampSigner(var_9)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads with various scenarios.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 3600
    var_8 = 7200
    var_9 = var_0 + var_8
    var_10 = 3600
    var_11 = str(var_4)
    var_12 = b'invalid-signature'
    var_13 = var_2.loads(var_12)
    var_14 = 'custom-salt'
    var_15 = 'wrong-salt'
    var_16 = 'utf-8'
    var_17 = 'list'
    var_18 = 'nested'
    var_19 = 'bool'
    var_20 = 'none'
    var_21 = 2
    var_22 = 3
    var_23 = [var_11, var_21, var_22]
    var_24 = 'a'
    var_25 = {var_24: var_11}
    var_26 = None
    var_27 = {var_17: var_23, var_18: var_25, var_19: var_11, var_20: var_26}
    var_28 = {}
    var_29 = 42
    var_30 = 'test string'
    var_31 = 'two'
    var_32 = [var_11, var_31, var_22]
    var_33 = -1
    var_34 = 0
    var_35 = 999999999
    var_36 = b'malformed-data-without-separator'
    var_37 = var_2.loads(var_36)
    var_38 = b''
    var_39 = var_2.loads(var_38)
    var_40 = None
    var_41 = var_2.loads(var_40)



# Parsed testcases at query #33
#--------------------------




# Parsed testcases at query #34
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
    var_8 = b'invalid-data'
    var_9 = b'bytes-payload'
    var_10 = -1
    var_11 = 'different-secret'
    var_12 = module_0.TimedSerializer(var_11)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = var_2.default_signer
    var_4 = 'test-salt'
    var_5 = module_0.TimedSerializer(var_1, var_4)
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}
    var_9 = module_0.TimedSerializer(var_1, signer_kwargs=var_8)
    var_10 = module_0.TimedSerializer(var_1)
    var_11 = var_10.iter_unsigners()
    var_12 = list(var_11)
    var_13 = len(var_12)



# Parsed testcases at query #36
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
    var_10 = var_1.sign(var_2)
    var_11 = -1
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = -1
    var_14 = var_10[:var_13]
    var_15 = -1
    var_16 = var_10[var_15:]
    var_17 = b'a'
    var_18 = var_16 == var_17
    var_19 = b'x'
    var_20 = var_19 if var_18 else var_17
    var_21 = var_14 + var_20
    var_22 = var_1.unsign(var_21)
    var_23 = b'test_value'
    var_24 = b'invalid_timestamp'



# Parsed testcases at query #37
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
    var_7 = 1000000
    var_8 = 3601
    var_9 = var_7 + var_8
    var_10 = 3600
    var_11 = 'invalid-signature'
    var_12 = var_1.loads(var_11)
    var_13 = 'custom-salt'
    var_14 = 'test string'
    var_15 = {}
    var_16 = 2
    var_17 = 3
    var_18 = [var_5, var_16, var_17]
    var_19 = None
    var_20 = 0



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor creates instance with correct default signer.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test-salt'
    var_4 = 'key_derivation'
    var_5 = 'hmac'
    var_6 = {var_4: var_5}
    var_7 = module_0.TimedSerializer(var_1, var_3, serializer_kwargs=var_6)



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.signer
    var_3 = 'custom-salt'
    var_4 = module_0.TimedSerializer(var_0, var_3)
    var_5 = 'sha256'
    var_6 = module_0.TimedSerializer(var_0)
    var_7 = 'hmac'
    var_8 = module_0.TimedSerializer(var_0)
    var_9 = 'default-key'
    var_10 = module_0.TimedSerializer(var_9)



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = ':'
    var_4 = 'none'
    var_5 = 'get_timestamp'
    var_6 = hasattr(var_1, var_5)
    var_7 = 'timestamp_to_datetime'
    var_8 = hasattr(var_1, var_7)
    var_9 = 'sign'
    var_10 = hasattr(var_1, var_9)
    var_11 = 'unsign'
    var_12 = hasattr(var_1, var_11)
    var_13 = 'validate'
    var_14 = hasattr(var_1, var_13)



# Parsed testcases at query #41
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimedSerializer(var_0, var_2)
    var_4 = 'key_derivation'
    var_5 = 'none'
    var_6 = {var_4: var_5}
    var_7 = module_0.TimedSerializer(var_0, signer_kwargs=var_6)
    var_8 = 'user_id'
    var_9 = 'username'
    var_10 = 1
    var_11 = 'test'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = True



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic initialization.'
    var_1 = module_0.TimedSerializer()
    var_2 = 'test-secret'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = 'test-salt'
    var_5 = module_0.TimedSerializer(var_4)
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}
    var_9 = module_0.TimedSerializer(signer_kwargs=var_8)
    var_10 = var_9.iter_unsigners()
    var_11 = list(var_10)
    var_12 = len(var_11)



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = 100
    var_10 = 'test_value'
    var_11 = var_8.sign(var_10)
    var_12 = 10
    var_13 = var_1.loads(var_11, var_12)
    var_14 = b'invalid_data'
    var_15 = var_1.loads(var_14)
    var_16 = 'custom_salt'
    var_17 = 'wrong_salt'



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'my-salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_1, signer_kwargs=var_7)
    var_9 = module_0.TimedSerializer(var_1)
    var_10 = var_9.iter_unsigners()
    var_11 = list(var_10)
    var_12 = len(var_11)



# Parsed testcases at query #45
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
    var_6 = var_1.sign(var_2)
    var_7 = 3600
    var_8 = var_1.unsign(var_6, var_7)
    var_9 = 100
    var_10 = var_1.sign(var_2)
    var_11 = 50
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = 1000
    var_14 = var_1.unsign(var_10, var_13)
    var_15 = b'invalid_signature'
    var_16 = var_1.unsign(var_15)
    var_17 = b'test'
    var_18 = b'invalid'
    var_19 = b'invalid_timestamp'
    var_20 = module_1.base64_encode(var_19)
    var_21 = 0
    var_22 = var_1.sign(var_2)



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimedSerializer(var_0, var_2)



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimestampSigner constructor and basic functionality.'
    var_1 = 'secret-key'
    var_2 = module_0.TimestampSigner(var_1)
    var_3 = 'secret'
    var_4 = 'custom-salt'
    var_5 = '|'
    var_6 = module_0.TimestampSigner(var_3, var_4, var_5)



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test the TimestampSigner class initialization and basic functionality.'
    var_1 = 'secret-key'
    var_2 = module_0.TimestampSigner(var_1)
    var_3 = 'custom-secret'
    var_4 = '-'
    var_5 = 'custom-salt'
    var_6 = 'none'
    var_7 = 'ascii'
    var_8 = var_2.get_timestamp()
    var_9 = var_2.timestamp_to_datetime(var_8)
    var_10 = 'test-value'
    var_11 = var_2.sign(var_10)
    var_12 = b'test-value'
    var_13 = var_2.unsign(var_11)
    assert var_13 == b'test-value'
    var_14 = True
    var_15 = var_2.validate(var_11)
    assert var_15 is True
    var_16 = b'invalid-signature'
    var_17 = var_2.validate(var_16)
    assert var_17 is False
    var_18 = module_0.TimestampSigner(var_1)
    var_19 = 100
    var_20 = var_18.sign(var_10)
    var_21 = 50
    var_22 = var_2.unsign(var_20, var_21)
    var_23 = b'test-value.abc.def'
    var_24 = var_2.unsign(var_23)
    var_25 = b'test-value'
    var_26 = var_2.unsign(var_25)



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 3600
    var_7 = True
    var_8 = -1
    var_9 = 'invalid-data'
    var_10 = var_2.loads(var_9)
    var_11 = 'custom-salt'
    var_12 = 'wrong-salt'



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'number'
    var_5 = 'value'
    var_6 = 42
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 3600
    var_9 = True
    var_10 = -1
    var_11 = 'invalid-signature'
    var_12 = var_2.loads(var_11)
    var_13 = ''
    var_14 = var_2.loads(var_13)
    var_15 = 'different-salt'
    var_16 = module_0.TimedSerializer(var_14, var_15)
    var_17 = 'wrong-salt'
    var_18 = 'utf-8'
    var_19 = 'list'
    var_20 = 'nested'
    var_21 = 'bool'
    var_22 = 'none'
    var_23 = 'float'
    var_24 = 2
    var_25 = 3
    var_26 = [var_9, var_24, var_25]
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_9, var_28: var_24}
    var_30 = None
    var_31 = 3.14
    var_32 = {var_19: var_26, var_20: var_29, var_21: var_9, var_22: var_30, var_23: var_31}
    var_33 = 'two'
    var_34 = {var_3: var_5}
    var_35 = [var_9, var_33, var_25, var_34]
    var_36 = 'simple string'
    var_37 = 12345
    var_38 = None
    var_39 = True
    var_40 = 3.14159
    var_41 = 'wrong-secret-key'
    var_42 = module_0.TimedSerializer(var_41)



# Parsed testcases at query #51
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
    var_8 = 0.1
    var_9 = 0
    var_10 = b'invalid_signature'
    var_11 = var_1.loads(var_10)
    var_12 = 'custom-salt'
    var_13 = 'wrong-salt'
    var_14 = 'key_derivation'
    var_15 = 'hmac'
    var_16 = {var_14: var_15}
    var_17 = module_0.TimedSerializer(var_13, signer_kwargs=var_16)



# Parsed testcases at query #52
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.TimedSerializer(var_0)
    var_6 = 3600
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = True
    var_9 = module_0.TimedSerializer(var_0)
    var_10 = module_0.TimedSerializer(var_0)
    var_11 = -1
    var_12 = module_0.TimedSerializer(var_11)
    var_13 = b'invalid-data'
    var_14 = var_12.loads(var_13)
    var_15 = 'custom-salt'
    var_16 = module_0.TimedSerializer(var_13, var_15)
    var_17 = module_0.TimedSerializer(var_13, var_15)
    var_18 = 'wrong-salt'
    var_19 = module_0.TimedSerializer(var_18)
    var_20 = b''
    var_21 = var_19.loads(var_20)
    var_22 = module_0.TimedSerializer(var_20)
    var_23 = module_0.TimedSerializer(var_20)
    var_24 = 'other-secret'
    var_25 = module_0.TimedSerializer(var_24)
    var_26 = module_0.TimedSerializer(var_20)
    var_27 = b'test-payload'
    var_28 = module_0.TimedSerializer(var_20)
    var_29 = b'binary-data'
    var_30 = module_0.TimedSerializer(var_20)
    var_31 = None
    var_32 = module_0.TimedSerializer(var_20)
    var_33 = 2
    var_34 = 3
    var_35 = [var_8, var_33, var_34]
    var_36 = module_0.TimedSerializer(var_20)
    var_37 = 42



# Parsed testcases at query #53
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = '|'
    var_5 = 'none'
    var_6 = var_1.get_timestamp()
    var_7 = var_1.timestamp_to_datetime(var_6)
    var_8 = 'test-value'
    var_9 = var_1.sign(var_8)
    var_10 = var_1.unsign(var_9)
    assert var_10 == b'test-value'
    var_11 = True
    var_12 = var_1.sign(var_8)
    var_13 = 3600
    var_14 = var_1.unsign(var_12, var_13)
    assert var_14 == b'test-value'
    var_15 = 7200
    var_16 = 'old-value'
    var_17 = var_1.sign(var_16)
    var_18 = 3600
    var_19 = var_1.unsign(var_17, var_18)
    var_20 = b'value.invalid-timestamp'
    var_21 = var_1.unsign(var_20)
    var_22 = b'value'
    var_23 = var_1.unsign(var_22)
    var_24 = var_1.sign(var_8)
    var_25 = b'corrupted'
    var_26 = var_24 + var_25
    var_27 = var_1.unsign(var_26)
    var_28 = var_1.sign(var_8)
    var_29 = var_1.validate(var_28)
    assert var_29 is True
    var_30 = var_1.validate(var_28, var_13)
    assert var_30 is True
    var_31 = b'invalid'
    var_32 = var_1.validate(var_31)
    assert var_32 is False
    var_33 = b'bytes-value'
    var_34 = var_1.sign(var_33)
    var_35 = var_1.unsign(var_34)
    assert var_35 == b'bytes-value'
    var_36 = 'string-value'
    var_37 = var_1.sign(var_36)
    var_38 = var_1.unsign(var_37)
    assert var_38 == b'string-value'



# Parsed testcases at query #54
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
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = var_1.sign(var_2)
    var_9 = 3600
    var_10 = var_1.unsign(var_8, var_9)
    assert var_10 == b'test_value'
    var_11 = 50
    var_12 = 50
    var_13 = 'wrong-key'
    var_14 = module_0.TimestampSigner(var_13)
    var_15 = var_1.sign(var_2)
    var_16 = var_14.unsign(var_15)
    var_17 = module_1.want_bytes(var_2)
    var_18 = var_1.sep
    var_19 = module_1.want_bytes(var_18)
    var_20 = b'!!!invalid!!!'
    var_21 = var_17 + var_19
    var_22 = var_21 + var_20
    var_23 = var_22 + var_19
    var_24 = var_17 + var_19
    var_25 = var_24 + var_20
    var_26 = var_17 + var_19
    var_27 = var_1.sign(var_2)
    var_28 = var_1.sign(var_2)
    var_29 = var_1.sign(var_2)
    var_30 = 0
    var_31 = var_1.unsign(var_29, var_30)



# Parsed testcases at query #55
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp
    var_3 = callable(var_2)
    var_4 = var_1.timestamp_to_datetime
    var_5 = callable(var_4)
    var_6 = var_1.sign
    var_7 = callable(var_6)
    var_8 = var_1.unsign
    var_9 = callable(var_8)
    var_10 = var_1.validate
    var_11 = callable(var_10)



# Parsed testcases at query #56
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = var_7.default_signer.get_timestamp
    var_9 = 10000
    var_10 = 1
    var_11 = b'invalid_signature'
    var_12 = var_1.loads(var_11)
    var_13 = 'custom_salt'
    var_14 = 'test_string'
    var_15 = 42
    var_16 = 2
    var_17 = 3
    var_18 = [var_6, var_16, var_17]
    var_19 = None
    var_20 = {}



# Parsed testcases at query #57
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'user'
    var_3 = 'role'
    var_4 = 'test_user'
    var_5 = 'admin'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = True
    var_8 = 3600
    var_9 = 0.1
    var_10 = 0
    var_11 = b'invalid-signature-data'
    var_12 = var_1.loads(var_11)
    var_13 = b'payload'
    var_14 = b'malformed-timestamp'
    var_15 = b''
    var_16 = var_1.loads(var_15)
    var_17 = 'different-key'
    var_18 = module_0.TimedSerializer(var_17)
    var_19 = 'custom-salt'
    var_20 = 'wrong-salt'
    var_21 = b'test-bytes-payload'
    var_22 = 'test-string-payload'
    var_23 = 'data'
    var_24 = 'timestamp'
    var_25 = 'test'
    var_26 = 'items'
    var_27 = 'nested'
    var_28 = 2
    var_29 = 3
    var_30 = [var_7, var_28, var_29]
    var_31 = 'key'
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = {var_26: var_30, var_27: var_33}



# Parsed testcases at query #58
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method with various scenarios.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 3600
    var_8 = module_0.TimedSerializer(var_1)
    var_9 = 10000
    var_10 = 1
    var_11 = 'invalid-signature'
    var_12 = var_2.loads(var_11)
    var_13 = 'custom-salt'
    var_14 = 'wrong-salt'



# Parsed testcases at query #59
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic properties.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'custom-salt'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = module_0.TimedSerializer(var_1, var_3, serializer_kwargs=var_6, signer_kwargs=var_9)



# Parsed testcases at query #60
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic initialization.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = var_2.signer



# Parsed testcases at query #61
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'my-salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)
    var_5 = 'skipkeys'
    var_6 = True
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_1, serializer_kwargs=var_7)
    var_9 = 'key_derivation'
    var_10 = 'hmac'
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(var_1, signer_kwargs=var_11)
    var_13 = 'sha256'
    var_14 = module_0.TimedSerializer(var_1)
    var_15 = module_0.TimedSerializer()



# Parsed testcases at query #62
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method with various scenarios.'
    var_1 = 'test_secret_key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 3600
    var_7 = True
    var_8 = 0
    var_9 = b'invalid_signature'
    var_10 = var_2.loads(var_9)
    var_11 = b''
    var_12 = var_2.loads(var_11)
    var_13 = 'custom_salt'
    var_14 = 'wrong_salt'
    var_15 = b'bytes_data'
    var_16 = 42



# Parsed testcases at query #63
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
    var_7 = 'old'
    var_8 = 'data'
    var_9 = {var_7: var_8}
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = 7200
    var_12 = 'old_data'
    var_13 = var_10.sign(var_12)
    var_14 = 1
    var_15 = var_1.loads(var_13, var_14)
    var_16 = b'invalid|data'
    var_17 = var_1.loads(var_16)
    var_18 = 'custom-salt'



# Parsed testcases at query #64
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret_key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 0.1
    var_8 = 0
    var_9 = 'invalid_signature'
    var_10 = var_1.loads(var_9)
    var_11 = ''
    var_12 = var_1.loads(var_11)
    var_13 = 'salt1'
    var_14 = 'wrong_salt'



# Parsed testcases at query #65
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
    var_9 = 'padsize'
    var_10 = 32
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(serializer_kwargs=var_11)
    var_13 = {var_5: var_6}
    var_14 = {var_9: var_10}
    var_15 = module_0.TimedSerializer(var_1, var_3, serializer_kwargs=var_14, signer_kwargs=var_13)



# Parsed testcases at query #66
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = b'test-secret'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = 'custom-salt'
    var_5 = module_0.TimedSerializer(var_0, var_4)
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}
    var_9 = module_0.TimedSerializer(var_0, signer_kwargs=var_8)
    var_10 = 'test'
    var_11 = 'data'
    var_12 = {var_10: var_11}



# Parsed testcases at query #67
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
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = -1
    var_9 = b'invalid-signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'custom-salt'
    var_12 = 'wrong-salt'
    var_13 = ''
    var_14 = None
    var_15 = 'list'
    var_16 = 'nested'
    var_17 = 'bool'
    var_18 = 'number'
    var_19 = 'float'
    var_20 = 2
    var_21 = 3
    var_22 = [var_5, var_20, var_21]
    var_23 = 'a'
    var_24 = 'b'
    var_25 = {var_23: var_5, var_24: var_20}
    var_26 = 42
    var_27 = 3.14
    var_28 = {var_15: var_22, var_16: var_25, var_17: var_5, var_18: var_26, var_19: var_27}
    var_29 = 'test-secret-2'
    var_30 = module_0.TimedSerializer(var_29)
    var_31 = [var_29]
    var_32 = module_0.TimedSerializer(var_12, fallback_signers=var_31)
    var_33 = 0



# Parsed testcases at query #68
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method with valid data.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 3600
    var_8 = 0.1
    var_9 = 0
    var_10 = 'invalid-data'
    var_11 = var_2.loads(var_10)
    var_12 = ''
    var_13 = var_2.loads(var_12)
    var_14 = 'custom-salt'
    var_15 = 'salt1'
    var_16 = 'salt2'
    var_17 = 'another-secret'
    var_18 = module_0.TimedSerializer(var_17)



# Parsed testcases at query #69
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)



# Parsed testcases at query #70
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic initialization.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = var_2.default_signer
    var_4 = 'signer_kwargs'
    var_5 = hasattr(var_2, var_4)
    var_6 = 'salt'
    var_7 = hasattr(var_2, var_6)
    var_8 = 'custom-salt'
    var_9 = module_0.TimedSerializer(var_1, var_8)
    var_10 = 'key_derivation'
    var_11 = 'hmac'
    var_12 = {var_10: var_11}
    var_13 = module_0.TimedSerializer(var_1, signer_kwargs=var_12)
    var_14 = var_2.iter_unsigners()
    var_15 = next(var_14)



# Parsed testcases at query #71
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = var_1.get_timestamp
    var_3 = callable(var_2)
    var_4 = var_1.timestamp_to_datetime
    var_5 = callable(var_4)
    var_6 = var_1.sign
    var_7 = callable(var_6)
    var_8 = var_1.unsign
    var_9 = callable(var_8)
    var_10 = var_1.validate
    var_11 = callable(var_10)



# Parsed testcases at query #72
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'my-salt'
    var_3 = module_0.TimestampSigner(var_0, var_2)
    var_4 = ':'
    var_5 = module_0.TimestampSigner(var_0, sep=var_4)
    var_6 = 'none'
    var_7 = module_0.TimestampSigner(var_0, key_derivation=var_6)
    var_8 = 'sha256'
    var_9 = module_0.TimestampSigner(var_0, digest_method=var_8)
    var_10 = 'hs256'
    var_11 = module_0.TimestampSigner(var_0, algorithm=var_10)



# Parsed testcases at query #73
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 7200
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = 3600
    var_10 = b'invalid_data'
    var_11 = var_1.loads(var_10)
    var_12 = 'utf-8'



# Parsed testcases at query #74
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
    var_8 = var_1.get_timestamp
    var_9 = 100
    var_10 = 10
    var_11 = var_1.unsign(var_3, var_10)
    var_12 = 3600
    var_13 = var_1.unsign(var_3, var_12)
    var_14 = b'test_value'
    var_15 = b'invalid_timestamp'
    var_16 = b'signature'
    var_17 = var_1.get_timestamp()
    var_18 = module_1.int_to_bytes(var_17)
    var_19 = module_1.base64_encode(var_18)
    var_20 = b'invalid_signature'



# Parsed testcases at query #75
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
    var_8 = 0.1
    var_9 = 0
    var_10 = b'invalid_data'
    var_11 = var_1.loads(var_10)
    var_12 = -1
    var_13 = b'x'
    var_14 = 'custom-salt'
    var_15 = 'wrong-salt'
    var_16 = 'string'
    var_17 = 'number'
    var_18 = 'list'
    var_19 = 'nested'
    var_20 = 'bool'
    var_21 = 'none'
    var_22 = 'hello'
    var_23 = 42
    var_24 = 2
    var_25 = 3
    var_26 = [var_6, var_24, var_25]
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_6, var_28: var_24}
    var_30 = None
    var_31 = {var_16: var_22, var_17: var_23, var_18: var_26, var_19: var_29, var_20: var_6, var_21: var_30}
    var_32 = {}



# Parsed testcases at query #76
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1
import src.itsdangerous.signer as module_2

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'test_payload'
    var_3 = True
    var_4 = 3600
    var_5 = module_0.TimestampSigner(var_0)
    var_6 = 7200
    var_7 = module_1.want_bytes(var_2)
    var_8 = var_5.sep
    var_9 = module_1.want_bytes(var_8)
    var_10 = var_7 + var_9
    var_11 = var_7 + var_9
    var_12 = 3600
    var_13 = 'invalid_data'
    var_14 = var_1.loads(var_13)
    var_15 = 3600
    var_16 = True
    var_17 = -1
    var_18 = -1
    var_19 = b'\x00'
    var_20 = b'\x01'
    var_21 = module_2.Signer(var_15)
    var_22 = var_21.sign(var_16)
    var_23 = var_1.loads(var_22)



# Parsed testcases at query #77
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
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6[var_5]
    var_9 = 3600
    var_10 = var_1.unsign(var_3, var_9)
    assert var_10 == b'test_string'
    var_11 = 100
    var_12 = var_1.sign(var_2)
    var_13 = 10
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = var_1.sign(var_2)
    var_16 = -1
    var_17 = var_1.unsign(var_15, var_16)
    var_18 = b'invalid_signature'
    var_19 = var_1.unsign(var_18)
    var_20 = b'test_value.invalidsig'
    var_21 = var_1.unsign(var_20)
    var_22 = module_0.TimestampSigner(var_20)
    var_23 = b'test_value.'
    var_24 = b'invalid_timestamp'
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_23 + var_25
    var_27 = b'.'
    var_28 = var_26 + var_27
    var_29 = module_1.base64_encode(var_24)
    var_30 = var_23 + var_29
    var_31 = 'different-key'
    var_32 = module_0.TimestampSigner(var_31)
    var_33 = var_32.sign(var_2)
    var_34 = var_1.unsign(var_33)
    var_35 = 'test_string'
    var_36 = var_1.sign(var_35)



# Parsed testcases at query #78
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 3600
    var_8 = module_0.TimedSerializer(var_1)
    var_9 = 3600
    var_10 = b'invalid-data'
    var_11 = var_2.loads(var_10)
    var_12 = 'custom-salt'
    var_13 = 'wrong-salt'
    var_14 = b''
    var_15 = var_2.loads(var_14)
    var_16 = 'utf-8'



# Parsed testcases at query #79
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
    var_7 = -1
    var_8 = 'invalid-signature'
    var_9 = var_1.loads(var_8)
    var_10 = 'custom-salt'
    var_11 = 'wrong-salt'



# Parsed testcases at query #80
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test the constructor of TimedSerializer class.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'custom-salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_1, signer_kwargs=var_7)
    var_9 = 'compress'
    var_10 = True
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(var_1, serializer_kwargs=var_11)
    var_13 = module_0.TimedSerializer(var_1)
    var_14 = 'dumps'
    var_15 = hasattr(var_2, var_14)
    var_16 = 'loads'
    var_17 = hasattr(var_2, var_16)
    var_18 = 'loads_unsafe'
    var_19 = hasattr(var_2, var_18)
    var_20 = b'test-secret-bytes'
    var_21 = module_0.TimedSerializer(var_20)



# Parsed testcases at query #81
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
    var_12 = 1000
    var_13 = var_10.sign(var_2)
    var_14 = 2000
    var_15 = 500
    var_16 = var_10.unsign(var_13, var_15)
    var_17 = module_0.TimestampSigner(var_15)
    var_18 = var_17.sign(var_16)
    var_19 = 3600
    var_20 = var_17.unsign(var_18, var_19)
    var_21 = b'invalid_signature'
    var_22 = var_1.unsign(var_21)
    var_23 = var_1.sign(var_22)
    var_24 = -1
    var_25 = var_23[:var_24]
    var_26 = -1
    var_27 = var_23[var_26:]
    var_28 = b'x'
    var_29 = var_27 != var_28
    var_30 = b'y'
    var_31 = var_28 if var_29 else var_30
    var_32 = var_25 + var_31
    var_33 = var_1.unsign(var_32)
    var_34 = b'value_without_timestamp'
    var_35 = var_34 + var_22
    var_36 = b'invalidsig'
    var_37 = var_35 + var_36
    var_38 = var_1.unsign(var_37)
    var_39 = var_1.sign(var_22)
    var_40 = 0
    var_41 = b'malformed_timestamp'
    var_42 = b''
    var_43 = var_1.sign(var_42)
    var_44 = var_1.unsign(var_43)
    assert var_44 == b''
    var_45 = b'bytes_value'
    var_46 = var_1.sign(var_45)
    var_47 = var_1.unsign(var_46)
    assert var_47 == b'bytes_value'
    assert var_47 == b'string_value'
    assert var_47 == b'test_value'
    var_48 = 'string_value'
    var_49 = var_1.sign(var_48)
    var_50 = var_1.sign(var_22)
    var_51 = var_1.sign(var_22)



# Parsed testcases at query #82
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    assert var_7 == b'string value'
    var_8 = var_1.sign(var_2)
    var_9 = 500
    var_10 = var_1.unsign(var_8, var_9)
    var_11 = var_1.sign(var_2)
    var_12 = 3600
    var_13 = var_1.unsign(var_11, var_12)
    var_14 = b'='
    var_15 = var_2 + var_14
    var_16 = 12345
    var_17 = module_1.int_to_bytes(var_16)
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_15 + var_18
    var_20 = var_1.unsign(var_19)
    var_21 = var_2 + var_14
    var_22 = b'=invalid'
    var_23 = var_2 + var_22
    var_24 = var_23 + var_14
    var_25 = var_2 + var_22
    var_26 = 'string value'
    var_27 = var_1.sign(var_26)



# Parsed testcases at query #83
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
    var_6 = var_1.sign(var_2)
    var_7 = 3600
    var_8 = var_1.unsign(var_6, var_7)
    assert var_8 == b'test_value'
    var_9 = var_1.get_timestamp
    var_10 = 100
    var_11 = var_1.sign(var_2)
    var_12 = 10
    var_13 = var_1.unsign(var_11, var_12)
    var_14 = var_1.sign(var_13)
    var_15 = 3600
    var_16 = var_1.unsign(var_14, var_15)
    var_17 = module_1.want_bytes(var_16)
    var_18 = var_1.sep
    var_19 = module_1.want_bytes(var_18)
    var_20 = var_17 + var_19
    var_21 = b'not-valid-timestamp'
    var_22 = var_20 + var_21
    var_23 = var_22 + var_19
    var_24 = var_17 + var_19
    var_25 = var_24 + var_21
    var_26 = module_1.want_bytes(var_16)
    var_27 = var_26 + var_19
    var_28 = b'invalid_signature'
    var_29 = var_1.unsign(var_28)
    var_30 = var_1.sign(var_29)
    var_31 = 0
    var_32 = b'modified_signature'
    var_33 = b'test'
    var_34 = var_33 + var_19
    var_35 = b'invalid'
    var_36 = var_34 + var_35
    var_37 = var_36 + var_19
    var_38 = b'sig'
    var_39 = var_37 + var_38
    var_40 = var_1.unsign(var_39)
    var_41 = b''
    var_42 = var_1.sign(var_41)
    var_43 = var_1.unsign(var_42)
    assert var_43 == b''
    var_44 = b'test_value'
    var_45 = var_1.sign(var_44)
    var_46 = var_1.unsign(var_45)
    assert var_46 == b'test_value'
    var_47 = var_1.sign(var_34)



# Parsed testcases at query #84
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = ':'
    var_4 = 'none'
    var_5 = 'test-value'
    var_6 = var_1.sign(var_5)
    var_7 = var_1.unsign(var_6)
    assert var_7 == b'test-value'



# Parsed testcases at query #85
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = True
    var_7 = var_2.sign(var_3)
    var_8 = 3600
    var_9 = var_2.unsign(var_7, var_8)
    var_10 = var_2.sign(var_3)
    var_11 = -1
    var_12 = var_2.unsign(var_10, var_11)
    var_13 = var_2.get_timestamp
    var_14 = 9999999999
    var_15 = var_2.sign(var_3)
    var_16 = 0
    var_17 = 3600
    var_18 = var_2.unsign(var_15, var_17)
    var_19 = var_2.sign(var_3)
    var_20 = -1
    var_21 = var_19[:var_20]
    var_22 = var_2.unsign(var_21)
    var_23 = var_2.sign(var_3)
    var_24 = b''
    var_25 = var_2.unsign(var_23)
    var_26 = var_2.sign(var_3)
    var_27 = b'invalid'
    var_28 = var_27 + var_26
    var_29 = var_2.unsign(var_28)
    var_30 = var_2.sign(var_3)
    var_31 = var_27 + var_30
    var_32 = True
    var_33 = var_2.unsign(var_31, return_timestamp=var_32)



# Parsed testcases at query #86
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
    var_8 = 7200
    var_9 = 'test-value'
    var_10 = var_7.sign(var_9)
    var_11 = 3600
    var_12 = var_1.loads(var_10, var_11)
    var_13 = 'invalid-signature'
    var_14 = var_1.loads(var_13)
    var_15 = 'custom-salt'
    var_16 = 'wrong-salt'



# Parsed testcases at query #87
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'secret-key-2'
    var_3 = '|'
    var_4 = 'custom-salt'
    var_5 = 'none'
    var_6 = 'sha256'
    var_7 = module_0.TimestampSigner(var_2, var_4, var_3, var_5, var_6)
    var_8 = b'bytes-secret'
    var_9 = module_0.TimestampSigner(var_8)
    var_10 = var_1.get_timestamp()
    var_11 = var_1.timestamp_to_datetime(var_10)



# Parsed testcases at query #88
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
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = 0.1
    var_9 = 0.05
    var_10 = 'invalid-data'
    var_11 = var_1.loads(var_10)
    var_12 = 'custom-salt'
    var_13 = module_0.TimedSerializer(var_10, var_12)
    var_14 = 'wrong-salt'
    var_15 = ''
    var_16 = None
    var_17 = 'list'
    var_18 = 'nested'
    var_19 = 'boolean'
    var_20 = 'number'
    var_21 = 2
    var_22 = 3
    var_23 = [var_5, var_21, var_22]
    var_24 = 'a'
    var_25 = {var_24: var_5}
    var_26 = 42.5
    var_27 = {var_17: var_23, var_18: var_25, var_19: var_5, var_20: var_26}



# Parsed testcases at query #89
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
    var_8 = 7200
    var_9 = b'test'
    var_10 = var_7.sign(var_9)
    var_11 = 3600
    var_12 = var_1.loads(var_10, var_11)
    var_13 = b'invalid-signature'
    var_14 = var_1.loads(var_13)
    var_15 = 'custom-salt'
    var_16 = 'wrong-salt'



# Parsed testcases at query #90
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = '|'
    var_4 = module_0.TimestampSigner(var_0, var_2, var_3)



# Parsed testcases at query #91
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = 'Test TimestampSigner.unsign method with various scenarios.'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.TimestampSigner(var_1, var_2)
    var_4 = b'test_value'
    var_5 = var_3.sign(var_4)
    var_6 = var_3.unsign(var_5)
    var_7 = var_3.sign(var_4)
    var_8 = True
    var_9 = 'Timestamp should be a datetime object'
    var_10 = 'test_string'
    var_11 = var_3.sign(var_10)
    var_12 = var_3.unsign(var_11)
    assert var_12 == b'test_string'
    var_13 = var_3.sign(var_4)
    var_14 = 3600
    var_15 = var_3.unsign(var_13, var_14)
    var_16 = module_0.TimestampSigner(var_1, var_2)
    var_17 = var_16.get_timestamp
    var_18 = 100
    var_19 = var_16.sign(var_4)
    var_20 = 50
    var_21 = var_16.unsign(var_19, var_20)
    var_22 = module_0.TimestampSigner(var_21, var_2)
    var_23 = var_22.sign(var_4)
    var_24 = 200
    var_25 = var_22.unsign(var_23, var_24)
    var_26 = var_3.sign(var_4)
    var_27 = -1
    var_28 = var_26[:var_27]
    var_29 = b'X'
    var_30 = var_28 + var_29
    var_31 = var_3.unsign(var_30)
    var_32 = module_1.Signer(var_25, var_2)
    var_33 = var_32.sign(var_4)
    var_34 = var_3.unsign(var_33)
    var_35 = b'invalid'
    var_36 = module_2.base64_encode(var_35)
    var_37 = var_3.sign(var_4)
    var_38 = var_3.sign(var_4)
    var_39 = var_3.unsign(var_38)
    var_40 = 'Result should be bytes'



# Parsed testcases at query #92
#--------------------------


import src.itsdangerous.timed as module_0

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
    var_11 = var_2.sign(var_3)
    var_12 = 0.1
    var_13 = 0
    var_14 = var_2.unsign(var_11, var_13)
    var_15 = 3600
    var_16 = var_2.unsign(var_11, var_15)
    var_17 = b'invalid-value'
    var_18 = var_2.unsign(var_17)
    var_19 = var_2.sign(var_3)
    var_20 = -1
    var_21 = var_19[:var_20]
    var_22 = -1
    var_23 = var_19[var_22:]
    var_24 = b'x'
    var_25 = var_23 != var_24
    var_26 = b'y'
    var_27 = var_24 if var_25 else var_26
    var_28 = var_21 + var_27
    var_29 = var_2.unsign(var_28)



# Parsed testcases at query #93
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
    var_8 = 100
    var_9 = 'test-value'
    var_10 = var_7.sign(var_9)
    var_11 = 10
    var_12 = var_1.loads(var_10, var_11)
    var_13 = module_0.TimestampSigner(var_11)
    var_14 = 10000
    var_15 = var_13.sign(var_9)
    var_16 = 3600
    var_17 = var_1.loads(var_15, var_16)
    var_18 = b'invalid-signature'
    var_19 = var_1.loads(var_18)
    var_20 = 'utf-8'
    var_21 = 'other-secret'
    var_22 = module_0.TimedSerializer(var_21)
    var_23 = b''
    var_24 = var_1.loads(var_23)
    var_25 = module_0.TimestampSigner(var_23)
    var_26 = 1000
    var_27 = var_25.sign(var_9)
    var_28 = 3600
    var_29 = var_1.loads(var_27, var_28)
    var_30 = 'custom-salt'
    var_31 = module_0.TimedSerializer(var_28, var_30)
    var_32 = 'nested'
    var_33 = 'data'
    var_34 = {var_32: var_33}
    var_35 = 'wrong-salt'



# Parsed testcases at query #94
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-key'
    var_3 = 'custom-salt'
    var_4 = 'sha512'
    var_5 = 'none'
    var_6 = module_0.TimestampSigner(var_2, var_3, key_derivation=var_5, digest_method=var_4)
    var_7 = b'bytes-key'
    var_8 = module_0.TimestampSigner(var_7)



# Parsed testcases at query #95
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
    var_8 = -1
    var_9 = b'invalid-data'
    var_10 = var_1.loads(var_9)
    var_11 = 'string data'
    var_12 = 12345
    var_13 = 3.14
    var_14 = None
    var_15 = 2
    var_16 = 3
    var_17 = [var_6, var_15, var_16]
    var_18 = 'nested'
    var_19 = 'data'
    var_20 = {var_19: var_3}
    var_21 = {var_18: var_20}
    var_22 = b'bytes data'
    var_23 = [var_11, var_12, var_13, var_6, var_14, var_17, var_21, var_22]
    var_24 = 'test-secret-2'
    var_25 = module_0.TimedSerializer(var_24)



# Parsed testcases at query #96
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
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = var_8.get_timestamp
    var_10 = 100
    var_11 = var_8.sign(var_2)
    var_12 = 50
    var_13 = var_8.unsign(var_11, var_12)
    var_14 = b'test-value'
    var_15 = b'invalid-timestamp'
    var_16 = b'signature'
    var_17 = b'some-data'
    var_18 = var_1.sign(var_2)
    var_19 = -1
    var_20 = var_18[:var_19]
    var_21 = b'0'
    var_22 = var_20 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = module_0.TimestampSigner(var_23)
    var_25 = 1000
    var_26 = var_24.sign(var_2)
    var_27 = 500
    var_28 = var_1.unsign(var_26, var_27)
    var_29 = var_1.sign(var_2)



# Parsed testcases at query #97
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = ':'
    var_5 = 'none'
    var_6 = 'sha256'
    var_7 = module_0.TimestampSigner(var_2, var_3, var_4, var_5, var_6)



# Parsed testcases at query #98
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method.'
    var_1 = 'test-secret-key-12345'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'number'
    var_5 = 'value'
    var_6 = 42
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 3600
    var_9 = True
    var_10 = 0
    var_11 = b'invalid-data'
    var_12 = var_2.loads(var_11)
    var_13 = -1
    var_14 = b'X'



# Parsed testcases at query #99
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
    var_7 = 0
    var_8 = b'invalid-data'
    var_9 = var_1.loads(var_8)
    var_10 = 'different-salt'
    var_11 = module_0.TimedSerializer(var_8, var_10)
    var_12 = module_0.TimedSerializer(var_8)
    var_13 = 999999999
    var_14 = {}
    var_15 = 2
    var_16 = 3
    var_17 = 'test'
    var_18 = [var_5, var_15, var_16, var_17]



# Parsed testcases at query #100
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
    var_8 = var_1.get_timestamp
    var_9 = 7200
    var_10 = 3600
    var_11 = var_1.unsign(var_3, var_10)
    var_12 = b'bad'
    var_13 = var_3 + var_12
    var_14 = var_1.unsign(var_13)
    var_15 = module_1.want_bytes(var_11)
    var_16 = var_1.sep
    var_17 = module_1.want_bytes(var_16)
    var_18 = var_15 + var_17
    var_19 = -1
    var_20 = var_3[:var_19]
    var_21 = b'X'
    var_22 = var_20 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = b'bytes_value'
    var_25 = var_1.sign(var_24)
    var_26 = var_1.unsign(var_25)
    assert var_26 == b'bytes_value'
    var_27 = var_1.validate(var_3)
    assert var_27 is True
    var_28 = b'invalid'
    var_29 = var_1.validate(var_28)
    assert var_29 is False
    var_30 = 1000
    var_31 = b'test_value'
    var_32 = var_31 + var_17
    var_33 = var_31 + var_17
    var_34 = 60



# Parsed testcases at query #101
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
    var_7 = 50
    var_8 = b'invalid_data'
    var_9 = var_1.loads(var_8)
    var_10 = 'salt1'
    var_11 = 'wrong_salt'



# Parsed testcases at query #102
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method.'
    var_1 = 'test_secret_key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 3600
    var_8 = module_0.TimedSerializer(var_1)
    var_9 = 100
    var_10 = 10
    var_11 = b'tampered'
    var_12 = b'no_timestamp'
    var_13 = var_2.loads(var_12)
    var_14 = 'custom_salt'
    var_15 = 'wrong_salt'
    var_16 = 'another_secret_key'
    var_17 = module_0.TimedSerializer(var_16)
    var_18 = b''
    var_19 = var_2.loads(var_18)
    var_20 = 'test_secret_key'
    var_21 = module_0.TimedSerializer(var_20)
    var_22 = 0



# Parsed testcases at query #103
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimestampSigner(var_0, var_2)
    var_4 = ':'
    var_5 = module_0.TimestampSigner(var_0, sep=var_4)
    var_6 = 'none'
    var_7 = module_0.TimestampSigner(var_0, key_derivation=var_6)
    var_8 = var_7.get_timestamp()
    var_9 = var_7.timestamp_to_datetime(var_8)



# Parsed testcases at query #104
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

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
    var_13 = 'old-value'
    var_14 = var_2.sign(var_13)
    var_15 = 50
    var_16 = var_2.unsign(var_14, var_15)
    var_17 = 'future-value'
    var_18 = var_2.sign(var_17)
    var_19 = 3600
    var_20 = var_2.unsign(var_18, var_19)
    var_21 = b'invalid-signature'
    var_22 = var_2.unsign(var_21)
    var_23 = var_2.sign(var_3)
    var_24 = b'extra'
    var_25 = var_23 + var_24
    var_26 = var_2.unsign(var_25)
    var_27 = module_1.Signer(var_26, var_22)
    var_28 = 'no-timestamp'
    var_29 = var_27.sign(var_28)
    var_30 = var_2.unsign(var_29)
    var_31 = var_2.sign
    var_32 = b'tampered'
    var_33 = var_2.sign(var_3)
    var_34 = var_2.unsign(var_33)
    var_35 = var_2.sign(var_3)
    var_36 = var_2.unsign(var_35)
    var_37 = var_2.sign(var_3)
    var_38 = var_2.unsign(var_37, return_timestamp=var_7)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = 0
    var_41 = var_38[var_40]
    var_42 = var_38[var_7]



# Parsed testcases at query #105
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor creates instance with correct default signer.'
    var_1 = 'secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'custom-salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_1, signer_kwargs=var_7)



# Parsed testcases at query #106
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
    var_8 = 100
    var_9 = 10
    var_10 = b'tampered'
    var_11 = 'custom-salt'
    var_12 = 'wrong-salt'



# Parsed testcases at query #107
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method with various scenarios.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 3600
    var_8 = module_0.TimedSerializer(var_1)
    var_9 = 3600
    var_10 = b'invalid-signature'
    var_11 = var_2.loads(var_10)
    var_12 = ''
    var_13 = var_2.loads(var_12)
    var_14 = 'different-salt'
    var_15 = 'wrong-salt'
    var_16 = 0
    var_17 = 'list'
    var_18 = 'nested'
    var_19 = 'bool'
    var_20 = 2
    var_21 = 3
    var_22 = [var_6, var_20, var_21]
    var_23 = 'a'
    var_24 = {var_23: var_6}
    var_25 = {var_17: var_22, var_18: var_24, var_19: var_6}
    var_26 = None
    var_27 = 42
    var_28 = 'test string'



# Parsed testcases at query #108
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = 100
    var_8 = module_0.TimedSerializer(var_0)
    var_9 = var_8.default_signer.get_timestamp
    var_10 = 10
    var_11 = b'invalid_data'
    var_12 = var_1.loads(var_11)
    var_13 = 'custom_salt'
    var_14 = 'wrong_salt'



# Parsed testcases at query #109
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic initialization.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = b'test-secret'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'custom-salt'
    var_6 = module_0.TimedSerializer(var_1, var_5)
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = module_0.TimedSerializer(var_1, signer_kwargs=var_9)
    var_11 = 'none'
    var_12 = {var_7: var_11}
    var_13 = module_0.TimedSerializer(var_1, serializer_kwargs=var_12)
    var_14 = module_0.TimedSerializer(var_1)



# Parsed testcases at query #110
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
    assert var_9 == b'test_value'
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = var_10.get_timestamp
    var_12 = 1000
    var_13 = var_10.sign(var_2)
    var_14 = 2000
    var_15 = 500
    var_16 = var_10.unsign(var_13, var_15)
    var_17 = 500
    var_18 = 3600
    var_19 = var_10.unsign(var_13, var_18)
    var_20 = 0
    var_21 = b'invalid_timestamp'
    var_22 = -1
    var_23 = var_13[:var_22]
    var_24 = b'x'
    var_25 = var_23 + var_24
    var_26 = var_1.unsign(var_25)
    var_27 = 'test_value'
    var_28 = var_1.sign(var_27)
    var_29 = var_1.get_timestamp()
    var_30 = var_1.timestamp_to_datetime(var_29)



# Parsed testcases at query #111
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 3600
    var_6 = True
    var_7 = 'custom-salt'
    var_8 = module_0.TimedSerializer(var_0, var_7)
    var_9 = -1
    var_10 = b'invalid-data'
    var_11 = var_1.loads(var_10)
    var_12 = 'utf-8'
    var_13 = 'different-key'
    var_14 = module_0.TimedSerializer(var_13)



# Parsed testcases at query #112
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = '|'
    var_4 = 'none'
    var_5 = 'sha256'
    var_6 = b'bytes-secret'
    var_7 = module_0.TimestampSigner(var_6)
    var_8 = var_1.get_timestamp()
    var_9 = var_1.get_timestamp()
    var_10 = var_1.get_timestamp()
    var_11 = var_1.timestamp_to_datetime(var_10)



# Parsed testcases at query #113
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimestampSigner(var_0, var_2)
    var_4 = ':'
    var_5 = 'none'
    var_6 = module_0.TimestampSigner(var_0, var_2, var_4, var_5)
    var_7 = var_1.get_timestamp()
    var_8 = var_1.get_timestamp()
    var_9 = var_1.timestamp_to_datetime(var_8)



# Parsed testcases at query #114
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1
import src.itsdangerous.encoding as module_2

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
    var_12 = var_10.sign(var_2)
    var_13 = 50
    var_14 = var_10.unsign(var_12, var_13)
    var_15 = module_0.TimestampSigner(var_13)
    var_16 = var_15.sign(var_14)
    var_17 = 3600
    var_18 = var_15.unsign(var_16, var_17)
    var_19 = b'invalid_signature'
    var_20 = var_1.unsign(var_19)
    var_21 = var_1.sign(var_20)
    var_22 = -1
    var_23 = var_21[:var_22]
    var_24 = -1
    var_25 = var_21[var_24:]
    var_26 = b'0'
    var_27 = var_25 == var_26
    var_28 = b'1'
    var_29 = var_28 if var_27 else var_26
    var_30 = var_23 + var_29
    var_31 = var_1.unsign(var_30)
    var_32 = module_1.Signer(var_31)
    var_33 = var_32.sign(var_20)
    var_34 = var_1.unsign(var_33)
    var_35 = module_2.want_bytes(var_20)
    var_36 = var_1.sep
    var_37 = module_2.want_bytes(var_36)
    var_38 = b'not_a_timestamp'
    var_39 = module_2.base64_encode(var_38)
    var_40 = var_35 + var_37
    var_41 = var_40 + var_39
    var_42 = var_41 + var_37
    var_43 = var_35 + var_37
    var_44 = var_43 + var_39
    var_45 = b'test_bytes'
    var_46 = var_1.sign(var_45)
    var_47 = var_1.unsign(var_46)
    assert var_47 == b'test_bytes'
    assert var_47 == b'test_string'
    assert var_47 == b'test_value'
    var_48 = 'test_string'
    var_49 = var_1.sign(var_48)
    var_50 = var_1.sign(var_20)



# Parsed testcases at query #115
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic functionality.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)



# Parsed testcases at query #116
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimestampSigner(var_0, var_2)
    var_4 = ':'
    var_5 = module_0.TimestampSigner(var_0, sep=var_4)
    var_6 = 'none'
    var_7 = module_0.TimestampSigner(var_0, key_derivation=var_6)
    var_8 = 'sign'
    var_9 = hasattr(var_1, var_8)
    var_10 = 'unsign'
    var_11 = hasattr(var_1, var_10)
    var_12 = 'validate'
    var_13 = hasattr(var_1, var_12)
    var_14 = 'get_timestamp'
    var_15 = hasattr(var_1, var_14)
    var_16 = 'timestamp_to_datetime'
    var_17 = hasattr(var_1, var_16)



# Parsed testcases at query #117
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test-value'
    var_5 = True
    var_6 = var_1.unsign(var_3, return_timestamp=var_5)
    var_7 = var_6[var_5]
    var_8 = 3600
    var_9 = var_1.unsign(var_3, var_8)
    assert var_9 == b'test-value'
    var_10 = var_1.get_timestamp
    var_11 = 10000
    var_12 = 1
    var_13 = var_1.unsign(var_3, var_12)
    var_14 = 3600
    var_15 = var_1.unsign(var_3, var_14)
    var_16 = b'bad'
    var_17 = var_3 + var_16
    var_18 = var_1.unsign(var_17)
    var_19 = b'test-value'
    var_20 = b'invalid'



# Parsed testcases at query #118
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
    var_12 = 1000
    var_13 = var_10.sign(var_2)
    var_14 = 10
    var_15 = var_10.unsign(var_13, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = 2000
    var_18 = var_16.sign(var_15)
    var_19 = 3600
    var_20 = var_16.unsign(var_18, var_19)
    var_21 = b'invalid_signature'
    var_22 = var_1.unsign(var_21)
    var_23 = module_0.TimestampSigner(var_21)
    var_24 = var_23.sign(var_22)
    var_25 = 0
    var_26 = b'invalid_base64'
    var_27 = b'test_value'



# Parsed testcases at query #119
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = 'custom-salt'
    var_4 = ':'
    var_5 = 'none'
    var_6 = 'sha256'
    var_7 = 'test'
    var_8 = None
    var_9 = module_0.TimestampSigner(var_7, key_derivation=var_8)
    var_10 = b'bytes-secret'
    var_11 = module_0.TimestampSigner(var_10)
    var_12 = 'fallback1'
    var_13 = 'fallback2'
    var_14 = [var_12, var_13]
    var_15 = module_0.TimestampSigner(var_7)



# Parsed testcases at query #120
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-secret'
    var_3 = '|'
    var_4 = 'custom-salt'
    var_5 = 'none'
    var_6 = 'hmac'
    var_7 = var_1.get_timestamp()
    var_8 = 1609459200
    var_9 = var_1.timestamp_to_datetime(var_8)
    var_10 = 'test-value'
    var_11 = var_1.sign(var_10)
    var_12 = var_1.unsign(var_11)
    assert var_12 == b'test-value'
    var_13 = True
    var_14 = var_1.unsign(var_11, return_timestamp=var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_14[var_13]
    var_17 = var_1.validate(var_11)
    assert var_17 is True
    var_18 = b'invalid-signature'
    var_19 = var_1.validate(var_18)
    assert var_19 is False
    var_20 = 'old-value'
    var_21 = var_1.sign(var_20)
    var_22 = 1000000
    var_23 = var_1.unsign(var_21, var_22)
    assert var_23 == b'old-value'
    var_24 = b'bytes-secret'
    var_25 = module_0.TimestampSigner(var_24)
    var_26 = ''
    var_27 = module_0.TimestampSigner(var_26)



# Parsed testcases at query #121
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
    var_8 = 7200
    var_9 = 'old-data'
    var_10 = var_7.sign(var_9)
    var_11 = 3600
    var_12 = var_1.loads(var_10, var_11)
    var_13 = 'invalid-signature'
    var_14 = var_1.loads(var_13)
    var_15 = 'custom-salt'
    var_16 = 'wrong-salt'
    var_17 = None
    var_18 = 2
    var_19 = 3
    var_20 = [var_5, var_18, var_19]
    var_21 = {}
    var_22 = 'test-string'
    var_23 = 'immediate'
    var_24 = 0
    var_25 = 'negative'
    var_26 = -1



# Parsed testcases at query #122
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimestampSigner(var_0, var_2)
    var_4 = '|'
    var_5 = module_0.TimestampSigner(var_0, sep=var_4)
    var_6 = 'none'
    var_7 = module_0.TimestampSigner(var_0, key_derivation=var_6)
    var_8 = var_1.get_timestamp()
    var_9 = var_1.timestamp_to_datetime(var_8)



# Parsed testcases at query #123
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 3600
    var_9 = var_1.unsign(var_7, var_8)
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = var_10.get_timestamp
    var_12 = var_10.sign(var_2)
    var_13 = 50
    var_14 = var_10.unsign(var_12, var_13)
    var_15 = b'invalid_signature'
    var_16 = var_1.unsign(var_15)
    var_17 = var_1.sign(var_2)
    var_18 = b'malformed'
    var_19 = var_17 + var_18
    var_20 = var_1.unsign(var_19)
    var_21 = module_0.TimestampSigner(var_20)
    var_22 = var_21.sign(var_2)
    var_23 = 3600
    var_24 = var_21.unsign(var_22, var_23)
    var_25 = 'test_string'
    var_26 = var_1.sign(var_25)
    var_27 = var_1.unsign(var_26)
    assert var_27 == b'test_string'
    var_28 = var_1.sign(var_2)



# Parsed testcases at query #124
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.TimestampSigner(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test-value'
    var_6 = True
    var_7 = 3600
    var_8 = var_2.unsign(var_4, var_7)
    assert var_8 == b'test-value'
    var_9 = 0
    var_10 = var_2.unsign(var_4, var_9)
    var_11 = module_0.TimestampSigner(var_9, var_10)
    var_12 = 1000
    var_13 = var_11.sign(var_3)
    var_14 = 3600
    var_15 = var_2.unsign(var_13, var_14)
    var_16 = b'invalid-timestamp'
    var_17 = b'test-value'
    var_18 = -1
    var_19 = var_4[:var_18]
    var_20 = b'X'
    var_21 = var_19 + var_20
    var_22 = var_2.unsign(var_21)
    var_23 = True
    var_24 = var_2.unsign(var_21, return_timestamp=var_23)
    var_25 = 'different-key'
    var_26 = module_0.TimestampSigner(var_25, var_24)
    var_27 = var_26.unsign(var_4)
    var_28 = var_2.unsign(var_4)
    assert var_28 == b'test-value'
    var_29 = 0
    var_30 = True
    var_31 = var_2.unsign(var_4, var_29, var_30)



# Parsed testcases at query #125
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 3600
    var_7 = module_0.TimestampSigner(var_0)
    var_8 = 7200
    var_9 = '{"key":"value"}'
    var_10 = module_1.want_bytes(var_9)
    var_11 = var_7.get_timestamp()
    var_12 = module_1.int_to_bytes(var_11)
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_7.sep
    var_15 = module_1.want_bytes(var_14)
    var_16 = var_10 + var_15
    var_17 = var_16 + var_13
    var_18 = var_17 + var_15
    var_19 = var_10 + var_15
    var_20 = var_19 + var_13
    var_21 = 3600
    var_22 = b'invalid-data'
    var_23 = var_1.loads(var_22)
    var_24 = 'key1'
    var_25 = 'key2'
    var_26 = [var_24, var_25]
    var_27 = module_0.TimedSerializer(var_26)
    var_28 = b'invalid-data'
    var_29 = 'test'



# Parsed testcases at query #126
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'my-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'my-salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)
    var_5 = 'signer_kwargs'
    var_6 = 'key_derivation'
    var_7 = 'hmac'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = module_0.TimedSerializer(var_1, serializer_kwargs=var_9)
    var_11 = 'digest_method'
    var_12 = 'sha256'
    var_13 = {var_11: var_12}
    var_14 = module_0.TimedSerializer(var_1, signer_kwargs=var_13)
    var_15 = 'test-key'
    var_16 = module_0.TimedSerializer(var_15)
    var_17 = 'user'
    var_18 = 'role'
    var_19 = 'test'
    var_20 = 'admin'
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = 3600
    var_23 = True
    var_24 = 'utf-8'
    var_25 = b'invalid-data'
    var_26 = 'salt1'
    var_27 = module_0.TimedSerializer(var_15, var_26)
    var_28 = 'salt2'
    var_29 = module_0.TimedSerializer(var_15, var_28)



# Parsed testcases at query #127
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'Test TimedSerializer constructor with default and custom parameters.'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'custom-secret'
    var_4 = 'custom-salt'
    var_5 = 'skipkeys'
    var_6 = True
    var_7 = {var_5: var_6}
    var_8 = 'key_derivation'
    var_9 = 'hmac'
    var_10 = {var_8: var_9}
    var_11 = 'json'
    var_12 = 'fallback'
    var_13 = module_1.Signer(var_12)
    var_14 = [var_13]
    var_15 = 'sha512'
    var_16 = module_1.Signer(var_12)
    var_17 = [var_16]
    var_18 = 'test-salt'
    var_19 = b'bytes-secret'
    var_20 = module_0.TimedSerializer(var_19)



# Parsed testcases at query #128
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer.loads method'
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 3600
    var_7 = True
    var_8 = module_0.TimestampSigner(var_1)
    var_9 = 7200
    var_10 = 'expired_data'
    var_11 = var_8.sign(var_10)
    var_12 = 3600
    var_13 = var_2.loads(var_11, var_12)
    var_14 = b'invalid_signature'
    var_15 = var_2.loads(var_14)
    var_16 = 'custom-salt'
    var_17 = 'utf-8'
    var_18 = 2
    var_19 = 3
    var_20 = [var_7, var_18, var_19]
    var_21 = None
    var_22 = b'invalid'
    var_23 = 'test-secret2'
    var_24 = module_0.TimedSerializer(var_23)



# Parsed testcases at query #129
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
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = 100
    var_10 = var_8.sign(var_2)
    var_11 = 10
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = module_0.TimestampSigner(var_11)
    var_14 = var_13.sign(var_12)
    var_15 = 3600
    var_16 = var_1.unsign(var_14, var_15)
    var_17 = b'invalid_signature'
    var_18 = var_1.unsign(var_17)
    var_19 = b'test_value'
    var_20 = b'tampered'
    var_21 = 0
    var_22 = b'not_base64'
    var_23 = module_0.TimestampSigner(var_17)
    var_24 = 1000
    var_25 = var_23.sign(var_18)
    var_26 = 10
    var_27 = var_1.unsign(var_25, var_26)



# Parsed testcases at query #130
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'custom-key'
    var_3 = 'custom-salt'
    var_4 = '-'
    var_5 = 'none'
    var_6 = 'sha256'
    var_7 = var_1.get_timestamp()
    var_8 = var_1.timestamp_to_datetime(var_7)
    var_9 = 'test-value'
    var_10 = var_1.sign(var_9)
    var_11 = var_1.unsign(var_10)
    assert var_11 == b'test-value'
    var_12 = True
    var_13 = var_1.unsign(var_10, return_timestamp=var_12)
    var_14 = var_13[var_12]
    var_15 = -1
    var_16 = var_1.unsign(var_10, var_15)
    var_17 = var_1.validate(var_10)
    assert var_17 is True
    var_18 = 1000
    var_19 = var_1.validate(var_10, var_18)
    assert var_19 is True
    var_20 = b'invalid-signature'
    var_21 = var_1.validate(var_20)
    assert var_21 is False
    var_22 = b'test-bytes'
    var_23 = var_1.sign(var_22)
    var_24 = var_1.unsign(var_23)
    assert var_24 == b'test-bytes'
    var_25 = 'different-key'
    var_26 = module_0.TimestampSigner(var_25)
    var_27 = var_26.unsign(var_10)



# Parsed testcases at query #131
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'test-secret-key'
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
    var_12 = 1000
    var_13 = var_10.sign(var_2)
    var_14 = 10
    var_15 = var_10.unsign(var_13, var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = var_16.sign(var_15)
    var_18 = 10
    var_19 = var_1.unsign(var_17, var_18)
    var_20 = b'invalid-data'
    var_21 = var_1.unsign(var_20)
    var_22 = module_1.Signer(var_20)
    var_23 = var_22.sign(var_21)
    var_24 = var_1.unsign(var_23)
    var_25 = var_1.sign(var_21)
    var_26 = 0
    var_27 = b'not-base64'
    var_28 = -1
    var_29 = var_1.unsign(var_25)
    var_30 = var_1.sign(var_21)
    var_31 = var_1.sign(var_21)



# Parsed testcases at query #132
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
    var_12 = var_10.sign(var_2)
    var_13 = 50
    var_14 = var_10.unsign(var_12, var_13)
    var_15 = var_10.sign(var_14)
    var_16 = 3600
    var_17 = var_10.unsign(var_15, var_16)
    var_18 = var_1.sign(var_17)
    var_19 = 0
    var_20 = b'invalid_timestamp'
    var_21 = var_1.sign(var_17)
    var_22 = var_1.sign(var_17)
    var_23 = b'wrong'
    var_24 = var_1.sign(var_17)



# Parsed testcases at query #133
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimestampSigner.unsign method with various scenarios.'
    var_1 = 'test-secret-key'
    var_2 = module_0.TimestampSigner(var_1)
    var_3 = b'test-data'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = True
    var_7 = var_2.unsign(var_4, return_timestamp=var_6)
    var_8 = 'Expected tuple when return_timestamp=True'
    var_9 = len(var_7)
    assert var_9 == 2
    var_10 = 'Expected datetime object'
    var_11 = 3600
    var_12 = var_2.unsign(var_4, var_11)
    var_13 = 100
    var_14 = var_2.sign(var_3)
    var_15 = 0
    var_16 = var_2.unsign(var_14, var_15)
    var_17 = module_0.TimestampSigner(var_16)
    var_18 = 1000
    var_19 = var_17.sign(var_3)
    var_20 = 3600
    var_21 = var_2.unsign(var_19, var_20)
    var_22 = -5
    var_23 = var_4[:var_22]
    var_24 = b'XXXXX'
    var_25 = var_23 + var_24
    var_26 = -2
    var_27 = var_4[var_26:]
    var_28 = var_25 + var_27
    var_29 = var_2.unsign(var_28)
    var_30 = b'.'
    var_31 = var_3 + var_30
    var_32 = -3
    var_33 = var_4[:var_32]
    var_34 = b'xyz'
    var_35 = var_33 + var_34
    var_36 = var_2.unsign(var_35)
    var_37 = 'test-string'
    var_38 = var_2.sign(var_37)
    var_39 = var_2.unsign(var_38)
    assert var_39 == b'test-string'
    var_40 = 999999
    var_41 = var_2.unsign(var_4, var_40)



# Parsed testcases at query #134
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'Test TimedSerializer constructor and basic initialization.'
    var_1 = 'test_secret_key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'test_salt'
    var_4 = module_0.TimedSerializer(var_1, var_3)
    var_5 = 'key_derivation'
    var_6 = 'hmac'
    var_7 = {var_5: var_6}
    var_8 = module_0.TimedSerializer(var_1, signer_kwargs=var_7)
    var_9 = 'serializer'
    var_10 = 'json'
    var_11 = {var_9: var_10}
    var_12 = module_0.TimedSerializer(var_1, serializer_kwargs=var_11)



# Parsed testcases at query #135
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test_secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'custom_salt'
    var_3 = module_0.TimedSerializer(var_0, var_2)
    var_4 = 'key_derivation'
    var_5 = 'hmac'
    var_6 = {var_4: var_5}
    var_7 = module_0.TimedSerializer(var_0, signer_kwargs=var_6)
    var_8 = 'compress'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = module_0.TimedSerializer(var_0, serializer_kwargs=var_10)
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = 3600



# Parsed testcases at query #136
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'custom-salt'
    var_3 = module_0.TimedSerializer(var_0, var_2)
    var_4 = b'bytes-salt'
    var_5 = module_0.TimedSerializer(var_0, var_4)
    var_6 = 'test-salt'
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = module_0.TimedSerializer(var_0, var_6, signer_kwargs=var_9)
    var_11 = 'digest_method'
    var_12 = var_1.iter_unsigners()
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 0
    var_16 = var_13[var_15]
    var_17 = 'specific-salt'
    var_18 = var_1.iter_unsigners(var_17)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = var_19[var_15]



# Parsed testcases at query #137
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = var_1.signer



# Parsed testcases at query #138
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'secret'
    var_3 = 'custom-salt'
    var_4 = module_0.TimestampSigner(var_2, var_3)
    var_5 = '-'
    var_6 = module_0.TimestampSigner(var_2, sep=var_5)
    var_7 = 'none'
    var_8 = module_0.TimestampSigner(var_2, key_derivation=var_7)
    var_9 = 'hs512'
    var_10 = module_0.TimestampSigner(var_2, algorithm=var_9)



