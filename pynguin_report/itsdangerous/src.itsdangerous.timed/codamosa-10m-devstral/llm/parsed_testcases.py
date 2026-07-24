####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
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
    var_5 = 10
    var_6 = True
    var_7 = 'salt'
    var_8 = 'invalid_signature'
    var_9 = var_1.loads(var_8)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = b'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'custom-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'json'
    var_4 = module_0.TimedSerializer(serializer=var_3)
    var_5 = ':'
    var_6 = module_0.TimedSerializer()
    var_7 = 'custom-salt'
    var_8 = 'sha256'
    var_9 = 'hmac'
    var_10 = 100000
    var_11 = module_0.TimedSerializer(var_1, var_7, var_3)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'test_value'
    var_11 = b':'
    var_12 = var_10 + var_11
    var_13 = b'malformed_timestamp'
    var_14 = var_12 + var_13
    var_15 = var_14 + var_11
    var_16 = b'signature'
    var_17 = var_15 + var_16
    var_18 = var_1.unsign(var_17)
    var_19 = var_10 + var_11
    var_20 = var_19 + var_16
    var_21 = var_1.unsign(var_20)
    var_22 = var_10 + var_11
    var_23 = b'timestamp'
    var_24 = var_22 + var_23
    var_25 = var_24 + var_11
    var_26 = b'bad_signature'
    var_27 = var_25 + var_26
    var_28 = var_1.unsign(var_27)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = '|'
    var_2 = module_0.TimedSerializer()
    var_3 = 'my-secret-key'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'sha256'
    var_6 = module_0.TimedSerializer()
    var_7 = 'salt1'
    var_8 = 'salt2'
    var_9 = 'value1'
    var_10 = 'value2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = module_0.TimedSerializer()



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = '|'
    var_2 = module_0.TimedSerializer()
    var_3 = module_1.Serializer()
    var_4 = module_0.TimedSerializer(serializer=var_3)
    var_5 = 'custom_salt'
    var_6 = module_0.TimedSerializer(var_5)
    var_7 = 'secret_key'
    var_8 = module_0.TimedSerializer(var_7)
    var_9 = 'sha256'
    var_10 = module_0.TimedSerializer()
    var_11 = 'hmac'
    var_12 = module_0.TimedSerializer()



# Parsed testcases at query #8
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
    var_7 = 'salt'
    var_8 = 0
    var_9 = 'invalid-signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'secret-key'
    var_12 = module_0.TimestampSigner(var_11)
    var_13 = 'data'
    var_14 = var_12.sign(var_13)
    var_15 = b'extra'
    var_16 = var_14 + var_15
    var_17 = var_1.loads(var_16)



# Parsed testcases at query #9
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
    var_7 = module_0.TimestampSigner(var_0)
    var_8 = 3601
    var_9 = 3600
    var_10 = 'invalid-signature'
    var_11 = var_1.loads(var_10)
    var_12 = 'wrong-key'
    var_13 = module_0.TimedSerializer(var_12)



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = b'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello world'
    assert var_2 == b'hello world'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = b'old value'
    var_7 = var_1.sign(var_6)
    var_8 = 0
    var_9 = var_1.unsign(var_7, var_8)
    var_10 = -1
    var_11 = var_1.unsign(var_3, var_10)
    var_12 = b'invalid:signature'
    var_13 = var_1.unsign(var_12)
    var_14 = b'value'
    var_15 = b':'
    var_16 = var_14 + var_15
    var_17 = b'not_a_timestamp'
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_16 + var_18
    var_20 = var_19 + var_15
    var_21 = b'sig'
    var_22 = var_20 + var_21
    var_23 = var_1.unsign(var_22)
    var_24 = var_14 + var_15
    var_25 = var_24 + var_21
    var_26 = var_1.unsign(var_25)
    var_27 = 'utf-8'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = True
    var_7 = 'salt'
    var_8 = 'signer'
    var_9 = 'invalid_signature'
    var_10 = var_1.loads(var_9)
    var_11 = module_1.Signer(var_0)
    var_12 = 'data'
    var_13 = var_11.sign(var_12)
    var_14 = var_1.loads(var_13)



# Parsed testcases at query #13
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
    var_8 = 'invalid-signature'
    var_9 = var_1.loads(var_8)
    var_10 = 'custom-salt'
    var_11 = 'wrong-salt'



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = True
    var_7 = 2
    var_8 = 1
    var_9 = 'invalid-signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'custom-salt'
    var_12 = 'wrong-salt'



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = ':'
    var_2 = module_0.TimedSerializer()
    var_3 = 'json'
    var_4 = module_0.TimedSerializer(serializer=var_3)
    var_5 = 'custom_salt'
    var_6 = module_0.TimedSerializer(var_5)



# Parsed testcases at query #16
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
    var_9 = 0
    var_10 = var_8.sign(var_2)
    var_11 = 1
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = b':'
    var_14 = var_2 + var_13
    var_15 = b'malformed_timestamp'
    var_16 = var_14 + var_15
    var_17 = var_16 + var_13
    var_18 = b'signature'
    var_19 = var_17 + var_18
    var_20 = var_1.unsign(var_19)
    var_21 = var_2 + var_13
    var_22 = var_21 + var_18
    var_23 = var_1.unsign(var_22)
    var_24 = var_2 + var_13
    var_25 = var_1.get_timestamp()
    var_26 = module_1.int_to_bytes(var_25)
    var_27 = module_1.base64_encode(var_26)
    var_28 = var_24 + var_27
    var_29 = var_28 + var_13
    var_30 = b'bad_signature'
    var_31 = var_29 + var_30
    var_32 = var_1.unsign(var_31)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'secret'
    var_5 = 'test_salt'
    var_6 = 10
    var_7 = True
    var_8 = module_0.TimestampSigner()
    var_9 = b'invalid_data'
    var_10 = var_0.loads(var_9)
    var_11 = 'wrong_salt'
    var_12 = b'data_without_timestamp'
    var_13 = var_0.loads(var_12)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = module_1.Serializer()
    var_2 = module_0.TimedSerializer(serializer=var_1)
    var_3 = var_2.serializer
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimedSerializer(signer=var_4)
    var_6 = var_5.signer
    var_7 = ':'
    var_8 = module_0.TimedSerializer()
    var_9 = 'test_salt'
    var_10 = module_0.TimedSerializer(var_9)



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = 1
    var_7 = True
    var_8 = 'salt'
    var_9 = 'bad-signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'signer'
    var_12 = 0



# Parsed testcases at query #20
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
    var_7 = 'test-salt'
    var_8 = 0
    var_9 = 'invalid-signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'wrong-salt'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = True
    var_7 = 'custom-salt'
    var_8 = 20
    var_9 = 10
    var_10 = b'bad-signature'
    var_11 = var_1.loads(var_10)
    var_12 = b'data'
    var_13 = b':'
    var_14 = var_12 + var_13
    var_15 = b'malformed'
    var_16 = var_14 + var_15
    var_17 = var_16 + var_13
    var_18 = b'sig'
    var_19 = var_17 + var_18
    var_20 = var_1.loads(var_19)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = 1
    var_7 = True
    var_8 = 'test-salt'
    var_9 = module_0.TimedSerializer(var_6, var_8)
    var_10 = 'bad-signature'
    var_11 = var_1.loads(var_10)
    var_12 = 'signer'
    var_13 = 0



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = '|'
    var_2 = module_0.TimedSerializer()
    var_3 = 'custom_salt'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'secret'
    var_6 = module_0.TimedSerializer(var_5)
    var_7 = 'sha256'
    var_8 = module_0.TimedSerializer()



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = True
    var_7 = 'test-salt'
    var_8 = 2
    var_9 = 1
    var_10 = 'invalid-signature'
    var_11 = var_1.loads(var_10)
    var_12 = 'secret-key'
    var_13 = module_0.TimestampSigner(var_12)
    var_14 = 'data'
    var_15 = var_13.sign(var_14)
    var_16 = b'.invalid'
    var_17 = var_15 + var_16
    var_18 = var_1.loads(var_17)



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = '|'
    var_4 = module_0.TimedSerializer()
    var_5 = 'json'
    var_6 = module_0.TimedSerializer(serializer=var_5)
    var_7 = 'key_derivation'
    var_8 = 'hmac'
    var_9 = {var_7: var_8}
    var_10 = module_0.TimedSerializer(signer_kwargs=var_9)



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = True
    var_7 = module_0.TimedSerializer(var_0)
    var_8 = 0
    var_9 = 'invalid_signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'test-salt'
    var_12 = 'wrong-salt'



# Parsed testcases at query #27
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
    var_7 = var_6[var_5]
    var_8 = 1000
    var_9 = var_1.unsign(var_3, var_8)
    assert var_9 == b'test_value'
    var_10 = 0
    var_11 = var_1.unsign(var_3, var_10)
    var_12 = b'malformed_signature'
    var_13 = var_1.unsign(var_12)
    var_14 = b'value'
    var_15 = b':'
    var_16 = var_14 + var_15
    var_17 = b'signature'
    var_18 = var_16 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = b'value'
    var_21 = b':'
    var_22 = var_20 + var_21
    var_23 = b'malformed_ts'
    var_24 = var_22 + var_23
    var_25 = var_24 + var_21
    var_26 = b'signature'
    var_27 = var_25 + var_26
    var_28 = var_1.unsign(var_27)
    var_29 = module_0.TimestampSigner(var_28)
    var_30 = 100
    var_31 = var_29.sign(var_2)
    var_32 = 50
    var_33 = var_1.unsign(var_31, var_32)



# Parsed testcases at query #28
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
    var_8 = 0
    var_9 = 'invalid-signature'
    var_10 = var_1.loads(var_9)
    var_11 = b'value:missing-timestamp'
    var_12 = var_1.loads(var_11)



# Parsed testcases at query #29
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
    var_7 = 'salt'
    var_8 = 0
    var_9 = 'invalid-signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'wrong-salt'



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test_key'
    var_2 = 'test_salt'
    var_3 = 'json'
    var_4 = module_0.TimedSerializer(var_1, var_2, var_3)
    var_5 = module_0.TimedSerializer()



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = True
    var_7 = 'salt'
    var_8 = 'signer'
    var_9 = 10
    var_10 = 'invalid_signature'
    var_11 = var_1.loads(var_10)
    var_12 = 5
    var_13 = True
    var_14 = 5



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = ':'
    var_3 = module_0.TimestampSigner(var_0, sep=var_2)
    var_4 = 'test-salt'
    var_5 = module_0.TimestampSigner(var_0, var_4)



# Parsed testcases at query #33
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
    var_7 = 2
    var_8 = 1
    var_9 = b'invalid-signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'salt'
    var_12 = 'wrong-salt'



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = 1
    var_7 = True
    var_8 = 'salt'
    var_9 = 'wrong-salt'
    var_10 = 'invalid-signature'
    var_11 = var_1.loads(var_10)
    var_12 = 'signer'
    var_13 = 0



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = 'test-salt'
    var_5 = module_0.TimestampSigner(var_4)
    var_6 = var_5.get_timestamp()
    var_7 = var_5.get_timestamp()
    var_8 = var_5.timestamp_to_datetime(var_7)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = var_1.unsign(var_3)
    var_6 = True
    var_7 = 3600
    var_8 = var_1.unsign(var_3, var_7)
    var_9 = module_0.TimestampSigner(var_0)
    var_10 = 3601
    var_11 = var_4 + var_10
    var_12 = 3600
    var_13 = var_9.unsign(var_3, var_12)
    var_14 = module_0.TimestampSigner(var_12)
    var_15 = var_4 - var_10
    var_16 = 3600
    var_17 = var_14.unsign(var_3, var_16)
    var_18 = b':'
    var_19 = var_2 + var_18
    var_20 = b'malformed'
    var_21 = var_19 + var_20
    var_22 = var_21 + var_18
    var_23 = var_2 + var_18
    var_24 = var_23 + var_20
    var_25 = var_2 + var_18
    var_26 = var_2 + var_18
    var_27 = module_1.int_to_bytes(var_4)
    var_28 = module_1.base64_encode(var_27)
    var_29 = var_26 + var_28
    var_30 = var_29 + var_18
    var_31 = b'bad_signature'
    var_32 = var_30 + var_31
    var_33 = var_1.unsign(var_32)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = 2
    var_7 = 0
    var_8 = True
    var_9 = 'custom-salt'
    var_10 = 'wrong-salt'
    var_11 = 'invalid-signature'
    var_12 = var_1.loads(var_11)
    var_13 = 0



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'my-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = '|'
    var_4 = module_0.TimedSerializer()
    var_5 = 'json'
    var_6 = module_0.TimedSerializer(serializer=var_5)
    var_7 = 'abcdefghijklmnopqrstuvwxyz'
    var_8 = module_0.TimedSerializer()
    var_9 = module_0.TimedSerializer(var_1, serializer=var_5)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.iter_unsigners()
    var_2 = 'custom-secret'
    var_3 = module_0.TimedSerializer(var_2)
    var_4 = '|'
    var_5 = module_0.TimedSerializer()
    var_6 = 'json'
    var_7 = module_0.TimedSerializer(serializer=var_6)
    var_8 = 'sha256'
    var_9 = module_0.TimedSerializer()



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = b'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #6
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
    var_8 = 'invalid-signature'
    var_9 = var_1.loads(var_8)
    var_10 = 'test-salt'
    var_11 = module_0.TimedSerializer(var_8, var_10)



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'malformed-signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'value'
    var_13 = var_12 + var_11
    var_14 = b'signature'
    var_15 = var_13 + var_14
    var_16 = var_1.unsign(var_15)
    var_17 = -1
    var_18 = var_1.unsign(var_3, var_17)



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = True
    var_7 = b'.'
    var_8 = module_1.base64_encode(var_0)
    var_9 = b'.'
    var_10 = var_3 + var_8
    var_11 = var_10 + var_9
    var_12 = b'invalid'
    var_13 = 'test-salt'
    var_14 = 'wrong-salt'



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = '|'
    var_4 = module_0.TimedSerializer()
    var_5 = 'test-salt'
    var_6 = module_0.TimedSerializer(var_5)
    var_7 = 10
    var_8 = module_0.TimedSerializer()
    var_9 = 'hmac'
    var_10 = module_0.TimedSerializer()



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = 0
    var_10 = var_8.sign(var_2)
    var_11 = 1
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = b'test_value:malformed_timestamp:signature'
    var_14 = var_1.unsign(var_13)
    var_15 = b'test_value:signature'
    var_16 = var_1.unsign(var_15)
    var_17 = b'test_value:timestamp:bad_signature'
    var_18 = var_1.unsign(var_17)



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = 'test-salt'
    var_5 = module_0.TimestampSigner(var_4)
    var_6 = var_5.get_timestamp()
    var_7 = var_5.get_timestamp()
    var_8 = var_5.timestamp_to_datetime(var_7)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = b'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'invalid_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b':'
    var_13 = var_2 + var_12
    var_14 = b'invalid_timestamp'
    var_15 = var_13 + var_14
    var_16 = var_15 + var_12
    var_17 = b'signature'
    var_18 = var_16 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = var_2 + var_12
    var_21 = var_20 + var_17
    var_22 = var_1.unsign(var_21)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = 10
    var_5 = var_1.unsign(var_3)
    var_6 = True
    var_7 = var_1.unsign(var_3, var_4)
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = var_8.sign(var_2)
    var_10 = var_1.unsign(var_9, var_4)
    var_11 = b':'
    var_12 = var_2 + var_11
    var_13 = b'malformed_timestamp'
    var_14 = var_12 + var_13
    var_15 = var_14 + var_11
    var_16 = b'signature'
    var_17 = var_15 + var_16
    var_18 = var_1.unsign(var_17)
    var_19 = var_2 + var_11
    var_20 = var_19 + var_16
    var_21 = var_1.unsign(var_20)
    var_22 = module_0.TimestampSigner(var_21)
    var_23 = var_22.sign(var_2)
    var_24 = var_1.unsign(var_23, var_4)



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'my-secret-key'
    var_4 = module_0.TimestampSigner(var_3)
    var_5 = module_0.TimestampSigner(var_3, sep=var_1)



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #17
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
    var_10 = b'malformed_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'test_value'
    var_13 = var_1.sep
    var_14 = var_12 + var_13
    var_15 = b'invalid_base64'
    var_16 = var_14 + var_15
    var_17 = var_1.unsign(var_16)
    var_18 = module_0.TimestampSigner(var_12)
    var_19 = int(var_16)
    var_20 = 100
    var_21 = var_19 + var_20
    var_22 = var_18.sign(var_2)
    var_23 = 50
    var_24 = var_1.unsign(var_22, var_23)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = 'test-salt'
    var_5 = module_0.TimestampSigner(var_4)
    var_6 = module_0.TimestampSigner(var_4, var_1)



# Parsed testcases at query #19
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
    var_8 = var_1.sign(var_2)
    var_9 = var_1.get_timestamp
    var_10 = 3601
    var_11 = 3600
    var_12 = var_1.unsign(var_8, var_11)
    var_13 = b'test_value:malformed_timestamp:signature'
    var_14 = var_1.unsign(var_13)
    var_15 = b'test_value:signature'
    var_16 = var_1.unsign(var_15)
    var_17 = b'test_value:timestamp:bad_signature'
    var_18 = var_1.unsign(var_17)



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()



# Parsed testcases at query #21
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
    var_8 = var_1.sign(var_2)
    var_9 = var_1.get_timestamp
    var_10 = 3601
    var_11 = 3600
    var_12 = var_1.unsign(var_8, var_11)
    var_13 = b'test_value:malformed_timestamp:signature'
    var_14 = var_1.unsign(var_13)
    var_15 = b'test_value:signature'
    var_16 = var_1.unsign(var_15)
    var_17 = b'test_value:timestamp:bad_signature'
    var_18 = var_1.unsign(var_17)



# Parsed testcases at query #22
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
    var_8 = 'invalid-signature'
    var_9 = var_1.loads(var_8)
    var_10 = 'custom-salt'
    var_11 = 'wrong-salt'



# Parsed testcases at query #23
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
    var_7 = var_6[var_5]
    var_8 = 1000
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = 0
    var_11 = var_1.unsign(var_3, var_10)
    var_12 = b'malformed_signature'
    var_13 = var_1.unsign(var_12)
    var_14 = b'value'
    var_15 = var_14 + var_13
    var_16 = var_1.unsign(var_15)
    var_17 = b'malformed_ts'
    var_18 = b'signature'
    var_19 = var_1.get_timestamp()
    var_20 = 100
    var_21 = var_19 + var_20
    var_22 = module_1.int_to_bytes(var_21)
    var_23 = module_1.base64_encode(var_22)



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #25
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
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'malformed_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'test_value:missing_timestamp'
    var_13 = var_1.unsign(var_12)
    var_14 = module_0.TimestampSigner(var_12)
    var_15 = 100
    var_16 = var_14.sign(var_2)
    var_17 = 50
    var_18 = var_1.unsign(var_16, var_17)



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = b'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)
    var_6 = var_5.get_timestamp()
    var_7 = var_5.get_timestamp()
    var_8 = var_5.timestamp_to_datetime(var_7)



# Parsed testcases at query #27
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
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b':'
    var_11 = var_2 + var_10
    var_12 = b'malformed_timestamp'
    var_13 = var_11 + var_12
    var_14 = var_13 + var_10
    var_15 = b'signature'
    var_16 = var_14 + var_15
    var_17 = var_1.unsign(var_16)
    var_18 = var_2 + var_10
    var_19 = var_18 + var_15
    var_20 = var_1.unsign(var_19)
    var_21 = var_2 + var_10
    var_22 = var_1.get_timestamp()
    var_23 = module_1.int_to_bytes(var_22)
    var_24 = module_1.base64_encode(var_23)
    var_25 = var_21 + var_24
    var_26 = var_25 + var_10
    var_27 = b'bad_signature'
    var_28 = var_26 + var_27
    var_29 = var_1.unsign(var_28)



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = True
    var_7 = 2
    var_8 = 0
    var_9 = 'invalid_signature'
    var_10 = var_1.loads(var_9, var_5)
    var_11 = 'salt'
    var_12 = 'wrong_salt'



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = b'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #30
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
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'invalid'
    var_11 = var_3 + var_10
    var_12 = var_1.unsign(var_11)
    var_13 = b'::'
    var_14 = var_2 + var_13
    var_15 = b'malformed_timestamp'
    var_16 = var_14 + var_15
    var_17 = var_1.unsign(var_16)
    var_18 = var_2 + var_13
    var_19 = var_1.unsign(var_18)



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = '|'
    var_2 = module_0.TimedSerializer()
    var_3 = 'sha256'
    var_4 = module_0.TimedSerializer()
    var_5 = 'hmac'
    var_6 = module_0.TimedSerializer()
    var_7 = 'secret'
    var_8 = module_0.TimedSerializer(var_7)
    var_9 = 'salt'
    var_10 = module_0.TimedSerializer(var_9)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = var_1.unsign(var_3)
    var_6 = True
    var_7 = 3600
    var_8 = var_1.unsign(var_3, var_7)
    var_9 = module_0.TimestampSigner(var_0)
    var_10 = 3601
    var_11 = var_4 - var_10
    var_12 = var_9.sign(var_2)
    var_13 = 3600
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = module_0.TimestampSigner(var_13)
    var_16 = var_4 + var_10
    var_17 = var_15.sign(var_2)
    var_18 = 3600
    var_19 = var_1.unsign(var_17, var_18)
    var_20 = b':'
    var_21 = var_2 + var_20
    var_22 = b'malformed_timestamp'
    var_23 = var_21 + var_22
    var_24 = var_23 + var_20
    var_25 = b'signature'
    var_26 = var_24 + var_25
    var_27 = var_1.unsign(var_26)
    var_28 = var_2 + var_20
    var_29 = var_28 + var_25
    var_30 = var_1.unsign(var_29)
    var_31 = var_2 + var_20
    var_32 = module_1.int_to_bytes(var_4)
    var_33 = module_1.base64_encode(var_32)
    var_34 = var_31 + var_33
    var_35 = var_34 + var_20
    var_36 = b'bad_signature'
    var_37 = var_35 + var_36
    var_38 = var_1.unsign(var_37)



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = var_1.unsign(var_3)
    var_6 = True
    var_7 = 1000
    var_8 = var_1.unsign(var_3, var_7)
    var_9 = module_0.TimestampSigner(var_0)
    var_10 = var_4 + var_7
    var_11 = var_10 + var_6
    var_12 = var_9.unsign(var_3, var_7)
    var_13 = module_0.TimestampSigner(var_12)
    var_14 = 100
    var_15 = var_4 - var_14
    var_16 = var_13.unsign(var_3, var_7)
    var_17 = b':'
    var_18 = var_2 + var_17
    var_19 = b'malformed_timestamp'
    var_20 = var_18 + var_19
    var_21 = var_20 + var_17
    var_22 = var_2 + var_17
    var_23 = var_22 + var_19
    var_24 = var_2 + var_17
    var_25 = var_2 + var_17
    var_26 = module_1.int_to_bytes(var_4)
    var_27 = module_1.base64_encode(var_26)
    var_28 = var_25 + var_27
    var_29 = var_28 + var_17
    var_30 = b'bad_signature'
    var_31 = var_29 + var_30
    var_32 = var_1.unsign(var_31)



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()



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
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = 2000
    var_10 = var_8.sign(var_2)
    var_11 = 1000
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = b':'
    var_14 = var_2 + var_13
    var_15 = b'malformed_timestamp'
    var_16 = var_14 + var_15
    var_17 = var_16 + var_13
    var_18 = b'signature'
    var_19 = var_17 + var_18
    var_20 = var_1.unsign(var_19)
    var_21 = var_2 + var_13
    var_22 = var_21 + var_18
    var_23 = var_1.unsign(var_22)
    var_24 = var_2 + var_13
    var_25 = var_1.get_timestamp()
    var_26 = module_1.int_to_bytes(var_25)
    var_27 = module_1.base64_encode(var_26)
    var_28 = var_24 + var_27
    var_29 = var_28 + var_13
    var_30 = b'bad_signature'
    var_31 = var_29 + var_30
    var_32 = var_1.unsign(var_31)



# Parsed testcases at query #36
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
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'malformed_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'test_value:signature_without_timestamp'
    var_13 = var_1.unsign(var_12)



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'bad_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'test_value'
    var_13 = b':'
    var_14 = var_12 + var_13
    var_15 = b'bad_timestamp'
    var_16 = var_14 + var_15
    var_17 = var_16 + var_13
    var_18 = b'signature'
    var_19 = var_17 + var_18
    var_20 = var_1.unsign(var_19)
    var_21 = b'test_value'
    var_22 = b':'
    var_23 = var_21 + var_22
    var_24 = b'signature'
    var_25 = var_23 + var_24
    var_26 = var_1.unsign(var_25)



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = var_1.sign(var_2)
    var_6 = True
    var_7 = var_1.sign(var_2)
    var_8 = 100
    var_9 = var_1.unsign(var_7, var_8)
    var_10 = var_1.sign(var_2)
    var_11 = 0
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = b':'
    var_14 = var_2 + var_13
    var_15 = b'malformed_timestamp'
    var_16 = var_14 + var_15
    var_17 = var_16 + var_13
    var_18 = b'signature'
    var_19 = var_17 + var_18
    var_20 = var_1.unsign(var_19)
    var_21 = var_2 + var_13
    var_22 = var_21 + var_18
    var_23 = var_1.unsign(var_22)
    var_24 = var_2 + var_13
    var_25 = var_1.get_timestamp()
    var_26 = module_1.int_to_bytes(var_25)
    var_27 = module_1.base64_encode(var_26)
    var_28 = var_24 + var_27
    var_29 = var_28 + var_13
    var_30 = b'bad_signature'
    var_31 = var_29 + var_30
    var_32 = var_1.unsign(var_31)
    var_33 = -1
    var_34 = var_1.unsign(var_10, var_33)



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = var_1.unsign(var_3)
    var_6 = True
    var_7 = 60
    var_8 = var_1.unsign(var_3, var_7)
    var_9 = 0
    var_10 = var_1.unsign(var_3, var_9)
    var_11 = -1
    var_12 = var_3[:var_11]
    var_13 = b'A'
    var_14 = var_12 + var_13
    var_15 = var_1.unsign(var_14)
    var_16 = b'test_value'
    var_17 = var_1.unsign(var_16)
    var_18 = b'test_value'
    var_19 = b':'
    var_20 = var_18 + var_19
    var_21 = module_1.int_to_bytes(var_4)
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_20 + var_22
    var_24 = b':bad_signature'
    var_25 = var_23 + var_24
    var_26 = var_1.unsign(var_25)
    var_27 = 100
    var_28 = var_4 + var_27
    var_29 = var_1.sign(var_2)
    var_30 = 0
    var_31 = var_1.unsign(var_29, var_30)



# Parsed testcases at query #40
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
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'malformed_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'test_value'
    var_13 = var_1.sep
    var_14 = var_12 + var_13
    var_15 = b'invalid_timestamp'
    var_16 = var_14 + var_15
    var_17 = var_1.unsign(var_16)
    var_18 = -1
    var_19 = var_1.unsign(var_3, var_18)



