####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.default_signer
    var_2 = var_0.default_signer
    var_3 = 'custom-secret'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = '|'
    var_6 = module_0.TimedSerializer()
    var_7 = 'json'
    var_8 = module_0.TimedSerializer(serializer=var_7)
    var_9 = 'sha256'
    var_10 = module_0.TimedSerializer()
    var_11 = 'custom-salt'
    var_12 = module_0.TimedSerializer(var_11)
    var_13 = 'salt1'
    var_14 = 'salt2'
    var_15 = [var_13, var_14]
    var_16 = module_0.TimedSerializer()
    var_17 = 'hmac'
    var_18 = module_0.TimedSerializer()
    var_19 = 'derivation-salt'
    var_20 = module_0.TimedSerializer()



# Parsed testcases at query #2
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
    var_7 = 100
    var_8 = var_1.unsign(var_3, var_7)
    var_9 = module_0.TimestampSigner(var_0)
    var_10 = 200
    var_11 = var_4 + var_10
    var_12 = 100
    var_13 = var_9.unsign(var_3, var_12)
    var_14 = module_0.TimestampSigner(var_12)
    var_15 = 100
    var_16 = var_4 - var_15
    var_17 = 100
    var_18 = var_14.unsign(var_3, var_17)
    var_19 = b'test_value:malformed_timestamp:signature'
    var_20 = var_1.unsign(var_19)
    var_21 = b'test_value:signature'
    var_22 = var_1.unsign(var_21)
    var_23 = b'test_value:'
    var_24 = module_1.int_to_bytes(var_4)
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_23 + var_25
    var_27 = b':badsig'
    var_28 = var_26 + var_27
    var_29 = var_1.unsign(var_28)



# Parsed testcases at query #3
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
    var_30 = var_1.get_timestamp()
    var_31 = 100
    var_32 = var_30 + var_31
    var_33 = var_2 + var_10
    var_34 = module_1.int_to_bytes(var_32)
    var_35 = module_1.base64_encode(var_34)
    var_36 = var_33 + var_35
    var_37 = var_36 + var_10
    var_38 = var_2 + var_10
    var_39 = module_1.int_to_bytes(var_32)
    var_40 = module_1.base64_encode(var_39)
    var_41 = var_38 + var_40



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.default_signer
    var_2 = var_0.default_signer
    var_3 = 'test_key'
    var_4 = 'test_salt'
    var_5 = module_0.TimedSerializer(var_3, var_4)



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test_value'
    var_3 = 10
    var_4 = var_1.sign(var_2)
    var_5 = var_1.unsign(var_4)
    var_6 = True
    var_7 = var_1.unsign(var_4, var_3)
    var_8 = 2
    var_9 = 0
    var_10 = var_1.unsign(var_4, var_9)
    var_11 = b':'
    var_12 = var_2 + var_11
    var_13 = b'invalid_timestamp'
    var_14 = var_12 + var_13
    var_15 = var_14 + var_11
    var_16 = b'signature'
    var_17 = var_15 + var_16
    var_18 = var_1.unsign(var_17)
    var_19 = var_2 + var_11
    var_20 = var_19 + var_16
    var_21 = var_1.unsign(var_20)
    var_22 = var_2 + var_11
    var_23 = var_1.get_timestamp()
    var_24 = module_1.int_to_bytes(var_23)
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_22 + var_25
    var_27 = var_26 + var_11
    var_28 = b'bad_signature'
    var_29 = var_27 + var_28
    var_30 = var_1.unsign(var_29)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = '|'
    var_2 = module_0.TimedSerializer()
    var_3 = 'my-secret-key'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = 'my-salt'
    var_6 = module_0.TimedSerializer(var_5)
    var_7 = 'sha256'
    var_8 = module_0.TimedSerializer()



# Parsed testcases at query #7
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
    var_7 = 'invalid-signature'
    var_8 = var_1.loads(var_7)
    var_9 = 'custom-salt'



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)
    var_6 = 'test-salt'
    var_7 = module_0.TimestampSigner(var_6)
    var_8 = module_0.TimestampSigner(var_6, var_1)



# Parsed testcases at query #9
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
    var_7 = 2
    var_8 = module_1.base64_encode(var_0)
    var_9 = var_3 + var_8
    var_10 = var_9 + var_6
    var_11 = b'invalid_signature'
    var_12 = var_1.loads(var_11)
    var_13 = 'secret-key'
    var_14 = module_0.TimestampSigner(var_13)
    var_15 = b'value'
    var_16 = var_15 + var_12
    var_17 = var_16 + var_9
    var_18 = var_1.loads(var_17)
    var_19 = b'value'
    var_20 = b'malformed_timestamp'
    var_21 = var_19 + var_13
    var_22 = var_21 + var_20
    var_23 = var_22 + var_9
    var_24 = var_19 + var_10
    var_25 = var_24 + var_20
    var_26 = var_1.loads(var_17)



# Parsed testcases at query #10
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
    var_9 = 'invalid'
    var_10 = var_1.loads(var_9)
    var_11 = 'signer'
    var_12 = 0



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.signer as module_1

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
    var_9 = b'invalid'
    var_10 = module_0.TimedSerializer(var_0)
    var_11 = -1
    var_12 = module_1.Signer(var_0)



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.default_signer
    var_2 = var_0.default_signer



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 3600
    var_5 = True
    var_6 = 'test_salt'
    var_7 = 2
    var_8 = 1
    var_9 = b'invalid_data'
    var_10 = var_0.loads(var_9)



# Parsed testcases at query #16
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
    var_7 = 1
    var_8 = 'invalid-signature'
    var_9 = var_1.loads(var_8)
    var_10 = 'test-salt'
    var_11 = module_0.TimedSerializer(var_8, var_10)
    var_12 = 'invalid-signature'



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
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = 2000
    var_9 = b':'
    var_10 = var_2 + var_9
    var_11 = var_2 + var_9
    var_12 = 1000
    var_13 = b':invalid_timestamp:signature'
    var_14 = var_2 + var_13
    var_15 = var_1.unsign(var_14)
    var_16 = b':signature'
    var_17 = var_2 + var_16
    var_18 = var_1.unsign(var_17)
    var_19 = b':timestamp:wrong_signature'
    var_20 = var_2 + var_19
    var_21 = var_1.unsign(var_20)



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'hello world'
    assert var_2 == b'hello world'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = var_1.sign(var_2)
    var_7 = 2
    var_8 = var_1.unsign(var_6, var_7)
    var_9 = 0
    var_10 = var_1.unsign(var_6, var_9)
    var_11 = b'invalid-signature'
    var_12 = var_1.unsign(var_11)
    var_13 = b'hello:world:invalid-timestamp'
    var_14 = var_1.unsign(var_13)
    var_15 = b'hello:world'
    var_16 = var_1.unsign(var_15)
    var_17 = -1
    var_18 = var_1.unsign(var_3, var_17)



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = module_0.TimestampSigner()
    var_2 = module_0.TimedSerializer(signer=var_1)



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = '|'
    var_2 = module_0.TimedSerializer()
    var_3 = module_1.Serializer()
    var_4 = module_0.TimedSerializer(serializer=var_3)
    var_5 = module_0.TimestampSigner()
    var_6 = module_0.TimedSerializer(signer=var_5)
    var_7 = 'custom_salt'
    var_8 = module_0.TimedSerializer(var_7)



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
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'malformed_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'test_value:missing_timestamp'
    var_13 = var_1.unsign(var_12)
    var_14 = b'test_value:malformed_ts:signature'
    var_15 = var_1.unsign(var_14)
    var_16 = module_0.TimestampSigner(var_15)
    var_17 = 100
    var_18 = var_16.sign(var_2)
    var_19 = 50
    var_20 = var_1.unsign(var_18, var_19)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.default_signer
    var_2 = var_0.default_signer
    var_3 = 'custom_separator'
    var_4 = module_0.TimedSerializer()
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = [var_7]
    var_9 = module_0.TimedSerializer()
    var_10 = {var_5: var_6}



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = '|'
    var_2 = module_0.TimedSerializer()
    var_3 = 'json'
    var_4 = module_0.TimedSerializer(serializer=var_3)
    var_5 = 'test-salt'
    var_6 = module_0.TimedSerializer(var_5)
    var_7 = 'test-secret'
    var_8 = module_0.TimedSerializer(var_7)
    var_9 = 'sha256'
    var_10 = module_0.TimedSerializer()



# Parsed testcases at query #24
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
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'|'
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



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = True
    var_7 = 'salt'
    var_8 = 2
    var_9 = 1
    var_10 = 'invalid_signature'
    var_11 = var_1.loads(var_10)
    var_12 = module_0.TimestampSigner(var_0)
    var_13 = 'data'
    var_14 = var_12.sign(var_13)
    var_15 = b'extra_data'
    var_16 = var_14 + var_15
    var_17 = var_1.loads(var_16)
    var_18 = module_0.TimestampSigner(var_0)
    var_19 = 'data'
    var_20 = var_18.sign(var_19)
    var_21 = b'='
    var_22 = b''
    var_23 = var_1.loads(var_16)



# Parsed testcases at query #26
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
    var_7 = 'test-salt'
    var_8 = b'.'
    var_9 = module_1.base64_encode(var_8)
    var_10 = 0
    var_11 = b'.'
    var_12 = var_3 + var_11
    var_13 = var_12 + var_9
    var_14 = var_13 + var_11
    var_15 = 2
    var_16 = 'invalid_signed_data'
    var_17 = var_1.loads(var_16)
    var_18 = 'salt1'
    var_19 = 'salt2'
    var_20 = 'invalid_signed_data'



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = var_1.unsign(var_3)
    var_6 = module_1.want_bytes(var_2)
    var_7 = True
    var_8 = module_1.want_bytes(var_2)
    var_9 = 3600
    var_10 = var_1.unsign(var_3, var_9)
    var_11 = module_0.TimestampSigner(var_0)
    var_12 = 3601
    var_13 = var_4 - var_12
    var_14 = var_11.sign(var_2)
    var_15 = 3600
    var_16 = var_1.unsign(var_14, var_15)
    var_17 = module_0.TimestampSigner(var_15)
    var_18 = var_4 + var_12
    var_19 = var_17.sign(var_2)
    var_20 = var_1.unsign(var_19)
    var_21 = b'test_value:malformed_timestamp:signature'
    var_22 = var_1.unsign(var_21)
    var_23 = b'test_value:signature'
    var_24 = var_1.unsign(var_23)
    var_25 = b'test_value:timestamp:bad_signature'
    var_26 = var_1.unsign(var_25)



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
    var_7 = 'salt'
    var_8 = -10
    var_9 = serializer.dumps(var_4)[:var_8]
    var_10 = -10
    var_11 = serializer.dumps(var_4)[var_10:]
    var_12 = 1
    var_13 = 'invalid-signature'
    var_14 = var_1.loads(var_13)
    var_15 = 'wrong-salt'



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.get_timestamp()
    var_5 = var_1.unsign(var_3)
    assert var_5 == b'test-value'
    var_6 = True
    var_7 = 3600
    var_8 = var_1.unsign(var_3, var_7)
    assert var_8 == b'test-value'
    var_9 = 0
    var_10 = var_1.unsign(var_3, var_9)
    var_11 = -1
    var_12 = var_1.unsign(var_3, var_11)
    var_13 = b'invalid-signature'
    var_14 = var_1.unsign(var_13)
    var_15 = b'test-value'
    var_16 = var_1.sep
    var_17 = var_15 + var_16
    var_18 = b'invalid-timestamp'
    var_19 = var_17 + var_18
    var_20 = var_1.sep
    var_21 = var_19 + var_20
    var_22 = b'signature'
    var_23 = var_21 + var_22
    var_24 = var_1.unsign(var_23)
    var_25 = var_1.sep
    var_26 = var_15 + var_25
    var_27 = var_26 + var_22
    var_28 = var_1.unsign(var_27)



# Parsed testcases at query #30
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
    var_8 = var_1.sign(var_2)
    var_9 = 2
    var_10 = 0
    var_11 = 100
    var_12 = var_1.unsign(var_8, var_11)
    var_13 = b'test_value'
    var_14 = b'malformed'
    var_15 = b'signature'
    var_16 = var_1.get_timestamp()
    var_17 = module_1.int_to_bytes(var_16)
    var_18 = module_1.base64_encode(var_17)
    var_19 = b'bad_signature'



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



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
    var_7 = 60
    var_8 = var_1.unsign(var_3, var_7)
    var_9 = -1
    var_10 = var_1.unsign(var_3, var_9)
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
    var_22 = var_2 + var_11
    var_23 = module_1.int_to_bytes(var_4)
    var_24 = module_1.base64_encode(var_23)
    var_25 = var_22 + var_24
    var_26 = var_25 + var_11
    var_27 = b'bad_signature'
    var_28 = var_26 + var_27
    var_29 = var_1.unsign(var_28)
    var_30 = 'utf-8'



# Parsed testcases at query #33
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
    var_10 = b'invalid_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = var_1.sep
    var_13 = 0
    var_14 = var_1.sep
    var_15 = b'tampered'
    var_16 = var_1.sep
    var_17 = 2
    var_18 = var_1.sep
    var_19 = var_2 + var_18
    var_20 = b'signature'
    var_21 = var_19 + var_20
    var_22 = var_1.unsign(var_21)
    var_23 = b'malformed'
    var_24 = module_1.base64_encode(var_23)
    var_25 = var_1.sep
    var_26 = var_2 + var_25
    var_27 = var_26 + var_24
    var_28 = var_1.sep
    var_29 = var_27 + var_28
    var_30 = b'signature'
    var_31 = var_29 + var_30
    var_32 = var_1.unsign(var_31)



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = var_0.get_timestamp
    var_4 = callable(var_3)
    var_5 = var_0.timestamp_to_datetime
    var_6 = callable(var_5)



# Parsed testcases at query #35
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
    var_8 = var_1.sign(var_2)
    var_9 = var_1.get_timestamp
    var_10 = var_1.unsign(var_8, var_6)
    var_11 = b'test_value:malformed_timestamp:signature'
    var_12 = var_1.unsign(var_11)
    var_13 = b'test_value:signature'
    var_14 = var_1.unsign(var_13)
    var_15 = b'test_value:timestamp:bad_signature'
    var_16 = var_1.unsign(var_15)



# Parsed testcases at query #36
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
    var_15 = var_2 + var_10
    var_16 = var_15 + var_12
    var_17 = var_2 + var_10
    var_18 = var_2 + var_10
    var_19 = var_1.get_timestamp()
    var_20 = module_1.int_to_bytes(var_19)
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_18 + var_21
    var_23 = var_22 + var_10
    var_24 = b'bad_signature'
    var_25 = var_23 + var_24
    var_26 = var_1.unsign(var_25)



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = -1
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'invalid-signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b':'
    var_13 = var_2 + var_12
    var_14 = b'invalid-timestamp'
    var_15 = var_13 + var_14
    var_16 = var_15 + var_12
    var_17 = b'signature'
    var_18 = var_16 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = var_2 + var_12
    var_21 = var_20 + var_17
    var_22 = var_1.unsign(var_21)



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



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = ':'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = 'sha256'
    var_6 = module_0.TimestampSigner(digest_method=var_5)



# Parsed testcases at query #41
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
    var_10 = b'invalid_signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b':'
    var_13 = var_2 + var_12
    var_14 = b'invalid_timestamp'
    var_15 = var_13 + var_14
    var_16 = var_15 + var_12
    var_17 = b'invalid_sig'
    var_18 = var_16 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = var_2 + var_12
    var_21 = var_20 + var_17
    var_22 = var_1.unsign(var_21)



# Parsed testcases at query #42
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
    var_9 = 1000
    var_10 = var_1.unsign(var_3, var_9)
    var_11 = module_0.TimestampSigner(var_0)
    var_12 = 0
    var_13 = var_11.sign(var_2)
    var_14 = 1
    var_15 = var_1.unsign(var_13, var_14)
    var_16 = b'test_value:malformed_timestamp:signature'
    var_17 = var_1.unsign(var_16)
    var_18 = b'test_value:signature'
    var_19 = var_1.unsign(var_18)
    var_20 = b'test_value:timestamp:bad_signature'
    var_21 = var_1.unsign(var_20)
    var_22 = module_0.TimestampSigner(var_21)
    var_23 = 9999999999
    var_24 = var_22.sign(var_2)
    var_25 = 100
    var_26 = var_1.unsign(var_24, var_25)



# Parsed testcases at query #43
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
    var_8 = var_1.sign(var_2)
    var_9 = var_1.sep
    var_10 = 0
    var_11 = var_1.sep
    var_12 = var_1.sep
    var_13 = 2
    var_14 = 10
    var_15 = var_1.unsign(var_8, var_14)
    var_16 = b'test'
    var_17 = var_1.sep
    var_18 = var_16 + var_17
    var_19 = b'malformed'
    var_20 = var_18 + var_19
    var_21 = var_1.sep
    var_22 = var_20 + var_21
    var_23 = b'sig'
    var_24 = var_22 + var_23
    var_25 = var_1.unsign(var_24)
    var_26 = var_1.sep
    var_27 = var_16 + var_26
    var_28 = var_27 + var_23
    var_29 = var_1.unsign(var_28)
    var_30 = var_1.sep
    var_31 = var_16 + var_30
    var_32 = var_1.sep
    var_33 = b'bad_sig'



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = '|'
    var_2 = module_0.TimedSerializer()
    var_3 = 'json'
    var_4 = 'pickle'
    var_5 = [var_3, var_4]
    var_6 = module_0.TimedSerializer()
    var_7 = 'my-secret-key'
    var_8 = module_0.TimedSerializer(var_7)
    var_9 = 'my-salt'
    var_10 = module_0.TimedSerializer(var_9)
    var_11 = 'sha256'
    var_12 = module_0.TimedSerializer()
    var_13 = 'hmac'
    var_14 = module_0.TimedSerializer()



# Parsed testcases at query #45
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
    var_8 = 'signer'
    var_9 = 1
    var_10 = 'bad-signature'
    var_11 = var_1.loads(var_10)
    var_12 = -10
    var_13 = serializer.dumps(var_4)[:var_12]
    var_14 = var_1.loads(var_13)
    var_15 = -5
    var_16 = serializer.dumps(var_4)[:var_15]
    var_17 = b'badts'
    var_18 = var_16 + var_17
    var_19 = var_1.loads(var_18)



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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
    var_7 = b':'
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = 1
    var_10 = b'invalid:signature'
    var_11 = var_1.loads(var_10)
    var_12 = b'missing:timestamp'
    var_13 = var_1.loads(var_12)



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_salt'
    var_6 = 10
    var_7 = True
    var_8 = 0
    var_9 = 'invalid_signature'
    var_10 = var_1.loads(var_9, salt=var_5)
    var_11 = 'wrong_salt'



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'my-secret-key'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = 'another-key'
    var_4 = ':'
    var_5 = 'sha256'



# Parsed testcases at query #50
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



# Parsed testcases at query #51
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
    var_12 = b'value'
    var_13 = var_1.sep
    var_14 = var_12 + var_13
    var_15 = b'signature'
    var_16 = var_14 + var_15
    var_17 = var_1.unsign(var_16)
    var_18 = b'value'
    var_19 = var_1.sep
    var_20 = var_18 + var_19
    var_21 = b'malformed_ts'
    var_22 = var_20 + var_21
    var_23 = var_1.sep
    var_24 = var_22 + var_23
    var_25 = b'signature'
    var_26 = var_24 + var_25
    var_27 = var_1.unsign(var_26)
    var_28 = module_0.TimestampSigner(var_27)
    var_29 = 100
    var_30 = var_28.sign(var_2)
    var_31 = 50
    var_32 = var_1.unsign(var_30, var_31)



# Parsed testcases at query #52
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



# Parsed testcases at query #53
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.default_signer
    var_2 = var_0.default_signer
    var_3 = 'test_key'
    var_4 = 'test_salt'
    var_5 = module_0.TimedSerializer(var_3, var_4)



# Parsed testcases at query #54
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = 3601
    var_10 = var_8.sign(var_2)
    var_11 = 3600
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = b':'
    var_14 = var_2 + var_13
    var_15 = b'malformed-timestamp'
    var_16 = var_14 + var_15
    var_17 = var_16 + var_13
    var_18 = var_2 + var_13
    var_19 = var_18 + var_15
    var_20 = var_2 + var_13
    var_21 = var_2 + var_13
    var_22 = var_1.get_timestamp()
    var_23 = module_1.int_to_bytes(var_22)
    var_24 = module_1.base64_encode(var_23)
    var_25 = var_21 + var_24
    var_26 = var_25 + var_13
    var_27 = b'bad-signature'
    var_28 = var_26 + var_27
    var_29 = var_1.unsign(var_28)



# Parsed testcases at query #55
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



# Parsed testcases at query #56
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.default_signer
    var_2 = var_0.default_signer
    var_3 = ':'
    var_4 = module_0.TimedSerializer()
    var_5 = 'json'
    var_6 = module_0.TimedSerializer(serializer=var_5)
    var_7 = 'custom_salt'
    var_8 = module_0.TimedSerializer(var_7)
    var_9 = 'salt1'
    var_10 = 'salt2'
    var_11 = [var_9, var_10]
    var_12 = module_0.TimedSerializer()



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
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
    var_5 = 3600
    var_6 = True
    var_7 = 'salt'
    var_8 = 0
    var_9 = 'invalid_signature'
    var_10 = var_1.loads(var_9)
    var_11 = 'wrong_salt'



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'test-secret-key'
    var_4 = module_0.TimestampSigner(var_3)
    var_5 = module_0.TimestampSigner(var_3, sep=var_1)



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = 'test-salt'
    var_6 = module_0.TimestampSigner(var_5)
    var_7 = module_0.TimestampSigner(var_5)
    var_8 = module_0.TimestampSigner(var_5, var_1)



# Parsed testcases at query #5
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
    var_8 = 4000
    var_9 = 3600
    var_10 = 'invalid-signature'
    var_11 = var_1.loads(var_10)
    var_12 = 'wrong-key'
    var_13 = module_0.TimedSerializer(var_12)



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = b'my-secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #7
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
    var_9 = b'invalid-signature'
    var_10 = var_1.loads(var_9)
    var_11 = 1
    var_12 = 'custom-salt'
    var_13 = 'wrong-salt'



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = '|'
    var_2 = module_0.TimedSerializer()
    var_3 = 'my-secret-key'
    var_4 = module_0.TimedSerializer(var_3)
    var_5 = module_0.TimedSerializer(var_3)
    var_6 = 'json'
    var_7 = module_0.TimedSerializer(serializer=var_6)



# Parsed testcases at query #9
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
    var_10 = b':'
    var_11 = var_2 + var_10
    var_12 = var_2 + var_10
    var_13 = 1
    var_14 = b':invalid_timestamp:signature'
    var_15 = var_2 + var_14
    var_16 = var_1.unsign(var_15)
    var_17 = b':signature'
    var_18 = var_2 + var_17
    var_19 = var_1.unsign(var_18)
    var_20 = b':timestamp:bad_signature'
    var_21 = var_2 + var_20
    var_22 = var_1.unsign(var_21)



# Parsed testcases at query #10
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
    var_8 = 1000
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



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = 0
    var_7 = True
    var_8 = 20
    var_9 = b'invalid'
    var_10 = 'salt'
    var_11 = 'wrong_salt'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.default_signer
    var_2 = var_0.default_signer
    var_3 = '.'
    var_4 = module_0.TimedSerializer()
    var_5 = 'json'
    var_6 = module_0.TimedSerializer(serializer=var_5)
    var_7 = 'custom_salt'
    var_8 = module_0.TimedSerializer(var_7)
    var_9 = 'secret'
    var_10 = module_0.TimedSerializer(var_9)
    var_11 = 'sha256'
    var_12 = module_0.TimedSerializer()



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()



# Parsed testcases at query #14
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
    var_12 = b'value'
    var_13 = var_12 + var_11
    var_14 = b'signature'
    var_15 = var_13 + var_14
    var_16 = var_1.unsign(var_15)
    var_17 = var_2 + var_15
    var_18 = b'malformed_ts'
    var_19 = var_17 + var_18
    var_20 = b'signature'
    var_21 = 100
    var_22 = 50



# Parsed testcases at query #15
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
    var_12 = -1
    var_13 = b'x'
    var_14 = 'custom-salt'
    var_15 = 'wrong-salt'



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.serializer as module_1

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.default_signer
    var_2 = var_0.default_signer
    var_3 = 'custom_separator'
    var_4 = module_0.TimedSerializer()
    var_5 = module_1.Serializer()
    var_6 = module_0.TimedSerializer(serializer=var_5)
    var_7 = module_0.TimestampSigner()
    var_8 = module_0.TimedSerializer(signer=var_7)
    var_9 = module_0.TimedSerializer(serializer=var_5, signer=var_7)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimedSerializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = 1
    var_7 = 1
    var_8 = True
    var_9 = 'salt'
    var_10 = 'bad-signature'
    var_11 = var_1.loads(var_10)
    var_12 = 'signer'
    var_13 = 1
    var_14 = 'salt1'
    var_15 = 'salt2'
    var_16 = [var_14, var_15]
    var_17 = module_0.TimedSerializer(var_13)



# Parsed testcases at query #18
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
    var_9 = 1
    var_10 = var_1.unsign(var_8, var_9)
    var_11 = var_1.sign(var_2)
    var_12 = var_1.unsign(var_11)
    var_13 = b':'
    var_14 = var_2 + var_13
    var_15 = b'invalid_timestamp'
    var_16 = var_14 + var_15
    var_17 = var_16 + var_13
    var_18 = b'signature'
    var_19 = var_17 + var_18
    var_20 = var_1.unsign(var_19)
    var_21 = var_2 + var_13
    var_22 = var_21 + var_18
    var_23 = var_1.unsign(var_22)
    var_24 = -1
    var_25 = var_3[:var_24]
    var_26 = b'x'
    var_27 = var_25 + var_26
    var_28 = var_1.unsign(var_27)



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = True
    var_6 = 3600
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'malformed-signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'test-value:missing-timestamp'
    var_13 = var_1.unsign(var_12)
    var_14 = b'test-value:malformed-timestamp:signature'
    var_15 = var_1.unsign(var_14)
    var_16 = module_0.TimestampSigner(var_14)
    var_17 = 100
    var_18 = var_16.sign(var_2)
    var_19 = 3600
    var_20 = var_1.unsign(var_18, var_19)



# Parsed testcases at query #20
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
    var_8 = var_1.get_timestamp()
    var_9 = var_8 - var_6
    var_10 = module_1.int_to_bytes(var_9)
    var_11 = b':'
    var_12 = var_2 + var_11
    var_13 = module_1.base64_encode(var_10)
    var_14 = var_12 + var_13
    var_15 = var_14 + var_11
    var_16 = var_2 + var_11
    var_17 = module_1.base64_encode(var_10)
    var_18 = var_16 + var_17
    var_19 = 1
    var_20 = var_2 + var_11
    var_21 = b'malformed'
    var_22 = var_20 + var_21
    var_23 = var_22 + var_11
    var_24 = var_2 + var_11
    var_25 = var_24 + var_21
    var_26 = var_2 + var_11
    var_27 = var_2 + var_11
    var_28 = var_1.get_timestamp()
    var_29 = module_1.int_to_bytes(var_28)
    var_30 = module_1.base64_encode(var_29)
    var_31 = var_27 + var_30
    var_32 = var_31 + var_11
    var_33 = b'bad_signature'
    var_34 = var_32 + var_33
    var_35 = var_1.unsign(var_34)



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
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = b'test_value:malformed_timestamp:signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'test_value:signature'
    var_13 = var_1.unsign(var_12)
    var_14 = b'test_value:timestamp:bad_signature'
    var_15 = var_1.unsign(var_14)



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimestampSigner()
    var_1 = b'|'
    var_2 = module_0.TimestampSigner(sep=var_1)
    var_3 = 'secret-key'
    var_4 = module_0.TimestampSigner()
    var_5 = module_0.TimestampSigner(sep=var_1)



# Parsed testcases at query #24
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
    var_10 = 'salt'
    var_11 = 'wrong-salt'



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.default_signer
    var_2 = var_0.default_signer



# Parsed testcases at query #26
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
    var_9 = -1
    var_10 = var_1.unsign(var_3, var_9)
    var_11 = b':'
    var_12 = var_2 + var_11
    var_13 = b'malformed'
    var_14 = var_12 + var_13
    var_15 = var_14 + var_11
    var_16 = b'signature'
    var_17 = var_15 + var_16
    var_18 = var_1.unsign(var_17)
    var_19 = var_2 + var_11
    var_20 = var_19 + var_16
    var_21 = var_1.unsign(var_20)
    var_22 = var_2 + var_11
    var_23 = module_1.int_to_bytes(var_4)
    var_24 = module_1.base64_encode(var_23)
    var_25 = var_22 + var_24
    var_26 = var_25 + var_11
    var_27 = b'bad_sig'
    var_28 = var_26 + var_27
    var_29 = var_1.unsign(var_28)



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.timed as module_0

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
    var_18 = var_1.unsign(var_17)
    var_19 = b'test_value:malformed_timestamp:signature'
    var_20 = var_1.unsign(var_19)
    var_21 = b'test_value:signature'
    var_22 = var_1.unsign(var_21)
    var_23 = b'test_value:timestamp:bad_signature'
    var_24 = var_1.unsign(var_23)



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.timed as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.TimestampSigner(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = module_1.want_bytes(var_2)
    var_6 = True
    var_7 = module_1.want_bytes(var_2)
    var_8 = 1000
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = module_0.TimestampSigner(var_0)
    var_11 = 2000
    var_12 = var_10.sign(var_2)
    var_13 = 1000
    var_14 = var_1.unsign(var_12, var_13)
    var_15 = module_0.TimestampSigner(var_13)
    var_16 = var_15.sign(var_2)
    var_17 = var_1.unsign(var_16)
    var_18 = b'test-value:malformed-timestamp:signature'
    var_19 = var_1.unsign(var_18)
    var_20 = b'test-value:signature'
    var_21 = var_1.unsign(var_20)
    var_22 = b'test-value:timestamp:wrong-signature'
    var_23 = var_1.unsign(var_22)



# Parsed testcases at query #30
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
    var_8 = b':'
    var_9 = var_2 + var_8
    var_10 = var_2 + var_8
    var_11 = 1
    var_12 = var_2 + var_8
    var_13 = b'invalid_timestamp'
    var_14 = var_12 + var_13
    var_15 = var_14 + var_8
    var_16 = b'signature'
    var_17 = var_15 + var_16
    var_18 = var_1.unsign(var_17)
    var_19 = var_2 + var_8
    var_20 = var_19 + var_16
    var_21 = var_1.unsign(var_20)
    var_22 = var_2 + var_8
    var_23 = var_1.get_timestamp()
    var_24 = module_1.int_to_bytes(var_23)
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_22 + var_25
    var_27 = var_26 + var_8
    var_28 = b'bad_signature'
    var_29 = var_27 + var_28
    var_30 = var_1.unsign(var_29)



# Parsed testcases at query #31
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
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = 'invalid-signature'
    var_11 = var_1.unsign(var_10)
    var_12 = b'test-value:malformed-timestamp:signature'
    var_13 = var_1.unsign(var_12)
    var_14 = b'test-value:signature'
    var_15 = var_1.unsign(var_14)



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = var_0.default_signer
    var_2 = var_0.default_signer
    var_3 = 'test-secret-key'
    var_4 = 'test-salt'
    var_5 = module_0.TimedSerializer(var_3, var_4)



# Parsed testcases at query #33
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
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    assert var_7 == b'test_value'
    var_8 = module_0.TimestampSigner(var_0)
    var_9 = 2000
    var_10 = var_8.sign(var_2)
    var_11 = 1000
    var_12 = var_1.unsign(var_10, var_11)
    var_13 = b'test_value:malformed_timestamp:signature'
    var_14 = var_1.unsign(var_13)
    var_15 = b'test_value:signature'
    var_16 = var_1.unsign(var_15)
    var_17 = b'test_value:timestamp:bad_signature'
    var_18 = var_1.unsign(var_17)



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.timed as module_0

def test_case_0():
    var_0 = module_0.TimedSerializer()
    var_1 = 'test-secret'
    var_2 = module_0.TimedSerializer(var_1)
    var_3 = ':'
    var_4 = module_0.TimedSerializer()
    var_5 = 'test-salt'
    var_6 = module_0.TimedSerializer(var_5)
    var_7 = 10
    var_8 = module_0.TimedSerializer()
    var_9 = 'hmac'
    var_10 = module_0.TimedSerializer()



# Parsed testcases at query #35
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
    var_9 = 'invalid'
    var_10 = var_1.loads(var_9)
    var_11 = -1
    var_12 = b'x'
    var_13 = 'x'



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
    var_6 = 1000
    var_7 = var_1.unsign(var_3, var_6)
    var_8 = 0
    var_9 = var_1.unsign(var_3, var_8)
    var_10 = -1
    var_11 = var_3[:var_10]
    var_12 = b'X'
    var_13 = var_11 + var_12
    var_14 = var_1.unsign(var_13)
    var_15 = b':'
    var_16 = var_2 + var_15
    var_17 = b'malformed'
    var_18 = var_16 + var_17
    var_19 = var_18 + var_15
    var_20 = b'signature'
    var_21 = var_19 + var_20
    var_22 = var_1.unsign(var_21)
    var_23 = var_2 + var_15
    var_24 = var_23 + var_20
    var_25 = var_1.unsign(var_24)



# Parsed testcases at query #37
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
    var_9 = 'invalid'
    var_10 = var_1.loads(var_9)
    var_11 = 'signer'
    var_12 = 0



# Parsed testcases at query #38
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



