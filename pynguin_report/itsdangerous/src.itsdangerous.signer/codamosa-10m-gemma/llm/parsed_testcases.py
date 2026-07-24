####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'.'
    var_3 = b'hello-world'
    var_4 = module_0.Signer(var_0, var_1, var_2)
    var_5 = var_4.sign(var_3)
    var_6 = var_4.unsign(var_5)
    var_7 = 'hello-world'
    var_8 = var_4.sign(var_7)
    var_9 = var_4.unsign(var_8)
    assert var_9 == b'hello-world'
    var_10 = b'no-separator-here'
    var_11 = var_4.unsign(var_10)
    var_12 = b'tampered-payload'
    var_13 = var_4.get_signature(var_3)
    var_14 = var_12 + var_2
    var_15 = var_14 + var_13
    var_16 = var_4.unsign(var_15)
    var_17 = var_3 + var_2
    var_18 = b'!!!'
    var_19 = var_17 + var_18
    var_20 = var_4.unsign(var_19)
    var_21 = b'old-secret'
    var_22 = b'new-secret'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23, var_1, var_2)
    var_25 = var_24.sign(var_3)
    var_26 = var_24.unsign(var_25)
    var_27 = var_24.derive_key(var_21)
    var_28 = var_3 + var_2
    var_29 = b'|'
    var_30 = module_0.Signer(var_0, var_1, var_29)
    var_31 = var_30.sign(var_3)
    var_32 = var_30.unsign(var_31)
    var_33 = b'A'
    var_34 = module_0.Signer(var_0, sep=var_33)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'.'
    var_3 = b'payload'
    var_4 = module_0.Signer(var_0, var_1, var_2)
    var_5 = var_4.sign(var_3)
    var_6 = var_4.unsign(var_5)
    var_7 = 'payload_string'
    var_8 = var_4.sign(var_7)
    var_9 = var_4.unsign(var_8)
    assert var_9 == b'payload_string'
    var_10 = b'nosetere'
    var_11 = var_4.unsign(var_10)
    var_12 = b'original_payload'
    var_13 = var_4.sign(var_12)
    var_14 = 1
    var_15 = b'tampered_payload'
    var_16 = var_15 + var_2
    var_17 = var_4.sign(var_12)
    var_18 = 0
    var_19 = 1
    var_20 = var_11 + var_19
    var_21 = 256
    var_22 = 0
    var_23 = b'old_secret'
    var_24 = b'new_secret'
    var_25 = [var_23, var_24]
    var_26 = module_0.Signer(var_25, var_1)
    var_27 = b'data'
    var_28 = var_26.sign(var_27)
    var_29 = var_26.unsign(var_28)
    assert var_29 == b'data'
    var_30 = var_26.derive_key(var_23)
    var_31 = var_27 + var_2
    var_32 = var_27 + var_2
    var_33 = b'!!!NotBase64!!!'
    var_34 = var_32 + var_33
    var_35 = var_4.unsign(var_34)



# Parsed testcases at query #3
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = b'other'
    var_6 = var_1 + var_5
    var_7 = module_0.digest()
    var_8 = 'django-concat'
    var_9 = b'signer'
    var_10 = var_1 + var_9
    var_11 = var_10 + var_0
    var_12 = module_0.digest()
    var_13 = var_1 + var_9
    var_14 = var_13 + var_5
    var_15 = module_0.digest()
    var_16 = 'hmac'
    var_17 = module_0.digest()
    var_18 = module_0.digest()
    var_19 = 'none'
    var_20 = b'old'
    var_21 = b'new'
    var_22 = [var_20, var_21]
    var_23 = module_1.Signer(var_22, var_1, key_derivation=var_19)
    var_24 = var_23.derive_key()
    assert var_24 == b'new'
    var_25 = var_23.derive_key(var_20)
    assert var_25 == b'old'
    var_26 = 'invalid_method'
    var_27 = module_1.Signer(var_0, key_derivation=var_26)
    var_28 = var_27.derive_key()



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Tests the verify_signature method of the Signer class.'
    var_1 = b'secret'
    var_2 = b'salt'
    var_3 = b'.'
    var_4 = b'payload'
    var_5 = module_0.Signer(var_1, var_2, var_3)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_4 + var_3
    var_8 = var_7 + var_6
    var_9 = var_5.verify_signature(var_4, var_6)
    assert var_9 is True
    var_10 = 1
    var_11 = signed_value_with_sig.split(var_3)[var_10]
    var_12 = var_5.verify_signature(var_4, var_11)
    assert var_12 is True
    var_13 = b'tampered_payload'
    var_14 = var_5.verify_signature(var_13, var_6)
    assert var_14 is False
    var_15 = b'wrong_sig'
    var_16 = module_1.base64_encode(var_15)
    var_17 = var_5.verify_signature(var_4, var_16)
    assert var_17 is False
    var_18 = b'old_secret'
    var_19 = b'new_secret'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20, var_2, var_3)
    var_22 = var_21.get_signature(var_4)
    var_23 = var_21.verify_signature(var_4, var_22)
    assert var_23 is True
    var_24 = var_21.derive_key(var_18)
    var_25 = module_0.HMACAlgorithm()
    var_26 = var_25.get_signature(var_24, var_4)
    var_27 = module_1.base64_encode(var_26)
    var_28 = var_21.verify_signature(var_4, var_27)
    assert var_28 is True
    var_29 = b'!!!NotBase64!!!'
    var_30 = var_5.verify_signature(var_4, var_29)
    assert var_30 is False
    var_31 = module_0.NoneAlgorithm()
    var_32 = module_0.Signer(var_1, algorithm=var_31)
    var_33 = b''
    var_34 = module_1.base64_encode(var_33)
    var_35 = var_32.verify_signature(var_4, var_34)
    assert var_35 is True



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_4 = var_1 + var_0
    var_5 = module_1.digest()
    var_6 = 'django-concat'
    var_7 = module_0.Signer(var_0, var_1, key_derivation=var_6)
    var_8 = b'signer'
    var_9 = var_1 + var_8
    var_10 = var_9 + var_0
    var_11 = module_1.digest()
    var_12 = 'hmac'
    var_13 = module_0.Signer(var_0, var_1, key_derivation=var_12)
    var_14 = module_1.digest()
    var_15 = 'none'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = b'alt_secret'
    var_18 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_19 = var_1 + var_17
    var_20 = module_1.digest()
    var_21 = var_18.derive_key(var_17)
    var_22 = b'old'
    var_23 = b'new'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24, var_1, key_derivation=var_15)
    var_26 = var_25.derive_key()
    assert var_26 == b'new'
    var_27 = 'invalid'
    var_28 = module_0.Signer(var_0, key_derivation=var_27)
    var_29 = var_28.derive_key()



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_4 = var_1 + var_0
    var_5 = module_1.digest()
    var_6 = 'django-concat'
    var_7 = module_0.Signer(var_0, var_1, key_derivation=var_6)
    var_8 = b'signer'
    var_9 = var_1 + var_8
    var_10 = var_9 + var_0
    var_11 = module_1.digest()
    var_12 = 'hmac'
    var_13 = module_0.Signer(var_0, var_1, key_derivation=var_12)
    var_14 = module_1.digest()
    var_15 = module_1.digest()
    var_16 = 'none'
    var_17 = module_0.Signer(var_0, var_1, key_derivation=var_16)
    var_18 = b'other'
    var_19 = var_1 + var_18
    var_20 = module_1.digest()
    var_21 = b'old'
    var_22 = b'new'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23, var_1, key_derivation=var_16)
    var_25 = var_24.derive_key(var_21)
    assert var_25 == b'old'
    var_26 = var_24.derive_key()
    assert var_26 == b'new'
    var_27 = 'invalid'
    var_28 = module_0.Signer(var_0, key_derivation=var_27)
    var_29 = var_28.derive_key()



# Parsed testcases at query #7
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = b'other'
    var_6 = var_1 + var_5
    var_7 = module_0.digest()
    var_8 = 'django-concat'
    var_9 = b'signer'
    var_10 = var_1 + var_9
    var_11 = var_10 + var_0
    var_12 = module_0.digest()
    var_13 = var_1 + var_9
    var_14 = var_13 + var_5
    var_15 = module_0.digest()
    var_16 = 'hmac'
    var_17 = module_0.digest()
    var_18 = module_0.digest()
    var_19 = 'none'
    var_20 = 'invalid'
    var_21 = module_1.Signer(var_0, key_derivation=var_20)
    var_22 = var_21.derive_key()
    var_23 = b'old'
    var_24 = b'new'
    var_25 = [var_23, var_24]
    var_26 = module_1.Signer(var_25, var_1, key_derivation=var_19)
    var_27 = var_26.derive_key()
    assert var_27 == b'new'
    var_28 = var_26.derive_key(var_23)
    assert var_28 == b'old'



# Parsed testcases at query #8
#--------------------------


import hmac as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'data'
    var_3 = b'.'
    var_4 = module_0.Signer(var_0, var_1, var_3)
    var_5 = var_4.get_signature(var_2)
    var_6 = var_2 + var_3
    var_7 = var_6 + var_5
    var_8 = 1
    var_9 = signed_payload.split(var_3)[var_8]
    var_10 = var_4.verify_signature(var_2, var_9)
    assert var_10 is True
    var_11 = b'bm90X3RoZV9zaWduYXR1cmU='
    var_12 = var_4.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = b'different_data'
    var_14 = var_4.verify_signature(var_13, var_9)
    assert var_14 is False
    var_15 = b'!!!not_base64!!!'
    var_16 = var_4.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = b'old_secret'
    var_18 = [var_17, var_0]
    var_19 = module_0.Signer(var_18, var_1)
    var_20 = var_19.derive_key(var_17)
    var_21 = module_0.HMACAlgorithm()
    var_22 = var_21.get_signature(var_20, var_2)
    var_23 = module_1.base64_encode(var_22)
    var_24 = var_19.verify_signature(var_2, var_23)
    assert var_24 is True
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_0, algorithm=var_25)
    var_27 = b''
    var_28 = var_26.verify_signature(var_2, var_27)
    assert var_28 is True
    var_29 = b'something'
    var_30 = var_26.verify_signature(var_2, var_29)
    assert var_30 is False
    var_31 = 'hmac'
    var_32 = module_0.Signer(var_0, var_1, key_derivation=var_31)
    var_33 = var_32.get_signature(var_2)
    var_34 = var_3 in var_33
    var_35 = sig_hmac.split(var_3)[var_8]
    var_36 = var_35 if var_34 else var_33
    var_37 = var_32.verify_signature(var_2, var_36)
    assert var_37 is True
    var_38 = b'|'
    var_39 = module_0.Signer(var_0, sep=var_38)
    var_40 = var_39.get_signature(var_2)
    var_41 = var_2 + var_38
    var_42 = var_41 + var_40



# Parsed testcases at query #10
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'none'
    var_3 = b'other'
    var_4 = 'concat'
    var_5 = var_1 + var_0
    var_6 = module_0.digest()
    var_7 = var_1 + var_3
    var_8 = module_0.digest()
    var_9 = 'django-concat'
    var_10 = b'signer'
    var_11 = var_1 + var_10
    var_12 = var_11 + var_0
    var_13 = module_0.digest()
    var_14 = var_1 + var_10
    var_15 = var_14 + var_3
    var_16 = module_0.digest()
    var_17 = 'hmac'
    var_18 = module_0.digest()
    var_19 = b'other_key'
    var_20 = module_0.digest()
    var_21 = b'old'
    var_22 = b'new'
    var_23 = [var_21, var_22]
    var_24 = module_1.Signer(var_23, var_1, key_derivation=var_2)
    var_25 = var_24.derive_key()
    assert var_25 == b'new'
    var_26 = var_24.derive_key(var_21)
    assert var_26 == b'old'
    var_27 = 'invalid_method'
    var_28 = module_1.Signer(var_0, key_derivation=var_27)
    var_29 = var_28.derive_key()



# Parsed testcases at query #11
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = 'django-concat'
    var_6 = b'signer'
    var_7 = var_1 + var_6
    var_8 = var_7 + var_0
    var_9 = module_0.digest()
    var_10 = 'hmac'
    var_11 = module_0.digest()
    var_12 = 'none'
    var_13 = b'alt_secret'
    var_14 = var_1 + var_13
    var_15 = module_0.digest()
    var_16 = 'invalid'
    var_17 = module_1.Signer(var_0, key_derivation=var_16)
    var_18 = var_17.derive_key()
    var_19 = b'old'
    var_20 = b'new'
    var_21 = [var_19, var_20]
    var_22 = module_1.Signer(var_21, var_1, key_derivation=var_12)
    var_23 = var_22.derive_key()
    assert var_23 == b'new'
    var_24 = var_22.derive_key(var_19)
    assert var_24 == b'old'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'hello-world'
    var_4 = var_2.sign(var_3)
    var_5 = 1
    var_6 = var_2.sep
    var_7 = signed_value.rsplit(var_6, var_5)[var_5]
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is True
    var_9 = b'hello-worle'
    var_10 = var_2.verify_signature(var_9, var_7)
    assert var_10 is False
    var_11 = module_1.base64_decode(var_7)
    var_12 = bytearray(var_11)
    var_13 = 0
    var_14 = var_12[var_13]
    var_15 = 255
    var_16 = var_14 ^ var_15
    var_17 = bytes(var_12)
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_2.verify_signature(var_3, var_18)
    assert var_19 is False
    var_20 = b'!!!not-base64!!!'
    var_21 = var_2.verify_signature(var_3, var_20)
    assert var_21 is False
    var_22 = b'old-key'
    var_23 = b'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24, var_1)
    var_26 = var_25.sign(var_3)
    var_27 = var_25.derive_key(var_22)
    var_28 = b'different-key'
    var_29 = module_0.Signer(var_28, var_1)
    var_30 = var_29.sign(var_3)
    var_31 = var_29.sep
    var_32 = var_15.rsplit(var_31, var_5)[var_5]
    var_33 = var_2.verify_signature(var_3, var_32)
    assert var_33 is False



# Parsed testcases at query #13
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = 'django-concat'
    var_6 = b'signer'
    var_7 = var_1 + var_6
    var_8 = var_7 + var_0
    var_9 = module_0.digest()
    var_10 = 'hmac'
    var_11 = module_0.digest()
    var_12 = 'none'
    var_13 = b'new_key'
    var_14 = var_1 + var_6
    var_15 = var_14 + var_13
    var_16 = module_0.digest()
    var_17 = b'old'
    var_18 = b'new'
    var_19 = [var_17, var_18]
    var_20 = module_1.Signer(var_19, var_1, key_derivation=var_12)
    var_21 = var_20.derive_key(var_17)
    assert var_21 == b'old'
    var_22 = var_20.derive_key(var_18)
    assert var_22 == b'new'
    var_23 = 'invalid_method'
    var_24 = module_1.Signer(var_0, key_derivation=var_23)
    var_25 = var_24.derive_key()



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'.'
    var_3 = b'payload'
    var_4 = module_0.Signer(var_0, var_1, var_2)
    var_5 = var_4.get_signature(var_3)
    var_6 = var_3 + var_2
    var_7 = var_6 + var_5
    var_8 = 1
    var_9 = signed_value.split(var_2)[var_8]
    var_10 = var_4.verify_signature(var_3, var_9)
    assert var_10 is True
    var_11 = b'bm90X3RoZV9zaWduYXR1cmU='
    var_12 = var_4.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = b'!!!'
    var_14 = var_4.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = b'old_secret'
    var_16 = b'new_secret'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17, var_1, var_2)
    var_19 = var_18.derive_key(var_15)
    var_20 = module_0.HMACAlgorithm()
    var_21 = var_20.get_signature(var_19, var_3)
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_18.verify_signature(var_3, var_22)
    assert var_23 is True
    var_24 = b'different_payload'
    var_25 = var_4.verify_signature(var_24, var_9)
    assert var_25 is False
    var_26 = b'valid_sig'
    var_27 = module_1.base64_encode(var_26)
    var_28 = b'wrong_value'



# Parsed testcases at query #15
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = b'other'
    var_6 = var_1 + var_5
    var_7 = module_0.digest()
    var_8 = 'django-concat'
    var_9 = b'signer'
    var_10 = var_1 + var_9
    var_11 = var_10 + var_0
    var_12 = module_0.digest()
    var_13 = var_1 + var_9
    var_14 = var_13 + var_5
    var_15 = module_0.digest()
    var_16 = 'hmac'
    var_17 = module_0.digest()
    var_18 = module_0.digest()
    var_19 = 'none'
    var_20 = module_1.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    var_22 = var_20.derive_key(var_5)
    assert var_22 == b'other'
    var_23 = b'old'
    var_24 = b'new'
    var_25 = [var_23, var_24]
    var_26 = module_1.Signer(var_25, var_1, key_derivation=var_19)
    var_27 = var_26.derive_key()
    assert var_27 == b'new'
    var_28 = var_26.derive_key(var_23)
    assert var_28 == b'old'
    var_29 = 'invalid_method'
    var_30 = module_1.Signer(var_0, key_derivation=var_29)
    var_31 = var_30.derive_key()



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'hello-world'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'wrong-value'
    var_7 = var_2.verify_signature(var_6, var_4)
    assert var_7 is False
    var_8 = -1
    var_9 = var_4[:var_8]
    var_10 = -1
    var_11 = var_4[var_10:]
    var_12 = b'A'
    var_13 = var_11 != var_12
    var_14 = b'B'
    var_15 = var_12 if var_13 else var_14
    var_16 = var_9 + var_15
    var_17 = var_2.verify_signature(var_3, var_16)
    assert var_17 is False
    var_18 = b'!!!not-base64!!!'
    var_19 = var_2.verify_signature(var_3, var_18)
    assert var_19 is False
    var_20 = b'old-key'
    var_21 = [var_20, var_0]
    var_22 = module_0.Signer(var_21, var_1)
    var_23 = var_22.get_signature(var_3)
    var_24 = var_2.verify_signature(var_3, var_23)
    assert var_24 is True
    var_25 = b'different-key'
    var_26 = module_0.Signer(var_25, var_1)
    var_27 = var_26.verify_signature(var_3, var_4)
    assert var_27 is False



# Parsed testcases at query #17
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest(var_3)
    var_5 = module_0.digest()
    var_6 = None
    var_7 = 'django-concat'
    var_8 = b'signer'
    var_9 = var_1 + var_8
    var_10 = var_9 + var_0
    var_11 = module_0.digest(var_10)
    var_12 = module_0.digest()
    var_13 = 'hmac'
    var_14 = module_0.digest()
    var_15 = 'none'
    var_16 = module_1.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = b'alt_secret'
    var_18 = var_1 + var_17
    var_19 = module_0.digest(var_18)
    var_20 = module_0.digest()
    var_21 = b'old'
    var_22 = b'new'
    var_23 = [var_21, var_22]
    var_24 = module_1.Signer(var_23, var_1, key_derivation=var_15)
    var_25 = var_24.derive_key(var_6)
    assert var_25 == b'new'
    var_26 = var_24.derive_key(var_21)
    assert var_26 == b'old'
    var_27 = 'unknown'
    var_28 = module_1.Signer(var_0, key_derivation=var_27)
    var_29 = var_28.derive_key()



# Parsed testcases at query #18
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = 'django-concat'
    var_6 = b'signer'
    var_7 = var_1 + var_6
    var_8 = var_7 + var_0
    var_9 = module_0.digest()
    var_10 = 'hmac'
    var_11 = module_0.digest()
    var_12 = 'none'
    var_13 = b'alternative'
    var_14 = var_1 + var_13
    var_15 = module_0.digest()
    var_16 = module_0.digest()
    var_17 = b'old'
    var_18 = b'new'
    var_19 = [var_17, var_18]
    var_20 = module_1.Signer(var_19, var_1, key_derivation=var_12)
    var_21 = var_20.derive_key()
    assert var_21 == b'new'
    var_22 = var_20.derive_key(var_17)
    assert var_22 == b'old'
    var_23 = 'invalid'
    var_24 = module_1.Signer(var_0, key_derivation=var_23)
    var_25 = var_24.derive_key()



# Parsed testcases at query #19
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'none'
    var_3 = b'other'
    var_4 = 'concat'
    var_5 = var_1 + var_0
    var_6 = module_0.digest()
    var_7 = var_1 + var_3
    var_8 = module_0.digest()
    var_9 = 'django-concat'
    var_10 = b'signer'
    var_11 = var_1 + var_10
    var_12 = var_11 + var_0
    var_13 = module_0.digest()
    var_14 = var_1 + var_10
    var_15 = var_14 + var_3
    var_16 = module_0.digest()
    var_17 = 'hmac'
    var_18 = module_0.digest()
    var_19 = b'another_secret'
    var_20 = module_0.digest()
    var_21 = b'old_key'
    var_22 = b'new_key'
    var_23 = [var_21, var_22]
    var_24 = module_1.Signer(var_23, var_1, key_derivation=var_2)
    var_25 = var_24.derive_key()
    assert var_25 == b'new_key'
    var_26 = var_24.derive_key(var_21)
    assert var_26 == b'old_key'
    var_27 = 'invalid_method'
    var_28 = module_1.Signer(var_0, key_derivation=var_27)
    var_29 = var_28.derive_key()



# Parsed testcases at query #20
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = 'django-concat'
    var_6 = b'signer'
    var_7 = var_1 + var_6
    var_8 = var_7 + var_0
    var_9 = module_0.digest()
    var_10 = 'hmac'
    var_11 = module_0.digest()
    var_12 = 'none'
    var_13 = b'other-key'
    var_14 = var_1 + var_13
    var_15 = module_0.digest()
    var_16 = 'invalid'
    var_17 = module_1.Signer(var_0, key_derivation=var_16)
    var_18 = var_17.derive_key()
    var_19 = b'old'
    var_20 = b'new'
    var_21 = [var_19, var_20]
    var_22 = module_1.Signer(var_21, var_1, key_derivation=var_12)
    var_23 = var_22.derive_key()
    assert var_23 == b'new'
    var_24 = var_22.derive_key(var_19)
    assert var_24 == b'old'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'payload'
    var_3 = b'.'
    var_4 = module_0.Signer(var_0, var_1, var_3)
    var_5 = var_4.get_signature(var_2)
    var_6 = var_2 + var_3
    var_7 = var_6 + var_5
    var_8 = 1
    var_9 = signed_content.split(var_3)[var_8]
    var_10 = var_4.verify_signature(var_2, var_9)
    assert var_10 is True
    var_11 = b'bm90X3RoZV9zaWduYXR1cmU='
    var_12 = var_4.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = b'tampered_payload'
    var_14 = var_4.verify_signature(var_13, var_9)
    assert var_14 is False
    var_15 = b'old_secret'
    var_16 = b'new_secret'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17, var_1, var_3)
    var_19 = var_18.derive_key(var_15)
    var_20 = module_0.HMACAlgorithm()
    var_21 = var_20.get_signature(var_19, var_2)
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_18.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = b'!!!notbase64!!!'
    var_25 = var_18.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_0, algorithm=var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_27.verify_signature(var_2, var_28)
    assert var_29 is True
    var_30 = 'hmac'
    var_31 = module_0.Signer(var_0, var_1, key_derivation=var_30)
    var_32 = var_31.get_signature(var_2)
    var_33 = var_31.verify_signature(var_2, var_32)
    assert var_33 is True



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = b'other'
    var_6 = var_1 + var_5
    var_7 = module_0.digest()
    var_8 = 'django-concat'
    var_9 = b'signer'
    var_10 = var_1 + var_9
    var_11 = var_10 + var_0
    var_12 = module_0.digest()
    var_13 = var_1 + var_9
    var_14 = var_13 + var_5
    var_15 = module_0.digest()
    var_16 = 'hmac'
    var_17 = module_0.digest()
    var_18 = module_0.digest()
    var_19 = 'none'
    var_20 = b'old'
    var_21 = b'new'
    var_22 = [var_20, var_21]
    var_23 = module_1.Signer(var_22, var_1, key_derivation=var_19)
    var_24 = var_23.derive_key()
    assert var_24 == b'new'
    var_25 = var_23.derive_key(var_20)
    assert var_25 == b'old'
    var_26 = 'invalid'
    var_27 = module_1.Signer(var_0, key_derivation=var_26)
    var_28 = var_27.derive_key()



# Parsed testcases at query #2
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = b'other'
    var_6 = var_1 + var_5
    var_7 = module_0.digest()
    var_8 = 'django-concat'
    var_9 = b'signer'
    var_10 = var_1 + var_9
    var_11 = var_10 + var_0
    var_12 = module_0.digest()
    var_13 = var_1 + var_9
    var_14 = var_13 + var_5
    var_15 = module_0.digest()
    var_16 = 'hmac'
    var_17 = module_0.digest()
    var_18 = module_0.digest()
    var_19 = module_0.digest()
    var_20 = 'none'
    var_21 = b'old'
    var_22 = b'new'
    var_23 = [var_21, var_22]
    var_24 = module_1.Signer(var_23, var_1, key_derivation=var_20)
    var_25 = var_24.derive_key()
    assert var_25 == b'new'
    var_26 = var_24.derive_key(var_21)
    assert var_26 == b'old'
    var_27 = 'invalid'
    var_28 = module_1.Signer(var_0, key_derivation=var_27)
    var_29 = var_28.derive_key()



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'super-secret-key'
    var_1 = b'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'hello-world'
    var_4 = var_2.get_signature(var_3)
    var_5 = b'.'
    var_6 = var_3 + var_5
    var_7 = var_6 + var_4
    var_8 = var_2.verify_signature(var_3, var_4)
    assert var_8 is True
    var_9 = len(var_3)
    var_10 = 1
    var_11 = var_9 + var_10
    var_12 = var_7[var_11:]
    var_13 = var_2.verify_signature(var_3, var_12)
    assert var_13 is True
    var_14 = b'tampered-world'
    var_15 = var_2.verify_signature(var_14, var_4)
    assert var_15 is False
    var_16 = module_1.base64_decode(var_4)
    var_17 = bytearray(var_16)
    var_18 = 0
    var_19 = var_17[var_18]
    var_20 = 255
    var_21 = var_19 ^ var_20
    var_22 = bytes(var_17)
    var_23 = module_1.base64_encode(var_22)
    var_24 = var_2.verify_signature(var_3, var_23)
    assert var_24 is False
    var_25 = b'not-base64-!!!'
    var_26 = var_2.verify_signature(var_3, var_25)
    assert var_26 is False
    var_27 = b'old-key'
    var_28 = [var_27, var_0]
    var_29 = module_0.Signer(var_28)
    var_30 = var_29.verify_signature(var_3, var_4)
    assert var_30 is True
    var_31 = b'different-key'
    var_32 = module_0.Signer(var_31, var_1)
    var_33 = var_32.get_signature(var_3)
    var_34 = var_2.verify_signature(var_3, var_33)
    assert var_34 is False
    var_35 = b'different-salt'
    var_36 = module_0.Signer(var_0, var_35)
    var_37 = var_36.verify_signature(var_3, var_4)
    assert var_37 is False



# Parsed testcases at query #4
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest(var_3)
    var_5 = module_0.digest()
    var_6 = 'django-concat'
    var_7 = b'signer'
    var_8 = var_1 + var_7
    var_9 = var_8 + var_0
    var_10 = module_0.digest(var_9)
    var_11 = module_0.digest()
    var_12 = 'hmac'
    var_13 = module_0.digest()
    var_14 = 'none'
    var_15 = module_1.Signer(var_0, var_1, key_derivation=var_14)
    var_16 = var_15.derive_key()
    var_17 = b'alternative'
    var_18 = var_1 + var_17
    var_19 = module_0.digest(var_18)
    var_20 = module_0.digest()
    var_21 = 'invalid'
    var_22 = module_1.Signer(var_0, key_derivation=var_21)
    var_23 = var_22.derive_key()
    var_24 = b'old'
    var_25 = b'new'
    var_26 = [var_24, var_25]
    var_27 = module_1.Signer(var_26, var_1, key_derivation=var_14)
    var_28 = var_27.derive_key()
    assert var_28 == b'new'
    var_29 = var_27.derive_key(var_24)
    assert var_29 == b'old'



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'.'
    var_3 = b'payload'
    var_4 = module_0.Signer(var_0, var_1, var_2)
    var_5 = var_4.get_signature(var_3)
    var_6 = var_3 + var_2
    var_7 = var_6 + var_5
    var_8 = 1
    var_9 = signed_value.split(var_2)[var_8]
    var_10 = var_4.verify_signature(var_3, var_9)
    assert var_10 is True
    var_11 = b'tampered_payload'
    var_12 = var_4.verify_signature(var_11, var_9)
    assert var_12 is False
    var_13 = b'different_value'
    var_14 = var_4.get_signature(var_13)
    var_15 = var_4.verify_signature(var_3, var_14)
    assert var_15 is False
    var_16 = b'!!!not-base64!!!'
    var_17 = var_4.verify_signature(var_3, var_16)
    assert var_17 is False
    var_18 = b'old_secret'
    var_19 = b'new_secret'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20, var_1, var_2)
    var_22 = var_21.derive_key(var_18)
    var_23 = 'hmac'
    var_24 = module_0.Signer(var_0, var_1, key_derivation=var_23)
    var_25 = var_24.get_signature(var_3)
    var_26 = var_24.verify_signature(var_3, var_25)
    assert var_26 is True
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, algorithm=var_27)
    var_29 = b''
    var_30 = var_28.verify_signature(var_3, var_29)
    assert var_30 is True



# Parsed testcases at query #6
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'django-concat'
    var_3 = b'signer'
    var_4 = var_1 + var_3
    var_5 = var_4 + var_0
    var_6 = module_0.digest(var_5)
    var_7 = module_0.digest()
    var_8 = 'concat'
    var_9 = var_1 + var_0
    var_10 = module_0.digest(var_9)
    var_11 = module_0.digest()
    var_12 = 'hmac'
    var_13 = module_0.digest()
    var_14 = 'none'
    var_15 = module_1.Signer(var_0, var_1, key_derivation=var_14)
    var_16 = var_15.derive_key()
    var_17 = b'alternative'
    var_18 = var_1 + var_3
    var_19 = var_18 + var_17
    var_20 = module_0.digest(var_19)
    var_21 = module_0.digest()
    var_22 = b'old_key'
    var_23 = b'new_key'
    var_24 = [var_22, var_23]
    var_25 = module_1.Signer(var_24, var_1, key_derivation=var_14)
    var_26 = var_25.derive_key()
    assert var_26 == b'new_key'
    var_27 = var_25.derive_key(var_22)
    assert var_27 == b'old_key'
    var_28 = 'invalid_method'
    var_29 = module_1.Signer(var_0, key_derivation=var_28)
    var_30 = var_29.derive_key()



# Parsed testcases at query #7
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'django-concat'
    var_3 = b'signer'
    var_4 = var_1 + var_3
    var_5 = var_4 + var_0
    var_6 = module_0.digest(var_5)
    var_7 = module_0.digest()
    var_8 = 'concat'
    var_9 = var_1 + var_0
    var_10 = module_0.digest(var_9)
    var_11 = module_0.digest()
    var_12 = 'hmac'
    var_13 = module_0.digest()
    var_14 = 'none'
    var_15 = b'alternative'
    var_16 = var_1 + var_3
    var_17 = var_16 + var_15
    var_18 = module_0.digest(var_17)
    var_19 = module_0.digest()
    var_20 = b'old'
    var_21 = b'new'
    var_22 = [var_20, var_21]
    var_23 = module_1.Signer(var_22, var_1, key_derivation=var_14)
    var_24 = var_23.derive_key()
    assert var_24 == b'new'
    var_25 = var_23.derive_key(var_20)
    assert var_25 == b'old'
    var_26 = 'invalid_method'
    var_27 = module_1.Signer(var_0, key_derivation=var_26)
    var_28 = var_27.derive_key()



# Parsed testcases at query #8
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret-key'
    var_1 = b'salt'
    var_2 = 'none'
    var_3 = b'other'
    var_4 = 'concat'
    var_5 = var_1 + var_0
    var_6 = module_0.digest()
    var_7 = var_1 + var_3
    var_8 = module_0.digest()
    var_9 = 'django-concat'
    var_10 = b'signer'
    var_11 = var_1 + var_10
    var_12 = var_11 + var_0
    var_13 = module_0.digest()
    var_14 = var_1 + var_10
    var_15 = var_14 + var_3
    var_16 = module_0.digest()
    var_17 = 'hmac'
    var_18 = module_0.digest()
    var_19 = b'other-key'
    var_20 = module_0.digest()
    var_21 = b'old-key'
    var_22 = b'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_1.Signer(var_23, var_1, key_derivation=var_2)
    var_25 = var_24.derive_key()
    assert var_25 == b'new-key'
    var_26 = var_24.derive_key(var_21)
    assert var_26 == b'old-key'
    var_27 = 'invalid'
    var_28 = module_1.Signer(var_0, key_derivation=var_27)
    var_29 = var_28.derive_key()
    var_30 = 'string-key'
    var_31 = module_1.Signer(var_30, var_1, key_derivation=var_29)
    var_32 = var_31.derive_key()
    assert var_32 == b'string-key'



# Parsed testcases at query #9
#--------------------------


import hmac as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest(var_3)
    var_5 = module_0.digest()



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'.'
    var_3 = b'payload'
    var_4 = module_0.Signer(var_0, var_1, var_2)
    var_5 = var_4.get_signature(var_3)
    var_6 = var_3 + var_2
    var_7 = var_6 + var_5
    var_8 = 1
    var_9 = signed_value.split(var_2)[var_8]
    var_10 = var_4.verify_signature(var_3, var_9)
    assert var_10 is True
    var_11 = b'tampered_payload'
    var_12 = var_4.verify_signature(var_11, var_9)
    assert var_12 is False
    var_13 = b'bm90X2FfcmVhbF9zaWduYXR1cmU='
    var_14 = var_4.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = b'old_secret'
    var_16 = [var_15, var_0]
    var_17 = module_0.Signer(var_16, var_1, var_2)
    var_18 = var_17.get_signature(var_3)
    var_19 = var_3 + var_2
    var_20 = var_19 + var_18
    var_21 = var_7.split(var_2)[var_8]
    var_22 = var_17.verify_signature(var_3, var_21)
    assert var_22 is True
    var_23 = b'!!!notbase64!!!'
    var_24 = var_17.verify_signature(var_3, var_23)
    assert var_24 is False
    var_25 = 'hmac'
    var_26 = module_0.Signer(var_0, var_1, var_2, var_25)
    var_27 = var_26.get_signature(var_3)
    var_28 = var_26.verify_signature(var_3, var_27)
    assert var_28 is True
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_0, algorithm=var_29)
    var_31 = b''
    var_32 = var_30.verify_signature(var_3, var_31)
    assert var_32 is True



# Parsed testcases at query #11
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = 'django-concat'
    var_6 = b'signer'
    var_7 = var_1 + var_6
    var_8 = var_7 + var_0
    var_9 = module_0.digest()
    var_10 = 'hmac'
    var_11 = module_0.digest()
    var_12 = 'none'
    var_13 = module_1.Signer(var_0, var_1, key_derivation=var_12)
    var_14 = var_13.derive_key()
    var_15 = b'alternative'
    var_16 = var_1 + var_15
    var_17 = module_0.digest()
    var_18 = 'invalid_method'
    var_19 = module_1.Signer(var_0, key_derivation=var_18)
    var_20 = var_19.derive_key()
    var_21 = b'old'
    var_22 = b'new'
    var_23 = [var_21, var_22]
    var_24 = module_1.Signer(var_23, var_1, key_derivation=var_12)
    var_25 = var_24.derive_key()
    assert var_25 == b'new'
    var_26 = var_24.derive_key(var_21)
    assert var_26 == b'old'



# Parsed testcases at query #12
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = 'django-concat'
    var_6 = b'signer'
    var_7 = var_1 + var_6
    var_8 = var_7 + var_0
    var_9 = module_0.digest()
    var_10 = 'hmac'
    var_11 = module_0.digest()
    var_12 = 'none'
    var_13 = module_1.Signer(var_0, var_1, key_derivation=var_12)
    var_14 = var_13.derive_key()
    var_15 = b'other'
    var_16 = var_1 + var_15
    var_17 = module_0.digest()
    var_18 = 'invalid'
    var_19 = module_1.Signer(var_0, key_derivation=var_18)
    var_20 = var_19.derive_key()
    var_21 = b'old'
    var_22 = b'new'
    var_23 = [var_21, var_22]
    var_24 = module_1.Signer(var_23, var_1, key_derivation=var_12)
    var_25 = var_24.derive_key()
    assert var_25 == b'new'
    var_26 = var_24.derive_key(var_21)
    assert var_26 == b'old'



# Parsed testcases at query #13
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest(var_3)
    var_5 = module_0.digest()
    var_6 = 'django-concat'
    var_7 = b'signer'
    var_8 = var_1 + var_7
    var_9 = var_8 + var_0
    var_10 = module_0.digest(var_9)
    var_11 = module_0.digest()
    var_12 = 'hmac'
    var_13 = module_0.digest()
    var_14 = 'none'
    var_15 = module_1.Signer(var_0, var_1, key_derivation=var_14)
    var_16 = b'alt-secret'
    var_17 = var_1 + var_16
    var_18 = module_0.digest(var_17)
    var_19 = module_0.digest()
    var_20 = b'old'
    var_21 = b'new'
    var_22 = [var_20, var_21]
    var_23 = module_1.Signer(var_22, var_1, key_derivation=var_14)
    var_24 = 'invalid_method'
    var_25 = module_1.Signer(var_0, key_derivation=var_24)
    var_26 = var_25.derive_key()



# Parsed testcases at query #14
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'super-secret'
    var_1 = b'test-salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = 'django-concat'
    var_6 = b'signer'
    var_7 = var_1 + var_6
    var_8 = var_7 + var_0
    var_9 = module_0.digest()
    var_10 = 'hmac'
    var_11 = module_0.digest()
    var_12 = 'none'
    var_13 = b'other-key'
    var_14 = var_1 + var_13
    var_15 = module_0.digest()
    var_16 = b'old-key'
    var_17 = b'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_1.Signer(var_18, var_1, key_derivation=var_12)
    var_20 = var_19.derive_key()
    assert var_20 == b'new-key'
    var_21 = var_19.derive_key(var_16)
    assert var_21 == b'old-key'
    var_22 = 'invalid'
    var_23 = module_1.Signer(var_0, key_derivation=var_22)
    var_24 = var_23.derive_key()



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'hello world'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.sep
    var_6 = var_3 + var_5
    var_7 = var_6 + var_4
    var_8 = var_2.verify_signature(var_3, var_4)
    assert var_8 is True
    var_9 = var_2.verify_signature(var_3, var_7)
    assert var_9 is True
    var_10 = b'tampered world'
    var_11 = var_2.verify_signature(var_10, var_4)
    assert var_11 is False
    var_12 = b'something else'
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = b'old-secret'
    var_16 = [var_15, var_0]
    var_17 = module_0.Signer(var_16, var_1)
    var_18 = var_17.get_signature(var_3)
    var_19 = var_17.verify_signature(var_3, var_18)
    assert var_19 is True
    var_20 = var_17.derive_key(var_15)
    var_21 = b'!!!not-base64!!!'
    var_22 = var_2.verify_signature(var_3, var_21)
    assert var_22 is False
    var_23 = b''
    var_24 = var_2.verify_signature(var_23, var_4)
    assert var_24 is False



# Parsed testcases at query #16
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = 'django-concat'
    var_6 = b'signer'
    var_7 = var_1 + var_6
    var_8 = var_7 + var_0
    var_9 = module_0.digest()
    var_10 = 'hmac'
    var_11 = module_0.digest()
    var_12 = 'none'
    var_13 = b'alternative'
    var_14 = var_1 + var_6
    var_15 = var_14 + var_13
    var_16 = module_0.digest()
    var_17 = b'old'
    var_18 = b'new'
    var_19 = [var_17, var_18]
    var_20 = module_1.Signer(var_19, var_1, key_derivation=var_12)
    var_21 = var_20.derive_key()
    assert var_21 == b'new'
    var_22 = var_20.derive_key(var_17)
    assert var_22 == b'old'
    var_23 = 'invalid'
    var_24 = module_1.Signer(var_0, key_derivation=var_23)
    var_25 = var_24.derive_key()



# Parsed testcases at query #17
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = b'other'
    var_6 = var_1 + var_5
    var_7 = module_0.digest()
    var_8 = 'django-concat'
    var_9 = b'signer'
    var_10 = var_1 + var_9
    var_11 = var_10 + var_0
    var_12 = module_0.digest()
    var_13 = var_1 + var_9
    var_14 = var_13 + var_5
    var_15 = module_0.digest()
    var_16 = 'hmac'
    var_17 = module_0.digest()
    var_18 = module_0.digest()
    var_19 = 'none'
    var_20 = 'invalid_method'
    var_21 = b'old_key'
    var_22 = b'new_key'
    var_23 = [var_21, var_22]
    var_24 = module_1.Signer(var_23, var_1, key_derivation=var_19)
    var_25 = var_24.derive_key()
    assert var_25 == b'new_key'
    var_26 = var_24.derive_key(var_21)
    assert var_26 == b'old_key'



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1
import src.itsdangerous.encoding as module_2

def test_case_0():
    var_0 = '\n    Tests the verify_signature method of the Signer class, covering:\n    - Successful verification with correct signature.\n    - Failed verification with incorrect signature.\n    - Failed verification with invalid base64 encoding.\n    - Verification using rotated keys (oldest to newest).\n    '
    var_1 = b'secret'
    var_2 = b'salt'
    var_3 = b'.'
    var_4 = b'payload'
    var_5 = module_0.Signer(var_1, var_2, var_3)
    var_6 = var_5.get_signature(var_4)
    var_7 = var_5.verify_signature(var_4, var_6)
    assert var_7 is True
    var_8 = b'wrong_payload'
    var_9 = var_5.verify_signature(var_8, var_6)
    assert var_9 is False
    var_10 = b'another_value'
    var_11 = var_5.get_signature(var_10)
    var_12 = var_5.verify_signature(var_4, var_11)
    assert var_12 is False
    var_13 = b'!!!not-base64!!!'
    var_14 = var_5.verify_signature(var_4, var_13)
    assert var_14 is False
    var_15 = b'old_secret'
    var_16 = b'new_secret'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17, var_2, var_3)
    var_19 = var_18.get_signature(var_4)
    var_20 = var_18.verify_signature(var_4, var_19)
    assert var_20 is True
    var_21 = var_18.derive_key(var_15)
    var_22 = module_1.digest()
    var_23 = module_2.base64_encode(var_22)
    var_24 = var_18.verify_signature(var_4, var_23)
    assert var_24 is True
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_1, algorithm=var_25)
    var_27 = b''
    var_28 = var_26.verify_signature(var_4, var_27)
    assert var_28 is True



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'payload'
    var_3 = b'.'
    var_4 = module_0.Signer(var_0, var_1, var_3)
    var_5 = var_4.get_signature(var_2)
    var_6 = var_4.verify_signature(var_2, var_5)
    assert var_6 is True
    var_7 = b'wrong_payload'



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_4 = var_1 + var_0
    var_5 = module_1.digest()
    var_6 = None
    var_7 = 'django-concat'
    var_8 = module_0.Signer(var_0, var_1, key_derivation=var_7)
    var_9 = b'signer'
    var_10 = var_1 + var_9
    var_11 = var_10 + var_0
    var_12 = module_1.digest()
    var_13 = 'hmac'
    var_14 = module_0.Signer(var_0, var_1, key_derivation=var_13)
    var_15 = module_1.digest()
    var_16 = 'none'
    var_17 = module_0.Signer(var_0, var_1, key_derivation=var_16)
    var_18 = b'other'
    var_19 = var_1 + var_18
    var_20 = module_1.digest()
    var_21 = 'invalid'
    var_22 = module_0.Signer(var_0, key_derivation=var_21)
    var_23 = var_22.derive_key()



# Parsed testcases at query #21
#--------------------------


import hmac as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()



# Parsed testcases at query #22
#--------------------------


import hmac as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()



# Parsed testcases at query #23
#--------------------------


import hmac as module_0
import src.itsdangerous.signer as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'concat'
    var_3 = var_1 + var_0
    var_4 = module_0.digest()
    var_5 = 'django-concat'
    var_6 = b'signer'
    var_7 = var_1 + var_6
    var_8 = var_7 + var_0
    var_9 = module_0.digest()
    var_10 = 'hmac'
    var_11 = module_0.digest()
    var_12 = 'none'
    var_13 = b'alternative'
    var_14 = module_0.digest()
    var_15 = 'invalid'
    var_16 = module_1.Signer(var_0, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = b'old'
    var_19 = b'new'
    var_20 = [var_18, var_19]
    var_21 = module_1.Signer(var_20, var_1, key_derivation=var_12)
    var_22 = var_21.derive_key()
    assert var_22 == b'new'
    var_23 = var_21.derive_key(var_18)
    assert var_23 == b'old'



