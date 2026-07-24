####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = -1
    var_6 = var_3[:var_5]
    var_7 = b'x'
    var_8 = var_6 + var_7
    var_9 = var_1.unsign(var_8)
    var_10 = b'no-separator'
    var_11 = var_1.unsign(var_10)
    var_12 = b'value.sep1.sep2'
    var_13 = var_1.unsign(var_12)
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = module_0.Signer(var_14)
    var_19 = var_18.sign(var_2)
    var_20 = module_0.Signer(var_15)
    var_21 = var_20.sign(var_2)
    var_22 = var_17.unsign(var_19)
    var_23 = var_17.unsign(var_21)
    var_24 = 'hmac'
    var_25 = module_0.Signer(var_13, key_derivation=var_24)
    var_26 = var_25.sign(var_2)
    var_27 = var_25.unsign(var_26)
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_13, algorithm=var_28)
    var_30 = var_29.sign(var_2)
    var_31 = var_29.unsign(var_30)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = module_0.Signer(var_0, var_1)
    var_7 = b'test-value.invalid-sig'
    var_8 = var_6.unsign(var_7)
    var_9 = module_0.Signer(var_7, var_8)
    var_10 = b'test-value'
    var_11 = var_9.unsign(var_10)
    var_12 = module_0.Signer(var_10, var_11)
    var_13 = b'test.value.with.separators'
    var_14 = var_12.sign(var_13)
    var_15 = var_12.unsign(var_14)
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_11)
    var_20 = b'test-value'
    var_21 = var_19.sign(var_20)
    var_22 = var_19.unsign(var_21)
    var_23 = [var_16, var_17]
    var_24 = module_0.Signer(var_23, var_11)
    var_25 = b'test-value'
    var_26 = module_0.Signer(var_16, var_11)
    var_27 = var_26.sign(var_25)
    var_28 = var_24.unsign(var_27)
    var_29 = 'hmac'
    var_30 = module_0.Signer(var_10, var_11, key_derivation=var_29)
    var_31 = b'test-value'
    var_32 = var_30.sign(var_31)
    var_33 = var_30.unsign(var_32)
    var_34 = b'test-value'
    var_35 = var_30.sign(var_34)
    var_36 = var_30.unsign(var_35)
    var_37 = module_0.NoneAlgorithm()
    var_38 = module_0.Signer(var_10, var_11, algorithm=var_37)
    var_39 = b'test-value'
    var_40 = var_38.sign(var_39)
    var_41 = var_38.unsign(var_40)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = 'another-secret'
    var_23 = var_2.derive_key(var_22)
    var_24 = var_4 + var_5
    var_25 = b'another-secret'
    var_26 = var_24 + var_25
    var_27 = module_1.digest()
    var_28 = module_0.Signer(var_7, var_4)
    var_29 = var_28.derive_key()
    var_30 = var_4 + var_5
    var_31 = var_30 + var_7
    var_32 = module_1.digest()
    var_33 = 'old-key'
    var_34 = 'new-key'
    var_35 = [var_33, var_34]
    var_36 = module_0.Signer(var_35, var_1)
    var_37 = var_36.derive_key()
    var_38 = var_4 + var_5
    var_39 = b'new-key'
    var_40 = var_38 + var_39
    var_41 = module_1.digest()
    var_42 = var_36.derive_key(var_33)
    var_43 = var_4 + var_5
    var_44 = b'old-key'
    var_45 = var_43 + var_44
    var_46 = module_1.digest()
    var_47 = var_4 + var_5
    var_48 = var_47 + var_7
    var_49 = module_1.digest()



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'different-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value'
    var_16 = module_0.Signer(var_11)
    var_17 = var_16.get_signature(var_15)
    var_18 = module_0.Signer(var_12)
    var_19 = var_18.get_signature(var_15)
    var_20 = var_14.verify_signature(var_15, var_17)
    assert var_20 is True
    var_21 = var_14.verify_signature(var_15, var_19)
    assert var_21 is True
    var_22 = var_14.verify_signature(var_15, var_5)
    assert var_22 is False
    var_23 = 'concat'
    var_24 = module_0.Signer(var_0, key_derivation=var_23)
    var_25 = var_24.get_signature(var_15)
    var_26 = var_24.verify_signature(var_15, var_25)
    assert var_26 is True
    var_27 = var_1.verify_signature(var_15, var_25)
    assert var_27 is False
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_0, algorithm=var_28)
    var_30 = var_29.get_signature(var_15)
    var_31 = var_29.verify_signature(var_15, var_30)
    assert var_31 is True
    var_32 = var_1.verify_signature(var_15, var_30)
    assert var_32 is False



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_0._lazy_sha1(var_8)
    var_10 = module_1.digest()
    var_11 = 'concat'
    var_12 = module_0.Signer(var_0, var_1, key_derivation=var_11)
    var_13 = var_12.derive_key()
    var_14 = var_4 + var_7
    var_15 = module_0._lazy_sha1(var_14)
    var_16 = module_1.digest()
    var_17 = 'hmac'
    var_18 = module_0.Signer(var_0, var_1, key_derivation=var_17)
    var_19 = var_18.derive_key()
    var_20 = module_1.digest()
    var_21 = 'none'
    var_22 = module_0.Signer(var_0, var_1, key_derivation=var_21)
    var_23 = var_22.derive_key()
    assert var_23 == b'secret-key'
    var_24 = module_0.Signer(var_0, var_1)
    var_25 = 'another-secret'
    var_26 = var_24.derive_key(var_25)
    var_27 = var_4 + var_5
    var_28 = b'another-secret'
    var_29 = var_27 + var_28
    var_30 = module_0._lazy_sha1(var_29)
    var_31 = module_1.digest()
    var_32 = module_0.Signer(var_7, var_4)
    var_33 = var_32.derive_key()
    var_34 = var_4 + var_5
    var_35 = var_34 + var_7
    var_36 = module_0._lazy_sha1(var_35)
    var_37 = module_1.digest()
    var_38 = module_0.Signer(var_0, var_4)
    var_39 = var_38.derive_key()
    var_40 = var_4 + var_5
    var_41 = var_40 + var_7
    var_42 = module_0._lazy_sha1(var_41)
    var_43 = module_1.digest()
    var_44 = 'invalid'
    var_45 = module_0.Signer(var_0, var_1, key_derivation=var_44)
    var_46 = var_45.derive_key()



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'invalid-base64!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'test-value'
    var_14 = module_0.Signer(var_9)
    var_15 = var_14.get_signature(var_13)
    var_16 = module_0.Signer(var_10)
    var_17 = var_16.get_signature(var_13)
    var_18 = var_12.verify_signature(var_13, var_15)
    assert var_18 is True
    var_19 = var_12.verify_signature(var_13, var_17)
    assert var_19 is True
    var_20 = var_12.verify_signature(var_13, var_5)
    assert var_20 is False
    var_21 = 'secret-key'
    var_22 = b'test-value'
    var_23 = var_1.get_signature(var_22)
    var_24 = var_1.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'wrong-signature'
    var_26 = var_1.verify_signature(var_22, var_25)
    assert var_26 is False
    var_27 = b'test-value'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_21, algorithm=var_28)
    var_30 = b'test-value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = var_29.verify_signature(var_30, var_25)
    assert var_33 is False



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.derive_key()
    var_3 = b'itsdangerous.Signersignersecret-key'
    var_4 = module_1.digest()
    var_5 = 'concat'
    var_6 = module_0.Signer(var_0, key_derivation=var_5)
    var_7 = var_6.derive_key()
    var_8 = b'itsdangerous.Signersecret-key'
    var_9 = module_1.digest()
    var_10 = 'hmac'
    var_11 = module_0.Signer(var_0, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = b'secret-key'
    var_14 = b'itsdangerous.Signer'
    var_15 = module_1.digest()
    var_16 = 'none'
    var_17 = module_0.Signer(var_0, key_derivation=var_16)
    var_18 = var_17.derive_key()
    assert var_18 == b'secret-key'
    var_19 = module_0.Signer(var_0)
    var_20 = 'other-secret'
    var_21 = var_19.derive_key(var_20)
    var_22 = b'itsdangerous.Signersignerother-secret'
    var_23 = module_1.digest()
    var_24 = module_0.Signer(var_13)
    var_25 = var_24.derive_key()
    var_26 = module_1.digest()
    var_27 = module_0.Signer(var_0)
    var_28 = b'other-secret'
    var_29 = var_27.derive_key(var_28)
    var_30 = module_1.digest()
    var_31 = 'invalid'
    var_32 = module_0.Signer(var_0, key_derivation=var_31)
    var_33 = var_32.derive_key()



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = var_13.get_signature(var_2)
    var_15 = var_13.verify_signature(var_2, var_14)
    assert var_15 is True
    var_16 = 'secret-key'
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_16, algorithm=var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = b'invalid-base64!'
    var_22 = var_1.verify_signature(var_2, var_21)
    assert var_22 is False



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = len(var_3)
    var_5 = 'concat'
    var_6 = module_0.Signer(var_0, var_1, key_derivation=var_5)
    var_7 = var_6.derive_key()
    var_8 = b'test-salt'
    var_9 = b'secret-key'
    var_10 = var_8 + var_9
    var_11 = module_1.digest()
    var_12 = 'django-concat'
    var_13 = module_0.Signer(var_0, var_1, key_derivation=var_12)
    var_14 = var_13.derive_key()
    var_15 = b'signer'
    var_16 = var_8 + var_15
    var_17 = var_16 + var_9
    var_18 = module_1.digest()
    var_19 = 'hmac'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    var_22 = module_1.digest()
    var_23 = 'none'
    var_24 = module_0.Signer(var_0, var_1, key_derivation=var_23)
    var_25 = var_24.derive_key()
    assert var_25 == b'secret-key'
    var_26 = module_0.Signer(var_0, var_1)
    var_27 = 'specific-key'
    var_28 = var_26.derive_key(var_27)
    var_29 = var_8 + var_15
    var_30 = b'specific-key'
    var_31 = var_29 + var_30
    var_32 = module_1.digest()
    var_33 = module_0.Signer(var_9, var_8)
    var_34 = var_33.derive_key()
    var_35 = var_8 + var_15
    var_36 = var_35 + var_9
    var_37 = module_1.digest()
    var_38 = 'unknown'
    var_39 = module_0.Signer(var_0, var_1, key_derivation=var_38)
    var_40 = var_39.derive_key()



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-salt'
    var_4 = b'signer'
    var_5 = var_3 + var_4
    var_6 = b'secret-key'
    var_7 = var_5 + var_6
    var_8 = module_1.digest()
    var_9 = var_2.derive_key()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_3 + var_6
    var_13 = module_1.digest()
    var_14 = var_11.derive_key()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = module_1.digest()
    var_18 = var_16.derive_key()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'other-key'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_3 + var_4
    var_26 = b'other-key'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = module_0.Signer(var_6, var_3)
    var_30 = var_29.derive_key()
    var_31 = var_3 + var_4
    var_32 = var_31 + var_6
    var_33 = module_1.digest()
    var_34 = module_0.Signer(var_0, var_3)
    var_35 = var_34.derive_key()
    var_36 = var_3 + var_4
    var_37 = var_36 + var_6
    var_38 = module_1.digest()
    var_39 = 'unknown'
    var_40 = module_0.Signer(var_0, var_1, key_derivation=var_39)
    var_41 = var_40.derive_key()



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = len(var_3)
    var_5 = 'concat'
    var_6 = module_0.Signer(var_0, var_1, key_derivation=var_5)
    var_7 = var_6.derive_key()
    var_8 = b'test-salt'
    var_9 = b'secret-key'
    var_10 = var_8 + var_9
    var_11 = module_1.digest()
    var_12 = 'hmac'
    var_13 = module_0.Signer(var_0, var_1, key_derivation=var_12)
    var_14 = var_13.derive_key()
    var_15 = module_1.digest()
    var_16 = 'none'
    var_17 = module_0.Signer(var_0, var_1, key_derivation=var_16)
    var_18 = var_17.derive_key()
    assert var_18 == b'secret-key'
    var_19 = module_0.Signer(var_0, var_1)
    var_20 = 'custom-secret'
    var_21 = var_19.derive_key(var_20)
    var_22 = b'custom-secret'
    var_23 = module_1.digest()
    var_24 = 'invalid'
    var_25 = module_0.Signer(var_0, var_1, key_derivation=var_24)
    var_26 = var_25.derive_key()



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'other-key'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'other-key'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = module_0.Signer(var_7, var_4)
    var_30 = var_29.derive_key()
    var_31 = var_4 + var_5
    var_32 = var_31 + var_7
    var_33 = module_1.digest()
    var_34 = module_0.Signer(var_0, var_4)
    var_35 = var_34.derive_key()
    var_36 = var_4 + var_5
    var_37 = var_36 + var_7
    var_38 = module_1.digest()
    var_39 = 'invalid'
    var_40 = module_0.Signer(var_0, var_1, key_derivation=var_39)
    var_41 = var_40.derive_key()



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'old-key'
    var_14 = var_12.get_signature(var_2)
    var_15 = b'new-key'
    var_16 = var_12.get_signature(var_2)
    var_17 = var_12.verify_signature(var_2, var_14)
    assert var_17 is True
    var_18 = var_12.verify_signature(var_2, var_16)
    assert var_18 is True
    var_19 = 'secret-key'
    var_20 = module_0.NoneAlgorithm()
    var_21 = module_0.Signer(var_19, algorithm=var_20)
    var_22 = var_21.get_signature(var_2)
    var_23 = var_21.verify_signature(var_2, var_22)
    assert var_23 is True



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = len(var_3)
    var_5 = 'concat'
    var_6 = module_0.Signer(var_0, var_1, key_derivation=var_5)
    var_7 = var_6.derive_key()
    var_8 = b'test_saltsecret'
    var_9 = module_1.digest()
    var_10 = 'django-concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = b'test_saltsignersecret'
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = b'secret'
    var_19 = b'test_salt'
    var_20 = module_1.digest()
    var_21 = 'none'
    var_22 = module_0.Signer(var_0, var_1, key_derivation=var_21)
    var_23 = var_22.derive_key()
    assert var_23 == b'secret'
    var_24 = module_0.Signer(var_0, var_1, key_derivation=var_5)
    var_25 = 'specific_secret'
    var_26 = var_24.derive_key(var_25)
    var_27 = b'specific_secret'
    var_28 = var_19 + var_27
    var_29 = module_1.digest()
    var_30 = module_0.Signer(var_18, var_19, key_derivation=var_5)
    var_31 = var_30.derive_key()
    var_32 = var_19 + var_18
    var_33 = module_1.digest()
    var_34 = var_30.derive_key()
    var_35 = module_1.digest()
    var_36 = 'invalid'
    var_37 = module_0.Signer(var_0, var_1, key_derivation=var_36)
    var_38 = var_37.derive_key()



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test_salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    var_22 = b'secret'
    var_23 = module_0.Signer(var_0, var_1)
    var_24 = 'other_secret'
    var_25 = var_23.derive_key(var_24)
    var_26 = var_4 + var_5
    var_27 = b'other_secret'
    var_28 = var_26 + var_27
    var_29 = module_1.digest()
    var_30 = module_0.Signer(var_7, var_1)
    var_31 = var_30.derive_key()
    var_32 = var_4 + var_5
    var_33 = var_32 + var_7
    var_34 = module_1.digest()
    var_35 = module_0.Signer(var_0, var_4)
    var_36 = var_35.derive_key()
    var_37 = var_4 + var_5
    var_38 = var_37 + var_7
    var_39 = module_1.digest()
    var_40 = var_35.derive_key()
    var_41 = var_4 + var_5
    var_42 = var_41 + var_7
    var_43 = module_1.digest()
    var_44 = 'invalid'
    var_45 = module_0.Signer(var_0, var_1, key_derivation=var_44)
    var_46 = var_45.derive_key()



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret'
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'other_secret'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'other_secret'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = module_0.Signer(var_7, var_4)
    var_30 = var_29.derive_key(var_26)
    var_31 = var_4 + var_5
    var_32 = var_31 + var_26
    var_33 = module_1.digest()
    var_34 = 'invalid'
    var_35 = module_0.Signer(var_0, var_1, key_derivation=var_34)
    var_36 = var_35.derive_key()



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret'
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'other_secret'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'other_secret'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = module_0.Signer(var_7, var_4)
    var_30 = var_29.derive_key()
    var_31 = var_4 + var_5
    var_32 = var_31 + var_7
    var_33 = module_1.digest()
    var_34 = 'secret'
    var_35 = 'salt'
    var_36 = 'invalid'
    var_37 = module_0.Signer(var_34, var_35, key_derivation=var_36)
    var_38 = var_37.derive_key()



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'other-secret'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'other-secret'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = module_0.Signer(var_7, var_4)
    var_30 = var_29.derive_key()
    var_31 = var_4 + var_5
    var_32 = var_31 + var_7
    var_33 = module_1.digest()
    var_34 = module_0.Signer(var_0, var_4)
    var_35 = var_34.derive_key()
    var_36 = var_4 + var_5
    var_37 = var_36 + var_7
    var_38 = module_1.digest()
    var_39 = 'invalid'
    var_40 = module_0.Signer(var_0, var_1, key_derivation=var_39)
    var_41 = var_40.derive_key()



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = len(var_3)
    var_5 = 'concat'
    var_6 = module_0.Signer(var_0, var_1, key_derivation=var_5)
    var_7 = var_6.derive_key()
    var_8 = 'hmac'
    var_9 = module_0.Signer(var_0, var_1, key_derivation=var_8)
    var_10 = var_9.derive_key()
    var_11 = 'none'
    var_12 = module_0.Signer(var_0, var_1, key_derivation=var_11)
    var_13 = var_12.derive_key()
    assert var_13 == b'secret'
    var_14 = 'custom-secret'
    var_15 = var_2.derive_key(var_14)
    var_16 = b'secret'
    var_17 = module_0.Signer(var_16, var_1)
    var_18 = var_17.derive_key()
    var_19 = 'invalid'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'wrong-sig'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_2.verify_signature(var_3, var_9)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13, var_1)
    var_15 = b'test-value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'secret-key'
    var_19 = 'test-salt'
    var_20 = b'test-value'
    var_21 = var_2.get_signature(var_20)
    var_22 = var_2.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'test-value'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_18, var_19, algorithm=var_24)
    var_26 = b'test-value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'invalid'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = var_13.get_signature(var_2)
    var_15 = var_13.verify_signature(var_2, var_14)
    assert var_15 is True
    var_16 = 'secret-key'
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_16, algorithm=var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'invalid-base64!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value'
    var_16 = module_0.Signer(var_11)
    var_17 = var_16.get_signature(var_15)
    var_18 = module_0.Signer(var_12)
    var_19 = var_18.get_signature(var_15)
    var_20 = var_14.verify_signature(var_15, var_17)
    assert var_20 is True
    var_21 = var_14.verify_signature(var_15, var_19)
    assert var_21 is True
    var_22 = var_14.verify_signature(var_15, var_5)
    assert var_22 is False
    var_23 = 'secret-key'
    var_24 = b'test-value'
    var_25 = var_1.get_signature(var_24)
    var_26 = var_1.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'wrong-signature'
    var_28 = var_1.verify_signature(var_24, var_27)
    assert var_28 is False



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'not-base64!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'old-key'
    var_14 = var_12.get_signature(var_2)
    var_15 = b'new-key'
    var_16 = var_12.get_signature(var_2)
    var_17 = var_12.verify_signature(var_2, var_14)
    assert var_17 is True
    var_18 = var_12.verify_signature(var_2, var_16)
    assert var_18 is True
    var_19 = 'secret-key'
    var_20 = module_0.NoneAlgorithm()
    var_21 = module_0.Signer(var_19, algorithm=var_20)
    var_22 = var_21.get_signature(var_2)
    var_23 = var_21.verify_signature(var_2, var_22)
    assert var_23 is True



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_12.verify_signature(var_2, var_13)
    assert var_14 is True
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, key_derivation=var_15)
    var_17 = var_16.get_signature(var_2)
    var_18 = var_16.verify_signature(var_2, var_17)
    assert var_18 is True
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_0, algorithm=var_19)
    var_21 = var_20.get_signature(var_2)
    var_22 = var_20.verify_signature(var_2, var_21)
    assert var_22 is True



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'different-value'
    var_11 = var_1.verify_signature(var_10, var_3)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = var_15.get_signature(var_2)
    var_17 = var_15.verify_signature(var_2, var_16)
    assert var_17 is True
    var_18 = b'invalid-base64!'
    var_19 = var_1.verify_signature(var_2, var_18)
    assert var_19 is False
    var_20 = b''
    var_21 = var_1.verify_signature(var_2, var_20)
    assert var_21 is False
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_0, algorithm=var_22)
    var_24 = var_23.get_signature(var_2)
    var_25 = var_23.verify_signature(var_2, var_24)
    assert var_25 is True
    var_26 = b'anything'
    var_27 = var_23.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = 'hmac'
    var_29 = module_0.Signer(var_0, key_derivation=var_28)
    var_30 = var_29.get_signature(var_2)
    var_31 = var_29.verify_signature(var_2, var_30)
    assert var_31 is True
    var_32 = var_1.verify_signature(var_2, var_30)
    assert var_32 is False



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'old-key'
    var_15 = var_13.get_signature(var_2)
    var_16 = b'new-key'
    var_17 = var_13.get_signature(var_2)
    var_18 = var_13.verify_signature(var_2, var_15)
    assert var_18 is True
    var_19 = var_13.verify_signature(var_2, var_17)
    assert var_19 is True
    var_20 = 'secret-key'
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_20, algorithm=var_21)
    var_23 = var_22.get_signature(var_2)
    var_24 = var_22.verify_signature(var_2, var_23)
    assert var_24 is True
    var_25 = b'invalid-base64!'
    var_26 = var_1.verify_signature(var_2, var_25)
    assert var_26 is False



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'different-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'test-value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = b'not-base64!'
    var_17 = var_1.verify_signature(var_13, var_16)
    assert var_17 is False
    var_18 = 'secret-key'
    var_19 = b'test-value'
    var_20 = var_1.get_signature(var_19)
    var_21 = var_1.verify_signature(var_19, var_20)
    assert var_21 is True



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'invalid-base64!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'test-value'
    var_14 = module_0.Signer(var_9)
    var_15 = var_14.get_signature(var_13)
    var_16 = module_0.Signer(var_10)
    var_17 = var_16.get_signature(var_13)
    var_18 = var_12.verify_signature(var_13, var_15)
    assert var_18 is True
    var_19 = var_12.verify_signature(var_13, var_17)
    assert var_19 is True
    var_20 = var_12.verify_signature(var_13, var_5)
    assert var_20 is False
    var_21 = 'secret-key'
    var_22 = b'test-value'
    var_23 = var_1.get_signature(var_22)
    var_24 = var_1.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'wrong-signature'
    var_26 = var_1.verify_signature(var_22, var_25)
    assert var_26 is False
    var_27 = b'test-value'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_21, algorithm=var_28)
    var_30 = b'test-value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = var_29.verify_signature(var_30, var_25)
    assert var_33 is False



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'old-key'
    var_14 = var_12.get_signature(var_2)
    var_15 = b'new-key'
    var_16 = var_12.get_signature(var_2)
    var_17 = var_12.verify_signature(var_2, var_14)
    assert var_17 is True
    var_18 = var_12.verify_signature(var_2, var_16)
    assert var_18 is True
    var_19 = var_12.verify_signature(var_2, var_5)
    assert var_19 is False
    var_20 = 'secret-key'
    var_21 = b'invalid-sig'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_20, algorithm=var_22)
    var_24 = var_23.get_signature(var_2)
    var_25 = var_23.verify_signature(var_2, var_24)
    assert var_25 is True
    var_26 = var_23.verify_signature(var_2, var_21)
    assert var_26 is False



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotated-value'
    var_17 = module_0.Signer(var_12)
    var_18 = var_17.get_signature(var_16)
    var_19 = module_0.Signer(var_13)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_18)
    assert var_21 is True
    var_22 = var_15.verify_signature(var_16, var_20)
    assert var_22 is True
    var_23 = var_15.verify_signature(var_16, var_7)
    assert var_23 is False
    var_24 = 'secret-key'
    var_25 = b'derivation-test'
    var_26 = var_1.get_signature(var_25)
    var_27 = var_1.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'sha256-test'
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_24, algorithm=var_29)
    var_31 = b'none-algo-test'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'test-value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'secret-key'
    var_17 = b'test-value'
    var_18 = var_1.get_signature(var_17)
    var_19 = var_1.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = b'test-value'
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_16, algorithm=var_21)
    var_23 = b'test-value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotated-value'
    var_17 = module_0.Signer(var_12)
    var_18 = var_17.get_signature(var_16)
    var_19 = module_0.Signer(var_13)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_20)
    assert var_21 is True
    var_22 = var_15.verify_signature(var_16, var_18)
    assert var_22 is True
    var_23 = 'other-key'
    var_24 = module_0.Signer(var_23)
    var_25 = var_24.get_signature(var_16)
    var_26 = var_15.verify_signature(var_16, var_25)
    assert var_26 is False
    var_27 = 'secret-key'
    var_28 = b'kd-value'
    var_29 = b'sha256-value'
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_27, algorithm=var_30)
    var_32 = b'none-value'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'invalid-base64!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value'
    var_16 = module_0.Signer(var_11)
    var_17 = var_16.get_signature(var_15)
    var_18 = module_0.Signer(var_12)
    var_19 = var_18.get_signature(var_15)
    var_20 = var_14.verify_signature(var_15, var_17)
    assert var_20 is True
    var_21 = var_14.verify_signature(var_15, var_19)
    assert var_21 is True
    var_22 = var_14.verify_signature(var_15, var_5)
    assert var_22 is False
    var_23 = 'secret-key'
    var_24 = b'test-value'
    var_25 = var_1.get_signature(var_24)
    var_26 = var_1.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'wrong-signature'
    var_28 = var_1.verify_signature(var_24, var_27)
    assert var_28 is False



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = module_1.base64_encode(var_6)
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'different-value'
    var_11 = var_2.verify_signature(var_10, var_4)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = b'test-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is False
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_0, var_1, algorithm=var_19)
    var_21 = b'test-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b''
    var_25 = var_20.verify_signature(var_21, var_24)
    assert var_25 is True



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = 'secret-key'
    var_9 = 'test-salt'
    var_10 = var_2.get_signature(var_3)
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is True
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_9)
    var_16 = b'test-value'
    var_17 = module_0.Signer(var_12, var_9)
    var_18 = var_17.get_signature(var_16)
    var_19 = module_0.Signer(var_13, var_9)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_18)
    assert var_21 is True
    var_22 = var_15.verify_signature(var_16, var_20)
    assert var_22 is True
    var_23 = var_15.verify_signature(var_16, var_6)
    assert var_23 is False
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_8, var_9, algorithm=var_24)
    var_26 = b'test-value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = var_25.verify_signature(var_26, var_6)
    assert var_29 is False
    var_30 = b'invalid-base64!'
    var_31 = var_25.verify_signature(var_26, var_30)
    assert var_31 is False



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'different-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = module_1.base64_encode(var_5)
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.get_signature(var_15)
    var_18 = var_14.verify_signature(var_15, var_17)
    assert var_18 is True
    var_19 = var_14.verify_signature(var_15, var_16)
    assert var_19 is False
    var_20 = 'concat'
    var_21 = module_0.Signer(var_0, key_derivation=var_20)
    var_22 = var_21.get_signature(var_15)
    var_23 = var_21.verify_signature(var_15, var_22)
    assert var_23 is True
    var_24 = var_1.verify_signature(var_15, var_22)
    assert var_24 is False
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_0, algorithm=var_25)
    var_27 = var_26.get_signature(var_15)
    var_28 = var_26.verify_signature(var_15, var_27)
    assert var_28 is True
    var_29 = var_1.verify_signature(var_15, var_27)
    assert var_29 is False



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'custom-key'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'custom-key'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = module_0.Signer(var_7, var_4)
    var_30 = var_29.derive_key()
    var_31 = var_4 + var_5
    var_32 = var_31 + var_7
    var_33 = module_1.digest()
    var_34 = module_0.Signer(var_0, var_4)
    var_35 = var_34.derive_key()
    var_36 = var_4 + var_5
    var_37 = var_36 + var_7
    var_38 = module_1.digest()
    var_39 = 'old-key'
    var_40 = 'new-key'
    var_41 = [var_39, var_40]
    var_42 = module_0.Signer(var_41, var_1)
    var_43 = var_42.derive_key()
    var_44 = var_4 + var_5
    var_45 = b'new-key'
    var_46 = var_44 + var_45
    var_47 = module_1.digest()
    var_48 = [var_39, var_40]
    var_49 = module_0.Signer(var_48, var_1)
    var_50 = var_49.derive_key(var_39)
    var_51 = var_4 + var_5
    var_52 = b'old-key'
    var_53 = var_51 + var_52
    var_54 = module_1.digest()
    var_55 = 'unknown'
    var_56 = module_0.Signer(var_0, var_1, key_derivation=var_55)
    var_57 = var_56.derive_key()



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'other-secret'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'other-secret'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = module_0.Signer(var_7, var_4)
    var_30 = var_29.derive_key()
    var_31 = var_4 + var_5
    var_32 = var_31 + var_7
    var_33 = module_1.digest()
    var_34 = module_0.Signer(var_0, var_4)
    var_35 = var_34.derive_key()
    var_36 = var_4 + var_5
    var_37 = var_36 + var_7
    var_38 = module_1.digest()
    var_39 = 'invalid'
    var_40 = module_0.Signer(var_0, var_1, key_derivation=var_39)
    var_41 = var_40.derive_key()



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'test-value'
    var_14 = module_0.Signer(var_9)
    var_15 = var_14.get_signature(var_13)
    var_16 = module_0.Signer(var_10)
    var_17 = var_16.get_signature(var_13)
    var_18 = var_12.verify_signature(var_13, var_15)
    assert var_18 is True
    var_19 = var_12.verify_signature(var_13, var_17)
    assert var_19 is True
    var_20 = 'secret-key'
    var_21 = b'test-value'
    var_22 = var_1.get_signature(var_21)
    var_23 = var_1.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'test-value'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_20, algorithm=var_25)
    var_27 = b'test-value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = module_0.Signer(var_20)
    var_31 = b'test-value'
    var_32 = b'test-signature'
    var_33 = module_1.base64_encode(var_32)
    var_34 = var_30.verify_signature(var_31, var_33)
    assert var_34 is False



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = var_2.salt
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = b'test-salt'
    var_14 = var_13 + var_7
    var_15 = module_1.digest()
    var_16 = 'hmac'
    var_17 = module_0.Signer(var_0, var_1, key_derivation=var_16)
    var_18 = var_17.derive_key()
    var_19 = var_17.digest_method
    var_20 = module_1.new(var_7, digestmod=var_19)
    var_21 = module_1.digest()
    var_22 = 'none'
    var_23 = module_0.Signer(var_0, var_1, key_derivation=var_22)
    var_24 = var_23.derive_key()
    assert var_24 == b'secret-key'
    var_25 = module_0.Signer(var_0, var_1)
    var_26 = 'other-secret'
    var_27 = var_25.derive_key(var_26)
    var_28 = var_13 + var_5
    var_29 = b'other-secret'
    var_30 = var_28 + var_29
    var_31 = module_1.digest()
    var_32 = module_0.Signer(var_7, var_13)
    var_33 = var_32.derive_key()
    var_34 = var_13 + var_5
    var_35 = var_34 + var_7
    var_36 = module_1.digest()
    var_37 = 'invalid'
    var_38 = module_0.Signer(var_0, var_1, key_derivation=var_37)
    var_39 = var_38.derive_key()



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'not-base64!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = module_0.Signer(var_9)
    var_14 = var_13.get_signature(var_2)
    var_15 = module_0.Signer(var_10)
    var_16 = var_15.get_signature(var_2)
    var_17 = var_12.verify_signature(var_2, var_14)
    assert var_17 is True
    var_18 = var_12.verify_signature(var_2, var_16)
    assert var_18 is True
    var_19 = b'wrong-sig'
    var_20 = var_12.verify_signature(var_2, var_19)
    assert var_20 is False
    var_21 = 'secret-key'
    var_22 = var_1.get_signature(var_2)
    var_23 = var_1.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_21, algorithm=var_24)
    var_26 = var_25.get_signature(var_2)
    var_27 = var_25.verify_signature(var_2, var_26)
    assert var_27 is True



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.derive_key()
    var_3 = var_1.salt
    var_4 = b'signer'
    var_5 = var_3 + var_4
    var_6 = b'secret-key'
    var_7 = var_5 + var_6
    var_8 = module_1.digest()
    var_9 = 'concat'
    var_10 = module_0.Signer(var_0, key_derivation=var_9)
    var_11 = var_10.derive_key()
    var_12 = var_10.salt
    var_13 = var_12 + var_6
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, key_derivation=var_15)
    var_17 = var_16.digest_method
    var_18 = module_1.new(var_6, digestmod=var_17)
    var_19 = var_16.salt
    var_20 = var_16.derive_key()
    var_21 = module_1.digest()
    var_22 = 'none'
    var_23 = module_0.Signer(var_0, key_derivation=var_22)
    var_24 = var_23.derive_key()
    assert var_24 == b'secret-key'
    var_25 = 'django-concat'
    var_26 = module_0.Signer(var_0, key_derivation=var_25)
    var_27 = 'another-secret'
    var_28 = var_26.derive_key(var_27)
    var_29 = var_26.salt
    var_30 = var_29 + var_4
    var_31 = b'another-secret'
    var_32 = var_30 + var_31
    var_33 = module_1.digest()
    var_34 = module_0.Signer(var_6, key_derivation=var_25)
    var_35 = var_34.derive_key()
    var_36 = var_34.salt
    var_37 = var_36 + var_4
    var_38 = var_37 + var_6
    var_39 = module_1.digest()
    var_40 = 'custom-salt'
    var_41 = module_0.Signer(var_0, var_40, key_derivation=var_25)
    var_42 = var_41.derive_key()
    var_43 = b'custom-salt'
    var_44 = var_43 + var_4
    var_45 = var_44 + var_6
    var_46 = module_1.digest()
    var_47 = 'unknown'
    var_48 = module_0.Signer(var_0, key_derivation=var_47)
    var_49 = var_48.derive_key()



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_0._lazy_sha1(var_8)
    var_10 = module_1.digest()
    var_11 = 'concat'
    var_12 = module_0.Signer(var_0, var_1, key_derivation=var_11)
    var_13 = var_12.derive_key()
    var_14 = var_4 + var_7
    var_15 = module_0._lazy_sha1(var_14)
    var_16 = module_1.digest()
    var_17 = 'hmac'
    var_18 = module_0.Signer(var_0, var_1, key_derivation=var_17)
    var_19 = var_18.derive_key()
    var_20 = module_1.digest()
    var_21 = 'none'
    var_22 = module_0.Signer(var_0, var_1, key_derivation=var_21)
    var_23 = var_22.derive_key()
    assert var_23 == b'secret-key'
    var_24 = module_0.Signer(var_0, var_1)
    var_25 = 'custom-key'
    var_26 = var_24.derive_key(var_25)
    var_27 = var_4 + var_5
    var_28 = b'custom-key'
    var_29 = var_27 + var_28
    var_30 = module_0._lazy_sha1(var_29)
    var_31 = module_1.digest()
    var_32 = module_0.Signer(var_7, var_4)
    var_33 = var_32.derive_key()
    var_34 = var_4 + var_5
    var_35 = var_34 + var_7
    var_36 = module_0._lazy_sha1(var_35)
    var_37 = module_1.digest()
    var_38 = module_0.Signer(var_0, var_4)
    var_39 = var_38.derive_key()
    var_40 = var_4 + var_5
    var_41 = var_40 + var_7
    var_42 = module_0._lazy_sha1(var_41)
    var_43 = module_1.digest()
    var_44 = 'unknown'
    var_45 = module_0.Signer(var_0, var_1, key_derivation=var_44)
    var_46 = var_45.derive_key()



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'test-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'test-value'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is False
    var_17 = 'hmac'
    var_18 = module_0.Signer(var_0, key_derivation=var_17)
    var_19 = b'test-value'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_0, algorithm=var_22)
    var_24 = b'test-value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'invalid-base64!'
    var_28 = var_1.verify_signature(var_24, var_27)
    assert var_28 is False



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'different-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_12.verify_signature(var_2, var_13)
    assert var_14 is True
    var_15 = b'not-base64!'
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'hmac'
    var_18 = module_0.Signer(var_0, key_derivation=var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = var_1.verify_signature(var_2, var_19)
    assert var_21 is False
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_0, algorithm=var_22)
    var_24 = var_23.get_signature(var_2)
    var_25 = var_23.verify_signature(var_2, var_24)
    assert var_25 is True



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'other-secret'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'other-secret'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = module_0.Signer(var_7, var_4)
    var_30 = var_29.derive_key()
    var_31 = var_4 + var_5
    var_32 = var_31 + var_7
    var_33 = module_1.digest()
    var_34 = module_0.Signer(var_0, var_4)
    var_35 = var_34.derive_key()
    var_36 = var_4 + var_5
    var_37 = var_36 + var_7
    var_38 = module_1.digest()
    var_39 = 'invalid'
    var_40 = module_0.Signer(var_0, var_1, key_derivation=var_39)
    var_41 = var_40.derive_key()



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1
import hmac as module_2

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'test-value'
    var_15 = var_13.get_signature(var_14)
    var_16 = module_1.want_bytes(var_10)
    var_17 = var_13.derive_key(var_16)
    var_18 = var_13.digest_method
    var_19 = module_2.new(var_17, var_14, var_18)
    var_20 = module_2.digest()
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_13.verify_signature(var_14, var_21)
    assert var_22 is True
    var_23 = var_13.verify_signature(var_14, var_15)
    assert var_23 is True
    var_24 = 'secret-key'
    var_25 = b'test-value'
    var_26 = var_1.get_signature(var_25)
    var_27 = var_1.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_24, algorithm=var_28)
    var_30 = b'test-value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'concat'
    var_2 = module_0.Signer(var_0, key_derivation=var_1)
    var_3 = var_2.derive_key()
    var_4 = b'itsdangerous.Signer'
    var_5 = b'secret'
    var_6 = var_4 + var_5
    var_7 = module_1.digest()
    var_8 = 'django-concat'
    var_9 = module_0.Signer(var_0, key_derivation=var_8)
    var_10 = var_9.derive_key()
    var_11 = b'signer'
    var_12 = var_4 + var_11
    var_13 = var_12 + var_5
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret'
    var_22 = 'custom'
    var_23 = module_0.Signer(var_0, var_22, key_derivation=var_1)
    var_24 = var_23.derive_key()
    var_25 = b'custom'
    var_26 = var_25 + var_5
    var_27 = module_1.digest()
    var_28 = module_0.Signer(var_0, key_derivation=var_1)
    var_29 = var_28.derive_key(var_22)
    var_30 = var_4 + var_25
    var_31 = module_1.digest()
    var_32 = module_0.Signer(var_5, key_derivation=var_1)
    var_33 = var_32.derive_key()
    var_34 = var_4 + var_5
    var_35 = module_1.digest()
    var_36 = module_0.Signer(var_0, var_25, key_derivation=var_1)
    var_37 = var_36.derive_key()
    var_38 = var_25 + var_5
    var_39 = module_1.digest()
    var_40 = 'invalid'
    var_41 = module_0.Signer(var_0, key_derivation=var_40)
    var_42 = var_41.derive_key()



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'another-secret-key'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'another-secret-key'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = module_0.Signer(var_7, var_1)
    var_30 = var_29.derive_key()
    var_31 = var_4 + var_5
    var_32 = var_31 + var_7
    var_33 = module_1.digest()
    var_34 = module_0.Signer(var_0, var_4)
    var_35 = var_34.derive_key()
    var_36 = var_4 + var_5
    var_37 = var_36 + var_7
    var_38 = module_1.digest()
    var_39 = 'unknown'
    var_40 = module_0.Signer(var_0, var_1, key_derivation=var_39)
    var_41 = var_40.derive_key()



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.derive_key()
    var_3 = var_1.salt
    var_4 = b'signer'
    var_5 = var_3 + var_4
    var_6 = b'secret'
    var_7 = var_5 + var_6
    var_8 = module_1.digest()
    var_9 = 'concat'
    var_10 = module_0.Signer(var_0, key_derivation=var_9)
    var_11 = var_10.derive_key()
    var_12 = var_10.salt
    var_13 = var_12 + var_6
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, key_derivation=var_15)
    var_17 = var_16.digest_method
    var_18 = module_1.new(var_6, digestmod=var_17)
    var_19 = var_16.salt
    var_20 = var_16.derive_key()
    var_21 = module_1.digest()
    var_22 = 'none'
    var_23 = module_0.Signer(var_0, key_derivation=var_22)
    var_24 = var_23.derive_key()
    assert var_24 == b'secret'
    var_25 = module_0.Signer(var_0)
    var_26 = b'other'
    var_27 = var_25.derive_key(var_26)
    var_28 = var_25.salt
    var_29 = var_28 + var_4
    var_30 = var_29 + var_26
    var_31 = module_1.digest()
    var_32 = module_0.Signer(var_6)
    var_33 = var_32.derive_key()
    var_34 = var_32.salt
    var_35 = var_34 + var_4
    var_36 = var_35 + var_6
    var_37 = module_1.digest()
    var_38 = 'unknown'
    var_39 = module_0.Signer(var_0, key_derivation=var_38)
    var_40 = var_39.derive_key()



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotated-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = var_15.salt
    var_20 = module_0.Signer(var_12, var_19)
    var_21 = var_20.get_signature(var_16)
    var_22 = var_15.verify_signature(var_16, var_21)
    assert var_22 is True
    var_23 = 'secret-key'
    var_24 = b'derivation-test'
    var_25 = var_1.get_signature(var_24)
    var_26 = var_1.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'sha256-test'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_23, algorithm=var_28)
    var_30 = b'none-algo-test'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'some-signature'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotated-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = module_0.Signer(var_12)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_20)
    assert var_21 is True
    var_22 = var_15.verify_signature(var_16, var_5)
    assert var_22 is False
    var_23 = 'secret-key'
    var_24 = b'test-value'
    var_25 = var_1.get_signature(var_24)
    var_26 = var_1.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'test-value'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_23, algorithm=var_28)
    var_30 = b'test-value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24, var_1)
    var_26 = var_25.derive_key(var_22)
    var_27 = var_4 + var_5
    var_28 = b'old-key'
    var_29 = var_27 + var_28
    var_30 = module_1.digest()



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'rotated-value'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = module_0.Signer(var_10)
    var_18 = var_17.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_18)
    assert var_19 is True
    var_20 = b'invalid-base64!'
    var_21 = var_1.verify_signature(var_14, var_20)
    assert var_21 is False
    var_22 = 'secret-key'
    var_23 = b'test-value'
    var_24 = var_1.get_signature(var_23)
    var_25 = var_1.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'test-value'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_22, algorithm=var_27)
    var_29 = b'test-value'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'test-value-multi'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = module_0.Signer(var_10)
    var_18 = var_17.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_18)
    assert var_19 is True
    var_20 = 'wrong-key'
    var_21 = module_0.Signer(var_20)
    var_22 = var_21.get_signature(var_2)
    var_23 = var_1.verify_signature(var_2, var_22)
    assert var_23 is False
    var_24 = 'hmac'
    var_25 = module_0.Signer(var_0, key_derivation=var_24)
    var_26 = var_25.get_signature(var_2)
    var_27 = var_25.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = var_1.verify_signature(var_2, var_26)
    assert var_28 is False
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_0, algorithm=var_29)
    var_31 = var_30.get_signature(var_2)
    var_32 = var_30.verify_signature(var_2, var_31)
    assert var_32 is True
    var_33 = var_1.verify_signature(var_2, var_31)
    assert var_33 is False



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    var_8 = module_1.base64_encode(var_6)
    var_9 = var_2.verify_signature(var_3, var_8)
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = var_13.get_signature(var_3)
    var_15 = var_13.verify_signature(var_3, var_14)
    var_16 = 'secret-key'
    var_17 = 'test-salt'
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_16, var_17, algorithm=var_18)
    var_20 = var_19.get_signature(var_3)
    var_21 = var_19.verify_signature(var_3, var_20)



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'old-key'
    var_15 = var_13.get_signature(var_2)
    var_16 = b'new-key'
    var_17 = var_13.get_signature(var_2)
    var_18 = var_13.verify_signature(var_2, var_15)
    assert var_18 is True
    var_19 = var_13.verify_signature(var_2, var_17)
    assert var_19 is True
    var_20 = 'secret-key'
    var_21 = var_1.get_signature(var_2)
    var_22 = var_1.verify_signature(var_2, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_20, algorithm=var_23)
    var_25 = var_24.get_signature(var_2)
    var_26 = var_24.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = b'invalid-base64!'
    var_28 = var_1.verify_signature(var_2, var_27)
    assert var_28 is False



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.salt
    var_4 = b'signer'
    var_5 = var_3 + var_4
    var_6 = b'secret-key'
    var_7 = var_5 + var_6
    var_8 = module_1.digest()
    var_9 = var_2.derive_key()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = b'test-salt'
    var_13 = var_12 + var_6
    var_14 = module_1.digest()
    var_15 = var_11.derive_key()
    var_16 = 'hmac'
    var_17 = module_0.Signer(var_0, var_1, key_derivation=var_16)
    var_18 = var_17.digest_method
    var_19 = module_1.new(var_6, digestmod=var_18)
    var_20 = module_1.digest()
    var_21 = var_17.derive_key()
    var_22 = 'none'
    var_23 = module_0.Signer(var_0, var_1, key_derivation=var_22)
    var_24 = var_23.derive_key()
    assert var_24 == b'secret-key'
    var_25 = module_0.Signer(var_0, var_1)
    var_26 = 'other-key'
    var_27 = var_25.derive_key(var_26)
    var_28 = var_12 + var_4
    var_29 = b'other-key'
    var_30 = var_28 + var_29
    var_31 = module_1.digest()
    var_32 = module_0.Signer(var_6, var_12)
    var_33 = var_12 + var_4
    var_34 = var_33 + var_6
    var_35 = module_1.digest()
    var_36 = var_32.derive_key()
    var_37 = module_0.Signer(var_0, var_12)
    var_38 = var_12 + var_4
    var_39 = var_38 + var_6
    var_40 = module_1.digest()
    var_41 = var_37.derive_key()
    var_42 = 'unknown'
    var_43 = module_0.Signer(var_0, var_1, key_derivation=var_42)
    var_44 = var_43.derive_key()



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'old-key'
    var_15 = var_13.get_signature(var_2)
    var_16 = b'new-key'
    var_17 = var_13.get_signature(var_2)
    var_18 = var_13.verify_signature(var_2, var_15)
    assert var_18 is True
    var_19 = var_13.verify_signature(var_2, var_17)
    assert var_19 is True
    var_20 = 'secret-key'
    var_21 = b'invalid-base64!'
    var_22 = var_1.verify_signature(var_2, var_21)
    assert var_22 is False



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'old-key'
    var_14 = var_12.get_signature(var_2)
    var_15 = b'new-key'
    var_16 = var_12.get_signature(var_2)
    var_17 = var_12.verify_signature(var_2, var_14)
    assert var_17 is True
    var_18 = var_12.verify_signature(var_2, var_16)
    assert var_18 is True
    var_19 = var_12.verify_signature(var_2, var_5)
    assert var_19 is False
    var_20 = 'secret-key'
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_20, algorithm=var_21)
    var_23 = var_22.get_signature(var_2)
    var_24 = var_22.verify_signature(var_2, var_23)
    assert var_24 is True



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'wrong-sig'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_2.verify_signature(var_3, var_9)
    assert var_10 is False
    var_11 = b'old-secret-key'
    var_12 = b'new-secret-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13, var_1)
    var_15 = b'test-value'
    var_16 = module_0.Signer(var_11, var_1)
    var_17 = var_16.get_signature(var_15)
    var_18 = module_0.Signer(var_12, var_1)
    var_19 = var_18.get_signature(var_15)
    var_20 = var_14.verify_signature(var_15, var_17)
    assert var_20 is True
    var_21 = var_14.verify_signature(var_15, var_19)
    assert var_21 is True
    var_22 = b'invalid-base64!'
    var_23 = var_14.verify_signature(var_15, var_22)
    assert var_23 is False
    var_24 = 'secret-key'
    var_25 = 'test-salt'
    var_26 = b'test-value'
    var_27 = var_14.get_signature(var_26)
    var_28 = var_14.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_24, var_25, algorithm=var_29)
    var_31 = b'test-value'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = var_1.derive_key()
    var_3 = len(var_2)
    var_4 = 'concat'
    var_5 = module_0.Signer(var_0, key_derivation=var_4)
    var_6 = var_5.derive_key()
    var_7 = len(var_6)
    var_8 = 'hmac'
    var_9 = module_0.Signer(var_0, key_derivation=var_8)
    var_10 = var_9.derive_key()
    var_11 = len(var_10)
    var_12 = 'none'
    var_13 = module_0.Signer(var_0, key_derivation=var_12)
    var_14 = var_13.derive_key()
    assert var_14 == b'secret-key'
    var_15 = 'custom-secret'
    var_16 = var_1.derive_key(var_15)
    var_17 = len(var_16)
    var_18 = b'secret-key'
    var_19 = module_0.Signer(var_18)
    var_20 = var_19.derive_key()
    var_21 = len(var_20)
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24)
    var_26 = var_25.derive_key()
    var_27 = len(var_26)
    var_28 = 'secret-key'
    var_29 = 'unknown'
    var_30 = module_0.Signer(var_28, key_derivation=var_29)
    var_31 = var_30.derive_key()



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'some-signature'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotated-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is False
    var_19 = 'secret-key'
    var_20 = b'derivation-test'
    var_21 = var_1.get_signature(var_20)
    var_22 = var_1.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'custom-digest-test'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_19, algorithm=var_24)
    var_26 = b'none-algorithm-test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = var_13.get_signature(var_2)
    var_15 = var_13.get_signature(var_2)
    var_16 = var_13.verify_signature(var_2, var_14)
    assert var_16 is True
    var_17 = var_13.verify_signature(var_2, var_15)
    assert var_17 is True
    var_18 = var_13.verify_signature(var_2, var_7)
    assert var_18 is False
    var_19 = 'secret-key'
    var_20 = var_1.get_signature(var_2)
    var_21 = var_1.verify_signature(var_2, var_20)
    assert var_21 is True
    var_22 = b'wrong-sig'
    var_23 = var_1.verify_signature(var_2, var_22)
    assert var_23 is False
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_19, algorithm=var_24)
    var_26 = b''
    var_27 = var_25.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = b'any-sig'
    var_29 = var_25.verify_signature(var_2, var_28)
    assert var_29 is False



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'secret-key'
    var_8 = var_6 + var_7
    var_9 = module_1.digest()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = var_4 + var_7
    var_14 = module_1.digest()
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = module_1.digest()
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'specific-key'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'specific-key'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = module_0.Signer(var_7, var_4)
    var_30 = var_29.derive_key()
    var_31 = var_4 + var_5
    var_32 = var_31 + var_7
    var_33 = module_1.digest()
    var_34 = var_29.derive_key()
    var_35 = var_4 + var_5
    var_36 = var_35 + var_7
    var_37 = module_1.digest()
    var_38 = 'old-key'
    var_39 = 'new-key'
    var_40 = [var_38, var_39]
    var_41 = module_0.Signer(var_40, var_1)
    var_42 = var_41.derive_key()
    var_43 = var_4 + var_5
    var_44 = b'new-key'
    var_45 = var_43 + var_44
    var_46 = module_1.digest()
    var_47 = [var_38, var_39]
    var_48 = module_0.Signer(var_47, var_1)
    var_49 = var_48.derive_key(var_38)
    var_50 = var_4 + var_5
    var_51 = b'old-key'
    var_52 = var_50 + var_51
    var_53 = module_1.digest()
    var_54 = None
    var_55 = module_0.Signer(var_0, var_54)
    var_56 = var_55.derive_key()
    var_57 = b'itsdangerous.Signer'
    var_58 = var_57 + var_5
    var_59 = var_58 + var_7
    var_60 = module_1.digest()



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong_signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'old-key'
    var_15 = var_13.get_signature(var_2)
    var_16 = b'new-key'
    var_17 = var_13.get_signature(var_2)
    var_18 = var_13.verify_signature(var_2, var_15)
    assert var_18 is True
    var_19 = var_13.verify_signature(var_2, var_17)
    assert var_19 is True
    var_20 = var_13.verify_signature(var_2, var_7)
    assert var_20 is False
    var_21 = 'secret-key'
    var_22 = var_1.get_signature(var_2)
    var_23 = var_1.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = b'wrong'
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_21, algorithm=var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_27.verify_signature(var_2, var_28)
    assert var_29 is True
    var_30 = var_27.verify_signature(var_2, var_7)
    assert var_30 is False



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'different-value'
    var_11 = var_1.verify_signature(var_10, var_3)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = module_0.Signer(var_12)
    var_17 = module_0.Signer(var_13)
    var_18 = b'old-value'
    var_19 = b'new-value'
    var_20 = var_16.get_signature(var_18)
    var_21 = var_17.get_signature(var_19)
    var_22 = var_15.verify_signature(var_18, var_20)
    assert var_22 is True
    var_23 = var_15.verify_signature(var_19, var_21)
    assert var_23 is True
    var_24 = var_15.verify_signature(var_19, var_20)
    assert var_24 is False
    var_25 = var_15.verify_signature(var_18, var_21)
    assert var_25 is False
    var_26 = 'secret-key'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = var_28.get_signature(var_2)
    var_30 = var_28.verify_signature(var_2, var_29)
    assert var_30 is True
    var_31 = b''
    var_32 = var_28.verify_signature(var_2, var_31)
    assert var_32 is True



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'different-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is False
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_0, algorithm=var_18)
    var_20 = b'test-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b''
    var_24 = var_19.verify_signature(var_20, var_23)
    assert var_24 is True
    var_25 = 'secret-key'
    var_26 = b'test-value'
    var_27 = var_1.get_signature(var_26)
    var_28 = var_1.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'invalid-base64!'
    var_30 = var_1.verify_signature(var_26, var_29)
    assert var_30 is False



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'test-value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'secret-key'
    var_17 = b'test-value'
    var_18 = var_1.get_signature(var_17)
    var_19 = var_1.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = b'test-value'
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_16, algorithm=var_21)
    var_23 = b'test-value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotated-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = var_15.get_signature(var_16)
    var_20 = var_15.verify_signature(var_16, var_19)
    assert var_20 is True
    var_21 = 'secret-key'
    var_22 = b'derivation-test'
    var_23 = b'sha256-test'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_21, algorithm=var_24)
    var_26 = b'none-algo-test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_12.verify_signature(var_2, var_13)
    assert var_14 is True
    var_15 = 'secret-key'
    var_16 = module_0.NoneAlgorithm()
    var_17 = module_0.Signer(var_15, algorithm=var_16)
    var_18 = var_17.get_signature(var_2)
    var_19 = var_17.verify_signature(var_2, var_18)
    assert var_19 is True
    var_20 = b''
    var_21 = var_1.get_signature(var_20)
    var_22 = var_1.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = var_1.verify_signature(var_2, var_20)
    assert var_23 is False



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = b'invalid'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = module_0.Signer(var_12)
    var_17 = module_0.Signer(var_13)
    var_18 = b'old-value'
    var_19 = b'new-value'
    var_20 = var_16.get_signature(var_18)
    var_21 = var_17.get_signature(var_19)
    var_22 = var_15.verify_signature(var_18, var_20)
    assert var_22 is True
    var_23 = var_15.verify_signature(var_19, var_21)
    assert var_23 is True
    var_24 = var_15.verify_signature(var_18, var_21)
    assert var_24 is False
    var_25 = var_15.verify_signature(var_19, var_20)
    assert var_25 is False
    var_26 = 'secret-key'
    var_27 = b'test-value'
    var_28 = var_1.get_signature(var_27)
    var_29 = var_1.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_26, algorithm=var_30)
    var_32 = b'test-value'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotated-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = var_15.get_signature(var_16)
    var_20 = var_15.verify_signature(var_16, var_19)
    assert var_20 is True
    var_21 = 'secret-key'
    var_22 = b'derivation-test'
    var_23 = var_1.get_signature(var_22)
    var_24 = var_1.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'sha256-test'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_21, algorithm=var_26)
    var_28 = b'none-algo-test'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = var_13.get_signature(var_2)
    var_15 = var_13.verify_signature(var_2, var_14)
    assert var_15 is True
    var_16 = 'hmac'
    var_17 = module_0.Signer(var_0, key_derivation=var_16)
    var_18 = var_17.get_signature(var_2)
    var_19 = var_17.verify_signature(var_2, var_18)
    assert var_19 is True
    var_20 = var_1.verify_signature(var_2, var_18)
    assert var_20 is False
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_0, algorithm=var_21)
    var_23 = var_22.get_signature(var_2)
    var_24 = var_22.verify_signature(var_2, var_23)
    assert var_24 is True
    var_25 = var_1.verify_signature(var_2, var_23)
    assert var_25 is False



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'old-key'
    var_14 = var_12.get_signature(var_2)
    var_15 = b'new-key'
    var_16 = var_12.get_signature(var_2)
    var_17 = var_12.verify_signature(var_2, var_14)
    assert var_17 is True
    var_18 = var_12.verify_signature(var_2, var_16)
    assert var_18 is True
    var_19 = var_12.verify_signature(var_2, var_5)
    assert var_19 is False
    var_20 = 'secret-key'
    var_21 = b'invalid-sig'
    var_22 = 'secret-key'
    var_23 = b'invalid-sig'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_22, algorithm=var_24)
    var_26 = var_25.get_signature(var_2)
    var_27 = var_25.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = var_25.verify_signature(var_2, var_23)
    assert var_28 is False



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'test_salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test_value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid_signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'different_value'
    var_9 = var_2.verify_signature(var_8, var_4)
    assert var_9 is False
    var_10 = b'invalid'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = b'old_secret'
    var_14 = b'new_secret'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15, var_1)
    var_17 = var_16.get_signature(var_3)
    var_18 = var_16.verify_signature(var_3, var_17)
    assert var_18 is True
    var_19 = module_0.Signer(var_13, var_1)
    var_20 = var_19.get_signature(var_3)
    var_21 = var_16.verify_signature(var_3, var_20)
    assert var_21 is True
    var_22 = 'hmac'
    var_23 = module_0.Signer(var_0, var_1, key_derivation=var_22)
    var_24 = var_23.get_signature(var_3)
    var_25 = var_23.verify_signature(var_3, var_24)
    assert var_25 is True
    var_26 = var_2.verify_signature(var_3, var_24)
    assert var_26 is False
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, var_1, algorithm=var_27)
    var_29 = var_28.get_signature(var_3)
    var_30 = var_28.verify_signature(var_3, var_29)
    assert var_30 is True



# Parsed testcases at query #41
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'some-signature'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotated-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.salt
    var_19 = module_0.Signer(var_12, var_18)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_20)
    assert var_21 is True
    var_22 = var_15.get_signature(var_16)
    var_23 = var_15.verify_signature(var_16, var_22)
    assert var_23 is True
    var_24 = 'secret-key'
    var_25 = b'derivation-test'
    var_26 = b'sha256-test'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_24, algorithm=var_27)
    var_29 = b'none-algo-test'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'different-value'
    var_9 = var_2.verify_signature(var_8, var_4)
    assert var_9 is False
    var_10 = b'not-base64!'
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = b'test-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'hmac'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = b'test-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_0, var_1, algorithm=var_24)
    var_26 = b'test-value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True



# Parsed testcases at query #43
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1
import hmac as module_2

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = module_1.base64_encode(var_6)
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = b'test-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = module_1.want_bytes(var_12)
    var_19 = var_15.derive_key(var_18)
    var_20 = var_15.digest_method
    var_21 = module_2.new(var_19, var_16, var_20)
    var_22 = module_2.digest()
    var_23 = module_1.base64_encode(var_22)
    var_24 = var_15.verify_signature(var_16, var_23)
    assert var_24 is True
    var_25 = var_15.verify_signature(var_16, var_17)
    assert var_25 is True
    var_26 = 'secret-key'
    var_27 = 'test-salt'
    var_28 = b'test-value'
    var_29 = var_2.get_signature(var_28)
    var_30 = var_2.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b'test-value'
    var_32 = module_0.NoneAlgorithm()
    var_33 = module_0.Signer(var_26, var_27, algorithm=var_32)
    var_34 = b'test-value'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'wrong-sig'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_2.verify_signature(var_3, var_9)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13, var_1)
    var_15 = b'test-value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is False
    var_18 = 'secret-key'
    var_19 = 'test-salt'
    var_20 = b'test-value'
    var_21 = var_2.get_signature(var_20)
    var_22 = var_2.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_18, var_19, algorithm=var_23)
    var_25 = b'test-value'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'test-value'



# Parsed testcases at query #45
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'wrong-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = var_13.get_signature(var_2)
    var_15 = var_13.get_signature(var_2)
    var_16 = var_13.verify_signature(var_2, var_14)
    assert var_16 is True
    var_17 = var_13.verify_signature(var_2, var_15)
    assert var_17 is True
    var_18 = 'secret-key'
    var_19 = var_13.get_signature(var_2)
    var_20 = var_13.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = var_13.get_signature(var_2)
    var_22 = var_13.verify_signature(var_2, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_18, algorithm=var_23)
    var_25 = var_24.get_signature(var_2)
    var_26 = var_24.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = b''
    var_28 = var_24.verify_signature(var_2, var_27)
    assert var_28 is False



