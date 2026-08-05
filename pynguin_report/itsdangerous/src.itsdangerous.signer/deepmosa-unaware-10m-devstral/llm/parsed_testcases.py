####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'itsdangerous.Signer'
    var_3 = b'signer'
    var_4 = var_2 + var_3
    var_5 = b'secret-key'
    var_6 = var_4 + var_5
    var_7 = module_0._lazy_sha1(var_6)
    var_8 = module_1.digest()
    var_9 = var_1.derive_key()
    var_10 = 'concat'
    var_11 = module_0.Signer(var_0, key_derivation=var_10)
    var_12 = var_2 + var_5
    var_13 = module_0._lazy_sha1(var_12)
    var_14 = module_1.digest()
    var_15 = var_11.derive_key()
    var_16 = 'hmac'
    var_17 = module_0.Signer(var_0, key_derivation=var_16)
    var_18 = module_1.digest()
    var_19 = var_17.derive_key()
    var_20 = 'none'
    var_21 = module_0.Signer(var_0, key_derivation=var_20)
    var_22 = var_21.derive_key()
    assert var_22 == b'secret-key'
    var_23 = 'custom-salt'
    var_24 = module_0.Signer(var_0, var_23)
    var_25 = b'custom-salt'
    var_26 = var_25 + var_3
    var_27 = var_26 + var_5
    var_28 = module_0._lazy_sha1(var_27)
    var_29 = module_1.digest()
    var_30 = var_24.derive_key()
    var_31 = module_0.Signer(var_0)
    var_32 = 'other-key'
    var_33 = var_31.derive_key(var_32)
    var_34 = var_2 + var_3
    var_35 = b'other-key'
    var_36 = var_34 + var_35
    var_37 = module_0._lazy_sha1(var_36)
    var_38 = module_1.digest()
    var_39 = module_0.Signer(var_5)
    var_40 = var_2 + var_3
    var_41 = var_40 + var_5
    var_42 = module_0._lazy_sha1(var_41)
    var_43 = module_1.digest()
    var_44 = var_39.derive_key()
    var_45 = 'old-key'
    var_46 = 'new-key'
    var_47 = [var_45, var_46]
    var_48 = module_0.Signer(var_47)
    var_49 = var_2 + var_3
    var_50 = b'new-key'
    var_51 = var_49 + var_50
    var_52 = module_0._lazy_sha1(var_51)
    var_53 = module_1.digest()
    var_54 = var_48.derive_key()
    var_55 = var_48.derive_key(var_45)
    var_56 = var_2 + var_3
    var_57 = b'old-key'
    var_58 = var_56 + var_57
    var_59 = module_0._lazy_sha1(var_58)
    var_60 = module_1.digest()
    var_61 = 'invalid'
    var_62 = module_0.Signer(var_0, key_derivation=var_61)
    var_63 = var_62.derive_key()



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = b'test-value.incorrect-signature'
    var_6 = var_1.unsign(var_5)
    var_7 = b'test-value'
    var_8 = var_1.unsign(var_7)
    var_9 = b'|'
    var_10 = module_0.Signer(var_7, sep=var_9)
    var_11 = b'test|value'
    var_12 = var_10.sign(var_11)
    var_13 = var_10.unsign(var_12)
    assert var_13 == b'test|value'
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = module_0.Signer(var_14)
    var_19 = b'test-value'
    var_20 = var_18.sign(var_19)
    var_21 = module_0.Signer(var_15)
    var_22 = var_21.sign(var_19)
    var_23 = var_17.unsign(var_20)
    assert var_23 == b'test-value'
    var_24 = var_17.unsign(var_22)
    assert var_24 == b'test-value'
    var_25 = 'hmac'
    var_26 = module_0.Signer(var_7, key_derivation=var_25)
    var_27 = var_26.sign(var_19)
    var_28 = var_26.unsign(var_27)
    assert var_28 == b'test-value'
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_7, algorithm=var_29)
    var_31 = var_30.sign(var_19)
    var_32 = var_30.unsign(var_31)
    assert var_32 == b'test-value'



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = len(var_3)
    var_5 = 'concat'
    var_6 = module_0.Signer(var_0, var_1, key_derivation=var_5)
    var_7 = var_6.derive_key()
    var_8 = len(var_7)
    var_9 = 'hmac'
    var_10 = module_0.Signer(var_0, var_1, key_derivation=var_9)
    var_11 = var_10.derive_key()
    var_12 = len(var_11)
    var_13 = 'none'
    var_14 = module_0.Signer(var_0, var_1, key_derivation=var_13)
    var_15 = var_14.derive_key()
    assert var_15 == b'secret-key'
    var_16 = module_0.Signer(var_0, var_1)
    var_17 = 'another-secret'
    var_18 = var_16.derive_key(var_17)
    var_19 = len(var_18)
    var_20 = 'invalid'
    var_21 = module_0.Signer(var_0, var_1, key_derivation=var_20)
    var_22 = var_21.derive_key()



# Parsed testcases at query #4
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
    var_6 = b'invalid-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'wrong-value'
    var_9 = var_2.verify_signature(var_8, var_4)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = var_15.get_signature(var_3)
    var_17 = var_15.verify_signature(var_3, var_16)
    assert var_17 is True
    var_18 = 'hmac'
    var_19 = module_0.Signer(var_0, var_1, key_derivation=var_18)
    var_20 = var_19.get_signature(var_3)
    var_21 = var_19.verify_signature(var_3, var_20)
    assert var_21 is True
    var_22 = var_2.verify_signature(var_3, var_20)
    assert var_22 is False
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, var_1, algorithm=var_23)
    var_25 = var_24.get_signature(var_3)
    var_26 = var_24.verify_signature(var_3, var_25)
    assert var_26 is True
    var_27 = var_2.verify_signature(var_3, var_25)
    assert var_27 is False



# Parsed testcases at query #5
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
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = module_1.base64_encode(var_5)
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'old-secret-key'
    var_10 = b'new-secret-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'test-value'
    var_14 = module_2.digest()
    var_15 = module_1.base64_encode(var_14)
    var_16 = var_12.verify_signature(var_13, var_15)
    assert var_16 is True
    var_17 = module_2.digest()
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_12.verify_signature(var_13, var_18)
    assert var_19 is True
    var_20 = b'invalid-secret-key'
    var_21 = module_2.digest()
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_12.verify_signature(var_13, var_22)
    assert var_23 is False
    var_24 = 'secret-key'
    var_25 = b'test-value'
    var_26 = var_12.get_signature(var_25)
    var_27 = var_12.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'secret-key'
    var_29 = b'test-value'
    var_30 = var_12.get_signature(var_29)
    var_31 = var_12.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = module_0.NoneAlgorithm()
    var_33 = module_0.Signer(var_28, algorithm=var_32)
    var_34 = b'test-value'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = module_0.NoneAlgorithm()
    var_38 = module_0.Signer(var_28, algorithm=var_37)
    var_39 = b'test-value'
    var_40 = b''
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = module_0.Signer(var_28)
    var_43 = b''
    var_44 = var_42.get_signature(var_43)
    var_45 = var_42.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = module_0.Signer(var_28)
    var_47 = 'test-value'
    var_48 = var_46.get_signature(var_47)
    var_49 = var_46.verify_signature(var_47, var_48)
    assert var_49 is True



# Parsed testcases at query #6
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
    var_10 = b'invalid-base64!'
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = b'test-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = b'old-key'
    var_19 = var_15.derive_key(var_18)
    var_20 = var_15.verify_signature(var_16, var_17)
    assert var_20 is True
    var_21 = 'secret-key'
    var_22 = 'test-salt'
    var_23 = b'test-value'
    var_24 = var_2.get_signature(var_23)
    var_25 = var_2.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'test-value'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_21, var_22, algorithm=var_27)
    var_29 = b'test-value'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong_signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'test_sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid_base64'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b'old_secret'
    var_13 = b'new_secret'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'test_value_rotated'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = module_0.Signer(var_12)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_20)
    assert var_21 is True
    var_22 = b'non_existent_sig'
    var_23 = var_15.verify_signature(var_16, var_22)
    assert var_23 is False



# Parsed testcases at query #8
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
    var_6 = b'wrong-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'invalid-base64'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'old-secret-key'
    var_11 = b'new-secret-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-value'
    var_15 = module_0.Signer(var_10, var_1)
    var_16 = var_15.get_signature(var_14)
    var_17 = var_13.verify_signature(var_14, var_16)
    assert var_17 is True
    var_18 = module_0.Signer(var_11, var_1)
    var_19 = var_18.get_signature(var_14)
    var_20 = var_13.verify_signature(var_14, var_19)
    assert var_20 is True
    var_21 = b'wrong-key'
    var_22 = module_0.Signer(var_21, var_1)
    var_23 = var_22.get_signature(var_14)
    var_24 = var_13.verify_signature(var_14, var_23)
    assert var_24 is False
    var_25 = 'secret-key'
    var_26 = 'test-salt'
    var_27 = b'test-value'
    var_28 = var_13.get_signature(var_27)
    var_29 = var_13.verify_signature(var_27, var_28)
    assert var_29 is True



# Parsed testcases at query #9
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
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'invalid-base64!'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'old-secret-key'
    var_11 = b'new-secret-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-value'
    var_15 = module_0.Signer(var_10, var_1)
    var_16 = var_15.get_signature(var_14)
    var_17 = module_0.Signer(var_11, var_1)
    var_18 = var_17.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_16)
    assert var_19 is True
    var_20 = var_13.verify_signature(var_14, var_18)
    assert var_20 is True
    var_21 = 'secret-key'
    var_22 = 'test-salt'
    var_23 = b'test-value'
    var_24 = var_13.get_signature(var_23)
    var_25 = var_13.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_21, var_22, algorithm=var_26)
    var_28 = b'test-value'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True



# Parsed testcases at query #10
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
    var_21 = var_1.get_signature(var_20)
    var_22 = var_1.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_16, algorithm=var_23)
    var_25 = b'test-value'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'invalid-base64!'
    var_29 = var_24.verify_signature(var_25, var_28)
    assert var_29 is False



# Parsed testcases at query #11
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'invalid'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b'different-value'
    var_13 = var_1.verify_signature(var_12, var_3)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test-value'
    var_19 = module_0.Signer(var_14)
    var_20 = var_19.get_signature(var_18)
    var_21 = module_0.Signer(var_15)
    var_22 = var_21.get_signature(var_18)
    var_23 = var_17.verify_signature(var_18, var_20)
    assert var_23 is True
    var_24 = var_17.verify_signature(var_18, var_22)
    assert var_24 is True
    var_25 = var_17.verify_signature(var_18, var_9)
    assert var_25 is False
    var_26 = 'secret-key'
    var_27 = b'test-value'
    var_28 = var_1.get_signature(var_27)
    var_29 = var_1.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'invalid'
    var_31 = var_1.verify_signature(var_27, var_30)
    assert var_31 is False
    var_32 = b'test-value'
    var_33 = module_0.NoneAlgorithm()
    var_34 = module_0.Signer(var_26, algorithm=var_33)
    var_35 = b'test-value'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = var_34.verify_signature(var_35, var_9)
    assert var_38 is False



# Parsed testcases at query #12
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
    var_18 = 'hmac'
    var_19 = module_0.Signer(var_0, key_derivation=var_18)
    var_20 = b'test-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, algorithm=var_23)
    var_25 = b'test-value'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True



# Parsed testcases at query #13
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
    var_6 = b'invalid-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'invalid-base64!'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-value'
    var_15 = module_0.Signer(var_10, var_1)
    var_16 = var_15.get_signature(var_14)
    var_17 = module_0.Signer(var_11, var_1)
    var_18 = var_17.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_16)
    assert var_19 is True
    var_20 = var_13.verify_signature(var_14, var_18)
    assert var_20 is True
    var_21 = var_13.verify_signature(var_14, var_6)
    assert var_21 is False
    var_22 = 'secret-key'
    var_23 = 'test-salt'
    var_24 = b'test-value'
    var_25 = var_2.get_signature(var_24)
    var_26 = var_2.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_22, var_23, algorithm=var_27)
    var_29 = b'test-value'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True



# Parsed testcases at query #14
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
    var_12 = 'test-value'
    var_13 = var_1.verify_signature(var_12, var_3)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'rotated-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is False
    var_21 = 'hmac'
    var_22 = module_0.Signer(var_0, key_derivation=var_21)
    var_23 = b'hmac-value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_0, algorithm=var_26)
    var_28 = b'none-value'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True



# Parsed testcases at query #15
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
    var_10 = b'old-secret-key'
    var_11 = b'new-secret-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-value'
    var_15 = module_0.Signer(var_10, var_1)
    var_16 = var_15.get_signature(var_14)
    var_17 = module_0.Signer(var_11, var_1)
    var_18 = var_17.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_16)
    assert var_19 is True
    var_20 = var_13.verify_signature(var_14, var_18)
    assert var_20 is True
    var_21 = var_13.verify_signature(var_14, var_6)
    assert var_21 is False
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_0, var_1, algorithm=var_22)
    var_24 = b'test-value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'anything-else'
    var_28 = var_23.verify_signature(var_24, var_27)
    assert var_28 is False
    var_29 = b'test-value'
    var_30 = b'malformed-base64!'
    var_31 = var_2.verify_signature(var_29, var_30)
    assert var_31 is False



# Parsed testcases at query #16
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
    var_6 = b'invalid-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = module_1.base64_encode(var_6)
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-value'
    var_15 = module_0.Signer(var_10, var_1)
    var_16 = var_15.get_signature(var_14)
    var_17 = module_0.Signer(var_11, var_1)
    var_18 = var_17.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_16)
    assert var_19 is True
    var_20 = var_13.verify_signature(var_14, var_18)
    assert var_20 is True
    var_21 = var_13.verify_signature(var_14, var_6)
    assert var_21 is False
    var_22 = 'secret-key'
    var_23 = 'test-salt'
    var_24 = b'test-value'
    var_25 = var_2.get_signature(var_24)
    var_26 = var_2.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'invalid-sig'
    var_28 = var_2.verify_signature(var_24, var_27)
    assert var_28 is False
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_22, var_23, algorithm=var_29)
    var_31 = b'test-value'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = var_30.verify_signature(var_31, var_27)
    assert var_34 is False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = module_0.Signer(var_0)
    var_6 = b'test_value.invalid_signature'
    var_7 = var_5.unsign(var_6)
    var_8 = module_0.Signer(var_7)
    var_9 = b'test_value'
    var_10 = var_8.unsign(var_9)
    var_11 = module_0.Signer(var_10)
    var_12 = b'test_value'
    var_13 = var_11.sign(var_12)
    var_14 = b'.extra'
    var_15 = var_13 + var_14
    var_16 = var_11.unsign(var_15)
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'test_value'
    var_22 = var_20.sign(var_21)
    var_23 = var_20.unsign(var_22)
    var_24 = module_0.Signer(var_17)
    var_25 = b'test_value'
    var_26 = var_24.sign(var_25)
    var_27 = [var_17, var_18]
    var_28 = module_0.Signer(var_27)
    var_29 = var_28.unsign(var_26)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    var_5 = module_0.Signer(var_0)
    var_6 = 'test-value'
    var_7 = var_5.sign(var_6)
    var_8 = var_5.unsign(var_7)
    assert var_8 == b'test-value'
    var_9 = module_0.Signer(var_0)
    var_10 = b'test-value'
    var_11 = var_9.sign(var_10)
    var_12 = -1
    var_13 = var_11[:var_12]
    var_14 = b'x'
    var_15 = var_13 + var_14
    var_16 = var_9.unsign(var_15)
    var_17 = module_0.Signer(var_16)
    var_18 = b'test-value'
    var_19 = var_17.unsign(var_18)
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = b'test-value'
    var_25 = var_23.sign(var_24)
    var_26 = var_23.unsign(var_25)
    var_27 = [var_20, var_21]
    var_28 = module_0.Signer(var_27)
    var_29 = b'test-value'
    var_30 = module_0.Signer(var_20)
    var_31 = var_30.sign(var_29)
    var_32 = var_28.unsign(var_31)
    var_33 = module_0.NoneAlgorithm()
    var_34 = module_0.Signer(var_18, algorithm=var_33)
    var_35 = b'test-value'
    var_36 = var_34.sign(var_35)
    var_37 = var_34.unsign(var_36)
    var_38 = 'hmac'
    var_39 = module_0.Signer(var_18, key_derivation=var_38)
    var_40 = b'test-value'
    var_41 = var_39.sign(var_40)
    var_42 = var_39.unsign(var_41)
    var_43 = b'test-value'
    var_44 = var_39.sign(var_43)
    var_45 = var_39.unsign(var_44)



# Parsed testcases at query #3
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
    var_15 = 'another-secret-key'
    var_16 = var_1.derive_key(var_15)
    var_17 = len(var_16)
    var_18 = 'django-concat'
    var_19 = module_0.Signer(var_0, key_derivation=var_18)
    var_20 = var_19.derive_key()
    var_21 = b'secret-key'
    var_22 = module_0.Signer(var_21)
    var_23 = var_22.derive_key()
    var_24 = len(var_23)
    var_25 = 'different-salt'
    var_26 = module_0.Signer(var_0, var_25)
    var_27 = var_26.derive_key()
    var_28 = 'invalid'
    var_29 = module_0.Signer(var_0, key_derivation=var_28)
    var_30 = var_29.derive_key()



# Parsed testcases at query #4
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
    var_7 = 'not-base64!'
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
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_21, algorithm=var_27)
    var_29 = b'test-value'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = var_28.verify_signature(var_29, var_25)
    assert var_32 is False



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    var_6 = 'test-value'
    var_7 = var_2.sign(var_6)
    var_8 = var_2.unsign(var_7)
    assert var_8 == b'test-value'
    var_9 = '|'
    var_10 = module_0.Signer(var_0, var_1, var_9)
    var_11 = var_10.sign(var_3)
    var_12 = var_10.unsign(var_11)
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15, var_1)
    var_17 = module_0.Signer(var_13, var_1)
    var_18 = var_17.sign(var_3)
    var_19 = module_0.Signer(var_14, var_1)
    var_20 = var_19.sign(var_3)
    var_21 = var_16.unsign(var_18)
    var_22 = var_16.unsign(var_20)
    var_23 = 'hmac'
    var_24 = module_0.Signer(var_0, var_1, key_derivation=var_23)
    var_25 = var_24.sign(var_3)
    var_26 = var_24.unsign(var_25)
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, var_1, algorithm=var_27)
    var_29 = var_28.sign(var_3)
    var_30 = var_28.unsign(var_29)
    var_31 = b'invalid-signature'
    var_32 = var_2.unsign(var_31)
    var_33 = -1
    var_34 = var_4[:var_33]
    var_35 = b'x'
    var_36 = var_34 + var_35
    var_37 = var_2.unsign(var_36)
    var_38 = var_2.unsign(var_3)
    var_39 = 'wrong-key'
    var_40 = module_0.Signer(var_39, var_32)
    var_41 = var_40.unsign(var_4)



# Parsed testcases at query #6
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
    var_29 = b'test-value'
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_23, algorithm=var_30)
    var_32 = b'test-value'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = var_31.verify_signature(var_32, var_27)
    assert var_35 is False



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = 'test'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test_value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'wrong_signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'wrong_sig'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_2.verify_signature(var_3, var_9)
    assert var_10 is False
    var_11 = 'old_key'
    var_12 = 'new_key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13, var_1)
    var_15 = b'rotated_value'
    var_16 = var_14.get_signature(var_15)
    var_17 = 'newer_key'
    var_18 = var_14.verify_signature(var_15, var_16)
    assert var_18 is False
    var_19 = 'secret'
    var_20 = 'test'
    var_21 = b'derivation_test'
    var_22 = var_2.get_signature(var_21)
    var_23 = var_2.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_19, var_20, algorithm=var_24)
    var_26 = b'custom_alg'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True



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
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    var_7 = b'some-signature'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    var_10 = b'invalid-base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'test-value'
    var_17 = module_0.Signer(var_12)
    var_18 = var_17.get_signature(var_16)
    var_19 = module_0.Signer(var_13)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_18)
    var_22 = var_15.verify_signature(var_16, var_20)
    var_23 = 'secret-key'
    var_24 = b'test-value'
    var_25 = var_1.get_signature(var_24)
    var_26 = var_1.verify_signature(var_24, var_25)
    var_27 = b'test-value'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_23, algorithm=var_28)
    var_30 = b'test-value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)



# Parsed testcases at query #9
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
    var_22 = 'other-old-key'
    var_23 = module_0.Signer(var_22)
    var_24 = var_23.get_signature(var_16)
    var_25 = var_15.verify_signature(var_16, var_24)
    assert var_25 is False
    var_26 = 'secret-key'
    var_27 = b'derivation-test'
    var_28 = var_1.get_signature(var_27)
    var_29 = var_1.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'sha256-test'
    var_31 = module_0.NoneAlgorithm()
    var_32 = module_0.Signer(var_26, algorithm=var_31)
    var_33 = b'none-algo-test'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True



# Parsed testcases at query #10
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
    var_7 = b'invalid'
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



# Parsed testcases at query #11
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
    var_26 = b'invalid-base64!'
    var_27 = var_1.verify_signature(var_23, var_26)
    assert var_27 is False



# Parsed testcases at query #12
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
    var_14 = b'test-value'
    var_15 = module_0.Signer(var_10)
    var_16 = var_15.get_signature(var_14)
    var_17 = module_0.Signer(var_11)
    var_18 = var_17.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_16)
    assert var_19 is True
    var_20 = var_13.verify_signature(var_14, var_18)
    assert var_20 is True
    var_21 = var_13.verify_signature(var_14, var_5)
    assert var_21 is False
    var_22 = 'secret-key'
    var_23 = b'test-value'
    var_24 = var_1.get_signature(var_23)
    var_25 = var_1.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'invalid-base64!'
    var_27 = var_1.verify_signature(var_23, var_26)
    assert var_27 is False



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'invalid_base64!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'old-secret-key'
    var_10 = b'new-secret-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'test_value'
    var_14 = module_0.Signer(var_9)
    var_15 = var_14.get_signature(var_13)
    var_16 = var_12.verify_signature(var_13, var_15)
    assert var_16 is True
    var_17 = module_0.Signer(var_10)
    var_18 = var_17.get_signature(var_13)
    var_19 = var_12.verify_signature(var_13, var_18)
    assert var_19 is True
    var_20 = b'invalid-secret-key'
    var_21 = module_0.Signer(var_20)
    var_22 = var_21.get_signature(var_13)
    var_23 = var_12.verify_signature(var_13, var_22)
    assert var_23 is False
    var_24 = 'secret-key'
    var_25 = b'test_value'
    var_26 = var_12.get_signature(var_25)
    var_27 = var_12.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_24, algorithm=var_28)
    var_30 = b'test_value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True



# Parsed testcases at query #14
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
    var_8 = b'not-base64!'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'old-secret-key'
    var_11 = b'new-secret-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-value'
    var_15 = module_0.Signer(var_10, var_1)
    var_16 = var_15.get_signature(var_14)
    var_17 = var_13.verify_signature(var_14, var_16)
    assert var_17 is True
    var_18 = module_0.Signer(var_11, var_1)
    var_19 = var_18.get_signature(var_14)
    var_20 = var_13.verify_signature(var_14, var_19)
    assert var_20 is True
    var_21 = b'invalid-key'
    var_22 = module_0.Signer(var_21, var_1)
    var_23 = var_22.get_signature(var_14)
    var_24 = var_13.verify_signature(var_14, var_23)
    assert var_24 is False
    var_25 = 'secret-key'
    var_26 = 'test-salt'
    var_27 = b'test-value'
    var_28 = var_2.get_signature(var_27)
    var_29 = var_2.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = 'secret-key'
    var_31 = 'test-salt'
    var_32 = b'test-value'
    var_33 = var_2.get_signature(var_32)
    var_34 = var_2.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = module_0.NoneAlgorithm()
    var_36 = module_0.Signer(var_30, var_31, algorithm=var_35)
    var_37 = b'test-value'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True



# Parsed testcases at query #15
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
    var_7 = b'invalid-base64!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-secret-key'
    var_10 = 'new-secret-key'
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
    var_23 = var_12.get_signature(var_22)
    var_24 = var_12.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'test-value'
    var_26 = var_12.get_signature(var_25)
    var_27 = var_12.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_21, algorithm=var_28)
    var_30 = b'test-value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True



# Parsed testcases at query #16
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
    var_8 = b'invalid-base64!'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'old-key'
    var_15 = var_13.get_signature(var_3)
    var_16 = b'new-key'
    var_17 = var_13.get_signature(var_3)
    var_18 = var_13.verify_signature(var_3, var_15)
    assert var_18 is True
    var_19 = var_13.verify_signature(var_3, var_17)
    assert var_19 is True
    var_20 = 'concat'
    var_21 = module_0.Signer(var_0, var_1, key_derivation=var_20)
    var_22 = var_21.get_signature(var_3)
    var_23 = var_21.verify_signature(var_3, var_22)
    assert var_23 is True
    var_24 = 'hmac'
    var_25 = module_0.Signer(var_0, var_1, key_derivation=var_24)
    var_26 = var_25.get_signature(var_3)
    var_27 = var_25.verify_signature(var_3, var_26)
    assert var_27 is True
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_0, var_1, algorithm=var_28)
    var_30 = var_29.get_signature(var_3)
    var_31 = var_29.verify_signature(var_3, var_30)
    assert var_31 is True



# Parsed testcases at query #17
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
    var_6 = b'invalid-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'different-value'
    var_9 = var_2.verify_signature(var_8, var_4)
    assert var_9 is False
    var_10 = b'invalid-base64!'
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = var_15.get_signature(var_3)
    var_17 = var_15.verify_signature(var_3, var_16)
    assert var_17 is False
    var_18 = var_15.verify_signature(var_3, var_16)
    assert var_18 is True
    var_19 = 'hmac'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.get_signature(var_3)
    var_22 = var_20.verify_signature(var_3, var_21)
    assert var_22 is True
    var_23 = var_2.verify_signature(var_3, var_21)
    assert var_23 is False
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_0, var_1, algorithm=var_24)
    var_26 = var_25.get_signature(var_3)
    var_27 = var_25.verify_signature(var_3, var_26)
    assert var_27 is True



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
    var_7 = b'test-sig'
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



# Parsed testcases at query #19
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
    var_7 = b'test_sig'
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
    var_16 = module_0.Signer(var_10)
    var_17 = var_16.get_signature(var_2)
    var_18 = var_13.verify_signature(var_2, var_17)
    assert var_18 is True
    var_19 = 'hmac'
    var_20 = module_0.Signer(var_0, key_derivation=var_19)
    var_21 = var_20.get_signature(var_2)
    var_22 = var_20.verify_signature(var_2, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, algorithm=var_23)
    var_25 = var_24.get_signature(var_2)
    var_26 = var_24.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = b'invalid_base64!'
    var_28 = var_1.verify_signature(var_2, var_27)
    assert var_28 is False



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
    var_15 = b'test-value-rotated'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = module_0.Signer(var_11, var_1)
    var_19 = var_18.get_signature(var_15)
    var_20 = var_14.verify_signature(var_15, var_19)
    assert var_20 is True
    var_21 = b'invalid-base64!'
    var_22 = var_2.verify_signature(var_3, var_21)
    assert var_22 is False
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, var_1, algorithm=var_23)
    var_25 = var_24.get_signature(var_3)
    var_26 = var_24.verify_signature(var_3, var_25)
    assert var_26 is True



# Parsed testcases at query #21
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
    assert var_15 is False
    var_16 = b'not-base64!'
    var_17 = var_1.verify_signature(var_13, var_16)
    assert var_17 is False
    var_18 = 'secret-key'
    var_19 = b'test-value'
    var_20 = var_1.get_signature(var_19)
    var_21 = var_1.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_18, algorithm=var_22)
    var_24 = b'test-value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True



# Parsed testcases at query #22
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
    var_7 = b'test'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'test-value'
    var_15 = var_13.get_signature(var_14)
    var_16 = 'newer-key'
    var_17 = var_13.verify_signature(var_14, var_15)
    assert var_17 is False
    var_18 = 'secret-key'
    var_19 = b'test-value'
    var_20 = var_1.get_signature(var_19)
    var_21 = var_1.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = b'test-value'
    var_23 = b'custom-sig'
    var_24 = var_1.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'wrong-sig'
    var_26 = var_1.verify_signature(var_22, var_25)
    assert var_26 is False



# Parsed testcases at query #24
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
    var_18 = 'hmac'
    var_19 = module_0.Signer(var_0, key_derivation=var_18)
    var_20 = b'test-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = var_1.verify_signature(var_20, var_21)
    assert var_23 is False
    var_24 = b'test-value'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_0, algorithm=var_25)
    var_27 = b'test-value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = var_1.verify_signature(var_27, var_28)
    assert var_30 is False



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
    var_7 = b'test-sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'test-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'secret-key'
    var_20 = b'test-value'
    var_21 = var_1.get_signature(var_20)
    var_22 = var_1.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'test-value'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_19, algorithm=var_24)
    var_26 = b'test-value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True



# Parsed testcases at query #26
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
    var_10 = 'not-base64'
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = b'test-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = module_0.Signer(var_12, var_1)
    var_19 = var_18.get_signature(var_16)
    var_20 = var_15.verify_signature(var_16, var_19)
    assert var_20 is True
    var_21 = var_15.verify_signature(var_16, var_17)
    assert var_21 is True
    var_22 = 'hmac'
    var_23 = module_0.Signer(var_0, var_1, key_derivation=var_22)
    var_24 = var_23.get_signature(var_16)
    var_25 = var_23.verify_signature(var_16, var_24)
    assert var_25 is True
    var_26 = var_2.verify_signature(var_16, var_24)
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
    assert var_15 is False
    var_16 = b'not-base64!'
    var_17 = var_1.verify_signature(var_13, var_16)
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



# Parsed testcases at query #28
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



# Parsed testcases at query #29
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
    var_7 = b'wrong_sig'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid_base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old_key'
    var_13 = 'new_key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotated_value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = var_15.salt
    var_20 = module_0.Signer(var_12, var_19)
    var_21 = var_20.get_signature(var_16)
    var_22 = var_15.verify_signature(var_16, var_21)
    assert var_22 is True
    var_23 = 'wrong_key'
    var_24 = module_0.Signer(var_23)
    var_25 = var_24.get_signature(var_16)
    var_26 = var_15.verify_signature(var_16, var_25)
    assert var_26 is False
    var_27 = 'hmac'
    var_28 = module_0.Signer(var_0, key_derivation=var_27)
    var_29 = var_28.get_signature(var_16)
    var_30 = var_28.verify_signature(var_16, var_29)
    assert var_30 is True
    var_31 = var_1.verify_signature(var_16, var_29)
    assert var_31 is False
    var_32 = module_0.NoneAlgorithm()
    var_33 = module_0.Signer(var_0, algorithm=var_32)
    var_34 = var_33.get_signature(var_16)
    var_35 = var_33.verify_signature(var_16, var_34)
    assert var_35 is True
    var_36 = var_1.verify_signature(var_16, var_34)
    assert var_36 is False



# Parsed testcases at query #30
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
    var_15 = module_0.Signer(var_10)
    var_16 = var_15.get_signature(var_13)
    var_17 = var_15.verify_signature(var_13, var_14)
    assert var_17 is False
    var_18 = var_12.verify_signature(var_13, var_16)
    assert var_18 is True
    var_19 = 'secret-key'
    var_20 = b'test-value'
    var_21 = var_1.get_signature(var_20)
    var_22 = var_1.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'test-value'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_19, algorithm=var_24)
    var_26 = b'test-value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'invalid-base64!'
    var_30 = var_1.verify_signature(var_26, var_29)
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
    var_24 = b'invalid-base64!'
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False



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
    var_7 = b'wrong'
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
    assert var_16 is True
    var_17 = 'secret-key'
    var_18 = b'test-value'
    var_19 = var_1.get_signature(var_18)
    var_20 = var_1.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = b'invalid-base64!'
    var_22 = var_1.verify_signature(var_18, var_21)
    assert var_22 is False



# Parsed testcases at query #33
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
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = module_0.NoneAlgorithm()
    var_11 = module_0.Signer(var_0, var_1, algorithm=var_10)
    var_12 = var_11.get_signature(var_3)
    var_13 = var_11.verify_signature(var_3, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16, var_1)
    var_18 = b'old-key'
    var_19 = var_17.get_signature(var_3)
    var_20 = b'new-key'
    var_21 = var_17.get_signature(var_3)
    var_22 = var_17.verify_signature(var_3, var_19)
    assert var_22 is True
    var_23 = var_17.verify_signature(var_3, var_21)
    assert var_23 is True
    var_24 = var_17.verify_signature(var_3, var_6)
    assert var_24 is False
    var_25 = 'secret-key'
    var_26 = 'test-salt'



# Parsed testcases at query #34
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
    var_6 = b'invalid-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'invalid-base64!'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'old-secret-key'
    var_11 = 'new-secret-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-value'
    var_15 = module_0.Signer(var_10, var_1)
    var_16 = var_15.get_signature(var_14)
    var_17 = var_13.verify_signature(var_14, var_16)
    assert var_17 is True
    var_18 = module_0.Signer(var_11, var_1)
    var_19 = var_18.get_signature(var_14)
    var_20 = var_13.verify_signature(var_14, var_19)
    assert var_20 is True
    var_21 = 'invalid-key'
    var_22 = module_0.Signer(var_21, var_1)
    var_23 = var_22.get_signature(var_14)
    var_24 = var_13.verify_signature(var_14, var_23)
    assert var_24 is False
    var_25 = 'secret-key'
    var_26 = 'test-salt'
    var_27 = b'test-value'
    var_28 = var_13.get_signature(var_27)
    var_29 = var_13.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_25, var_26, algorithm=var_30)
    var_32 = b'test-value'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True



# Parsed testcases at query #35
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
    var_29 = var_24.get_signature(var_28)
    var_30 = var_24.verify_signature(var_28, var_29)
    assert var_30 is True



# Parsed testcases at query #36
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
    var_7 = b'different-value'
    var_8 = var_1.verify_signature(var_7, var_3)
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
    var_21 = var_1.get_signature(var_20)
    var_22 = var_1.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'test-value'
    var_24 = var_1.get_signature(var_23)
    var_25 = var_1.verify_signature(var_23, var_24)
    assert var_25 is True



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
    var_15 = var_14.get_signature(var_2)
    var_16 = var_14.verify_signature(var_2, var_15)
    assert var_16 is True
    var_17 = 'secret-key'
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_17, algorithm=var_18)
    var_20 = var_19.get_signature(var_2)
    var_21 = var_19.verify_signature(var_2, var_20)
    assert var_21 is True



# Parsed testcases at query #39
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
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'invalid-base64!'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'old-key'
    var_15 = var_13.get_signature(var_3)
    var_16 = b'new-key'
    var_17 = var_13.get_signature(var_3)
    var_18 = var_13.verify_signature(var_3, var_15)
    assert var_18 is True
    var_19 = var_13.verify_signature(var_3, var_17)
    assert var_19 is True
    var_20 = 'secret-key'
    var_21 = 'test-salt'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_20, var_21, algorithm=var_22)
    var_24 = var_23.get_signature(var_3)
    var_25 = var_23.verify_signature(var_3, var_24)
    assert var_25 is True



# Parsed testcases at query #40
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
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_0, algorithm=var_17)
    var_19 = b''
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = b'anything'
    var_22 = var_18.verify_signature(var_2, var_21)
    assert var_22 is False



# Parsed testcases at query #41
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
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-value'
    var_15 = var_13.get_signature(var_14)
    var_16 = 'newer-key'
    var_17 = var_13.verify_signature(var_14, var_15)
    assert var_17 is False
    var_18 = 'invalid-base64'
    var_19 = var_2.verify_signature(var_14, var_18)
    assert var_19 is False
    var_20 = 'hmac'
    var_21 = module_0.Signer(var_0, var_1, key_derivation=var_20)
    var_22 = var_21.get_signature(var_14)
    var_23 = var_21.verify_signature(var_14, var_22)
    assert var_23 is True
    var_24 = var_2.verify_signature(var_14, var_22)
    assert var_24 is False
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_0, var_1, algorithm=var_25)
    var_27 = var_26.get_signature(var_14)
    var_28 = var_26.verify_signature(var_14, var_27)
    assert var_28 is True



# Parsed testcases at query #42
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
    var_15 = var_14.get_signature(var_2)
    var_16 = var_14.verify_signature(var_2, var_15)
    assert var_16 is True
    var_17 = 'secret-key'
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_17, algorithm=var_18)
    var_20 = var_19.get_signature(var_2)
    var_21 = var_19.verify_signature(var_2, var_20)
    assert var_21 is True



# Parsed testcases at query #43
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
    var_7 = b'different-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_12.get_signature(var_2)
    var_15 = var_12.verify_signature(var_2, var_13)
    assert var_15 is True
    var_16 = var_12.verify_signature(var_2, var_14)
    assert var_16 is True
    var_17 = var_12.verify_signature(var_2, var_5)
    assert var_17 is False
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_0, algorithm=var_18)
    var_20 = var_19.get_signature(var_2)
    var_21 = var_19.verify_signature(var_2, var_20)
    assert var_21 is True
    var_22 = var_19.verify_signature(var_2, var_5)
    assert var_22 is False
    var_23 = b'invalid-base64!'
    var_24 = var_1.verify_signature(var_2, var_23)
    assert var_24 is False



# Parsed testcases at query #44
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
    var_7 = b'not-base64!'
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



# Parsed testcases at query #45
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
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = module_0.NoneAlgorithm()
    var_11 = module_0.Signer(var_0, var_1, algorithm=var_10)
    var_12 = var_11.get_signature(var_3)
    var_13 = var_11.verify_signature(var_3, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16, var_1)
    var_18 = b'rotated-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.Signer(var_14, var_1)
    var_22 = var_21.get_signature(var_18)
    var_23 = var_17.verify_signature(var_18, var_22)
    assert var_23 is True
    var_24 = b'invalid-base64!'
    var_25 = var_2.verify_signature(var_3, var_24)
    assert var_25 is False



# Parsed testcases at query #46
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
    var_6 = b'wrong-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'invalid-base64'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'old-secret-key'
    var_11 = b'new-secret-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-value'
    var_15 = module_0.Signer(var_10, var_1)
    var_16 = var_15.get_signature(var_14)
    var_17 = module_0.Signer(var_11, var_1)
    var_18 = var_17.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_16)
    assert var_19 is True
    var_20 = var_13.verify_signature(var_14, var_18)
    assert var_20 is True
    var_21 = 'wrong-secret-key'
    var_22 = module_0.Signer(var_21, var_1)
    var_23 = var_22.get_signature(var_14)
    var_24 = var_13.verify_signature(var_14, var_23)
    assert var_24 is False
    var_25 = 'hmac'
    var_26 = module_0.Signer(var_0, var_1, key_derivation=var_25)
    var_27 = var_26.get_signature(var_14)
    var_28 = var_26.verify_signature(var_14, var_27)
    assert var_28 is True
    var_29 = var_13.verify_signature(var_14, var_27)
    assert var_29 is False
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_0, var_1, algorithm=var_30)
    var_32 = var_31.get_signature(var_14)
    var_33 = var_31.verify_signature(var_14, var_32)
    assert var_33 is True
    var_34 = var_13.verify_signature(var_14, var_32)
    assert var_34 is False



# Parsed testcases at query #47
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
    var_11 = b'invalid-base64!'
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15, var_1)
    var_17 = b'test-value'
    var_18 = module_0.Signer(var_13, var_1)
    var_19 = var_18.get_signature(var_17)
    var_20 = module_0.Signer(var_14, var_1)
    var_21 = var_20.get_signature(var_17)
    var_22 = var_16.verify_signature(var_17, var_19)
    assert var_22 is True
    var_23 = var_16.verify_signature(var_17, var_21)
    assert var_23 is True
    var_24 = var_16.verify_signature(var_17, var_8)
    assert var_24 is False
    var_25 = 'secret-key'
    var_26 = 'test-salt'
    var_27 = b'test-value'
    var_28 = var_2.get_signature(var_27)
    var_29 = var_2.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'wrong-sig'
    var_31 = var_2.verify_signature(var_27, var_30)
    assert var_31 is False
    var_32 = module_0.NoneAlgorithm()
    var_33 = module_0.Signer(var_25, var_26, algorithm=var_32)
    var_34 = b'test-value'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = var_33.verify_signature(var_34, var_8)
    assert var_37 is False



# Parsed testcases at query #48
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
    var_7 = b'invalid-base64!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'old-secret-key'
    var_10 = b'new-secret-key'
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
    var_20 = b'invalid-secret-key'
    var_21 = module_0.Signer(var_20)
    var_22 = var_21.get_signature(var_13)
    var_23 = var_12.verify_signature(var_13, var_22)
    assert var_23 is False
    var_24 = 'secret-key'
    var_25 = b'test-value'
    var_26 = var_12.get_signature(var_25)
    var_27 = var_12.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_24, algorithm=var_28)
    var_30 = b'test-value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True



