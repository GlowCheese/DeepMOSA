####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = len(var_3)
    var_5 = 'custom-secret'
    var_6 = var_2.derive_key(var_5)
    var_7 = 'concat'
    var_8 = module_0.Signer(var_0, var_1, key_derivation=var_7)
    var_9 = var_8.derive_key()
    var_10 = len(var_9)
    var_11 = 'django-concat'
    var_12 = module_0.Signer(var_0, var_1, key_derivation=var_11)
    var_13 = var_12.derive_key()
    var_14 = len(var_13)
    var_15 = 'hmac'
    var_16 = module_0.Signer(var_0, var_1, key_derivation=var_15)
    var_17 = var_16.derive_key()
    var_18 = len(var_17)
    var_19 = 'none'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = var_20.derive_key()
    assert var_21 == b'secret-key'
    var_22 = 'salt1'
    var_23 = module_0.Signer(var_0, var_22)
    var_24 = 'salt2'
    var_25 = module_0.Signer(var_0, var_24)
    var_26 = var_23.derive_key()
    var_27 = var_25.derive_key()
    var_28 = 'secret1'
    var_29 = module_0.Signer(var_28)
    var_30 = 'secret2'
    var_31 = module_0.Signer(var_30)
    var_32 = var_29.derive_key()
    var_33 = var_31.derive_key()
    var_34 = 'old-key'
    var_35 = 'new-key'
    var_36 = [var_34, var_35]
    var_37 = module_0.Signer(var_36)
    var_38 = var_37.derive_key()
    var_39 = var_37.derive_key(var_34)
    var_40 = var_37.derive_key(var_35)
    var_41 = module_0.Signer(var_0)
    var_42 = 'another-secret'
    var_43 = var_41.derive_key(var_42)
    var_44 = b'byte-secret'
    var_45 = var_41.derive_key(var_44)
    var_46 = 'invalid'
    var_47 = module_0.Signer(var_0, key_derivation=var_46)
    var_48 = var_47.derive_key()



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test-value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test-value'
    var_5 = b'test-bytes'
    var_6 = var_1.sign(var_5)
    var_7 = var_1.unsign(var_6)
    assert var_7 == b'test-bytes'
    var_8 = 'old-key'
    var_9 = 'new-key'
    var_10 = [var_8, var_9]
    var_11 = module_0.Signer(var_10)
    var_12 = 'value'
    var_13 = var_11.sign(var_12)
    var_14 = var_11.unsign(var_13)
    assert var_14 == b'value'
    var_15 = 'no-separator-here'
    var_16 = var_1.unsign(var_15)
    var_17 = b'value.invalid-signature'
    var_18 = var_1.unsign(var_17)
    var_19 = 'key'
    var_20 = b'-'
    var_21 = module_0.Signer(var_19, sep=var_20)
    var_22 = 'test'
    var_23 = var_21.sign(var_22)
    var_24 = var_21.unsign(var_23)
    assert var_24 == b'test'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_19, algorithm=var_25)
    var_27 = var_26.sign(var_12)
    var_28 = var_26.unsign(var_27)
    assert var_28 == b'value'



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
    var_22 = module_0.Signer(var_0, var_1)
    var_23 = 'other-key'
    var_24 = var_22.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'other-key'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = 'old-key'
    var_30 = 'new-key'
    var_31 = [var_29, var_30]
    var_32 = module_0.Signer(var_31, var_1)
    var_33 = var_32.derive_key()
    var_34 = var_4 + var_5
    var_35 = b'new-key'
    var_36 = var_34 + var_35
    var_37 = module_1.digest()
    var_38 = b'bytes-key'
    var_39 = module_0.Signer(var_38, var_4)
    var_40 = var_39.derive_key()
    var_41 = var_4 + var_5
    var_42 = var_41 + var_38
    var_43 = module_1.digest()
    var_44 = 'invalid'
    var_45 = module_0.Signer(var_0, var_1, key_derivation=var_44)
    var_46 = var_45.derive_key()



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
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = 'not-base64!@#'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = b'other-value'
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = var_1.verify_signature(var_11, var_3)
    assert var_14 is False
    var_15 = 'test-value'
    var_16 = 'invalid'
    var_17 = var_1.verify_signature(var_15, var_16)
    assert var_17 is False
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = b'test-rotated'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = module_0.Signer(var_18)
    var_26 = var_25.get_signature(var_22)
    var_27 = var_21.verify_signature(var_22, var_26)
    assert var_27 is True
    var_28 = 'wrong-key'
    var_29 = module_0.Signer(var_28)
    var_30 = var_29.get_signature(var_22)
    var_31 = var_21.verify_signature(var_22, var_30)
    assert var_31 is False
    var_32 = 'secret'
    var_33 = module_0.NoneAlgorithm()
    var_34 = module_0.Signer(var_32, algorithm=var_33)
    var_35 = b'test-none'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = module_1.base64_decode(var_3)
    var_39 = var_1.verify_signature(var_2, var_38)
    assert var_39 is True



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = 'test-value'
    var_7 = var_2.get_signature(var_6)
    var_8 = var_2.verify_signature(var_6, var_7)
    assert var_8 is True
    var_9 = 'utf-8'
    var_10 = b'wrong-signature'
    var_11 = module_1.base64_encode(var_10)
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = b'!!!invalid-base64!!!'
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = b''
    var_16 = var_2.get_signature(var_15)
    var_17 = var_2.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = b'rotation-test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_1, algorithm=var_25)
    var_27 = b'none-alg-test'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'custom-salt'
    var_31 = module_0.Signer(var_1, var_30)
    var_32 = b'custom-salt-test'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = b'_'
    var_36 = module_0.Signer(var_1, sep=var_35)
    var_37 = b'separator-test'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = 'key1'
    var_41 = module_0.Signer(var_40)
    var_42 = 'key2'
    var_43 = module_0.Signer(var_42)
    var_44 = b'different-keys'
    var_45 = var_41.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is False
    var_47 = 256
    var_48 = range(var_47)
    var_49 = bytes(var_48)
    var_50 = var_2.get_signature(var_49)
    var_51 = var_2.verify_signature(var_49, var_50)
    assert var_51 is True



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
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = 'test-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is True
    var_9 = 'utf-8'
    var_10 = b'-'
    var_11 = module_0.Signer(var_0, sep=var_10)
    var_12 = b'another-value'
    var_13 = var_11.get_signature(var_12)
    var_14 = var_11.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'test-rotation'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'custom-salt'
    var_23 = module_0.Signer(var_0, var_22)
    var_24 = b'test-salt'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, algorithm=var_27)
    var_29 = b'test-none-algo'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b''
    var_33 = var_28.verify_signature(var_29, var_32)
    assert var_33 is True
    var_34 = b'\x00'
    var_35 = 1
    var_36 = var_3[var_35:]
    var_37 = var_34 + var_36
    var_38 = var_1.verify_signature(var_2, var_37)
    assert var_38 is False
    var_39 = b''
    var_40 = var_1.get_signature(var_39)
    var_41 = var_1.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = b'test@#$%^&*()'
    var_43 = var_1.get_signature(var_42)
    var_44 = var_1.verify_signature(var_42, var_43)
    assert var_44 is True
    var_45 = b'!!!invalid-base64!!!'
    var_46 = var_1.verify_signature(var_2, var_45)
    assert var_46 is False
    var_47 = var_1.verify_signature(var_2, var_32)
    assert var_47 is False
    var_48 = 'key1'
    var_49 = 'key2'
    var_50 = 'key3'
    var_51 = [var_48, var_49, var_50]
    var_52 = module_0.Signer(var_51)
    var_53 = b'test-rotation-all'
    var_54 = var_52.get_signature(var_53)
    var_55 = var_52.verify_signature(var_53, var_54)
    assert var_55 is True



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'!!!invalid!!!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'test value'
    var_12 = var_1.verify_signature(var_11, var_3)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = var_16.get_signature(var_2)
    var_18 = var_16.verify_signature(var_2, var_17)
    assert var_18 is True
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_0, algorithm=var_19)
    var_21 = var_20.get_signature(var_2)
    var_22 = var_20.verify_signature(var_2, var_21)
    assert var_22 is True
    var_23 = 'different-salt'
    var_24 = module_0.Signer(var_0, var_23)
    var_25 = var_24.verify_signature(var_2, var_3)
    assert var_25 is False
    var_26 = b'|'
    var_27 = module_0.Signer(var_0, sep=var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_27.verify_signature(var_2, var_28)
    assert var_29 is True
    var_30 = var_1.verify_signature(var_2, var_28)
    assert var_30 is False



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test verify_signature method for various scenarios.'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'original value'
    var_8 = var_3.get_signature(var_7)
    var_9 = b'different value'
    var_10 = var_3.verify_signature(var_9, var_8)
    assert var_10 is False
    var_11 = b'test'
    var_12 = b'invalid-sig'
    var_13 = var_3.verify_signature(var_11, var_12)
    assert var_13 is False
    var_14 = b''
    var_15 = var_3.get_signature(var_14)
    var_16 = var_3.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'test string'
    var_18 = var_3.get_signature(var_17)
    var_19 = var_3.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22, var_2)
    var_24 = var_23.get_signature(var_11)
    var_25 = var_23.verify_signature(var_11, var_24)
    assert var_25 is True
    var_26 = 'secret'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = var_28.get_signature(var_11)
    var_30 = var_28.verify_signature(var_11, var_29)
    assert var_30 is True
    var_31 = b'important data'
    var_32 = var_3.get_signature(var_31)
    var_33 = b'a'
    var_34 = 1
    var_35 = var_32[var_34:]
    var_36 = var_33 + var_35
    var_37 = var_3.verify_signature(var_31, var_36)
    assert var_37 is False
    var_38 = b'bytes test'
    var_39 = var_3.get_signature(var_38)
    var_40 = var_3.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = 'different-salt'
    var_42 = module_0.Signer(var_1, var_41)
    var_43 = var_3.get_signature(var_11)
    var_44 = var_42.verify_signature(var_11, var_43)
    assert var_44 is False



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
    var_6 = b'invalid-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'test-value'
    var_11 = var_2.get_signature(var_10)
    var_12 = var_2.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15, var_1)
    var_17 = b'test-rotated'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = module_0.Signer(var_13, var_1)
    var_21 = var_20.get_signature(var_17)
    var_22 = var_16.verify_signature(var_17, var_21)
    assert var_22 is True
    var_23 = b'|'
    var_24 = module_0.Signer(var_0, var_1, var_23)
    var_25 = b'test-custom'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_0, var_1, algorithm=var_28)
    var_30 = b'test-none'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = b'test-hmac'
    var_34 = 'concat'
    var_35 = module_0.Signer(var_0, var_1, key_derivation=var_34)
    var_36 = b'test-concat'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = 'hmac'
    var_40 = module_0.Signer(var_0, var_1, key_derivation=var_39)
    var_41 = b'test-hmac-der'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = 'none'
    var_45 = module_0.Signer(var_0, var_1, key_derivation=var_44)
    var_46 = b'test-none-der'
    var_47 = var_45.get_signature(var_46)
    var_48 = var_45.verify_signature(var_46, var_47)
    assert var_48 is True
    var_49 = b'!!!invalid-base64!!!'
    var_50 = var_2.verify_signature(var_3, var_49)
    assert var_50 is False
    var_51 = 'different-salt'
    var_52 = module_0.Signer(var_0, var_51)
    var_53 = var_52.verify_signature(var_3, var_4)
    assert var_53 is False



# Parsed testcases at query #10
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
    var_8 = b''
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'other-value'
    var_11 = var_2.get_signature(var_10)
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = 'ascii'
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16, var_1)
    var_18 = b'test-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'secret'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = b'test-value'
    var_25 = var_23.get_signature(var_24)
    assert var_25 == b''
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'!!!invalid-base64!!!'
    var_28 = var_2.verify_signature(var_24, var_27)
    assert var_28 is False
    var_29 = b''
    var_30 = var_2.get_signature(var_29)
    var_31 = var_2.verify_signature(var_29, var_30)
    assert var_31 is True



# Parsed testcases at query #11
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = 'different-secret'
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'secret'
    var_16 = module_0.NoneAlgorithm()
    var_17 = module_0.Signer(var_15, algorithm=var_16)
    var_18 = var_17.get_signature(var_2)
    var_19 = var_17.verify_signature(var_2, var_18)
    assert var_19 is True
    var_20 = b'!!!invalid-base64!!!'
    var_21 = var_1.verify_signature(var_2, var_20)
    assert var_21 is False
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24)
    var_26 = b'rotation-test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = module_0.Signer(var_22)
    var_30 = var_29.get_signature(var_26)
    var_31 = var_25.verify_signature(var_26, var_30)
    assert var_31 is True
    var_32 = var_1.get_signature(var_2)
    var_33 = var_1.verify_signature(var_2, var_32)
    assert var_33 is True
    var_34 = 'ascii'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret'
    assert var_1 is True
    var_2 = module_0.Signer(var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'original'
    var_13 = var_2.get_signature(var_12)
    var_14 = b'tampered'
    var_15 = var_2.verify_signature(var_14, var_13)
    assert var_15 is False
    var_16 = b'!!!invalid!!!'
    var_17 = var_2.verify_signature(var_3, var_16)
    assert var_17 is False
    var_18 = 'old_key'
    var_19 = 'new_key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = b'rotation test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = module_0.Signer(var_18)
    var_26 = var_25.get_signature(var_22)
    var_27 = var_21.verify_signature(var_22, var_26)
    assert var_27 is True
    var_28 = 'test value'
    var_29 = var_2.verify_signature(var_28, var_23)
    assert var_29 is True
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_1, algorithm=var_30)
    var_32 = b'test'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = b'custom_salt'
    var_36 = module_0.Signer(var_1, var_35)
    var_37 = b'test'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = b'different_salt'
    var_41 = module_0.Signer(var_1, var_40)
    var_42 = var_41.verify_signature(var_37, var_38)
    assert var_42 is False
    var_43 = b'|'
    var_44 = module_0.Signer(var_1, sep=var_43)
    var_45 = b'test'
    var_46 = var_44.get_signature(var_45)
    var_47 = var_44.verify_signature(var_45, var_46)
    assert var_47 is True
    var_48 = 'secret'
    var_49 = b'test'
    var_50 = b'x'
    var_51 = 10000
    var_52 = var_50 * var_51
    var_53 = var_2.get_signature(var_52)
    var_54 = var_2.verify_signature(var_52, var_53)
    assert var_54 is True
    var_55 = 'héllo wörld'
    var_56 = var_2.get_signature(var_55)
    var_57 = var_2.verify_signature(var_55, var_56)
    assert var_57 is True
    var_58 = b'test.value'
    var_59 = var_2.get_signature(var_58)
    var_60 = var_2.verify_signature(var_58, var_59)
    assert var_60 is True
    var_61 = b''
    var_62 = var_2.verify_signature(var_49, var_61)
    assert var_62 is False
    var_63 = None
    var_64 = var_2.verify_signature(var_63, var_23)



# Parsed testcases at query #13
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
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.verify_signature(var_3, var_9)
    assert var_10 is False
    var_11 = b'!!!invalid-base64!!!'
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = 'test-value'
    var_14 = var_2.verify_signature(var_13, var_4)
    assert var_14 is True
    var_15 = 'ascii'
    var_16 = 'different-secret'
    var_17 = module_0.Signer(var_16, var_1)
    var_18 = var_17.get_signature(var_3)
    var_19 = var_2.verify_signature(var_3, var_18)
    assert var_19 is False
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22, var_1)
    var_24 = b'test-value-3'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = module_0.Signer(var_20, var_1)
    var_28 = var_27.get_signature(var_24)
    var_29 = var_23.verify_signature(var_24, var_28)
    assert var_29 is True
    var_30 = 'different-salt'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = var_31.get_signature(var_3)
    var_33 = var_2.verify_signature(var_3, var_32)
    assert var_33 is False



# Parsed testcases at query #14
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_7)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-rotation'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = b'corrupted'
    var_19 = 3
    var_20 = var_16[var_19:]
    var_21 = var_18 + var_20
    var_22 = var_1.verify_signature(var_15, var_21)
    assert var_22 is False
    var_23 = 'secret'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_23, algorithm=var_24)
    var_26 = b'test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'!!!invalid-base64!!!'
    var_30 = var_1.verify_signature(var_26, var_29)
    assert var_30 is False



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid_sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'test value'
    var_12 = var_2.get_signature(var_11)
    var_13 = b'!!!invalid_base64!!!'
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17, var_1)
    var_19 = b'rotate test'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_0, algorithm=var_22)
    var_24 = b'none algo test'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'|'
    var_28 = module_0.Signer(var_0, sep=var_27)
    var_29 = b'separator test'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'different value'
    var_33 = var_2.verify_signature(var_32, var_4)
    assert var_33 is False



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test Signer.verify_signature method with various scenarios.'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid-signature'
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_3.get_signature(var_9)
    var_11 = var_3.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b''
    var_13 = var_3.verify_signature(var_4, var_12)
    assert var_13 is False
    var_14 = b'corrupted'
    var_15 = var_3.verify_signature(var_4, var_14)
    assert var_15 is False
    var_16 = 'test-value'
    var_17 = var_3.verify_signature(var_16, var_5)
    assert var_17 is True
    var_18 = 'ascii'
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21, var_2)
    var_23 = b'test-value-rotated'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'wrong-signature'
    var_27 = var_22.verify_signature(var_23, var_26)
    assert var_27 is False
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_1, var_2, algorithm=var_28)
    var_30 = b'test-value-none'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = b'|'
    var_34 = module_0.Signer(var_1, var_2, var_33)
    var_35 = b'test-value-sep'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = b'test-value-digest'
    var_39 = 'concat'
    var_40 = module_0.Signer(var_1, var_2, key_derivation=var_39)
    var_41 = b'test-value-concat'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = 'hmac'
    var_45 = module_0.Signer(var_1, var_2, key_derivation=var_44)
    var_46 = b'test-value-hmac'
    var_47 = var_45.get_signature(var_46)
    var_48 = var_45.verify_signature(var_46, var_47)
    assert var_48 is True
    var_49 = 'none'
    var_50 = module_0.Signer(var_1, var_2, key_derivation=var_49)
    var_51 = b'test-value-none-der'
    var_52 = var_50.get_signature(var_51)
    var_53 = var_50.verify_signature(var_51, var_52)
    assert var_53 is True
    var_54 = 'secret1'
    var_55 = module_0.Signer(var_54, var_2)
    var_56 = 'secret2'
    var_57 = module_0.Signer(var_56, var_2)
    var_58 = b'test-value-diff'
    var_59 = var_55.get_signature(var_58)
    var_60 = var_57.verify_signature(var_58, var_59)
    assert var_60 is False



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
    var_6 = 'test-value'
    var_7 = var_2.verify_signature(var_6, var_4)
    assert var_7 is True
    var_8 = b'invalid-sig'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!!!'
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = b''
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17, var_1)
    var_19 = b'test-value-2'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'secret'
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_22, algorithm=var_23)
    var_25 = b'unsigned-value'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = var_24.verify_signature(var_25, var_12)
    assert var_28 is True
    var_29 = b'|'
    var_30 = module_0.Signer(var_22, sep=var_29)
    var_31 = b'another-value'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b'bytes-key'
    var_35 = b'bytes-salt'
    var_36 = module_0.Signer(var_34, var_35)
    var_37 = b'bytes-value'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = 'correct-key'
    var_41 = module_0.Signer(var_40)
    var_42 = 'wrong-key'
    var_43 = module_0.Signer(var_42)
    var_44 = b'sensitive-data'
    var_45 = var_41.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is False



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = b'!!!invalid_base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'test string'
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'different-salt'
    var_16 = module_0.Signer(var_0, var_15)
    var_17 = var_16.get_signature(var_2)
    var_18 = var_16.verify_signature(var_2, var_17)
    assert var_18 is True
    var_19 = var_1.verify_signature(var_2, var_17)
    assert var_19 is False
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = b'test value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = [var_20, var_21]
    var_28 = module_0.Signer(var_27)
    var_29 = var_28.get_signature(var_24)
    var_30 = var_23.verify_signature(var_24, var_29)
    assert var_30 is True
    var_31 = 'secret'
    var_32 = 'concat'
    var_33 = module_0.Signer(var_31, key_derivation=var_32)
    var_34 = var_33.get_signature(var_24)
    var_35 = var_33.verify_signature(var_24, var_34)
    assert var_35 is True
    var_36 = 'hmac'
    var_37 = module_0.Signer(var_31, key_derivation=var_36)
    var_38 = var_37.get_signature(var_24)
    var_39 = var_37.verify_signature(var_24, var_38)
    assert var_39 is True
    var_40 = 'none'
    var_41 = module_0.Signer(var_31, key_derivation=var_40)
    var_42 = var_41.get_signature(var_24)
    var_43 = var_41.verify_signature(var_24, var_42)
    assert var_43 is True
    var_44 = module_0.NoneAlgorithm()
    var_45 = module_0.Signer(var_31, algorithm=var_44)
    var_46 = var_45.get_signature(var_24)
    var_47 = var_45.verify_signature(var_24, var_46)
    assert var_47 is True



# Parsed testcases at query #19
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
    var_8 = b'wrong-value'
    var_9 = var_2.verify_signature(var_8, var_4)
    assert var_9 is False
    var_10 = 'test-value'
    var_11 = var_2.get_signature(var_10)
    var_12 = var_2.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = b'not-base64!!'
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = b''
    var_16 = var_2.verify_signature(var_3, var_15)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'test'
    var_22 = module_0.Signer(var_17)
    var_23 = var_22.get_signature(var_21)
    var_24 = var_20.verify_signature(var_21, var_23)
    assert var_24 is True
    var_25 = 'unknown-key'
    var_26 = module_0.Signer(var_25)
    var_27 = var_26.get_signature(var_21)
    var_28 = var_20.verify_signature(var_21, var_27)
    assert var_28 is False
    var_29 = 'secret'
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_29, algorithm=var_30)
    var_32 = b'test'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = b'test'
    var_36 = b'wrong'



# Parsed testcases at query #20
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'test-string'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = module_0.NoneAlgorithm()
    var_14 = module_0.Signer(var_0, algorithm=var_13)
    var_15 = var_14.get_signature(var_2)
    var_16 = var_14.verify_signature(var_2, var_15)
    assert var_16 is True
    var_17 = b'!!!invalid-base64!!!'
    var_18 = var_1.verify_signature(var_2, var_17)
    assert var_18 is False
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21)
    var_23 = b'rotation-test'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = module_0.Signer(var_19)
    var_27 = var_26.get_signature(var_23)
    var_28 = var_22.verify_signature(var_23, var_27)
    assert var_28 is True
    var_29 = 'salt1'
    var_30 = module_0.Signer(var_0, var_29)
    var_31 = 'salt2'
    var_32 = module_0.Signer(var_0, var_31)
    var_33 = var_30.get_signature(var_2)
    var_34 = var_30.verify_signature(var_2, var_33)
    assert var_34 is True
    var_35 = var_32.verify_signature(var_2, var_33)
    assert var_35 is False



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = var_1.verify_signature(var_2, var_7)
    assert var_10 is False
    var_11 = b'original'
    var_12 = var_1.get_signature(var_11)
    var_13 = b'modified'
    var_14 = var_1.verify_signature(var_13, var_12)
    assert var_14 is False
    var_15 = 'different-key'
    var_16 = module_0.Signer(var_15)
    var_17 = var_16.get_signature(var_2)
    var_18 = var_1.verify_signature(var_2, var_17)
    assert var_18 is False
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21)
    var_23 = module_0.Signer(var_19)
    var_24 = var_23.get_signature(var_2)
    var_25 = var_22.verify_signature(var_2, var_24)
    assert var_25 is True
    var_26 = var_22.get_signature(var_2)
    var_27 = var_22.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = 'secret'
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_28, algorithm=var_29)
    var_31 = var_30.get_signature(var_2)
    var_32 = var_30.verify_signature(var_2, var_31)
    assert var_32 is True
    var_33 = b'any'
    var_34 = var_30.verify_signature(var_2, var_33)
    assert var_34 is True
    var_35 = 'test value'
    var_36 = var_1.verify_signature(var_35, var_3)
    assert var_36 is True
    var_37 = b'!!!invalid base64!!!'
    var_38 = var_1.verify_signature(var_2, var_37)
    assert var_38 is False
    var_39 = 'different'
    var_40 = module_0.Signer(var_28, var_39)
    var_41 = var_40.get_signature(var_2)
    var_42 = var_1.verify_signature(var_2, var_41)
    assert var_42 is False
    var_43 = b'-'
    var_44 = module_0.Signer(var_28, sep=var_43)
    var_45 = var_44.get_signature(var_2)
    var_46 = module_0.Signer(var_28)
    var_47 = var_46.verify_signature(var_2, var_45)
    assert var_47 is False



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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'not-base64!!!'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'rotate-test'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = module_0.Signer(var_13)
    var_21 = var_20.get_signature(var_17)
    var_22 = var_16.verify_signature(var_17, var_21)
    assert var_22 is True
    var_23 = 'key'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_23, algorithm=var_24)
    var_26 = b'none-algo'
    var_27 = var_25.get_signature(var_26)
    var_28 = b''
    var_29 = module_1.base64_encode(var_28)
    var_30 = var_25.verify_signature(var_26, var_27)
    assert var_30 is True
    var_31 = b'custom-salt'
    var_32 = module_0.Signer(var_23, var_31)
    var_33 = b'salt-test'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = 'test-value'
    var_37 = var_1.verify_signature(var_36, var_3)
    assert var_37 is True
    var_38 = var_1.verify_signature(var_2, var_28)
    assert var_38 is False



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'corrupted-signature'
    var_10 = var_2.verify_signature(var_3, var_9)
    assert var_10 is False
    var_11 = b''
    var_12 = var_2.get_signature(var_11)
    var_13 = var_2.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'test-string'
    var_15 = var_2.get_signature(var_14)
    var_16 = var_2.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'rotation-test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.Signer(var_17)
    var_25 = var_24.get_signature(var_21)
    var_26 = var_20.verify_signature(var_21, var_25)
    assert var_26 is True
    var_27 = 'wrong-key'
    var_28 = module_0.Signer(var_27)
    var_29 = var_28.get_signature(var_21)
    var_30 = var_20.verify_signature(var_21, var_29)
    assert var_30 is False
    var_31 = module_0.NoneAlgorithm()
    var_32 = 'test'
    var_33 = module_0.Signer(var_32, algorithm=var_31)
    var_34 = b'test-value'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = b'!!!not-base64!!!'
    var_38 = var_2.verify_signature(var_3, var_37)
    assert var_38 is False
    var_39 = 'salt1'
    var_40 = module_0.Signer(var_32, var_39)
    var_41 = 'salt2'
    var_42 = module_0.Signer(var_32, var_41)
    var_43 = b'test-with-salt'
    var_44 = var_40.get_signature(var_43)
    var_45 = var_40.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = var_42.verify_signature(var_43, var_44)
    assert var_46 is False
    var_47 = 'concat'
    var_48 = module_0.Signer(var_32, key_derivation=var_47)
    var_49 = 'django-concat'
    var_50 = module_0.Signer(var_32, key_derivation=var_49)
    var_51 = 'hmac'
    var_52 = module_0.Signer(var_32, key_derivation=var_51)
    var_53 = 'none'
    var_54 = module_0.Signer(var_32, key_derivation=var_53)
    var_55 = b'derivation-test'
    var_56 = var_48.get_signature(var_55)
    var_57 = var_50.get_signature(var_55)
    var_58 = var_52.get_signature(var_55)
    var_59 = var_54.get_signature(var_55)
    var_60 = var_48.verify_signature(var_55, var_56)
    assert var_60 is True
    var_61 = var_50.verify_signature(var_55, var_57)
    assert var_61 is True
    var_62 = var_52.verify_signature(var_55, var_58)
    assert var_62 is True
    var_63 = var_54.verify_signature(var_55, var_59)
    assert var_63 is True
    var_64 = var_48.verify_signature(var_55, var_57)
    assert var_64 is False
    var_65 = var_50.verify_signature(var_55, var_56)
    assert var_65 is False



# Parsed testcases at query #24
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
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'test-value'
    var_12 = var_2.verify_signature(var_11, var_4)
    assert var_12 is True
    var_13 = b'!!!invalid-base64!!!'
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17, var_1)
    var_19 = b'test-value'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = module_0.Signer(var_15, var_1)
    var_23 = var_22.get_signature(var_19)
    var_24 = var_18.verify_signature(var_19, var_23)
    assert var_24 is True
    var_25 = b'.'
    var_26 = module_0.Signer(var_0, var_1, var_25)
    var_27 = b'-'
    var_28 = module_0.Signer(var_0, var_1, var_27)
    var_29 = b'test-value'
    var_30 = var_26.get_signature(var_29)
    var_31 = var_28.get_signature(var_29)
    var_32 = var_26.verify_signature(var_29, var_30)
    assert var_32 is True
    var_33 = var_28.verify_signature(var_29, var_31)
    assert var_33 is True
    var_34 = module_0.NoneAlgorithm()
    var_35 = module_0.Signer(var_0, var_1, algorithm=var_34)
    var_36 = b'test-value'
    var_37 = b''
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = b'test-value'
    var_40 = 'concat'
    var_41 = module_0.Signer(var_0, var_1, key_derivation=var_40)
    var_42 = b'test-value'
    var_43 = var_41.get_signature(var_42)
    var_44 = var_41.verify_signature(var_42, var_43)
    assert var_44 is True
    var_45 = 'hmac'
    var_46 = module_0.Signer(var_0, var_1, key_derivation=var_45)
    var_47 = b'test-value'
    var_48 = var_46.get_signature(var_47)
    var_49 = var_46.verify_signature(var_47, var_48)
    assert var_49 is True
    var_50 = 'none'
    var_51 = module_0.Signer(var_0, var_1, key_derivation=var_50)
    var_52 = b'test-value'
    var_53 = var_51.get_signature(var_52)
    var_54 = var_51.verify_signature(var_52, var_53)
    assert var_54 is True



# Parsed testcases at query #25
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = '!!!invalid-base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'different-secret'
    var_13 = module_0.Signer(var_12)
    var_14 = var_13.get_signature(var_2)
    var_15 = var_1.verify_signature(var_2, var_14)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'rotation-test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = 'secret'
    var_25 = module_0.Signer(var_24, algorithm=var_23)
    var_26 = var_25.get_signature(var_2)
    var_27 = var_25.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = 'key1'
    var_29 = 'key2'
    var_30 = 'key3'
    var_31 = [var_28, var_29, var_30]
    var_32 = module_0.Signer(var_31)
    var_33 = b'multi-key-test'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = 'wrong-secret'
    var_37 = module_0.Signer(var_36)
    var_38 = var_37.get_signature(var_33)
    var_39 = var_32.verify_signature(var_33, var_38)
    assert var_39 is False
    var_40 = 'salt1'
    var_41 = module_0.Signer(var_24, var_40)
    var_42 = 'salt2'
    var_43 = module_0.Signer(var_24, var_42)
    var_44 = b'salt-test'
    var_45 = var_41.get_signature(var_44)
    var_46 = var_41.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = var_43.verify_signature(var_44, var_45)
    assert var_47 is False



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = 'test-string-value'
    var_7 = var_2.get_signature(var_6)
    var_8 = var_2.verify_signature(var_6, var_7)
    assert var_8 is True
    var_9 = b'invalid-signature'
    var_10 = var_2.verify_signature(var_3, var_9)
    assert var_10 is False
    var_11 = b'tampered'
    var_12 = 1
    var_13 = var_4[var_12:]
    var_14 = var_11 + var_13
    var_15 = var_2.verify_signature(var_3, var_14)
    assert var_15 is False
    var_16 = b''
    var_17 = var_2.get_signature(var_16)
    var_18 = var_2.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = module_0.Signer(var_1)
    var_20 = var_19.verify_signature(var_3, var_4)
    assert var_20 is True
    var_21 = 'different-secret-key'
    var_22 = module_0.Signer(var_21)
    var_23 = var_22.verify_signature(var_3, var_4)
    assert var_23 is False
    var_24 = b'mixed-value'
    var_25 = var_2.get_signature(var_24)
    var_26 = 'mixed-value'
    var_27 = var_2.verify_signature(var_26, var_25)
    assert var_27 is True
    var_28 = b'!!!invalid-base64!!!'
    var_29 = var_2.verify_signature(var_3, var_28)
    assert var_29 is False
    var_30 = 'old-key'
    var_31 = 'newer-key'
    var_32 = 'newest-key'
    var_33 = [var_30, var_31, var_32]
    var_34 = module_0.Signer(var_33)
    var_35 = b'rotation-test'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = [var_30]
    var_39 = module_0.Signer(var_38)
    var_40 = var_39.get_signature(var_35)
    var_41 = var_34.verify_signature(var_35, var_40)
    assert var_41 is True
    var_42 = 'test'
    var_43 = 'concat'
    var_44 = module_0.Signer(var_42, key_derivation=var_43)
    var_45 = b'concat-test'
    var_46 = var_44.get_signature(var_45)
    var_47 = var_44.verify_signature(var_45, var_46)
    assert var_47 is True
    var_48 = 'hmac'
    var_49 = module_0.Signer(var_42, key_derivation=var_48)
    var_50 = b'hmac-test'
    var_51 = var_49.get_signature(var_50)
    var_52 = var_49.verify_signature(var_50, var_51)
    assert var_52 is True
    var_53 = 'none'
    var_54 = module_0.Signer(var_42, key_derivation=var_53)
    var_55 = b'none-test'
    var_56 = var_54.get_signature(var_55)
    var_57 = var_54.verify_signature(var_55, var_56)
    assert var_57 is True
    var_58 = b'sha256-test'



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
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = module_0.NoneAlgorithm()
    var_10 = module_0.Signer(var_0, algorithm=var_9)
    var_11 = var_10.get_signature(var_2)
    var_12 = var_10.verify_signature(var_2, var_11)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = var_16.get_signature(var_2)
    var_18 = module_0.Signer(var_13)
    var_19 = var_18.verify_signature(var_2, var_17)
    assert var_19 is True
    var_20 = 'concat'
    var_21 = module_0.Signer(var_0, key_derivation=var_20)
    var_22 = var_21.get_signature(var_2)
    var_23 = var_21.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = 'hmac'
    var_25 = module_0.Signer(var_0, key_derivation=var_24)
    var_26 = var_25.get_signature(var_2)
    var_27 = var_25.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = 'key1'
    var_29 = module_0.Signer(var_28)
    var_30 = 'key2'
    var_31 = module_0.Signer(var_30)
    var_32 = var_29.get_signature(var_2)
    var_33 = var_31.verify_signature(var_2, var_32)
    assert var_33 is False



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
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'test-value'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = [var_10, var_11]
    var_18 = module_0.Signer(var_17)
    var_19 = module_0.Signer(var_10)
    var_20 = b'test-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_18.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid-base64!!!'
    var_24 = var_1.verify_signature(var_20, var_23)
    assert var_24 is False
    var_25 = 'test-value-string'
    var_26 = var_1.get_signature(var_25)
    var_27 = var_1.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'test'
    var_29 = var_1.get_signature(var_28)
    var_30 = 'test'
    var_31 = var_1.verify_signature(var_30, var_29)
    assert var_31 is True
    var_32 = var_1.get_signature(var_30)
    var_33 = var_1.verify_signature(var_28, var_32)
    assert var_33 is True
    var_34 = var_1.verify_signature(var_20, var_7)
    assert var_34 is False
    var_35 = module_0.NoneAlgorithm()
    var_36 = module_0.Signer(var_0, algorithm=var_35)
    var_37 = b'test'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = b'different-value'
    var_41 = var_36.verify_signature(var_40, var_38)
    assert var_41 is False
    var_42 = b'test'
    var_43 = b'wrong-value'



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid_sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'string value'
    var_12 = var_2.get_signature(var_11)
    var_13 = var_2.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = b'!!!invalid_base64!!!'
    var_15 = var_2.verify_signature(var_3, var_14)
    assert var_15 is False
    var_16 = b''
    var_17 = var_2.verify_signature(var_3, var_16)
    assert var_17 is False
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_0, algorithm=var_18)
    var_20 = b'test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'old-key'
    var_24 = 'new-key'
    var_25 = [var_23, var_24]
    var_26 = module_0.Signer(var_25)
    var_27 = b'rotated value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = 'secret'
    var_31 = 'salt1'
    var_32 = module_0.Signer(var_30, var_31)
    var_33 = 'salt2'
    var_34 = module_0.Signer(var_30, var_33)
    var_35 = var_32.get_signature(var_27)
    var_36 = var_34.verify_signature(var_27, var_35)
    assert var_36 is False



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b'tampered value'
    var_9 = var_1.verify_signature(var_8, var_3)
    assert var_9 is False
    var_10 = b'!!!invalid-base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b''
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'different-secret'
    var_16 = module_0.Signer(var_15)
    var_17 = var_16.verify_signature(var_2, var_3)
    assert var_17 is False
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = b'rotation test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = 'key'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = b'test'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b'any'
    var_32 = var_27.verify_signature(var_28, var_31)
    assert var_32 is False
    var_33 = 'string value'
    var_34 = var_1.get_signature(var_33)
    var_35 = var_1.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = b'bytes value'
    var_37 = var_1.get_signature(var_36)
    var_38 = var_1.verify_signature(var_36, var_37)
    assert var_38 is True



# Parsed testcases at query #31
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = b'not-base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = var_1.verify_signature(var_2, var_7)
    assert var_12 is False
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'test-value'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'test-value'
    var_21 = var_16.verify_signature(var_20, var_18)
    assert var_21 is True
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_0, algorithm=var_22)
    var_24 = var_23.get_signature(var_17)
    var_25 = var_23.verify_signature(var_17, var_24)
    assert var_25 is True
    var_26 = var_23.verify_signature(var_17, var_7)
    assert var_26 is True
    var_27 = b'any-sig'
    var_28 = var_23.verify_signature(var_17, var_27)
    assert var_28 is True



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
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'different-secret-key'
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'not-base64!!'
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = b''
    var_18 = var_1.verify_signature(var_2, var_17)
    assert var_18 is False
    var_19 = 'key'
    var_20 = module_0.NoneAlgorithm()
    var_21 = module_0.Signer(var_19, algorithm=var_20)
    var_22 = b'test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = 'old-key'
    var_26 = 'new-key'
    var_27 = [var_25, var_26]
    var_28 = module_0.Signer(var_27)
    var_29 = b'rotated-value'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = var_28.derive_key(var_25)
    var_33 = 'string-value'
    var_34 = var_1.verify_signature(var_33, var_3)
    assert var_34 is True
    var_35 = 'ascii'
    var_36 = b'|'
    var_37 = module_0.Signer(var_19, sep=var_36)
    var_38 = b'custom-sep'
    var_39 = var_37.get_signature(var_38)
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = b'x'
    var_42 = 10000
    var_43 = var_41 * var_42
    var_44 = var_1.get_signature(var_43)
    var_45 = var_1.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = b'custom-salt'
    var_47 = module_0.Signer(var_19, var_46)
    var_48 = b'salted-value'
    var_49 = var_47.get_signature(var_48)
    var_50 = var_47.verify_signature(var_48, var_49)
    assert var_50 is True
    var_51 = var_1.verify_signature(var_48, var_49)
    assert var_51 is False
    var_52 = b'hello\nworld\t!'
    var_53 = var_1.get_signature(var_52)
    var_54 = var_1.verify_signature(var_52, var_53)
    assert var_54 is True
    var_55 = var_1.verify_signature(var_2, var_3)
    assert var_55 is True
    var_56 = var_1.verify_signature(var_2, var_3)
    assert var_56 is True



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
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'rotation-test'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_0, var_1, algorithm=var_17)
    var_19 = b'test-none'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = b'secret-key'
    var_23 = b'test-salt'
    var_24 = module_0.Signer(var_22, var_23)
    var_25 = b'bytes-value'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'string-value'
    var_29 = var_24.get_signature(var_28)
    var_30 = var_24.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b'!!!invalid-base64!!!'
    var_32 = var_2.verify_signature(var_3, var_31)
    assert var_32 is False
    var_33 = b'modified-value'
    var_34 = var_2.verify_signature(var_33, var_4)
    assert var_34 is False



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
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'not-base64!!'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = ''
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b'other-value'
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'test-value'
    var_16 = var_1.verify_signature(var_15, var_3)
    assert var_16 is True
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'rotation-test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.Signer(var_17)
    var_25 = var_24.get_signature(var_21)
    var_26 = var_20.verify_signature(var_21, var_25)
    assert var_26 is True
    var_27 = 'wrong-key'
    var_28 = module_0.Signer(var_27)
    var_29 = var_28.get_signature(var_21)
    var_30 = var_20.verify_signature(var_21, var_29)
    assert var_30 is False
    var_31 = 'different-salt'
    var_32 = module_0.Signer(var_0, var_31)
    var_33 = var_32.get_signature(var_2)
    var_34 = var_1.verify_signature(var_2, var_33)
    assert var_34 is False
    var_35 = var_32.verify_signature(var_2, var_33)
    assert var_35 is True
    var_36 = module_0.NoneAlgorithm()
    var_37 = module_0.Signer(var_0, algorithm=var_36)
    var_38 = var_37.get_signature(var_2)
    var_39 = var_37.verify_signature(var_2, var_38)
    assert var_39 is True
    var_40 = b''
    var_41 = var_37.verify_signature(var_2, var_40)
    assert var_41 is True
    var_42 = b'anything'
    var_43 = var_37.verify_signature(var_2, var_42)
    assert var_43 is False



# Parsed testcases at query #35
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
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'!!!invalid-base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b'other-value'
    var_13 = var_1.verify_signature(var_12, var_3)
    assert var_13 is False
    var_14 = 'test-string'
    var_15 = var_1.get_signature(var_14)
    var_16 = module_0.NoneAlgorithm()
    var_17 = module_0.Signer(var_0, algorithm=var_16)
    var_18 = b'no-sig-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = b'some-sig'
    var_22 = var_17.verify_signature(var_18, var_21)
    assert var_22 is False
    var_23 = 'old-key'
    var_24 = 'new-key'
    var_25 = [var_23, var_24]
    var_26 = module_0.Signer(var_25)
    var_27 = b'multi-key-value'
    var_28 = module_0.Signer(var_23)
    var_29 = var_28.get_signature(var_27)
    var_30 = module_0.Signer(var_24)
    var_31 = var_30.get_signature(var_27)
    var_32 = var_26.verify_signature(var_27, var_29)
    assert var_32 is True
    var_33 = var_26.verify_signature(var_27, var_31)
    assert var_33 is True
    var_34 = 'unknown-key'
    var_35 = module_0.Signer(var_34)
    var_36 = var_35.get_signature(var_27)
    var_37 = var_26.verify_signature(var_27, var_36)
    assert var_37 is False
    var_38 = 'secret'
    var_39 = 'salt1'
    var_40 = module_0.Signer(var_38, var_39)
    var_41 = 'salt2'
    var_42 = module_0.Signer(var_38, var_41)
    var_43 = b'salt-test'
    var_44 = var_40.get_signature(var_43)
    var_45 = var_40.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = var_42.verify_signature(var_43, var_44)
    assert var_46 is False
    var_47 = 'concat'
    var_48 = module_0.Signer(var_38, key_derivation=var_47)
    var_49 = 'hmac'
    var_50 = module_0.Signer(var_38, key_derivation=var_49)
    var_51 = b'derivation-test'
    var_52 = var_48.get_signature(var_51)
    var_53 = var_50.get_signature(var_51)
    var_54 = var_48.verify_signature(var_51, var_52)
    assert var_54 is True
    var_55 = var_48.verify_signature(var_51, var_53)
    assert var_55 is False
    var_56 = var_50.verify_signature(var_51, var_53)
    assert var_56 is True
    var_57 = var_50.verify_signature(var_51, var_52)
    assert var_57 is False
    var_58 = b'digest-test'



# Parsed testcases at query #36
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'test-string'
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'not-base64!@#'
    var_16 = var_2.verify_signature(var_3, var_15)
    assert var_16 is False
    var_17 = b''
    var_18 = var_2.verify_signature(var_3, var_17)
    assert var_18 is False
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21, var_1)
    var_23 = b'test-value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True



# Parsed testcases at query #37
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'test value'
    var_10 = b'modified value'
    var_11 = var_1.verify_signature(var_10, var_3)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'test value 2'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = module_0.Signer(var_12)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_20)
    assert var_21 is True
    var_22 = 'different-salt'
    var_23 = module_0.Signer(var_0, var_22)
    var_24 = var_23.get_signature(var_2)
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = b'!!!invalid_base64!!!'
    var_27 = var_1.verify_signature(var_2, var_26)
    assert var_27 is False
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_0, algorithm=var_28)
    var_30 = var_29.get_signature(var_2)
    var_31 = var_29.verify_signature(var_2, var_30)
    assert var_31 is True
    var_32 = b'any_sig'
    var_33 = var_29.verify_signature(var_2, var_32)
    assert var_33 is True
    var_34 = module_0.Signer(var_7)
    var_35 = var_34.get_signature(var_2)
    var_36 = var_34.verify_signature(var_2, var_35)
    assert var_36 is True



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
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'original-value'
    var_12 = var_1.get_signature(var_11)
    var_13 = b'modified-value'
    var_14 = var_1.verify_signature(var_13, var_12)
    assert var_14 is False
    var_15 = b'!!invalid-base64!!'
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'rotation-test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = 'salt1'
    var_25 = module_0.Signer(var_0, var_24)
    var_26 = 'salt2'
    var_27 = module_0.Signer(var_0, var_26)
    var_28 = b'salt-test'
    var_29 = var_25.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is False
    var_31 = module_0.NoneAlgorithm()
    var_32 = module_0.Signer(var_0, algorithm=var_31)
    var_33 = b'none-algorithm-test'
    var_34 = var_32.get_signature(var_33)
    var_35 = b''
    var_36 = module_1.base64_encode(var_35)
    var_37 = var_32.verify_signature(var_33, var_34)
    assert var_37 is True
    var_38 = 'string-value'
    var_39 = var_1.get_signature(var_38)
    var_40 = var_1.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = var_1.derive_key()
    var_42 = b':test'
    var_43 = var_41 + var_42
    var_44 = module_1.base64_encode(var_43)



# Parsed testcases at query #39
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = b'test rotation'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_0, var_1, algorithm=var_19)
    var_21 = b'test none algorithm'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b':'
    var_25 = module_0.Signer(var_0, var_1, var_24)
    var_26 = b'test separator'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'test string'
    var_30 = var_2.get_signature(var_29)
    var_31 = var_2.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = -1
    var_33 = var_4[:var_32]
    var_34 = b'X'
    var_35 = var_33 + var_34
    var_36 = var_2.verify_signature(var_3, var_35)
    assert var_36 is False
    var_37 = b'different value'
    var_38 = var_2.verify_signature(var_37, var_4)
    assert var_38 is False



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
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = module_0.NoneAlgorithm()
    var_12 = module_0.Signer(var_0, algorithm=var_11)
    var_13 = var_12.verify_signature(var_2, var_7)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'salt1'
    var_22 = module_0.Signer(var_0, var_21)
    var_23 = 'salt2'
    var_24 = module_0.Signer(var_0, var_23)
    var_25 = b'test-value'
    var_26 = var_22.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is False
    var_28 = b'|'
    var_29 = module_0.Signer(var_0, sep=var_28)
    var_30 = b'test-value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = 'secret-key'
    var_34 = b'test-value'
    var_35 = var_1.get_signature(var_34)
    var_36 = var_1.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = b'!!!invalid-base64!!!'
    var_38 = var_1.verify_signature(var_34, var_37)
    assert var_38 is False
    var_39 = b'test-value'
    var_40 = var_1.get_signature(var_39)
    var_41 = b'modified-value'
    var_42 = var_1.verify_signature(var_41, var_40)
    assert var_42 is False



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
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b'!!!invalid-base64!!!'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.verify_signature(var_10, var_3)
    assert var_11 is False
    var_12 = var_1.verify_signature(var_2, var_10)
    assert var_12 is False
    var_13 = b'different-value'
    var_14 = var_1.verify_signature(var_13, var_3)
    assert var_14 is False
    var_15 = 'test-value'
    var_16 = var_1.verify_signature(var_15, var_3)
    assert var_16 is True
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'rotation-test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = [var_17]
    var_25 = module_0.Signer(var_24)
    var_26 = var_25.verify_signature(var_21, var_22)
    assert var_26 is False
    var_27 = 'secret'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_27, algorithm=var_28)
    var_30 = b'test'
    var_31 = var_29.get_signature(var_30)
    var_32 = module_1.base64_encode(var_10)
    var_33 = var_29.verify_signature(var_30, var_31)
    assert var_33 is True
    var_34 = b'sha256-test'
    var_35 = 'secret'
    var_36 = b'kd-test'



# Parsed testcases at query #42
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'wrong-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = 'not-base64!!'
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
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_0, var_1, algorithm=var_18)
    var_20 = b'test-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True



# Parsed testcases at query #43
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
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'different-value'
    var_12 = var_2.verify_signature(var_11, var_4)
    assert var_12 is False
    var_13 = 'test-string-value'
    var_14 = var_2.get_signature(var_13)
    var_15 = var_2.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_1)
    var_20 = b'test-value-rotation'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, algorithm=var_23)
    var_25 = b'test'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'!!!invalid-base64!!!'
    var_29 = var_2.verify_signature(var_3, var_28)
    assert var_29 is False
    var_30 = b''
    var_31 = var_2.verify_signature(var_3, var_30)
    assert var_31 is False
    var_32 = 'different-salt'
    var_33 = module_0.Signer(var_0, var_32)
    var_34 = var_2.get_signature(var_25)
    var_35 = var_33.verify_signature(var_25, var_34)
    assert var_35 is False



# Parsed testcases at query #44
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid_sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13, var_1)
    var_15 = b'test value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = b'test value'
    var_19 = var_2.verify_signature(var_18, var_4)
    assert var_19 is True
    var_20 = b'AAAA'
    var_21 = 4
    var_22 = var_4[var_21:]
    var_23 = var_20 + var_22
    var_24 = var_2.verify_signature(var_3, var_23)
    assert var_24 is False
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_0, algorithm=var_25)
    var_27 = var_26.get_signature(var_3)
    var_28 = var_26.verify_signature(var_3, var_27)
    assert var_28 is True
    var_29 = b'!!!invalid_base64!!!'
    var_30 = var_2.verify_signature(var_3, var_29)
    assert var_30 is False
    var_31 = 'different-key'
    var_32 = module_0.Signer(var_31, var_1)
    var_33 = var_32.get_signature(var_3)
    var_34 = var_2.verify_signature(var_3, var_33)
    assert var_34 is False



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
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_8, var_3)
    assert var_9 is False
    var_10 = b'different-value'
    var_11 = var_1.verify_signature(var_10, var_3)
    assert var_11 is False
    var_12 = 'test-value'
    var_13 = module_1.base64_encode(var_3)
    var_14 = var_1.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'invalid-base64!'
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'rotation-test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_0, algorithm=var_24)
    var_26 = b'none-algorithm-test'
    var_27 = var_25.get_signature(var_26)
    var_28 = module_1.base64_encode(var_8)
    var_29 = var_25.verify_signature(var_26, var_27)
    assert var_29 is True
    var_30 = 'custom-salt'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = b'custom-salt-test'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = b'|'
    var_36 = module_0.Signer(var_0, sep=var_35)
    var_37 = b'separator-test'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = b'hmac-test'
    var_41 = module_0.Signer(var_0)
    var_42 = b'empty-sig-test'
    var_43 = var_41.verify_signature(var_42, var_8)
    assert var_43 is False
    var_44 = b'x'
    var_45 = 10000
    var_46 = var_44 * var_45
    var_47 = var_1.get_signature(var_46)
    var_48 = var_1.verify_signature(var_46, var_47)
    assert var_48 is True



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test_value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test_value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'invalid_sig'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = module_0.NoneAlgorithm()
    var_12 = module_0.Signer(var_0, algorithm=var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_12.verify_signature(var_2, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'test_value_2'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = module_0.Signer(var_15)
    var_23 = var_22.get_signature(var_19)
    var_24 = var_18.verify_signature(var_19, var_23)
    assert var_24 is True
    var_25 = b'!!!invalid_base64!!!'
    var_26 = var_1.verify_signature(var_2, var_25)
    assert var_26 is False
    var_27 = 'different-salt'
    var_28 = module_0.Signer(var_0, var_27)
    var_29 = var_28.get_signature(var_2)
    var_30 = var_28.verify_signature(var_2, var_29)
    assert var_30 is True
    var_31 = var_1.verify_signature(var_2, var_29)
    assert var_31 is False
    var_32 = b'different_value'
    var_33 = var_1.verify_signature(var_32, var_3)
    assert var_33 is False



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
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'different-value'
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is False
    var_11 = 'not-base64'
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = b''
    var_14 = var_2.get_signature(var_13)
    var_15 = var_2.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_1)
    var_20 = b'test-value-2'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, var_1, algorithm=var_23)
    var_25 = b'test'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'|'
    var_29 = module_0.Signer(var_0, var_1, var_28)
    var_30 = b'test-value-3'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = b'fake-signature'
    var_34 = var_19.verify_signature(var_20, var_33)
    assert var_34 is False



# Parsed testcases at query #48
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'different-secret-key'
    var_12 = module_0.Signer(var_11, var_1)
    var_13 = var_12.verify_signature(var_3, var_4)
    assert var_13 is False
    var_14 = 'different-salt'
    var_15 = module_0.Signer(var_0, var_14)
    var_16 = var_15.verify_signature(var_3, var_4)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19, var_1)
    var_21 = b'test-value-2'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = [var_17]
    var_25 = module_0.Signer(var_24, var_1)
    var_26 = var_25.get_signature(var_21)
    var_27 = var_20.verify_signature(var_21, var_26)
    assert var_27 is True
    var_28 = 'test-key'
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_28, algorithm=var_29)
    var_31 = b'test'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b'any-sig'
    var_35 = var_30.verify_signature(var_31, var_34)
    assert var_35 is True
    var_36 = b'!!!invalid-base64!!!'
    var_37 = var_2.verify_signature(var_3, var_36)
    assert var_37 is False
    var_38 = 'string-value'
    var_39 = var_2.get_signature(var_38)
    var_40 = var_2.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = 'ascii'



# Parsed testcases at query #49
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'different-salt'
    var_11 = module_0.Signer(var_0, var_10)
    var_12 = var_11.get_signature(var_2)
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'different-secret-key'
    var_15 = module_0.Signer(var_14)
    var_16 = var_15.get_signature(var_2)
    var_17 = var_1.verify_signature(var_2, var_16)
    assert var_17 is False
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = var_21.get_signature(var_2)
    var_23 = var_21.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = module_0.Signer(var_18)
    var_25 = var_24.get_signature(var_2)
    var_26 = var_21.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, algorithm=var_27)
    var_29 = var_28.get_signature(var_2)
    var_30 = var_28.verify_signature(var_2, var_29)
    assert var_30 is True
    var_31 = 'test string'
    var_32 = var_1.get_signature(var_31)
    var_33 = var_1.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b'!!!invalid_base64!!!'
    var_35 = var_1.verify_signature(var_2, var_34)
    assert var_35 is False
    var_36 = 'concat'
    var_37 = module_0.Signer(var_0, key_derivation=var_36)
    var_38 = var_37.get_signature(var_2)
    var_39 = var_37.verify_signature(var_2, var_38)
    assert var_39 is True
    var_40 = 'hmac'
    var_41 = module_0.Signer(var_0, key_derivation=var_40)
    var_42 = var_41.get_signature(var_2)
    var_43 = var_41.verify_signature(var_2, var_42)
    assert var_43 is True
    var_44 = 'none'
    var_45 = module_0.Signer(var_0, key_derivation=var_44)
    var_46 = var_45.get_signature(var_2)
    var_47 = var_45.verify_signature(var_2, var_46)
    assert var_47 is True



# Parsed testcases at query #50
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that verify_signature correctly validates and invalidates signatures.'
    var_1 = 'test-secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'different-value'
    var_8 = var_3.verify_signature(var_7, var_5)
    assert var_8 is False
    var_9 = b'tampered-signature'
    var_10 = var_3.verify_signature(var_4, var_9)
    assert var_10 is False
    var_11 = b''
    var_12 = var_3.get_signature(var_11)
    var_13 = var_3.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16, var_2)
    var_18 = b'rotation-test'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.Signer(var_14, var_2)
    var_22 = var_21.get_signature(var_18)
    var_23 = var_17.verify_signature(var_18, var_22)
    assert var_23 is True
    var_24 = 'test-key'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = b'none-test'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'any-signature'
    var_31 = var_26.verify_signature(var_27, var_30)
    assert var_31 is True
    var_32 = b'!!!invalid-base64!!!'
    var_33 = b'test'
    var_34 = var_3.verify_signature(var_33, var_32)
    assert var_34 is False
    var_35 = b'bytes-value'
    var_36 = var_3.get_signature(var_35)
    var_37 = 'ascii'
    var_38 = b''
    var_39 = var_3.verify_signature(var_33, var_38)
    assert var_39 is False



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'string-value'
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'not-base64!!!'
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = b''
    var_14 = var_1.get_signature(var_13)
    var_15 = var_1.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'test-key'
    var_17 = 'concat'
    var_18 = module_0.Signer(var_16, key_derivation=var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = 'hmac'
    var_22 = module_0.Signer(var_16, key_derivation=var_21)
    var_23 = var_22.get_signature(var_2)
    var_24 = var_22.verify_signature(var_2, var_23)
    assert var_24 is True
    var_25 = 'none'
    var_26 = module_0.Signer(var_16, key_derivation=var_25)
    var_27 = var_26.get_signature(var_2)
    var_28 = var_26.verify_signature(var_2, var_27)
    assert var_28 is True
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_16, algorithm=var_29)
    var_31 = var_30.get_signature(var_2)
    var_32 = var_30.verify_signature(var_2, var_31)
    assert var_32 is True
    var_33 = 'old-key'
    var_34 = 'new-key'
    var_35 = [var_33, var_34]
    var_36 = module_0.Signer(var_35)
    var_37 = var_36.get_signature(var_2)
    var_38 = [var_33, var_34]
    var_39 = module_0.Signer(var_38)
    var_40 = var_39.verify_signature(var_2, var_37)
    assert var_40 is True



# Parsed testcases at query #52
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
    var_5 = b'invalid_signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = -1
    var_9 = var_3[:var_8]
    var_10 = b'x'
    var_11 = var_9 + var_10
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = b''
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = b'!!!invalid!!!'
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'test_value'
    var_18 = var_1.verify_signature(var_17, var_3)
    assert var_18 is True
    var_19 = b'test_value'
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = 'test'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = var_23.get_signature(var_24)
    var_28 = var_23.verify_signature(var_24, var_27)
    assert var_28 is True
    var_29 = module_0.NoneAlgorithm()
    var_30 = 'secret'
    var_31 = module_0.Signer(var_30, algorithm=var_29)
    var_32 = var_31.get_signature(var_24)
    var_33 = var_31.verify_signature(var_24, var_32)
    assert var_33 is True
    var_34 = 'custom-salt'
    var_35 = module_0.Signer(var_30, var_34)
    var_36 = var_35.get_signature(var_24)
    var_37 = var_35.verify_signature(var_24, var_36)
    assert var_37 is True
    var_38 = b':'
    var_39 = module_0.Signer(var_30, sep=var_38)
    var_40 = var_39.get_signature(var_24)
    var_41 = var_39.verify_signature(var_24, var_40)
    assert var_41 is True
    var_42 = 'different-secret'
    var_43 = module_0.Signer(var_42)
    var_44 = var_43.verify_signature(var_2, var_3)
    assert var_44 is False
    var_45 = var_1.verify_signature(var_13, var_3)
    assert var_45 is False
    var_46 = b'test with \x00 null byte and \xff binary'
    var_47 = var_1.get_signature(var_46)
    var_48 = var_1.verify_signature(var_46, var_47)
    assert var_48 is True
    var_49 = 10000
    var_50 = var_10 * var_49
    var_51 = var_1.get_signature(var_50)
    var_52 = var_1.verify_signature(var_50, var_51)
    assert var_52 is True



# Parsed testcases at query #53
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = 'ascii'
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotation-test'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = b'!!!invalid-base64!!!'
    var_20 = var_1.verify_signature(var_2, var_19)
    assert var_20 is False
    var_21 = 'secret'
    var_22 = b'test'
    var_23 = b'custom-sig'
    var_24 = b'wrong-sig'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_21, algorithm=var_25)
    var_27 = var_26.verify_signature(var_22, var_7)
    assert var_27 is True
    var_28 = b'something'
    var_29 = var_26.verify_signature(var_22, var_28)
    assert var_29 is False
    var_30 = 'custom-salt'
    var_31 = module_0.Signer(var_21, var_30)
    var_32 = b'salt-test'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = 'different-salt'
    var_36 = module_0.Signer(var_21, var_35)
    var_37 = var_36.verify_signature(var_32, var_33)
    assert var_37 is False
    var_38 = b'|'
    var_39 = module_0.Signer(var_21, sep=var_38)
    var_40 = b'sep-test'
    var_41 = var_39.get_signature(var_40)
    var_42 = var_39.verify_signature(var_40, var_41)
    assert var_42 is True
    var_43 = 'concat'
    var_44 = module_0.Signer(var_21, key_derivation=var_43)
    var_45 = b'concat-test'
    var_46 = var_44.get_signature(var_45)
    var_47 = var_44.verify_signature(var_45, var_46)
    assert var_47 is True
    var_48 = 'hmac'
    var_49 = module_0.Signer(var_21, key_derivation=var_48)
    var_50 = b'hmac-test'
    var_51 = var_49.get_signature(var_50)
    var_52 = var_49.verify_signature(var_50, var_51)
    assert var_52 is True
    var_53 = 'none'
    var_54 = module_0.Signer(var_21, key_derivation=var_53)
    var_55 = b'none-test'
    var_56 = var_54.get_signature(var_55)
    var_57 = var_54.verify_signature(var_55, var_56)
    assert var_57 is True
    var_58 = module_0.Signer(var_21)
    var_59 = b'mixed'
    var_60 = var_58.get_signature(var_59)
    var_61 = b'mixed'
    var_62 = var_58.verify_signature(var_61, var_60)
    assert var_62 is True
    var_63 = 'mixed'
    var_64 = var_58.verify_signature(var_63, var_60)
    assert var_64 is True
    var_65 = b'modified-'
    var_66 = var_65 + var_2
    var_67 = var_1.verify_signature(var_66, var_3)
    assert var_67 is False



# Parsed testcases at query #54
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'bytes-value'
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'string-value'
    var_12 = var_2.get_signature(var_11)
    var_13 = var_2.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = b'original'
    var_15 = var_2.get_signature(var_14)
    var_16 = b'modified'
    var_17 = var_2.verify_signature(var_16, var_15)
    assert var_17 is False
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20, var_1)
    var_22 = b'test-rotation'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = module_0.Signer(var_18, var_1)
    var_26 = var_25.get_signature(var_22)
    var_27 = var_21.verify_signature(var_22, var_26)
    assert var_27 is True
    var_28 = 'test-key'
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_28, algorithm=var_29)
    var_31 = b'test-none'
    var_32 = var_30.get_signature(var_31)
    assert var_32 == b''
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b'test-sha256'
    var_35 = b''
    var_36 = var_2.get_signature(var_35)
    var_37 = var_2.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = b'value with spaces and !@#$%^&*()'
    var_39 = var_2.get_signature(var_38)
    var_40 = var_2.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = b'\xff\xfe\xfd'
    var_42 = var_2.verify_signature(var_3, var_41)
    assert var_42 is False



# Parsed testcases at query #55
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = module_0.Signer(var_11)
    var_19 = var_18.get_signature(var_15)
    var_20 = var_14.verify_signature(var_15, var_19)
    assert var_20 is True
    var_21 = 'wrong-key'
    var_22 = module_0.Signer(var_21)
    var_23 = var_22.get_signature(var_15)
    var_24 = var_14.verify_signature(var_15, var_23)
    assert var_24 is False
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_0, algorithm=var_25)
    var_27 = b'test-value'
    var_28 = var_26.get_signature(var_27)
    assert var_28 == b''
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'!!!invalid-base64!!!'
    var_31 = var_26.verify_signature(var_27, var_30)
    assert var_31 is False



# Parsed testcases at query #56
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test string'
    var_6 = var_1.get_signature(var_5)
    var_7 = var_1.verify_signature(var_5, var_6)
    assert var_7 is True
    var_8 = b'invalid_sig'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b'!!!invalid_base64!!!'
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'different-secret'
    var_15 = module_0.Signer(var_14)
    var_16 = var_15.get_signature(var_2)
    var_17 = var_1.verify_signature(var_2, var_16)
    assert var_17 is False
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = module_0.Signer(var_18)
    var_23 = var_22.get_signature(var_2)
    var_24 = var_21.get_signature(var_2)
    var_25 = var_21.verify_signature(var_2, var_23)
    assert var_25 is True
    var_26 = var_21.verify_signature(var_2, var_24)
    assert var_26 is True
    var_27 = b'different value'
    var_28 = var_1.verify_signature(var_27, var_3)
    assert var_28 is False
    var_29 = 'secret'
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_29, algorithm=var_30)
    var_32 = var_31.get_signature(var_2)
    var_33 = var_31.verify_signature(var_2, var_32)
    assert var_33 is True
    var_34 = var_31.verify_signature(var_2, var_10)
    assert var_34 is True
    var_35 = b'|'
    var_36 = module_0.Signer(var_29, sep=var_35)
    var_37 = b'test'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True



# Parsed testcases at query #57
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'test-value'
    var_11 = var_1.verify_signature(var_10, var_3)
    assert var_11 is True
    var_12 = 'old-secret-key'
    var_13 = module_0.Signer(var_12)
    var_14 = b'old-test-value'
    var_15 = var_13.get_signature(var_14)
    var_16 = 'new-secret-key'
    var_17 = [var_12, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = var_18.verify_signature(var_14, var_15)
    assert var_19 is True
    var_20 = 'different-salt'
    var_21 = module_0.Signer(var_0, var_20)
    var_22 = var_21.get_signature(var_2)
    var_23 = var_1.verify_signature(var_2, var_22)
    assert var_23 is False
    var_24 = b'!!!invalid-base64!!!'
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_0, algorithm=var_26)
    var_28 = b'test-none'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b'different'
    var_32 = var_27.verify_signature(var_28, var_31)
    assert var_32 is False
    var_33 = 'test-secret-key'
    var_34 = b'test-diff-derivation'
    var_35 = b'wrong'
    var_36 = b'value1'
    var_37 = b'value2'
    var_38 = var_1.get_signature(var_36)
    var_39 = var_1.verify_signature(var_36, var_38)
    assert var_39 is True
    var_40 = var_1.verify_signature(var_37, var_38)
    assert var_40 is False



# Parsed testcases at query #58
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'invalid-sig'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'modified-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is False
    var_11 = 'different-key'
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = module_0.Signer(var_15)
    var_21 = var_20.get_signature(var_2)
    var_22 = var_18.verify_signature(var_2, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid-base64!!!'
    var_24 = var_1.verify_signature(var_2, var_23)
    assert var_24 is False
    var_25 = b''
    var_26 = var_1.get_signature(var_25)
    var_27 = var_1.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = var_1.verify_signature(var_2, var_25)
    assert var_28 is False
    var_29 = 'key'
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_29, algorithm=var_30)
    var_32 = var_31.get_signature(var_2)
    var_33 = var_31.verify_signature(var_2, var_32)
    assert var_33 is True
    var_34 = b'different-value'
    var_35 = var_31.verify_signature(var_34, var_32)
    assert var_35 is True



# Parsed testcases at query #59
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'!!!invalid-base64!!!'
    var_13 = var_2.verify_signature(var_3, var_12)
    assert var_13 is False
    var_14 = module_0.NoneAlgorithm()
    var_15 = module_0.Signer(var_0, algorithm=var_14)
    var_16 = b'test'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21, var_1)
    var_23 = b'test value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'|'
    var_27 = module_0.Signer(var_0, var_1, var_26)
    var_28 = b'test value'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = module_0.HMACAlgorithm()
    var_32 = module_0.Signer(var_0, var_1, algorithm=var_31)
    var_33 = b'test value'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = 'test string'
    var_37 = var_2.get_signature(var_36)
    var_38 = var_2.verify_signature(var_36, var_37)
    assert var_38 is True



# Parsed testcases at query #60
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = b'different-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is False
    var_11 = b'!!!invalid-base64!!!'
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = 'test-value'
    var_14 = var_1.verify_signature(var_13, var_3)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_0, algorithm=var_21)
    var_23 = var_22.get_signature(var_2)
    var_24 = var_22.verify_signature(var_2, var_23)
    assert var_24 is True



# Parsed testcases at query #61
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = 'wrong-key'
    var_10 = module_0.Signer(var_9, var_1)
    var_11 = var_10.get_signature(var_3)
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = 'test value'
    var_14 = var_2.verify_signature(var_13, var_4)
    assert var_14 is True
    var_15 = 'not-base64!!!'
    var_16 = var_2.verify_signature(var_3, var_15)
    assert var_16 is False
    var_17 = ''
    var_18 = var_2.verify_signature(var_3, var_17)
    assert var_18 is False
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21, var_1)
    var_23 = b'test value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True



# Parsed testcases at query #62
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'wrong value'
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13, var_1)
    var_15 = b'test value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'not-base64!@#'
    var_19 = var_2.verify_signature(var_15, var_18)
    assert var_19 is False
    var_20 = b''
    var_21 = var_2.verify_signature(var_15, var_20)
    assert var_21 is False
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_0, algorithm=var_22)
    var_24 = b'test value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'test value'
    var_28 = b'wrong'



# Parsed testcases at query #63
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = 'wrong-value'
    var_7 = var_2.verify_signature(var_6, var_4)
    assert var_7 is False
    var_8 = 'invalid-sig'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'!@#$%'
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = module_0.Signer(var_12, var_1)
    var_17 = var_16.get_signature(var_3)
    var_18 = module_0.Signer(var_13, var_1)
    var_19 = var_18.get_signature(var_3)
    var_20 = var_15.verify_signature(var_3, var_17)
    assert var_20 is True
    var_21 = var_15.verify_signature(var_3, var_19)
    assert var_21 is True



# Parsed testcases at query #64
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test Signer.verify_signature method.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'different value'
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is False
    var_11 = 'not-base64!'
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = b''
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = module_0.NoneAlgorithm()
    var_16 = module_0.Signer(var_1, algorithm=var_15)
    var_17 = b'test value'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = b'any-signature'
    var_21 = var_16.verify_signature(var_17, var_20)
    assert var_21 is False
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24)
    var_26 = b'test value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'|'
    var_30 = module_0.Signer(var_1, sep=var_29)
    var_31 = b'test value'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b'different-salt'
    var_35 = module_0.Signer(var_1, var_34)
    var_36 = b'test value'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = module_0.Signer(var_1)
    var_40 = var_39.verify_signature(var_36, var_37)
    assert var_40 is False



# Parsed testcases at query #65
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'test string'
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'different-key'
    var_16 = module_0.Signer(var_15)
    var_17 = var_16.get_signature(var_3)
    var_18 = var_2.verify_signature(var_3, var_17)
    assert var_18 is False
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21)
    var_23 = b'rotated test'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = 'key'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = b'none alg'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'!invalid_base64!'
    var_33 = var_2.verify_signature(var_29, var_32)
    assert var_33 is False
    var_34 = 'custom-salt'
    var_35 = module_0.Signer(var_26, var_34)
    var_36 = b'salt test'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = b':'
    var_40 = module_0.Signer(var_26, sep=var_39)
    var_41 = b'sep test'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = b'sha256 test'
    var_45 = 'hmac'
    var_46 = module_0.Signer(var_26, key_derivation=var_45)
    var_47 = b'hmac test'
    var_48 = var_46.get_signature(var_47)
    var_49 = var_46.verify_signature(var_47, var_48)
    assert var_49 is True



# Parsed testcases at query #66
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
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'test-value'
    var_11 = var_1.verify_signature(var_10, var_3)
    assert var_11 is True
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'test-value-2'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_0, algorithm=var_19)
    var_21 = b'test-value-3'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'!!!invalid-base64!!!'
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = b''
    var_27 = var_1.get_signature(var_26)
    var_28 = var_1.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'custom-salt'
    var_30 = module_0.Signer(var_0, var_29)
    var_31 = b'test-value-4'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = 'other-salt'
    var_35 = module_0.Signer(var_0, var_34)
    var_36 = var_35.verify_signature(var_31, var_32)
    assert var_36 is False
    var_37 = b'|'
    var_38 = module_0.Signer(var_0, sep=var_37)
    var_39 = b'test-value-5'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = 'hmac'
    var_43 = module_0.Signer(var_0, key_derivation=var_42)
    var_44 = b'test-value-6'
    var_45 = var_43.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = 'concat'
    var_48 = module_0.Signer(var_0, key_derivation=var_47)
    var_49 = b'test-value-7'
    var_50 = var_48.get_signature(var_49)
    var_51 = var_48.verify_signature(var_49, var_50)
    assert var_51 is True



# Parsed testcases at query #67
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
    var_5 = b'invalid_signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'!!!invalid_base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b'other_value'
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = b'string_value'
    var_16 = var_1.get_signature(var_15)
    var_17 = 'string_value'
    var_18 = var_1.verify_signature(var_17, var_16)
    assert var_18 is True
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21)
    var_23 = b'test'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = 'wrong-key'
    var_27 = module_0.Signer(var_26)
    var_28 = var_27.get_signature(var_23)
    var_29 = var_22.verify_signature(var_23, var_28)
    assert var_29 is False
    var_30 = 'key'
    var_31 = module_0.NoneAlgorithm()
    var_32 = module_0.Signer(var_30, algorithm=var_31)
    var_33 = b'test'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = var_32.verify_signature(var_33, var_8)
    assert var_36 is True



# Parsed testcases at query #68
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'different-key'
    var_11 = module_0.Signer(var_10)
    var_12 = var_11.get_signature(var_2)
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'string-value'
    var_15 = var_1.get_signature(var_14)
    var_16 = var_1.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'utf-8'
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = var_21.get_signature(var_2)
    var_23 = var_21.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = 'secret'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = var_26.get_signature(var_2)
    var_28 = var_26.verify_signature(var_2, var_27)
    assert var_28 is True
    var_29 = b'!!!invalid-base64!!!'
    var_30 = var_1.verify_signature(var_2, var_29)
    assert var_30 is False
    var_31 = 'different-salt'
    var_32 = module_0.Signer(var_0, var_31)
    var_33 = var_32.get_signature(var_2)
    var_34 = var_1.verify_signature(var_2, var_33)
    assert var_34 is False



# Parsed testcases at query #69
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'modified-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is False
    var_7 = b'invalid-signature'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'\x00\x01\x02'
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'string-value'
    var_16 = var_1.get_signature(var_15)
    var_17 = var_1.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_0, algorithm=var_18)
    var_20 = b'test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid-base64!!!'
    var_24 = var_1.verify_signature(var_2, var_23)
    assert var_24 is False
    var_25 = b''
    var_26 = var_1.verify_signature(var_2, var_25)
    assert var_26 is False
    var_27 = 'old-key'
    var_28 = 'new-key'
    var_29 = [var_27, var_28]
    var_30 = b'test-salt'
    var_31 = module_0.Signer(var_29, var_30)
    var_32 = b'rotation-test'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = b'custom-salt'
    var_36 = module_0.Signer(var_0, var_35)
    var_37 = b'value-with-salt'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = var_1.verify_signature(var_37, var_38)
    assert var_40 is False



# Parsed testcases at query #70
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'invalid'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'secret'
    var_12 = module_0.NoneAlgorithm()
    var_13 = module_0.Signer(var_11, algorithm=var_12)
    var_14 = b'test'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = 'concat'
    var_25 = module_0.Signer(var_11, key_derivation=var_24)
    var_26 = var_25.get_signature(var_21)
    var_27 = var_25.verify_signature(var_21, var_26)
    assert var_27 is True
    var_28 = 'hmac'
    var_29 = module_0.Signer(var_11, key_derivation=var_28)
    var_30 = var_29.get_signature(var_21)
    var_31 = var_29.verify_signature(var_21, var_30)
    assert var_31 is True
    var_32 = 'none'
    var_33 = module_0.Signer(var_11, key_derivation=var_32)
    var_34 = var_33.get_signature(var_21)
    var_35 = var_33.verify_signature(var_21, var_34)
    assert var_35 is True
    var_36 = b':'
    var_37 = module_0.Signer(var_11, sep=var_36)
    var_38 = var_37.get_signature(var_21)
    var_39 = var_37.verify_signature(var_21, var_38)
    assert var_39 is True
    var_40 = b'custom-salt'
    var_41 = module_0.Signer(var_11, var_40)
    var_42 = var_41.get_signature(var_21)
    var_43 = var_41.verify_signature(var_21, var_42)
    assert var_43 is True
    var_44 = b'secret-key'
    var_45 = module_0.Signer(var_44)
    var_46 = var_45.get_signature(var_21)
    var_47 = var_45.verify_signature(var_21, var_46)
    assert var_47 is True
    var_48 = '!!!invalid base64!!!'
    var_49 = var_1.verify_signature(var_21, var_48)
    assert var_49 is False
    var_50 = None
    var_51 = module_0.Signer(var_11, var_50)
    var_52 = var_51.get_signature(var_21)
    var_53 = var_51.verify_signature(var_21, var_52)
    assert var_53 is True
    var_54 = var_1.get_signature(var_21)
    var_55 = b'corrupted'
    var_56 = var_1.verify_signature(var_55, var_54)
    assert var_56 is False



# Parsed testcases at query #71
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is False
    var_7 = b'test-value'
    var_8 = b'invalid-signature'
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'test-value-2'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'different-salt'
    var_18 = module_0.Signer(var_0, var_17)
    var_19 = var_18.verify_signature(var_2, var_3)
    assert var_19 is False



# Parsed testcases at query #72
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'string value'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'test value'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = module_0.Signer(var_13)
    var_21 = var_20.get_signature(var_17)
    var_22 = var_16.verify_signature(var_17, var_21)
    assert var_22 is True
    var_23 = b'original value'
    var_24 = var_16.get_signature(var_23)
    var_25 = b'modified value'
    var_26 = var_16.verify_signature(var_25, var_24)
    assert var_26 is False
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, algorithm=var_27)
    var_29 = b'test'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = '!!!invalid_base64!!!'
    var_33 = var_28.verify_signature(var_29, var_32)
    assert var_33 is False
    var_34 = b''
    var_35 = var_28.verify_signature(var_29, var_34)
    assert var_35 is False



# Parsed testcases at query #73
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
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = b'!!!invalid-base64!!!'
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = b''
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = module_0.NoneAlgorithm()
    var_16 = module_0.Signer(var_0, algorithm=var_15)
    var_17 = b'another-value'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = b'rotated-value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = module_0.Signer(var_20)
    var_28 = var_27.get_signature(var_24)
    var_29 = var_23.verify_signature(var_24, var_28)
    assert var_29 is True
    var_30 = module_0.HMACAlgorithm()
    var_31 = module_0.Signer(var_0, algorithm=var_30)
    var_32 = b'hmac-value'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = b'custom-digest'
    var_36 = 'custom-salt'
    var_37 = module_0.Signer(var_0, var_36)
    var_38 = b'salted-value'
    var_39 = var_37.get_signature(var_38)
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = 'other-salt'
    var_42 = module_0.Signer(var_0, var_41)
    var_43 = var_42.get_signature(var_38)
    var_44 = var_37.verify_signature(var_38, var_43)
    assert var_44 is False
    var_45 = b'bytes-value'
    var_46 = var_1.get_signature(var_13)
    var_47 = var_1.verify_signature(var_13, var_46)
    assert var_47 is True
    var_48 = -1
    var_49 = var_3[:var_48]
    var_50 = b'x'
    var_51 = var_49 + var_50
    var_52 = var_51 if var_3 else var_50
    var_53 = var_1.verify_signature(var_2, var_52)
    assert var_53 is False



# Parsed testcases at query #74
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
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'invalid-base64!!'
    var_13 = var_2.verify_signature(var_3, var_12)
    assert var_13 is False
    var_14 = 'different-secret'
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = var_15.verify_signature(var_3, var_4)
    assert var_16 is False
    var_17 = 'different-salt'
    var_18 = module_0.Signer(var_0, var_17)
    var_19 = var_18.verify_signature(var_3, var_4)
    assert var_19 is False
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22, var_1)
    var_24 = var_23.get_signature(var_3)
    var_25 = var_23.verify_signature(var_3, var_24)
    assert var_25 is True
    var_26 = 'string-value'
    var_27 = var_2.get_signature(var_26)
    var_28 = var_2.verify_signature(var_26, var_27)
    assert var_28 is True



# Parsed testcases at query #75
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
    var_7 = b'other-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer()
    var_13 = b'test-value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'secret'
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_16, algorithm=var_17)
    var_19 = b'test-value'
    var_20 = var_18.get_signature(var_19)
    assert var_20 == b''
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = b'!!!invalid-base64!!!'
    var_23 = var_1.verify_signature(var_19, var_22)
    assert var_23 is False
    var_24 = 'test-value'
    var_25 = var_1.verify_signature(var_24, var_20)
    assert var_25 is True
    var_26 = 'invalid-sig'
    var_27 = var_1.verify_signature(var_24, var_26)
    assert var_27 is False
    var_28 = b''
    var_29 = var_1.get_signature(var_28)
    var_30 = var_1.verify_signature(var_28, var_29)
    assert var_30 is True



# Parsed testcases at query #76
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that verify_signature correctly validates and invalidates signatures.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'different-value'
    var_11 = var_2.verify_signature(var_10, var_4)
    assert var_11 is False
    var_12 = 'test-string-value'
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'test-value-2'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = b'!!!not-valid-base64!!!'
    var_23 = var_2.verify_signature(var_3, var_22)
    assert var_23 is False
    var_24 = 'test-key'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = b'test-value-3'
    var_28 = var_26.get_signature(var_27)
    assert var_28 == b''
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = 'salt1'
    var_31 = module_0.Signer(var_24, var_30)
    var_32 = 'salt2'
    var_33 = module_0.Signer(var_24, var_32)
    var_34 = b'test-value-4'
    var_35 = var_31.get_signature(var_34)
    var_36 = var_33.get_signature(var_34)
    var_37 = var_31.verify_signature(var_34, var_35)
    assert var_37 is True
    var_38 = var_31.verify_signature(var_34, var_36)
    assert var_38 is False
    var_39 = var_33.verify_signature(var_34, var_36)
    assert var_39 is True
    var_40 = var_33.verify_signature(var_34, var_35)
    assert var_40 is False



# Parsed testcases at query #77
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test Signer.verify_signature method'
    var_1 = 'test-secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'other-value'
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is False
    var_11 = 'test-value'
    var_12 = var_2.verify_signature(var_11, var_4)
    assert var_12 is True
    var_13 = module_1.base64_encode(var_4)
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is True
    var_15 = b''
    var_16 = var_2.get_signature(var_15)
    var_17 = var_2.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = b'!!!invalid-base64!!!'
    var_19 = var_2.verify_signature(var_3, var_18)
    assert var_19 is False
    var_20 = b''
    var_21 = var_2.verify_signature(var_3, var_20)
    assert var_21 is False
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24)
    var_26 = b'rotated-value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = module_0.Signer(var_22)
    var_30 = var_29.get_signature(var_26)
    var_31 = var_25.verify_signature(var_26, var_30)
    assert var_31 is True
    var_32 = 'test-key'
    var_33 = module_0.NoneAlgorithm()
    var_34 = module_0.Signer(var_32, algorithm=var_33)
    var_35 = b'none-algo-value'
    var_36 = var_34.get_signature(var_35)
    var_37 = module_1.base64_encode(var_20)
    var_38 = var_34.verify_signature(var_35, var_36)
    assert var_38 is True
    var_39 = var_34.verify_signature(var_35, var_20)
    assert var_39 is True
    var_40 = b'sha256-value'
    var_41 = 'concat'
    var_42 = module_0.Signer(var_32, key_derivation=var_41)
    var_43 = 'hmac'
    var_44 = module_0.Signer(var_32, key_derivation=var_43)
    var_45 = 'none'
    var_46 = module_0.Signer(var_32, key_derivation=var_45)
    var_47 = b'derivation-test'
    var_48 = var_42.get_signature(var_47)
    var_49 = var_44.get_signature(var_47)
    var_50 = var_46.get_signature(var_47)
    var_51 = var_42.verify_signature(var_47, var_48)
    assert var_51 is True
    var_52 = var_44.verify_signature(var_47, var_49)
    assert var_52 is True
    var_53 = var_46.verify_signature(var_47, var_50)
    assert var_53 is True
    var_54 = var_42.verify_signature(var_47, var_49)
    assert var_54 is False
    var_55 = var_42.verify_signature(var_47, var_50)
    assert var_55 is False



# Parsed testcases at query #78
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'different-secret-key'
    var_13 = module_0.Signer(var_12)
    var_14 = var_13.verify_signature(var_3, var_4)
    assert var_14 is False
    var_15 = b'invalid-base64!!!'
    var_16 = var_2.verify_signature(var_3, var_15)
    assert var_16 is False
    var_17 = 'test-value'
    var_18 = var_2.verify_signature(var_17, var_4)
    assert var_18 is True
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21)
    var_23 = b'rotation-test'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = module_0.Signer(var_19)
    var_27 = var_26.get_signature(var_23)
    var_28 = var_22.verify_signature(var_23, var_27)
    assert var_28 is True
    var_29 = 'wrong-key'
    var_30 = module_0.Signer(var_29)
    var_31 = var_30.get_signature(var_23)
    var_32 = var_22.verify_signature(var_23, var_31)
    assert var_32 is False
    var_33 = module_0.NoneAlgorithm()
    var_34 = 'custom-key'
    var_35 = module_0.Signer(var_34, algorithm=var_33)
    var_36 = b'custom-value'
    var_37 = b''
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = b'anything'
    var_40 = var_35.verify_signature(var_36, var_39)
    assert var_40 is False



# Parsed testcases at query #79
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is False
    var_11 = 'other-key'
    var_12 = module_0.Signer(var_11, var_1)
    var_13 = var_12.get_signature(var_3)
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17, var_1)
    var_19 = module_0.Signer(var_15, var_1)
    var_20 = var_19.get_signature(var_3)
    var_21 = var_18.verify_signature(var_3, var_20)
    assert var_21 is True
    var_22 = b'!!!'
    var_23 = var_2.verify_signature(var_3, var_22)
    assert var_23 is False
    var_24 = 'test value'
    var_25 = var_2.verify_signature(var_24, var_4)
    assert var_25 is True
    var_26 = 'ascii'
    var_27 = var_2.verify_signature(var_3, var_9)
    assert var_27 is False
    var_28 = 'secret'
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_28, algorithm=var_29)
    var_31 = var_30.get_signature(var_3)
    var_32 = var_30.verify_signature(var_3, var_31)
    assert var_32 is True



# Parsed testcases at query #80
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
    var_5 = 'test-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = 'ascii'
    var_8 = b'wrong-signature'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = b''
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = b'!!!invalid-base64!!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'test-value-2'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'key'
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_22, algorithm=var_23)
    var_25 = b'test-value-3'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'test-value-4'
    var_29 = 'key'
    var_30 = b'test-value-5'
    var_31 = b'different-value'
    var_32 = var_1.verify_signature(var_31, var_3)
    assert var_32 is False
    var_33 = b'custom-salt'
    var_34 = module_0.Signer(var_22, var_33)
    var_35 = b'test-value-6'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = b'|'
    var_39 = module_0.Signer(var_22, sep=var_38)
    var_40 = b'test-value-7'
    var_41 = var_39.get_signature(var_40)
    var_42 = var_39.verify_signature(var_40, var_41)
    assert var_42 is True



# Parsed testcases at query #81
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret-key-12345'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value-to-sign'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'\x00'
    var_6 = 20
    var_7 = var_5 * var_6
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'different-value'
    var_11 = var_1.verify_signature(var_10, var_3)
    assert var_11 is False
    var_12 = b'!!!invalid-base64!!!'
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = b''
    var_15 = var_1.verify_signature(var_2, var_14)
    assert var_15 is False
    var_16 = b'test-value-to-sign'
    var_17 = var_1.verify_signature(var_16, var_3)
    assert var_17 is True
    var_18 = 'test-value-to-sign'
    var_19 = var_1.verify_signature(var_18, var_3)
    assert var_19 is True
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = b'test-salt'
    var_24 = module_0.Signer(var_22, var_23)
    var_25 = b'rotation-test-value'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'test-key'
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_28, algorithm=var_29)
    var_31 = b'test'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True



# Parsed testcases at query #82
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'different-secret-key'
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.verify_signature(var_2, var_3)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = var_17.get_signature(var_2)
    var_19 = var_17.verify_signature(var_2, var_18)
    assert var_19 is True
    var_20 = 'test-value'
    var_21 = var_1.get_signature(var_20)
    var_22 = var_1.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'not-valid-base64!!'
    var_24 = var_1.verify_signature(var_2, var_23)
    assert var_24 is False
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_0, algorithm=var_25)
    var_27 = b'test-value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'salt1'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = b'salt2'
    var_33 = module_0.Signer(var_0, var_32)
    var_34 = var_31.get_signature(var_27)
    var_35 = var_33.verify_signature(var_27, var_34)
    assert var_35 is False



# Parsed testcases at query #83
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = b'test-secret-key-12345'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b'different-value'
    var_9 = var_1.verify_signature(var_8, var_3)
    assert var_9 is False
    var_10 = '!!!invalid-base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = ''
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'concat'
    var_15 = module_0.Signer(var_0, key_derivation=var_14)
    var_16 = b'another-test'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = b'wrong-signature'
    var_20 = var_15.verify_signature(var_16, var_19)
    assert var_20 is False
    var_21 = b'old-key'
    var_22 = b'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23)
    var_25 = b'rotation-test'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_0, algorithm=var_28)
    var_30 = b'none-algorithm-test'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True



# Parsed testcases at query #84
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = b'!!!invalid base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'string value'
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'old_key'
    var_16 = 'new_key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'test with rotation'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'different_salt'
    var_23 = module_0.Signer(var_0, var_22)
    var_24 = b'test value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = var_1.verify_signature(var_24, var_25)
    assert var_27 is False
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_0, algorithm=var_28)
    var_30 = b'test'
    var_31 = var_29.get_signature(var_30)
    assert var_31 == b''
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = 'secret-key'
    var_34 = b'test value'
    var_35 = [var_15, var_16]
    var_36 = module_0.Signer(var_35)
    var_37 = b'test'
    var_38 = var_36.get_signature(var_37)
    var_39 = [var_16]
    var_40 = module_0.Signer(var_39)
    var_41 = var_40.verify_signature(var_37, var_38)
    assert var_41 is False
    var_42 = b'|'
    var_43 = module_0.Signer(var_33, sep=var_42)
    var_44 = b'test value'
    var_45 = var_43.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is True



# Parsed testcases at query #85
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'string value'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = '!!!invalid base64!!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = module_0.NoneAlgorithm()
    var_16 = module_0.Signer(var_0, algorithm=var_15)
    var_17 = var_16.get_signature(var_2)
    var_18 = var_16.verify_signature(var_2, var_17)
    assert var_18 is True
    var_19 = 'different-salt'
    var_20 = module_0.Signer(var_0, var_19)
    var_21 = var_20.get_signature(var_2)
    var_22 = var_20.verify_signature(var_2, var_21)
    assert var_22 is True
    var_23 = var_1.verify_signature(var_2, var_21)
    assert var_23 is False
    var_24 = 'old-key'
    var_25 = 'new-key'
    var_26 = [var_24, var_25]
    var_27 = module_0.Signer(var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_27.verify_signature(var_2, var_28)
    assert var_29 is True
    var_30 = module_0.Signer(var_24)
    var_31 = var_30.verify_signature(var_2, var_28)
    assert var_31 is True



# Parsed testcases at query #86
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
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'different-key'
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = 'key'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = var_23.get_signature(var_2)
    var_25 = var_23.verify_signature(var_2, var_24)
    assert var_25 is True
    var_26 = b''
    var_27 = var_23.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = b'not-base64!!'
    var_29 = var_1.verify_signature(var_2, var_28)
    assert var_29 is False
    var_30 = 'string-value'
    var_31 = var_1.get_signature(var_30)
    var_32 = var_1.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = b'bytes-value'
    var_34 = var_1.get_signature(var_33)
    var_35 = b'original'
    var_36 = b'tampered'
    var_37 = var_1.get_signature(var_35)
    var_38 = var_1.verify_signature(var_36, var_37)
    assert var_38 is False
    var_39 = 'custom-salt'
    var_40 = module_0.Signer(var_0, var_39)
    var_41 = var_40.get_signature(var_2)
    var_42 = var_1.verify_signature(var_2, var_41)
    assert var_42 is False
    var_43 = var_40.verify_signature(var_2, var_41)
    assert var_43 is True
    var_44 = module_0.Signer(var_21)
    var_45 = var_44.verify_signature(var_2, var_26)
    assert var_45 is False



# Parsed testcases at query #87
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret-key'
    assert var_1 is True
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is True
    var_11 = b'.'
    var_12 = module_0.Signer(var_1, sep=var_11)
    var_13 = b'test-value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'custom-salt'
    var_17 = module_0.Signer(var_1, var_16)
    var_18 = b'test-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = var_2.verify_signature(var_18, var_19)
    assert var_21 is False
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24)
    var_26 = b'test-value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_1, algorithm=var_29)
    var_31 = b'test-value'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b'test-value'
    var_35 = 'not-base64!!'
    var_36 = var_2.verify_signature(var_3, var_35)
    assert var_36 is False
    var_37 = b''
    var_38 = var_2.get_signature(var_37)
    var_39 = var_2.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = b'test with spaces and !@#$%^&*()'
    var_41 = var_2.get_signature(var_40)
    var_42 = var_2.verify_signature(var_40, var_41)
    assert var_42 is True
    var_43 = 'secret-key'
    var_44 = b'test-value'
    var_45 = module_1.base64_decode(var_4)
    var_46 = var_2.verify_signature(var_3, var_45)
    assert var_46 is True
    var_47 = 'key1'
    var_48 = module_0.Signer(var_47)
    var_49 = 'key2'
    var_50 = module_0.Signer(var_49)
    var_51 = b'test-value'
    var_52 = var_48.get_signature(var_51)
    var_53 = var_48.verify_signature(var_51, var_52)
    assert var_53 is True
    var_54 = var_50.verify_signature(var_51, var_52)
    assert var_54 is False



# Parsed testcases at query #88
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = b'!!!invalid-base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'different-secret-key'
    var_13 = module_0.Signer(var_12)
    var_14 = var_13.verify_signature(var_2, var_3)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'rotation-test'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'custom-salt'
    var_23 = module_0.Signer(var_0, var_22)
    var_24 = var_23.verify_signature(var_2, var_3)
    assert var_24 is False
    var_25 = 'test-value'
    var_26 = var_1.verify_signature(var_25, var_3)
    assert var_26 is True
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, algorithm=var_27)
    var_29 = var_28.get_signature(var_2)
    var_30 = var_28.verify_signature(var_2, var_29)
    assert var_30 is True
    var_31 = 'secret-key'
    var_32 = b'test-value'
    var_33 = var_1.verify_signature(var_32, var_3)
    assert var_33 is True
    var_34 = var_1.verify_signature(var_25, var_3)
    assert var_34 is True



# Parsed testcases at query #89
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
    var_7 = 'test-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is True
    var_9 = 'different-key'
    var_10 = module_0.Signer(var_9)
    var_11 = var_10.get_signature(var_2)
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = module_0.Signer(var_13)
    var_18 = var_17.get_signature(var_2)
    var_19 = var_16.verify_signature(var_2, var_18)
    assert var_19 is True
    var_20 = b'!!!invalid-base64!!!'
    var_21 = var_1.verify_signature(var_2, var_20)
    assert var_21 is False
    var_22 = b''
    var_23 = var_1.get_signature(var_22)
    var_24 = var_1.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = 'key'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_27.verify_signature(var_2, var_28)
    assert var_29 is True
    var_30 = b'wrong'
    var_31 = 'concat'
    var_32 = module_0.Signer(var_25, key_derivation=var_31)
    var_33 = var_32.get_signature(var_2)
    var_34 = var_32.verify_signature(var_2, var_33)
    assert var_34 is True
    var_35 = 'hmac'
    var_36 = module_0.Signer(var_25, key_derivation=var_35)
    var_37 = var_36.get_signature(var_2)
    var_38 = var_36.verify_signature(var_2, var_37)
    assert var_38 is True



# Parsed testcases at query #90
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
    var_7 = b'!!!invalid-base64!!!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = var_14.get_signature(var_2)
    var_16 = var_14.verify_signature(var_2, var_15)
    assert var_16 is True
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_0, algorithm=var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = 'concat'
    var_22 = module_0.Signer(var_0, key_derivation=var_21)
    var_23 = var_22.get_signature(var_2)
    var_24 = var_22.verify_signature(var_2, var_23)
    assert var_24 is True
    var_25 = 'hmac'
    var_26 = module_0.Signer(var_0, key_derivation=var_25)
    var_27 = var_26.get_signature(var_2)
    var_28 = var_26.verify_signature(var_2, var_27)
    assert var_28 is True
    var_29 = 'none'
    var_30 = module_0.Signer(var_0, key_derivation=var_29)
    var_31 = var_30.get_signature(var_2)
    var_32 = var_30.verify_signature(var_2, var_31)
    assert var_32 is True



# Parsed testcases at query #91
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = module_0.NoneAlgorithm()
    var_10 = module_0.Signer(var_0, algorithm=var_9)
    var_11 = var_10.get_signature(var_2)
    var_12 = var_10.verify_signature(var_2, var_11)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = module_0.Signer(var_13)
    var_18 = var_17.get_signature(var_2)
    var_19 = module_0.Signer(var_14)
    var_20 = var_19.get_signature(var_2)
    var_21 = var_16.verify_signature(var_2, var_18)
    assert var_21 is True
    var_22 = var_16.verify_signature(var_2, var_20)
    assert var_22 is True
    var_23 = 'test-string'
    var_24 = var_1.get_signature(var_23)
    var_25 = var_1.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'!!!invalid-base64!!!'
    var_27 = var_1.verify_signature(var_2, var_26)
    assert var_27 is False
    var_28 = 'concat'
    var_29 = module_0.Signer(var_0, key_derivation=var_28)
    var_30 = var_29.get_signature(var_2)
    var_31 = var_29.verify_signature(var_2, var_30)
    assert var_31 is True
    var_32 = 'hmac'
    var_33 = module_0.Signer(var_0, key_derivation=var_32)
    var_34 = var_33.get_signature(var_2)
    var_35 = var_33.verify_signature(var_2, var_34)
    assert var_35 is True
    var_36 = b'custom-salt'
    var_37 = module_0.Signer(var_0, var_36)
    var_38 = var_37.get_signature(var_2)
    var_39 = var_37.verify_signature(var_2, var_38)
    assert var_39 is True



# Parsed testcases at query #92
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = 'test-string-value'
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b''
    var_12 = var_2.get_signature(var_11)
    var_13 = var_2.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'concat'
    var_15 = module_0.Signer(var_0, var_1, key_derivation=var_14)
    var_16 = b'test-value-concat'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'hmac'
    var_20 = module_0.Signer(var_0, var_1, key_derivation=var_19)
    var_21 = b'test-value-hmac'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = 'none'
    var_25 = module_0.Signer(var_0, var_1, key_derivation=var_24)
    var_26 = b'test-value-none'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'old-key'
    var_30 = 'new-key'
    var_31 = [var_29, var_30]
    var_32 = module_0.Signer(var_31, var_1)
    var_33 = b'test-value-rotation'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = module_0.NoneAlgorithm()
    var_37 = module_0.Signer(var_0, var_1, algorithm=var_36)
    var_38 = b'test-value-none-alg'
    var_39 = var_37.get_signature(var_38)
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = b'!!!invalid-base64!!!'
    var_42 = var_2.verify_signature(var_3, var_41)
    assert var_42 is False
    var_43 = b'wrong-value'
    var_44 = var_2.verify_signature(var_43, var_4)
    assert var_44 is False
    var_45 = 'different-salt'
    var_46 = module_0.Signer(var_0, var_45)
    var_47 = var_46.get_signature(var_3)
    var_48 = var_2.verify_signature(var_3, var_47)
    assert var_48 is False



# Parsed testcases at query #93
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = module_0.NoneAlgorithm()
    var_10 = module_0.Signer(var_0, algorithm=var_9)
    var_11 = b'test value'
    var_12 = var_10.get_signature(var_11)
    var_13 = var_10.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = b'modified value'
    var_15 = var_1.verify_signature(var_14, var_12)
    assert var_15 is False
    var_16 = 'different-secret'
    var_17 = module_0.Signer(var_16)
    var_18 = var_17.get_signature(var_11)
    var_19 = var_1.verify_signature(var_11, var_18)
    assert var_19 is False
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = module_0.Signer(var_20)
    var_25 = b'test'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_23.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = var_23.verify_signature(var_25, var_5)
    assert var_28 is False
    var_29 = b'test'
    var_30 = var_1.verify_signature(var_29, var_12)
    assert var_30 is True
    var_31 = 'test'
    var_32 = b'!!!invalid_base64!!!'
    var_33 = var_1.verify_signature(var_25, var_32)
    assert var_33 is False



# Parsed testcases at query #94
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'test-bytes'
    var_7 = var_2.get_signature(var_6)
    var_8 = var_2.verify_signature(var_6, var_7)
    assert var_8 is True
    var_9 = 'test-value'
    var_10 = b'invalid-sig'
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is False
    var_12 = 'original-value'
    var_13 = var_2.get_signature(var_12)
    var_14 = 'tampered-value'
    var_15 = var_2.verify_signature(var_14, var_13)
    assert var_15 is False
    var_16 = ''
    var_17 = var_2.get_signature(var_16)
    var_18 = var_2.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_0, algorithm=var_19)
    var_21 = 'test-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'!!!invalid-base64!!!'
    var_25 = var_2.verify_signature(var_9, var_24)
    assert var_25 is False
    var_26 = 'old-key'
    var_27 = 'new-key'
    var_28 = [var_26, var_27]
    var_29 = module_0.Signer(var_28, var_1)
    var_30 = 'test-value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = module_0.Signer(var_26, var_1)
    var_34 = var_33.verify_signature(var_30, var_31)
    assert var_34 is True
    var_35 = 'wrong-key'
    var_36 = module_0.Signer(var_35, var_1)
    var_37 = var_36.verify_signature(var_30, var_31)
    assert var_37 is False
    var_38 = b'|'
    var_39 = module_0.Signer(var_0, var_1, var_38)
    var_40 = 'test-value'
    var_41 = var_39.get_signature(var_40)
    var_42 = var_39.verify_signature(var_40, var_41)
    assert var_42 is True



# Parsed testcases at query #95
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'invalid-sig'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = module_0.NoneAlgorithm()
    var_12 = module_0.Signer(var_0, algorithm=var_11)
    var_13 = b'test-value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'test-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'salt1'
    var_24 = module_0.Signer(var_0, var_23)
    var_25 = 'salt2'
    var_26 = module_0.Signer(var_0, var_25)
    var_27 = b'test-value'
    var_28 = var_24.get_signature(var_27)
    var_29 = var_26.get_signature(var_27)
    var_30 = var_24.verify_signature(var_27, var_29)
    assert var_30 is False
    var_31 = var_26.verify_signature(var_27, var_28)
    assert var_31 is False
    var_32 = 'concat'
    var_33 = module_0.Signer(var_0, key_derivation=var_32)
    var_34 = b'test-value'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = 'hmac'
    var_38 = module_0.Signer(var_0, key_derivation=var_37)
    var_39 = b'test-value'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = 'none'
    var_43 = module_0.Signer(var_0, key_derivation=var_42)
    var_44 = b'test-value'
    var_45 = var_43.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = b'!!!invalid-base64!!!'
    var_48 = var_1.verify_signature(var_44, var_47)
    assert var_48 is False



# Parsed testcases at query #96
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'string-value'
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'not-base64!!!'
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = b''
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = b'modified-value'
    var_16 = var_1.verify_signature(var_15, var_3)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = b'test-salt'
    var_21 = module_0.Signer(var_19, var_20)
    var_22 = b'rotation-test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = module_0.NoneAlgorithm()
    var_26 = 'test-key'
    var_27 = module_0.Signer(var_26, algorithm=var_25)
    var_28 = b'none-algo-test'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True



# Parsed testcases at query #97
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = module_0.NoneAlgorithm()
    var_10 = module_0.Signer(var_0, algorithm=var_9)
    var_11 = b'test value'
    var_12 = var_10.get_signature(var_11)
    var_13 = var_10.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'secret-key'
    var_22 = b'test value'
    var_23 = var_1.get_signature(var_22)
    var_24 = var_1.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = module_0.Signer(var_21)
    var_26 = b'original value'
    var_27 = var_25.get_signature(var_26)
    var_28 = b'tampered value'
    var_29 = var_25.verify_signature(var_28, var_27)
    assert var_29 is False
    var_30 = 'test value'
    var_31 = var_25.verify_signature(var_30, var_27)
    assert var_31 is False
    var_32 = 'original value'
    var_33 = var_25.verify_signature(var_32, var_27)
    assert var_33 is True
    var_34 = b'!!!invalid-base64!!!'
    var_35 = var_25.verify_signature(var_26, var_34)
    assert var_35 is False
    var_36 = b'test value'



# Parsed testcases at query #98
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = b'test-secret-key-12345'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'modified-value'
    var_13 = var_2.verify_signature(var_12, var_4)
    assert var_13 is False
    var_14 = 'string-value'
    var_15 = var_2.get_signature(var_14)
    var_16 = var_2.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = b'not-valid-base64!!!'
    var_18 = var_2.verify_signature(var_3, var_17)
    assert var_18 is False
    var_19 = b''
    var_20 = var_2.verify_signature(var_3, var_19)
    assert var_20 is False
    var_21 = b'old-secret-key'
    var_22 = b'new-secret-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23)
    var_25 = module_0.Signer(var_21)
    var_26 = var_25.get_signature(var_3)
    var_27 = var_24.verify_signature(var_3, var_26)
    assert var_27 is True
    var_28 = var_24.get_signature(var_3)
    var_29 = var_24.verify_signature(var_3, var_28)
    assert var_29 is True
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_1, algorithm=var_30)
    var_32 = var_31.get_signature(var_3)
    var_33 = var_31.verify_signature(var_3, var_32)
    assert var_33 is True
    var_34 = b'different-salt'
    var_35 = module_0.Signer(var_1, var_34)
    var_36 = var_35.get_signature(var_3)
    var_37 = var_35.verify_signature(var_3, var_36)
    assert var_37 is True
    var_38 = var_2.verify_signature(var_3, var_36)
    assert var_38 is False
    var_39 = b'|'
    var_40 = module_0.Signer(var_1, sep=var_39)
    var_41 = b'test-with-custom-sep'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = b'test\nwith\tspecial\x00chars'
    var_45 = var_2.get_signature(var_44)
    var_46 = var_2.verify_signature(var_44, var_45)
    assert var_46 is True



# Parsed testcases at query #99
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'different value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = module_1.base64_encode(var_3)
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is True
    var_14 = b'!!!invalid-base64!!!'
    var_15 = var_1.verify_signature(var_2, var_14)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'rotation test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'string value'
    var_24 = var_1.get_signature(var_23)
    var_25 = var_1.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = 'key'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = b'test'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b''
    var_33 = var_28.verify_signature(var_29, var_32)
    assert var_33 is True



# Parsed testcases at query #100
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test-value'
    var_6 = var_1.get_signature(var_5)
    var_7 = var_1.verify_signature(var_5, var_6)
    assert var_7 is True
    var_8 = b'invalid-sig'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b'!!!invalid-base64!!!'
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = module_0.NoneAlgorithm()
    var_15 = module_0.Signer(var_0, algorithm=var_14)
    var_16 = b'test-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = var_15.verify_signature(var_16, var_10)
    assert var_19 is True
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = b'test-value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = 'secret-key'
    var_28 = b'test-value'
    var_29 = b'|'
    var_30 = module_0.Signer(var_27, sep=var_29)
    var_31 = b'test-value'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b'custom-salt'
    var_35 = module_0.Signer(var_27, var_34)
    var_36 = b'test-value'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = 'key1'
    var_40 = module_0.Signer(var_39)
    var_41 = 'key2'
    var_42 = module_0.Signer(var_41)
    var_43 = b'test-value'
    var_44 = var_40.get_signature(var_43)
    var_45 = var_42.verify_signature(var_43, var_44)
    assert var_45 is False
    var_46 = b'bytes-key'
    var_47 = module_0.Signer(var_46)
    var_48 = b'test-value'
    var_49 = var_47.get_signature(var_48)
    var_50 = var_47.verify_signature(var_48, var_49)
    assert var_50 is True
    var_51 = module_0.HMACAlgorithm()
    var_52 = module_0.Signer(var_27, algorithm=var_51)
    var_53 = b'test-value'
    var_54 = var_52.get_signature(var_53)
    var_55 = var_52.verify_signature(var_53, var_54)
    assert var_55 is True



# Parsed testcases at query #101
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
    var_8 = 'old-key'
    var_9 = 'new-key'
    var_10 = [var_8, var_9]
    var_11 = module_0.Signer(var_10, var_1)
    var_12 = b'test-value-2'
    var_13 = var_11.get_signature(var_12)
    var_14 = var_11.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'test-value-2'
    var_16 = var_11.verify_signature(var_15, var_13)
    assert var_16 is True
    var_17 = 'secret'
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_17, algorithm=var_18)
    var_20 = b'test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'test'
    var_24 = b'!!!invalid-base64!!!'
    var_25 = var_2.verify_signature(var_23, var_24)
    assert var_25 is False
    var_26 = b''
    var_27 = var_2.get_signature(var_26)
    var_28 = var_2.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = var_2.verify_signature(var_23, var_4)
    assert var_29 is True
    var_30 = 'test'
    var_31 = var_2.verify_signature(var_30, var_4)
    assert var_31 is True



# Parsed testcases at query #102
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
    var_5 = 'test-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'invalid'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'not-valid-base64!!'
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.Signer(var_14)
    var_22 = var_21.get_signature(var_18)
    var_23 = var_17.verify_signature(var_18, var_22)
    assert var_23 is True
    var_24 = 'wrong-key'
    var_25 = module_0.Signer(var_24)
    var_26 = var_25.get_signature(var_18)
    var_27 = var_17.verify_signature(var_18, var_26)
    assert var_27 is False
    var_28 = 'salt1'
    var_29 = module_0.Signer(var_0, var_28)
    var_30 = 'salt2'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = b'test-value'
    var_33 = var_29.get_signature(var_32)
    var_34 = var_29.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = var_31.verify_signature(var_32, var_33)
    assert var_35 is False
    var_36 = module_0.NoneAlgorithm()
    var_37 = module_0.Signer(var_0, algorithm=var_36)
    var_38 = b'test-value'
    var_39 = var_37.get_signature(var_38)
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = var_37.verify_signature(var_38, var_10)
    assert var_41 is True
    var_42 = b'test-value'



# Parsed testcases at query #103
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'different-secret'
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = var_13.get_signature(var_3)
    var_15 = var_2.verify_signature(var_3, var_14)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_1)
    var_20 = var_19.get_signature(var_3)
    var_21 = var_19.verify_signature(var_3, var_20)
    assert var_21 is True
    var_22 = 'not-base64!!'
    var_23 = var_2.verify_signature(var_3, var_22)
    assert var_23 is False
    var_24 = b''
    var_25 = var_2.verify_signature(var_3, var_24)
    assert var_25 is False
    var_26 = 'key'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = var_28.get_signature(var_3)
    var_30 = var_28.verify_signature(var_3, var_29)
    assert var_30 is True
    var_31 = 'hello world'
    var_32 = var_2.get_signature(var_31)
    var_33 = var_2.verify_signature(var_31, var_32)
    assert var_33 is True



# Parsed testcases at query #104
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid-signature'
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_3.get_signature(var_9)
    var_11 = var_3.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'test-value'
    var_13 = var_3.verify_signature(var_12, var_5)
    assert var_13 is True
    var_14 = 'other-secret'
    var_15 = module_0.Signer(var_14, var_2)
    var_16 = var_15.verify_signature(var_4, var_5)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19, var_2)
    var_21 = b'old-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = 'secret'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = b'test'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b''
    var_31 = var_26.verify_signature(var_27, var_30)
    assert var_31 is True
    var_32 = b'!!!invalid-base64!!!'
    var_33 = var_3.verify_signature(var_4, var_32)
    assert var_33 is False
    var_34 = var_3.verify_signature(var_4, var_30)
    assert var_34 is False
    var_35 = 'different-salt'
    var_36 = module_0.Signer(var_1, var_35)
    var_37 = var_36.verify_signature(var_4, var_5)
    assert var_37 is False
    var_38 = 'test-string-value'



# Parsed testcases at query #105
#--------------------------




# Parsed testcases at query #106
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test-string'
    var_6 = var_1.get_signature(var_5)
    var_7 = var_1.verify_signature(var_5, var_6)
    assert var_7 is True
    var_8 = b'invalid-signature'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'different-salt'
    var_14 = module_0.Signer(var_0, var_13)
    var_15 = var_14.get_signature(var_2)
    var_16 = var_14.verify_signature(var_2, var_15)
    assert var_16 is True
    var_17 = var_1.verify_signature(var_2, var_15)
    assert var_17 is False
    var_18 = '!!!invalid-base64!!!'
    var_19 = var_1.verify_signature(var_2, var_18)
    assert var_19 is False
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = b'rotation-test'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = var_1.get_signature(var_2)
    var_28 = var_1.verify_signature(var_2, var_27)
    assert var_28 is True
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_0, algorithm=var_29)
    var_31 = b'none-algorithm'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b''
    var_35 = var_30.verify_signature(var_31, var_34)
    assert var_35 is True



# Parsed testcases at query #107
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'test-key'
    var_11 = module_0.NoneAlgorithm()
    var_12 = module_0.Signer(var_10, algorithm=var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_12.verify_signature(var_2, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'rotation test'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'key'
    var_23 = b'salt1'
    var_24 = module_0.Signer(var_22, var_23)
    var_25 = b'salt2'
    var_26 = module_0.Signer(var_22, var_25)
    var_27 = var_24.get_signature(var_19)
    var_28 = var_26.verify_signature(var_19, var_27)
    assert var_28 is False
    var_29 = 'string value'
    var_30 = var_1.get_signature(var_29)
    var_31 = var_1.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'custom digest'
    var_33 = b'!!!invalid base64!!!'
    var_34 = var_1.verify_signature(var_32, var_33)
    assert var_34 is False
    var_35 = var_1.get_signature(var_32)
    var_36 = var_1.verify_signature(var_32, var_35)
    assert var_36 is True
    var_37 = 'concat'
    var_38 = module_0.Signer(var_22, key_derivation=var_37)
    var_39 = b'concat test'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = 'hmac'
    var_43 = module_0.Signer(var_22, key_derivation=var_42)
    var_44 = var_43.get_signature(var_39)
    var_45 = var_43.verify_signature(var_39, var_44)
    assert var_45 is True
    var_46 = 'none'
    var_47 = module_0.Signer(var_22, key_derivation=var_46)
    var_48 = var_47.get_signature(var_39)
    var_49 = var_47.verify_signature(var_39, var_48)
    assert var_49 is True



# Parsed testcases at query #108
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'wrong value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'test value'
    var_19 = b'test value'
    var_20 = var_1.get_signature(var_19)
    var_21 = module_1.base64_encode(var_20)
    var_22 = var_1.verify_signature(var_18, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid base64!!!'
    var_24 = var_1.verify_signature(var_15, var_23)
    assert var_24 is False
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_0, algorithm=var_25)
    var_27 = b'test value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'test value'



# Parsed testcases at query #109
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b'corrupted-signature'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'string-value'
    var_14 = var_1.get_signature(var_13)
    var_15 = var_1.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'rotation-test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.Signer(var_16)
    var_24 = var_23.get_signature(var_20)
    var_25 = var_19.verify_signature(var_20, var_24)
    assert var_25 is True
    var_26 = 'custom-salt'
    var_27 = module_0.Signer(var_0, var_26)
    var_28 = b'salt-test'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b'|'
    var_32 = module_0.Signer(var_0, sep=var_31)
    var_33 = b'sep-test'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = module_0.NoneAlgorithm()
    var_37 = module_0.Signer(var_0, algorithm=var_36)
    var_38 = b'none-test'
    var_39 = var_37.get_signature(var_38)
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = b'md5-test'
    var_42 = 'key1'
    var_43 = module_0.Signer(var_42)
    var_44 = 'key2'
    var_45 = module_0.Signer(var_44)
    var_46 = b'different-keys'
    var_47 = var_43.get_signature(var_46)
    var_48 = var_45.verify_signature(var_46, var_47)
    assert var_48 is False
    var_49 = b'hello@world!test_123'
    var_50 = var_1.get_signature(var_49)
    var_51 = var_1.verify_signature(var_49, var_50)
    assert var_51 is True
    var_52 = b'x'
    var_53 = 10000
    var_54 = var_52 * var_53
    var_55 = var_1.get_signature(var_54)
    var_56 = var_1.verify_signature(var_54, var_55)
    assert var_56 is True
    var_57 = 256
    var_58 = range(var_57)
    var_59 = bytes(var_58)
    var_60 = var_1.get_signature(var_59)
    var_61 = var_1.verify_signature(var_59, var_60)
    assert var_61 is True
    var_62 = 'none'
    var_63 = module_0.Signer(var_0, key_derivation=var_62)
    var_64 = b'no-derivation'
    var_65 = var_63.get_signature(var_64)
    var_66 = var_63.verify_signature(var_64, var_65)
    assert var_66 is True
    var_67 = 'concat'
    var_68 = module_0.Signer(var_0, key_derivation=var_67)
    var_69 = b'concat-test'
    var_70 = var_68.get_signature(var_69)
    var_71 = var_68.verify_signature(var_69, var_70)
    assert var_71 is True
    var_72 = 'hmac'
    var_73 = module_0.Signer(var_0, key_derivation=var_72)
    var_74 = b'hmac-derivation'
    var_75 = var_73.get_signature(var_74)
    var_76 = var_73.verify_signature(var_74, var_75)
    assert var_76 is True
    var_77 = b''
    var_78 = var_1.verify_signature(var_2, var_77)
    assert var_78 is False
    var_79 = b'None'
    var_80 = var_1.verify_signature(var_2, var_79)
    assert var_80 is False



# Parsed testcases at query #110
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = b'test-secret-key-12345'
    var_2 = b'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid-signature'
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_3.get_signature(var_9)
    var_11 = var_3.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'different-key'
    var_13 = module_0.Signer(var_12, var_2)
    var_14 = var_13.get_signature(var_4)
    var_15 = var_3.verify_signature(var_4, var_14)
    assert var_15 is False
    var_16 = b'old-key'
    var_17 = b'newer-key'
    var_18 = b'newest-key'
    var_19 = [var_16, var_17, var_18]
    var_20 = module_0.Signer(var_19, var_2)
    var_21 = b'rotation-test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_1, var_2, algorithm=var_24)
    var_26 = b'none-algorithm-test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'string-value'
    var_30 = var_3.get_signature(var_29)
    var_31 = var_3.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'!!!invalid-base64!!!'
    var_33 = var_3.verify_signature(var_4, var_32)
    assert var_33 is False
    var_34 = b'a'
    var_35 = 10000
    var_36 = var_34 * var_35
    var_37 = var_3.get_signature(var_36)
    var_38 = var_3.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = 'concat'
    var_40 = module_0.Signer(var_1, var_2, key_derivation=var_39)
    var_41 = b'concat-test'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = 'hmac'
    var_45 = module_0.Signer(var_1, var_2, key_derivation=var_44)
    var_46 = b'hmac-test'
    var_47 = var_45.get_signature(var_46)
    var_48 = var_45.verify_signature(var_46, var_47)
    assert var_48 is True
    var_49 = 'none'
    var_50 = module_0.Signer(var_1, var_2, key_derivation=var_49)
    var_51 = b'none-derivation-test'
    var_52 = var_50.get_signature(var_51)
    var_53 = var_50.verify_signature(var_51, var_52)
    assert var_53 is True



# Parsed testcases at query #111
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'tampered-value'
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is False
    var_11 = b'!@#$%^'
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = b''
    var_14 = var_2.get_signature(var_13)
    var_15 = var_2.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_1)
    var_20 = b'test-rotation'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'test-key'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_23, algorithm=var_24)
    var_26 = b'test-none'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'|'
    var_30 = module_0.Signer(var_23, sep=var_29)
    var_31 = b'test-custom'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True



# Parsed testcases at query #112
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'test-string'
    var_12 = var_2.get_signature(var_11)
    var_13 = var_2.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16, var_1)
    var_18 = b'test-value-rotation'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'test-key'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = b'test-value-none'
    var_25 = var_23.get_signature(var_24)
    assert var_25 == b''
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'!!!invalid-base64!!!'
    var_28 = b'test'
    var_29 = var_2.verify_signature(var_28, var_27)
    assert var_29 is False
    var_30 = b''
    var_31 = var_2.verify_signature(var_28, var_30)
    assert var_31 is False
    var_32 = 'different-salt'
    var_33 = module_0.Signer(var_0, var_32)
    var_34 = b'test-value'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_2.verify_signature(var_34, var_35)
    assert var_36 is False



# Parsed testcases at query #113
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_7)
    assert var_8 is False
    var_9 = b'!!!invalid-base64!!!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'test-value'
    var_19 = var_14.verify_signature(var_18, var_16)
    assert var_19 is True
    var_20 = module_0.NoneAlgorithm()
    var_21 = module_0.Signer(var_0, algorithm=var_20)
    var_22 = var_21.get_signature(var_15)
    var_23 = var_21.verify_signature(var_15, var_22)
    assert var_23 is True
    var_24 = module_0.Signer(var_0)
    var_25 = var_24.get_signature(var_15)
    var_26 = var_21.verify_signature(var_15, var_25)
    assert var_26 is False



# Parsed testcases at query #114
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = 'string-value'
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = var_1.get_signature(var_2)
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is True
    var_12 = b'!!!invalid-base64!!!'
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = b''
    var_15 = var_1.get_signature(var_14)
    var_16 = var_1.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'old-key'
    var_18 = 'newer-key'
    var_19 = 'newest-key'
    var_20 = [var_17, var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = b'rotated-value'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = var_21.get_signature(var_22)
    var_26 = var_21.verify_signature(var_22, var_25)
    assert var_26 is True
    var_27 = 'key'
    var_28 = 'salt1'
    var_29 = module_0.Signer(var_27, var_28)
    var_30 = 'salt2'
    var_31 = module_0.Signer(var_27, var_30)
    var_32 = b'test'
    var_33 = var_29.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is False
    var_35 = b'-'
    var_36 = module_0.Signer(var_27, sep=var_35)
    var_37 = b'test-with-custom-sep'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True



# Parsed testcases at query #115
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
    var_8 = 'test-value'
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'test-value'
    var_12 = var_2.get_signature(var_11)
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15, var_1)
    var_17 = b'test-value'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = module_0.Signer(var_13, var_1)
    var_21 = var_20.verify_signature(var_17, var_18)
    assert var_21 is True
    var_22 = var_20.get_signature(var_17)
    var_23 = var_16.verify_signature(var_17, var_22)
    assert var_23 is True
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_0, algorithm=var_24)
    var_26 = b'test-value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'not-base64!!'
    var_30 = var_2.verify_signature(var_26, var_29)
    assert var_30 is False
    var_31 = b''
    var_32 = var_2.get_signature(var_31)
    var_33 = var_2.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = 'different-salt'
    var_35 = module_0.Signer(var_0, var_34)
    var_36 = var_35.verify_signature(var_26, var_27)
    assert var_36 is False
    var_37 = b':'
    var_38 = module_0.Signer(var_0, var_1, var_37)
    var_39 = var_38.verify_signature(var_26, var_27)
    assert var_39 is False
    var_40 = module_0.HMACAlgorithm()
    var_41 = module_0.Signer(var_0, algorithm=var_40)
    var_42 = var_41.get_signature(var_26)
    var_43 = var_41.verify_signature(var_26, var_42)
    assert var_43 is True



# Parsed testcases at query #116
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'invalid'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = b'wrong value'
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test'
    var_19 = module_0.Signer(var_14)
    var_20 = var_19.get_signature(var_18)
    var_21 = module_0.Signer(var_15)
    var_22 = var_21.get_signature(var_18)
    var_23 = var_17.verify_signature(var_18, var_20)
    assert var_23 is True
    var_24 = var_17.verify_signature(var_18, var_22)
    assert var_24 is True
    var_25 = 'wrong-key'
    var_26 = module_0.Signer(var_25)
    var_27 = var_26.get_signature(var_18)
    var_28 = var_17.verify_signature(var_18, var_27)
    assert var_28 is False
    var_29 = 'custom-salt'
    var_30 = module_0.Signer(var_0, var_29)
    var_31 = b'test'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = 'different-salt'
    var_35 = module_0.Signer(var_0, var_34)
    var_36 = var_35.verify_signature(var_31, var_32)
    assert var_36 is False
    var_37 = module_0.NoneAlgorithm()
    var_38 = module_0.Signer(var_0, algorithm=var_37)
    var_39 = b'test'
    var_40 = var_38.verify_signature(var_39, var_9)
    assert var_40 is True
    var_41 = b'something'
    var_42 = var_38.verify_signature(var_39, var_41)
    assert var_42 is False
    var_43 = b'test'
    var_44 = module_0.Signer(var_0)
    var_45 = b'test'
    var_46 = var_44.derive_key()
    var_47 = var_44.verify_signature(var_45, var_3)
    assert var_47 is True
    var_48 = b'!!!invalid-base64!!!'
    var_49 = var_44.verify_signature(var_45, var_48)
    assert var_49 is False
    var_50 = var_44.get_signature(var_9)
    var_51 = var_44.verify_signature(var_9, var_50)
    assert var_51 is True



# Parsed testcases at query #117
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
    var_7 = b'wrong-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'test-value'
    var_13 = var_1.verify_signature(var_12, var_3)
    assert var_13 is True
    var_14 = 'invalid-sig'
    var_15 = var_1.verify_signature(var_12, var_14)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'test-rotation'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, algorithm=var_23)
    var_25 = b'test-none'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = var_24.verify_signature(var_25, var_9)
    assert var_28 is True
    var_29 = b'anything'
    var_30 = var_24.verify_signature(var_25, var_29)
    assert var_30 is True
    var_31 = b'!!!invalid-base64!!!'
    var_32 = var_1.verify_signature(var_2, var_31)
    assert var_32 is False
    var_33 = var_1.verify_signature(var_2, var_9)
    assert var_33 is False
    var_34 = b'test-value'
    var_35 = var_1.verify_signature(var_34, var_3)
    assert var_35 is True



# Parsed testcases at query #118
#--------------------------




# Parsed testcases at query #119
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid-signature'
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_3.get_signature(var_9)
    var_11 = var_3.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b''
    var_13 = var_3.verify_signature(var_4, var_12)
    assert var_13 is False
    var_14 = module_0.NoneAlgorithm()
    var_15 = module_0.Signer(var_1, algorithm=var_14)
    var_16 = b'test'
    var_17 = var_15.get_signature(var_16)
    assert var_17 == b''
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'concat'
    var_20 = module_0.Signer(var_1, key_derivation=var_19)
    var_21 = b'test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = 'hmac'
    var_25 = module_0.Signer(var_1, key_derivation=var_24)
    var_26 = b'test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'none'
    var_30 = module_0.Signer(var_1, key_derivation=var_29)
    var_31 = b'test'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = 'old-key'
    var_35 = 'new-key'
    var_36 = [var_34, var_35]
    var_37 = 'rotation-salt'
    var_38 = module_0.Signer(var_36, var_37)
    var_39 = b'test'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = b'!!!invalid-base64!!!'
    var_43 = var_3.verify_signature(var_4, var_42)
    assert var_43 is False
    var_44 = 'test-value'
    var_45 = var_3.verify_signature(var_44, var_5)
    assert var_45 is True
    var_46 = 'invalid-sig'
    var_47 = var_3.verify_signature(var_44, var_46)
    assert var_47 is False
    var_48 = b'test'
    var_49 = b'|'
    var_50 = module_0.Signer(var_1, sep=var_49)
    var_51 = b'test'
    var_52 = var_50.get_signature(var_51)
    var_53 = var_50.verify_signature(var_51, var_52)
    assert var_53 is True
    var_54 = None
    var_55 = module_0.Signer(var_1, var_54)
    var_56 = b'test'
    var_57 = var_55.get_signature(var_56)
    var_58 = var_55.verify_signature(var_56, var_57)
    assert var_58 is True



# Parsed testcases at query #120
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = module_0.NoneAlgorithm()
    var_10 = module_0.Signer(var_0, algorithm=var_9)
    var_11 = b'test value'
    var_12 = var_10.get_signature(var_11)
    var_13 = var_10.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.Signer(var_0)
    var_22 = 'test string'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = 'different-salt'
    var_26 = module_0.Signer(var_0, var_25)
    var_27 = b'test value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b':'
    var_31 = module_0.Signer(var_0, sep=var_30)
    var_32 = b'test value'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = 'hmac'
    var_36 = module_0.Signer(var_0, key_derivation=var_35)
    var_37 = b'test value'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = 'concat'
    var_41 = module_0.Signer(var_0, key_derivation=var_40)
    var_42 = b'test value'
    var_43 = var_41.get_signature(var_42)
    var_44 = var_41.verify_signature(var_42, var_43)
    assert var_44 is True
    var_45 = 'none'
    var_46 = module_0.Signer(var_0, key_derivation=var_45)
    var_47 = b'test value'
    var_48 = var_46.get_signature(var_47)
    var_49 = var_46.verify_signature(var_47, var_48)
    assert var_49 is True



# Parsed testcases at query #121
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
    var_8 = b''
    var_9 = b'invalid'
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is False
    var_11 = b'!!!invalid-base64!!!'
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15, var_1)
    var_17 = b'test-value'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = module_0.NoneAlgorithm()
    var_21 = module_0.Signer(var_0, algorithm=var_20)
    var_22 = b'test-value'
    var_23 = b''
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'|'
    var_26 = module_0.Signer(var_0, sep=var_25)
    var_27 = b'test-value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = 'test-value'
    var_31 = var_2.verify_signature(var_30, var_28)
    assert var_31 is True
    var_32 = 'ascii'



# Parsed testcases at query #122
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = var_1.verify_signature(var_2, var_7)
    assert var_9 is False
    var_10 = module_0.NoneAlgorithm()
    var_11 = module_0.Signer(var_0, algorithm=var_10)
    var_12 = var_11.verify_signature(var_2, var_7)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'old-value'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = [var_13, var_14]
    var_21 = module_0.Signer(var_20)
    var_22 = b'new-value'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'salt1'
    var_26 = module_0.Signer(var_0, var_25)
    var_27 = b'salt2'
    var_28 = module_0.Signer(var_0, var_27)
    var_29 = b'test'
    var_30 = var_26.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is False
    var_32 = b'bytes-value'
    var_33 = var_1.verify_signature(var_32, var_3)
    assert var_33 is False
    var_34 = 'ascii'
    var_35 = 'string-value'
    var_36 = b'!!!invalid-base64!!!'
    var_37 = var_1.verify_signature(var_29, var_36)
    assert var_37 is False
    var_38 = 'hmac'
    var_39 = module_0.Signer(var_0, key_derivation=var_38)
    var_40 = var_39.get_signature(var_29)
    var_41 = var_39.verify_signature(var_29, var_40)
    assert var_41 is True



# Parsed testcases at query #123
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
    var_6 = b'bad-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'test-value'
    var_13 = var_2.verify_signature(var_12, var_4)
    assert var_13 is True
    var_14 = 'invalid-base64!!!'
    var_15 = var_2.verify_signature(var_3, var_14)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_1)
    var_20 = b'test'
    var_21 = var_19.sign(var_20)
    var_22 = var_19.sep
    var_23 = 1
    var_24 = 'secret'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = b'test'
    var_28 = var_26.get_signature(var_27)
    var_29 = b''
    var_30 = module_1.base64_encode(var_29)
    var_31 = var_26.verify_signature(var_27, var_28)
    assert var_31 is True



# Parsed testcases at query #124
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
    var_9 = module_0.NoneAlgorithm()
    var_10 = module_0.Signer(var_0, algorithm=var_9)
    var_11 = b'test-value'
    var_12 = var_10.get_signature(var_11)
    var_13 = var_10.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = b'test'
    var_22 = b'!!!invalid-base64!!!'
    var_23 = var_1.verify_signature(var_21, var_22)
    assert var_23 is False
    var_24 = 'test-value'
    var_25 = var_1.verify_signature(var_24, var_3)
    assert var_25 is True
    var_26 = 'invalid-sig'
    var_27 = var_1.verify_signature(var_24, var_26)
    assert var_27 is False
    var_28 = b''
    var_29 = var_1.verify_signature(var_28, var_28)
    assert var_29 is False
    var_30 = 'ascii'
    var_31 = 'hmac'
    var_32 = module_0.Signer(var_0, key_derivation=var_31)
    var_33 = var_32.get_signature(var_2)
    var_34 = var_32.verify_signature(var_2, var_33)
    assert var_34 is True
    var_35 = 'concat'
    var_36 = module_0.Signer(var_0, key_derivation=var_35)
    var_37 = var_36.get_signature(var_2)
    var_38 = var_36.verify_signature(var_2, var_37)
    assert var_38 is True



# Parsed testcases at query #125
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test value'
    var_6 = var_1.get_signature(var_5)
    var_7 = var_1.verify_signature(var_5, var_6)
    assert var_7 is True
    var_8 = b'invalid'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = b'raw_signature'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = module_0.NoneAlgorithm()
    var_16 = module_0.Signer(var_0, algorithm=var_15)
    var_17 = b'test value'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = b''
    var_21 = var_16.verify_signature(var_17, var_20)
    assert var_21 is True
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24)
    var_26 = b'test value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'|'
    var_30 = module_0.Signer(var_0, sep=var_29)
    var_31 = b'test|value'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b'salt1'
    var_35 = module_0.Signer(var_0, var_34)
    var_36 = b'salt2'
    var_37 = module_0.Signer(var_0, var_36)
    var_38 = b'test value'
    var_39 = var_35.get_signature(var_38)
    var_40 = var_35.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = var_37.verify_signature(var_38, var_39)
    assert var_41 is False
    var_42 = b'test value'
    var_43 = 'hmac'
    var_44 = module_0.Signer(var_0, key_derivation=var_43)
    var_45 = b'test value'
    var_46 = var_44.get_signature(var_45)
    var_47 = var_44.verify_signature(var_45, var_46)
    assert var_47 is True
    var_48 = 'concat'
    var_49 = module_0.Signer(var_0, key_derivation=var_48)
    var_50 = b'test value'
    var_51 = var_49.get_signature(var_50)
    var_52 = var_49.verify_signature(var_50, var_51)
    assert var_52 is True
    var_53 = 'none'
    var_54 = module_0.Signer(var_0, key_derivation=var_53)
    var_55 = b'test value'
    var_56 = var_54.get_signature(var_55)
    var_57 = var_54.verify_signature(var_55, var_56)
    assert var_57 is True
    var_58 = b'secret-key'
    var_59 = module_0.Signer(var_58)
    var_60 = b'test value'
    var_61 = var_59.get_signature(var_60)
    var_62 = var_59.verify_signature(var_60, var_61)
    assert var_62 is True
    var_63 = b'old-key'
    var_64 = b'new-key'
    var_65 = [var_63, var_64]
    var_66 = module_0.Signer(var_65)
    var_67 = b'test value'
    var_68 = var_66.get_signature(var_67)
    var_69 = var_66.verify_signature(var_67, var_68)
    assert var_69 is True
    var_70 = b'!!!invalid-base64!!!'
    var_71 = var_1.verify_signature(var_67, var_70)
    assert var_71 is False



# Parsed testcases at query #126
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'test_string'
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = b'test'
    var_15 = b'not-base64!!!'
    var_16 = var_1.verify_signature(var_14, var_15)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'rotation_test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = 'different-salt'
    var_25 = module_0.Signer(var_0, var_24)
    var_26 = var_25.get_signature(var_21)
    var_27 = var_25.verify_signature(var_21, var_26)
    assert var_27 is True
    var_28 = var_1.verify_signature(var_21, var_26)
    assert var_28 is False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that unsign correctly validates and returns the original value.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test-value'
    var_6 = b'test-value'
    var_7 = var_2.sign(var_6)
    var_8 = var_2.unsign(var_7)
    assert var_8 == b'test-value'
    var_9 = var_2.sign(var_3)
    var_10 = -1
    var_11 = var_9[:var_10]
    var_12 = b'X'
    var_13 = var_11 + var_12
    var_14 = var_2.unsign(var_13)
    var_15 = b'no-separator-here'
    var_16 = var_2.unsign(var_15)
    var_17 = b'|'
    var_18 = module_0.Signer(var_16, sep=var_17)
    var_19 = var_18.sign(var_3)
    var_20 = var_18.unsign(var_19)
    assert var_20 == b'test-value'
    var_21 = 'old-key'
    var_22 = 'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23)
    var_25 = var_24.sign(var_3)
    var_26 = var_24.unsign(var_25)
    assert var_26 == b'test-value'
    var_27 = module_0.Signer(var_16)
    var_28 = var_27.sign(var_3)
    var_29 = b'tampered'
    var_30 = 7
    var_31 = var_28[var_30:]
    var_32 = var_29 + var_31
    var_33 = var_27.unsign(var_32)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that unsign correctly validates and returns the original value.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test-value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that unsign raises BadSignature when no separator is found.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value-without-separator'
    var_4 = var_2.unsign(var_3)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that unsign raises BadSignature for invalid signature.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = b'different-value'
    var_6 = 12
    var_7 = var_4[var_6:]
    var_8 = var_5 + var_7
    var_9 = var_2.unsign(var_8)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that unsign works with key rotation (multiple secret keys).'
    var_1 = 'old-key'
    var_2 = 'new-key'
    var_3 = [var_1, var_2]
    var_4 = module_0.Signer(var_3)
    var_5 = 'test-value'
    var_6 = var_4.sign(var_5)
    var_7 = var_4.unsign(var_6)
    assert var_7 == b'test-value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that unsign can verify signatures created with older keys.'
    var_1 = 'old-key'
    var_2 = module_0.Signer(var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = 'new-key'
    var_6 = [var_1, var_5]
    var_7 = module_0.Signer(var_6)
    var_8 = var_7.unsign(var_4)
    assert var_8 == b'test-value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that unsign works with a custom separator.'
    var_1 = 'secret-key'
    var_2 = b'|'
    var_3 = module_0.Signer(var_1, sep=var_2)
    var_4 = 'test-value'
    var_5 = var_3.sign(var_4)
    var_6 = var_3.unsign(var_5)
    assert var_6 == b'test-value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that unsign works with a custom salt.'
    var_1 = 'secret-key'
    var_2 = 'custom-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = 'test-value'
    var_5 = var_3.sign(var_4)
    var_6 = var_3.unsign(var_5)
    assert var_6 == b'test-value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that unsign works with different algorithms.'
    var_1 = 'secret-key'
    var_2 = module_0.NoneAlgorithm()
    var_3 = module_0.Signer(var_1, algorithm=var_2)
    var_4 = 'test-value'
    var_5 = var_3.sign(var_4)
    var_6 = var_3.unsign(var_5)
    assert var_6 == b'test-value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that unsign works with bytes input.'
    var_1 = b'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = var_2.unsign(var_4)
    assert var_5 == b'test-value'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that BadSignature exception contains the payload.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = 'test-value'
    var_4 = var_2.sign(var_3)
    var_5 = b'different'
    var_6 = 9
    var_7 = var_4[var_6:]
    var_8 = var_5 + var_7
    var_9 = var_2.unsign(var_8)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'my-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = len(var_3)
    var_5 = 'other-key'
    var_6 = var_2.derive_key(var_5)
    var_7 = 'concat'
    var_8 = module_0.Signer(var_0, var_1, key_derivation=var_7)
    var_9 = var_8.derive_key()
    var_10 = 'hmac'
    var_11 = module_0.Signer(var_0, var_1, key_derivation=var_10)
    var_12 = var_11.derive_key()
    var_13 = 'none'
    var_14 = module_0.Signer(var_0, var_1, key_derivation=var_13)
    var_15 = var_14.derive_key()
    assert var_15 == b'secret-key'
    var_16 = 'invalid'
    var_17 = module_0.Signer(var_0, var_1, key_derivation=var_16)
    var_18 = var_17.derive_key()
    var_19 = 'key1'
    var_20 = 'salt'
    var_21 = module_0.Signer(var_19, var_20)
    var_22 = 'key2'
    var_23 = module_0.Signer(var_22, var_20)
    var_24 = var_21.derive_key()
    var_25 = var_23.derive_key()
    var_26 = 'key'
    var_27 = 'salt1'
    var_28 = module_0.Signer(var_26, var_27)
    var_29 = 'salt2'
    var_30 = module_0.Signer(var_26, var_29)
    var_31 = var_28.derive_key()
    var_32 = var_30.derive_key()



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
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = 'test-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is True
    var_9 = 'ascii'
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'rotation-test'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'secret'
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_17, algorithm=var_18)
    var_20 = b'test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid-base64!!!'
    var_24 = var_1.verify_signature(var_2, var_23)
    assert var_24 is False
    var_25 = module_0.Signer(var_17)
    var_26 = b''
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'salt-a'
    var_30 = module_0.Signer(var_17, var_29)
    var_31 = b'salt-b'
    var_32 = module_0.Signer(var_17, var_31)
    var_33 = b'cross-salt-test'
    var_34 = var_30.get_signature(var_33)
    var_35 = var_30.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = var_32.verify_signature(var_33, var_34)
    assert var_36 is False
    var_37 = module_0.HMACAlgorithm()
    var_38 = module_0.Signer(var_17, algorithm=var_37)
    var_39 = b'hmac-test'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.sign(var_2)
    var_4 = var_1.unsign(var_3)
    assert var_4 == b'test value'



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = '!!!invalid-base64!!!'
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15, var_1)
    var_17 = b'test value 2'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'wrong-key'
    var_21 = module_0.Signer(var_20, var_1)
    var_22 = var_21.verify_signature(var_17, var_18)
    assert var_22 is False
    var_23 = 'test value'
    var_24 = var_2.verify_signature(var_23, var_4)
    assert var_24 is True
    var_25 = b':'
    var_26 = module_0.Signer(var_0, sep=var_25)
    var_27 = b'another value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_0, algorithm=var_30)
    var_32 = b'value with no signature'
    var_33 = var_31.get_signature(var_32)
    assert var_33 == b''
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = b''
    var_36 = var_31.verify_signature(var_32, var_35)
    assert var_36 is True
    var_37 = module_0.NoneAlgorithm()
    var_38 = module_0.Signer(var_0, algorithm=var_37)
    var_39 = b'value'
    var_40 = b'some-sig'
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = b'x'
    var_43 = 1000
    var_44 = var_42 * var_43
    var_45 = var_2.get_signature(var_44)
    var_46 = var_2.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = 256
    var_48 = range(var_47)
    var_49 = bytes(var_48)
    var_50 = var_2.get_signature(var_49)
    var_51 = var_2.verify_signature(var_49, var_50)
    assert var_51 is True



# Parsed testcases at query #7
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'!!!invalid-base64!!!'
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = b''
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = module_0.NoneAlgorithm()
    var_16 = module_0.Signer(var_0, algorithm=var_15)
    var_17 = b'test-value'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = b'test-value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = 'test-string'
    var_28 = var_1.get_signature(var_27)
    var_29 = var_1.verify_signature(var_27, var_28)
    assert var_29 is True



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'different-key'
    var_11 = module_0.Signer(var_10)
    var_12 = var_11.get_signature(var_2)
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'test value'
    var_15 = '!!!'
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'rotate test'
    var_22 = module_0.Signer(var_17)
    var_23 = var_22.get_signature(var_21)
    var_24 = var_20.verify_signature(var_21, var_23)
    assert var_24 is True
    var_25 = 'key'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_27.verify_signature(var_2, var_28)
    assert var_29 is True



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test Signer.verify_signature method with various scenarios.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = 'test-value'
    var_9 = var_2.verify_signature(var_8, var_4)
    assert var_9 is True
    var_10 = 'utf-8'
    var_11 = b'modified-value'
    var_12 = var_2.verify_signature(var_11, var_4)
    assert var_12 is False
    var_13 = b'!!!invalid-base64!!!'
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = b''
    var_16 = var_2.get_signature(var_15)
    var_17 = var_2.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = b'|'
    var_19 = module_0.Signer(var_1, sep=var_18)
    var_20 = b'custom-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'old-key'
    var_24 = 'new-key'
    var_25 = [var_23, var_24]
    var_26 = module_0.Signer(var_25)
    var_27 = b'rotation-value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = module_0.Signer(var_23)
    var_31 = var_30.get_signature(var_27)
    var_32 = var_26.verify_signature(var_27, var_31)
    assert var_32 is True
    var_33 = module_0.NoneAlgorithm()
    var_34 = module_0.Signer(var_1, algorithm=var_33)
    var_35 = b'no-signature'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = b'custom-salt'
    var_39 = module_0.Signer(var_1, var_38)
    var_40 = b'custom-salt-value'
    var_41 = var_39.get_signature(var_40)
    var_42 = var_39.verify_signature(var_40, var_41)
    assert var_42 is True
    var_43 = module_0.Signer(var_1)
    var_44 = var_43.verify_signature(var_40, var_41)
    assert var_44 is False
    var_45 = 'concat'
    var_46 = module_0.Signer(var_1, key_derivation=var_45)
    var_47 = b'concat-value'
    var_48 = var_46.get_signature(var_47)
    var_49 = var_46.verify_signature(var_47, var_48)
    assert var_49 is True
    var_50 = 'hmac'
    var_51 = module_0.Signer(var_1, key_derivation=var_50)
    var_52 = b'hmac-value'
    var_53 = var_51.get_signature(var_52)
    var_54 = var_51.verify_signature(var_52, var_53)
    assert var_54 is True
    var_55 = 'none'
    var_56 = module_0.Signer(var_1, key_derivation=var_55)
    var_57 = b'none-derivation-value'
    var_58 = var_56.get_signature(var_57)
    var_59 = var_56.verify_signature(var_57, var_58)
    assert var_59 is True
    var_60 = b'bytes-secret-key'
    var_61 = module_0.Signer(var_60)
    var_62 = b'bytes-value'
    var_63 = var_61.get_signature(var_62)
    var_64 = var_61.verify_signature(var_62, var_63)
    assert var_64 is True
    var_65 = b'x'
    var_66 = 10000
    var_67 = var_65 * var_66
    var_68 = var_2.get_signature(var_67)
    var_69 = var_2.verify_signature(var_67, var_68)
    assert var_69 is True
    var_70 = b'unicode-value-\xe2\x9c\x93'
    var_71 = var_2.get_signature(var_70)
    var_72 = var_2.verify_signature(var_70, var_71)
    assert var_72 is True
    var_73 = b'YWJj'
    var_74 = var_2.verify_signature(var_3, var_73)
    assert var_74 is False



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'test string'
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = b'!!!invalid-base64!!!'
    var_16 = var_2.verify_signature(var_3, var_15)
    assert var_16 is False
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_0, var_1, algorithm=var_17)
    var_19 = b'test'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24, var_1)
    var_26 = b'test rotation'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = var_2.get_signature(var_3)
    var_30 = b'tampered value'
    var_31 = var_2.verify_signature(var_30, var_29)
    assert var_31 is False
    var_32 = b''
    var_33 = var_2.verify_signature(var_3, var_32)
    assert var_33 is False
    var_34 = module_0.NoneAlgorithm()
    var_35 = module_0.Signer(var_0, var_1, algorithm=var_34)
    var_36 = b'test'
    var_37 = var_35.verify_signature(var_36, var_32)
    assert var_37 is True



# Parsed testcases at query #11
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'different-salt'
    var_11 = module_0.Signer(var_0, var_10)
    var_12 = var_11.get_signature(var_7)
    var_13 = var_1.verify_signature(var_7, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.Signer(var_14)
    var_22 = var_21.get_signature(var_18)
    var_23 = var_17.verify_signature(var_18, var_22)
    assert var_23 is True
    var_24 = 'test-string'
    var_25 = var_1.get_signature(var_24)
    var_26 = var_1.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'test'
    var_28 = b'!!!invalid-base64!!!'
    var_29 = var_1.verify_signature(var_27, var_28)
    assert var_29 is False
    var_30 = b''
    var_31 = var_1.verify_signature(var_27, var_30)
    assert var_31 is False



# Parsed testcases at query #12
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'test-string'
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotation-test'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = module_0.Signer(var_12)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_20)
    assert var_21 is True
    var_22 = module_0.NoneAlgorithm()
    var_23 = 'secret'
    var_24 = module_0.Signer(var_23, algorithm=var_22)
    var_25 = b'none-test'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = var_24.verify_signature(var_25, var_7)
    assert var_28 is True
    var_29 = b'sha256-test'
    var_30 = b'wrong'
    var_31 = 'concat'
    var_32 = module_0.Signer(var_23, key_derivation=var_31)
    var_33 = b'concat-test'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = 'hmac'
    var_37 = module_0.Signer(var_23, key_derivation=var_36)
    var_38 = b'hmac-test'
    var_39 = var_37.get_signature(var_38)
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = 'none'
    var_42 = module_0.Signer(var_23, key_derivation=var_41)
    var_43 = b'none-deriv-test'
    var_44 = var_42.get_signature(var_43)
    var_45 = var_42.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = b'!!!invalid-base64!!!'
    var_47 = var_1.verify_signature(var_2, var_46)
    assert var_47 is False
    var_48 = b'|'
    var_49 = module_0.Signer(var_23, sep=var_48)
    var_50 = b'sep-test'
    var_51 = var_49.get_signature(var_50)
    var_52 = var_49.verify_signature(var_50, var_51)
    assert var_52 is True
    var_53 = None
    var_54 = module_0.Signer(var_23, var_53)
    var_55 = b'no-salt-test'
    var_56 = var_54.get_signature(var_55)
    var_57 = var_54.verify_signature(var_55, var_56)
    assert var_57 is True



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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'different-key'
    var_9 = module_0.Signer(var_8)
    var_10 = var_9.get_signature(var_2)
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b''
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'test-value'
    var_16 = var_1.get_signature(var_15)
    var_17 = var_1.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = b'rotation-test'
    var_23 = module_0.Signer(var_18)
    var_24 = var_23.get_signature(var_22)
    var_25 = module_0.Signer(var_19)
    var_26 = var_25.get_signature(var_22)
    var_27 = var_21.verify_signature(var_22, var_24)
    assert var_27 is True
    var_28 = var_21.verify_signature(var_22, var_26)
    assert var_28 is True
    var_29 = b'not-base64!!!'
    var_30 = var_1.verify_signature(var_2, var_29)
    assert var_30 is False
    var_31 = var_1.get_signature(var_2)
    var_32 = var_1.verify_signature(var_2, var_31)
    assert var_32 is True
    var_33 = module_0.NoneAlgorithm()
    var_34 = module_0.Signer(var_0, algorithm=var_33)
    var_35 = var_34.get_signature(var_2)
    var_36 = var_34.verify_signature(var_2, var_35)
    assert var_36 is True
    var_37 = var_34.verify_signature(var_2, var_3)
    assert var_37 is False



# Parsed testcases at query #14
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
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'tampered-value'
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is False
    var_11 = b''
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = 'test-value'
    var_14 = var_2.verify_signature(var_13, var_4)
    assert var_14 is True
    var_15 = 'utf-8'
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_1)
    var_20 = b'test-value-rotation'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, var_1, algorithm=var_23)
    var_25 = b'test-value-none'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = '!!!invalid-base64!!!'
    var_29 = var_2.verify_signature(var_3, var_28)
    assert var_29 is False
    var_30 = 'different-salt'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = var_31.verify_signature(var_3, var_4)
    assert var_32 is False



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = 'ascii'
    var_8 = 'utf-8'
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'different-salt'
    var_13 = module_0.Signer(var_0, var_12)
    var_14 = var_13.get_signature(var_2)
    var_15 = var_1.verify_signature(var_2, var_14)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, algorithm=var_23)
    var_25 = b'test'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'!!!invalid base64!!!'
    var_29 = var_1.verify_signature(var_2, var_28)
    assert var_29 is False
    var_30 = b''
    var_31 = var_1.verify_signature(var_2, var_30)
    assert var_31 is False



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'not-base64!'
    var_13 = var_2.verify_signature(var_3, var_12)
    assert var_13 is False
    var_14 = module_0.NoneAlgorithm()
    var_15 = module_0.Signer(var_0, var_1, algorithm=var_14)
    var_16 = b'test'
    var_17 = var_15.get_signature(var_16)
    var_18 = b''
    var_19 = module_1.base64_encode(var_18)
    var_20 = var_15.verify_signature(var_16, var_17)
    assert var_20 is True
    var_21 = 'old-key'
    var_22 = 'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23, var_1)
    var_25 = b'rotation test'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'secret-key'
    var_29 = 'salt'
    var_30 = b'derivation test'
    var_31 = b'sha256 test'
    var_32 = module_0.Signer(var_28, var_29)
    var_33 = 'string value'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = b'bytes value'
    var_37 = var_32.get_signature(var_36)
    var_38 = module_1.base64_encode(var_37)



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = ''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'different value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is False
    var_11 = 'different-secret'
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = module_0.Signer(var_15)
    var_20 = var_19.get_signature(var_2)
    var_21 = var_18.get_signature(var_2)
    var_22 = var_18.verify_signature(var_2, var_20)
    assert var_22 is True
    var_23 = var_18.verify_signature(var_2, var_21)
    assert var_23 is True
    var_24 = [var_15, var_16]
    var_25 = module_0.Signer(var_24)
    var_26 = 'other-key'
    var_27 = module_0.Signer(var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_25.verify_signature(var_2, var_28)
    assert var_29 is False
    var_30 = 'secret'
    var_31 = module_0.NoneAlgorithm()
    var_32 = module_0.Signer(var_30, algorithm=var_31)
    var_33 = var_32.get_signature(var_2)
    var_34 = var_32.verify_signature(var_2, var_33)
    assert var_34 is True
    var_35 = b'anything'
    var_36 = var_32.verify_signature(var_2, var_35)
    assert var_36 is True
    var_37 = b'|'
    var_38 = module_0.Signer(var_30, sep=var_37)
    var_39 = b'test|value'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True



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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'test-value'
    var_9 = var_1.verify_signature(var_8, var_3)
    assert var_9 is True
    var_10 = b''
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = b'modified-value'
    var_14 = var_1.verify_signature(var_13, var_3)
    assert var_14 is False
    var_15 = '!!!invalid-base64!!!'
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = var_1.verify_signature(var_2, var_10)
    assert var_17 is False
    var_18 = 'old-secret'
    var_19 = module_0.Signer(var_18)
    var_20 = var_19.get_signature(var_2)
    var_21 = 'new-secret'
    var_22 = [var_18, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = var_23.verify_signature(var_2, var_20)
    assert var_24 is True
    var_25 = var_23.get_signature(var_2)
    var_26 = var_23.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = 'wrong-secret'
    var_28 = module_0.Signer(var_27)
    var_29 = var_28.get_signature(var_2)
    var_30 = var_23.verify_signature(var_2, var_29)
    assert var_30 is False



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
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b'wrong-value'
    var_9 = var_1.verify_signature(var_8, var_3)
    assert var_9 is False
    var_10 = 'string-value'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = b''
    var_14 = var_1.get_signature(var_13)
    var_15 = var_1.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-secret'
    var_17 = 'new-secret'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'rotation-test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'different-salt'
    var_24 = module_0.Signer(var_0, var_23)
    var_25 = b'another-value'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_1.verify_signature(var_25, var_26)
    assert var_27 is False
    var_28 = module_0.HMACAlgorithm()
    var_29 = module_0.Signer(var_0, algorithm=var_28)
    var_30 = b'hmac-test'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = module_0.NoneAlgorithm()
    var_34 = module_0.Signer(var_0, algorithm=var_33)
    var_35 = b'none-test'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = b''
    var_39 = var_34.verify_signature(var_35, var_38)
    assert var_39 is True
    var_40 = '!!!invalid-base64!!!'
    var_41 = var_1.verify_signature(var_20, var_40)
    assert var_41 is False
    var_42 = var_21
    var_43 = var_1.verify_signature(var_20, var_42)
    assert var_43 is True
    var_44 = b'x'
    var_45 = 10000
    var_46 = var_44 * var_45
    var_47 = var_1.get_signature(var_46)
    var_48 = var_1.verify_signature(var_46, var_47)
    assert var_48 is True
    var_49 = 'héllo wörld 🔐'
    var_50 = var_1.get_signature(var_49)
    var_51 = var_1.verify_signature(var_49, var_50)
    assert var_51 is True



# Parsed testcases at query #20
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'test-value'
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-rotation'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'secret'
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_18, algorithm=var_19)
    var_21 = b'test-none'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'!!!invalid-base64!!!'
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = 'different-salt'
    var_27 = module_0.Signer(var_0, var_26)
    var_28 = var_27.verify_signature(var_2, var_3)
    assert var_28 is False
    var_29 = b'|'
    var_30 = module_0.Signer(var_0, sep=var_29)
    var_31 = b'test-sep'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = var_1.get_signature(var_2)
    var_35 = var_1.verify_signature(var_2, var_34)
    assert var_35 is True
    var_36 = b'corrupted'
    var_37 = 3
    var_38 = var_3[var_37:]
    var_39 = var_36 + var_38
    var_40 = var_1.verify_signature(var_2, var_39)
    assert var_40 is False



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'string value'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'test with key rotation'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'secret'
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_20, algorithm=var_21)
    var_23 = b'test none algorithm'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'!!!invalid base64!!!'
    var_27 = var_1.verify_signature(var_2, var_26)
    assert var_27 is False
    var_28 = b'custom-salt'
    var_29 = module_0.Signer(var_20, var_28)
    var_30 = b'test with custom salt'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = var_1.get_signature(var_2)
    var_34 = var_1.verify_signature(var_2, var_33)
    assert var_34 is True



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
    var_5 = 'test-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'invalid-sig'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value'
    var_16 = module_0.Signer(var_11)
    var_17 = var_16.get_signature(var_15)
    var_18 = var_14.get_signature(var_15)
    var_19 = var_14.verify_signature(var_15, var_17)
    assert var_19 is True
    var_20 = var_14.verify_signature(var_15, var_18)
    assert var_20 is True
    var_21 = b'custom-salt'
    var_22 = module_0.Signer(var_0, var_21)
    var_23 = b'test-value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'|'
    var_27 = module_0.Signer(var_0, sep=var_26)
    var_28 = b'test-value'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = module_0.NoneAlgorithm()
    var_32 = module_0.Signer(var_0, algorithm=var_31)
    var_33 = b'test-value'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = b'!!!invalid-base64!!!'
    var_37 = var_1.verify_signature(var_33, var_36)
    assert var_37 is False
    var_38 = b''
    var_39 = var_1.get_signature(var_38)
    var_40 = var_1.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = 'héllo'
    var_42 = var_1.get_signature(var_41)
    var_43 = var_1.verify_signature(var_41, var_42)
    assert var_43 is True



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
    var_7 = b'!!!invalid-base64!!!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-rotation'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'secret'
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_18, algorithm=var_19)
    var_21 = b'test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b''
    var_25 = var_20.verify_signature(var_21, var_24)
    assert var_25 is True
    var_26 = b'|'
    var_27 = module_0.Signer(var_18, sep=var_26)
    var_28 = b'test-custom'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = var_1.get_signature(var_24)
    var_32 = var_1.verify_signature(var_24, var_31)
    assert var_32 is True
    var_33 = b'test.value'
    var_34 = var_1.get_signature(var_33)
    var_35 = var_1.verify_signature(var_33, var_34)
    assert var_35 is True



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test Signer.verify_signature method.'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_3.verify_signature(var_4, var_8)
    assert var_9 is False
    var_10 = b'modified value'
    var_11 = var_3.verify_signature(var_10, var_5)
    assert var_11 is False
    var_12 = 'string value'
    var_13 = var_3.get_signature(var_12)
    var_14 = var_3.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = b'!!!invalid-base64!!!'
    var_16 = var_3.verify_signature(var_4, var_15)
    assert var_16 is False
    var_17 = b''
    var_18 = var_3.get_signature(var_17)
    var_19 = var_3.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = 'rotation-salt'
    var_24 = module_0.Signer(var_22, var_23)
    var_25 = b'rotation test'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'none-algo-salt'
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_1, var_28, algorithm=var_29)
    var_31 = b'none algorithm test'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b''
    var_35 = var_30.verify_signature(var_31, var_34)
    assert var_35 is True



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test message'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid_sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'!!!invalid_base64!!!'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'another message'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = module_0.Signer(var_10, var_1)
    var_18 = var_17.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_18)
    assert var_19 is True
    var_20 = 'different-salt'
    var_21 = module_0.Signer(var_0, var_20)
    var_22 = var_21.get_signature(var_3)
    var_23 = var_2.verify_signature(var_3, var_22)
    assert var_23 is False
    var_24 = b''
    var_25 = var_2.get_signature(var_24)
    var_26 = var_2.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = 'key'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_27, algorithm=var_28)
    var_30 = b'test'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True



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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b'tampered-value'
    var_9 = var_1.verify_signature(var_8, var_3)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'not-base64!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = b''
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_0, algorithm=var_17)
    var_19 = b'test'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = var_18.verify_signature(var_19, var_5)
    assert var_22 is False
    var_23 = 'old-key'
    var_24 = 'new-key'
    var_25 = [var_23, var_24]
    var_26 = module_0.Signer(var_25)
    var_27 = b'test'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = 'different-salt'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = b'test'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = var_1.verify_signature(var_32, var_33)
    assert var_35 is False
    var_36 = module_0.HMACAlgorithm()
    var_37 = module_0.Signer(var_0, algorithm=var_36)
    var_38 = b'test-hmac'
    var_39 = var_37.get_signature(var_38)
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True



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
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = 'test-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is True
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'\x00\x01\x02'
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = b'!!!invalid-base64!!!'
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = b''
    var_18 = var_1.verify_signature(var_2, var_17)
    assert var_18 is False
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_0, algorithm=var_19)
    var_21 = b'test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = var_20.verify_signature(var_21, var_17)
    assert var_24 is True
    var_25 = 'custom-salt'
    var_26 = module_0.Signer(var_0, var_25)
    var_27 = b'test'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = module_0.Signer(var_0)
    var_31 = var_30.verify_signature(var_27, var_28)
    assert var_31 is False
    var_32 = 'old-key'
    var_33 = 'new-key'
    var_34 = [var_32, var_33]
    var_35 = module_0.Signer(var_34)
    var_36 = b'test'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = b'|'
    var_40 = module_0.Signer(var_0, sep=var_39)
    var_41 = b'test'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = 'secret-key'
    var_45 = b'test'
    var_46 = b'test'
    var_47 = b'modified-value'
    var_48 = var_1.verify_signature(var_47, var_3)
    assert var_48 is False
    var_49 = 123
    var_50 = var_1.verify_signature(var_49, var_3)
    assert var_50 is True
    var_51 = b'test\nwith\tspaces and \x00 null'
    var_52 = var_1.get_signature(var_51)
    var_53 = var_1.verify_signature(var_51, var_52)
    assert var_53 is True
    var_54 = b'x'
    var_55 = 10000
    var_56 = var_54 * var_55
    var_57 = var_1.get_signature(var_56)
    var_58 = var_1.verify_signature(var_56, var_57)
    assert var_58 is True
    var_59 = None
    var_60 = module_0.Signer(var_44, var_59)
    var_61 = b'test'
    var_62 = var_60.get_signature(var_61)
    var_63 = var_60.verify_signature(var_61, var_62)
    assert var_63 is True
    var_64 = b'a'
    var_65 = 1
    var_66 = var_3[var_65:]
    var_67 = var_64 + var_66
    var_68 = var_1.verify_signature(var_2, var_67)
    assert var_68 is False
    var_69 = -1
    var_70 = var_3[:var_69]
    var_71 = var_1.verify_signature(var_2, var_70)
    assert var_71 is False
    var_72 = b'extra'
    var_73 = var_3 + var_72
    var_74 = var_1.verify_signature(var_2, var_73)
    assert var_74 is False



# Parsed testcases at query #28
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'string_value'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = module_1.base64_encode(var_3)
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is True
    var_15 = b'!!!invalid_base64!!!'
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'old_key'
    var_18 = 'new_key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = var_20.sign(var_2)
    var_22 = var_20.get_signature(var_2)
    var_23 = var_20.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = 'key'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = var_26.get_signature(var_2)
    var_28 = var_26.verify_signature(var_2, var_27)
    assert var_28 is True
    var_29 = 'concat'
    var_30 = module_0.Signer(var_24, key_derivation=var_29)
    var_31 = var_30.get_signature(var_2)
    var_32 = var_30.verify_signature(var_2, var_31)
    assert var_32 is True
    var_33 = 'hmac'
    var_34 = module_0.Signer(var_24, key_derivation=var_33)
    var_35 = var_34.get_signature(var_2)
    var_36 = var_34.verify_signature(var_2, var_35)
    assert var_36 is True
    var_37 = 'none'
    var_38 = module_0.Signer(var_24, key_derivation=var_37)
    var_39 = var_38.get_signature(var_2)
    var_40 = var_38.verify_signature(var_2, var_39)
    assert var_40 is True
    var_41 = 'different_salt'
    var_42 = module_0.Signer(var_24, var_41)
    var_43 = var_42.get_signature(var_2)
    var_44 = var_42.verify_signature(var_2, var_43)
    assert var_44 is True
    var_45 = var_1.verify_signature(var_2, var_43)
    assert var_45 is False
    var_46 = b'original_value'
    var_47 = var_1.get_signature(var_46)
    var_48 = b'modified_value'
    var_49 = var_1.verify_signature(var_48, var_47)
    assert var_49 is False
    var_50 = b''
    var_51 = var_1.verify_signature(var_2, var_50)
    assert var_51 is False
    var_52 = module_0.NoneAlgorithm()
    var_53 = module_0.Signer(var_24, algorithm=var_52)
    var_54 = b'any_value'
    var_55 = var_53.verify_signature(var_54, var_50)
    assert var_55 is True
    var_56 = b'non_empty'
    var_57 = var_53.verify_signature(var_54, var_56)
    assert var_57 is False



# Parsed testcases at query #29
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
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'test-value'
    var_12 = var_2.verify_signature(var_11, var_4)
    assert var_12 is True
    var_13 = 'utf-8'
    var_14 = b'!!!invalid-base64!!!'
    var_15 = var_2.verify_signature(var_3, var_14)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_1)
    var_20 = b'test-rotation'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.Signer(var_16, var_1)
    var_24 = var_23.get_signature(var_20)
    var_25 = var_19.verify_signature(var_20, var_24)
    assert var_25 is True
    var_26 = 'secret'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = b'test-none'
    var_30 = var_28.get_signature(var_29)
    assert var_30 == b''
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'anything'
    var_33 = var_28.verify_signature(var_29, var_32)
    assert var_33 is True
    var_34 = module_0.HMACAlgorithm()
    var_35 = module_0.Signer(var_26, algorithm=var_34)
    var_36 = b'test-hmac'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = 'test'
    var_40 = b'|'
    var_41 = module_0.Signer(var_26, var_39, var_40)
    var_42 = b'test-sep'
    var_43 = var_41.get_signature(var_42)
    var_44 = var_41.verify_signature(var_42, var_43)
    assert var_44 is True
    var_45 = []
    var_46 = module_0.Signer(var_45, var_39)
    var_47 = b'test-empty'
    var_48 = var_46.get_signature(var_47)
    var_49 = var_46.verify_signature(var_47, var_48)
    assert var_49 is True



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
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_7)
    assert var_8 is False
    var_9 = b'modified-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value-2'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = module_0.Signer(var_11)
    var_19 = var_18.get_signature(var_15)
    var_20 = var_14.verify_signature(var_15, var_19)
    assert var_20 is True
    var_21 = 'test-value'
    var_22 = var_1.verify_signature(var_21, var_3)
    assert var_22 is True
    var_23 = b'test-value'
    var_24 = var_1.verify_signature(var_23, var_3)
    assert var_24 is True
    var_25 = 'secret'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = b'test-value-3'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b'anything'
    var_32 = var_27.verify_signature(var_28, var_31)
    assert var_32 is True



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = b'bytes-value'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'string-value'
    var_14 = var_1.get_signature(var_13)
    var_15 = var_1.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = b'!!!invalid-base64!!!'
    var_17 = var_1.verify_signature(var_2, var_16)
    assert var_17 is False
    var_18 = b''
    var_19 = var_1.verify_signature(var_2, var_18)
    assert var_19 is False
    var_20 = 'test'
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_20, algorithm=var_21)
    var_23 = b'test-value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'something'
    var_27 = var_22.verify_signature(var_23, var_26)
    assert var_27 is False
    var_28 = 'old-key'
    var_29 = 'new-key'
    var_30 = [var_28, var_29]
    var_31 = module_0.Signer(var_30)
    var_32 = b'value-signed-with-old-key'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = b'wrong-value'
    var_36 = var_1.verify_signature(var_35, var_3)
    assert var_36 is False
    var_37 = b'value-with-special-chars!@#$%^&*()'
    var_38 = var_1.get_signature(var_37)
    var_39 = var_1.verify_signature(var_37, var_38)
    assert var_39 is True



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'test-salt'
    var_2 = '.'
    var_3 = module_0.Signer(var_0, var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid-signature'
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_3.get_signature(var_9)
    var_11 = var_3.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'utf-8'
    var_13 = 'test-string'
    var_14 = var_3.get_signature(var_13)
    var_15 = var_3.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_1, var_2)
    var_20 = module_0.Signer(var_16, var_1, var_2)
    var_21 = var_20.get_signature(var_4)
    var_22 = var_19.verify_signature(var_4, var_21)
    assert var_22 is True
    var_23 = b'!@#$%^&*()'
    var_24 = var_19.verify_signature(var_4, var_23)
    assert var_24 is False
    var_25 = 'different-salt'
    var_26 = module_0.Signer(var_0, var_25, var_2)
    var_27 = var_26.get_signature(var_4)
    var_28 = var_3.verify_signature(var_4, var_27)
    assert var_28 is False



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
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'tampered-value'
    var_12 = var_1.verify_signature(var_11, var_3)
    assert var_12 is False
    var_13 = b'not-base64!!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = module_0.NoneAlgorithm()
    var_16 = module_0.Signer(var_0, algorithm=var_15)
    var_17 = var_16.get_signature(var_2)
    var_18 = var_16.verify_signature(var_2, var_17)
    assert var_18 is True
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21)
    var_23 = var_22.get_signature(var_2)
    var_24 = var_22.verify_signature(var_2, var_23)
    assert var_24 is True
    var_25 = b'different-salt'
    var_26 = module_0.Signer(var_0, var_25)
    var_27 = var_26.verify_signature(var_2, var_3)
    assert var_27 is False



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
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'different-secret-key'
    var_12 = module_0.Signer(var_11, var_1)
    var_13 = var_12.get_signature(var_3)
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = 'test-value'
    var_16 = var_2.verify_signature(var_15, var_4)
    assert var_16 is True
    var_17 = 'ascii'
    var_18 = b'!!!invalid-base64!!!'
    var_19 = var_2.verify_signature(var_3, var_18)
    assert var_19 is False
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22, var_1)
    var_24 = var_23.get_signature(var_3)
    var_25 = var_23.verify_signature(var_3, var_24)
    assert var_25 is True
    var_26 = var_23.get_signature(var_3)
    var_27 = var_23.verify_signature(var_3, var_26)
    assert var_27 is True
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_0, algorithm=var_28)
    var_30 = var_29.get_signature(var_3)
    var_31 = var_29.verify_signature(var_3, var_30)
    assert var_31 is True



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'modified value'
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is False
    var_11 = b''
    var_12 = var_2.get_signature(var_11)
    var_13 = var_2.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'not-valid-base64!!!'
    var_15 = var_2.verify_signature(var_3, var_14)
    assert var_15 is False
    var_16 = 'test value'
    var_17 = var_2.verify_signature(var_16, var_4)
    assert var_17 is True
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20, var_1)
    var_22 = b'test value'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = 'secret'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = b'test'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
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
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'test-rotated'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'secret'
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_17, algorithm=var_18)
    var_20 = b'test-none'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid-base64!!!'
    var_24 = var_1.verify_signature(var_2, var_23)
    assert var_24 is False
    var_25 = 'test-string'
    var_26 = var_1.get_signature(var_25)
    var_27 = var_1.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'ascii'



# Parsed testcases at query #37
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'test-value'
    var_11 = var_1.get_signature(var_10)
    var_12 = '!!!invalid-base64!!!'
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = -1
    var_15 = var_3[:var_14]
    var_16 = -1
    var_17 = var_3[var_16:]
    var_18 = b'X'
    var_19 = var_17 != var_18
    var_20 = b'Y'
    var_21 = var_18 if var_19 else var_20
    var_22 = var_15 + var_21
    var_23 = var_1.verify_signature(var_2, var_22)
    assert var_23 is False
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_0, algorithm=var_24)
    var_26 = var_25.get_signature(var_2)
    var_27 = var_25.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = 'old-key'
    var_29 = 'new-key'
    var_30 = [var_28, var_29]
    var_31 = module_0.Signer(var_30)
    var_32 = var_31.get_signature(var_2)
    var_33 = var_31.verify_signature(var_2, var_32)
    assert var_33 is True
    var_34 = b'custom-salt'
    var_35 = module_0.Signer(var_0, var_34)
    var_36 = var_35.get_signature(var_2)
    var_37 = var_35.verify_signature(var_2, var_36)
    assert var_37 is True
    var_38 = module_0.Signer(var_0)
    var_39 = var_38.get_signature(var_2)
    var_40 = b':'
    var_41 = module_0.Signer(var_0, sep=var_40)
    var_42 = var_41.get_signature(var_2)
    var_43 = var_41.verify_signature(var_2, var_42)
    assert var_43 is True
    var_44 = var_1.verify_signature(var_2, var_7)
    assert var_44 is False
    var_45 = b'x'
    var_46 = 10000
    var_47 = var_45 * var_46
    var_48 = var_1.get_signature(var_47)
    var_49 = var_1.verify_signature(var_47, var_48)
    assert var_49 is True
    var_50 = 'secret-key'
    var_51 = b'wrong-value'
    var_52 = var_1.verify_signature(var_51, var_3)
    assert var_52 is False



# Parsed testcases at query #38
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test Signer.verify_signature method.'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_3.verify_signature(var_4, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_3.get_signature(var_10)
    var_12 = var_3.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'utf-8'
    var_14 = b'!!!invalid!!!'
    var_15 = var_3.verify_signature(var_4, var_14)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_2)
    var_20 = module_0.Signer(var_16, var_2)
    var_21 = b'test rotation'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_19.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = var_19.get_signature(var_21)
    var_25 = var_19.verify_signature(var_21, var_24)
    assert var_25 is True



# Parsed testcases at query #39
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = b'!!!invalid-base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b''
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'rotation-test'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'string-value'
    var_22 = var_1.get_signature(var_21)
    var_23 = var_1.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'bytes-value'
    var_25 = var_1.get_signature(var_24)
    var_26 = var_1.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = 'secret'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_27, algorithm=var_28)
    var_30 = b'none-algorithm'
    var_31 = var_29.get_signature(var_30)
    assert var_31 == b''
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = var_29.verify_signature(var_30, var_12)
    assert var_33 is True
    var_34 = b'any-sig'
    var_35 = var_29.verify_signature(var_30, var_34)
    assert var_35 is False



# Parsed testcases at query #40
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = var_2.get_signature(var_3)
    var_10 = b'tampered value'
    var_11 = var_2.verify_signature(var_10, var_9)
    assert var_11 is False
    var_12 = b''
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = b'!!invalid-base64!!'
    var_16 = var_2.verify_signature(var_3, var_15)
    assert var_16 is False
    var_17 = b'test'
    var_18 = var_2.get_signature(var_17)
    var_19 = 'test'
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22, var_1)
    var_24 = b'rotation test'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, algorithm=var_27)
    var_29 = b'test with none algorithm'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = var_2.verify_signature(var_3, var_4)
    assert var_32 is True
    var_33 = b'string test'
    var_34 = var_2.get_signature(var_33)
    var_35 = 'string test'
    var_36 = var_2.verify_signature(var_35, var_34)
    assert var_36 is True



# Parsed testcases at query #41
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
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'!!!invalid-base64!!!'
    var_13 = var_2.verify_signature(var_3, var_12)
    assert var_13 is False
    var_14 = 'secret-key'
    var_15 = 'test-salt'
    var_16 = b'test-value'
    var_17 = var_2.get_signature(var_16)
    var_18 = var_2.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = b'wrong-signature'
    var_20 = var_2.verify_signature(var_16, var_19)
    assert var_20 is False
    var_21 = 'old-key'
    var_22 = 'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23, var_15)
    var_25 = b'test-value'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_14, var_15, algorithm=var_28)
    var_30 = b'test-value'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = b''
    var_34 = var_29.verify_signature(var_30, var_33)
    assert var_34 is True
    var_35 = b'test-value'
    var_36 = var_29.get_signature(var_35)
    var_37 = var_29.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = b'wrong-signature'
    var_39 = var_29.verify_signature(var_35, var_38)
    assert var_39 is False
    var_40 = module_0.Signer(var_14, var_15)
    var_41 = 'test-string'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = b'wrong'
    var_45 = var_40.verify_signature(var_41, var_44)
    assert var_45 is False
    var_46 = module_0.Signer(var_14, var_15)
    var_47 = b'test-value'
    var_48 = var_46.get_signature(var_47)
    var_49 = var_46.verify_signature(var_47, var_48)
    assert var_49 is True
    var_50 = b'invalid-sig'
    var_51 = var_46.verify_signature(var_47, var_50)
    assert var_51 is False



# Parsed testcases at query #42
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
    var_7 = b'test-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is True
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = var_14.get_signature(var_2)
    var_16 = var_14.verify_signature(var_2, var_15)
    assert var_16 is True
    var_17 = 'secret'
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_17, algorithm=var_18)
    var_20 = b'test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid-base64!!!'
    var_24 = var_1.verify_signature(var_20, var_23)
    assert var_24 is False



# Parsed testcases at query #43
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
    var_8 = b'!!!invalid-base64!!!'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'test-value'
    var_11 = var_2.verify_signature(var_10, var_4)
    assert var_11 is True
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14, var_1)
    var_16 = var_15.get_signature(var_3)
    var_17 = var_15.verify_signature(var_3, var_16)
    assert var_17 is True
    var_18 = [var_12, var_13]
    var_19 = module_0.Signer(var_18, var_1)
    var_20 = var_19.get_signature(var_3)
    var_21 = var_19.verify_signature(var_3, var_20)
    assert var_21 is True
    var_22 = module_0.HMACAlgorithm()
    var_23 = module_0.Signer(var_0, algorithm=var_22)
    var_24 = var_23.get_signature(var_3)
    var_25 = var_23.verify_signature(var_3, var_24)
    assert var_25 is True
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_0, algorithm=var_26)
    var_28 = var_27.get_signature(var_3)
    var_29 = var_27.verify_signature(var_3, var_28)
    assert var_29 is True
    var_30 = 'secret-key'
    var_31 = 'test-salt'
    var_32 = b''
    var_33 = var_2.get_signature(var_32)
    var_34 = var_2.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = var_2.get_signature(var_3)
    var_36 = var_2.verify_signature(var_3, var_35)
    assert var_36 is True
    var_37 = module_0.NoneAlgorithm()
    var_38 = module_0.Signer(var_30, algorithm=var_37)
    var_39 = b'test'
    var_40 = var_38.verify_signature(var_39, var_32)
    assert var_40 is True



# Parsed testcases at query #44
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'string-value'
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'not-base64!'
    var_15 = var_1.verify_signature(var_2, var_14)
    assert var_15 is False
    var_16 = ''
    var_17 = var_1.verify_signature(var_2, var_16)
    assert var_17 is False
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = b'rotation-test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = 'secret'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = b''
    var_30 = module_1.base64_encode(var_29)
    var_31 = var_27.verify_signature(var_2, var_28)
    assert var_31 is True
    var_32 = b'anything'
    var_33 = module_1.base64_encode(var_32)
    var_34 = var_27.verify_signature(var_2, var_33)
    assert var_34 is True
    var_35 = b'.'
    var_36 = module_0.Signer(var_25, sep=var_35)
    var_37 = b'-'
    var_38 = module_0.Signer(var_25, sep=var_37)
    var_39 = b'test'
    var_40 = var_36.get_signature(var_39)
    var_41 = var_38.get_signature(var_39)
    var_42 = var_36.verify_signature(var_39, var_40)
    assert var_42 is True
    var_43 = var_38.verify_signature(var_39, var_41)
    assert var_43 is True
    var_44 = var_36.verify_signature(var_39, var_41)
    assert var_44 is False
    var_45 = 'concat'
    var_46 = module_0.Signer(var_25, key_derivation=var_45)
    var_47 = 'hmac'
    var_48 = module_0.Signer(var_25, key_derivation=var_47)
    var_49 = 'none'
    var_50 = module_0.Signer(var_25, key_derivation=var_49)
    var_51 = b'derivation-test'
    var_52 = var_46.get_signature(var_51)
    var_53 = var_48.get_signature(var_51)
    var_54 = var_50.get_signature(var_51)
    var_55 = var_46.verify_signature(var_51, var_52)
    assert var_55 is True
    var_56 = var_48.verify_signature(var_51, var_53)
    assert var_56 is True
    var_57 = var_50.verify_signature(var_51, var_54)
    assert var_57 is True
    var_58 = var_46.verify_signature(var_51, var_53)
    assert var_58 is False



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
    var_5 = b'fake-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'invalid-base64!!!'
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = module_0.NoneAlgorithm()
    var_14 = module_0.Signer(var_0, algorithm=var_13)
    var_15 = b'test'
    var_16 = b''
    var_17 = module_1.base64_encode(var_16)
    var_18 = var_14.verify_signature(var_15, var_17)
    assert var_18 is True
    var_19 = b'something'
    var_20 = module_1.base64_encode(var_19)
    var_21 = var_14.verify_signature(var_15, var_20)
    assert var_21 is False
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24)
    var_26 = b'test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = var_25.get_signature(var_26)
    var_30 = 'even-older-key'
    var_31 = var_25.verify_signature(var_26, var_29)
    assert var_31 is True
    var_32 = 'key'
    var_33 = 'salt1'
    var_34 = module_0.Signer(var_32, var_33)
    var_35 = 'salt2'
    var_36 = module_0.Signer(var_32, var_35)
    var_37 = b'test'
    var_38 = var_34.get_signature(var_37)
    var_39 = var_34.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = var_36.verify_signature(var_37, var_38)
    assert var_40 is False
    var_41 = b'|'
    var_42 = module_0.Signer(var_32, sep=var_41)
    var_43 = b'test'
    var_44 = var_42.get_signature(var_43)
    var_45 = var_42.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = 'test-value'
    var_47 = var_1.verify_signature(var_46, var_27)
    assert var_47 is True
    var_48 = b'test-value'



# Parsed testcases at query #46
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'different-key'
    var_10 = module_0.Signer(var_9)
    var_11 = var_10.verify_signature(var_2, var_3)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = module_0.Signer(var_12)
    var_17 = var_16.get_signature(var_2)
    var_18 = var_15.verify_signature(var_2, var_17)
    assert var_18 is True
    var_19 = module_0.Signer(var_13)
    var_20 = var_19.get_signature(var_2)
    var_21 = var_15.verify_signature(var_2, var_20)
    assert var_21 is True
    var_22 = 'unknown-key'
    var_23 = module_0.Signer(var_22)
    var_24 = var_23.get_signature(var_2)
    var_25 = var_15.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = 'secret'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = var_28.get_signature(var_2)
    var_30 = var_28.verify_signature(var_2, var_29)
    assert var_30 is True
    var_31 = '!!!invalid_base64!!!'
    var_32 = var_1.verify_signature(var_2, var_31)
    assert var_32 is False
    var_33 = module_0.Signer(var_26)
    var_34 = var_33.derive_key()
    var_35 = module_1.base64_encode(var_34)
    var_36 = b'extra'
    var_37 = var_35 + var_36
    var_38 = var_1.verify_signature(var_2, var_37)
    assert var_38 is False
    var_39 = 'test value'
    var_40 = var_1.verify_signature(var_39, var_3)
    assert var_40 is True
    var_41 = b'custom_salt'
    var_42 = module_0.Signer(var_26, var_41)
    var_43 = var_42.get_signature(var_2)
    var_44 = var_1.verify_signature(var_2, var_43)
    assert var_44 is False
    var_45 = var_42.verify_signature(var_2, var_43)
    assert var_45 is True



# Parsed testcases at query #47
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test verify_signature method with various scenarios.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'other-value'
    var_11 = var_2.get_signature(var_10)
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = 'test-string'
    var_14 = var_2.get_signature(var_13)
    var_15 = var_2.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = b'!!invalid-base64!!'
    var_17 = var_2.verify_signature(var_3, var_16)
    assert var_17 is False
    var_18 = 'different-salt'
    var_19 = module_0.Signer(var_1, var_18)
    var_20 = var_19.verify_signature(var_3, var_4)
    assert var_20 is False
    var_21 = 'different-secret-key'
    var_22 = module_0.Signer(var_21)
    var_23 = var_22.verify_signature(var_3, var_4)
    assert var_23 is False
    var_24 = 'old-key'
    var_25 = 'new-key'
    var_26 = [var_24, var_25]
    var_27 = module_0.Signer(var_26)
    var_28 = module_0.Signer(var_24)
    var_29 = var_28.get_signature(var_3)
    var_30 = var_27.get_signature(var_3)
    var_31 = var_27.verify_signature(var_3, var_29)
    assert var_31 is True
    var_32 = var_27.verify_signature(var_3, var_30)
    assert var_32 is True
    var_33 = 'test-key'
    var_34 = module_0.NoneAlgorithm()
    var_35 = module_0.Signer(var_33, algorithm=var_34)
    var_36 = var_35.get_signature(var_3)
    assert var_36 == b''
    var_37 = var_35.verify_signature(var_3, var_8)
    assert var_37 is True
    var_38 = b'anything'
    var_39 = var_35.verify_signature(var_3, var_38)
    assert var_39 is False
    var_40 = var_2.get_signature(var_8)
    var_41 = var_2.verify_signature(var_8, var_40)
    assert var_41 is True
    var_42 = b'value with \x00 null and \xff bytes'
    var_43 = var_2.get_signature(var_42)
    var_44 = var_2.verify_signature(var_42, var_43)
    assert var_44 is True
    var_45 = b'a'
    var_46 = 10000
    var_47 = var_45 * var_46
    var_48 = var_2.get_signature(var_47)
    var_49 = var_2.verify_signature(var_47, var_48)
    assert var_49 is True
    var_50 = 256
    var_51 = range(var_50)
    var_52 = bytes(var_51)
    var_53 = var_2.get_signature(var_52)
    var_54 = var_2.verify_signature(var_52, var_53)
    assert var_54 is True



# Parsed testcases at query #48
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b''
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = module_0.NoneAlgorithm()
    var_14 = module_0.Signer(var_0, algorithm=var_13)
    var_15 = var_14.get_signature(var_2)
    var_16 = var_14.verify_signature(var_2, var_15)
    assert var_16 is True
    var_17 = 'hmac'
    var_18 = module_0.Signer(var_0, key_derivation=var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = b'old-secret'
    var_22 = b'new-secret'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23)
    var_25 = var_24.get_signature(var_2)
    var_26 = var_24.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = b'!!!invalid-base64!!!'
    var_28 = var_24.verify_signature(var_2, var_27)
    assert var_28 is False
    var_29 = b'test'
    var_30 = var_1.get_signature(var_29)
    var_31 = 'test'
    var_32 = var_1.verify_signature(var_31, var_30)
    assert var_32 is True



# Parsed testcases at query #49
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_8, var_3)
    assert var_9 is False
    var_10 = var_1.verify_signature(var_2, var_8)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value-2'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = module_0.Signer(var_11)
    var_19 = var_18.get_signature(var_15)
    var_20 = var_14.verify_signature(var_15, var_19)
    assert var_20 is True
    var_21 = 'secret'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = b'test-value-3'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'test-value-4'
    var_28 = b'!!!invalid-base64!!!'
    var_29 = var_1.verify_signature(var_2, var_28)
    assert var_29 is False
    var_30 = b'test.value'
    var_31 = var_1.get_signature(var_30)
    var_32 = var_1.verify_signature(var_30, var_31)
    assert var_32 is True



# Parsed testcases at query #50
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
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'!!!invalid-base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b'different-value'
    var_13 = var_1.verify_signature(var_12, var_3)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test-value-old'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'secret'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = b'test-none'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = 'salt1'
    var_28 = module_0.Signer(var_21, var_27)
    var_29 = 'salt2'
    var_30 = module_0.Signer(var_21, var_29)
    var_31 = b'test-salt'
    var_32 = var_28.get_signature(var_31)
    var_33 = var_28.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = var_30.verify_signature(var_31, var_32)
    assert var_34 is False
    var_35 = 'test-string'
    var_36 = var_1.get_signature(var_35)
    var_37 = var_1.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = var_1.derive_key()
    var_39 = module_1.base64_encode(var_38)



# Parsed testcases at query #51
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    var_5 = b'invalid-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    var_9 = 'different-secret-key'
    var_10 = module_0.Signer(var_9)
    var_11 = var_10.get_signature(var_2)
    var_12 = var_1.verify_signature(var_2, var_11)
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'test-rotation'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    var_20 = module_0.Signer(var_13)
    var_21 = var_20.verify_signature(var_17, var_18)
    var_22 = module_0.Signer(var_14)
    var_23 = var_22.verify_signature(var_17, var_18)
    var_24 = b'garbage'
    var_25 = var_16.verify_signature(var_17, var_24)
    var_26 = var_16.verify_signature(var_7, var_7)
    var_27 = 'secret'
    var_28 = b'|'
    var_29 = module_0.Signer(var_27, sep=var_28)
    var_30 = b'test-sep'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    var_33 = b'invalid'
    var_34 = var_29.verify_signature(var_30, var_33)



# Parsed testcases at query #52
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'test value'
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'ascii'
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'test value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'custom-salt'
    var_20 = module_0.Signer(var_0, var_19)
    var_21 = b'test value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = 'concat'
    var_25 = module_0.Signer(var_0, key_derivation=var_24)
    var_26 = b'test value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'hmac'
    var_30 = module_0.Signer(var_0, key_derivation=var_29)
    var_31 = b'test value'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = 'none'
    var_35 = module_0.Signer(var_0, key_derivation=var_34)
    var_36 = b'test value'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = module_0.NoneAlgorithm()
    var_40 = module_0.Signer(var_0, algorithm=var_39)
    var_41 = b'test value'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = '!!!invalid base64!!!'
    var_45 = var_1.verify_signature(var_41, var_44)
    assert var_45 is False
    var_46 = b''
    var_47 = var_1.get_signature(var_46)
    var_48 = var_1.verify_signature(var_46, var_47)
    assert var_48 is True
    var_49 = 'héllo wörld'
    var_50 = var_1.get_signature(var_49)
    var_51 = var_1.verify_signature(var_49, var_50)
    assert var_51 is True



# Parsed testcases at query #53
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret-key'
    assert var_1 is True
    assert var_1 is False
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = var_2.get_signature(var_3)
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is True
    var_15 = b'original-value'
    var_16 = var_2.get_signature(var_15)
    var_17 = b'modified-value'
    var_18 = var_2.verify_signature(var_17, var_16)
    assert var_18 is False
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21)
    var_23 = b'test-rotation'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = 'custom-salt'
    var_27 = module_0.Signer(var_1, var_26)
    var_28 = b'test-salt'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = var_2.verify_signature(var_28, var_29)
    assert var_31 is False
    var_32 = module_0.NoneAlgorithm()
    var_33 = module_0.Signer(var_1, algorithm=var_32)
    var_34 = b'test-none'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = b''
    var_38 = module_1.base64_encode(var_37)
    var_39 = b'!!!invalid-base64!!!'
    var_40 = var_2.verify_signature(var_3, var_39)
    assert var_40 is False
    var_41 = var_2.get_signature(var_37)
    var_42 = var_2.verify_signature(var_37, var_41)
    assert var_42 is True
    var_43 = var_2.get_signature(var_3)
    var_44 = var_2.verify_signature(var_3, var_43)
    assert var_44 is True
    var_45 = 'secret-key'
    var_46 = b'test-kd'
    var_47 = 'secret-key'
    var_48 = module_0.Signer(var_47)
    var_49 = b'test-sha256'
    var_50 = b'|'
    var_51 = module_0.Signer(var_1, sep=var_50)
    var_52 = b'test-custom-sep'
    var_53 = var_51.get_signature(var_52)
    var_54 = var_51.verify_signature(var_52, var_53)
    assert var_54 is True
    var_55 = b'a'
    var_56 = 10000
    var_57 = var_55 * var_56
    var_58 = var_2.get_signature(var_57)
    var_59 = var_2.verify_signature(var_57, var_58)
    assert var_59 is True
    var_60 = b'test.sep.value'
    var_61 = var_2.get_signature(var_60)
    var_62 = var_2.verify_signature(var_60, var_61)
    assert var_62 is True
    var_63 = 'héllo wörld'
    var_64 = var_2.get_signature(var_63)
    var_65 = var_2.verify_signature(var_63, var_64)
    assert var_65 is True



# Parsed testcases at query #54
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
    var_5 = 'test-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = 'ascii'
    var_8 = b'wrong-signature'
    var_9 = module_1.base64_encode(var_8)
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = b'wrong-value'
    var_12 = var_1.verify_signature(var_11, var_3)
    assert var_12 is False
    var_13 = b'!!!invalid!!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = b''
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'|'
    var_25 = module_0.Signer(var_0, sep=var_24)
    var_26 = b'test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_0, algorithm=var_29)
    var_31 = b'test'
    var_32 = var_30.get_signature(var_31)
    var_33 = module_1.base64_encode(var_15)
    var_34 = var_30.verify_signature(var_31, var_32)
    assert var_34 is True
    var_35 = b'test'
    var_36 = 'secret-key'
    var_37 = b'test'



# Parsed testcases at query #55
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
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = var_2.verify_signature(var_3, var_9)
    assert var_11 is False
    var_12 = 'test-string'
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = module_0.NoneAlgorithm()
    var_16 = module_0.Signer(var_0, algorithm=var_15)
    var_17 = b'test-value'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = b'!!!invalid-base64!!!'
    var_21 = var_2.verify_signature(var_3, var_20)
    assert var_21 is False
    var_22 = b''
    var_23 = var_2.verify_signature(var_3, var_22)
    assert var_23 is False
    var_24 = 'old-key'
    var_25 = 'new-key'
    var_26 = [var_24, var_25]
    var_27 = module_0.Signer(var_26, var_1)
    var_28 = b'test-value'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = module_0.Signer(var_24, var_1)
    var_32 = var_31.get_signature(var_28)
    var_33 = var_27.verify_signature(var_28, var_32)
    assert var_33 is True
    var_34 = b'|'
    var_35 = module_0.Signer(var_0, sep=var_34)
    var_36 = b'test-value'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = module_0.HMACAlgorithm()
    var_40 = module_0.Signer(var_0, algorithm=var_39)
    var_41 = b'test-value'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = b'test-value'



# Parsed testcases at query #56
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'different-secret'
    var_11 = module_0.Signer(var_10)
    var_12 = var_11.get_signature(var_2)
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = var_17.get_signature(var_2)
    var_19 = var_17.verify_signature(var_2, var_18)
    assert var_19 is True
    var_20 = 'ascii'
    var_21 = '!!!invalid-base64!!!'
    var_22 = var_1.verify_signature(var_2, var_21)
    assert var_22 is False
    var_23 = module_0.NoneAlgorithm()
    var_24 = 'secret'
    var_25 = module_0.Signer(var_24, algorithm=var_23)
    var_26 = b'test'
    var_27 = b''
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'sig'
    var_30 = var_25.verify_signature(var_26, var_29)
    assert var_30 is False
    var_31 = 'test-value'
    var_32 = var_1.verify_signature(var_31, var_3)
    assert var_32 is True
    var_33 = b'x'
    var_34 = 10000
    var_35 = var_33 * var_34
    var_36 = var_1.get_signature(var_35)
    var_37 = var_1.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = b'test\nwith\tspecial\x00chars'
    var_39 = var_1.get_signature(var_38)
    var_40 = var_1.verify_signature(var_38, var_39)
    assert var_40 is True



# Parsed testcases at query #57
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'different-secret'
    var_11 = module_0.Signer(var_10)
    var_12 = var_11.get_signature(var_2)
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test-rotated'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.Signer(var_14)
    var_22 = var_21.get_signature(var_18)
    var_23 = var_17.verify_signature(var_18, var_22)
    assert var_23 is True
    var_24 = b'???invalid-base64???'
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = 'key'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = b'test-none'
    var_30 = var_28.get_signature(var_29)
    assert var_30 == b''
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'test-custom'
    var_33 = b'wrong-sig'
    var_34 = 'string-value'
    var_35 = var_1.get_signature(var_34)
    var_36 = var_1.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = var_1.verify_signature(var_34, var_33)
    assert var_37 is False



# Parsed testcases at query #58
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'!!!invalid_base64!!!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = module_0.NoneAlgorithm()
    var_12 = module_0.Signer(var_0, algorithm=var_11)
    var_13 = b'test value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = b'test value'
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'test value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.Signer(var_0)
    var_25 = 'test string'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = module_0.Signer(var_0)
    var_29 = b'test value'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = module_0.Signer(var_0)
    var_33 = b'test value'
    var_34 = var_32.get_signature(var_33)



# Parsed testcases at query #59
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'original value'
    var_11 = var_1.get_signature(var_10)
    var_12 = b'different value'
    var_13 = var_1.verify_signature(var_12, var_11)
    assert var_13 is False
    var_14 = 'test value'
    var_15 = var_1.get_signature(var_14)
    var_16 = var_1.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = b'!!!invalid base64!!!'
    var_18 = var_1.verify_signature(var_2, var_17)
    assert var_18 is False
    var_19 = 'old-key'
    var_20 = 'new-key'
    var_21 = [var_19, var_20]
    var_22 = module_0.Signer(var_21)
    var_23 = b'test value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True



# Parsed testcases at query #60
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = module_0.NoneAlgorithm()
    var_10 = module_0.Signer(var_0, algorithm=var_9)
    var_11 = b'test-value'
    var_12 = var_10.get_signature(var_11)
    var_13 = var_10.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'secret'
    var_22 = 'salt1'
    var_23 = module_0.Signer(var_21, var_22)
    var_24 = 'salt2'
    var_25 = module_0.Signer(var_21, var_24)
    var_26 = b'test'
    var_27 = var_23.get_signature(var_26)
    var_28 = var_25.get_signature(var_26)
    var_29 = var_23.verify_signature(var_26, var_27)
    assert var_29 is True
    var_30 = var_25.verify_signature(var_26, var_28)
    assert var_30 is True
    var_31 = var_23.verify_signature(var_26, var_28)
    assert var_31 is False
    var_32 = var_25.verify_signature(var_26, var_27)
    assert var_32 is False
    var_33 = b'test'
    var_34 = module_0.Signer(var_21)
    var_35 = b'original-value'
    var_36 = var_34.get_signature(var_35)
    var_37 = b'modified-value'
    var_38 = var_34.verify_signature(var_37, var_36)
    assert var_38 is False
    var_39 = module_0.Signer(var_21)
    var_40 = 'string-value'
    var_41 = var_39.get_signature(var_40)
    var_42 = var_39.verify_signature(var_40, var_41)
    assert var_42 is True



# Parsed testcases at query #61
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'string-value'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'rotation-test'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = b'AAAA'
    var_21 = 4
    var_22 = var_3[var_21:]
    var_23 = var_20 + var_22
    var_24 = var_16.verify_signature(var_17, var_23)
    assert var_24 is False
    var_25 = 'secret'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = b'test'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b'any-sig'
    var_32 = var_27.verify_signature(var_28, var_31)
    assert var_32 is True
    var_33 = b'!!!invalid-base64!!!'
    var_34 = var_1.verify_signature(var_2, var_33)
    assert var_34 is False
    var_35 = 'salt1'
    var_36 = module_0.Signer(var_25, var_35)
    var_37 = 'salt2'
    var_38 = module_0.Signer(var_25, var_37)
    var_39 = b'test-salt'
    var_40 = var_36.get_signature(var_39)
    var_41 = var_36.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = var_38.verify_signature(var_39, var_40)
    assert var_42 is False
    var_43 = 'concat'
    var_44 = module_0.Signer(var_25, key_derivation=var_43)
    var_45 = b'test-concat'
    var_46 = var_44.get_signature(var_45)
    var_47 = var_44.verify_signature(var_45, var_46)
    assert var_47 is True
    var_48 = 'hmac'
    var_49 = module_0.Signer(var_25, key_derivation=var_48)
    var_50 = b'test-hmac'
    var_51 = var_49.get_signature(var_50)
    var_52 = var_49.verify_signature(var_50, var_51)
    assert var_52 is True
    var_53 = 'none'
    var_54 = module_0.Signer(var_25, key_derivation=var_53)
    var_55 = b'test-none'
    var_56 = var_54.get_signature(var_55)
    var_57 = var_54.verify_signature(var_55, var_56)
    assert var_57 is True



# Parsed testcases at query #62
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'!!!invalid-base64!!!'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_2.get_signature(var_10)
    var_12 = var_2.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'test-value'
    var_14 = var_2.verify_signature(var_13, var_4)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17, var_1)
    var_19 = b'test-value-rotation'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'test-key'
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_22, algorithm=var_23)
    var_25 = b'test-none-value'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b''
    var_29 = var_24.verify_signature(var_25, var_28)
    assert var_29 is True
    var_30 = b'test-hmac-value'
    var_31 = b'different-value'
    var_32 = var_2.verify_signature(var_31, var_4)
    assert var_32 is False



# Parsed testcases at query #63
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'!!!invalid-base64!!!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = module_0.NoneAlgorithm()
    var_12 = module_0.Signer(var_0, algorithm=var_11)
    var_13 = b'test-value'
    var_14 = var_12.get_signature(var_13)
    assert var_14 == b''
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'hmac'
    var_17 = module_0.Signer(var_0, key_derivation=var_16)
    var_18 = b'test-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'old-key'
    var_22 = 'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23)
    var_25 = b'test-value'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'test-value'
    var_29 = var_1.get_signature(var_28)
    var_30 = var_1.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b'test-value'
    var_32 = var_1.get_signature(var_31)
    var_33 = var_1.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = var_1.get_signature(var_31)
    var_35 = 'ascii'



# Parsed testcases at query #64
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test Signer.verify_signature method.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b'modified-value'
    var_9 = var_2.verify_signature(var_8, var_4)
    assert var_9 is False
    var_10 = 'test-value'
    var_11 = var_2.verify_signature(var_10, var_4)
    assert var_11 is True
    var_12 = b''
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'test-rotation'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = module_0.Signer(var_15)
    var_23 = var_22.get_signature(var_19)
    var_24 = var_18.verify_signature(var_19, var_23)
    assert var_24 is True
    var_25 = 'test-key'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = b'test-none'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b''
    var_32 = var_27.verify_signature(var_28, var_31)
    assert var_32 is True
    var_33 = b'!!!invalid-base64!!!'
    var_34 = var_2.verify_signature(var_28, var_33)
    assert var_34 is False
    var_35 = var_2.verify_signature(var_28, var_31)
    assert var_35 is False



# Parsed testcases at query #65
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'not-base64!!!'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'test'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'string value'
    var_21 = var_1.get_signature(var_20)
    var_22 = var_1.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'key'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_23, algorithm=var_24)
    var_26 = b'test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'test'



# Parsed testcases at query #66
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-sig'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = 'test-value'
    var_9 = var_2.verify_signature(var_8, var_4)
    assert var_9 is True
    var_10 = 'ascii'
    var_11 = b''
    var_12 = var_2.get_signature(var_11)
    var_13 = var_2.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = b'different-value'
    var_15 = var_2.verify_signature(var_14, var_4)
    assert var_15 is False
    var_16 = b'!!!invalid-base64!!!'
    var_17 = var_2.verify_signature(var_3, var_16)
    assert var_17 is False
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20, var_1)
    var_22 = var_21.get_signature(var_3)
    var_23 = var_21.verify_signature(var_3, var_22)
    assert var_23 is True
    var_24 = 'test-key'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, var_1, algorithm=var_25)
    var_27 = var_26.get_signature(var_3)
    var_28 = var_26.verify_signature(var_3, var_27)
    assert var_28 is True
    var_29 = b'modified-value'
    var_30 = var_2.verify_signature(var_29, var_4)
    assert var_30 is False



# Parsed testcases at query #67
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'different-key'
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = var_13.verify_signature(var_3, var_4)
    assert var_14 is False
    var_15 = 'different-salt'
    var_16 = module_0.Signer(var_0, var_15)
    var_17 = var_16.verify_signature(var_3, var_4)
    assert var_17 is False
    var_18 = 'test-string-value'
    var_19 = var_2.get_signature(var_18)
    var_20 = var_2.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'not-base64!!'
    var_22 = var_2.verify_signature(var_3, var_21)
    assert var_22 is False
    var_23 = 'key'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_23, algorithm=var_24)
    var_26 = b'test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'old-key'
    var_30 = 'new-key'
    var_31 = [var_29, var_30]
    var_32 = 'test'
    var_33 = module_0.Signer(var_31, var_32)
    var_34 = b'test'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = b'hello\x00world'
    var_38 = var_2.get_signature(var_37)
    var_39 = var_2.verify_signature(var_37, var_38)
    assert var_39 is True



# Parsed testcases at query #68
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = var_1.get_signature(var_2)
    var_9 = b'modified-value'
    var_10 = var_1.verify_signature(var_9, var_8)
    assert var_10 is False
    var_11 = b''
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'test-value'
    var_15 = var_1.get_signature(var_14)
    var_16 = var_1.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'not-base64!!'
    var_18 = var_1.verify_signature(var_2, var_17)
    assert var_18 is False
    var_19 = var_1.verify_signature(var_2, var_11)
    assert var_19 is False
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = b'test-value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'|'
    var_28 = module_0.Signer(var_0, sep=var_27)
    var_29 = b'test-value'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = module_0.NoneAlgorithm()
    var_33 = module_0.Signer(var_0, algorithm=var_32)
    var_34 = b'test-value'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = b'test-value'
    var_38 = 'hmac'
    var_39 = module_0.Signer(var_0, key_derivation=var_38)
    var_40 = b'test-value'
    var_41 = var_39.get_signature(var_40)
    var_42 = var_39.verify_signature(var_40, var_41)
    assert var_42 is True
    var_43 = 'concat'
    var_44 = module_0.Signer(var_0, key_derivation=var_43)
    var_45 = var_44.get_signature(var_40)
    var_46 = var_44.verify_signature(var_40, var_45)
    assert var_46 is True
    var_47 = 'none'
    var_48 = module_0.Signer(var_0, key_derivation=var_47)
    var_49 = var_48.get_signature(var_40)
    var_50 = var_48.verify_signature(var_40, var_49)
    assert var_50 is True
    var_51 = None
    var_52 = module_0.Signer(var_0, var_51)
    var_53 = b'test-value'
    var_54 = var_52.get_signature(var_53)
    var_55 = var_52.verify_signature(var_53, var_54)
    assert var_55 is True



# Parsed testcases at query #69
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
    var_8 = b''
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = module_0.NoneAlgorithm()
    var_11 = module_0.Signer(var_0, algorithm=var_10)
    var_12 = var_11.get_signature(var_3)
    var_13 = var_11.verify_signature(var_3, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16, var_1)
    var_18 = b'test-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = b'|'
    var_22 = module_0.Signer(var_0, sep=var_21)
    var_23 = b'test-value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True



# Parsed testcases at query #70
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
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'different-secret'
    var_11 = module_0.Signer(var_10)
    var_12 = var_11.get_signature(var_7)
    var_13 = var_1.verify_signature(var_7, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'rotation-test'
    var_19 = module_0.Signer(var_14)
    var_20 = var_19.get_signature(var_18)
    var_21 = var_17.get_signature(var_18)
    var_22 = var_17.verify_signature(var_18, var_20)
    assert var_22 is True
    var_23 = var_17.verify_signature(var_18, var_21)
    assert var_23 is True
    var_24 = 'string-value'
    var_25 = var_1.get_signature(var_24)
    var_26 = var_1.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'!!!invalid-base64!!!'
    var_28 = var_1.verify_signature(var_18, var_27)
    assert var_28 is False
    var_29 = 'secret'
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_29, algorithm=var_30)
    var_32 = b'none-test'
    var_33 = var_31.get_signature(var_32)
    var_34 = b''
    var_35 = module_1.base64_encode(var_34)
    var_36 = var_31.verify_signature(var_32, var_33)
    assert var_36 is True
    var_37 = b'something'
    var_38 = var_31.verify_signature(var_32, var_37)
    assert var_38 is False



# Parsed testcases at query #71
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
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'test-value'
    var_12 = var_1.verify_signature(var_11, var_3)
    assert var_12 is True
    var_13 = 'not-base64!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = b''
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'different-salt'
    var_18 = module_0.Signer(var_0, var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_1.verify_signature(var_2, var_19)
    assert var_20 is False
    var_21 = var_18.verify_signature(var_2, var_19)
    assert var_21 is True
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24)
    var_26 = var_25.get_signature(var_2)
    var_27 = var_25.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = 'concat'
    var_29 = module_0.Signer(var_0, key_derivation=var_28)
    var_30 = var_29.get_signature(var_2)
    var_31 = var_29.verify_signature(var_2, var_30)
    assert var_31 is True
    var_32 = 'hmac'
    var_33 = module_0.Signer(var_0, key_derivation=var_32)
    var_34 = var_33.get_signature(var_2)
    var_35 = var_33.verify_signature(var_2, var_34)
    assert var_35 is True
    var_36 = 'none'
    var_37 = module_0.Signer(var_0, key_derivation=var_36)
    var_38 = var_37.get_signature(var_2)
    var_39 = var_37.verify_signature(var_2, var_38)
    assert var_39 is True
    var_40 = module_0.NoneAlgorithm()
    var_41 = module_0.Signer(var_0, algorithm=var_40)
    var_42 = var_41.get_signature(var_2)
    var_43 = var_41.verify_signature(var_2, var_42)
    assert var_43 is True
    var_44 = b'modified-value'
    var_45 = var_1.verify_signature(var_44, var_3)
    assert var_45 is False
    var_46 = b'value.with.dots'
    var_47 = var_1.get_signature(var_46)
    var_48 = var_1.verify_signature(var_46, var_47)
    assert var_48 is True
    var_49 = -1
    var_50 = var_3[:var_49]
    var_51 = -1
    var_52 = var_3[var_51:]
    var_53 = b'\x00'
    var_54 = var_52 != var_53
    var_55 = b'\x01'
    var_56 = var_53 if var_54 else var_55
    var_57 = var_50 + var_56
    var_58 = var_1.verify_signature(var_2, var_57)
    assert var_58 is False



# Parsed testcases at query #72
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
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'different-secret'
    var_12 = module_0.Signer(var_11, var_1)
    var_13 = var_12.get_signature(var_3)
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = b'!!!invalid-base64!!!'
    var_16 = var_2.verify_signature(var_3, var_15)
    assert var_16 is False
    var_17 = b''
    var_18 = var_2.verify_signature(var_3, var_17)
    assert var_18 is False
    var_19 = 'string-value'
    var_20 = var_2.get_signature(var_19)
    var_21 = var_2.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = var_2.get_signature(var_3)
    var_23 = 'old-key'
    var_24 = 'new-key'
    var_25 = [var_23, var_24]
    var_26 = 'rotation-salt'
    var_27 = module_0.Signer(var_25, var_26)
    var_28 = b'rotation-test'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = module_0.NoneAlgorithm()
    var_32 = 'secret'
    var_33 = 'test'
    var_34 = module_0.Signer(var_32, var_33, algorithm=var_31)
    var_35 = b'test-value'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = b'invalid'
    var_39 = var_34.verify_signature(var_35, var_38)
    assert var_39 is False
    var_40 = b'ab'
    var_41 = var_2.verify_signature(var_3, var_40)
    assert var_41 is False
    var_42 = b'a'
    var_43 = 10000
    var_44 = var_42 * var_43
    var_45 = var_2.get_signature(var_44)
    var_46 = var_2.verify_signature(var_44, var_45)
    assert var_46 is True



# Parsed testcases at query #73
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = b'different-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is False
    var_11 = 'test-value'
    var_12 = var_1.verify_signature(var_11, var_3)
    assert var_12 is True
    var_13 = 'ascii'
    var_14 = b'!!!invalid-base64!!!'
    var_15 = var_1.verify_signature(var_2, var_14)
    assert var_15 is False
    var_16 = var_1.verify_signature(var_2, var_7)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'test-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.Signer(var_17)
    var_25 = var_24.get_signature(var_21)
    var_26 = var_20.verify_signature(var_21, var_25)
    assert var_26 is True
    var_27 = 'wrong-key'
    var_28 = module_0.Signer(var_27)
    var_29 = var_28.get_signature(var_21)
    var_30 = var_20.verify_signature(var_21, var_29)
    assert var_30 is False
    var_31 = 'custom-salt'
    var_32 = module_0.Signer(var_0, var_31)
    var_33 = var_32.get_signature(var_21)
    var_34 = var_32.verify_signature(var_21, var_33)
    assert var_34 is True
    var_35 = module_0.Signer(var_0)
    var_36 = var_35.get_signature(var_21)
    var_37 = var_32.verify_signature(var_21, var_36)
    assert var_37 is False
    var_38 = b'|'
    var_39 = module_0.Signer(var_0, sep=var_38)
    var_40 = b'test-value'
    var_41 = var_39.get_signature(var_40)
    var_42 = var_39.verify_signature(var_40, var_41)
    assert var_42 is True
    var_43 = module_0.NoneAlgorithm()
    var_44 = module_0.Signer(var_0, algorithm=var_43)
    var_45 = b'test-value'
    var_46 = var_44.get_signature(var_45)
    var_47 = var_44.verify_signature(var_45, var_46)
    assert var_47 is True
    var_48 = b'test-value'
    var_49 = module_0.Signer(var_0)
    var_50 = var_49.get_signature(var_48)



# Parsed testcases at query #74
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'test@#$%^&*()'
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17, var_1)
    var_19 = b'rotation test'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = b'!!!invalid-base64!!!'
    var_23 = var_2.verify_signature(var_3, var_22)
    assert var_23 is False
    var_24 = b'test bytes'
    var_25 = var_2.get_signature(var_24)
    var_26 = 'secret'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = b'test none algorithm'
    var_30 = var_28.get_signature(var_29)
    var_31 = b''
    var_32 = module_1.base64_encode(var_31)
    var_33 = var_28.verify_signature(var_29, var_30)
    assert var_33 is True
    var_34 = b'test sha256'
    var_35 = b'|'
    var_36 = module_0.Signer(var_26, sep=var_35)
    var_37 = b'custom sep'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True



# Parsed testcases at query #75
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'test-string'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = b'!!!invalid-base64!!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'rotation-test'
    var_20 = var_18.get_signature(var_19)
    var_21 = module_0.Signer(var_15)
    var_22 = var_21.verify_signature(var_19, var_20)
    assert var_22 is False
    var_23 = var_18.verify_signature(var_19, var_20)
    assert var_23 is True
    var_24 = 'key'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = b'none-test'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b''
    var_31 = module_1.base64_encode(var_30)
    var_32 = 'concat'
    var_33 = module_0.Signer(var_24, key_derivation=var_32)
    var_34 = b'concat-test'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = 'none'
    var_38 = module_0.Signer(var_24, key_derivation=var_37)
    var_39 = b'none-derivation-test'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = 'hmac'
    var_43 = module_0.Signer(var_24, key_derivation=var_42)
    var_44 = b'hmac-test'
    var_45 = var_43.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = 'key1'
    var_48 = module_0.Signer(var_47)
    var_49 = 'key2'
    var_50 = module_0.Signer(var_49)
    var_51 = b'diff-test'
    var_52 = var_48.get_signature(var_51)
    var_53 = var_50.get_signature(var_51)
    var_54 = var_48.verify_signature(var_51, var_52)
    assert var_54 is True
    var_55 = var_50.verify_signature(var_51, var_53)
    assert var_55 is True
    var_56 = var_48.verify_signature(var_51, var_53)
    assert var_56 is False
    var_57 = var_50.verify_signature(var_51, var_52)
    assert var_57 is False



# Parsed testcases at query #76
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
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'not-valid-base64!!'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'test-value'
    var_15 = 0
    var_16 = var_13.secret_keys[var_15]
    var_17 = var_13.derive_key(var_16)
    var_18 = var_13.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_18)
    assert var_19 is True
    var_20 = 'different-salt'
    var_21 = module_0.Signer(var_0, var_20)
    var_22 = var_21.get_signature(var_14)
    var_23 = var_21.verify_signature(var_14, var_22)
    assert var_23 is True
    var_24 = var_1.verify_signature(var_14, var_22)
    assert var_24 is False
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_0, algorithm=var_25)
    var_27 = var_26.get_signature(var_14)
    var_28 = var_26.verify_signature(var_14, var_27)
    assert var_28 is True
    var_29 = b''
    var_30 = var_1.get_signature(var_29)
    var_31 = var_1.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'bytes-value'
    var_33 = var_1.get_signature(var_32)
    var_34 = var_1.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = 'string-value'
    var_36 = var_1.get_signature(var_35)
    var_37 = var_1.verify_signature(var_35, var_36)
    assert var_37 is True



# Parsed testcases at query #77
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'test!@#$%^&*()_+-=[]{}|;\':",./<>?`~'
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 256
    var_16 = range(var_15)
    var_17 = bytes(var_16)
    var_18 = var_2.get_signature(var_17)
    var_19 = var_2.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'string value'
    var_21 = var_2.get_signature(var_20)
    var_22 = var_2.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'old-key'
    var_24 = 'new-key'
    var_25 = [var_23, var_24]
    var_26 = module_0.Signer(var_25, var_1)
    var_27 = b'test rotation'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = var_26.verify_signature(var_27, var_28)
    assert var_30 is True
    var_31 = 'wrong-key'
    var_32 = module_0.Signer(var_31, var_1)
    var_33 = var_32.verify_signature(var_3, var_4)
    assert var_33 is False
    var_34 = 'secret'
    var_35 = module_0.NoneAlgorithm()
    var_36 = module_0.Signer(var_34, algorithm=var_35)
    var_37 = b'test'
    var_38 = var_36.get_signature(var_37)
    var_39 = b''
    var_40 = module_1.base64_encode(var_39)
    var_41 = var_36.verify_signature(var_37, var_38)
    assert var_41 is True
    var_42 = b'!!!invalid-base64!!!'
    var_43 = var_2.verify_signature(var_3, var_42)
    assert var_43 is False
    var_44 = var_2.verify_signature(var_3, var_39)
    assert var_44 is False
    var_45 = 'different-salt'
    var_46 = module_0.Signer(var_0, var_45)
    var_47 = var_46.verify_signature(var_3, var_4)
    assert var_47 is False
    var_48 = b'custom digest'



# Parsed testcases at query #78
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'test-secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid-signature'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_3.verify_signature(var_4, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_3.get_signature(var_10)
    var_12 = var_3.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'not-base64'
    var_14 = var_3.verify_signature(var_4, var_13)
    assert var_14 is False
    var_15 = 'string-value'
    var_16 = var_3.get_signature(var_15)
    var_17 = var_3.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'different-salt'
    var_19 = module_0.Signer(var_1, var_18)
    var_20 = var_19.get_signature(var_4)
    var_21 = var_3.verify_signature(var_4, var_20)
    assert var_21 is False
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24, var_2)
    var_26 = b'rotated-value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'test-key'
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_29, algorithm=var_30)
    var_32 = b'none-alg-value'
    var_33 = var_31.get_signature(var_32)
    assert var_33 == b''
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = 'concat'
    var_36 = module_0.Signer(var_29, key_derivation=var_35)
    var_37 = b'concat-value'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = 'hmac'
    var_41 = module_0.Signer(var_29, key_derivation=var_40)
    var_42 = b'hmac-value'
    var_43 = var_41.get_signature(var_42)
    var_44 = var_41.verify_signature(var_42, var_43)
    assert var_44 is True



# Parsed testcases at query #79
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
    var_7 = b'modified-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = 'utf-8'
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'test-rotation'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = module_0.Signer(var_12)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_20)
    assert var_21 is True
    var_22 = b'!!!invalid-base64!!!'
    var_23 = var_1.verify_signature(var_2, var_22)
    assert var_23 is False
    var_24 = b''
    var_25 = var_1.get_signature(var_24)
    var_26 = var_1.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'value-with-special-chars!@#$%^&*()'
    var_28 = var_1.get_signature(var_27)
    var_29 = var_1.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = 'custom-salt'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = b'test-salt'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = var_1.verify_signature(var_32, var_33)
    assert var_35 is False
    var_36 = module_0.NoneAlgorithm()
    var_37 = module_0.Signer(var_0, algorithm=var_36)
    var_38 = b'test-none'
    var_39 = var_37.get_signature(var_38)
    assert var_39 == b''
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True



# Parsed testcases at query #80
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b'tampered-value'
    var_9 = var_1.verify_signature(var_8, var_3)
    assert var_9 is False
    var_10 = 'not-base64!@#'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b''
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'old-value'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'secret'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = b'test'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'different'
    var_28 = module_1.base64_encode(var_27)
    var_29 = var_23.verify_signature(var_24, var_28)
    assert var_29 is False
    var_30 = 'salt1'
    var_31 = module_0.Signer(var_21, var_30)
    var_32 = 'salt2'
    var_33 = module_0.Signer(var_21, var_32)
    var_34 = b'test'
    var_35 = var_31.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is False
    var_37 = 'test-value'
    var_38 = var_1.verify_signature(var_37, var_3)
    assert var_38 is True
    var_39 = var_1.verify_signature(var_2, var_3)
    assert var_39 is True
    var_40 = b''
    var_41 = var_1.get_signature(var_40)
    var_42 = var_1.verify_signature(var_40, var_41)
    assert var_42 is True
    var_43 = b'\xff\xfe\xfd'
    var_44 = var_1.verify_signature(var_2, var_43)
    assert var_44 is False



# Parsed testcases at query #81
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
    var_6 = b'wrong'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'!!!invalid-base64!!!'
    var_10 = var_2.verify_signature(var_3, var_9)
    assert var_10 is False
    var_11 = b''
    var_12 = var_2.get_signature(var_11)
    var_13 = var_2.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16, var_1)
    var_18 = b'test-value-2'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.Signer(var_14, var_1)
    var_22 = var_21.get_signature(var_18)
    var_23 = var_17.verify_signature(var_18, var_22)
    assert var_23 is True
    var_24 = 'wrong-key'
    var_25 = module_0.Signer(var_24, var_1)
    var_26 = var_25.get_signature(var_18)
    var_27 = var_17.verify_signature(var_18, var_26)
    assert var_27 is False
    var_28 = 'test-value'
    var_29 = var_2.verify_signature(var_28, var_4)
    assert var_29 is True
    var_30 = 'utf-8'
    var_31 = 'secret'
    var_32 = module_0.NoneAlgorithm()
    var_33 = module_0.Signer(var_31, algorithm=var_32)
    var_34 = b'test'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True



# Parsed testcases at query #82
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_3.verify_signature(var_4, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_3.get_signature(var_10)
    var_12 = var_3.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'string-value'
    var_14 = var_3.get_signature(var_13)
    var_15 = var_3.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = -1
    var_17 = var_5[:var_16]
    var_18 = -1
    var_19 = var_5[var_18:]
    var_20 = b'x'
    var_21 = var_19 != var_20
    var_22 = b'y'
    var_23 = var_20 if var_21 else var_22
    var_24 = var_17 + var_23
    var_25 = var_3.verify_signature(var_4, var_24)
    assert var_25 is False
    var_26 = 'key'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = b'test'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = 'old-key'
    var_33 = 'new-key'
    var_34 = [var_32, var_33]
    var_35 = module_0.Signer(var_34, var_2)
    var_36 = b'rotation-test'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = 0
    var_40 = var_35.secret_keys[var_39]
    var_41 = var_35.derive_key(var_40)
    var_42 = module_0.HMACAlgorithm()
    var_43 = var_42.get_signature(var_41, var_36)
    var_44 = module_1.base64_encode(var_43)
    var_45 = var_35.verify_signature(var_36, var_44)
    assert var_45 is True
    var_46 = b'!!!invalid-base64!!!'
    var_47 = var_3.verify_signature(var_29, var_46)
    assert var_47 is False
    var_48 = b'sha256-test'
    var_49 = 'concat'
    var_50 = module_0.Signer(var_26, key_derivation=var_49)
    var_51 = b'concat-test'
    var_52 = var_50.get_signature(var_51)
    var_53 = var_50.verify_signature(var_51, var_52)
    assert var_53 is True
    var_54 = 'hmac'
    var_55 = module_0.Signer(var_26, key_derivation=var_54)
    var_56 = b'hmac-test'
    var_57 = var_55.get_signature(var_56)
    var_58 = var_55.verify_signature(var_56, var_57)
    assert var_58 is True
    var_59 = 'none'
    var_60 = module_0.Signer(var_26, key_derivation=var_59)
    var_61 = b'none-derivation-test'
    var_62 = var_60.get_signature(var_61)
    var_63 = var_60.verify_signature(var_61, var_62)
    assert var_63 is True
    var_64 = None
    var_65 = module_0.Signer(var_26, var_64)
    var_66 = b'none-salt-test'
    var_67 = var_65.get_signature(var_66)
    var_68 = var_65.verify_signature(var_66, var_67)
    assert var_68 is True
    var_69 = b'original-value'
    var_70 = var_3.get_signature(var_69)
    var_71 = b'different-value'
    var_72 = var_3.verify_signature(var_71, var_70)
    assert var_72 is False



# Parsed testcases at query #83
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret-key'
    var_2 = 'salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_3.verify_signature(var_4, var_8)
    assert var_9 is False
    var_10 = b'wrong-value'
    var_11 = var_3.verify_signature(var_10, var_5)
    assert var_11 is False
    var_12 = '!!!invalid-base64!!!'
    var_13 = var_3.verify_signature(var_4, var_12)
    assert var_13 is False
    var_14 = b''
    var_15 = var_3.verify_signature(var_4, var_14)
    assert var_15 is False
    var_16 = module_0.NoneAlgorithm()
    var_17 = module_0.Signer(var_1, algorithm=var_16)
    var_18 = var_17.get_signature(var_4)
    var_19 = var_17.verify_signature(var_4, var_18)
    assert var_19 is True
    var_20 = b''
    var_21 = var_3.get_signature(var_20)
    var_22 = var_3.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = var_3.verify_signature(var_20, var_7)
    assert var_23 is False
    var_24 = 'old-key'
    var_25 = 'new-key'
    var_26 = [var_24, var_25]
    var_27 = module_0.Signer(var_26, var_2)
    var_28 = var_27.sign(var_4)
    var_29 = var_27.get_signature(var_4)
    var_30 = var_27.verify_signature(var_4, var_29)
    assert var_30 is True
    var_31 = 'test-value'
    var_32 = var_3.verify_signature(var_31, var_5)
    assert var_32 is True
    var_33 = 'invalid'
    var_34 = var_3.verify_signature(var_31, var_33)
    assert var_34 is False



# Parsed testcases at query #84
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
    var_6 = 'test-string'
    var_7 = var_2.get_signature(var_6)
    var_8 = var_2.verify_signature(var_6, var_7)
    assert var_8 is True
    var_9 = b'fake-signature'
    var_10 = module_1.base64_encode(var_9)
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = 'invalid-base64!!'
    var_13 = var_2.verify_signature(var_3, var_12)
    assert var_13 is False
    var_14 = b''
    var_15 = var_2.verify_signature(var_3, var_14)
    assert var_15 is False
    var_16 = var_2.get_signature(var_14)
    var_17 = var_2.verify_signature(var_14, var_16)
    assert var_17 is True
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = 'rotation-salt'
    var_22 = module_0.Signer(var_20, var_21)
    var_23 = b'rotation-test'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = 'concat-salt'
    var_27 = 'concat'
    var_28 = module_0.Signer(var_0, var_26, key_derivation=var_27)
    var_29 = b'concat-test'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = 'hmac-salt'
    var_33 = 'hmac'
    var_34 = module_0.Signer(var_0, var_32, key_derivation=var_33)
    var_35 = b'hmac-test'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = 'none-salt'
    var_39 = 'none'
    var_40 = module_0.Signer(var_0, var_38, key_derivation=var_39)
    var_41 = b'none-test'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = module_0.NoneAlgorithm()
    var_45 = module_0.Signer(var_0, algorithm=var_44)
    var_46 = b'none-alg-test'
    var_47 = var_45.get_signature(var_46)
    var_48 = var_45.verify_signature(var_46, var_47)
    assert var_48 is True
    var_49 = var_2.get_signature(var_3)
    var_50 = module_1.base64_decode(var_49)
    var_51 = var_2.verify_signature(var_3, var_50)
    assert var_51 is False
    var_52 = var_2.verify_signature(var_3, var_49)
    assert var_52 is True



# Parsed testcases at query #85
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid-signature'
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False
    var_9 = 'test-string'
    var_10 = var_3.get_signature(var_9)
    var_11 = var_3.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b''
    var_13 = var_3.get_signature(var_12)
    var_14 = var_3.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'key1'
    var_16 = 'salt1'
    var_17 = module_0.Signer(var_15, var_16)
    var_18 = 'key2'
    var_19 = module_0.Signer(var_18, var_16)
    var_20 = b'test'
    var_21 = var_17.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is False
    var_23 = 'old-key'
    var_24 = 'new-key'
    var_25 = [var_23, var_24]
    var_26 = module_0.Signer(var_25, var_2)
    var_27 = b'test-value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'!!!invalid-base64!!!'
    var_31 = var_3.verify_signature(var_4, var_30)
    assert var_31 is False
    var_32 = module_0.NoneAlgorithm()
    var_33 = 'key'
    var_34 = 'salt'
    var_35 = module_0.Signer(var_33, var_34, algorithm=var_32)
    var_36 = b'test-value'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = b''
    var_40 = var_35.verify_signature(var_36, var_39)
    assert var_40 is True
    var_41 = b'anything'
    var_42 = var_35.verify_signature(var_36, var_41)
    assert var_42 is False



# Parsed testcases at query #86
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
    var_7 = b'wrong-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'!!!invalid-base64!!!'
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = b''
    var_15 = var_1.verify_signature(var_2, var_14)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'test-rotation'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'string-value'
    var_24 = var_1.get_signature(var_23)
    var_25 = var_1.verify_signature(var_23, var_24)
    assert var_25 is True



# Parsed testcases at query #87
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = b'!!!invalid-base64!!!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'test-value'
    var_12 = var_1.verify_signature(var_11, var_3)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'test-value-2'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'secret'
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_20, algorithm=var_21)
    var_23 = b'test'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = var_22.verify_signature(var_23, var_7)
    assert var_26 is True



# Parsed testcases at query #88
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b'tampered value'
    var_9 = var_1.verify_signature(var_8, var_3)
    assert var_9 is False
    var_10 = b'not-base64!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b''
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'rotation test'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = module_0.Signer(var_15)
    var_23 = var_22.get_signature(var_19)
    var_24 = var_18.verify_signature(var_19, var_23)
    assert var_24 is True
    var_25 = 'different-salt'
    var_26 = module_0.Signer(var_0, var_25)
    var_27 = var_26.get_signature(var_2)
    var_28 = var_26.verify_signature(var_2, var_27)
    assert var_28 is True
    var_29 = var_1.verify_signature(var_2, var_27)
    assert var_29 is False
    var_30 = b'-'
    var_31 = module_0.Signer(var_0, sep=var_30)
    var_32 = var_31.get_signature(var_2)
    var_33 = var_31.verify_signature(var_2, var_32)
    assert var_33 is True
    var_34 = var_1.verify_signature(var_2, var_32)
    assert var_34 is False
    var_35 = module_0.NoneAlgorithm()
    var_36 = module_0.Signer(var_0, algorithm=var_35)
    var_37 = var_36.get_signature(var_2)
    var_38 = var_36.verify_signature(var_2, var_37)
    assert var_38 is True



# Parsed testcases at query #89
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'invalid-sig'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = b'modified-value'
    var_12 = var_1.verify_signature(var_11, var_3)
    assert var_12 is False
    var_13 = module_0.NoneAlgorithm()
    var_14 = module_0.Signer(var_0, algorithm=var_13)
    var_15 = b'test'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = b'test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = 'secret'
    var_26 = b'salt1'
    var_27 = module_0.Signer(var_25, var_26)
    var_28 = b'salt2'
    var_29 = module_0.Signer(var_25, var_28)
    var_30 = b'test'
    var_31 = var_27.get_signature(var_30)
    var_32 = var_27.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = var_29.verify_signature(var_30, var_31)
    assert var_33 is False
    var_34 = b'|'
    var_35 = module_0.Signer(var_25, sep=var_34)
    var_36 = b'test'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = 'secret'
    var_40 = b'test'
    var_41 = var_1.get_signature(var_40)
    var_42 = var_1.verify_signature(var_40, var_41)
    assert var_42 is True
    var_43 = b'test'
    var_44 = b'!!!invalid-base64!!!'
    var_45 = var_1.verify_signature(var_43, var_44)
    assert var_45 is False



# Parsed testcases at query #90
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test that verify_signature returns True for valid signatures and False for invalid ones.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'wrong-value'
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is False
    var_11 = 'old-key'
    var_12 = 'new-key'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = module_0.Signer(var_11)
    var_19 = var_18.get_signature(var_15)
    var_20 = var_14.verify_signature(var_15, var_19)
    assert var_20 is True
    var_21 = 'secret'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = b'test-value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = 'not-base64!!'
    var_28 = var_2.verify_signature(var_24, var_27)
    assert var_28 is False
    var_29 = b''
    var_30 = var_2.get_signature(var_29)
    var_31 = var_2.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = var_2.verify_signature(var_24, var_30)
    assert var_32 is True
    var_33 = 'ascii'



# Parsed testcases at query #91
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
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'old-secret'
    var_12 = 'new-secret'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'rotated-value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = module_0.Signer(var_11)
    var_19 = var_18.get_signature(var_15)
    var_20 = var_14.verify_signature(var_15, var_19)
    assert var_20 is True
    var_21 = 'secret'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = b'test-none'
    var_25 = var_23.get_signature(var_24)
    var_26 = b''
    var_27 = module_1.base64_encode(var_26)
    var_28 = var_23.verify_signature(var_24, var_25)
    assert var_28 is True
    var_29 = b'not-base64!!'
    var_30 = var_1.verify_signature(var_2, var_29)
    assert var_30 is False
    var_31 = var_1.verify_signature(var_2, var_3)
    assert var_31 is True



# Parsed testcases at query #92
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = 'test string'
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = b''
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'different-salt'
    var_14 = module_0.Signer(var_0, var_13)
    var_15 = var_14.get_signature(var_2)
    var_16 = var_14.verify_signature(var_2, var_15)
    assert var_16 is True
    var_17 = var_1.verify_signature(var_2, var_15)
    assert var_17 is False
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = var_21.get_signature(var_2)
    var_23 = var_21.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = b'!!!invalid-base64!!!'
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = 'secret'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = var_28.get_signature(var_2)
    var_30 = var_28.verify_signature(var_2, var_29)
    assert var_30 is True
    var_31 = b'bytes-key'
    var_32 = module_0.Signer(var_31)
    var_33 = b'bytes-value'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = b'a'
    var_37 = 100
    var_38 = var_36 * var_37
    var_39 = var_1.verify_signature(var_2, var_38)
    assert var_39 is False



# Parsed testcases at query #93
#--------------------------




# Parsed testcases at query #94
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = var_1.verify_signature(var_2, var_7)
    assert var_9 is False
    var_10 = 'different-key'
    var_11 = module_0.Signer(var_10)
    var_12 = var_11.get_signature(var_2)
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = var_17.get_signature(var_2)
    var_19 = var_17.verify_signature(var_2, var_18)
    assert var_19 is True
    var_20 = module_0.NoneAlgorithm()
    var_21 = module_0.Signer(var_0, algorithm=var_20)
    var_22 = var_21.get_signature(var_2)
    assert var_22 == b''
    var_23 = var_21.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = '!!!invalid-base64!!!'
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = 'test-value'
    var_27 = var_1.verify_signature(var_26, var_3)
    assert var_27 is True
    var_28 = 'ascii'
    var_29 = 'different-salt'
    var_30 = module_0.Signer(var_0, var_29)
    var_31 = var_30.get_signature(var_2)
    var_32 = var_1.verify_signature(var_2, var_31)
    assert var_32 is False



# Parsed testcases at query #95
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'test-value'
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b''
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = b'.'
    var_15 = module_0.Signer(var_0, sep=var_14)
    var_16 = b'test-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'custom-salt'
    var_20 = module_0.Signer(var_0, var_19)
    var_21 = b'test-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_0, algorithm=var_24)
    var_26 = b'test-value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'old-key'
    var_30 = 'new-key'
    var_31 = [var_29, var_30]
    var_32 = module_0.Signer(var_31)
    var_33 = b'test-value'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = b'test'
    var_37 = b'!!!invalid-base64!!!'
    var_38 = var_1.verify_signature(var_36, var_37)
    assert var_38 is False
    var_39 = var_1.verify_signature(var_36, var_11)
    assert var_39 is False



# Parsed testcases at query #96
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test_value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid_signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'different_value'
    var_12 = var_2.verify_signature(var_11, var_4)
    assert var_12 is False
    var_13 = -1
    var_14 = var_4[:var_13]
    var_15 = b'x'
    var_16 = var_14 + var_15
    var_17 = var_2.verify_signature(var_3, var_16)
    assert var_17 is False
    var_18 = 'test_string'
    var_19 = var_2.get_signature(var_18)
    var_20 = var_2.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'old-key'
    var_22 = 'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23, var_1)
    var_25 = var_24.get_signature(var_3)
    var_26 = var_24.verify_signature(var_3, var_25)
    assert var_26 is True
    var_27 = b'!!!invalid_base64!!!'
    var_28 = var_2.verify_signature(var_3, var_27)
    assert var_28 is False
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_0, var_1, algorithm=var_29)
    var_31 = var_30.get_signature(var_3)
    var_32 = var_30.verify_signature(var_3, var_31)
    assert var_32 is True
    var_33 = 'concat'
    var_34 = module_0.Signer(var_0, var_1, key_derivation=var_33)
    var_35 = var_34.get_signature(var_3)
    var_36 = var_34.verify_signature(var_3, var_35)
    assert var_36 is True



# Parsed testcases at query #97
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
    var_10 = module_0.NoneAlgorithm()
    var_11 = module_0.Signer(var_0, var_1, algorithm=var_10)
    var_12 = b'test-value'
    var_13 = var_11.get_signature(var_12)
    var_14 = var_11.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = b''
    var_16 = var_11.verify_signature(var_12, var_15)
    assert var_16 is True
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19, var_1)
    var_21 = b'test-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'!!!invalid-base64!!!'
    var_25 = var_2.verify_signature(var_21, var_24)
    assert var_25 is False
    var_26 = 'test-value'
    var_27 = var_2.verify_signature(var_26, var_22)
    assert var_27 is True
    var_28 = var_2.get_signature(var_15)
    var_29 = var_2.verify_signature(var_15, var_28)
    assert var_29 is True
    var_30 = var_2.verify_signature(var_15, var_15)
    assert var_30 is False



# Parsed testcases at query #98
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
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'test-value'
    var_12 = var_2.get_signature(var_11)
    var_13 = var_2.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'not-base64!@#'
    var_15 = var_2.verify_signature(var_3, var_14)
    assert var_15 is False
    var_16 = b''
    var_17 = var_2.verify_signature(var_3, var_16)
    assert var_17 is False
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_0, algorithm=var_18)
    var_20 = b'test-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'old-key'
    var_24 = 'new-key'
    var_25 = [var_23, var_24]
    var_26 = module_0.Signer(var_25, var_1)
    var_27 = b'test-value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True



# Parsed testcases at query #99
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'not-base64!@#'
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = b''
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'different-secret'
    var_16 = module_0.Signer(var_15)
    var_17 = var_16.get_signature(var_2)
    var_18 = var_1.verify_signature(var_2, var_17)
    assert var_18 is False
    var_19 = 'secret'
    var_20 = module_0.NoneAlgorithm()
    var_21 = module_0.Signer(var_19, algorithm=var_20)
    var_22 = var_21.get_signature(var_2)
    var_23 = var_21.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = 'old-key'
    var_25 = 'new-key'
    var_26 = [var_24, var_25]
    var_27 = module_0.Signer(var_26)
    var_28 = b'rotation-test'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = module_0.Signer(var_24)
    var_32 = var_31.get_signature(var_28)
    var_33 = var_27.verify_signature(var_28, var_32)
    assert var_33 is True
    var_34 = 'unrelated-key'
    var_35 = module_0.Signer(var_34)
    var_36 = var_35.get_signature(var_28)
    var_37 = var_27.verify_signature(var_28, var_36)
    assert var_37 is False
    var_38 = 'test-value'
    var_39 = var_1.verify_signature(var_38, var_3)
    assert var_39 is True
    var_40 = var_1.verify_signature(var_38, var_5)
    assert var_40 is False
    var_41 = 'custom-salt'
    var_42 = module_0.Signer(var_19, var_41)
    var_43 = var_42.get_signature(var_2)
    var_44 = var_42.verify_signature(var_2, var_43)
    assert var_44 is True
    var_45 = module_0.Signer(var_19)
    var_46 = var_45.verify_signature(var_2, var_43)
    assert var_46 is False
    var_47 = 'concat'
    var_48 = module_0.Signer(var_19, key_derivation=var_47)
    var_49 = var_48.get_signature(var_2)
    var_50 = var_48.verify_signature(var_2, var_49)
    assert var_50 is True
    var_51 = 'hmac'
    var_52 = module_0.Signer(var_19, key_derivation=var_51)
    var_53 = var_52.get_signature(var_2)
    var_54 = var_52.verify_signature(var_2, var_53)
    assert var_54 is True
    var_55 = 'none'
    var_56 = module_0.Signer(var_19, key_derivation=var_55)
    var_57 = var_56.get_signature(var_2)
    var_58 = var_56.verify_signature(var_2, var_57)
    assert var_58 is True



# Parsed testcases at query #100
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
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'not-valid-base64'
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = b'test-value-with-special-chars!@#$%^&*()'
    var_14 = var_1.get_signature(var_13)
    var_15 = var_1.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'test-rotation'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.Signer(var_16)
    var_24 = var_23.get_signature(var_20)
    var_25 = var_19.verify_signature(var_20, var_24)
    assert var_25 is True
    var_26 = 'different-key'
    var_27 = module_0.Signer(var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_1.verify_signature(var_2, var_28)
    assert var_29 is False
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_0, algorithm=var_30)
    var_32 = var_31.get_signature(var_2)
    var_33 = b''
    var_34 = module_1.base64_encode(var_33)
    var_35 = var_31.verify_signature(var_2, var_32)
    assert var_35 is True
    var_36 = 'ascii'
    var_37 = 'invalid'
    var_38 = var_1.verify_signature(var_2, var_37)
    assert var_38 is False



# Parsed testcases at query #101
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
    var_7 = 'ascii'
    var_8 = 'utf-8'
    var_9 = b'!!!invalid-base64!!!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = b''
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = var_17.get_signature(var_2)
    var_19 = var_17.verify_signature(var_2, var_18)
    assert var_19 is True
    var_20 = module_0.Signer(var_14)
    var_21 = var_20.get_signature(var_2)
    var_22 = var_17.verify_signature(var_2, var_21)
    assert var_22 is True
    var_23 = 'custom-salt'
    var_24 = module_0.Signer(var_0, var_23)
    var_25 = var_24.get_signature(var_2)
    var_26 = var_24.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = var_1.verify_signature(var_2, var_25)
    assert var_27 is False
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_0, algorithm=var_28)
    var_30 = var_29.get_signature(var_2)
    assert var_30 == b''
    var_31 = var_29.verify_signature(var_2, var_30)
    assert var_31 is True
    var_32 = b'|'
    var_33 = module_0.Signer(var_0, sep=var_32)
    var_34 = b'test|value'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = 'héllo wörld'
    var_38 = var_1.get_signature(var_37)
    var_39 = var_1.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = 256
    var_41 = range(var_40)
    var_42 = bytes(var_41)
    var_43 = var_1.get_signature(var_42)
    var_44 = var_1.verify_signature(var_42, var_43)
    assert var_44 is True
    var_45 = b''
    var_46 = var_1.verify_signature(var_2, var_45)
    assert var_46 is False



# Parsed testcases at query #102
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'!!!invalid base64!!!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = module_0.NoneAlgorithm()
    var_12 = module_0.Signer(var_0, algorithm=var_11)
    var_13 = b'test value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'test value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'test value'
    var_24 = module_0.Signer(var_0)
    var_25 = 'test string'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'utf-8'
    var_29 = module_0.Signer(var_0)
    var_30 = var_29.get_signature(var_9)
    var_31 = var_29.verify_signature(var_9, var_30)
    assert var_31 is True



# Parsed testcases at query #103
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'test value'
    var_6 = b'invalid_sig'
    var_7 = var_1.verify_signature(var_5, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_5, var_8)
    assert var_9 is False
    var_10 = b'!!!invalid!!!'
    var_11 = var_1.verify_signature(var_5, var_10)
    assert var_11 is False
    var_12 = 'ascii'
    var_13 = 'test value'
    var_14 = var_1.verify_signature(var_13, var_3)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'test with rotation'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_0, algorithm=var_22)
    var_24 = b'test with none algorithm'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'test with hmac sha256'
    var_28 = b'modified value'
    var_29 = var_1.verify_signature(var_28, var_25)
    assert var_29 is False
    var_30 = 'different-salt'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = b'test with different salt'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_1.verify_signature(var_32, var_33)
    assert var_34 is False
    var_35 = var_31.verify_signature(var_32, var_33)
    assert var_35 is True



# Parsed testcases at query #104
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid_sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'string value'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = b'!!!invalid_base64!!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'rotation test'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = b'custom-salt'
    var_23 = module_0.Signer(var_0, var_22)
    var_24 = b'salted value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'different-salt'
    var_28 = module_0.Signer(var_0, var_27)
    var_29 = var_28.verify_signature(var_24, var_25)
    assert var_29 is False
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_0, algorithm=var_30)
    var_32 = b'none algorithm'
    var_33 = var_31.get_signature(var_32)
    assert var_33 == b''
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = bytearray(var_3)
    var_36 = -1
    var_37 = var_35[var_36]
    var_38 = 255
    var_39 = var_37 ^ var_38
    var_40 = bytes(var_35)
    var_41 = var_1.verify_signature(var_2, var_40)
    assert var_41 is False
    var_42 = b''
    var_43 = module_0.Signer(var_42)
    var_44 = b'test'
    var_45 = var_43.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = 'hmac'
    var_48 = module_0.Signer(var_0, key_derivation=var_47)
    var_49 = b'hmac test'
    var_50 = var_48.get_signature(var_49)
    var_51 = var_48.verify_signature(var_49, var_50)
    assert var_51 is True
    var_52 = 'concat'
    var_53 = module_0.Signer(var_0, key_derivation=var_52)
    var_54 = b'concat test'
    var_55 = var_53.get_signature(var_54)
    var_56 = var_53.verify_signature(var_54, var_55)
    assert var_56 is True
    var_57 = 'none'
    var_58 = module_0.Signer(var_0, key_derivation=var_57)
    var_59 = b'none derivation'
    var_60 = var_58.get_signature(var_59)
    var_61 = var_58.verify_signature(var_59, var_60)
    assert var_61 is True
    var_62 = b'wrong value'
    var_63 = var_1.verify_signature(var_62, var_3)
    assert var_63 is False
    var_64 = b'bytes-key'
    var_65 = module_0.Signer(var_64)
    var_66 = b'bytes key test'
    var_67 = var_65.get_signature(var_66)
    var_68 = var_65.verify_signature(var_66, var_67)
    assert var_68 is True
    var_69 = 'string-key'
    var_70 = module_0.Signer(var_69)
    var_71 = b'string key test'
    var_72 = var_70.get_signature(var_71)
    var_73 = var_70.verify_signature(var_71, var_72)
    assert var_73 is True
    var_74 = b'key1'
    var_75 = b'key2'
    var_76 = [var_74, var_75]
    var_77 = module_0.Signer(var_76)
    var_78 = b'iterable bytes test'
    var_79 = var_77.get_signature(var_78)
    var_80 = var_77.verify_signature(var_78, var_79)
    assert var_80 is True
    var_81 = 'old'
    var_82 = 'newest'
    var_83 = [var_81, var_82]
    var_84 = module_0.Signer(var_83)
    var_85 = b'rotation verify'
    var_86 = var_84.get_signature(var_85)
    var_87 = var_84.verify_signature(var_85, var_86)
    assert var_87 is True
    var_88 = b'sha256 test'



# Parsed testcases at query #105
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'modified-value'
    var_10 = var_2.verify_signature(var_9, var_4)
    assert var_10 is False
    var_11 = module_0.Signer(var_1)
    var_12 = 'test-string-value'
    var_13 = var_11.get_signature(var_12)
    var_14 = var_11.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'test-value'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = b'not-base64!@#'
    var_23 = var_18.verify_signature(var_19, var_22)
    assert var_23 is False
    var_24 = b''
    var_25 = var_18.verify_signature(var_19, var_24)
    assert var_25 is False
    var_26 = 'test-key'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = b'test-value'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'test-value'
    var_33 = 'salt1'
    var_34 = module_0.Signer(var_26, var_33)
    var_35 = 'salt2'
    var_36 = module_0.Signer(var_26, var_35)
    var_37 = b'test-value'
    var_38 = var_34.get_signature(var_37)
    var_39 = var_36.get_signature(var_37)
    var_40 = var_36.verify_signature(var_37, var_38)
    assert var_40 is False
    var_41 = var_34.verify_signature(var_37, var_39)
    assert var_41 is False
    var_42 = b'-'
    var_43 = module_0.Signer(var_26, sep=var_42)
    var_44 = b'test-value'
    var_45 = var_43.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = 'test-key'
    var_48 = b'test-value'
    var_49 = var_18.get_signature(var_48)
    var_50 = var_18.verify_signature(var_48, var_49)
    assert var_50 is True



# Parsed testcases at query #106
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'test string'
    var_12 = var_1.verify_signature(var_11, var_3)
    assert var_12 is False
    var_13 = 'not-base64!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'test with rotation'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = module_0.Signer(var_15)
    var_23 = var_22.verify_signature(var_19, var_20)
    assert var_23 is True
    var_24 = 'wrong-key'
    var_25 = module_0.Signer(var_24)
    var_26 = var_25.verify_signature(var_19, var_20)
    assert var_26 is False
    var_27 = 'salt1'
    var_28 = module_0.Signer(var_0, var_27)
    var_29 = 'salt2'
    var_30 = module_0.Signer(var_0, var_29)
    var_31 = b'test with salt'
    var_32 = var_28.get_signature(var_31)
    var_33 = var_28.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = var_30.verify_signature(var_31, var_32)
    assert var_34 is False
    var_35 = b'|'
    var_36 = module_0.Signer(var_0, sep=var_35)
    var_37 = b'test with custom sep'
    var_38 = var_36.get_signature(var_37)
    var_39 = module_1.base64_encode(var_38)
    var_40 = var_36.verify_signature(var_37, var_39)
    assert var_40 is True
    var_41 = module_0.NoneAlgorithm()
    var_42 = module_0.Signer(var_0, algorithm=var_41)
    var_43 = b'test no signature'
    var_44 = var_42.get_signature(var_43)
    assert var_44 == b''
    var_45 = var_42.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = module_1.base64_decode(var_3)
    var_47 = var_1.verify_signature(var_2, var_46)
    assert var_47 is True
    var_48 = b'test sha256'



# Parsed testcases at query #107
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-rotated-value'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'secret'
    var_18 = module_0.NoneAlgorithm()
    var_19 = module_0.Signer(var_17, var_1, algorithm=var_18)
    var_20 = b'test-none'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = var_19.verify_signature(var_20, var_8)
    assert var_23 is True
    var_24 = b'test-hmac'
    var_25 = b'wrong-signature'
    var_26 = 'secret'
    var_27 = 'salt'
    var_28 = b'test-derivation'
    var_29 = b'fake-sig'



# Parsed testcases at query #108
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'invalid-sig'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = module_0.NoneAlgorithm()
    var_12 = module_0.Signer(var_0, algorithm=var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_12.verify_signature(var_2, var_13)
    assert var_14 is True
    var_15 = b'something'
    var_16 = var_12.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'old-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = 'different-key'
    var_25 = module_0.Signer(var_24)
    var_26 = var_25.get_signature(var_2)
    var_27 = var_1.verify_signature(var_2, var_26)
    assert var_27 is False
    var_28 = b'!!!invalid-base64!!!'
    var_29 = var_1.verify_signature(var_2, var_28)
    assert var_29 is False
    var_30 = 'custom-salt'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = var_31.get_signature(var_2)
    var_33 = var_31.verify_signature(var_2, var_32)
    assert var_33 is True
    var_34 = var_1.verify_signature(var_2, var_32)
    assert var_34 is False
    var_35 = b'-'
    var_36 = module_0.Signer(var_0, sep=var_35)
    var_37 = var_36.get_signature(var_2)
    var_38 = var_36.verify_signature(var_2, var_37)
    assert var_38 is True



# Parsed testcases at query #109
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'invalid-sig'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'test-value-2'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'custom-salt'
    var_20 = module_0.Signer(var_0, var_19)
    var_21 = b'test-value-3'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_0, algorithm=var_24)
    var_26 = b'test-value-4'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'!!!invalid-base64!!!'
    var_30 = var_1.verify_signature(var_2, var_29)
    assert var_30 is False
    var_31 = b'x'
    var_32 = 10000
    var_33 = var_31 * var_32
    var_34 = var_1.get_signature(var_33)
    var_35 = var_1.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = 'héllo wörld'
    var_37 = var_1.get_signature(var_36)
    var_38 = var_1.verify_signature(var_36, var_37)
    assert var_38 is True



# Parsed testcases at query #110
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
    var_7 = 'old-key'
    var_8 = 'new-key'
    var_9 = [var_7, var_8]
    var_10 = module_0.Signer(var_9)
    var_11 = b'test-old'
    var_12 = var_10.get_signature(var_11)
    var_13 = var_10.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'custom-salt'
    var_15 = module_0.Signer(var_0, var_14)
    var_16 = b'test-salt'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = var_1.verify_signature(var_16, var_17)
    assert var_19 is False
    var_20 = 'string-value'
    var_21 = var_1.get_signature(var_20)
    var_22 = var_1.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'test'
    var_24 = b'!!!invalid-base64!!!'
    var_25 = var_1.verify_signature(var_23, var_24)
    assert var_25 is False
    var_26 = b'test-sha256'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, algorithm=var_27)
    var_29 = b'test-none'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = 'secret-key'
    var_33 = b'test-derivation'
    var_34 = b''
    var_35 = var_1.get_signature(var_34)
    var_36 = var_1.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = 'different-secret'
    var_38 = module_0.Signer(var_37)
    var_39 = b'test-value2'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_1.verify_signature(var_39, var_40)
    assert var_41 is False



# Parsed testcases at query #111
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
    var_8 = b''
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'different-value'
    var_11 = var_2.verify_signature(var_10, var_4)
    assert var_11 is False
    var_12 = module_0.NoneAlgorithm()
    var_13 = module_0.Signer(var_0, algorithm=var_12)
    var_14 = b'test-value'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19, var_1)
    var_21 = b'test-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'!!!invalid-base64!!!'
    var_25 = var_2.verify_signature(var_21, var_24)
    assert var_25 is False
    var_26 = b'test-value'
    var_27 = var_2.verify_signature(var_26, var_22)
    assert var_27 is True
    var_28 = 'test-value'
    var_29 = var_2.verify_signature(var_28, var_22)
    assert var_29 is True



# Parsed testcases at query #112
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'test-string'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'wrong-key'
    var_14 = module_0.Signer(var_13)
    var_15 = var_14.get_signature(var_2)
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'test2'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.Signer(var_17)
    var_25 = var_24.get_signature(var_21)
    var_26 = var_20.verify_signature(var_21, var_25)
    assert var_26 is True
    var_27 = b'!!!invalid-base64!!!'
    var_28 = var_1.verify_signature(var_2, var_27)
    assert var_28 is False
    var_29 = 'key'
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_29, algorithm=var_30)
    var_32 = b'test3'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = b''
    var_36 = var_31.verify_signature(var_32, var_35)
    assert var_36 is True
    var_37 = b'anything'
    var_38 = var_31.verify_signature(var_32, var_37)
    assert var_38 is False



# Parsed testcases at query #113
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = 'old-key'
    var_9 = 'new-key'
    var_10 = [var_8, var_9]
    var_11 = module_0.Signer(var_10)
    var_12 = b'test-value-2'
    var_13 = var_11.get_signature(var_12)
    var_14 = var_11.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'secret'
    var_16 = module_0.NoneAlgorithm()
    var_17 = module_0.Signer(var_15, algorithm=var_16)
    var_18 = b'test-value-3'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = b'!!!invalid-base64!!!'
    var_22 = var_1.verify_signature(var_2, var_21)
    assert var_22 is False
    var_23 = 'string-value'
    var_24 = var_1.verify_signature(var_23, var_3)
    assert var_24 is True
    var_25 = b''
    var_26 = var_1.get_signature(var_25)
    var_27 = var_1.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'different-salt'
    var_29 = module_0.Signer(var_15, var_28)
    var_30 = b'test-value-4'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = var_1.verify_signature(var_30, var_31)
    assert var_33 is False



# Parsed testcases at query #114
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test that verify_signature correctly validates signatures.'
    var_1 = b'test-secret-key-12345'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'different-value'
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = b'not-base64!!!'
    var_13 = var_2.verify_signature(var_3, var_12)
    assert var_13 is False
    var_14 = b''
    var_15 = var_2.verify_signature(var_3, var_14)
    assert var_15 is False
    var_16 = b'test-value'
    var_17 = var_2.verify_signature(var_16, var_4)
    assert var_17 is True
    var_18 = 'test-value'
    var_19 = var_2.verify_signature(var_18, var_4)
    assert var_19 is True
    var_20 = b'old-secret-key'
    var_21 = [var_20, var_1]
    var_22 = module_0.Signer(var_21)
    var_23 = module_0.Signer(var_20)
    var_24 = var_23.get_signature(var_3)
    var_25 = var_22.verify_signature(var_3, var_24)
    assert var_25 is True
    var_26 = b'newer-secret-key'
    var_27 = [var_1, var_26]
    var_28 = module_0.Signer(var_27)
    var_29 = var_28.verify_signature(var_3, var_24)
    assert var_29 is False
    var_30 = b'different-salt'
    var_31 = module_0.Signer(var_1, var_30)
    var_32 = var_31.get_signature(var_3)
    var_33 = var_2.verify_signature(var_3, var_32)
    assert var_33 is False
    var_34 = module_0.HMACAlgorithm()
    var_35 = module_0.Signer(var_1, algorithm=var_34)
    var_36 = var_35.get_signature(var_3)
    var_37 = var_35.verify_signature(var_3, var_36)
    assert var_37 is True
    var_38 = module_0.NoneAlgorithm()
    var_39 = module_0.Signer(var_1, algorithm=var_38)
    var_40 = var_39.get_signature(var_3)
    var_41 = var_39.verify_signature(var_3, var_40)
    assert var_41 is True
    var_42 = module_1.base64_encode(var_14)



# Parsed testcases at query #115
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
    var_10 = 'test-value'
    var_11 = 'invalid-sig'
    var_12 = var_2.verify_signature(var_10, var_11)
    assert var_12 is False
    var_13 = b''
    var_14 = var_2.get_signature(var_13)
    var_15 = var_2.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = b'empty-sig'
    var_17 = var_2.verify_signature(var_13, var_16)
    assert var_17 is False
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20, var_1)
    var_22 = b'test-value'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'!!!invalid-base64!!!'
    var_26 = var_2.verify_signature(var_22, var_25)
    assert var_26 is False
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, algorithm=var_27)
    var_29 = var_28.get_signature(var_22)
    var_30 = var_28.verify_signature(var_22, var_29)
    assert var_30 is True
    var_31 = b'wrong-sig'
    var_32 = var_2.verify_signature(var_10, var_4)
    assert var_32 is True



# Parsed testcases at query #116
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'non-empty'
    var_12 = var_2.verify_signature(var_11, var_9)
    assert var_12 is False
    var_13 = b'!!!invalid-base64!!!'
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = b'test'
    var_16 = var_2.verify_signature(var_15, var_4)
    assert var_16 is True
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19, var_1)
    var_21 = b'test value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'modified value'
    var_25 = var_2.verify_signature(var_24, var_4)
    assert var_25 is False
    var_26 = 'secret'
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_26, algorithm=var_27)
    var_29 = b'test'
    var_30 = b''
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'any'
    var_33 = var_28.verify_signature(var_29, var_32)
    assert var_33 is False



# Parsed testcases at query #117
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = 'test-value'
    var_6 = var_1.get_signature(var_5)
    var_7 = var_1.verify_signature(var_5, var_6)
    assert var_7 is True
    var_8 = b'invalid-sig'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = bytearray(var_3)
    var_13 = 0
    var_14 = var_12[var_13]
    var_15 = 1
    var_16 = var_14 ^ var_15
    var_17 = bytes(var_12)
    var_18 = var_1.verify_signature(var_2, var_17)
    assert var_18 is False
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_13, algorithm=var_19)
    var_21 = b'test-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = 'old-key'
    var_25 = 'new-key'
    var_26 = [var_24, var_25]
    var_27 = module_0.Signer(var_26)
    var_28 = b'test-value'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = module_0.Signer(var_24)
    var_32 = var_31.get_signature(var_28)
    var_33 = var_27.verify_signature(var_28, var_32)
    assert var_33 is True
    var_34 = b'test-value'
    var_35 = b'|'
    var_36 = module_0.Signer(var_13, sep=var_35)
    var_37 = b'test-value'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = None
    var_41 = module_0.Signer(var_13, var_40)
    var_42 = b'test-value'
    var_43 = var_41.get_signature(var_42)
    var_44 = var_41.verify_signature(var_42, var_43)
    assert var_44 is True
    var_45 = 'concat'
    var_46 = module_0.Signer(var_13, key_derivation=var_45)
    var_47 = b'test-value'
    var_48 = var_46.get_signature(var_47)
    var_49 = var_46.verify_signature(var_47, var_48)
    assert var_49 is True
    var_50 = module_0.NoneAlgorithm()
    var_51 = module_0.Signer(var_13, algorithm=var_50)
    var_52 = b'anything'
    var_53 = var_51.verify_signature(var_52, var_10)
    assert var_53 is True
    var_54 = module_0.Signer(var_13)
    var_55 = b'test'
    var_56 = b'!!!invalid_base64!!!'
    var_57 = var_54.verify_signature(var_55, var_56)
    assert var_57 is False



# Parsed testcases at query #118
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is False
    var_7 = -1
    var_8 = var_3[:var_7]
    var_9 = -1
    var_10 = var_3[var_9:]
    var_11 = b'\x00'
    var_12 = var_10 != var_11
    var_13 = b'\x01'
    var_14 = var_11 if var_12 else var_13
    var_15 = var_8 + var_14
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'test-value-2'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'|'
    var_25 = module_0.Signer(var_0, sep=var_24)
    var_26 = b'test-value-3'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_0, algorithm=var_29)
    var_31 = b'test-value-4'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b'not-base64!!'
    var_35 = var_1.verify_signature(var_2, var_34)
    assert var_35 is False
    var_36 = b''
    var_37 = var_1.verify_signature(var_36, var_3)
    assert var_37 is False
    var_38 = var_1.verify_signature(var_2, var_3)
    assert var_38 is True
    var_39 = 'test-value'
    var_40 = var_1.verify_signature(var_39, var_3)
    assert var_40 is True
    var_41 = 'ascii'



# Parsed testcases at query #119
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = var_2.get_signature(var_3)
    var_10 = b'modified value'
    var_11 = var_2.verify_signature(var_10, var_9)
    assert var_11 is False
    var_12 = b''
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = b'not-base64!!!'
    var_16 = var_2.verify_signature(var_3, var_15)
    assert var_16 is False
    var_17 = 'secret-key'
    var_18 = 'test-salt'
    var_19 = b'test value'
    var_20 = var_2.get_signature(var_19)
    var_21 = var_2.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24, var_18)
    var_26 = b'test value'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_17, var_18, algorithm=var_29)
    var_31 = b'test value'
    var_32 = var_30.get_signature(var_31)
    var_33 = b''
    var_34 = module_1.base64_encode(var_33)
    var_35 = var_30.verify_signature(var_31, var_32)
    assert var_35 is True
    var_36 = module_1.base64_encode(var_33)
    var_37 = var_30.verify_signature(var_31, var_36)
    assert var_37 is True
    var_38 = b'different value'
    var_39 = var_30.verify_signature(var_38, var_32)
    assert var_39 is True
    var_40 = 'test string'
    var_41 = var_30.get_signature(var_40)
    var_42 = var_30.verify_signature(var_40, var_41)
    assert var_42 is True
    var_43 = b'test string'
    var_44 = var_30.verify_signature(var_43, var_41)
    assert var_44 is True



# Parsed testcases at query #120
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
    var_7 = b''
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'invalid-sig'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'old-secret'
    var_12 = 'new-secret'
    var_13 = [var_11, var_12]
    var_14 = module_0.Signer(var_13)
    var_15 = b'test-rotation'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'secret'
    var_19 = module_0.NoneAlgorithm()
    var_20 = module_0.Signer(var_18, algorithm=var_19)
    var_21 = b'test-none'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = module_0.Signer(var_18)
    var_25 = b'test-b64'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'modified-value'
    var_29 = var_1.verify_signature(var_28, var_3)
    assert var_29 is False



# Parsed testcases at query #121
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'test-string'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = '!!!invalid-base64!!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'test-rotation'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'secret'
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_22, algorithm=var_23)
    var_25 = b'test-none'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'test-sha256'



# Parsed testcases at query #122
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'different-key'
    var_11 = module_0.Signer(var_10)
    var_12 = var_11.verify_signature(var_2, var_3)
    assert var_12 is False
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'test-value-2'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = b'!!!invalid-base64!!!'
    var_21 = var_1.verify_signature(var_2, var_20)
    assert var_21 is False
    var_22 = 'test-value'
    var_23 = var_1.verify_signature(var_22, var_3)
    assert var_23 is True
    var_24 = 'secret'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = var_26.get_signature(var_2)
    var_28 = var_26.verify_signature(var_2, var_27)
    assert var_28 is True



# Parsed testcases at query #123
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = 'string-value'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'different-secret-key'
    var_14 = module_0.Signer(var_13)
    var_15 = var_14.verify_signature(var_2, var_3)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'test-value-2'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'key'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_23, algorithm=var_24)
    var_26 = b'test-value-3'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'!!!invalid-base64!!!'
    var_30 = var_1.verify_signature(var_2, var_29)
    assert var_30 is False
    var_31 = b'|'
    var_32 = module_0.Signer(var_0, sep=var_31)
    var_33 = b'test-value-4'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = var_1.verify_signature(var_33, var_34)
    assert var_36 is False
    var_37 = b'custom-salt'
    var_38 = module_0.Signer(var_0, var_37)
    var_39 = b'test-value-5'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = var_1.verify_signature(var_39, var_40)
    assert var_42 is False



# Parsed testcases at query #124
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = 'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'secret-key'
    var_6 = module_0.Signer(var_5)
    var_7 = b'test-value'
    var_8 = var_6.get_signature(var_7)
    var_9 = var_6.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = module_0.Signer(var_0)
    var_11 = 'test-value'
    var_12 = var_10.get_signature(var_11)
    var_13 = b'invalid-signature'
    var_14 = module_1.base64_encode(var_13)
    var_15 = var_10.verify_signature(var_11, var_14)
    assert var_15 is False
    var_16 = module_0.Signer(var_0)
    var_17 = ''
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'secret-key-1'
    var_21 = module_0.Signer(var_20)
    var_22 = 'secret-key-2'
    var_23 = module_0.Signer(var_22)
    var_24 = 'test-value'
    var_25 = var_21.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is False
    var_27 = 'old-key'
    var_28 = 'new-key'
    var_29 = [var_27, var_28]
    var_30 = module_0.Signer(var_29)
    var_31 = 'test-value'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = module_0.Signer(var_0)
    var_35 = 'test-value'
    var_36 = 'not-base64!!'
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is False
    var_38 = module_0.Signer(var_0)
    var_39 = 'test-value'
    var_40 = ''
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is False
    var_42 = module_0.NoneAlgorithm()
    var_43 = module_0.Signer(var_0, algorithm=var_42)
    var_44 = 'test-value'
    var_45 = var_43.get_signature(var_44)
    var_46 = b''
    var_47 = module_1.base64_encode(var_46)
    var_48 = var_43.verify_signature(var_44, var_45)
    assert var_48 is True
    var_49 = 'salt1'
    var_50 = module_0.Signer(var_0, var_49)
    var_51 = 'salt2'
    var_52 = module_0.Signer(var_0, var_51)
    var_53 = 'test-value'
    var_54 = var_50.get_signature(var_53)
    var_55 = var_50.verify_signature(var_53, var_54)
    assert var_55 is True
    var_56 = var_52.verify_signature(var_53, var_54)
    assert var_56 is False
    var_57 = b'|'
    var_58 = module_0.Signer(var_0, sep=var_57)
    var_59 = 'test-value'
    var_60 = var_58.get_signature(var_59)
    var_61 = var_58.verify_signature(var_59, var_60)
    assert var_61 is True



# Parsed testcases at query #125
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
    var_7 = b''
    var_8 = var_1.get_signature(var_7)
    var_9 = var_1.verify_signature(var_7, var_8)
    assert var_9 is True
    var_10 = b'bytes-value'
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = 'string-value'
    var_14 = var_1.get_signature(var_13)
    var_15 = var_1.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'rotation-test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.Signer(var_16)
    var_24 = var_23.get_signature(var_20)
    var_25 = var_19.verify_signature(var_20, var_24)
    assert var_25 is True
    var_26 = b'!!!invalid-base64!!!'
    var_27 = var_1.verify_signature(var_2, var_26)
    assert var_27 is False
    var_28 = b''
    var_29 = var_1.verify_signature(var_2, var_28)
    assert var_29 is False
    var_30 = 'secret'
    var_31 = module_0.NoneAlgorithm()
    var_32 = module_0.Signer(var_30, algorithm=var_31)
    var_33 = b'none-test'
    var_34 = var_32.get_signature(var_33)
    assert var_34 == b''
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = b'anything'
    var_37 = var_32.verify_signature(var_33, var_36)
    assert var_37 is True
    var_38 = b'sha256-test'
    var_39 = b'wrong-sig'



# Parsed testcases at query #126
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
    var_8 = b''
    var_9 = var_2.get_signature(var_8)
    var_10 = var_2.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b''
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = module_0.NoneAlgorithm()
    var_14 = module_0.Signer(var_0, var_1, algorithm=var_13)
    var_15 = var_14.verify_signature(var_3, var_11)
    assert var_15 is True
    var_16 = b'any-signature'
    var_17 = var_14.verify_signature(var_3, var_16)
    assert var_17 is True
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20, var_1)
    var_22 = b'rotation-test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = 'test-value'
    var_26 = var_2.verify_signature(var_25, var_23)
    assert var_26 is True
    var_27 = b'!!!invalid-base64!!!'
    var_28 = var_2.verify_signature(var_22, var_27)
    assert var_28 is False
    var_29 = 'different-salt'
    var_30 = module_0.Signer(var_0, var_29)
    var_31 = var_30.get_signature(var_22)
    var_32 = var_2.verify_signature(var_22, var_31)
    assert var_32 is False



# Parsed testcases at query #127
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'\x00\x01\x02\xff'
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'different-salt'
    var_15 = module_0.Signer(var_0, var_14)
    var_16 = var_15.get_signature(var_2)
    var_17 = var_1.verify_signature(var_2, var_16)
    assert var_17 is False
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = var_21.get_signature(var_2)
    var_23 = var_21.verify_signature(var_2, var_22)
    assert var_23 is True
    var_24 = '!!!invalid-base64!!!'
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_0, algorithm=var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_27.verify_signature(var_2, var_28)
    assert var_29 is True
    var_30 = 'string value'
    var_31 = var_1.get_signature(var_30)
    var_32 = var_1.verify_signature(var_30, var_31)
    assert var_32 is True



# Parsed testcases at query #128
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'other-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'test-value-2'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = module_0.Signer(var_9)
    var_17 = var_16.get_signature(var_13)
    var_18 = var_12.verify_signature(var_13, var_17)
    assert var_18 is True
    var_19 = b''
    var_20 = var_1.verify_signature(var_2, var_19)
    assert var_20 is False
    var_21 = 'key'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = var_23.get_signature(var_2)
    var_25 = var_23.verify_signature(var_2, var_24)
    assert var_25 is True
    var_26 = b'something'
    var_27 = var_23.verify_signature(var_2, var_26)
    assert var_27 is False



# Parsed testcases at query #129
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test Signer.verify_signature with various scenarios.'
    var_1 = 'secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid-signature'
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_3.get_signature(var_9)
    var_11 = var_3.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'modified-value'
    var_13 = var_3.verify_signature(var_12, var_5)
    assert var_13 is False
    var_14 = 'test-string'
    var_15 = var_3.get_signature(var_14)
    var_16 = var_3.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = var_3.get_signature(var_4)
    var_18 = module_1.base64_encode(var_17)
    var_19 = var_3.verify_signature(var_4, var_18)
    assert var_19 is True
    var_20 = b'!!!invalid-base64!!!'
    var_21 = var_3.verify_signature(var_4, var_20)
    assert var_21 is False
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24, var_2)
    var_26 = b'rotation-test'
    var_27 = module_0.Signer(var_22, var_2)
    var_28 = var_27.get_signature(var_26)
    var_29 = var_25.verify_signature(var_26, var_28)
    assert var_29 is True
    var_30 = var_25.get_signature(var_26)
    var_31 = var_25.verify_signature(var_26, var_30)
    assert var_31 is True
    var_32 = 'wrong-key'
    var_33 = module_0.Signer(var_32, var_2)
    var_34 = var_33.get_signature(var_26)
    var_35 = var_25.verify_signature(var_26, var_34)
    assert var_35 is False
    var_36 = module_0.NoneAlgorithm()
    var_37 = module_0.Signer(var_1, var_2, algorithm=var_36)
    var_38 = b'none-algorithm-test'
    var_39 = var_37.get_signature(var_38)
    var_40 = b''
    var_41 = module_1.base64_encode(var_40)
    var_42 = var_37.verify_signature(var_38, var_39)
    assert var_42 is True
    var_43 = b'|'
    var_44 = module_0.Signer(var_1, var_2, var_43)
    var_45 = b'custom-sep-test'
    var_46 = var_44.get_signature(var_45)
    var_47 = var_44.verify_signature(var_45, var_46)
    assert var_47 is True
    var_48 = b'wrong-sig'
    var_49 = var_44.verify_signature(var_45, var_48)
    assert var_49 is False



# Parsed testcases at query #130
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'!!!invalid-base64!!!'
    var_12 = var_1.verify_signature(var_2, var_11)
    assert var_12 is False
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'rotated test'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = 'key'
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_20, algorithm=var_21)
    var_23 = b'none algo'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'|'
    var_27 = module_0.Signer(var_20, sep=var_26)
    var_28 = b'custom sep'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True



# Parsed testcases at query #131
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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b'tampered-value'
    var_9 = var_1.verify_signature(var_8, var_3)
    assert var_9 is False
    var_10 = 'not-base64!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b''
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = b'rotation-test'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'key'
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_22, algorithm=var_23)
    var_25 = b'test-none'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'test-value'
    var_29 = var_1.verify_signature(var_28, var_3)
    assert var_29 is True



