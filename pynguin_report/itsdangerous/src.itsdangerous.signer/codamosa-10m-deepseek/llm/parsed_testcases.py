####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
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
    assert var_4 == b'test-value'
    var_5 = 'test-value'
    var_6 = var_1.sign(var_5)
    var_7 = var_1.unsign(var_6)
    assert var_7 == b'test-value'
    var_8 = 'old-secret'
    var_9 = module_0.Signer(var_8)
    var_10 = b'value-signed-with-old-key'
    var_11 = var_9.sign(var_10)
    var_12 = 'new-secret'
    var_13 = [var_12, var_8]
    var_14 = module_0.Signer(var_13)
    var_15 = var_14.unsign(var_11)
    assert var_15 == b'value-signed-with-old-key'
    var_16 = b'no-separator'
    var_17 = var_1.unsign(var_16)
    var_18 = b'value.invalid-signature'
    var_19 = var_1.unsign(var_18)
    var_20 = b'|'
    var_21 = module_0.Signer(var_18, sep=var_20)
    var_22 = var_21.sign(var_19)
    var_23 = var_21.unsign(var_22)
    assert var_23 == b'test-value'
    var_24 = var_1.unsign(var_22)
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = var_26.sign(var_19)
    var_28 = var_26.unsign(var_27)
    assert var_28 == b'test-value'
    var_29 = 'secret-key'
    var_30 = b'test-value'
    var_31 = var_1.sign(var_30)
    var_32 = var_1.unsign(var_31)
    assert var_32 == b'test-value'
    var_33 = b''
    var_34 = var_1.sign(var_33)
    var_35 = var_1.unsign(var_34)
    assert var_35 == b''
    var_36 = b'value-with-special-chars:./\\'
    var_37 = var_1.sign(var_36)
    var_38 = var_1.unsign(var_37)



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
    var_5 = b'|'
    var_6 = module_0.Signer(var_0, sep=var_5)
    var_7 = var_6.sign(var_2)
    var_8 = var_6.unsign(var_7)
    assert var_8 == b'test-value'
    var_9 = b'secret-key'
    var_10 = module_0.Signer(var_9)
    var_11 = b'test-value'
    var_12 = var_10.sign(var_11)
    var_13 = var_10.unsign(var_12)
    assert var_13 == b'test-value'
    var_14 = 'custom-salt'
    var_15 = module_0.Signer(var_0, var_14)
    var_16 = var_15.sign(var_2)
    var_17 = var_15.unsign(var_16)
    assert var_17 == b'test-value'
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = var_21.sign(var_2)
    var_23 = var_21.unsign(var_22)
    assert var_23 == b'test-value'
    var_24 = 'secret-key'
    var_25 = 'test-value'
    var_26 = var_21.sign(var_25)
    var_27 = var_21.unsign(var_26)
    assert var_27 == b'test-value'
    var_28 = var_21.sign(var_25)
    var_29 = var_21.unsign(var_28)
    assert var_29 == b'test-value'
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_24, algorithm=var_30)
    var_32 = var_31.sign(var_25)
    var_33 = var_31.unsign(var_32)
    assert var_33 == b'test-value'
    var_34 = module_0.Signer(var_24)
    var_35 = b'no-separator'
    var_36 = var_34.unsign(var_35)
    var_37 = module_0.Signer(var_35)
    var_38 = b'value.invalid-signature'
    var_39 = var_37.unsign(var_38)
    var_40 = module_0.Signer(var_38)
    var_41 = b''
    var_42 = var_40.sign(var_41)
    var_43 = var_40.unsign(var_42)
    assert var_43 == b''
    var_44 = module_0.Signer(var_38)
    var_45 = 'value with spaces and !@#$%'
    var_46 = var_44.sign(var_45)
    var_47 = var_44.unsign(var_46)
    assert var_47 == b'value with spaces and !@#$%'
    var_48 = module_0.Signer(var_38)
    var_49 = 'café'
    var_50 = var_48.sign(var_49)
    var_51 = var_48.unsign(var_50)
    var_52 = module_0.Signer(var_38)
    var_53 = 'a'
    var_54 = 10000
    var_55 = var_53 * var_54
    var_56 = var_52.sign(var_55)
    var_57 = var_52.unsign(var_56)
    var_58 = module_0.Signer(var_38)
    var_59 = b'value'
    var_60 = b'invalid-sig'
    var_61 = var_58.verify_signature(var_59, var_60)
    var_62 = 'hmac'
    var_63 = module_0.Signer(var_38, key_derivation=var_62)
    var_64 = var_63.sign(var_39)
    var_65 = var_63.unsign(var_64)
    assert var_65 == b'test-value'
    var_66 = module_0.NoneAlgorithm()
    var_67 = module_0.Signer(var_38, algorithm=var_66)
    var_68 = b'value.'
    var_69 = var_67.unsign(var_68)
    assert var_69 == b'value'
    var_70 = b'value.anything'
    var_71 = var_67.unsign(var_70)
    assert var_71 == b'value'
    var_72 = 'secret-key'
    var_73 = b'a'
    var_74 = module_0.Signer(var_72, sep=var_73)
    var_75 = 12345
    var_76 = module_0.Signer(var_75)
    var_77 = module_0.Signer(var_75)
    var_78 = var_77.sign(var_76)
    var_79 = var_77.validate(var_78)
    var_80 = b'invalid-value'
    var_81 = var_77.validate(var_80)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = len(var_3)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = 'custom-secret'
    var_4 = var_2.derive_key(var_3)
    var_5 = len(var_4)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = 'concat'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_4 = var_3.derive_key()
    var_5 = len(var_4)
    assert var_5 == 20

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = 'django-concat'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_4 = var_3.derive_key()
    var_5 = len(var_4)
    assert var_5 == 20

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = 'hmac'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_4 = var_3.derive_key()
    var_5 = len(var_4)
    assert var_5 == 20

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = 'none'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_4 = var_3.derive_key()
    assert var_4 == b'test-secret'

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = 'invalid'
    var_3 = module_0.Signer(var_0, var_1, key_derivation=var_2)
    var_4 = var_3.derive_key()

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'old-secret'
    var_1 = 'new-secret'
    var_2 = [var_0, var_1]
    var_3 = 'test-salt'
    var_4 = module_0.Signer(var_2, var_3)
    var_5 = var_4.derive_key()
    var_6 = var_4.derive_key(var_0)
    var_7 = var_4.derive_key(var_1)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = b'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = len(var_3)

import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = module_0.Signer(var_0, var_1)
    var_4 = var_2.derive_key()
    var_5 = var_3.derive_key()



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
    var_5 = 'old-key'
    var_6 = 'new-key'
    var_7 = [var_5, var_6]
    var_8 = module_0.Signer(var_7)
    var_9 = b'test-rotation'
    var_10 = var_8.get_signature(var_9)
    var_11 = var_8.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'wrong-signature'
    var_13 = module_1.base64_encode(var_12)
    var_14 = var_1.verify_signature(var_9, var_13)
    assert var_14 is False
    var_15 = b''
    var_16 = var_1.get_signature(var_15)
    var_17 = var_1.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'invalid-base64!!'
    var_19 = var_1.verify_signature(var_9, var_18)
    assert var_19 is False
    var_20 = 'hmac-key'
    var_21 = b'hmac-test'
    var_22 = 'none-key'
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_22, algorithm=var_23)
    var_25 = b'none-test'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b''
    var_29 = var_24.verify_signature(var_25, var_28)
    assert var_29 is True
    var_30 = 'concat-key'
    var_31 = 'concat'
    var_32 = module_0.Signer(var_30, key_derivation=var_31)
    var_33 = b'concat-test'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = 'hmac-derived-key'
    var_37 = 'hmac'
    var_38 = module_0.Signer(var_36, key_derivation=var_37)
    var_39 = b'hmac-derived-test'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = 'string-key'
    var_43 = module_0.Signer(var_42)
    var_44 = 'string-value'
    var_45 = var_43.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = 'salt-key'
    var_48 = b'custom-salt'
    var_49 = module_0.Signer(var_47, var_48)
    var_50 = b'salt-test'
    var_51 = var_49.get_signature(var_50)
    var_52 = var_49.verify_signature(var_50, var_51)
    assert var_52 is True
    var_53 = 'sep-key'
    var_54 = b'|'
    var_55 = module_0.Signer(var_53, sep=var_54)
    var_56 = b'sep-test'
    var_57 = var_55.get_signature(var_56)
    var_58 = var_55.verify_signature(var_56, var_57)
    assert var_58 is True
    var_59 = 'none-salt-key'
    var_60 = None
    var_61 = module_0.Signer(var_59, var_60)
    var_62 = b'none-salt-test'
    var_63 = var_61.get_signature(var_62)
    var_64 = var_61.verify_signature(var_62, var_63)
    assert var_64 is True
    var_65 = 'int-key'
    var_66 = module_0.Signer(var_65)
    var_67 = b'12345'
    var_68 = var_66.get_signature(var_67)
    var_69 = var_66.verify_signature(var_67, var_68)
    assert var_69 is True



# Parsed testcases at query #5
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
    var_34 = 'key1'
    var_35 = 'key2'
    var_36 = 'key3'
    var_37 = [var_34, var_35, var_36]
    var_38 = module_0.Signer(var_37, var_1)
    var_39 = var_38.derive_key()
    var_40 = var_4 + var_5
    var_41 = b'key3'
    var_42 = var_40 + var_41
    var_43 = module_1.digest()
    var_44 = var_38.derive_key()
    var_45 = var_4 + var_5
    var_46 = var_45 + var_7
    var_47 = module_1.digest()
    var_48 = module_0.Signer(var_0, var_1)
    var_49 = var_48.derive_key()
    var_50 = 'unknown'
    var_51 = module_0.Signer(var_0, var_1, key_derivation=var_50)
    var_52 = var_51.derive_key()



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
    var_7 = b'different-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
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
    var_21 = 'key'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = var_23.get_signature(var_2)
    var_25 = var_23.verify_signature(var_2, var_24)
    assert var_25 is True
    var_26 = 'custom-salt'
    var_27 = module_0.Signer(var_21, var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_27.verify_signature(var_2, var_28)
    assert var_29 is True
    var_30 = 'different-salt'
    var_31 = module_0.Signer(var_21, var_30)
    var_32 = var_31.verify_signature(var_2, var_28)
    assert var_32 is False
    var_33 = b'!!!invalid-base64!!!'
    var_34 = var_1.verify_signature(var_2, var_33)
    assert var_34 is False
    var_35 = b''
    var_36 = var_1.get_signature(var_35)
    var_37 = var_1.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = b'|'
    var_39 = module_0.Signer(var_21, sep=var_38)
    var_40 = var_39.get_signature(var_2)
    var_41 = var_39.verify_signature(var_2, var_40)
    assert var_41 is True



# Parsed testcases at query #7
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
    var_7 = b'!!!not-base64!!!'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'test-value-2'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = b'custom-salt'
    var_17 = module_0.Signer(var_0, var_16)
    var_18 = b'test-value-3'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = b'different-salt'
    var_22 = module_0.Signer(var_0, var_21)
    var_23 = var_22.verify_signature(var_18, var_19)
    assert var_23 is False
    var_24 = 'hmac'
    var_25 = module_0.Signer(var_0, key_derivation=var_24)
    var_26 = b'test-value-5'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = module_0.Signer(var_0)
    var_30 = b''
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = module_0.Signer(var_0)
    var_34 = 'test-string'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True



# Parsed testcases at query #8
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
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = b'!!!invalid-base64!!!'
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = module_0.NoneAlgorithm()
    var_16 = module_0.Signer(var_0, algorithm=var_15)
    var_17 = b'test-value'
    var_18 = b''
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
    var_27 = module_0.Signer(var_20)
    var_28 = var_27.get_signature(var_24)
    var_29 = var_23.verify_signature(var_24, var_28)
    assert var_29 is True
    var_30 = 'custom-salt'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = b'test-value'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = 'different-salt'
    var_36 = module_0.Signer(var_0, var_35)
    var_37 = var_36.verify_signature(var_32, var_33)
    assert var_37 is False
    var_38 = 'hmac'
    var_39 = module_0.Signer(var_0, key_derivation=var_38)
    var_40 = b'test-value'
    var_41 = var_39.get_signature(var_40)
    var_42 = var_39.verify_signature(var_40, var_41)
    assert var_42 is True
    var_43 = 'concat'
    var_44 = module_0.Signer(var_0, key_derivation=var_43)
    var_45 = b'test-value'
    var_46 = var_44.get_signature(var_45)
    var_47 = var_44.verify_signature(var_45, var_46)
    assert var_47 is True
    var_48 = b'test-value'
    var_49 = 256
    var_50 = range(var_49)
    var_51 = bytes(var_50)
    var_52 = var_1.get_signature(var_51)
    var_53 = var_1.verify_signature(var_51, var_52)
    assert var_53 is True
    var_54 = 'héllo wörld 🎉'
    var_55 = var_1.get_signature(var_54)
    var_56 = var_1.verify_signature(var_54, var_55)
    assert var_56 is True
    var_57 = b''
    var_58 = var_1.verify_signature(var_2, var_57)
    assert var_58 is False



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
    var_13 = b'rotation-test'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = b'different-salt'
    var_17 = module_0.Signer(var_0, var_16)
    var_18 = var_17.verify_signature(var_2, var_3)
    assert var_18 is False
    var_19 = b''
    var_20 = var_1.verify_signature(var_2, var_19)
    assert var_20 is False
    var_21 = b'!!!invalid_base64!!!'
    var_22 = var_1.verify_signature(var_2, var_21)
    assert var_22 is False
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, algorithm=var_23)
    var_25 = b'no-signature'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'|'
    var_29 = module_0.Signer(var_0, sep=var_28)
    var_30 = b'custom-sep'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True



# Parsed testcases at query #10
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
    var_11 = b'!!!invalid_base64!!!'
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = module_0.NoneAlgorithm()
    var_14 = module_0.Signer(var_0, algorithm=var_13)
    var_15 = b'test'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'old-secret'
    var_19 = 'new-secret'
    var_20 = [var_18, var_19]
    var_21 = 'rotation-test'
    var_22 = module_0.Signer(var_20, var_21)
    var_23 = b'rotation value'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = 'string value'
    var_27 = var_2.get_signature(var_26)
    var_28 = var_2.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'modified value'
    var_30 = var_2.verify_signature(var_29, var_4)
    assert var_30 is False



# Parsed testcases at query #11
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
    var_11 = 'test-string'
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'not-base64!'
    var_15 = var_1.verify_signature(var_2, var_14)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'rotated-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'test'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_23, algorithm=var_24)
    var_26 = b'none-algorithm'
    var_27 = var_25.get_signature(var_26)
    var_28 = b''
    var_29 = module_1.base64_encode(var_28)
    var_30 = var_25.verify_signature(var_26, var_27)
    assert var_30 is True
    var_31 = module_0.HMACAlgorithm()
    var_32 = module_0.Signer(var_23, algorithm=var_31)
    var_33 = b'hmac-test'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = b'sha256-test'
    var_37 = b'|'
    var_38 = module_0.Signer(var_23, sep=var_37)
    var_39 = b'separator-test'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = None
    var_43 = module_0.Signer(var_23, var_42)
    var_44 = b'no-salt'
    var_45 = var_43.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = b'custom-salt'
    var_48 = module_0.Signer(var_23, var_47)
    var_49 = b'custom-salt-value'
    var_50 = var_48.get_signature(var_49)
    var_51 = var_48.verify_signature(var_49, var_50)
    assert var_51 is True
    var_52 = 'concat'
    var_53 = module_0.Signer(var_23, key_derivation=var_52)
    var_54 = b'concat-test'
    var_55 = var_53.get_signature(var_54)
    var_56 = var_53.verify_signature(var_54, var_55)
    assert var_56 is True
    var_57 = 'hmac'
    var_58 = module_0.Signer(var_23, key_derivation=var_57)
    var_59 = b'hmac-derived'
    var_60 = var_58.get_signature(var_59)
    var_61 = var_58.verify_signature(var_59, var_60)
    assert var_61 is True
    var_62 = 'none'
    var_63 = module_0.Signer(var_23, key_derivation=var_62)
    var_64 = b'none-derived'
    var_65 = var_63.get_signature(var_64)
    var_66 = var_63.verify_signature(var_64, var_65)
    assert var_66 is True



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
    var_7 = 'test-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is True
    var_9 = 'ascii'
    var_10 = b'!!!invalid-base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b'modified-value'
    var_13 = var_1.verify_signature(var_12, var_3)
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
    var_24 = 'different-salt'
    var_25 = module_0.Signer(var_0, var_24)
    var_26 = var_25.get_signature(var_2)
    var_27 = var_1.verify_signature(var_2, var_26)
    assert var_27 is False
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_0, algorithm=var_28)
    var_30 = b'test-none'
    var_31 = var_29.get_signature(var_30)
    assert var_31 == b''
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = b'test-sha256'
    var_34 = 'concat'
    var_35 = module_0.Signer(var_0, key_derivation=var_34)
    var_36 = b'test-concat'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = 'hmac'
    var_40 = module_0.Signer(var_0, key_derivation=var_39)
    var_41 = b'test-hmac'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = 'none'
    var_45 = module_0.Signer(var_0, key_derivation=var_44)
    var_46 = b'test-none-deriv'
    var_47 = var_45.get_signature(var_46)
    var_48 = var_45.verify_signature(var_46, var_47)
    assert var_48 is True
    var_49 = b''
    var_50 = var_1.get_signature(var_49)
    var_51 = var_1.verify_signature(var_49, var_50)
    assert var_51 is True
    var_52 = b'test with spaces and !@#$%^&*()'
    var_53 = var_1.get_signature(var_52)
    var_54 = var_1.verify_signature(var_52, var_53)
    assert var_54 is True



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
    var_13 = 'ascii'
    var_14 = module_0.NoneAlgorithm()
    var_15 = module_0.Signer(var_0, algorithm=var_14)
    var_16 = var_15.get_signature(var_3)
    var_17 = var_15.verify_signature(var_3, var_16)
    assert var_17 is True
    var_18 = b''
    var_19 = var_15.verify_signature(var_3, var_18)
    assert var_19 is True
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22, var_1)
    var_24 = module_0.Signer(var_20, var_1)
    var_25 = var_24.get_signature(var_3)
    var_26 = var_23.verify_signature(var_3, var_25)
    assert var_26 is True
    var_27 = var_23.get_signature(var_3)
    var_28 = var_23.verify_signature(var_3, var_27)
    assert var_28 is True
    var_29 = b'!!!invalid-base64!!!'
    var_30 = var_2.verify_signature(var_3, var_29)
    assert var_30 is False
    var_31 = 'different-salt'
    var_32 = module_0.Signer(var_0, var_31)
    var_33 = var_32.get_signature(var_3)
    var_34 = var_2.verify_signature(var_3, var_33)
    assert var_34 is False
    var_35 = 'different-key'
    var_36 = module_0.Signer(var_35, var_1)
    var_37 = var_36.get_signature(var_3)
    var_38 = var_2.verify_signature(var_3, var_37)
    assert var_38 is False
    var_39 = module_0.HMACAlgorithm()
    var_40 = module_0.Signer(var_0, algorithm=var_39)
    var_41 = var_40.get_signature(var_3)
    var_42 = var_40.verify_signature(var_3, var_41)
    assert var_42 is True



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test verify_signature method with various scenarios.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = 'test-string'
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b''
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = b'modified-value'
    var_16 = var_2.verify_signature(var_15, var_4)
    assert var_16 is False
    var_17 = b'!!!invalid-base64!!!'
    var_18 = var_2.verify_signature(var_3, var_17)
    assert var_18 is False
    var_19 = 'different-salt'
    var_20 = module_0.Signer(var_1, var_19)
    var_21 = var_20.get_signature(var_3)
    var_22 = var_2.verify_signature(var_3, var_21)
    assert var_22 is False
    var_23 = var_20.verify_signature(var_3, var_21)
    assert var_23 is True
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_1, algorithm=var_24)
    var_26 = b'test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'old-key'
    var_30 = 'new-key'
    var_31 = [var_29, var_30]
    var_32 = module_0.Signer(var_31)
    var_33 = var_32.get_signature(var_26)
    var_34 = var_32.verify_signature(var_26, var_33)
    assert var_34 is True
    var_35 = 'concat'
    var_36 = module_0.Signer(var_1, key_derivation=var_35)
    var_37 = var_36.get_signature(var_26)
    var_38 = var_36.verify_signature(var_26, var_37)
    assert var_38 is True
    var_39 = b''
    var_40 = var_2.verify_signature(var_26, var_39)
    assert var_40 is False
    var_41 = b'a'
    var_42 = 10000
    var_43 = var_41 * var_42
    var_44 = var_2.get_signature(var_43)
    var_45 = var_2.verify_signature(var_43, var_44)
    assert var_45 is True



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
    var_20 = b'not-base64!!'
    var_21 = var_1.verify_signature(var_2, var_20)
    assert var_21 is False
    var_22 = 'key'
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_22, algorithm=var_23)
    var_25 = var_24.get_signature(var_2)
    var_26 = var_24.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = b'anything'
    var_28 = var_24.verify_signature(var_2, var_27)
    assert var_28 is True
    var_29 = 'string-value'
    var_30 = var_1.get_signature(var_29)
    var_31 = var_1.verify_signature(var_29, var_30)
    assert var_31 is True



# Parsed testcases at query #16
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
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'test-string'
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'utf-8'
    var_15 = -1
    var_16 = var_3[:var_15]
    var_17 = -1
    var_18 = var_3[var_17]
    var_19 = 255
    var_20 = var_18 ^ var_19
    var_21 = [var_20]
    var_22 = bytes(var_21)
    var_23 = var_16 + var_22
    var_24 = var_1.verify_signature(var_2, var_23)
    assert var_24 is False
    var_25 = '!!!invalid-base64!!!'
    var_26 = var_1.verify_signature(var_2, var_25)
    assert var_26 is False
    var_27 = b''
    var_28 = var_1.verify_signature(var_2, var_27)
    assert var_28 is False
    var_29 = 'test'
    var_30 = module_0.NoneAlgorithm()
    var_31 = module_0.Signer(var_29, algorithm=var_30)
    var_32 = b'test-value'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = 'old-key'
    var_36 = 'new-key'
    var_37 = [var_35, var_36]
    var_38 = module_0.Signer(var_37)
    var_39 = module_0.Signer(var_35)
    var_40 = var_39.get_signature(var_2)
    var_41 = var_38.verify_signature(var_2, var_40)
    assert var_41 is True
    var_42 = var_38.get_signature(var_2)
    var_43 = var_38.verify_signature(var_2, var_42)
    assert var_43 is True
    var_44 = 'different-key'
    var_45 = module_0.Signer(var_44)
    var_46 = var_45.get_signature(var_2)
    var_47 = var_1.verify_signature(var_2, var_46)
    assert var_47 is False



# Parsed testcases at query #17
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
    var_20 = b'test-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid-base64!!!'
    var_24 = var_2.verify_signature(var_20, var_23)
    assert var_24 is False
    var_25 = 'different-salt'
    var_26 = module_0.Signer(var_0, var_25)
    var_27 = var_26.get_signature(var_20)
    var_28 = var_2.verify_signature(var_20, var_27)
    assert var_28 is False
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_0, algorithm=var_29)
    var_31 = var_30.get_signature(var_20)
    var_32 = var_30.verify_signature(var_20, var_31)
    assert var_32 is True
    var_33 = b'anything'
    var_34 = var_30.verify_signature(var_20, var_33)
    assert var_34 is True



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test verify_signature method of Signer class.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test_value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = 'test_string'
    var_7 = var_2.get_signature(var_6)
    var_8 = var_2.verify_signature(var_6, var_7)
    assert var_8 is True
    var_9 = b'invalid_sig'
    var_10 = var_2.verify_signature(var_3, var_9)
    assert var_10 is False
    var_11 = b''
    var_12 = var_2.verify_signature(var_3, var_11)
    assert var_12 is False
    var_13 = b'!!!invalid_base64!!!'
    var_14 = var_2.verify_signature(var_3, var_13)
    assert var_14 is False
    var_15 = 'concat'
    var_16 = module_0.Signer(var_1, key_derivation=var_15)
    var_17 = var_16.get_signature(var_3)
    var_18 = var_16.verify_signature(var_3, var_17)
    assert var_18 is True
    var_19 = 'hmac'
    var_20 = module_0.Signer(var_1, key_derivation=var_19)
    var_21 = var_20.get_signature(var_3)
    var_22 = var_20.verify_signature(var_3, var_21)
    assert var_22 is True
    var_23 = 'none'
    var_24 = module_0.Signer(var_1, key_derivation=var_23)
    var_25 = var_24.get_signature(var_3)
    var_26 = var_24.verify_signature(var_3, var_25)
    assert var_26 is True
    var_27 = b'custom_salt'
    var_28 = module_0.Signer(var_1, var_27)
    var_29 = var_28.get_signature(var_3)
    var_30 = var_28.verify_signature(var_3, var_29)
    assert var_30 is True
    var_31 = 'old-key'
    var_32 = 'new-key'
    var_33 = [var_31, var_32]
    var_34 = module_0.Signer(var_33)
    var_35 = var_34.get_signature(var_3)
    var_36 = var_34.verify_signature(var_3, var_35)
    assert var_36 is True
    var_37 = module_0.Signer(var_31)
    var_38 = var_37.get_signature(var_3)
    var_39 = var_34.verify_signature(var_3, var_38)
    assert var_39 is True
    var_40 = module_0.NoneAlgorithm()
    var_41 = module_0.Signer(var_1, algorithm=var_40)
    var_42 = var_41.get_signature(var_3)
    assert var_42 == b''
    var_43 = var_41.verify_signature(var_3, var_42)
    assert var_43 is True
    var_44 = 'key1'
    var_45 = module_0.Signer(var_44)
    var_46 = 'key2'
    var_47 = module_0.Signer(var_46)
    var_48 = var_45.get_signature(var_3)
    var_49 = var_47.get_signature(var_3)
    var_50 = var_45.verify_signature(var_3, var_48)
    assert var_50 is True
    var_51 = var_45.verify_signature(var_3, var_49)
    assert var_51 is False
    var_52 = var_47.verify_signature(var_3, var_49)
    assert var_52 is True
    var_53 = var_47.verify_signature(var_3, var_48)
    assert var_53 is False
    var_54 = 'héllo'
    var_55 = var_2.get_signature(var_54)
    var_56 = var_2.verify_signature(var_54, var_55)
    assert var_56 is True



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'my-secret-key'
    var_1 = 'my-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test_value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'wrong_signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.get_signature(var_9)
    var_11 = var_2.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'invalid_base64!!'
    var_13 = var_2.verify_signature(var_3, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'rotation_test'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.Signer(var_14)
    var_22 = var_21.get_signature(var_18)
    var_23 = var_17.verify_signature(var_18, var_22)
    assert var_23 is True
    var_24 = 'key'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = b'test_none'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test that verify_signature returns True for valid signatures and False for invalid ones.'
    var_1 = 'test-secret-key'
    var_2 = module_0.Signer(var_1)
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
    var_10 = 'test-string'
    var_11 = var_2.get_signature(var_10)
    var_12 = var_2.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = b''
    var_14 = var_2.get_signature(var_13)
    var_15 = var_2.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'test-key'
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_16, algorithm=var_17)
    var_19 = b'test'
    var_20 = var_18.get_signature(var_19)
    var_21 = var_18.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = b'test'
    var_23 = b'!!!'
    var_24 = var_2.verify_signature(var_22, var_23)
    assert var_24 is False
    var_25 = 'old-key'
    var_26 = 'new-key'
    var_27 = [var_25, var_26]
    var_28 = module_0.Signer(var_27)
    var_29 = b'test-rotation'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = 'concat'
    var_33 = module_0.Signer(var_16, key_derivation=var_32)
    var_34 = b'test-concat'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = 'hmac'
    var_38 = module_0.Signer(var_16, key_derivation=var_37)
    var_39 = b'test-hmac'
    var_40 = var_38.get_signature(var_39)
    var_41 = var_38.verify_signature(var_39, var_40)
    assert var_41 is True
    var_42 = 'none'
    var_43 = module_0.Signer(var_16, key_derivation=var_42)
    var_44 = b'test-none'
    var_45 = var_43.get_signature(var_44)
    var_46 = var_43.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = b'|'
    var_48 = module_0.Signer(var_16, sep=var_47)
    var_49 = b'test-custom-sep'
    var_50 = var_48.get_signature(var_49)
    var_51 = var_48.verify_signature(var_49, var_50)
    assert var_51 is True



# Parsed testcases at query #21
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
    var_7 = b'invalid-sig'
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_3.get_signature(var_9)
    var_11 = var_3.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = module_1.base64_encode(var_5)
    var_13 = var_3.verify_signature(var_4, var_12)
    assert var_13 is True
    var_14 = 'string-value'
    var_15 = var_3.get_signature(var_14)
    var_16 = var_3.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = b'!!!invalid-base64!!!'
    var_18 = var_3.verify_signature(var_4, var_17)
    assert var_18 is False
    var_19 = 'old-secret-key'
    var_20 = module_0.Signer(var_19, var_2)
    var_21 = b'old-value'
    var_22 = var_20.get_signature(var_21)
    var_23 = 'new-secret-key'
    var_24 = [var_19, var_23]
    var_25 = module_0.Signer(var_24, var_2)
    var_26 = var_25.verify_signature(var_21, var_22)
    assert var_26 is True
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_1, algorithm=var_27)
    var_29 = var_28.get_signature(var_4)
    var_30 = var_28.verify_signature(var_4, var_29)
    assert var_30 is True
    var_31 = 'hmac'
    var_32 = module_0.Signer(var_1, var_2, key_derivation=var_31)
    var_33 = var_32.get_signature(var_4)
    var_34 = var_32.verify_signature(var_4, var_33)
    assert var_34 is True
    var_35 = None
    var_36 = module_0.Signer(var_1, var_35)
    var_37 = var_36.get_signature(var_4)
    var_38 = var_36.verify_signature(var_4, var_37)
    assert var_38 is True
    var_39 = b'bytes-salt'
    var_40 = module_0.Signer(var_1, var_39)
    var_41 = var_40.get_signature(var_4)
    var_42 = var_40.verify_signature(var_4, var_41)
    assert var_42 is True
    var_43 = ''
    var_44 = var_3.get_signature(var_43)
    var_45 = var_3.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = module_0.NoneAlgorithm()
    var_47 = module_0.Signer(var_1, var_2, algorithm=var_46)
    var_48 = var_47.get_signature(var_4)
    var_49 = b''
    var_50 = module_1.base64_encode(var_49)
    var_51 = var_47.verify_signature(var_4, var_48)
    assert var_51 is True



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
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = 'ascii'
    var_11 = 'utf-8'
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'another-value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'key'
    var_20 = module_0.NoneAlgorithm()
    var_21 = module_0.Signer(var_19, algorithm=var_20)
    var_22 = b'test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'modified-value'
    var_26 = var_1.verify_signature(var_25, var_3)
    assert var_26 is False
    var_27 = b'different-salt'
    var_28 = module_0.Signer(var_0, var_27)
    var_29 = var_28.verify_signature(var_2, var_3)
    assert var_29 is False



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    var_1 = 'salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = b'test value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid_signature'
    var_7 = var_2.verify_signature(var_3, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_2.verify_signature(var_8, var_4)
    assert var_9 is False
    var_10 = b'!!!not_base64!!!'
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = 'test string'
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
    var_22 = module_0.Signer(var_15, var_1)
    var_23 = var_22.get_signature(var_19)
    var_24 = var_18.verify_signature(var_19, var_23)
    assert var_24 is True
    var_25 = module_0.NoneAlgorithm()
    var_26 = 'secret'
    var_27 = module_0.Signer(var_26, algorithm=var_25)
    var_28 = b'none algo'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = var_27.verify_signature(var_28, var_8)
    assert var_31 is True
    var_32 = b'hmac test'



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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = b'test'
    var_12 = b'!!!invalid-base64!!!'
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test-multi'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.Signer(var_14)
    var_22 = var_21.verify_signature(var_18, var_19)
    assert var_22 is False
    var_23 = 'test-string'
    var_24 = var_1.verify_signature(var_23, var_3)
    assert var_24 is False
    var_25 = var_1.get_signature(var_23)
    var_26 = var_1.verify_signature(var_23, var_25)
    assert var_26 is True
    var_27 = 'key'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_27, algorithm=var_28)
    var_30 = b'test-none'
    var_31 = var_29.get_signature(var_30)
    var_32 = b''
    var_33 = module_1.base64_encode(var_32)
    var_34 = var_29.verify_signature(var_30, var_31)
    assert var_34 is True
    var_35 = module_1.base64_encode(var_32)
    var_36 = var_29.verify_signature(var_30, var_35)
    assert var_36 is True
    var_37 = b'x'
    var_38 = module_1.base64_encode(var_37)
    var_39 = var_29.verify_signature(var_30, var_38)
    assert var_39 is False
    var_40 = b'test-sha256'
    var_41 = module_0.Signer(var_27)



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
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'rotation-test'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 256
    var_18 = range(var_17)
    var_19 = bytes(var_18)
    var_20 = var_1.get_signature(var_19)
    var_21 = var_1.verify_signature(var_19, var_20)
    assert var_21 is True
    var_22 = 'test-string'
    var_23 = var_1.get_signature(var_22)
    var_24 = var_1.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = 'secret'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = b'test-none'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = 'secret'
    var_32 = b'derivation-test'
    var_33 = b'!!!invalid-base64!!!'
    var_34 = var_1.verify_signature(var_32, var_33)
    assert var_34 is False
    var_35 = b''
    var_36 = var_1.verify_signature(var_32, var_35)
    assert var_36 is False
    var_37 = module_0.NoneAlgorithm()
    var_38 = module_0.Signer(var_25, algorithm=var_37)
    var_39 = b'test'
    var_40 = var_38.verify_signature(var_39, var_35)
    assert var_40 is True



# Parsed testcases at query #26
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
    var_7 = 'test value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is True
    var_9 = b'modified value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is False
    var_11 = b''
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'rotation test'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.Signer(var_14)
    var_22 = var_21.get_signature(var_18)
    var_23 = var_17.verify_signature(var_18, var_22)
    assert var_23 is True
    var_24 = 'secret'
    var_25 = module_0.NoneAlgorithm()
    var_26 = module_0.Signer(var_24, algorithm=var_25)
    var_27 = b'test'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'!!!invalid base64!!!'
    var_31 = var_1.verify_signature(var_27, var_30)
    assert var_31 is False
    var_32 = 'custom-salt'
    var_33 = module_0.Signer(var_24, var_32)
    var_34 = var_33.get_signature(var_27)
    var_35 = var_33.verify_signature(var_27, var_34)
    assert var_35 is True
    var_36 = b'|'
    var_37 = module_0.Signer(var_24, sep=var_36)
    var_38 = var_37.get_signature(var_27)
    var_39 = var_37.verify_signature(var_27, var_38)
    assert var_39 is True
    var_40 = 'concat'
    var_41 = module_0.Signer(var_24, key_derivation=var_40)
    var_42 = var_41.get_signature(var_27)
    var_43 = var_41.verify_signature(var_27, var_42)
    assert var_43 is True
    var_44 = 'hmac'
    var_45 = module_0.Signer(var_24, key_derivation=var_44)
    var_46 = var_45.get_signature(var_27)
    var_47 = var_45.verify_signature(var_27, var_46)
    assert var_47 is True
    var_48 = 'none'
    var_49 = module_0.Signer(var_24, key_derivation=var_48)
    var_50 = var_49.get_signature(var_27)
    var_51 = var_49.verify_signature(var_27, var_50)
    assert var_51 is True
    var_52 = b'bytes-key'
    var_53 = module_0.Signer(var_52)
    var_54 = var_53.get_signature(var_27)
    var_55 = var_53.verify_signature(var_27, var_54)
    assert var_55 is True
    var_56 = module_0.NoneAlgorithm()
    var_57 = b'key'
    var_58 = b'value'
    var_59 = b'anything'



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
    var_5 = 'test-string'
    var_6 = var_1.get_signature(var_5)
    var_7 = var_1.verify_signature(var_5, var_6)
    assert var_7 is True
    var_8 = b'invalid-sig'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'secret'
    var_13 = module_0.NoneAlgorithm()
    var_14 = module_0.Signer(var_12, algorithm=var_13)
    var_15 = var_14.get_signature(var_2)
    var_16 = var_14.verify_signature(var_2, var_15)
    assert var_16 is True
    var_17 = var_14.verify_signature(var_2, var_10)
    assert var_17 is True
    var_18 = b'!!!invalid-base64!!!'
    var_19 = var_1.verify_signature(var_2, var_18)
    assert var_19 is False
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = b'rotate-test'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = 'hmac'
    var_28 = module_0.Signer(var_12, key_derivation=var_27)
    var_29 = var_28.get_signature(var_2)
    var_30 = var_28.verify_signature(var_2, var_29)
    assert var_30 is True
    var_31 = 'concat'
    var_32 = module_0.Signer(var_12, key_derivation=var_31)
    var_33 = var_32.get_signature(var_2)
    var_34 = var_32.verify_signature(var_2, var_33)
    assert var_34 is True
    var_35 = b'|'
    var_36 = module_0.Signer(var_12, sep=var_35)
    var_37 = b'test-sep'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True



# Parsed testcases at query #28
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
    var_8 = b'invalid'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotated value'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'secret'
    var_20 = module_0.NoneAlgorithm()
    var_21 = module_0.Signer(var_19, algorithm=var_20)
    var_22 = b'none test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'custom test'
    var_26 = 'other-key'
    var_27 = module_0.Signer(var_26)
    var_28 = b'different key'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_1.verify_signature(var_28, var_29)
    assert var_30 is False
    var_31 = 'secret'
    var_32 = b'derivation test'
    var_33 = module_0.Signer(var_12)
    var_34 = b'old value'
    var_35 = var_33.get_signature(var_34)
    var_36 = [var_12, var_13]
    var_37 = module_0.Signer(var_36)
    var_38 = var_37.verify_signature(var_34, var_35)
    assert var_38 is True
    var_39 = b'test'
    var_40 = b'!!!invalid base64!!!'
    var_41 = var_1.verify_signature(var_39, var_40)
    assert var_41 is False
    var_42 = b'.'
    var_43 = var_39 + var_42
    var_44 = b'value'
    var_45 = var_43 + var_44
    var_46 = var_1.get_signature(var_45)
    var_47 = var_1.verify_signature(var_45, var_46)
    assert var_47 is True
    var_48 = 'salt1'
    var_49 = module_0.Signer(var_19, var_48)
    var_50 = 'salt2'
    var_51 = module_0.Signer(var_19, var_50)
    var_52 = b'salt test'
    var_53 = var_49.get_signature(var_52)
    var_54 = var_49.verify_signature(var_52, var_53)
    assert var_54 is True
    var_55 = var_51.verify_signature(var_52, var_53)
    assert var_55 is False



# Parsed testcases at query #29
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
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_3.get_signature(var_9)
    var_11 = var_3.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'test-value'
    var_13 = var_3.verify_signature(var_12, var_5)
    assert var_13 is True
    var_14 = b'modified-value'
    var_15 = var_3.verify_signature(var_14, var_5)
    assert var_15 is False
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_2)
    var_20 = var_19.get_signature(var_4)
    var_21 = var_19.verify_signature(var_4, var_20)
    assert var_21 is True
    var_22 = module_0.NoneAlgorithm()
    var_23 = 'test-key'
    var_24 = module_0.Signer(var_23, algorithm=var_22)
    var_25 = var_24.get_signature(var_4)
    var_26 = var_24.verify_signature(var_4, var_25)
    assert var_26 is True
    var_27 = b'!!!invalid-base64!!!'
    var_28 = var_3.verify_signature(var_4, var_27)
    assert var_28 is False
    var_29 = b'\xff\xff\xff'
    var_30 = module_1.base64_encode(var_29)
    var_31 = var_3.verify_signature(var_4, var_30)
    assert var_31 is False



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
    var_15 = b'test-value-2'
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
    var_25 = 'custom-salt'
    var_26 = module_0.Signer(var_0, var_25)
    var_27 = b'test-value-3'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = module_0.Signer(var_0)
    var_31 = var_30.get_signature(var_27)
    var_32 = var_26.verify_signature(var_27, var_31)
    assert var_32 is False
    var_33 = b'|'
    var_34 = module_0.Signer(var_0, sep=var_33)
    var_35 = b'test-value-4'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = module_0.NoneAlgorithm()
    var_39 = module_0.Signer(var_0, algorithm=var_38)
    var_40 = b'test-value-5'
    var_41 = var_39.get_signature(var_40)
    var_42 = var_39.verify_signature(var_40, var_41)
    assert var_42 is True



# Parsed testcases at query #31
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
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = b'!!!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'test value'
    var_12 = var_1.verify_signature(var_11, var_3)
    assert var_12 is True
    var_13 = 'old-key'
    var_14 = 'new-key'
    var_15 = [var_13, var_14]
    var_16 = module_0.Signer(var_15)
    var_17 = b'test with rotation'
    var_18 = var_16.get_signature(var_17)
    var_19 = var_16.verify_signature(var_17, var_18)
    assert var_19 is True
    var_20 = b'different-salt'
    var_21 = module_0.Signer(var_0, var_20)
    var_22 = var_21.verify_signature(var_17, var_18)
    assert var_22 is False
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, algorithm=var_23)
    var_25 = b'test none algorithm'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = b'test hmac sha256'
    var_29 = b'wrong'



# Parsed testcases at query #32
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
    var_7 = b'invalid-signature'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_3.verify_signature(var_4, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_3.get_signature(var_10)
    var_12 = var_3.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = module_0.NoneAlgorithm()
    var_14 = module_0.Signer(var_1, var_2, algorithm=var_13)
    var_15 = var_14.get_signature(var_4)
    var_16 = var_14.verify_signature(var_4, var_15)
    assert var_16 is True
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19, var_2)
    var_21 = var_20.get_signature(var_4)
    var_22 = var_20.verify_signature(var_4, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid-base64!!!'
    var_24 = var_3.verify_signature(var_4, var_23)
    assert var_24 is False
    var_25 = 'test-string'
    var_26 = var_3.get_signature(var_25)
    var_27 = var_3.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'different-salt'
    var_29 = module_0.Signer(var_1, var_28)
    var_30 = var_29.get_signature(var_4)
    var_31 = var_3.verify_signature(var_4, var_30)
    assert var_31 is False
    var_32 = var_29.verify_signature(var_4, var_30)
    assert var_32 is True
    var_33 = b'|'
    var_34 = module_0.Signer(var_1, var_2, var_33)
    var_35 = var_34.get_signature(var_4)
    var_36 = var_34.verify_signature(var_4, var_35)
    assert var_36 is True
    var_37 = b''
    var_38 = var_3.verify_signature(var_4, var_37)
    assert var_38 is False
    var_39 = b'test.value'
    var_40 = var_3.get_signature(var_39)
    var_41 = var_3.verify_signature(var_39, var_40)
    assert var_41 is True



# Parsed testcases at query #33
#--------------------------




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
    var_6 = b'invalid-signature'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_2.verify_signature(var_3, var_9)
    assert var_10 is False
    var_11 = module_0.NoneAlgorithm()
    var_12 = module_0.Signer(var_0, var_1, algorithm=var_11)
    var_13 = b'test-value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18, var_1)
    var_20 = b'test-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.Signer(var_16, var_1)
    var_24 = var_23.get_signature(var_20)
    var_25 = var_19.verify_signature(var_20, var_24)
    assert var_25 is True
    var_26 = 'test-value'
    var_27 = var_2.verify_signature(var_26, var_21)
    assert var_27 is True
    var_28 = 'not-base64!!'
    var_29 = var_2.verify_signature(var_20, var_28)
    assert var_29 is False
    var_30 = b'-'
    var_31 = module_0.Signer(var_0, var_1, var_30)
    var_32 = b'test-value'
    var_33 = var_31.get_signature(var_32)
    var_34 = var_31.verify_signature(var_32, var_33)
    assert var_34 is True
    var_35 = var_2.verify_signature(var_9, var_33)
    assert var_35 is False
    var_36 = 'different-salt'
    var_37 = module_0.Signer(var_0, var_36)
    var_38 = var_37.verify_signature(var_32, var_33)
    assert var_38 is False
    var_39 = 'concat'
    var_40 = module_0.Signer(var_0, var_1, key_derivation=var_39)
    var_41 = b'test-value'
    var_42 = var_40.get_signature(var_41)
    var_43 = var_40.verify_signature(var_41, var_42)
    assert var_43 is True
    var_44 = var_2.verify_signature(var_41, var_42)
    assert var_44 is False



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'Test verify_signature method with various scenarios.'
    var_1 = 'test-secret-key'
    var_2 = 'test-salt'
    var_3 = module_0.Signer(var_1, var_2)
    var_4 = b'test-value'
    var_5 = var_3.get_signature(var_4)
    var_6 = var_3.verify_signature(var_4, var_5)
    assert var_6 is True
    var_7 = b'invalid-signature'
    var_8 = var_3.verify_signature(var_4, var_7)
    assert var_8 is False
    var_9 = b'different-value'
    var_10 = var_3.get_signature(var_9)
    var_11 = var_3.verify_signature(var_4, var_10)
    assert var_11 is False
    var_12 = 'ascii'
    var_13 = 'utf-8'
    var_14 = b'!!!invalid-base64!!!'
    var_15 = var_3.verify_signature(var_4, var_14)
    assert var_15 is False
    var_16 = b''
    var_17 = var_3.get_signature(var_16)
    var_18 = var_3.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = var_3.verify_signature(var_4, var_16)
    assert var_19 is False
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22, var_2)
    var_24 = b'test-value'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_1, algorithm=var_27)
    var_29 = var_28.get_signature(var_24)
    var_30 = var_28.verify_signature(var_24, var_29)
    assert var_30 is True
    var_31 = b'|'
    var_32 = module_0.Signer(var_1, sep=var_31)
    var_33 = b'test-value'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = b'test-value'
    var_37 = 'salt1'
    var_38 = module_0.Signer(var_1, var_37)
    var_39 = 'salt2'
    var_40 = module_0.Signer(var_1, var_39)
    var_41 = var_38.get_signature(var_36)
    var_42 = var_40.get_signature(var_36)
    var_43 = var_40.verify_signature(var_36, var_41)
    assert var_43 is False
    var_44 = 'héllo wörld'
    var_45 = var_3.get_signature(var_44)
    var_46 = var_3.verify_signature(var_44, var_45)
    assert var_46 is True



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
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
    var_34 = 'unknown'
    var_35 = module_0.Signer(var_0, var_1, key_derivation=var_34)
    var_36 = var_35.derive_key()



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.signer as module_0
import hmac as module_1

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = 'test-salt'
    var_2 = module_0.Signer(var_0, var_1)
    var_3 = var_2.derive_key()
    var_4 = b'test-salt'
    var_5 = b'signer'
    var_6 = var_4 + var_5
    var_7 = b'test-secret-key'
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
    var_22 = b'test-secret-key'
    var_23 = 'explicit-key'
    var_24 = var_2.derive_key(var_23)
    var_25 = var_4 + var_5
    var_26 = b'explicit-key'
    var_27 = var_25 + var_26
    var_28 = module_1.digest()
    var_29 = b'bytes-secret'
    var_30 = b'bytes-salt'
    var_31 = module_0.Signer(var_29, var_30)
    var_32 = var_31.derive_key()
    var_33 = var_30 + var_5
    var_34 = var_33 + var_29
    var_35 = module_1.digest()
    var_36 = 'test-secret'
    var_37 = var_4 + var_5
    var_38 = b'test-secret'
    var_39 = var_37 + var_38
    var_40 = module_1.digest()
    var_41 = 'old-key'
    var_42 = 'new-key'
    var_43 = [var_41, var_42]
    var_44 = module_0.Signer(var_43, var_1)
    var_45 = var_44.derive_key()
    var_46 = var_4 + var_5
    var_47 = b'new-key'
    var_48 = var_46 + var_47
    var_49 = module_1.digest()
    var_50 = var_44.derive_key(var_41)
    var_51 = var_4 + var_5
    var_52 = b'old-key'
    var_53 = var_51 + var_52
    var_54 = module_1.digest()
    var_55 = 'test-key'
    var_56 = 'invalid'
    var_57 = module_0.Signer(var_55, var_1, key_derivation=var_56)
    var_58 = var_57.derive_key()
    var_59 = None
    var_60 = module_0.Signer(var_55, var_59)
    var_61 = var_60.derive_key()
    var_62 = b'itsdangerous.Signer'
    var_63 = var_62 + var_5
    var_64 = b'test-key'
    var_65 = var_63 + var_64
    var_66 = module_1.digest()
    var_67 = 'custom-salt'
    var_68 = module_0.Signer(var_55, var_67)
    var_69 = var_68.derive_key()
    var_70 = b'custom-salt'
    var_71 = var_70 + var_5
    var_72 = var_71 + var_64
    var_73 = module_1.digest()



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
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'!!!invalid-base64!!!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'test-string-value'
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'rotation-test'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'custom-salt'
    var_22 = module_0.Signer(var_0, var_21)
    var_23 = b'salt-test'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = 'different-salt'
    var_27 = module_0.Signer(var_0, var_26)
    var_28 = var_27.verify_signature(var_23, var_24)
    assert var_28 is False
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_0, algorithm=var_29)
    var_31 = b'none-alg-test'
    var_32 = var_30.get_signature(var_31)
    var_33 = module_1.base64_encode(var_7)
    var_34 = var_30.verify_signature(var_31, var_32)
    assert var_34 is True
    var_35 = b'sha256-test'
    var_36 = 'hmac'
    var_37 = module_0.Signer(var_0, key_derivation=var_36)
    var_38 = b'hmac-derivation-test'
    var_39 = var_37.get_signature(var_38)
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = 'concat'
    var_42 = module_0.Signer(var_0, key_derivation=var_41)
    var_43 = b'concat-derivation-test'
    var_44 = var_42.get_signature(var_43)
    var_45 = var_42.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = 'none'
    var_47 = module_0.Signer(var_0, key_derivation=var_46)
    var_48 = b'none-derivation-test'
    var_49 = var_47.get_signature(var_48)
    var_50 = var_47.verify_signature(var_48, var_49)
    assert var_50 is True
    var_51 = module_0.Signer(var_0)
    var_52 = b'bytes-value'
    var_53 = 'string-value'
    var_54 = var_51.get_signature(var_52)
    var_55 = var_51.get_signature(var_53)
    var_56 = var_51.verify_signature(var_52, var_54)
    assert var_56 is True
    var_57 = var_51.verify_signature(var_53, var_55)
    assert var_57 is True
    var_58 = var_51.verify_signature(var_52, var_55)
    assert var_58 is False
    var_59 = var_51.verify_signature(var_53, var_54)
    assert var_59 is False



# Parsed testcases at query #4
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
    var_22 = 'custom-secret'
    var_23 = var_2.derive_key(var_22)
    var_24 = var_4 + var_5
    var_25 = b'custom-secret'
    var_26 = var_24 + var_25
    var_27 = module_1.digest()
    var_28 = b'secret-key-bytes'
    var_29 = b'test-salt-bytes'
    var_30 = module_0.Signer(var_28, var_29)
    var_31 = var_30.derive_key()
    var_32 = var_29 + var_5
    var_33 = var_32 + var_28
    var_34 = module_1.digest()
    var_35 = None
    var_36 = module_0.Signer(var_0, var_35)
    var_37 = var_36.derive_key()
    var_38 = b'itsdangerous.Signer'
    var_39 = var_38 + var_5
    var_40 = var_39 + var_7
    var_41 = module_1.digest()
    var_42 = 'unknown'
    var_43 = module_0.Signer(var_0, key_derivation=var_42)
    var_44 = var_43.derive_key()



# Parsed testcases at query #5
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
    var_12 = 'string-value'
    var_13 = var_2.get_signature(var_12)
    var_14 = var_2.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = 'rotation-salt'
    var_19 = module_0.Signer(var_17, var_18)
    var_20 = b'rotation-value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'secret'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_23, algorithm=var_24)
    var_26 = b'test'
    var_27 = var_25.get_signature(var_26)
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = b'!@#$%^&*'
    var_30 = var_2.verify_signature(var_26, var_29)
    assert var_30 is False
    var_31 = module_0.HMACAlgorithm()
    var_32 = module_0.Signer(var_0, algorithm=var_31)
    var_33 = b'hmac-test'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = 'concat'
    var_37 = module_0.Signer(var_0, key_derivation=var_36)
    var_38 = b'concat-test'
    var_39 = var_37.get_signature(var_38)
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = 'django-concat'
    var_42 = module_0.Signer(var_0, key_derivation=var_41)
    var_43 = b'django-test'
    var_44 = var_42.get_signature(var_43)
    var_45 = var_42.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = 'hmac'
    var_47 = module_0.Signer(var_0, key_derivation=var_46)
    var_48 = b'hmac-derivation-test'
    var_49 = var_47.get_signature(var_48)
    var_50 = var_47.verify_signature(var_48, var_49)
    assert var_50 is True
    var_51 = 'none'
    var_52 = module_0.Signer(var_0, key_derivation=var_51)
    var_53 = b'none-derivation-test'
    var_54 = var_52.get_signature(var_53)
    var_55 = var_52.verify_signature(var_53, var_54)
    assert var_55 is True
    var_56 = b'custom-digest'
    var_57 = b'|'
    var_58 = module_0.Signer(var_0, sep=var_57)
    var_59 = b'sep-test'
    var_60 = var_58.get_signature(var_59)
    var_61 = var_58.verify_signature(var_59, var_60)
    assert var_61 is True



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
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'test-string'
    var_12 = var_1.get_signature(var_11)
    var_13 = var_1.verify_signature(var_11, var_12)
    assert var_13 is True
    var_14 = '!!!invalid-base64!!!'
    var_15 = var_1.verify_signature(var_2, var_14)
    assert var_15 is False
    var_16 = module_0.NoneAlgorithm()
    var_17 = module_0.Signer(var_0, algorithm=var_16)
    var_18 = b'test-value-2'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'old-key'
    var_22 = 'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23)
    var_25 = b'test-value-3'
    var_26 = var_24.get_signature(var_25)
    var_27 = var_24.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = 'custom-salt'
    var_29 = module_0.Signer(var_0, var_28)
    var_30 = b'test-value-4'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True
    var_33 = 'different-salt'
    var_34 = module_0.Signer(var_0, var_33)
    var_35 = var_34.verify_signature(var_30, var_31)
    assert var_35 is False
    var_36 = 'concat'
    var_37 = module_0.Signer(var_0, key_derivation=var_36)
    var_38 = b'test-value-5'
    var_39 = var_37.get_signature(var_38)
    var_40 = var_37.verify_signature(var_38, var_39)
    assert var_40 is True
    var_41 = 'hmac'
    var_42 = module_0.Signer(var_0, key_derivation=var_41)
    var_43 = b'test-value-6'
    var_44 = var_42.get_signature(var_43)
    var_45 = var_42.verify_signature(var_43, var_44)
    assert var_45 is True



# Parsed testcases at query #7
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
    var_8 = b'!!!invalid-base64!!!'
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12, var_1)
    var_14 = b'test-value-2'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_0, algorithm=var_17)
    var_19 = b'test-value-3'
    var_20 = var_18.get_signature(var_19)
    var_21 = b''
    var_22 = module_1.base64_encode(var_21)
    var_23 = var_18.verify_signature(var_19, var_20)
    assert var_23 is True
    var_24 = 'test-value'
    var_25 = var_2.verify_signature(var_24, var_4)
    assert var_25 is True
    var_26 = var_2.get_signature(var_21)
    var_27 = var_2.verify_signature(var_21, var_26)
    assert var_27 is True
    var_28 = var_18.verify_signature(var_21, var_21)
    assert var_28 is True



# Parsed testcases at query #8
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
    var_11 = b'|'
    var_12 = module_0.Signer(var_0, sep=var_11)
    var_13 = b'another-value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'rotation-test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = 'secret'
    var_24 = module_0.NoneAlgorithm()
    var_25 = module_0.Signer(var_23, algorithm=var_24)
    var_26 = b'none-algorithm'
    var_27 = var_25.get_signature(var_26)
    assert var_27 == b''
    var_28 = var_25.verify_signature(var_26, var_27)
    assert var_28 is True
    var_29 = 'concat'
    var_30 = module_0.Signer(var_23, key_derivation=var_29)
    var_31 = b'concat-derivation'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = b'sha256-test'
    var_35 = b'custom-salt'
    var_36 = module_0.Signer(var_23, var_35)
    var_37 = b'salted-value'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = b'!!!invalid-base64!!!'
    var_41 = var_1.verify_signature(var_2, var_40)
    assert var_41 is False
    var_42 = 'different-key'
    var_43 = module_0.Signer(var_42)
    var_44 = var_43.verify_signature(var_2, var_3)
    assert var_44 is False
    var_45 = var_1.get_signature(var_9)
    var_46 = var_1.verify_signature(var_9, var_45)
    assert var_46 is True
    var_47 = b'x'
    var_48 = 10000
    var_49 = var_47 * var_48
    var_50 = var_1.get_signature(var_49)
    var_51 = var_1.verify_signature(var_49, var_50)
    assert var_51 is True



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
    var_16 = b'test'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = var_15.salt
    var_20 = var_15.sep
    var_21 = module_0.Signer(var_12, var_19, var_20)
    var_22 = var_21.get_signature(var_16)
    var_23 = var_15.verify_signature(var_16, var_22)
    assert var_23 is True
    var_24 = 'different-salt'
    var_25 = module_0.Signer(var_0, var_24)
    var_26 = var_25.get_signature(var_16)
    var_27 = var_1.verify_signature(var_16, var_26)
    assert var_27 is False
    var_28 = b'|'
    var_29 = module_0.Signer(var_0, sep=var_28)
    var_30 = var_29.get_signature(var_16)
    var_31 = var_29.verify_signature(var_16, var_30)
    assert var_31 is True
    var_32 = b'!!!invalid-base64!!!'
    var_33 = var_1.verify_signature(var_16, var_32)
    assert var_33 is False
    var_34 = module_0.NoneAlgorithm()
    var_35 = module_0.Signer(var_0, algorithm=var_34)
    var_36 = var_35.get_signature(var_16)
    var_37 = var_35.verify_signature(var_16, var_36)
    assert var_37 is True
    var_38 = 'secret-key'



# Parsed testcases at query #10
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
    var_7 = b'invalid_sig'
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = module_0.NoneAlgorithm()
    var_13 = module_0.Signer(var_0, algorithm=var_12)
    var_14 = b'test value 2'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'rotation test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'!!!invalid base64!!!'
    var_25 = var_1.verify_signature(var_2, var_24)
    assert var_25 is False
    var_26 = b'original value'
    var_27 = var_1.get_signature(var_26)
    var_28 = b'modified value'
    var_29 = var_1.verify_signature(var_28, var_27)
    assert var_29 is False
    var_30 = b'different-salt'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = var_31.get_signature(var_2)
    var_33 = var_31.verify_signature(var_2, var_32)
    assert var_33 is True
    var_34 = var_1.verify_signature(var_2, var_32)
    assert var_34 is False



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
    var_7 = b'modified-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is False
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = 'different-salt'
    var_13 = module_0.Signer(var_0, var_12)
    var_14 = var_13.get_signature(var_2)
    var_15 = var_1.verify_signature(var_2, var_14)
    assert var_15 is False
    var_16 = 'concat'
    var_17 = module_0.Signer(var_0, key_derivation=var_16)
    var_18 = var_17.get_signature(var_2)
    var_19 = var_1.verify_signature(var_2, var_18)
    assert var_19 is True
    var_20 = var_1.verify_signature(var_2, var_3)
    assert var_20 is False
    var_21 = 'old-key'
    var_22 = 'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23)
    var_25 = var_24.get_signature(var_2)
    var_26 = var_24.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = b'!!!invalid-base64!!!'
    var_28 = var_1.verify_signature(var_2, var_27)
    assert var_28 is False
    var_29 = 'test-value'
    var_30 = var_1.verify_signature(var_29, var_3)
    assert var_30 is True
    var_31 = module_0.NoneAlgorithm()
    var_32 = module_0.Signer(var_0, algorithm=var_31)
    var_33 = var_32.get_signature(var_2)
    var_34 = var_32.verify_signature(var_2, var_33)
    assert var_34 is True



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'secret-key'
    assert var_0 is True
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
    var_18 = b'test-value-2'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'different-salt'
    var_22 = module_0.Signer(var_0, var_21)
    var_23 = b'test-value-3'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = var_1.verify_signature(var_23, var_24)
    assert var_26 is False
    var_27 = b'|'
    var_28 = module_0.Signer(var_0, sep=var_27)
    var_29 = b'test-value-4'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = module_0.NoneAlgorithm()
    var_33 = module_0.Signer(var_0, algorithm=var_32)
    var_34 = b'test-value-5'
    var_35 = var_33.get_signature(var_34)
    var_36 = var_33.verify_signature(var_34, var_35)
    assert var_36 is True
    var_37 = b'anything'
    var_38 = var_33.verify_signature(var_34, var_37)
    assert var_38 is True



# Parsed testcases at query #13
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
    var_8 = b'wrong_signature'
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.get_signature(var_10)
    var_12 = var_1.verify_signature(var_10, var_11)
    assert var_12 is True
    var_13 = b'|'
    var_14 = module_0.Signer(var_0, sep=var_13)
    var_15 = b'another value'
    var_16 = var_14.get_signature(var_15)
    var_17 = var_14.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = b'rotated key value'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = b'!!!invalid_base64!!!'
    var_26 = var_1.verify_signature(var_2, var_25)
    assert var_26 is False
    var_27 = module_0.NoneAlgorithm()
    var_28 = module_0.Signer(var_0, algorithm=var_27)
    var_29 = b'none algorithm test'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'sha256 test'
    var_33 = b'custom-salt'
    var_34 = module_0.Signer(var_0, var_33)
    var_35 = b'salted value'
    var_36 = var_34.get_signature(var_35)
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True



# Parsed testcases at query #14
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
    var_9 = var_2.verify_signature(var_3, var_8)
    assert var_9 is False
    var_10 = b'!!!'
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = 'test value'
    var_13 = var_2.verify_signature(var_12, var_4)
    assert var_13 is True
    var_14 = module_0.NoneAlgorithm()
    var_15 = module_0.Signer(var_0, algorithm=var_14)
    var_16 = var_15.get_signature(var_3)
    var_17 = var_15.verify_signature(var_3, var_16)
    assert var_17 is True
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20, var_1)
    var_22 = b'test value'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = module_0.Signer(var_18, var_1)
    var_26 = var_25.get_signature(var_22)
    var_27 = var_21.verify_signature(var_22, var_26)
    assert var_27 is True



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.signer as module_0

def test_case_0():
    var_0 = 'test-secret-key'
    var_1 = module_0.Signer(var_0)
    var_2 = b'test-value'
    var_3 = var_1.get_signature(var_2)
    var_4 = var_1.verify_signature(var_2, var_3)
    assert var_4 is True
    var_5 = b'wrong-signature'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = 'test-value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is True
    var_9 = 'old-key'
    var_10 = 'new-key'
    var_11 = [var_9, var_10]
    var_12 = module_0.Signer(var_11)
    var_13 = b'rotation-test'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = b'!!!invalid-base64!!!'
    var_17 = var_1.verify_signature(var_13, var_16)
    assert var_17 is False
    var_18 = b''
    var_19 = var_1.get_signature(var_18)
    var_20 = var_1.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = 'test-secret'
    var_22 = b'|'
    var_23 = module_0.Signer(var_21, sep=var_22)
    var_24 = b'custom-sep'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'custom-salt'
    var_28 = module_0.Signer(var_21, var_27)
    var_29 = b'salt-test'
    var_30 = var_28.get_signature(var_29)
    var_31 = var_28.verify_signature(var_29, var_30)
    assert var_31 is True
    var_32 = b'different-salt'
    var_33 = module_0.Signer(var_21, var_32)
    var_34 = var_33.verify_signature(var_29, var_30)
    assert var_34 is False
    var_35 = 'hmac'
    var_36 = module_0.Signer(var_21, key_derivation=var_35)
    var_37 = b'hmac-test'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = 'concat'
    var_41 = module_0.Signer(var_21, key_derivation=var_40)
    var_42 = b'concat-test'
    var_43 = var_41.get_signature(var_42)
    var_44 = var_41.verify_signature(var_42, var_43)
    assert var_44 is True
    var_45 = module_0.NoneAlgorithm()
    var_46 = module_0.Signer(var_21, algorithm=var_45)
    var_47 = b'none-algorithm'
    var_48 = var_46.get_signature(var_47)
    var_49 = var_46.verify_signature(var_47, var_48)
    assert var_49 is True
    var_50 = b'bytes-secret-key'
    var_51 = module_0.Signer(var_50)
    var_52 = b'bytes-test'
    var_53 = var_51.get_signature(var_52)
    var_54 = var_51.verify_signature(var_52, var_53)
    assert var_54 is True



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
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b''
    var_8 = var_1.verify_signature(var_2, var_7)
    assert var_8 is False
    var_9 = b'!!!invalid-base64!!!'
    var_10 = var_1.verify_signature(var_2, var_9)
    assert var_10 is False
    var_11 = 'test-value'
    var_12 = var_1.verify_signature(var_11, var_3)
    assert var_12 is True
    var_13 = 'different-salt'
    var_14 = module_0.Signer(var_0, var_13)
    var_15 = var_14.get_signature(var_2)
    var_16 = var_1.verify_signature(var_2, var_15)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = var_20.get_signature(var_2)
    var_22 = var_20.verify_signature(var_2, var_21)
    assert var_22 is True
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_0, algorithm=var_23)
    var_25 = var_24.get_signature(var_2)
    var_26 = var_24.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = b'anything'
    var_28 = var_24.verify_signature(var_2, var_27)
    assert var_28 is False
    var_29 = module_0.NoneAlgorithm()
    var_30 = module_0.Signer(var_0, algorithm=var_29)
    var_31 = module_1.base64_encode(var_7)
    var_32 = var_30.verify_signature(var_2, var_31)
    assert var_32 is True
    var_33 = b'wrong'
    var_34 = 'test-key'
    var_35 = b'|'
    var_36 = module_0.Signer(var_34, sep=var_35)
    var_37 = b'test|value'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True



# Parsed testcases at query #17
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
    var_9 = 'test-value'
    var_10 = var_1.verify_signature(var_9, var_3)
    assert var_10 is True
    var_11 = b'|'
    var_12 = module_0.Signer(var_0, sep=var_11)
    var_13 = b'another-value'
    var_14 = var_12.get_signature(var_13)
    var_15 = var_12.verify_signature(var_13, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'key-rotation-test'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = module_0.Signer(var_16)
    var_24 = var_23.verify_signature(var_20, var_21)
    assert var_24 is False
    var_25 = b'custom-salt'
    var_26 = module_0.Signer(var_0, var_25)
    var_27 = b'salted-value'
    var_28 = var_26.get_signature(var_27)
    var_29 = var_26.verify_signature(var_27, var_28)
    assert var_29 is True
    var_30 = b'different-salt'
    var_31 = module_0.Signer(var_0, var_30)
    var_32 = var_31.verify_signature(var_27, var_28)
    assert var_32 is False
    var_33 = module_0.NoneAlgorithm()
    var_34 = module_0.Signer(var_0, algorithm=var_33)
    var_35 = b'none-algorithm'
    var_36 = var_34.get_signature(var_35)
    assert var_36 == b''
    var_37 = var_34.verify_signature(var_35, var_36)
    assert var_37 is True
    var_38 = b'hmac-sha256'
    var_39 = b'corrupt'
    var_40 = var_3 + var_39
    var_41 = var_1.verify_signature(var_2, var_40)
    assert var_41 is False
    var_42 = b'different-value'
    var_43 = var_1.verify_signature(var_42, var_3)
    assert var_43 is False
    var_44 = '!!invalid-base64!!'
    var_45 = var_1.verify_signature(var_2, var_44)
    assert var_45 is False



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
    var_5 = 'test-value'
    var_6 = var_1.verify_signature(var_5, var_3)
    assert var_6 is True
    var_7 = b'wrong-signature'
    var_8 = module_1.base64_encode(var_7)
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b''
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b'!!!invalid-base64!!!'
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
    var_21 = 'salt1'
    var_22 = module_0.Signer(var_0, var_21)
    var_23 = 'salt2'
    var_24 = module_0.Signer(var_0, var_23)
    var_25 = b'test-value'
    var_26 = var_22.get_signature(var_25)
    var_27 = var_22.verify_signature(var_25, var_26)
    assert var_27 is True
    var_28 = var_24.verify_signature(var_25, var_26)
    assert var_28 is False
    var_29 = b'|'
    var_30 = module_0.Signer(var_0, sep=var_29)
    var_31 = b'test-value'
    var_32 = var_30.get_signature(var_31)
    var_33 = var_30.verify_signature(var_31, var_32)
    assert var_33 is True
    var_34 = module_0.NoneAlgorithm()
    var_35 = module_0.Signer(var_0, algorithm=var_34)
    var_36 = b'test-value'
    var_37 = var_35.get_signature(var_36)
    var_38 = var_35.verify_signature(var_36, var_37)
    assert var_38 is True
    var_39 = var_35.verify_signature(var_36, var_10)
    assert var_39 is True
    var_40 = b'test-value'
    var_41 = var_35.get_signature(var_40)
    var_42 = var_35.verify_signature(var_40, var_41)
    assert var_42 is True
    var_43 = 'secret-key'
    var_44 = b'test-value'
    var_45 = var_35.get_signature(var_44)
    var_46 = var_35.verify_signature(var_44, var_45)
    assert var_46 is True
    var_47 = module_0.Signer(var_43)
    var_48 = b''
    var_49 = var_47.get_signature(var_48)
    var_50 = var_47.verify_signature(var_48, var_49)
    assert var_50 is True
    var_51 = module_0.Signer(var_43)
    var_52 = b'test.value'
    var_53 = var_51.get_signature(var_52)
    var_54 = var_51.verify_signature(var_52, var_53)
    assert var_54 is True



# Parsed testcases at query #19
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
    var_15 = 'old-key'
    var_16 = 'new-key'
    var_17 = [var_15, var_16]
    var_18 = module_0.Signer(var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = 'different-salt'
    var_22 = module_0.Signer(var_0, var_21)
    var_23 = var_22.get_signature(var_2)
    var_24 = var_1.verify_signature(var_2, var_23)
    assert var_24 is False
    var_25 = var_22.verify_signature(var_2, var_23)
    assert var_25 is True
    var_26 = 'concat'
    var_27 = module_0.Signer(var_0, key_derivation=var_26)
    var_28 = var_27.get_signature(var_2)
    var_29 = var_27.verify_signature(var_2, var_28)
    assert var_29 is True
    var_30 = var_1.verify_signature(var_2, var_28)
    assert var_30 is False
    var_31 = 'hmac'
    var_32 = module_0.Signer(var_0, key_derivation=var_31)
    var_33 = var_32.get_signature(var_2)
    var_34 = var_32.verify_signature(var_2, var_33)
    assert var_34 is True
    var_35 = var_1.verify_signature(var_2, var_33)
    assert var_35 is False
    var_36 = 'none'
    var_37 = module_0.Signer(var_0, key_derivation=var_36)
    var_38 = var_37.get_signature(var_2)
    var_39 = var_37.verify_signature(var_2, var_38)
    assert var_39 is True
    var_40 = var_1.verify_signature(var_2, var_38)
    assert var_40 is False



# Parsed testcases at query #20
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
    var_7 = 'test value'
    var_8 = var_1.verify_signature(var_7, var_3)
    assert var_8 is True
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'!!!invalid!!!'
    var_13 = var_1.verify_signature(var_2, var_12)
    assert var_13 is False
    var_14 = 'old-key'
    var_15 = 'new-key'
    var_16 = [var_14, var_15]
    var_17 = module_0.Signer(var_16)
    var_18 = b'test value 2'
    var_19 = var_17.get_signature(var_18)
    var_20 = var_17.verify_signature(var_18, var_19)
    assert var_20 is True
    var_21 = module_0.NoneAlgorithm()
    var_22 = module_0.Signer(var_0, algorithm=var_21)
    var_23 = b'test value 3'
    var_24 = var_22.get_signature(var_23)
    var_25 = var_22.verify_signature(var_23, var_24)
    assert var_25 is True
    var_26 = b'|'
    var_27 = module_0.Signer(var_0, sep=var_26)
    var_28 = b'test value 4'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = 'hmac'
    var_32 = module_0.Signer(var_0, key_derivation=var_31)
    var_33 = b'test value 5'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = b'wrong value'
    var_37 = var_1.verify_signature(var_36, var_3)
    assert var_37 is False
    var_38 = var_1.verify_signature(var_2, var_3)
    assert var_38 is True
    var_39 = 'different-secret'
    var_40 = module_0.Signer(var_39)
    var_41 = var_40.get_signature(var_2)
    var_42 = var_1.verify_signature(var_2, var_41)
    assert var_42 is False
    var_43 = b'test value with \xc3\xa9'
    var_44 = var_1.get_signature(var_43)
    var_45 = var_1.verify_signature(var_43, var_44)
    assert var_45 is True



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
    var_5 = b'invalid'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'wrong-value'
    var_11 = var_1.verify_signature(var_10, var_3)
    assert var_11 is False
    var_12 = 'old-key'
    var_13 = 'new-key'
    var_14 = [var_12, var_13]
    var_15 = module_0.Signer(var_14)
    var_16 = b'rotation-test'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = 'test-value'
    var_20 = var_1.verify_signature(var_19, var_3)
    assert var_20 is True
    var_21 = 'key'
    var_22 = module_0.NoneAlgorithm()
    var_23 = module_0.Signer(var_21, algorithm=var_22)
    var_24 = b'none-test'
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = b'!!!invalid-base64!!!'
    var_28 = var_1.verify_signature(var_2, var_27)
    assert var_28 is False



# Parsed testcases at query #22
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
    var_11 = b'test'
    var_12 = var_2.verify_signature(var_11, var_4)
    assert var_12 is False
    var_13 = 'test-value'
    var_14 = var_2.verify_signature(var_13, var_4)
    assert var_14 is True
    var_15 = b'!!!invalid-base64!!!'
    var_16 = var_2.verify_signature(var_3, var_15)
    assert var_16 is False
    var_17 = module_0.NoneAlgorithm()
    var_18 = module_0.Signer(var_0, algorithm=var_17)
    var_19 = var_18.get_signature(var_3)
    var_20 = var_18.verify_signature(var_3, var_19)
    assert var_20 is True
    var_21 = 'old-key'
    var_22 = 'new-key'
    var_23 = [var_21, var_22]
    var_24 = module_0.Signer(var_23, var_1)
    var_25 = var_24.get_signature(var_3)
    var_26 = var_24.verify_signature(var_3, var_25)
    assert var_26 is True
    var_27 = b'wrong-sig'
    var_28 = 'hmac'
    var_29 = module_0.Signer(var_0, key_derivation=var_28)
    var_30 = var_29.get_signature(var_3)
    var_31 = var_29.verify_signature(var_3, var_30)
    assert var_31 is True
    var_32 = 'none'
    var_33 = module_0.Signer(var_0, key_derivation=var_32)
    var_34 = var_33.get_signature(var_3)
    var_35 = var_33.verify_signature(var_3, var_34)
    assert var_35 is True



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
    var_9 = b''
    var_10 = var_1.get_signature(var_9)
    var_11 = var_1.verify_signature(var_9, var_10)
    assert var_11 is True
    var_12 = b'different-salt'
    var_13 = module_0.Signer(var_0, var_12)
    var_14 = var_13.get_signature(var_2)
    var_15 = var_13.verify_signature(var_2, var_14)
    assert var_15 is True
    var_16 = var_1.verify_signature(var_2, var_14)
    assert var_16 is False
    var_17 = b':'
    var_18 = module_0.Signer(var_0, sep=var_17)
    var_19 = var_18.get_signature(var_2)
    var_20 = var_18.verify_signature(var_2, var_19)
    assert var_20 is True
    var_21 = var_1.verify_signature(var_2, var_19)
    assert var_21 is False
    var_22 = 'old-key'
    var_23 = 'new-key'
    var_24 = [var_22, var_23]
    var_25 = module_0.Signer(var_24)
    var_26 = var_25.get_signature(var_2)
    var_27 = var_25.verify_signature(var_2, var_26)
    assert var_27 is True
    var_28 = module_0.Signer(var_22)
    var_29 = var_28.get_signature(var_2)
    var_30 = var_25.verify_signature(var_2, var_29)
    assert var_30 is True
    var_31 = 'test-value'
    var_32 = var_1.verify_signature(var_31, var_3)
    assert var_32 is True
    var_33 = 'ascii'
    var_34 = module_0.NoneAlgorithm()
    var_35 = module_0.Signer(var_0, algorithm=var_34)
    var_36 = var_35.get_signature(var_2)
    var_37 = var_35.verify_signature(var_2, var_36)
    assert var_37 is True
    var_38 = b'anything'
    var_39 = var_35.verify_signature(var_2, var_38)
    assert var_39 is True
    var_40 = 'secret-key'
    var_41 = b'a'
    var_42 = 10000
    var_43 = var_41 * var_42
    var_44 = var_1.get_signature(var_43)
    var_45 = var_1.verify_signature(var_43, var_44)
    assert var_45 is True
    var_46 = b'\x00\x01\x02\xff'
    var_47 = var_1.get_signature(var_46)
    var_48 = var_1.verify_signature(var_46, var_47)
    assert var_48 is True



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.signer as module_0
import src.itsdangerous.encoding as module_1

def test_case_0():
    var_0 = 'Test Signer.verify_signature method.'
    var_1 = 'secret-key'
    var_2 = module_0.Signer(var_1)
    var_3 = b'test-value'
    var_4 = var_2.get_signature(var_3)
    var_5 = var_2.verify_signature(var_3, var_4)
    assert var_5 is True
    var_6 = b'invalid'
    var_7 = module_1.base64_encode(var_6)
    var_8 = var_2.verify_signature(var_3, var_7)
    assert var_8 is False
    var_9 = b'x'
    var_10 = var_4 + var_9
    var_11 = var_2.verify_signature(var_3, var_10)
    assert var_11 is False
    var_12 = b''
    var_13 = var_2.verify_signature(var_3, var_12)
    assert var_13 is False
    var_14 = 'test-value'
    var_15 = var_2.verify_signature(var_14, var_4)
    assert var_15 is True
    var_16 = 'different-key'
    var_17 = module_0.Signer(var_16)
    var_18 = var_17.get_signature(var_3)
    var_19 = var_2.verify_signature(var_3, var_18)
    assert var_19 is False
    var_20 = 'old-key'
    var_21 = 'new-key'
    var_22 = [var_20, var_21]
    var_23 = module_0.Signer(var_22)
    var_24 = module_0.Signer(var_20)
    var_25 = var_24.get_signature(var_3)
    var_26 = var_23.verify_signature(var_3, var_25)
    assert var_26 is True
    var_27 = 'key'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_27, algorithm=var_28)
    var_30 = var_29.get_signature(var_3)
    var_31 = var_29.verify_signature(var_3, var_30)
    assert var_31 is True
    var_32 = b'!!!invalid!!!'
    var_33 = var_2.verify_signature(var_3, var_32)
    assert var_33 is False



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
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.verify_signature(var_2, var_8)
    assert var_9 is False
    var_10 = b'invalid-base64!!!'
    var_11 = var_1.verify_signature(var_2, var_10)
    assert var_11 is False
    var_12 = b''
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_12, var_13)
    assert var_14 is True
    var_15 = 'test-string'
    var_16 = var_1.get_signature(var_15)
    var_17 = var_1.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'old-key'
    var_19 = 'new-key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Signer(var_20)
    var_22 = b'rotate-test'
    var_23 = var_21.get_signature(var_22)
    var_24 = var_21.verify_signature(var_22, var_23)
    assert var_24 is True
    var_25 = 'test'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = b'none-alg-test'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b'different-salt'
    var_32 = module_0.Signer(var_0, var_31)
    var_33 = b'salt-test'
    var_34 = var_32.get_signature(var_33)
    var_35 = var_32.verify_signature(var_33, var_34)
    assert var_35 is True
    var_36 = var_1.verify_signature(var_33, var_34)
    assert var_36 is False



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
    var_13 = var_1.get_signature(var_12)
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'test-value'
    var_16 = var_1.verify_signature(var_15, var_13)
    assert var_16 is False
    var_17 = 'old-key'
    var_18 = 'new-key'
    var_19 = [var_17, var_18]
    var_20 = module_0.Signer(var_19)
    var_21 = b'test'
    var_22 = var_20.get_signature(var_21)
    var_23 = var_20.verify_signature(var_21, var_22)
    assert var_23 is True
    var_24 = b'wrong-sig'
    var_25 = module_1.base64_encode(var_24)
    var_26 = var_20.verify_signature(var_21, var_25)
    assert var_26 is False
    var_27 = var_20.get_signature(var_21)
    var_28 = var_20.verify_signature(var_21, var_27)
    assert var_28 is True
    var_29 = b'test'
    var_30 = var_20.get_signature(var_29)
    var_31 = 'test'
    var_32 = var_20.verify_signature(var_31, var_30)
    assert var_32 is True
    var_33 = var_20.get_signature(var_8)
    var_34 = var_20.verify_signature(var_8, var_33)
    assert var_34 is True
    var_35 = var_20.verify_signature(var_8, var_8)
    assert var_35 is False



# Parsed testcases at query #27
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
    var_8 = 'test value'
    var_9 = var_2.verify_signature(var_8, var_4)
    assert var_9 is True
    var_10 = b'modified value'
    var_11 = var_2.verify_signature(var_10, var_4)
    assert var_11 is False
    var_12 = module_0.NoneAlgorithm()
    var_13 = module_0.Signer(var_0, algorithm=var_12)
    var_14 = var_13.get_signature(var_3)
    var_15 = var_13.verify_signature(var_3, var_14)
    assert var_15 is True
    var_16 = 'old-key'
    var_17 = 'new-key'
    var_18 = [var_16, var_17]
    var_19 = module_0.Signer(var_18)
    var_20 = b'test value'
    var_21 = var_19.get_signature(var_20)
    var_22 = var_19.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid base64!!!'
    var_24 = var_2.verify_signature(var_20, var_23)
    assert var_24 is False
    var_25 = b''
    var_26 = var_2.get_signature(var_25)
    var_27 = var_2.verify_signature(var_25, var_26)
    assert var_27 is True



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
    var_5 = b'invalid-sig'
    var_6 = var_1.verify_signature(var_2, var_5)
    assert var_6 is False
    var_7 = b'!!!invalid-base64!!!'
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
    var_16 = b'test-value-2'
    var_17 = var_15.get_signature(var_16)
    var_18 = var_15.verify_signature(var_16, var_17)
    assert var_18 is True
    var_19 = module_0.Signer(var_12)
    var_20 = var_19.get_signature(var_16)
    var_21 = var_15.verify_signature(var_16, var_20)
    assert var_21 is True
    var_22 = 'another-key'
    var_23 = module_0.Signer(var_22)
    var_24 = b''
    var_25 = var_23.get_signature(var_24)
    var_26 = var_23.verify_signature(var_24, var_25)
    assert var_26 is True
    var_27 = 'key'
    var_28 = module_0.NoneAlgorithm()
    var_29 = module_0.Signer(var_27, algorithm=var_28)
    var_30 = b'test'
    var_31 = var_29.get_signature(var_30)
    var_32 = var_29.verify_signature(var_30, var_31)
    assert var_32 is True



# Parsed testcases at query #29
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
    var_10 = 'old-key'
    var_11 = 'new-key'
    var_12 = [var_10, var_11]
    var_13 = module_0.Signer(var_12)
    var_14 = b'rotation test'
    var_15 = var_13.get_signature(var_14)
    var_16 = var_13.verify_signature(var_14, var_15)
    assert var_16 is True
    var_17 = module_0.Signer(var_10)
    var_18 = var_17.get_signature(var_14)
    var_19 = var_13.verify_signature(var_14, var_18)
    assert var_19 is True
    var_20 = 'string value'
    var_21 = var_1.get_signature(var_20)
    var_22 = var_1.verify_signature(var_20, var_21)
    assert var_22 is True
    var_23 = b'!!!invalid-base64!!!'
    var_24 = var_1.verify_signature(var_14, var_23)
    assert var_24 is False
    var_25 = 'key'
    var_26 = module_0.NoneAlgorithm()
    var_27 = module_0.Signer(var_25, algorithm=var_26)
    var_28 = b'none algo'
    var_29 = var_27.get_signature(var_28)
    var_30 = var_27.verify_signature(var_28, var_29)
    assert var_30 is True
    var_31 = b''
    var_32 = var_27.verify_signature(var_28, var_31)
    assert var_32 is True
    var_33 = b'anything'
    var_34 = var_27.verify_signature(var_28, var_33)
    assert var_34 is False
    var_35 = 'different-salt'
    var_36 = module_0.Signer(var_25, var_35)
    var_37 = b'salt test'
    var_38 = var_36.get_signature(var_37)
    var_39 = var_36.verify_signature(var_37, var_38)
    assert var_39 is True
    var_40 = var_1.verify_signature(var_37, var_38)
    assert var_40 is False



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
    var_5 = b'invalid-signature'
    var_6 = module_1.base64_encode(var_5)
    var_7 = var_1.verify_signature(var_2, var_6)
    assert var_7 is False
    var_8 = b''
    var_9 = var_1.get_signature(var_8)
    var_10 = var_1.verify_signature(var_8, var_9)
    assert var_10 is True
    var_11 = 'different-secret'
    var_12 = module_0.Signer(var_11)
    var_13 = var_12.get_signature(var_2)
    var_14 = var_1.verify_signature(var_2, var_13)
    assert var_14 is False
    var_15 = 'string-value'
    var_16 = var_1.get_signature(var_15)
    var_17 = var_1.verify_signature(var_15, var_16)
    assert var_17 is True
    var_18 = 'not-base64!!'
    var_19 = var_1.verify_signature(var_2, var_18)
    assert var_19 is False
    var_20 = b''
    var_21 = var_1.verify_signature(var_2, var_20)
    assert var_21 is False
    var_22 = 'secret'
    var_23 = module_0.NoneAlgorithm()
    var_24 = module_0.Signer(var_22, algorithm=var_23)
    var_25 = var_24.get_signature(var_2)
    var_26 = var_24.verify_signature(var_2, var_25)
    assert var_26 is True
    var_27 = 'old-key'
    var_28 = 'new-key'
    var_29 = [var_27, var_28]
    var_30 = module_0.Signer(var_29)
    var_31 = module_0.Signer(var_27)
    var_32 = var_31.get_signature(var_2)
    var_33 = var_30.verify_signature(var_2, var_32)
    assert var_33 is True
    var_34 = module_0.Signer(var_28)
    var_35 = var_34.get_signature(var_2)
    var_36 = var_30.verify_signature(var_2, var_35)
    assert var_36 is True
    var_37 = 'unknown-key'
    var_38 = module_0.Signer(var_37)
    var_39 = var_38.get_signature(var_2)
    var_40 = var_30.verify_signature(var_2, var_39)
    assert var_40 is False
    var_41 = b'custom-salt'
    var_42 = module_0.Signer(var_22, var_41)
    var_43 = var_42.get_signature(var_2)
    var_44 = var_42.verify_signature(var_2, var_43)
    assert var_44 is True
    var_45 = var_1.verify_signature(var_2, var_43)
    assert var_45 is False



