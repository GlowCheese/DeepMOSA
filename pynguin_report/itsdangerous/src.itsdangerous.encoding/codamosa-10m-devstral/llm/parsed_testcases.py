####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'aGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZ25hbWU+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<big name>'
    var_12 = 'PGJpZ25hbWU'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<big name>'
    var_14 = 'YWJjZGVmZ2g='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abcdefgh'
    var_16 = 'SGVsbG8!\n'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = 'SGVsbG8?'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!='
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8=SGVsbG8='
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = module_0.base64_decode(var_4)
    assert var_13 == b'Hello'
    var_14 = 'SGVsbG8!@#'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!@#$%^&*()'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8!@#$%^&*()'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!@#$%^&*()'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8!@#$%^&*()'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8!@#$%^&*()'
    var_27 = module_0.base64_decode(var_26)



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'PGJhc2U2NF90ZXN0Pg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<base64_test>'
    var_10 = 'PGJhc2U2NF90ZXN0Pg'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<base64_test>'
    var_12 = '8J+YgA=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'\x10\xff\x00'
    var_14 = 'SGVsbG8!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8=!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8==='
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_2)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_2)
    assert var_9 == b'Hello'
    var_10 = module_0.base64_decode(var_0)
    assert var_10 == b'Hello'
    var_11 = module_0.base64_decode(var_2)
    assert var_11 == b'Hello'
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = module_0.base64_decode(var_2)
    assert var_13 == b'Hello'
    var_14 = module_0.base64_decode(var_0)
    assert var_14 == b'Hello'
    var_15 = module_0.base64_decode(var_2)
    assert var_15 == b'Hello'
    var_16 = module_0.base64_decode(var_0)
    assert var_16 == b'Hello'
    var_17 = module_0.base64_decode(var_2)
    assert var_17 == b'Hello'
    var_18 = module_0.base64_decode(var_0)
    assert var_18 == b'Hello'
    var_19 = module_0.base64_decode(var_2)
    assert var_19 == b'Hello'
    var_20 = module_0.base64_decode(var_0)
    assert var_20 == b'Hello'
    var_21 = module_0.base64_decode(var_2)
    assert var_21 == b'Hello'
    var_22 = module_0.base64_decode(var_0)
    assert var_22 == b'Hello'
    var_23 = module_0.base64_decode(var_2)
    assert var_23 == b'Hello'
    var_24 = module_0.base64_decode(var_0)
    assert var_24 == b'Hello'
    var_25 = module_0.base64_decode(var_2)
    assert var_25 == b'Hello'
    var_26 = module_0.base64_decode(var_0)
    assert var_26 == b'Hello'
    var_27 = module_0.base64_decode(var_2)
    assert var_27 == b'Hello'
    var_28 = module_0.base64_decode(var_0)
    assert var_28 == b'Hello'
    var_29 = module_0.base64_decode(var_2)
    assert var_29 == b'Hello'
    var_30 = module_0.base64_decode(var_0)
    assert var_30 == b'Hello'
    var_31 = module_0.base64_decode(var_2)
    assert var_31 == b'Hello'
    var_32 = module_0.base64_decode(var_0)
    assert var_32 == b'Hello'
    var_33 = module_0.base64_decode(var_2)
    assert var_33 == b'Hello'
    var_34 = module_0.base64_decode(var_0)
    assert var_34 == b'Hello'
    var_35 = module_0.base64_decode(var_2)
    assert var_35 == b'Hello'
    var_36 = module_0.base64_decode(var_0)
    assert var_36 == b'Hello'
    var_37 = module_0.base64_decode(var_2)
    assert var_37 == b'Hello'
    var_38 = module_0.base64_decode(var_0)
    assert var_38 == b'Hello'
    var_39 = module_0.base64_decode(var_2)
    assert var_39 == b'Hello'
    var_40 = module_0.base64_decode(var_0)
    assert var_40 == b'Hello'
    var_41 = module_0.base64_decode(var_2)
    assert var_41 == b'Hello'
    var_42 = module_0.base64_decode(var_0)
    assert var_42 == b'Hello'
    var_43 = module_0.base64_decode(var_2)
    assert var_43 == b'Hello'
    var_44 = module_0.base64_decode(var_0)
    assert var_44 == b'Hello'
    var_45 = module_0.base64_decode(var_2)
    assert var_45 == b'Hello'
    var_46 = module_0.base64_decode(var_0)
    assert var_46 == b'Hello'
    var_47 = module_0.base64_decode(var_2)
    assert var_47 == b'Hello'
    var_48 = module_0.base64_decode(var_0)
    assert var_48 == b'Hello'
    var_49 = module_0.base64_decode(var_2)
    assert var_49 == b'Hello'
    var_50 = module_0.base64_decode(var_0)
    assert var_50 == b'Hello'
    var_51 = module_0.base64_decode(var_2)
    assert var_51 == b'Hello'
    var_52 = module_0.base64_decode(var_0)
    assert var_52 == b'Hello'
    var_53 = module_0.base64_decode(var_2)
    assert var_53 == b'Hello'
    var_54 = module_0.base64_decode(var_0)
    assert var_54 == b'Hello'
    var_55 = module_0.base64_decode(var_2)
    assert var_55 == b'Hello'
    var_56 = module_0.base64_decode(var_0)
    assert var_56 == b'Hello'
    var_57 = module_0.base64_decode(var_2)
    assert var_57 == b'Hello'
    var_58 = module_0.base64_decode(var_0)
    assert var_58 == b'Hello'
    var_59 = module_0.base64_decode(var_2)
    assert var_59 == b'Hello'
    var_60 = module_0.base64_decode(var_0)
    assert var_60 == b'Hello'
    var_61 = module_0.base64_decode(var_2)
    assert var_61 == b'Hello'
    var_62 = module_0.base64_decode(var_0)
    assert var_62 == b'Hello'
    var_63 = module_0.base64_decode(var_2)
    assert var_63 == b'Hello'
    var_64 = module_0.base64_decode(var_0)
    assert var_64 == b'Hello'
    var_65 = module_0.base64_decode(var_2)
    assert var_65 == b'Hello'
    var_66 = module_0.base64_decode(var_0)
    assert var_66 == b'Hello'
    var_67 = module_0.base64_decode(var_2)
    assert var_67 == b'Hello'
    var_68 = module_0.base64_decode(var_0)
    assert var_68 == b'Hello'
    var_69 = module_0.base64_decode(var_2)
    assert var_69 == b'Hello'
    var_70 = module_0.base64_decode(var_0)
    assert var_70 == b'Hello'
    var_71 = module_0.base64_decode(var_2)
    assert var_71 == b'Hello'
    var_72 = module_0.base64_decode(var_0)
    assert var_72 == b'Hello'
    var_73 = module_0.base64_decode(var_2)
    assert var_73 == b'Hello'
    var_74 = module_0.base64_decode(var_0)
    assert var_74 == b'Hello'
    var_75 = module_0.base64_decode(var_2)
    assert var_75 == b'Hello'
    var_76 = module_0.base64_decode(var_0)
    assert var_76 == b'Hello'
    var_77 = module_0.base64_decode(var_2)
    assert var_77 == b'Hello'
    var_78 = module_0.base64_decode(var_0)
    assert var_78 == b'Hello'
    var_79 = module_0.base64_decode(var_2)
    assert var_79 == b'Hello'
    var_80 = module_0.base64_decode(var_0)
    assert var_80 == b'Hello'
    var_81 = module_0.base64_decode(var_2)
    assert var_81 == b'Hello'
    var_82 = module_0.base64_decode(var_0)
    assert var_82 == b'Hello'
    var_83 = module_0.base64_decode(var_2)
    assert var_83 == b'Hello'
    var_84 = module_0.base64_decode(var_0)
    assert var_84 == b'Hello'
    var_85 = module_0.base64_decode(var_2)
    assert var_85 == b'Hello'
    var_86 = module_0.base64_decode(var_0)
    assert var_86 == b'Hello'
    var_87 = module_0.base64_decode(var_2)
    assert var_87 == b'Hello'
    var_88 = module_0.base64_decode(var_0)
    assert var_88 == b'Hello'
    var_89 = module_0.base64_decode(var_2)
    assert var_89 == b'Hello'
    var_90 = module_0.base64_decode(var_0)
    assert var_90 == b'Hello'
    var_91 = module_0.base64_decode(var_2)
    assert var_91 == b'Hello'
    var_92 = module_0.base64_decode(var_0)
    assert var_92 == b'Hello'
    var_93 = module_0.base64_decode(var_2)
    assert var_93 == b'Hello'
    var_94 = module_0.base64_decode(var_0)
    assert var_94 == b'Hello'
    var_95 = module_0.base64_decode(var_2)
    assert var_95 == b'Hello'
    var_96 = module_0.base64_decode(var_0)
    assert var_96 == b'Hello'
    var_97 = module_0.base64_decode(var_2)
    assert var_97 == b'Hello'
    var_98 = module_0.base64_decode(var_0)
    assert var_98 == b'Hello'
    var_99 = module_0.base64_decode(var_2)
    assert var_99 == b'Hello'
    var_100 = module_0.base64_decode(var_0)
    assert var_100 == b'Hello'
    var_101 = module_0.base64_decode(var_2)
    assert var_101 == b'Hello'
    var_102 = module_0.base64_decode(var_0)
    assert var_102 == b'Hello'
    var_103 = module_0.base64_decode(var_2)
    assert var_103 == b'Hello'
    var_104 = module_0.base64_decode(var_0)
    assert var_104 == b'Hello'
    var_105 = module_0.base64_decode(var_2)
    assert var_105 == b'Hello'
    var_106 = module_0.base64_decode(var_0)
    assert var_106 == b'Hello'
    var_107 = module_0.base64_decode(var_2)
    assert var_107 == b'Hello'
    var_108 = module_0.base64_decode(var_0)
    assert var_108 == b'Hello'
    var_109 = module_0.base64_decode(var_2)
    assert var_109 == b'Hello'
    var_110 = module_0.base64_decode(var_0)
    assert var_110 == b'Hello'



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = module_0.base64_decode(var_0)
    assert var_4 == b'Hello'
    var_5 = module_0.base64_decode(var_2)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_2)
    assert var_6 == b'Hello'
    var_7 = ''
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b''
    var_9 = b'SGVsbG8='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = b'SGVsbG8'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = 'Invalid!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'SGVsbG8!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8='
    var_18 = 100
    var_19 = var_17 * var_18
    var_20 = '!'
    var_21 = var_19 + var_20
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8ÿ'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'
    var_25 = 'SGVsbG8ÿÿ'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'
    var_27 = 'SGVsbG8-'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello-'
    var_29 = 'SGVsbG8_'
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello_'
    var_31 = module_0.base64_decode(var_17)
    assert var_31 == b'Hello'
    var_32 = 'SGVsbG8=='
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'Hello'



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_0.base64_decode(var_0)
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=😊'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'SGVsbG8=ñ'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = module_0.base64_decode(var_2)
    assert var_24 == b'Hello'
    var_25 = module_0.base64_decode(var_18)
    assert var_25 == b'Hello'
    var_26 = 'SGVsbG8=='
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZ2ZpbG0+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigfilm>'
    var_12 = 'PGJpZ2ZpbG0'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfilm>'
    var_14 = 'YWJjZGVmZw=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abcdefg'
    var_16 = 'YWJjZGVmZw'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abcdefg'
    var_18 = '8J+YgA=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'\x10\xff\xee'
    var_20 = '8J+YgA'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'\x10\xff\xee'
    var_22 = 'Invalid!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8=!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8=ÿ'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'
    var_30 = 'SGVsbG8=ÿþ'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'aGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZ2Zvb3Q+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigfoot>'
    var_12 = 'PGJpZ2Zvb3Q'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = 'SGVsbG8!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8='
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8==='
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJyb2FkY2FzdD4='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<broadcast>'
    var_12 = 'PGJyb2FkY2FzdD4'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<broadcast>'
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = 'YWJjZA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abcd'
    var_18 = 'YWJjZGU='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abcde'
    var_20 = '!!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=😊'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = 'SGVsbG8=ñ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = ' SGVsbG8= '
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'
    var_30 = 'SG Vs bG8='
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = module_0.base64_decode(var_2)
    var_10 = module_0.base64_decode(var_2)
    var_11 = module_0.base64_decode(var_2)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'Invalid!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'Invalid!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8ÿ'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'PGJpZCBpZD0iMTAwIj5UZXN0PC9iaWQ+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bid id="100">Test</bid>'
    var_14 = 'PGJpZCBpZD0iMTAwIj5UZXN0PC9iaWQ-'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bid id="100">Test</bid>'
    var_16 = 'Invalid@Base64!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=😊'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = b'SGVsbG8=\xff'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello?'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = 'SGVsbG8ÿ'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = 'SGVsbG8!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8#'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8$'
    var_20 = module_0.base64_decode(var_19)



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello?'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'SGVsbG8!@#'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hell'



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8!@#'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = b'SGVsbG8!@#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8='
    var_23 = 100
    var_24 = var_22 * var_23
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8é'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = b'SGVsbG8\xff'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'Invalid!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_0.base64_decode(var_2)
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'PGJpZ25hbWU+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<big-name>'
    var_14 = 'PGJpZ25hbWU'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<big-name>'
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=😊'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = b'SGVsbG8=\xff'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJhcj4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<bar>'
    var_10 = 'PGJhcj4'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bar>'
    var_12 = module_0.base64_decode(var_8)
    assert var_12 == b'<bar>'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'SGVsbG8!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8#'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8$'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8=é'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'
    var_23 = 'SGVsbG8=😊'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = module_1.urlsafe_b64decode(var_0)
    var_10 = module_0.base64_decode(var_2)
    var_11 = module_1.urlsafe_b64decode(var_2)
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'Invalid!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8=ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = 'SGVsbG8-'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = 'SGVsbG8_'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8@'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8#'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8=ÿ'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'
    var_25 = b'SGVsbG8=\xff'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_0.base64_decode(var_2)
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZj4='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bif>'
    var_12 = 'PGJpZj4'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bif>'
    var_14 = 'Invalid!@#'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8=ÿ'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = 'SGVsbG8ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVs'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hel'
    var_22 = 'SGV'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'He'



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJhc2U2NGVudGVzdD4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<base64test>'
    var_10 = 'PGJhc2U2NGVudGVzdD4'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<base64test>'
    var_12 = module_0.base64_decode(var_8)
    assert var_12 == b'<base64test>'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'invalid!base64'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8=ÿ'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'
    var_21 = 'SGVsbG8ÿ'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'Hello'



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'PGJpZCBieT0iYSI+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bid by="a">'
    var_14 = 'PGJpZCBieT0iYSI'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bid by="a">'
    var_16 = 'SGVsbG8!\n'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = b'SGVsbG8!\n'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = '!!!invalid!!!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'!!!invalid!!!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'é'
    var_25 = var_22 + var_24
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'
    var_27 = 'utf-8'
    var_28 = module_1.encode(var_27)
    var_29 = var_4 + var_28
    var_30 = module_0.base64_decode(var_29)
    assert var_30 == b'Hello'



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'aGVsbG8='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'helloo'
    var_14 = 'aGVsbG8'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'helloo'
    var_16 = 'PGJvZHk+'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'<body>'
    var_18 = 'PGJvZHk'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'<body>'
    var_20 = 'invalid!base64'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8=😊'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = b'SGVsbG8=\xff'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8-'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8=='
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = 'SGVsbG8!@#'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'Hello'
    var_15 = 'InvalidBase64!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = module_0.base64_decode(var_4)
    assert var_17 == b'Hello'
    var_18 = module_0.base64_decode(var_15)
    assert var_18 == b'Hello'
    var_19 = module_0.base64_decode(var_9)
    assert var_19 == b'Hello'



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8gd29ybGQ='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello world'
    var_2 = 'SGVsbG8gd29ybGQ'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello world'
    var_4 = 'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = b'SGVsbG8gd29ybGQ='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello world'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'SGVsbG8gd29ybGQ!@#'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello world'
    var_14 = 'Invalid!!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8='
    var_17 = module_0.base64_decode(var_16)



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZz5IZWxsbyBXb3JsZCE8L2JpZz4='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<big>Hello World!</big>'
    var_12 = 'PGJpZz5IZWxsbyBXb3JsZCE8L2JpZz4'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<big>Hello World!</big>'
    var_14 = 'YQ=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'a'
    var_16 = 'YWE='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'aa'
    var_18 = 'YWFh'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'aaa'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8$'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8#'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8ÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = 'SGVsbG8\x00'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'
    var_30 = 'QUJD'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'ABC'
    var_32 = 'QUJDRA=='
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'ABCD'
    var_34 = 'QUJDREU='
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'ABCDE'
    var_36 = 'QUJDREVG'
    var_37 = module_0.base64_decode(var_36)
    assert var_37 == b'ABCDEF'



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_0.base64_decode(var_0)
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbÿG8='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = module_0.base64_decode(var_4)
    assert var_7 == b'Hello'
    var_8 = 'SGVsbG8-'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8_'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'Invalid!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'PGJpZ2ZpbG0+'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<bigfilm>'
    var_10 = 'PGJpZ2ZpbG0'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigfilm>'
    var_12 = 'YWJjZGVmZw=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'abcdefg'
    var_14 = 'YWJjZGVmZw'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abcdefg'
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbÿG8='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hell'



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'aGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'hello'
    var_2 = b'aGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'hello'
    var_4 = 'aGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = module_0.base64_decode(var_4)
    assert var_8 == b'hello'
    var_9 = 'aGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'hello'
    var_11 = 'aGVsbG8_'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'invalid!base64'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'aGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = module_0.base64_decode(var_17)
    assert var_19 == b'hello'
    var_20 = 'aGVsbG8=ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'hello'



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8= '
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8= ='
    var_23 = module_0.base64_decode(var_22)
    var_24 = 123
    var_25 = module_0.base64_decode(var_24)



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'SGVsbG8-'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8_'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'QUJDRA=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'ABCD'
    var_14 = 'QUJD'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'ABCD'
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8=ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8====='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = module_0.base64_decode(var_0)
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    var_12 = 'SGVsbG8+'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'YQ=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'a'
    var_18 = 'YWI='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'ab'
    var_20 = 'YWJj'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'abc'
    var_22 = 'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8='
    var_25 = module_0.base64_decode(var_24)
    var_26 = 123
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8=ÿ'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'



# Parsed testcases at query #2
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8gV29ybGQh'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello World!'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'SGVsbG8=😊'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = 'Invalid!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = module_0.base64_decode(var_14)
    assert var_16 == b'Hello'
    var_17 = 'SGVsbG8=='
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'
    var_19 = 'PGJpZ25hbWU+'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'<big name>'
    var_21 = module_0.base64_decode(var_19)
    assert var_21 == b'<big name>'



# Parsed testcases at query #3
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8gV29ybGQ'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello World'
    var_10 = 'PGJpZ2ZpbG0+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigfilm>'
    var_12 = 'PGJpZ2ZpbG0'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfilm>'
    var_14 = 'PGJpZ2ZpbG0-'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigfilm>'
    var_16 = ''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = b''
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b''
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8@'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8#'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8=😊'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = 'SGVsbG8=äöü'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'
    var_30 = 'a'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'\xed'
    var_32 = 'ab'
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'\xed\x95'
    var_34 = 'abc'
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'\xed\x95\x9c'



# Parsed testcases at query #4
#--------------------------


import src.itsdangerous.encoding as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'PGJpZ2Zvb3Q+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = 'PGJpZ2Zvb3Q'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigfoot>'
    var_16 = 'PGJpZ2Zvb3Q='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'<bigfoot>'
    var_18 = 'PGJpZ2Zvb3Q=='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'<bigfoot>'
    var_20 = 'SGVsbG8!@#'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = b'SGVsbG8!@#'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = 'SGVsbG8é'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = b'SGVsbG8\xff'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = 'SGVsbG'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hell'
    var_30 = module_0.base64_decode(var_0)
    assert var_30 == b'Hello'
    var_31 = 'SGVsbG8=='
    var_32 = module_0.base64_decode(var_31)
    assert var_32 == b'Hello'
    var_33 = 'SGVsbG8!'
    var_34 = module_0.base64_decode(var_33)
    var_35 = 'SGVsbG8='
    var_36 = 100
    var_37 = var_35 * var_36
    var_38 = '!'
    var_39 = var_37 + var_38
    var_40 = module_0.base64_decode(var_39)
    var_41 = 'a'
    var_42 = 1000
    var_43 = var_41 * var_42
    var_44 = module_1.encode()
    var_45 = module_1.urlsafe_b64encode(var_44)
    var_46 = b'='
    var_47 = module_1.encode()



# Parsed testcases at query #5
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'PGJpZ2Zvb3Q+'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'<bigfoot>'
    var_8 = 'PGJpZ2Zvb3Q'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<bigfoot>'
    var_10 = 'PGJpZ2Zvb3Q='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigfoot>'
    var_12 = 'PGJpZ2Zvb3Q=='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfoot>'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8#'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8$'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8ÿ'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = 'SGVsbG8\x00'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = 'a'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'\xab'
    var_28 = 'ab'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'\xab\xc0'
    var_30 = 'abc'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'\xab\xcd\xef'



# Parsed testcases at query #6
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello'
    var_11 = 'SGVsbG8_'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'Invalid!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8ÿ'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



# Parsed testcases at query #7
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'SGVsbG8\x00'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = 'VGVzdCBkYXRhIHRoYXQgaXMgYSB0ZXN0IGZvciBib29rZWVwaW5n'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Test data that is a test for bookkeeping'



# Parsed testcases at query #8
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = 'Zg=='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'f'
    var_10 = 'Zm8='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'fo'
    var_12 = 'Zm9v'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'foo'
    var_14 = 'Zm9vYg=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'foob'
    var_16 = 'Zm9vYmE='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'fooba'
    var_18 = 'Zm9vYmFy'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'foobar'
    var_20 = module_0.base64_decode(var_0)
    var_21 = module_0.base64_decode(var_2)
    var_22 = 'PGJyPg=='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'<br>'
    var_24 = 'PGJyPg'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'<br>'
    var_26 = 'SGVsbG8!'
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8=!'
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8==='
    var_31 = module_0.base64_decode(var_30)
    var_32 = 'SGVsbG'
    var_33 = module_0.base64_decode(var_32)
    var_34 = 123
    var_35 = module_0.base64_decode(var_34)
    var_36 = '4pyT'
    var_37 = module_0.base64_decode(var_36)
    assert var_37 == b'\xe4\xbd\xa0\xe5\xa5\xbd'
    var_38 = 'Ã¤Ã¶Ã¼'
    var_39 = module_0.base64_decode(var_38)
    assert var_39 == b''
    var_40 = module_0.base64_decode(var_38)
    assert var_40 == b''



# Parsed testcases at query #9
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello?'
    var_12 = 'SGVsbG8!@#'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8==!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = module_0.base64_decode(var_22)
    assert var_24 == b'Hello'
    var_25 = 'SGVsbG8=ÿ'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'



# Parsed testcases at query #10
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_2)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8-'
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello?'
    var_9 = 'SGVsbG8_'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello/'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = 'Invalid!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = 'SGVsbG8ÿ'
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'Hello'



# Parsed testcases at query #11
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_2)
    assert var_9 == b'Hello'
    var_10 = module_0.base64_decode(var_2)
    assert var_10 == b'Hello'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = 'YQ'
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b'a'
    var_15 = 'YQ=='
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b'a'
    var_17 = 'YWI'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'ab'
    var_19 = 'YWI='
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'ab'
    var_21 = 'YWJj'
    var_22 = module_0.base64_decode(var_21)
    assert var_22 == b'abc'
    var_23 = module_0.base64_decode(var_21)
    assert var_23 == b'abc'
    var_24 = 'Invalid!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'YQ#='
    var_27 = module_0.base64_decode(var_26)
    var_28 = 12345
    var_29 = module_0.base64_decode(var_28)
    var_30 = 'SGVsbG8ÿ'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'



# Parsed testcases at query #12
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'SGVsbG8-'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = 'SGVsbG8_'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8ÿ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'SGVsbG8\x00'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'



# Parsed testcases at query #13
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQh'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World!'
    var_8 = module_0.base64_decode(var_6)
    assert var_8 == b'Hello World!'
    var_9 = 'SGVsbG8-V29ybGQh'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello+World!'
    var_11 = 'SGVsbG8_V29ybGQh'
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b'Hello/World!'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'Invalid!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8=ÿ'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



# Parsed testcases at query #14
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJvZHk+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<body>'
    var_12 = 'PGJvZHk-'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<body>'
    var_14 = 'Invalid!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8=😊'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #15
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'PGJpZ2ZpbG0+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfilm>'
    var_14 = 'PGJpZ2ZpbG0'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigfilm>'
    var_16 = 'SGVsbG8!@#$'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = b'SGVsbG8\xff\xfe'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hell'
    var_22 = 'SGVsb'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hel'
    var_24 = 'SGVsbG8!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8=='
    var_27 = module_0.base64_decode(var_26)
    var_28 = 'SGVsbG8=!'
    var_29 = module_0.base64_decode(var_28)



# Parsed testcases at query #16
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello?'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8!@#'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = b'SGVsbG8!@#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = b'SGVsbG8!'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8é'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'
    var_26 = b'SGVsbG8\xc3\xa9'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'



# Parsed testcases at query #17
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = module_0.base64_decode(var_0)
    assert var_12 == b'Hello'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = b''
    var_16 = module_0.base64_decode(var_15)
    assert var_16 == b''
    var_17 = 'Invalid!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8=!'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8ÿ'
    var_24 = module_0.base64_decode(var_23)
    assert var_24 == b'Hello'
    var_25 = 'SGVsbG8\x00'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'



# Parsed testcases at query #18
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'PGJvZHk+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<body>'
    var_14 = 'PGJvZHk'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<body>'
    var_16 = 'YWJjX2RlZg=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abc_def'
    var_18 = 'YWJjX2RlZg'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abc_def'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8='
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8==='
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8=ÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = module_0.base64_decode(var_26)
    assert var_28 == b'Hello'



# Parsed testcases at query #19
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'SGVsbG8gd29ybGQ='
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello world'
    var_14 = 'SGVsbG8gd29ybGQ'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello world'
    var_16 = 'SGVsbG8gd29ybGQh'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello world!'
    var_18 = module_0.base64_decode(var_16)
    assert var_18 == b'Hello world!'
    var_19 = 'SGVsbG8!'
    var_20 = module_0.base64_decode(var_19)
    var_21 = 'SGVsbG8@'
    var_22 = module_0.base64_decode(var_21)
    var_23 = 'SGVsbG8#'
    var_24 = module_0.base64_decode(var_23)
    var_25 = 'SGVsbG8ÿ'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'
    var_27 = b'SGVsbG8\xff'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'
    var_29 = module_0.base64_decode(var_23)
    assert var_29 == b'Hello'
    var_30 = 'SGVsbG8=='
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'



# Parsed testcases at query #20
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'aGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'hello'
    var_6 = ''
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b''
    var_8 = b'SGVsbG8='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello'
    var_10 = b'SGVsbG8'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'PGJpZ25hbWU+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigname>'
    var_14 = 'PGJpZ25hbWU'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<bigname>'
    var_16 = 'YWJjZGVmZw=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abcdefg'
    var_18 = 'YWJjZGVmZw'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abcdefg'
    var_20 = 'SGVsbG8!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8#'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8$'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8ÿ'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = 'SGVsbG8\x00'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'Hello'
    var_30 = 'SGVs'
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hel'
    var_32 = 'SGV'
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'He'
    var_34 = 'SG'
    var_35 = module_0.base64_decode(var_34)
    assert var_35 == b'H'



# Parsed testcases at query #21
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    assert var_10 == b'Hello?'
    var_11 = ''
    var_12 = module_0.base64_decode(var_11)
    assert var_12 == b''
    var_13 = 'Invalid!'
    var_14 = module_0.base64_decode(var_13)
    var_15 = b'Invalid!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8ÿ'
    var_18 = module_0.base64_decode(var_17)
    assert var_18 == b'Hello'



# Parsed testcases at query #22
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJhc2U2ND4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<base64>'
    var_10 = 'PGJhc2U2ND4'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<base64>'
    var_12 = module_0.base64_decode(var_8)
    assert var_12 == b'<base64>'
    var_13 = ''
    var_14 = module_0.base64_decode(var_13)
    assert var_14 == b''
    var_15 = 'Invalid!'
    var_16 = module_0.base64_decode(var_15)
    var_17 = 'SGVsbG8!'
    var_18 = module_0.base64_decode(var_17)
    var_19 = 'SGVsbG8=😊'
    var_20 = module_0.base64_decode(var_19)
    assert var_20 == b'Hello'



# Parsed testcases at query #23
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_0)
    var_9 = 'SGVsbG8-'
    var_10 = module_0.base64_decode(var_9)
    var_11 = module_0.base64_decode(var_0)
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'Invalid!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'Invalid!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8=ÿ'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = b'SGVsbG8=\xff'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'



# Parsed testcases at query #24
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'aGVsbG8gd29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'hello world'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZj4='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bif>'
    var_12 = 'PGJpZj4'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bif>'
    var_14 = 'YWJjZGVmZw=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abcdefg'
    var_16 = 'YWJj'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'ab'
    var_18 = 'YWJjZGU='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abcde'
    var_20 = 'YWJjZGVm'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'abcdef'
    var_22 = 'invalid!base64'
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'SGVsbG8!'
    var_25 = module_0.base64_decode(var_24)
    var_26 = 'SGVsbG8=☺'
    var_27 = module_0.base64_decode(var_26)
    assert var_27 == b'Hello'
    var_28 = 'aGVsbG8gd29ybGQ=😊'
    var_29 = module_0.base64_decode(var_28)
    assert var_29 == b'hello world'
    var_30 = ' SGVsbG8= '
    var_31 = module_0.base64_decode(var_30)
    assert var_31 == b'Hello'
    var_32 = 'aGVs bG8g d29y bGQ='
    var_33 = module_0.base64_decode(var_32)
    assert var_33 == b'hello world'



# Parsed testcases at query #25
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello?'
    var_12 = 'SGVsbG8_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello/'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'SGVsbG8!@#'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'Invalid!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8é'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'



# Parsed testcases at query #26
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8gV29ybGQh'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello World!'
    var_10 = 'SGVsbG8gV29ybGQ-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello World?'
    var_12 = 'SGVsbG8gV29ybGQ_'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello World/'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8!@#'
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'Hello'
    var_18 = 'Invalid@@@'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=Invalid'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'ascii'
    var_23 = module_0.base64_decode(var_20)
    assert var_23 == b'Hello'



# Parsed testcases at query #27
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = b''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8#'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8$'
    var_21 = module_0.base64_decode(var_20)
    var_22 = module_0.base64_decode(var_2)
    assert var_22 == b'Hello'
    var_23 = module_0.base64_decode(var_20)
    assert var_23 == b'Hello'
    var_24 = module_0.base64_decode(var_10)
    assert var_24 == b'Hello'
    var_25 = 'SGVsbG8ÿ'
    var_26 = module_0.base64_decode(var_25)
    assert var_26 == b'Hello'
    var_27 = 'SGVsbG8\x00'
    var_28 = module_0.base64_decode(var_27)
    assert var_28 == b'Hello'



# Parsed testcases at query #28
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJpZ2ZpbG0+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bigfilm>'
    var_12 = 'PGJpZ2ZpbG0'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<bigfilm>'
    var_14 = '8J+YgA=='
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'\xe4\xbd\xa0\xe5\xa5\xbd'
    var_16 = '!!!invalid!!!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'
    var_20 = 'SGVsbG8ÿ='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'
    var_22 = 'SGVsbG8===='
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'



# Parsed testcases at query #29
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = 'PGJhcj4='
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'<bar>'
    var_10 = 'PGJhcj4'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<bar>'
    var_12 = module_0.base64_decode(var_8)
    assert var_12 == b'<bar>'
    var_13 = module_0.base64_decode(var_10)
    assert var_13 == b'<bar>'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = b''
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b''
    var_18 = 'Invalid!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = b'Invalid!'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8=😊'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'
    var_24 = b'SGVsbG8=\xff'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'Hello'



# Parsed testcases at query #30
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = 'PGJvZHk+'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'<body>'
    var_12 = 'PGJvZHk-'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<body>'
    var_14 = 'PGJvZHk_'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<body>'
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8=ÿþ'
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hello'



# Parsed testcases at query #31
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = ''
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b''
    var_10 = b''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = 'PGJpZ25hbWU+'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'<big name>'
    var_14 = 'PGJpZ25hbWU'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'<big name>'
    var_16 = 'Invalid!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8!'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbÿG8='
    var_21 = module_0.base64_decode(var_20)
    assert var_21 == b'Hell'



# Parsed testcases at query #32
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = module_0.base64_decode(var_0)
    assert var_6 == b'Hello'
    var_7 = 'SGVsbG8=='
    var_8 = module_0.base64_decode(var_7)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_4)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8-'
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello?'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'SGVsbG8!\n'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'Hello'
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8===='
    var_19 = module_0.base64_decode(var_18)



# Parsed testcases at query #33
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = ''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'Invalid!'
    var_15 = module_0.base64_decode(var_14)
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8ÿ'
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'Hello'



# Parsed testcases at query #34
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = 'SGVsbG8'
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = b'SGVsbG8='
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = b'SGVsbG8'
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello'
    var_8 = module_0.base64_decode(var_2)
    assert var_8 == b'Hello'
    var_9 = module_0.base64_decode(var_0)
    assert var_9 == b'Hello'
    var_10 = 'SGVsbG8=='
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b'Hello'
    var_12 = 'SGVsbG8---'
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b'Hello'
    var_14 = ''
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b''
    var_16 = 'SGVsbG8!'
    var_17 = module_0.base64_decode(var_16)
    var_18 = 'SGVsbG8@'
    var_19 = module_0.base64_decode(var_18)
    var_20 = 'SGVsbG8#'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'SGVsbG8ÿ'
    var_23 = module_0.base64_decode(var_22)
    assert var_23 == b'Hello'



# Parsed testcases at query #35
#--------------------------


import src.itsdangerous.encoding as module_0

def test_case_0():
    var_0 = 'SGVsbG8='
    var_1 = module_0.base64_decode(var_0)
    assert var_1 == b'Hello'
    var_2 = b'SGVsbG8='
    var_3 = module_0.base64_decode(var_2)
    assert var_3 == b'Hello'
    var_4 = 'SGVsbG8'
    var_5 = module_0.base64_decode(var_4)
    assert var_5 == b'Hello'
    var_6 = 'SGVsbG8gV29ybGQ='
    var_7 = module_0.base64_decode(var_6)
    assert var_7 == b'Hello World'
    var_8 = 'SGVsbG8gV29ybGQh'
    var_9 = module_0.base64_decode(var_8)
    assert var_9 == b'Hello World!'
    var_10 = ''
    var_11 = module_0.base64_decode(var_10)
    assert var_11 == b''
    var_12 = b''
    var_13 = module_0.base64_decode(var_12)
    assert var_13 == b''
    var_14 = 'YWJj'
    var_15 = module_0.base64_decode(var_14)
    assert var_15 == b'abc'
    var_16 = 'YWJjZA=='
    var_17 = module_0.base64_decode(var_16)
    assert var_17 == b'abcd'
    var_18 = 'YWJjZGU='
    var_19 = module_0.base64_decode(var_18)
    assert var_19 == b'abcde'
    var_20 = 'Invalid!Base64@String'
    var_21 = module_0.base64_decode(var_20)
    var_22 = 'YWJj!ZA=='
    var_23 = module_0.base64_decode(var_22)
    var_24 = 'YWJjÿþ'
    var_25 = module_0.base64_decode(var_24)
    assert var_25 == b'abc'



